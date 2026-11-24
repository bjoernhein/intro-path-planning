
"""
This code is part of a series of notebooks regarding  "Introduction to robot path planning".

Author: Gergely Soti, adapted by Bjoern Hein

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""
from IPPlanarManipulator import PlanarJoint, PlanarRobot
from IPEnvironment import CollisionChecker, OBSTACLE_COLOR, MIN_PENETRATION_DEPTH
from IPPerfMonitor import IPPerfMonitor
from shapely.geometry import Point, Polygon, LineString
from shapely import plotting
import numpy as np
import copy

def interpolate_line(startPos, endPos, step_l):
    """Configurations on the straight line from startPos to endPos with the distance step_l (the last step to
    endPos may be shorter). startPos and endPos are included."""
    steps = []
    line = np.array(endPos, dtype=float) - np.array(startPos, dtype=float)
    line_l = np.linalg.norm(line)
    if line_l == 0.0:
        return [np.array(endPos, dtype=float)]
    step = line / line_l * step_l
    n_steps = np.floor(line_l / step_l).astype(np.int32)
    c_step = np.array(startPos, dtype=float)
    for i in range(n_steps):
        steps.append(copy.deepcopy(c_step))
        c_step += step
    if not (c_step == np.array(endPos)).all():
        steps.append(np.array(endPos))
    return steps


class KinChainCollisionChecker(CollisionChecker):
    """Collision checker for a planar robot (kinematic chain, see IPPlanarManipulator.py and IP-10-0).

    A configuration is the list of the joint angles. For a collision check the robot is moved to the configuration,
    the positions of its joints in the work space are computed (forward kinematics) and every link is checked against
    the obstacles (shapely geometries in the work space).
    Not checked: the joint limits and collisions of the robot with itself.
    """

    def __init__(self, kin_chain, scene, limits=[[-3.0, 3.0], [-3.0, 3.0]], statistic=None, fk_resolution=0.1,
                 link_radius=0.0, skeleton="axis"):
        """kin_chain: the robot (PlanarRobot)
        scene: dict of obstacles in the work space
        limits: limits of the joint angles [rad], used by the planners for sampling
        fk_resolution: distance in the joint space between two checked configurations of a line (lineInCollision)
        link_radius: the links are lines (0.0) or capsules with this radius around the line
        skeleton: what a link is shrunk to by blowUp() (BB-Method):
            "axis":  the line between its joints, only the radius is blown up
            "joint": the joint at its base, the whole link (length and radius) is scaled
        """
        super(KinChainCollisionChecker, self).__init__(scene, limits, statistic)
        if len(limits) != kin_chain.dim:
            raise ValueError("Limits must match the dimension of the kinematic chain. Default values are for a 2-dof planar manipulator. If you use dof>2 you have to specify the limits explicitly")
        if skeleton not in ("axis", "joint"):
            raise ValueError('skeleton must be "axis" or "joint"')
        self.kin_chain = kin_chain
        self.fk_resolution = fk_resolution
        self.link_radius = link_radius
        self.skeleton = skeleton
        self.dim = self.kin_chain.dim
        self._cspace = None   # sampled C-obstacles (only for figures, see computeCSpace)

    def getDim(self):
        return self.dim

    def robotInCollision(self, pos):
        """The collision check itself: move the robot to pos and check every link in the work space."""
        self.kin_chain.move(pos)
        joint_positions = self.kin_chain.get_transforms()
        for i in range(1, len(joint_positions)):
            if self.segmentInCollision(joint_positions[i-1], joint_positions[i]):
                return True
        return False

    @IPPerfMonitor
    def pointInCollision(self, pos):
        return self.robotInCollision(pos)
    
    @IPPerfMonitor
    def lineInCollision(self, startPos, endPos, steps=None):
        assert (len(startPos) == self.getDim())
        assert (len(endPos) == self.getDim())
        steps = interpolate_line(startPos, endPos, self.fk_resolution)
        for pos in steps:
            if self.pointInCollision(pos):
                return True
        return False
    
    def segmentInCollision(self, startPos, endPos, radius=None):
        """Link from startPos to endPos in collision? radius: of the capsule (default: link_radius)"""
        radius = self.link_radius if radius is None else radius
        link = LineString([(startPos[0], startPos[1]), (endPos[0], endPos[1])])
        for key, value in self.scene.items():
            if radius > 0.0:
                # capsule around the line: exact test via the distance
                if value.distance(link) <= radius:
                    return True
            elif value.intersects(link):
                return True
        return False

    def blowUp(self, pos, tolerance=1e-3):
        """"Blow up" of the BB-Method: every link is shrunk to its skeleton and blown up again link by link
        from the base (see skeleton in __init__).
        Returns (i_coll, scale): i_coll is the first colliding link (1, 2, ...), scale in [0, 1] the largest scale
        of this link without collision (0: already the skeleton collides). Collision-free: (None, 1.0)
        """
        self.kin_chain.move(pos)
        joint_positions = [np.asarray(p, dtype=float) for p in self.kin_chain.get_transforms()]
        for i in range(1, len(joint_positions)):
            start, end = joint_positions[i-1], joint_positions[i]
            if not self.segmentInCollision(start, end):
                continue
            if self.skeleton == "axis":
                # the largest radius without collision is the distance of the axis to the obstacles
                if self.link_radius == 0.0:
                    return i, 0.0
                axis = LineString([tuple(start), tuple(end)])
                return i, min(value.distance(axis) for value in self.scene.values()) / self.link_radius
            # "joint": bisection of the scale of the whole link
            low, high = 0.0, 1.0
            while high - low > tolerance:
                scale = (low + high) / 2
                if self.segmentInCollision(start, start + scale * (end - start), scale * self.link_radius):
                    high = scale
                else:
                    low = scale
            return i, low
        return None, 1.0

    def qScale(self, pos):
        """Rating of the BB-Method: Q_scale = (i_coll - 1) + scale_icoll in [0, dim], collision-free: dim"""
        i_coll, scale = self.blowUp(pos)
        return float(self.dim) if i_coll is None else (i_coll - 1) + scale

    @IPPerfMonitor
    def penetrationDepth(self, pos):
        """Virtual penetration depth for the BB-Method: dim - Q_scale (free -> 0.0, see blowUp)"""
        assert (len(pos) == self.getDim())
        i_coll, scale = self.blowUp(pos)
        if i_coll is None:
            return 0.0
        return max(self.dim - (i_coll - 1) - scale, MIN_PENETRATION_DEPTH)

    def computeCSpace(self, resolution=150):
        """Sample the C-obstacles of a robot with two joints: collision check on a grid with resolution x resolution
        configurations inside the limits. Only for figures: afterwards drawObstacles(ax) draws the C-obstacles, so
        the visualizations of the planners (IPVIS*.py) show them. The checks are not recorded by IPPerfMonitor.
        Returns the grid: grid[i, j] is True, if the configuration (q1[j], q2[i]) is in collision."""
        if self.dim != 2:
            raise ValueError("The C-space can only be drawn for a robot with two joints")
        q1 = np.linspace(self.limits[0][0], self.limits[0][1], resolution)
        q2 = np.linspace(self.limits[1][0], self.limits[1][1], resolution)
        self._cspace = np.array([[self.robotInCollision([a, b]) for a in q1] for b in q2])
        return self._cspace

    def drawObstacles(self, ax, inWorkspace=False):
        """inWorkspace=True: the obstacles in the work space.
        inWorkspace=False (the call of the planner visualizations): the sampled C-obstacles, if they were computed
        with computeCSpace() - otherwise nothing, the C-obstacles are not known."""
        if inWorkspace:
            for key, value in self.scene.items():
                plotting.plot_polygon(value, add_points=False, color=OBSTACLE_COLOR, ax=ax)
        elif self._cspace is not None:
            from matplotlib.colors import ListedColormap, to_rgba
            ax.imshow(self._cspace, origin="lower", interpolation="nearest", zorder=0,
                      extent=(self.limits[0][0], self.limits[0][1], self.limits[1][0], self.limits[1][1]),
                      cmap=ListedColormap([(1, 1, 1, 0), to_rgba(OBSTACLE_COLOR, 0.3)]), vmin=0, vmax=1)



def planarRobotVisualize(kin_chain, ax, color='g', linewidth=1.5):
    """Draw the links of the robot in its current configuration as lines."""
    joint_positions = kin_chain.get_transforms()
    for i in range(1, len(joint_positions)):
        xs = [joint_positions[i-1][0], joint_positions[i][0]]
        ys = [joint_positions[i-1][1], joint_positions[i][1]]
        ax.plot(xs, ys, color=color, linewidth=linewidth)


import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML, display

matplotlib.rcParams['animation.embed_limit'] = 64


def animateSolution(planner, environment, solution, visualizer, workSpaceLimits=[[-3,3],[-3,3]]):
    """Animation of the robot along a solution path (list of nodes of planner.graph).
    Left: the robot in the work space. Right, only for a robot with two joints: the graph of the planner in the
    joint space (drawn by visualizer, e.g. IPVISLazyPRM.lazyPRMVisualize) with the current configuration."""
    twoJoints = environment.getDim() == 2
    fig_local = plt.figure(figsize=(14, 7) if twoJoints else (7, 7))
    ax1 = fig_local.add_subplot(1, 2 if twoJoints else 1, 1)
    ax2 = fig_local.add_subplot(1, 2, 2) if twoJoints else None

    ## get positions for solution and interpolate to obtain a smoother movement
    solution_pos = [planner.graph.nodes[node]['pos'] for node in solution]
    i_solution_pos = [solution_pos[0]]
    for i in range(1, len(solution_pos)):
        i_solution_pos = i_solution_pos + interpolate_line(solution_pos[i-1], solution_pos[i], 0.1)[1:]

    r = environment.kin_chain

    def animate(t):
        ## work space: obstacles and robot
        ax1.cla()
        ax1.set_xlim(workSpaceLimits[0])
        ax1.set_ylim(workSpaceLimits[1])
        ax1.set_aspect("equal")
        environment.drawObstacles(ax1, inWorkspace=True)
        r.move(i_solution_pos[t])
        planarRobotVisualize(r, ax1)
        if ax2 is not None:
            ## joint space: graph, path and current configuration
            ax2.cla()
            visualizer(planner, solution, ax2)
            ax2.scatter(i_solution_pos[t][0], i_solution_pos[t][1], color='r', zorder=10, s=250)

    ani = matplotlib.animation.FuncAnimation(fig_local, animate, frames=len(i_solution_pos))
    display(HTML(ani.to_jshtml()))
    plt.close(fig_local)
