# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

Figures of the notebook IP-10-2-ShapeRobot: robots with a shape (IPEnvironmentShapeRobot.py), which translate
(configuration [x, y]) or translate and rotate (configuration [x, y, theta]) in the plane.

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import animation
from matplotlib.colors import ListedColormap, to_rgba
from shapely import plotting
from shapely.affinity import scale, translate
from shapely.geometry import MultiPoint
from shapely.ops import unary_union

from IPEnvironment import OBSTACLE_COLOR
from IPVISStyle import (COLLISION_COLOR, COMPONENT_COLOR, EDGE_COLOR, GOAL_COLOR, MUTED, NODE_COLOR, PATH_COLOR,
                        START_COLOR, drawGraph, drawSolution)

COBSTACLE_COLOR = COMPONENT_COLOR     # C-obstacles (the obstacles themselves are gray)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def robotGeometry(collisionChecker, q):
    """The geometry of the robot at the configuration q (moves the robot of the collision checker)"""
    collisionChecker.robot.setTo(list(q))
    return collisionChecker.robot.geometry


def minkowskiSum(polygon, convexPolygon):
    """Minkowski sum of a polygon (may be non-convex, without holes) and a convex polygon.
    P + Q = (boundary of P + Q) united with P shifted by a point of Q; edge + Q is the convex hull of Q at both ends
    of the edge."""
    q = list(convexPolygon.exterior.coords)[:-1]
    parts = [translate(polygon, xoff=q[0][0], yoff=q[0][1])]
    boundary = list(polygon.exterior.coords)
    for a, b in zip(boundary[:-1], boundary[1:]):
        parts.append(MultiPoint([(p[0] + x, p[1] + y) for p in (a, b) for x, y in q]).convex_hull)
    return unary_union(parts)


def cObstacles(collisionChecker, theta=None):
    """The C-obstacles of a convex robot, which only translates (for a robot with orientation: at the orientation
    theta): every obstacle grown by the robot mirrored at its reference point (Minkowski sum).
    Returns one geometry (the union of all C-obstacles)."""
    origin = [0.0, 0.0] if theta is None else [0.0, 0.0, theta]
    mirrored = scale(robotGeometry(collisionChecker, origin), xfact=-1, yfact=-1, origin=(0, 0))
    return unary_union([minkowskiSum(obstacle, mirrored) for obstacle in collisionChecker.scene.values()])


def cspaceGrid(collisionChecker, resolution=120, theta=None):
    """Collision check on a grid of positions (for a robot with orientation: at the orientation theta).
    Returns grid[i, j]: True, if the robot at (x[j], y[i]) is in collision."""
    limits = collisionChecker.getEnvironmentLimits()
    xs = np.linspace(limits[0][0], limits[0][1], resolution)
    ys = np.linspace(limits[1][0], limits[1][1], resolution)
    rest = [] if theta is None else [theta]
    return np.array([[collisionChecker.robotInCollision([x, y] + rest) for x in xs] for y in ys])


def interpolatePath(path, stepSize):
    """Configurations along a path (list of configurations) with a maximal distance of stepSize"""
    result = [list(path[0])]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, math.ceil(math.dist(a, b) / stepSize))
        result += [[u + k / n * (v - u) for u, v in zip(a, b)] for k in range(1, n + 1)]
    return result


def solutionConfigurations(planner, solution):
    """The configurations of a solution path (list of nodes of planner.graph)"""
    return [list(planner.graph.nodes[node]["pos"]) for node in solution]


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def setupAxes(ax, collisionChecker, title=None):
    limits = collisionChecker.getEnvironmentLimits()
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_on()
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    if title:
        ax.set_title(title, fontsize=10, loc="left")


def drawObstacles(ax, collisionChecker):
    """Only the obstacles (collisionChecker.drawObstacles also draws the robot at its last configuration)"""
    for obstacle in collisionChecker.scene.values():
        plotting.plot_polygon(obstacle, ax=ax, add_points=False, color=OBSTACLE_COLOR)


def drawRobot(ax, collisionChecker, q, color=PATH_COLOR, alpha=0.8, reference=True, zorder=5):
    """The robot at the configuration q and its reference point"""
    plotting.plot_polygon(robotGeometry(collisionChecker, q), ax=ax, add_points=False, facecolor=to_rgba(color, alpha),
                          edgecolor=color, linewidth=1.2, zorder=zorder)
    if reference:
        ax.plot(q[0], q[1], "o", ms=4, color="white", mec=EDGE_COLOR, mew=0.8, zorder=zorder + 0.1)


def configurationColor(collisionChecker, q, color=PATH_COLOR):
    """color, or orange if the configuration is in collision"""
    return COLLISION_COLOR if collisionChecker.robotInCollision(list(q)) else color


def drawCObstacles(ax, collisionChecker, theta=None, outline=False):
    """The exact C-obstacles (Minkowski sum): filled, or only their outline"""
    geometry = cObstacles(collisionChecker, theta)
    if outline:
        plotting.plot_polygon(geometry, ax=ax, add_points=False, facecolor="none", edgecolor=EDGE_COLOR,
                              linewidth=1.2, linestyle="--", zorder=3)
    else:
        plotting.plot_polygon(geometry, ax=ax, add_points=False, facecolor=to_rgba(COBSTACLE_COLOR, 0.3),
                              edgecolor=COBSTACLE_COLOR, linewidth=1.0, zorder=1)


def drawGrid(ax, collisionChecker, grid):
    """Result of cspaceGrid: configurations in collision as colored cells"""
    limits = collisionChecker.getEnvironmentLimits()
    ax.imshow(grid, origin="lower", extent=(limits[0][0], limits[0][1], limits[1][0], limits[1][1]),
              cmap=ListedColormap([(1, 1, 1, 0), to_rgba(COBSTACLE_COLOR, 0.35)]), vmin=0, vmax=1,
              interpolation="nearest", zorder=1)


def showConfigurations(collisionChecker, configurations, title=None, labels=None, figsize=(5.5, 5.5)):
    """The robot at some configurations in the work space (orange: in collision)"""
    fig, ax = plt.subplots(figsize=figsize)
    drawObstacles(ax, collisionChecker)
    for k, q in enumerate(configurations):
        drawRobot(ax, collisionChecker, q, color=configurationColor(collisionChecker, q))
        if labels:
            ax.annotate(labels[k], q[:2], xytext=(8, 8), textcoords="offset points", fontsize=10, color=EDGE_COLOR,
                        zorder=9)
    setupAxes(ax, collisionChecker, title)
    fig.tight_layout()
    plt.show()


def showCSpace(collisionChecker, configurations=(), resolution=120, theta=None, panelWidth=5.2):
    """Left: the C-obstacles by sampling (collision check on a grid). Right: the exact C-obstacles (Minkowski sum).
    In both panels the obstacles of the work space in gray and the robot at the given configurations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * panelWidth, panelWidth + 0.3))
    grid = cspaceGrid(collisionChecker, resolution, theta)
    drawGrid(ax1, collisionChecker, grid)
    drawCObstacles(ax2, collisionChecker, theta)
    for ax, title in ((ax1, f"sampled: {grid.size} collision checks"), (ax2, "exact: Minkowski sum")):
        drawObstacles(ax, collisionChecker)
        for q in configurations:
            drawRobot(ax, collisionChecker, q, color=configurationColor(collisionChecker, q))
        setupAxes(ax, collisionChecker, title)
    fig.tight_layout()
    plt.show()
    return grid


def showSlices(collisionChecker, thetas, position=None, panelWidth=3.6):
    """The C-obstacles of a robot with orientation for some fixed orientations: slices of the 3D C-space.
    position: the robot is drawn there with the orientation of the slice"""
    fig, axes = plt.subplots(1, len(thetas), figsize=(len(thetas) * panelWidth, panelWidth + 0.4))
    for ax, theta in zip(np.atleast_1d(axes), thetas):
        drawCObstacles(ax, collisionChecker, theta)
        drawObstacles(ax, collisionChecker)
        if position is not None:
            q = list(position) + [theta]
            drawRobot(ax, collisionChecker, q, color=configurationColor(collisionChecker, q))
        setupAxes(ax, collisionChecker, f"$\\theta$ = {math.degrees(theta):.0f}°")
    fig.tight_layout()
    plt.show()


def drawRoadmap(ax, planner, solution, nodeSize=20):
    """Graph of a planner and its solution, projected on x and y"""
    pos = {node: (p[0], p[1]) for node, p in nx.get_node_attributes(planner.graph, "pos").items()}
    drawGraph(planner.graph, pos, ax, nodeSize, edgeColor=MUTED, edgeWidth=0.6)
    drawSolution(planner.graph, pos, solution, ax, width=3.0)


def drawMotion(ax, collisionChecker, path, stepSize=1.0):
    """The robot along a path: start green, goal red, the configurations in between blue (orange: in collision)"""
    configurations = interpolatePath(path, stepSize)
    for k, q in enumerate(configurations):
        first, last = k == 0, k == len(configurations) - 1
        color = START_COLOR if first else GOAL_COLOR if last else NODE_COLOR
        drawRobot(ax, collisionChecker, q, color=configurationColor(collisionChecker, q, color),
                  alpha=0.9 if first or last else 0.55, reference=False, zorder=8 if first or last else 5)
    points = np.array(interpolatePath(path, 0.05))
    ax.plot(points[:, 0], points[:, 1], color=PATH_COLOR, linewidth=2.0, zorder=7)


def showSolution(collisionChecker, planner, solution, title=None, cObstacle=True, stepSize=1.0, nodeSize=20,
                 panelWidth=5.2):
    """Left: the graph of the planner (projected on x and y) on the C-obstacles (only for a robot without
    orientation). Right: the robot along the solution path in the work space."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * panelWidth, panelWidth + 0.3))
    if cObstacle and collisionChecker.getDim() == 2:
        drawCObstacles(ax1, collisionChecker)
    drawObstacles(ax1, collisionChecker)
    drawRoadmap(ax1, planner, solution, nodeSize)
    setupAxes(ax1, collisionChecker, title or "roadmap")
    drawObstacles(ax2, collisionChecker)
    path = solutionConfigurations(planner, solution)
    if len(path) > 1:
        drawMotion(ax2, collisionChecker, path, stepSize)
    setupAxes(ax2, collisionChecker, "the robot along the path" if len(path) > 1 else "no path found")
    fig.tight_layout()
    plt.show()


def animateSolution(collisionChecker, path, stepSize=0.4, interval=80, figsize=(6, 6)):
    """Animation of the robot along a path (list of configurations).
    Show it in a notebook with: HTML(animateSolution(...).to_jshtml())"""
    configurations = interpolatePath(path, stepSize)
    points = np.array(configurations)
    fig, ax = plt.subplots(figsize=figsize)

    def drawFrame(frame):
        ax.cla()
        drawObstacles(ax, collisionChecker)
        ax.plot(points[:, 0], points[:, 1], color=MUTED, linewidth=1.0, zorder=3)
        ax.plot(points[:frame + 1, 0], points[:frame + 1, 1], color=PATH_COLOR, linewidth=2.0, zorder=4)
        q = configurations[frame]
        drawRobot(ax, collisionChecker, q, color=configurationColor(collisionChecker, q))
        setupAxes(ax, collisionChecker, f"configuration {frame + 1} of {len(configurations)}")
        return []

    result = animation.FuncAnimation(fig, drawFrame, frames=len(configurations), interval=interval)
    plt.close(fig)   # only the animation is shown, not the empty figure
    return result
