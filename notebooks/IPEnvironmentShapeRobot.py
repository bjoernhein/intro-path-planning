# coding: utf-8

"""
Planar robot environment helpers used within the introduction to robot path planning notebooks.

This code is part of a series of notebooks regarding  "Introduction to robot path planning".

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import math

from shapely import plotting
from shapely.affinity import rotate, translate

from IPEnvironment import CollisionChecker
from IPPerfMonitor import IPPerfMonitor


class ShapeRobot:
    """Simple planar robot wrapper that stores a Shapely geometry."""

    def __init__(self, geometry, limits=[[0.0, 22.0], [0.0, 22.0]]):
        self.geometry = geometry
        self._pose = [0.0, 0.0]
        # Store workspace limits so they can be reused by the collision checker.
        self.limits = [list(bound) for bound in limits]

    def setTo(self, pos): 
        x_pos, y_pos = pos
        """Move the robot geometry to a new absolute position."""
        dx = x_pos - self._pose[0]
        dy = y_pos - self._pose[1]
        self.geometry = translate(self.geometry, xoff=dx, yoff=dy)
        self._pose = (x_pos, y_pos)
        
    def getLimits(self):
        """Return the configuration limits of the robot."""
        return self.limits

    def drawRobot(self, ax, color="blue", **kwargs):
        """Draw the current robot geometry on the provided axes."""
        plotting.plot_polygon(self.geometry, ax=ax, add_points=False, color=color, **kwargs)
        
class ShapeRobotWithOrientation(ShapeRobot):
    """Planar robot wrapper that also tracks orientation (theta)."""

    def __init__(
        self,
        geometry,
        limits=[[0.0, 22.0], [0.0, 22.0], [-math.pi, math.pi]],
        anchor=(0.0, 0.0),
        use_radians=True,
    ):
        super().__init__(geometry, limits=limits)
        self._template = geometry
        self._anchor = anchor
        self._orientation = 0.0
        self._use_radians = use_radians

    def setTo(self, pos):
        x_pos, y_pos, orientation = pos
        """Move the robot to an absolute pose (x, y, theta)."""
        if orientation == self._orientation:
            dx = x_pos - self._pose[0]
            dy = y_pos - self._pose[1]
            if dx == 0.0 and dy == 0.0:
                return
            self.geometry = translate(self.geometry, xoff=dx, yoff=dy)
        else:
            rotated = rotate(
                self._template,
                orientation,
                origin=self._anchor,
                use_radians=self._use_radians,
            )
            dx = x_pos - self._anchor[0]
            dy = y_pos - self._anchor[1]
            if dx != 0.0 or dy != 0.0:
                rotated = translate(rotated, xoff=dx, yoff=dy)
            self.geometry = rotated
        self._pose = (x_pos, y_pos)
        self._orientation = orientation
        


class CollisionCheckerShapeRobot(CollisionChecker):
    """Collision checker that evaluates intersections against a movable robot geometry."""

    def __init__(self, scene, robot, limits=None, statistic=None):
        if limits is None:
            limits = getattr(robot, "limits", None)
        if limits is None:
            limits = [[0.0, 22.0], [0.0, 22.0]]
        super().__init__(scene=scene, limits=limits, statistic=statistic)
        self.robot = robot
        self.dim = len(self.limits)
        
    def getDim(self):
        return self.dim
        
    def drawObstacles(self, ax):
        """Draw all obstacles plus the robot at its current configuration."""
        super().drawObstacles(ax)
        self.robot.drawRobot(ax)
        
    def getEnvironmentLimits(self): 
        """Return the configuration limits of the robot.
           Currently this seems not optimal, as there are two sources of limits (robot and environment),
           but for now this is kept for compatibility reasons. Dominating limits are the ones of the robot in any case."""
        return self.robot.getLimits()

    @IPPerfMonitor
    def pointInCollision(self, pos):
        """Check whether the robot placed at the given position collides with the scene."""
        assert len(pos) == self.getDim()
        self.robot.setTo(pos)
        for obstacle in self.scene.values():
            if obstacle.intersects(self.robot.geometry):
                return True
        return False
