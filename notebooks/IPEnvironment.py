# coding: utf-8

"""
This code is part of a series of notebooks regarding  "Introduction to robot path planning".

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from IPPerfMonitor import IPPerfMonitor

from shapely.geometry import Point, Polygon, LineString
from shapely import plotting

# color of the obstacles in all notebooks (start: green, goal: red)
OBSTACLE_COLOR = "#52514e"

class CollisionChecker(object):

    def __init__(self, scene, limits=None, statistic=None):
        self.scene = scene
        self.limits = [[0.0, 22.0], [0.0, 22.0]] if limits is None else limits
        self.statistic = statistic

    def getDim(self):
        """Return dimension of Environment (Shapely should currently always be 2)."""
        return 2

    def getEnvironmentLimits(self):
        """Return limits of Environment."""
        return list(self.limits)

    def isInLimits(self, pos):
        """Return whether a configuration lies inside the configured limits."""
        assert (len(pos) == self.getDim())
        return (self.limits[0][0] <= pos[0] <= self.limits[0][1]
                and self.limits[1][0] <= pos[1] <= self.limits[1][1])

    @IPPerfMonitor
    def pointInCollision(self, pos):
        """Return whether a configuration is invalid.
        Collision or outside limits -> True
        Free and inside limits -> False
        """
        assert (len(pos) == self.getDim())
        if not self.isInLimits(pos):
            return True
        for key, value in self.scene.items():
            if value.intersects(Point(pos[0], pos[1])):
                return True
        return False

    @IPPerfMonitor
    def lineInCollision(self, startPos, endPos, steps=40):
        """Check a line by sampling intermediate points.
        This illustrates local planner discretization, but it may miss thin obstacles.
        """
        assert (len(startPos) == self.getDim())
        assert (len(endPos) == self.getDim())
        dim = self.getDim()
        for i in range(steps + 1):
            t = i / steps
            testPoint = [startPos[d] + t * (endPos[d] - startPos[d]) for d in range(dim)]
            if self.pointInCollision(testPoint):
                return True
        return False

    @IPPerfMonitor
    def lineInCollisionExact(self, startPos, endPos):
        """Check whether the exact line segment from startPos to endPos collides."""
        assert (len(startPos) == self.getDim())
        assert (len(endPos) == self.getDim())
        if self.pointInCollision(startPos) or self.pointInCollision(endPos):
            return True
        line = LineString([(startPos[0], startPos[1]), (endPos[0], endPos[1])])
        for key, value in self.scene.items():
            if value.intersects(line):
                return True
        return False

    def drawObstacles(self, ax):
        for key, value in self.scene.items():
            plotting.plot_polygon(value, add_points=False, ax=ax, color=OBSTACLE_COLOR)
            
