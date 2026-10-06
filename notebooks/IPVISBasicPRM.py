# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import networkx as nx

from IPVISStyle import drawGraph, drawLargestComponent, drawSolution, drawStartGoal


def basicPRMVisualize(planner, solution, ax = None, nodeSize = 100):
    """ Draw graph, obstacles and solution in a axis environment of matplotib.
    The largest connected component is underlaid in yellow.
    """
    graph = planner.graph
    planner._collisionChecker.drawObstacles(ax)

    # get a list of positions of all nodes by returning the content of the attribute 'pos'
    pos = nx.get_node_attributes(graph,'pos')
    _drawRoadmap(graph, pos, solution, ax, nodeSize)


def basicPRMVisualizeWspace(planner, solution, ax = None, nodeSize = 100):
    """ Same as basicPRMVisualize, but only the first two dimensions of the configurations
    are drawn (e.g. the position of a shape robot in the work space).
    """
    graph = planner.graph
    planner._collisionChecker.drawObstacles(ax)

    pos = {node: (p[0], p[1]) for node, p in nx.get_node_attributes(graph,'pos').items()}
    _drawRoadmap(graph, pos, solution, ax, nodeSize)


def _drawRoadmap(graph, pos, solution, ax, nodeSize):
    drawLargestComponent(graph, pos, ax)
    drawGraph(graph, pos, ax, nodeSize)
    drawSolution(graph, pos, solution, ax)
    drawStartGoal(graph, pos, ax, nodeSize)
