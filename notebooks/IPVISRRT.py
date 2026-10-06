# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import networkx as nx

from IPVISStyle import drawGraph, drawSolution, drawStartGoal


def rrtPRMVisualize(planner, solution, ax = None, nodeSize = 300):
    """ Draw graph, obstacles and solution in a axis environment of matplotib.
    """
    graph = planner.graph
    planner._collisionChecker.drawObstacles(ax)

    # get a list of positions of all nodes by returning the content of the attribute 'pos'
    pos = nx.get_node_attributes(graph,'pos')

    drawGraph(graph, pos, ax, nodeSize)
    drawSolution(graph, pos, solution, ax)
    drawStartGoal(graph, pos, ax, nodeSize)
