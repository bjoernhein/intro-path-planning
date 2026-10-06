# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein). It is based on the slides given during the course, so please **read the information in theses slides first**

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import networkx as nx

from IPVISStyle import (COLLISION_COLOR, FREE_COLOR, MUTED, drawEdges, drawGraph, drawLargestComponent,
                        drawSolution, drawStartGoal)


def lazyPRMVisualize(planner, solution = [] , ax=None, nodeSize = 300):
    """ Draw the roadmap of the Lazy-PRM: edges not yet checked (thin, gray), edges checked
    collision-free (blue) and edges found in collision (orange, dashed; already removed from the roadmap).
    The largest connected component is underlaid in yellow.
    """
    graph = planner.graph
    planner._collisionChecker.drawObstacles(ax)

    # get a list of positions of all nodes by returning the content of the attribute 'pos'
    pos = nx.get_node_attributes(graph,'pos')

    drawLargestComponent(graph, pos, ax)
    drawEdges(planner.collidingEdges, pos, ax, edge_color=COLLISION_COLOR, width=2.5, style="dashed", alpha=0.8)
    drawEdges(planner.nonCollidingEdges, pos, ax, edge_color=FREE_COLOR, width=3.0, alpha=0.4)
    drawGraph(graph, pos, ax, nodeSize, edgeColor=MUTED, edgeWidth=0.8)
    drawSolution(graph, pos, solution, ax)
    drawStartGoal(graph, pos, ax, nodeSize)
