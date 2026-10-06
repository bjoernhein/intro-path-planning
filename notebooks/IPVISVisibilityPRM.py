# coding: utf-8

"""
This code is part of the course 'Innovative Programmiermethoden für Industrieroboter' (Author: Bjoern Hein). It is based on the slides given during the course, so please **read the information in theses slides first**

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import networkx as nx

from IPVISStyle import EDGE_COLOR, MUTED, SPECIAL_COLOR, drawGraph, drawLargestComponent, drawSolution, drawStartGoal


def visibilityPRMVisualize(planner, solution, ax = None, nodeSize = 300):
    """ Draw the roadmap of the Visibility-PRM: guard nodes (violet) and connection nodes (light blue).
    The visibility tests recorded by the statistics handler are drawn in the background (gray),
    the largest connected component is underlaid in yellow.
    """
    graph = planner.graph
    statsHandler = planner.statsHandler
    planner._collisionChecker.drawObstacles(ax)

    # get a list of positions of all nodes by returning the content of the attribute 'pos'
    pos = nx.get_node_attributes(graph,'pos')

    if statsHandler:
        statPos = nx.get_node_attributes(statsHandler.graph,'pos')
        nx.draw_networkx_edges(statsHandler.graph, statPos, ax=ax, edge_color=MUTED, alpha=0.25)
        nx.draw_networkx_nodes(statsHandler.graph, statPos, ax=ax, node_size=nodeSize * 0.5, node_color=MUTED,
                               alpha=0.25)

    drawLargestComponent(graph, pos, ax)
    drawGraph(graph, pos, ax, nodeSize)
    guards = [node for node, nodeType in graph.nodes(data='nodeType') if nodeType == 'Guard']
    nx.draw_networkx_nodes(graph, pos, nodelist=guards, ax=ax, node_size=nodeSize, node_color=SPECIAL_COLOR,
                           edgecolors=EDGE_COLOR, linewidths=0.5)
    drawSolution(graph, pos, solution, ax)
    drawStartGoal(graph, pos, ax, nodeSize)
