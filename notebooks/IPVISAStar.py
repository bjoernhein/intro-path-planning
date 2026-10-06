# coding: utf-8

"""
This code is part of the course 'Innovative Programmiermethoden für Industrieroboter' (Author: Bjoern Hein). It is based on the slides given during the course, so please **read the information in theses slides first**

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import networkx as nx

from IPVISStyle import EDGE_COLOR, MUTED, drawSolution, drawStartGoal


def aStarVisualize(planner, solution, ax = None, nodeSize = 300):
    """ Draw the search graph of A*: expanded nodes (closed list, gray) and nodes of the
    open list (white), the solution path and start/goal.
    """
    graph = planner.graph
    planner._collisionChecker.drawObstacles(ax)

    # get a list of positions of all nodes by returning the content of the attribute 'pos'
    pos = nx.get_node_attributes(graph,'pos')

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=EDGE_COLOR, width=1.0, arrows=False)
    for status, color in (("closed", MUTED), ("open", "white")):
        nodes = [node for node, attribute in graph.nodes(data=True) if attribute['status'] == status]
        nx.draw_networkx_nodes(graph, pos, nodelist=nodes, ax=ax, node_size=nodeSize, node_color=color,
                               edgecolors=EDGE_COLOR, linewidths=0.8)

    drawSolution(graph, pos, solution, ax, arrows=False)
    if solution:
        drawStartGoal(graph, pos, ax, nodeSize, start=solution[0], goal=solution[-1])
