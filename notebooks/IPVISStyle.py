# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

Common colors and drawing helpers for the visualizations of all notebooks:
obstacles gray, start green, goal red, solution path blue, collisions orange.

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import networkx as nx

from IPEnvironment import OBSTACLE_COLOR

NODE_COLOR = "#b7d3f6"        # nodes of a graph / roadmap
EDGE_COLOR = "#52514e"        # edges of a graph and outlines of nodes
START_COLOR = "#0ca30c"
GOAL_COLOR = "#d03b3b"
PATH_COLOR = "#2a78d6"        # solution path
FREE_COLOR = "#2a78d6"        # checked: collision-free
COLLISION_COLOR = "#eb6834"   # checked: in collision
COMPONENT_COLOR = "#eda100"   # (largest) connected component
SPECIAL_COLOR = "#4a3aa7"     # special nodes, e.g. guards of the Visibility-PRM
MUTED = "#898781"             # secondary information


def drawGraph(graph, pos, ax=None, nodeSize=100, nodelist=None, nodeColor=NODE_COLOR, edgeColor=EDGE_COLOR,
              edgeWidth=1.0, **edgeKwargs):
    """Edges and nodes of a graph in the common style."""
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edgeColor, width=edgeWidth, **edgeKwargs)
    nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=nodelist, node_size=nodeSize, node_color=nodeColor,
                           edgecolors=EDGE_COLOR, linewidths=0.5)


def drawEdges(edges, pos, ax=None, **kwargs):
    """Draw a list of edges (they do not have to be part of a graph any more)."""
    graph = nx.Graph()
    graph.add_nodes_from(pos)
    graph.add_edges_from(e for e in edges if e[0] in pos and e[1] in pos)
    nx.draw_networkx_edges(graph, pos, ax=ax, **kwargs)


def drawLargestComponent(graph, pos, ax=None, width=4.0):
    """Underlay the edges of the largest connected component (undirected graphs)."""
    if graph.number_of_edges() == 0:
        return
    largest = max(nx.connected_components(graph), key=len)
    nx.draw_networkx_edges(graph.subgraph(largest), pos, ax=ax, edge_color=COMPONENT_COLOR, alpha=0.3, width=width)


def drawSolution(graph, pos, solution, ax=None, width=4.0, **kwargs):
    """Edges of the solution path (list of nodes)."""
    if solution and len(solution) > 1:
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=list(zip(solution[:-1], solution[1:])),
                               edge_color=PATH_COLOR, width=width, **kwargs)


def drawStartGoal(graph, pos, ax=None, nodeSize=100, start="start", goal="goal"):
    """Start (green, "S") and goal (red, "G") node."""
    size = max(1.5 * nodeSize, 150)
    for node, color, label in ((start, START_COLOR, "S"), (goal, GOAL_COLOR, "G")):
        if node is not None and node in graph:
            nx.draw_networkx_nodes(graph, pos, nodelist=[node], ax=ax, node_size=size, node_color=color,
                                   edgecolors="white", linewidths=1.0)
            nx.draw_networkx_labels(graph, pos, labels={node: label}, ax=ax, font_size=8, font_color="white",
                                    font_weight="bold")
