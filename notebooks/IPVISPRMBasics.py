# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

Visualization helpers for the notebook IP-5-0-PRM_Basics: the Basic-PRM step by step.

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import math
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from shapely import affinity, plotting
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

# colors used for all figures of the notebook (common style of all notebooks, see IPVISStyle.py)
from IPVISStyle import (COLLISION_COLOR, COMPONENT_COLOR, EDGE_COLOR, FREE_COLOR, GOAL_COLOR, MUTED, NODE_COLOR,
                        PATH_COLOR, START_COLOR)

SAMPLE_COLOR = FREE_COLOR     # newly generated configuration and its neighbourhood


# ---------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------

def _rounded(polygon, r=0.25):
    """Round the convex and the concave corners of a polygon."""
    return polygon.buffer(-r).buffer(2 * r).buffer(-r)


def lectureScene():
    """Obstacles similar to the example of the lecture slides. Returns (scene, limits)."""
    limits = [[0.0, 22.0], [0.0, 11.0]]
    scene = dict()

    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 11)[:-1]
    radii = [2.3 if i % 2 == 0 else 1.0 for i in range(10)]
    star = Polygon([(4.3 + r * np.cos(a), 7.4 + r * np.sin(a)) for a, r in zip(angles, radii)])
    scene["star"] = _rounded(affinity.rotate(star, 12, origin="centroid"), 0.2)

    scene["triangle"] = _rounded(Polygon([(9.6, 6.5), (9.6, 7.7), (11.3, 6.5)]), 0.2)

    ellipse = affinity.scale(Point(10.3, 2.6).buffer(1.0), 2.2, 1.15)
    scene["ellipse"] = affinity.rotate(ellipse, 20, origin="centroid")

    arms = unary_union([affinity.rotate(box(15.2, 7.6, 19.2, 8.6), angle, origin=(17.2, 8.1)) for angle in (45, -45)])
    body = affinity.rotate(affinity.scale(Point(18.0, 5.6).buffer(1.0), 2.0, 1.2), -15, origin="centroid")
    scene["cross"] = _rounded(unary_union([arms, box(15.8, 5.6, 18.6, 8.1), body]), 0.2)
    return scene, limits


# ---------------------------------------------------------------------------
# Basic PRM with recording of every step
# ---------------------------------------------------------------------------

def traceBasicPRM(collisionChecker, numNodes, radius, maxNeighbours=3, seed=None, collisionCheckingSteps=40):
    """Basic PRM as implemented in the notebook (learnRoadmapNearestNeighbour), but every step is recorded.

    Returns the roadmap and a list of events:
    {"type": "sample", "node": id, "pos": [x, y], "neighbours": [ids sorted by distance]}
    {"type": "connect", "node": id, "neighbour": id, "result": "added" | "collision" | "sameComponent"}
    """
    rng = random.Random(seed)
    limits = collisionChecker.getEnvironmentLimits()
    graph = nx.Graph()
    events = []
    for nodeID in range(1, numNodes + 1):
        pos = [rng.uniform(low, high) for low, high in limits]
        while collisionChecker.pointInCollision(pos):
            pos = [rng.uniform(low, high) for low, high in limits]

        candidates = sorted((math.dist(p, pos), n) for n, p in graph.nodes(data="pos") if math.dist(p, pos) < radius)
        neighbours = [n for _, n in candidates[:maxNeighbours]]
        graph.add_node(nodeID, pos=pos)
        events.append({"type": "sample", "node": nodeID, "pos": pos, "neighbours": neighbours})

        for n in neighbours:
            neighbourPos = graph.nodes[n]["pos"]
            if nx.has_path(graph, nodeID, n):
                result = "sameComponent"
            elif collisionChecker.lineInCollision(pos, neighbourPos, steps=collisionCheckingSteps):
                result = "collision"
            else:
                graph.add_edge(nodeID, n, weight=math.dist(pos, neighbourPos))
                result = "added"
            events.append({"type": "connect", "node": nodeID, "neighbour": n, "result": result})
    return graph, events


def graphAtStep(events, step):
    """Roadmap after the given event."""
    graph = nx.Graph()
    for event in events[:step + 1]:
        if event["type"] == "sample":
            graph.add_node(event["node"], pos=event["pos"])
        elif event["result"] == "added":
            p, q = graph.nodes[event["node"]]["pos"], graph.nodes[event["neighbour"]]["pos"]
            graph.add_edge(event["node"], event["neighbour"], weight=math.dist(p, q))
    return graph


def keySteps(events, minLength=2.5):
    """Index of the first event of every typical situation (used for the static figures).
    Connections shorter than minLength are skipped, so the situation is clearly visible."""
    steps = dict()
    numNodes = 0
    graph = nx.Graph()
    for k, event in enumerate(events):
        if event["type"] == "sample":
            numNodes += 1
            graph.add_node(event["node"], pos=event["pos"])
            if numNodes == 1:
                steps.setdefault("first", k)
            elif not event["neighbours"]:
                steps.setdefault("noNeighbours", k)
            else:
                steps.setdefault("neighbours", k)
            continue
        visible = math.dist(graph.nodes[event["node"]]["pos"], graph.nodes[event["neighbour"]]["pos"]) >= minLength
        if event["result"] == "added":
            nodeComponent = nx.node_connected_component(graph, event["node"])
            neighbourComponent = nx.node_connected_component(graph, event["neighbour"])
            if visible and len(nodeComponent) > 1 and len(neighbourComponent) > 1:
                steps.setdefault("merge", k)
            if visible:
                steps.setdefault("added", k)
            graph.add_edge(event["node"], event["neighbour"])
        elif visible:
            steps.setdefault(event["result"], k)
    steps["last"] = len(events) - 1
    return steps


CAPTIONS = {
    "first": "first free configuration: no neighbours yet",
    "noNeighbours": "new configuration: no nodes within the radius",
    "neighbours": "new configuration with neighbours $V_c$ in the radius",
    "collision": "connection to a neighbour: collision, no edge",
    "added": "connection collision-free: edge is added",
    "merge": "new edge connects two components",
    "sameComponent": "neighbour already in the same component: not tested",
    "last": "roadmap at the end of the learning phase",
}


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def drawEnvironment(ax, collisionChecker, title=None):
    """Obstacles and axes set up for the environment."""
    collisionChecker.drawObstacles(ax)
    setupAxes(ax, collisionChecker, title)


def setupAxes(ax, collisionChecker, title=None):
    """Limits, equal aspect and muted axes (without drawing the obstacles)."""
    limits = collisionChecker.getEnvironmentLimits()
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    if title:
        ax.set_title(title, fontsize=11, loc="left")


def drawComponents(ax, graph, margin=0.35):
    """Every connected component with more than one node is shown as a light area around its edges."""
    pos = nx.get_node_attributes(graph, "pos")
    for component in nx.connected_components(graph):
        if len(component) < 2:
            continue
        area = unary_union([LineString([pos[u], pos[v]]).buffer(margin) for u, v in graph.subgraph(component).edges])
        plotting.plot_polygon(area, ax=ax, add_points=False, facecolor=to_rgba(COMPONENT_COLOR, 0.2),
                              edgecolor=to_rgba(COMPONENT_COLOR, 0.8), linewidth=1.0)


def drawRoadmap(ax, graph, nodeSize=40, edgeWidth=1.2):
    pos = nx.get_node_attributes(graph, "pos")
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=EDGE_COLOR, width=edgeWidth)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=nodeSize, node_color=NODE_COLOR, edgecolors=EDGE_COLOR,
                           linewidths=0.8)


def drawPRMState(ax, collisionChecker, events, step, radius, caption=None, highlight=True):
    """State of the learning phase after the given event (event `step` is highlighted)."""
    graph = graphAtStep(events, step)
    pos = nx.get_node_attributes(graph, "pos")
    event = events[step]
    node = event["node"]
    sample = next(e for e in events[:step + 1][::-1] if e["type"] == "sample" and e["node"] == node)

    drawEnvironment(ax, collisionChecker)
    drawComponents(ax, graph)
    drawRoadmap(ax, graph)
    if not highlight:
        ax.set_title(caption, fontsize=10, loc="left")
        return

    # current configuration c, its neighbourhood and its candidate neighbours V_c
    ax.add_patch(Circle(pos[node], radius, facecolor=to_rgba(SAMPLE_COLOR, 0.08), edgecolor=SAMPLE_COLOR,
                        linestyle="--", linewidth=1.0))
    if sample["neighbours"]:
        nx.draw_networkx_nodes(graph, pos, nodelist=sample["neighbours"], ax=ax, node_size=90, node_color="none",
                               edgecolors=SAMPLE_COLOR, linewidths=1.5)
    nx.draw_networkx_nodes(graph, pos, nodelist=[node], ax=ax, node_size=70, node_color=SAMPLE_COLOR,
                           edgecolors="white", linewidths=1.0)

    # tested connection
    if event["type"] == "connect":
        p, q = pos[node], pos[event["neighbour"]]
        style = {"added": dict(color=SAMPLE_COLOR, linestyle="-", linewidth=2.5),
                 "collision": dict(color=COLLISION_COLOR, linestyle="--", linewidth=2.0),
                 "sameComponent": dict(color=MUTED, linestyle=":", linewidth=2.0)}[event["result"]]
        ax.plot([p[0], q[0]], [p[1], q[1]], zorder=5, **style)

    text = caption if caption else f"step {step}: " + (
        f"new node {node}, {len(sample['neighbours'])} neighbour(s)" if event["type"] == "sample"
        else f"connect {node} - {event['neighbour']}: {event['result']}")
    ax.set_title(text, fontsize=10, loc="left")


def legendLearningPhase():
    return [Line2D([], [], marker="o", linestyle="", markerfacecolor=SAMPLE_COLOR, markeredgecolor="white",
                   markersize=8, label="new configuration $c$"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=SAMPLE_COLOR,
                   markersize=9, markeredgewidth=1.5, label="neighbours $V_c$ (max. 3)"),
            Line2D([], [], color=SAMPLE_COLOR, linestyle="--", linewidth=1.0, label="radius"),
            Line2D([], [], color=COLLISION_COLOR, linestyle="--", linewidth=2.0, label="connection: collision"),
            Line2D([], [], color=MUTED, linestyle=":", linewidth=2.0, label="not tested: same component"),
            Patch(facecolor=to_rgba(COMPONENT_COLOR, 0.15), edgecolor=to_rgba(COMPONENT_COLOR, 0.7),
                  label="connected component")]


def showKeySteps(collisionChecker, events, radius, situations=None, columns=2, panelWidth=6.0):
    """Static figure: one panel per typical situation of the learning phase."""
    steps = keySteps(events)
    situations = situations if situations else sorted((s for s in CAPTIONS if s in steps), key=lambda s: steps[s])
    rows = math.ceil(len(situations) / columns)
    limits = collisionChecker.getEnvironmentLimits()
    aspect = (limits[1][1] - limits[1][0]) / (limits[0][1] - limits[0][0])
    fig, axes = plt.subplots(rows, columns, figsize=(columns * panelWidth, rows * panelWidth * aspect + 0.9),
                             squeeze=False)
    for ax, situation in zip(axes.flat, situations):
        drawPRMState(ax, collisionChecker, events, steps[situation], radius,
                     caption=f"{situations.index(situation) + 1}. {CAPTIONS[situation]}",
                     highlight=situation != "last")
    for ax in list(axes.flat)[len(situations):]:
        ax.set_visible(False)
    fig.legend(handles=legendLearningPhase(), loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    plt.show()


def plotStep(collisionChecker, events, step, radius, caption=None, legend=True, highlight=True, figsize=(9.5, 4.2)):
    """One step of the learning phase (also used for the interactive slider)."""
    fig, ax = plt.subplots(figsize=figsize)
    drawPRMState(ax, collisionChecker, events, step, radius, caption=caption, highlight=highlight)
    if legend:
        ax.legend(handles=legendLearningPhase(), loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False,
                  fontsize=9)
    fig.tight_layout()
    plt.show()


def plotSituation(collisionChecker, events, situation, radius, legend=False):
    """The first step of a typical situation of the learning phase, see keySteps()."""
    plotStep(collisionChecker, events, keySteps(events)[situation], radius, caption=CAPTIONS[situation], legend=legend,
             highlight=situation != "last")


def plotEnvironment(collisionChecker, title=None, figsize=(7, 3.8)):
    fig, ax = plt.subplots(figsize=figsize)
    drawEnvironment(ax, collisionChecker, title)
    fig.tight_layout()
    return ax


def showNeighbours(collisionChecker, graph, pos, radius, result, figsize=(7, 3.8)):
    """Result of a nearest neighbour search: the found neighbours are numbered by increasing distance."""
    ax = plotEnvironment(collisionChecker, f"neighbours of the new configuration (radius {radius})", figsize)
    drawRoadmap(ax, graph)
    ax.add_patch(Circle(pos, radius, facecolor=to_rgba(SAMPLE_COLOR, 0.08), edgecolor=SAMPLE_COLOR, linestyle="--"))
    ax.scatter(*pos, s=70, color=SAMPLE_COLOR, edgecolor="white", zorder=5)
    for k, (distance, (node, data)) in enumerate(result, start=1):
        ax.scatter(*data["pos"], s=110, facecolor="none", edgecolor=SAMPLE_COLOR, linewidth=1.5, zorder=5)
        ax.annotate(str(k), data["pos"], xytext=(5, 5), textcoords="offset points", fontsize=11, color=SAMPLE_COLOR)
    plt.show()


# ---------------------------------------------------------------------------
# query phase
# ---------------------------------------------------------------------------

def connectToRoadmap(graph, pos, collisionChecker, radius, collisionCheckingSteps=40):
    """Nearest roadmap node within the radius, which can be connected collision-free (None if there is none)."""
    candidates = sorted((math.dist(p, pos), n) for n, p in graph.nodes(data="pos") if math.dist(p, pos) < radius)
    for _, n in candidates:
        if not collisionChecker.lineInCollision(pos, graph.nodes[n]["pos"], steps=collisionCheckingSteps):
            return n
    return None


def query(graph, start, goal, collisionChecker, radius):
    """Query phase on a copy of the roadmap. Returns (graph, s_hat, g_hat, path)."""
    graph = graph.copy()
    sHat = connectToRoadmap(graph, start, collisionChecker, radius)
    gHat = connectToRoadmap(graph, goal, collisionChecker, radius)
    for name, p, hat in (("start", start, sHat), ("goal", goal, gHat)):
        graph.add_node(name, pos=p)
        if hat is not None:
            graph.add_edge(name, hat, weight=math.dist(p, graph.nodes[hat]["pos"]))
    try:
        path = nx.shortest_path(graph, "start", "goal", weight="weight")
    except nx.NetworkXNoPath:
        path = []
    return graph, sHat, gHat, path


def drawPath(ax, graph, path, width=4.0):
    pos = nx.get_node_attributes(graph, "pos")
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=list(zip(path[:-1], path[1:])), edge_color=PATH_COLOR,
                           width=width)


def drawStartGoal(ax, graph, nodeSize=120, labels=True):
    pos = nx.get_node_attributes(graph, "pos")
    for name, color, label in (("start", START_COLOR, "$s$"), ("goal", GOAL_COLOR, "$g$")):
        if name in graph:
            nx.draw_networkx_nodes(graph, pos, nodelist=[name], ax=ax, node_size=nodeSize, node_color=color,
                                   edgecolors="white", linewidths=1.0)
            if labels:
                ax.annotate(label, pos[name], xytext=(6, 6), textcoords="offset points", fontsize=12)


def showQueryPhase(collisionChecker, roadmap, start, goal, radius):
    """Left: start and goal connected to the roadmap. Right: path found by the graph search."""
    graph, sHat, gHat, path = query(roadmap, start, goal, collisionChecker, radius)
    pos = nx.get_node_attributes(graph, "pos")
    limits = collisionChecker.getEnvironmentLimits()
    aspect = (limits[1][1] - limits[1][0]) / (limits[0][1] - limits[0][0])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6 * aspect + 0.6))
    titles = (r"connect $s$ and $g$ to the nearest nodes $\hat{s}$ and $\hat{g}$", "graph search: path from $s$ to $g$")
    for k, ax in enumerate(axes):
        drawEnvironment(ax, collisionChecker, titles[k])
        drawRoadmap(ax, graph.subgraph([n for n in graph if n not in ("start", "goal")]))
        if k == 0:
            for name, hat, label in (("start", sHat, r"$\hat{s}$"), ("goal", gHat, r"$\hat{g}$")):
                if hat is not None:
                    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=[(name, hat)], edge_color=PATH_COLOR,
                                           width=2.5, style="--")
                    ax.annotate(label, pos[hat], xytext=(6, -14), textcoords="offset points", fontsize=12)
        elif path:
            drawPath(ax, graph, path)
        drawStartGoal(ax, graph)
    fig.tight_layout()
    plt.show()
    return graph, path


def pathLength(graph, path):
    return nx.path_weight(graph, path, weight="weight") if path else float("inf")


def showNotShortest(collisionChecker, roadmap, start, goal, radius, collisionCheckingSteps=40):
    """Basic PRM (no edges within a component: the roadmap is a forest) vs. the same nodes with all
    collision-free edges within the radius."""
    full = nx.Graph()
    full.add_nodes_from(roadmap.nodes(data=True))
    nodes = list(roadmap.nodes(data="pos"))
    for i, (n, p) in enumerate(nodes):
        for m, q in nodes[i + 1:]:
            if math.dist(p, q) < radius and not collisionChecker.lineInCollision(p, q, steps=collisionCheckingSteps):
                full.add_edge(n, m, weight=math.dist(p, q))

    limits = collisionChecker.getEnvironmentLimits()
    aspect = (limits[1][1] - limits[1][0]) / (limits[0][1] - limits[0][0])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6 * aspect + 0.6))
    for ax, graph, name in zip(axes, (roadmap, full), ("Basic PRM", "all collision-free edges in the radius")):
        g, sHat, gHat, path = query(graph, start, goal, collisionChecker, radius)
        drawEnvironment(ax, collisionChecker, f"{name}: path length {pathLength(g, path):.1f}")
        drawRoadmap(ax, graph)
        if path:
            drawPath(ax, g, path)
        drawStartGoal(ax, g)
        print(f"{name}: {graph.number_of_edges()} edges, {len(nx.cycle_basis(graph))} independent cycles, "
              f"path length {pathLength(g, path):.2f}")
    fig.tight_layout()
    plt.show()
