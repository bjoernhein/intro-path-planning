# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

Visualization of the smoothing (IPSmoothing.py) and helpers for the notebook IP-12-0-Smoothing.
A tested shortcut is given as dict (see Smoother._optimize), the figures work with the steps of Smoother as well as
with the steps of the implementation in the notebook.

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import math

import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import box

from IPSmoothing import pathLength, pointAt, positions, quality
from IPVISPRMBasics import drawEnvironment, setupAxes
from IPVISStyle import (COLLISION_COLOR, COMPONENT_COLOR, EDGE_COLOR, FREE_COLOR, GOAL_COLOR, MUTED, NODE_COLOR,
                        PATH_COLOR, SPECIAL_COLOR, START_COLOR, drawStartGoal)

MARK_COLOR = SPECIAL_COLOR     # l_rand and the worst path point p(l_w)
STRATEGY_COLORS = {"random": COMPONENT_COLOR, "latombe": SPECIAL_COLOR, "deterministic": PATH_COLOR}
STRATEGY_NAMES = {"random": "pure probabilistic", "latombe": "extended probabilistic (Latombe)",
                  "deterministic": "deterministic (Berchtold & Glavina)"}


# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------

def lectureExample():
    """Obstacle and path P1 ... P7 of the lecture slides: the lengths of the segments are chosen such that
    l_i = 0, 0.2, 0.4, 0.5, 0.6, 0.85, 1 as on the slides. Returns (scene, limits, path)."""
    scene = {"Obs": box(4.1, 4.25, 8.85, 8.4).buffer(-0.35).buffer(0.35)}
    path = [(1.3, 2.4)]
    for (dx, dy), length in zip([(1, 5.25), (-1, 4.15), (1, 0), (1.05, -2.15), (1, 0), (1.05, -4.15)],
                                [4.8, 4.8, 2.4, 2.4, 6.0, 3.6]):
        n = math.hypot(dx, dy)
        path.append((round(path[-1][0] + length * dx / n, 3), round(path[-1][1] + length * dy / n, 3)))
    return scene, [[0.0, 13.0], [0.0, 13.0]], path


# ---------------------------------------------------------------------------
# basic drawing
# ---------------------------------------------------------------------------

def drawPath(ax, path, color=EDGE_COLOR, width=2.0, nodeSize=30, linestyle="-", nodes=True, nodeColor=NODE_COLOR,
             zorder=4):
    """Path given as list of configurations"""
    xs, ys = [q[0] for q in path], [q[1] for q in path]
    ax.plot(xs, ys, color=color, linewidth=width, linestyle=linestyle, solid_capstyle="round", zorder=zorder)
    if nodes:
        ax.scatter(xs, ys, s=nodeSize, color=nodeColor, edgecolors=color, linewidths=0.8, zorder=zorder + 0.5)


def drawStartGoalPoints(ax, start, goal, size=110):
    for q, color in ((start, START_COLOR), (goal, GOAL_COLOR)):
        ax.scatter(q[0], q[1], s=size, color=color, edgecolor="white", linewidth=1.0, zorder=8)


def drawPoint(ax, q, label=None, color=MARK_COLOR, size=60, offset=(6, 6), zorder=9):
    ax.scatter(q[0], q[1], s=size, color=color, edgecolor=EDGE_COLOR, linewidth=1.0, zorder=zorder)
    if label:
        ax.annotate(label, (q[0], q[1]), xytext=offset, textcoords="offset points", fontsize=11, color=color,
                    zorder=zorder)


def subPath(path, lA, lB):
    """The part of the path from p(l_A) to p(l_B)"""
    ls = positions(path)
    qa, ka = pointAt(path, lA, ls)
    qb, kb = pointAt(path, lB, ls)
    return [qa] + [path[i] for i in range(ka + 1, kb + 1) if lA < ls[i] < lB] + [qb]


def drawQualities(ax, path, worst=None, fontsize=9):
    """Rating Q at every inner path point (worst point bold)"""
    for i in range(1, len(path) - 1):
        q = quality(path, i)
        if q < 1 + 1e-6:
            continue
        hot = worst is not None and tuple(path[i]) == tuple(worst)
        ax.annotate(f"{q:.2f}", path[i], xytext=(5, -12), textcoords="offset points", zorder=10,
                    fontsize=fontsize + (1 if hot else 0), color=MARK_COLOR if hot else MUTED,
                    fontweight="bold" if hot else "normal")


def showPath(collisionChecker, path, title=None, labels=False, figsize=(5.5, 5.2)):
    """A path; labels: P_i = p(l_i) at every point"""
    fig, ax = plt.subplots(figsize=figsize)
    drawEnvironment(ax, collisionChecker, title)
    drawPath(ax, path, nodeSize=40)
    if labels:
        for i, (q, l) in enumerate(zip(path, positions(path))):
            ax.annotate(f"$P_{i + 1}$ ({l:.2g})", q, xytext=(6, 4), textcoords="offset points", fontsize=10)
    plt.show()


def showQualities(collisionChecker, path, title=None, figsize=(5.5, 5.2)):
    """A path with the rating Q at every inner point"""
    fig, ax = plt.subplots(figsize=figsize)
    drawEnvironment(ax, collisionChecker, title)
    drawPath(ax, path, nodeSize=40)
    worst = max(range(1, len(path) - 1), key=lambda i: quality(path, i))
    drawQualities(ax, path, path[worst])
    plt.show()


# ---------------------------------------------------------------------------
# a tested shortcut
# ---------------------------------------------------------------------------

def drawShortcut(ax, previous, newPath, qa, qb, free, lA=None, lB=None):
    """Tested shortcut from p(l_A) to p(l_B): the replaced part of the path underlaid, free (blue) or in
    collision (orange, dashed); if free, the previous path is drawn muted and the new one on top"""
    if lA is not None:
        part = subPath(previous, lA, lB)
        ax.plot([q[0] for q in part], [q[1] for q in part], color=FREE_COLOR if free else COLLISION_COLOR,
                linewidth=9, alpha=0.2, solid_capstyle="round", zorder=3)
    if free:
        drawPath(ax, previous, color=MUTED, width=1.2, nodeSize=18, nodeColor="white", zorder=2)
        drawPath(ax, newPath, nodeSize=30)
        ax.plot([qa[0], qb[0]], [qa[1], qb[1]], color=FREE_COLOR, linewidth=3.0, zorder=6)
    else:
        drawPath(ax, previous, nodeSize=30)
        ax.plot([qa[0], qb[0]], [qa[1], qb[1]], color=COLLISION_COLOR, linewidth=2.5, linestyle="--", zorder=6)
    color = FREE_COLOR if free else COLLISION_COLOR
    ax.scatter([qa[0], qb[0]], [qa[1], qb[1]], s=30, color=color, edgecolor=EDGE_COLOR, linewidth=0.8, zorder=7)


def showShortcut(collisionChecker, path, lA, lB, newPath, free, figsize=(5.5, 5.2)):
    fig, ax = plt.subplots(figsize=figsize)
    qa, qb = pointAt(path, lA)[0], pointAt(path, lB)[0]
    drawEnvironment(ax, collisionChecker, f"shortcut ({lA:.2g}, {lB:.2g}): " + ("free" if free else "collision")
                    + f", length {pathLength(path):.1f} → {pathLength(newPath):.1f}")
    drawShortcut(ax, path, newPath, qa, qb, free, lA, lB)
    drawPoint(ax, qa, "$p(l_A)$", color=EDGE_COLOR, size=1, offset=(8, -14))
    drawPoint(ax, qb, "$p(l_B)$", color=EDGE_COLOR, size=1, offset=(6, 4))
    plt.show()


def stepTitle(step, k):
    """Title of the k-th tested shortcut"""
    result = "free" if step["free"] else "collision"
    strategy = step.get("strategy")
    if strategy == "latombe":
        how = f"$l_{{rand}} = {step['lRand']:.2f}$" + (", halved" if step["chosen"] == "halve" else "")
    elif strategy == "deterministic":
        how = "worst point" if step["chosen"] == "new" else "halved"
    else:
        how = f"$(l_A, l_B) = ({step['lA']:.2f}, {step['lB']:.2f})$"
    return f"check {k}: {how}: {result}\nlength {pathLength(step['path']):.1f}, {len(step['path'])} points"


def drawStep(ax, collisionChecker, step, qualities=False):
    drawShortcut(ax, step["previous"], step["path"], step["qa"], step["qb"], step["free"], step["lA"], step["lB"])
    if step.get("strategy") == "latombe" and step.get("lRand") is not None:
        drawPoint(ax, pointAt(step["previous"], step["lRand"])[0], "$p(l_{rand})$", size=45)
    if step.get("strategy") == "deterministic" and step.get("worst") is not None:
        drawPoint(ax, step["worst"], "$p(l_w)$", size=60)
        if qualities:
            drawQualities(ax, step["previous"], step["worst"])


def showSteps(collisionChecker, steps, initialPath, columns=3, panelWidth=4.0, qualities=False, count=None):
    """Small multiples: the path before and every tested shortcut (count: only the first ones)"""
    steps = steps[:count] if count else steps
    panels = len(steps) + 1
    rows = math.ceil(panels / columns)
    limits = collisionChecker.getEnvironmentLimits()
    aspect = (limits[1][1] - limits[1][0]) / (limits[0][1] - limits[0][0])
    fig, axes = plt.subplots(rows, columns, figsize=(columns * panelWidth, rows * (panelWidth * aspect + 0.6)),
                             squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= panels:
            ax.axis("off")
            continue
        if k == 0:
            drawEnvironment(ax, collisionChecker, f"path: length {pathLength(initialPath):.1f}, "
                                                  f"{len(initialPath)} points")
            drawPath(ax, initialPath, nodeSize=30)
            if qualities:
                drawQualities(ax, initialPath)
        else:
            drawEnvironment(ax, collisionChecker, stepTitle(steps[k - 1], k))
            drawStep(ax, collisionChecker, steps[k - 1], qualities)
    fig.tight_layout()
    plt.show()


def plotStep(collisionChecker, steps, initialPath, k, qualities=False, figsize=(6, 5.6)):
    """One tested shortcut (k = 0: the path before), e.g. for a slider"""
    fig, ax = plt.subplots(figsize=figsize)
    if k == 0:
        drawEnvironment(ax, collisionChecker, f"path: length {pathLength(initialPath):.1f}, "
                                              f"{len(initialPath)} points")
        drawPath(ax, initialPath, nodeSize=30)
    else:
        drawEnvironment(ax, collisionChecker, stepTitle(steps[k - 1], k))
        drawStep(ax, collisionChecker, steps[k - 1], qualities)
    plt.show()


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

def cspaceGrid(collisionChecker, n=150, reuse=None):
    """Sampled C-obstacles of a planar robot with two joints, only for the figures: calls
    KinChainCollisionChecker.computeCSpace(n) (n * n collision checks, see IP-10-0). Afterwards the collision checker
    draws its C-obstacles itself, so smoothingVisualize and the visualizations of the planners show them.
    reuse: a collision checker with the same robot, scene and limits, whose grid is taken instead (no checks)."""
    if reuse is not None:
        collisionChecker._cspace = reuse._cspace
        return collisionChecker._cspace
    return collisionChecker.computeCSpace(n)


def drawObstacles(ax, collisionChecker):
    """Obstacles of the collision checker (for a planar robot the sampled C-obstacles, if they were computed with
    computeCSpace)"""
    collisionChecker.drawObstacles(ax)


def drawCollisions(ax, path, check, width=3.0, zorder=6):
    """Segments of the path, which collide according to check (function (qa, qb) -> True if in collision)"""
    for a, b in zip(path[:-1], path[1:]):
        if check(list(a), list(b)):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=COLLISION_COLOR, linewidth=width, linestyle="--", zorder=zorder)


def smoothingVisualize(smoother, solution, ax=None, nodeSize=40, check=None):
    """Smoothed path (blue) and the path before (dashed), same interface as the visualizations of the planners.
    check (optional): function (qa, qb) -> True if in collision, e.g. an exact check; colliding segments of both paths
    are drawn orange."""
    graph = smoother.graph
    drawObstacles(ax, smoother._collisionChecker)
    if smoother.inputPath:
        drawPath(ax, smoother.inputPath, color=MUTED, width=1.2, nodeSize=12, linestyle="--", nodeColor="white",
                 zorder=2)
        if check is not None:
            drawCollisions(ax, smoother.inputPath, check, width=1.8, zorder=2.5)
    pos = nx.get_node_attributes(graph, "pos")
    drawPath(ax, smoother.path, color=PATH_COLOR, width=3.0, nodes=False)
    if check is not None:
        drawCollisions(ax, smoother.path, check)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=nodeSize, node_color=NODE_COLOR, edgecolors=EDGE_COLOR,
                           linewidths=0.8)
    drawStartGoal(graph, pos, ax, nodeSize)


def resultTitle(smoother, title=None):
    before, after = smoother.inputPath, smoother.path
    text = f"length {pathLength(before):.1f} → {pathLength(after):.1f}, points {len(before)} → {len(after)}"
    return text if title is None else f"{title}\n{text}"


def showResults(results, columns=None, panelWidth=4.5, check=None):
    """Results of several smoothings side by side: results = [(title, smoother, solution), ...]
    check (optional): colliding segments are drawn orange, see smoothingVisualize"""
    columns = columns or len(results)
    rows = math.ceil(len(results) / columns)
    limits = results[0][1]._collisionChecker.getEnvironmentLimits()
    aspect = (limits[1][1] - limits[1][0]) / (limits[0][1] - limits[0][0])
    fig, axes = plt.subplots(rows, columns, figsize=(columns * panelWidth, rows * (panelWidth * aspect + 0.7)),
                             squeeze=False)
    for ax, (title, smoother, solution) in zip(axes.flat, results):
        smoothingVisualize(smoother, solution, ax=ax, nodeSize=25, check=check)
        setupAxes(ax, smoother._collisionChecker, resultTitle(smoother, title))
    for ax in list(axes.flat)[len(results):]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def showPlannerResult(planner, solution, visualizer, path, title, figsize=(11, 5.2)):
    """Left: the result of the planner with its visualization, right: the path given to the smoothing"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    visualizer(planner, solution, ax=ax1, nodeSize=20)
    setupAxes(ax1, planner._collisionChecker, title)
    drawEnvironment(ax2, planner._collisionChecker, f"path: length {pathLength(path):.1f}, {len(path)} points")
    drawPath(ax2, path, color=PATH_COLOR, width=2.5, nodeSize=25)
    drawStartGoalPoints(ax2, path[0], path[-1])
    fig.tight_layout()
    plt.show()


def showComparison(curves, title=None, panelWidth=6.0):
    """Median (line) and 1st to 3rd quartile (band) over the number of segment collision checks:
    curves = {strategy: {"relative length": [(q1, median, q3), ...], "points": [...]}}"""
    keys = ("relative length", "points")
    fig, axes = plt.subplots(1, 2, figsize=(2 * panelWidth, 0.62 * panelWidth))
    for ax, key in zip(axes, keys):
        for strategy, data in curves.items():
            lo, mid, hi = zip(*data[key])
            x = range(len(mid))
            ax.fill_between(x, lo, hi, color=STRATEGY_COLORS[strategy], alpha=0.15, linewidth=0)
            ax.plot(x, mid, color=STRATEGY_COLORS[strategy], linewidth=2.2, label=STRATEGY_NAMES[strategy])
        ax.set_xlabel("number of segment collision checks")
        ax.set_title("relative path length $l / \\min(l)$" if key == keys[0] else "number of path points",
                     fontsize=11, loc="left")
        ax.grid(color="#e6e6e6", linewidth=0.8)
        ax.set_xlim(0, len(mid) - 1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    plt.show()
