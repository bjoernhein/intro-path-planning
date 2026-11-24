# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

Figures of the notebook IP-10-0-PlanarManipulator: planar robots (IPPlanarManipulator.py) with the collision checker
KinChainCollisionChecker (IPEnvironmentKin.py) in the work space and in the joint space.

Robots with a link radius and the work space are drawn with the functions of IPVISBBMethod.py (drawPlanarRobot,
setupWorkspace).

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from IPVISBBMethod import drawPlanarRobot, interpolatePath, setupWorkspace
from IPVISStyle import (COLLISION_COLOR, COMPONENT_COLOR, EDGE_COLOR, FREE_COLOR, GOAL_COLOR, MUTED, NODE_COLOR,
                        PATH_COLOR, START_COLOR)


def configurationColor(collisionChecker, q, color=NODE_COLOR):
    """color, or orange if the configuration is in collision"""
    return COLLISION_COLOR if collisionChecker.robotInCollision(q) else color


def sampleLine(qa, qb, resolution):
    """Configurations on the line from qa to qb (both included) with a maximal distance of resolution"""
    n = max(1, math.ceil(math.dist(qa, qb) / resolution))
    return [[a + k / n * (b - a) for a, b in zip(qa, qb)] for k in range(n + 1)]


def tcpPath(collisionChecker, configurations):
    """Positions of the end of the last link for a list of configurations"""
    chain = collisionChecker.kin_chain
    points = []
    for q in configurations:
        chain.move(q)
        points.append(np.asarray(chain.get_transforms()[-1], dtype=float))
    return np.array(points)


# ---------------------------------------------------------------------------
# work space
# ---------------------------------------------------------------------------

def drawRobot(ax, collisionChecker, q, color=NODE_COLOR, alpha=1.0, zorder=5):
    """The robot at the configuration q: links as capsules (link_radius > 0) or as thick lines (link_radius = 0)"""
    if collisionChecker.link_radius > 0.0:
        drawPlanarRobot(ax, collisionChecker, q, color=color, alpha=alpha, zorder=zorder)
        return
    chain = collisionChecker.kin_chain
    chain.move(q)
    points = np.array(chain.get_transforms(), dtype=float)
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=4.0, alpha=alpha, solid_capstyle="round",
            solid_joinstyle="round", zorder=zorder)
    ax.scatter(points[1:-1, 0], points[1:-1, 1], s=30, color="white", edgecolors=color, linewidths=1.5, alpha=alpha,
               zorder=zorder + 0.1)


def drawMotion(ax, collisionChecker, path, poses=12, title=None, limits=None):
    """The robot at configurations with equal distances along the path: start green, goal red, in collision orange,
    the poses in between the more opaque, the later"""
    configurations = interpolatePath(path, poses)
    setupWorkspace(ax, collisionChecker, title, limits)
    for k, q in enumerate(configurations):
        first, last = k == 0, k == len(configurations) - 1
        color = START_COLOR if first else GOAL_COLOR if last else MUTED
        alpha = 1.0 if first or last else 0.25 + 0.5 * k / (len(configurations) - 1)
        drawRobot(ax, collisionChecker, q, color=configurationColor(collisionChecker, q, color), alpha=alpha,
                  zorder=20 if first or last else 5 + k)

def showConfigurations(collisionChecker, configurations, titles=None, colors=None, panelWidth=4.2, limits=None):
    """The robot at some configurations, one panel per configuration (orange: in collision)"""
    fig, axes = plt.subplots(1, len(configurations), figsize=(len(configurations) * panelWidth, panelWidth + 0.3))
    for k, (ax, q) in enumerate(zip(np.atleast_1d(axes), configurations)):
        title = titles[k] if titles else "q = [" + ", ".join(f"{value:.2f}" for value in q) + "]"
        setupWorkspace(ax, collisionChecker, title, limits)
        drawRobot(ax, collisionChecker, q,
                  color=configurationColor(collisionChecker, q, colors[k] if colors else PATH_COLOR))
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# joint space (C-space) of a robot with two joints
# ---------------------------------------------------------------------------

def setupCSpace(ax, collisionChecker, title=None, obstacles=True):
    """Axes of the joint space with the sampled C-obstacles (computed here, if not done yet).
    obstacles=False: they are already drawn, e.g. by the visualization of a planner"""
    if collisionChecker._cspace is None:
        collisionChecker.computeCSpace()
    if obstacles:
        collisionChecker.drawObstacles(ax)
    limits = collisionChecker.getEnvironmentLimits()
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_aspect("equal")
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, colors=MUTED)
    ax.set_xlabel("$q_1$ [rad]")
    ax.set_ylabel("$q_2$ [rad]")
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    if title:
        ax.set_title(title, fontsize=10, loc="left")


def drawConfigurationPoint(ax, q, color, label=None, size=110):
    ax.scatter([q[0]], [q[1]], s=size, color=color, edgecolors="white", linewidths=1.0, zorder=8)
    if label:
        ax.annotate(label, q[:2], xytext=(7, 7), textcoords="offset points", fontsize=10, color=EDGE_COLOR, zorder=9)


def showWorkAndCSpace(collisionChecker, configurations, labels=None, colors=None, panelWidth=5.2, limits=None,
                      titles=("work space", "joint space (C-space)")):
    """Left: the robot at the configurations in the work space. Right: the same configurations as points in the
    joint space with the sampled C-obstacles. Configurations in collision are drawn orange."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * panelWidth + 0.6, panelWidth))
    setupWorkspace(ax1, collisionChecker, titles[0], limits)
    setupCSpace(ax2, collisionChecker, titles[1])
    for k, q in enumerate(configurations):
        color = configurationColor(collisionChecker, q, colors[k] if colors else PATH_COLOR)
        drawRobot(ax1, collisionChecker, q, color=color, zorder=5 + k)
        drawConfigurationPoint(ax2, q, color, labels[k] if labels else None)
        if labels:   # label at the end of the first link
            collisionChecker.kin_chain.move(q)
            elbow = np.asarray(collisionChecker.kin_chain.get_transforms()[1], dtype=float)
            ax1.annotate(labels[k], elbow, xytext=(7, 7), textcoords="offset points", fontsize=10, color=EDGE_COLOR,
                         zorder=20)
    fig.tight_layout()
    plt.show()


def showLineCheck(collisionChecker, qa, qb, resolutions=(0.2, 0.01), panelWidth=5.0, limits=None):
    """The check of a line in the joint space by sampling, one row per resolution.
    Left: the sampled configurations in the joint space, right: the robot at these configurations and the path of
    its tip. Blue: free, orange: in collision."""
    fig, axes = plt.subplots(len(resolutions), 2, figsize=(2 * panelWidth + 0.6, len(resolutions) * panelWidth),
                             squeeze=False)
    fine = sampleLine(qa, qb, 0.005)
    tip = tcpPath(collisionChecker, fine)
    for (ax1, ax2), resolution in zip(axes, resolutions):
        samples = sampleLine(qa, qb, resolution)
        collisions = [collisionChecker.robotInCollision(q) for q in samples]
        result = "in collision" if any(collisions) else "free"
        setupCSpace(ax1, collisionChecker, f"resolution {resolution}: {len(samples)} checks, result: {result}")
        ax1.plot([qa[0], qb[0]], [qa[1], qb[1]], color=MUTED, linewidth=1.0, zorder=5)
        ax1.scatter([q[0] for q in samples], [q[1] for q in samples], s=18 if len(samples) < 60 else 5, zorder=6,
                    color=[COLLISION_COLOR if c else FREE_COLOR for c in collisions])
        setupWorkspace(ax2, collisionChecker, "work space: the robot at the checked configurations", limits)
        alpha = 0.8 if len(samples) < 30 else 0.06
        for q, collision in zip(samples, collisions):
            drawRobot(ax2, collisionChecker, q, color=COLLISION_COLOR if collision else FREE_COLOR,
                      alpha=0.9 if collision else alpha, zorder=7 if collision else 5)
        ax2.plot(tip[:, 0], tip[:, 1], color=EDGE_COLOR, linewidth=1.0, linestyle=":", zorder=9)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# results of planners
# ---------------------------------------------------------------------------

def solutionConfigurations(planner, solution):
    """The configurations of a solution path (list of nodes of planner.graph)"""
    return [list(planner.graph.nodes[node]["pos"]) for node in solution]


def drawTcpPath(ax, collisionChecker, path, resolution=0.02):
    """Path of the tip of the robot along a path in the joint space"""
    dense = [q for a, b in zip(path[:-1], path[1:]) for q in sampleLine(a, b, resolution)]
    if dense:
        tip = tcpPath(collisionChecker, dense)
        ax.plot(tip[:, 0], tip[:, 1], color=PATH_COLOR, linewidth=2.0, zorder=30)


def showSolution(collisionChecker, planner, solution, visualizer, title=None, poses=12, nodeSize=40, panelWidth=5.2,
                 limits=None):
    """Left: the graph of the planner and its solution in the joint space (drawn by its visualizer, e.g.
    IPVISLazyPRM.lazyPRMVisualize). Right: the robot along the solution path and the path of its tip (blue)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * panelWidth + 0.6, panelWidth))
    if collisionChecker._cspace is None:
        collisionChecker.computeCSpace()
    visualizer(planner, solution, ax=ax1, nodeSize=nodeSize)   # draws the C-obstacles via drawObstacles(ax)
    setupCSpace(ax1, collisionChecker, title or "joint space", obstacles=False)
    path = solutionConfigurations(planner, solution)
    if len(path) > 1:
        drawMotion(ax2, collisionChecker, path, poses, "work space: the robot along the path", limits)
        drawTcpPath(ax2, collisionChecker, path)
    else:
        setupWorkspace(ax2, collisionChecker, "work space: no path found", limits)
    fig.tight_layout()
    plt.show()


def showMotion(collisionChecker, path, title=None, poses=12, panelWidth=5.5, limits=None):
    """The robot along a path in the joint space (any number of joints) and the path of its tip"""
    fig, ax = plt.subplots(figsize=(panelWidth, panelWidth + 0.3))
    drawMotion(ax, collisionChecker, path, poses, title, limits)
    drawTcpPath(ax, collisionChecker, path)
    fig.tight_layout()
    plt.show()


def showMotions(collisionChecker, motions, poses=12, panelWidth=5.0, limits=None):
    """Several motions of the same robot side by side, e.g. before and after smoothing:
    motions = [(title, path), ...] with path as list of configurations"""
    fig, axes = plt.subplots(1, len(motions), figsize=(len(motions) * panelWidth, panelWidth + 0.3))
    for ax, (title, path) in zip(np.atleast_1d(axes), motions):
        drawMotion(ax, collisionChecker, path, poses, title, limits)
        drawTcpPath(ax, collisionChecker, path)
    fig.tight_layout()
    plt.show()


def tcpPathLength(collisionChecker, path, resolution=0.02):
    """Length of the path of the tip of the robot in the work space along a path in the joint space"""
    dense = [q for a, b in zip(path[:-1], path[1:]) for q in sampleLine(a, b, resolution)]
    tip = tcpPath(collisionChecker, dense)
    return float(np.linalg.norm(np.diff(tip, axis=0), axis=1).sum())


def showJointAngles(path, count=100, figsize=(7, 3.2), titles=None):
    """The joint angles along a path (list of configurations). A list of paths gives one panel per path with the
    same scale (titles: their titles), e.g. before and after smoothing."""
    paths = path if isinstance(path[0][0], (list, tuple, np.ndarray)) else [path]
    fig, axes = plt.subplots(1, len(paths), figsize=(figsize[0] * len(paths) / (1 if len(paths) == 1 else 1.4),
                                                    figsize[1]), sharey=True, squeeze=False)
    for k, (ax, single) in enumerate(zip(axes[0], paths)):
        configurations = np.array(interpolatePath(single, count))
        for joint in range(configurations.shape[1]):
            ax.plot(np.linspace(0, 1, count), configurations[:, joint], label=f"$q_{joint + 1}$")
        ax.set_xlabel("path parameter")
        ax.grid(linewidth=0.3)
        if titles:
            ax.set_title(titles[k], fontsize=10, loc="left")
    axes[0][0].set_ylabel("joint angle [rad]")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center", ncol=len(paths[0][0]), fontsize=9,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()
