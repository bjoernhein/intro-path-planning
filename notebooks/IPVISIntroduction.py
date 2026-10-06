# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

Visualization helpers for the notebook IP-2-0-Pathplanning-Introduction (work space vs. configuration space).

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import heapq
import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch
from shapely import affinity, plotting
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from IPEnvironment import OBSTACLE_COLOR
from IPVISStyle import COLLISION_COLOR, FREE_COLOR, GOAL_COLOR, START_COLOR

# colors used for all figures of the notebook
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
OBSTACLE = OBSTACLE_COLOR
C_OBSTACLE = COLLISION_COLOR
START = START_COLOR
GOAL = GOAL_COLOR
FREE = FREE_COLOR
COLLISION = COLLISION_COLOR
OBSTACLE_COLORS = ["#2a78d6", "#eb6834", "#4a3aa7"]   # several obstacles, each with its own color


# ---------------------------------------------------------------------------
# general helpers
# ---------------------------------------------------------------------------

def styleAxes(ax, limits, title=None, xlabel="x", ylabel="y", grid=True):
    """Common look: equal aspect, light grid, muted spines."""
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=12, color=INK, loc="left")
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(INK_MUTED)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def setJointAngleTicks(ax, axis="both", dim3=False):
    """Ticks at multiples of pi/2 for joint angle axes."""
    ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    labels = [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]
    setters = []
    if axis in ("x", "both"):
        setters.append((ax.set_xticks, ax.set_xticklabels))
    if axis in ("y", "both"):
        setters.append((ax.set_yticks, ax.set_yticklabels))
    if dim3:
        setters.append((ax.set_zticks, ax.set_zticklabels))
    for setTicks, setLabels in setters:
        setTicks(ticks)
        setLabels(labels)


def drawGeometry(ax, geometry, facecolor, edgecolor=None, alpha=1.0, linewidth=0.0, zorder=1, **kwargs):
    """Draw a shapely (multi-)polygon."""
    geometries = getattr(geometry, "geoms", [geometry])
    for geom in geometries:
        if geom.is_empty or not isinstance(geom, Polygon):
            continue
        plotting.plot_polygon(geom, ax=ax, add_points=False, facecolor=facecolor,
                              edgecolor=edgecolor if edgecolor else facecolor,
                              alpha=alpha, linewidth=linewidth, zorder=zorder, **kwargs)


def drawStartGoal(ax, start, goal, size=90, labelOffset=(0.4, 0.4), zorder=10):
    """Start (green) and goal (red) marker, both labeled (start label above, goal label below the marker)."""
    for q, color, label, sign, va in ((start, START, r"$q_s$", 1, "bottom"), (goal, GOAL, r"$q_g$", -1, "top")):
        ax.scatter([q[0]], [q[1]], s=size, color=color, edgecolor="white", linewidth=1.5, zorder=zorder)
        ax.annotate(label, (q[0], q[1]), xytext=(q[0] + labelOffset[0], q[1] + sign * labelOffset[1]),
                    fontsize=12, color=INK, va=va, zorder=zorder)


def addArrowBetweenAxes(fig, axFrom, axTo, text, y=0.5, color=INK, textOffset=0.02):
    """Horizontal arrow (figure coordinates) in the gap between two neighbouring axes."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    toFigure = fig.transFigure.inverted()
    boxFrom = toFigure.transform_bbox(axFrom.get_tightbbox(renderer))
    boxTo = toFigure.transform_bbox(axTo.get_tightbbox(renderer))
    if boxFrom.x1 < boxTo.x0:
        x0, x1 = boxFrom.x1 + 0.01, boxTo.x0 - 0.01
    else:
        x0, x1 = boxFrom.x0 - 0.01, boxTo.x1 + 0.01
    fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=18, color=color, linewidth=1.5))
    fig.text((x0 + x1) / 2, y + textOffset, text, ha="center", va="bottom", fontsize=10, color=color)


# ---------------------------------------------------------------------------
# piano mover's problem
# ---------------------------------------------------------------------------

def pianoShape():
    """Top view of a grand piano (units: m). Keyboard along the local x-axis, the tail points to +y.
    The reference point (origin of the local frame) is roughly the center of the piano."""
    left = [(0.0, 0.0), (0.0, 1.75)]
    tail = [(0.45 + 0.42 * np.cos(a), 1.62 + 0.42 * np.sin(a)) for a in np.linspace(np.radians(160), 0, 12)]
    t = np.linspace(1, 0, 14)
    smooth = 3 * t ** 2 - 2 * t ** 3
    treble = [(1.5 - 0.63 * s, 0.35 + 1.27 * s) for s in smooth]
    outline = Polygon(left + tail + treble + [(1.5, 0.0)])
    keyboard = box(0.05, 0.0, 1.45, 0.22)
    center = (0.75, 0.85)
    return (affinity.translate(outline, -center[0], -center[1]),
            affinity.translate(keyboard, -center[0], -center[1]))


def placeShape(shape, q):
    """Move a shape given in local coordinates to configuration q = (x, y, theta)."""
    rotated = affinity.rotate(shape, q[2], origin=(0, 0), use_radians=True)
    return affinity.translate(rotated, q[0], q[1])


def interpolatePoses(waypoints, stepSize=0.05):
    """Linear interpolation of (x, y, theta) between waypoints."""
    poses = [np.array(waypoints[0], dtype=float)]
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        a, b = np.array(a, dtype=float), np.array(b, dtype=float)
        n = max(1, int(np.ceil(max(np.linalg.norm(b[:2] - a[:2]), abs(b[2] - a[2])) / stepSize)))
        poses += [a + (b - a) * i / n for i in range(1, n + 1)]
    return poses


def plotPianoMovers(roomObstacles, waypoints, roomLimits, numberOfPoses=6, figsize=(9, 6.2)):
    """Room (top view) with a grand piano moved along the waypoints. Poses in collision are drawn orange."""
    outline, keyboard = pianoShape()
    obstacles = unary_union(list(roomObstacles.values()))
    poses = interpolatePoses(waypoints)
    inCollision = [obstacles.intersects(placeShape(outline, q)) for q in poses]

    fig, ax = plt.subplots(figsize=figsize)
    styleAxes(ax, roomLimits, "Piano mover's problem (top view)", "x [m]", "y [m]")
    drawGeometry(ax, obstacles, OBSTACLE, zorder=2)

    # path of the reference point
    ax.plot([q[0] for q in poses], [q[1] for q in poses], color=INK_MUTED, linewidth=1.2,
            linestyle=(0, (4, 3)), zorder=3)

    # intermediate poses
    shown = np.linspace(0, len(poses) - 1, numberOfPoses + 2).round().astype(int)[1:-1]
    for i in shown:
        color = COLLISION if inCollision[i] else INK_MUTED
        drawGeometry(ax, placeShape(outline, poses[i]), "none", edgecolor=color, linewidth=1.0, zorder=4)

    # start and goal pose
    for q, color, label in ((poses[0], START, r"$q_s$"), (poses[-1], GOAL, r"$q_g$")):
        drawGeometry(ax, placeShape(outline, q), color, alpha=0.25, zorder=5)
        drawGeometry(ax, placeShape(outline, q), "none", edgecolor=color, linewidth=2, zorder=6)
        drawGeometry(ax, placeShape(keyboard, q), color, alpha=0.6, zorder=6)
        ax.scatter([q[0]], [q[1]], s=25, color=INK, zorder=7)
        ax.annotate(label, (q[0], q[1]), xytext=(q[0] + 0.12, q[1] - 0.32), fontsize=13, color=INK, zorder=7)

    # local frame at the start pose: x', y' and rotation theta
    q = poses[0]
    for angle, name in ((q[2], r"$x'$"), (q[2] + np.pi / 2, r"$y'$")):
        d = 0.75 * np.array([np.cos(angle), np.sin(angle)])
        ax.annotate("", xy=(q[0] + d[0], q[1] + d[1]), xytext=(q[0], q[1]),
                    arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.5), zorder=8)
        ax.text(q[0] + 1.2 * d[0], q[1] + 1.2 * d[1], name, fontsize=11, ha="center", va="center", zorder=8)
    ax.text(0.01, -0.13, r"configuration $q = (x, y, \theta)$: 3 DoF.  Dark gray: walls and furniture (static obstacles).",
            transform=ax.transAxes, fontsize=10, color=INK_SECONDARY)
    plt.show()
    if any(inCollision):
        print(f"{sum(inCollision)} of {len(poses)} interpolated poses are in collision (drawn orange).")


# ---------------------------------------------------------------------------
# point robot and disc robot
# ---------------------------------------------------------------------------

def plotPointRobot(scene, path, limits=((0, 22), (0, 22)), figsize=(11, 5.4)):
    """Point robot: work space and configuration space are identical."""
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    titles = (("W-space", "x", "y"), (r"C-space ($\equiv$ W-space)", r"$q_1 = x$", r"$q_2 = y$"))
    for ax, (title, xlabel, ylabel) in zip(axs, titles):
        styleAxes(ax, limits, title, xlabel, ylabel)
        for obstacle in scene.values():
            drawGeometry(ax, obstacle, OBSTACLE, zorder=2)
        ax.plot([p[0] for p in path], [p[1] for p in path], color=INK, linewidth=1.5,
                linestyle=(0, (4, 3)), zorder=5)
        drawStartGoal(ax, path[0], path[-1], labelOffset=(0.6, 0.6))
    fig.tight_layout()
    plt.show()


def plotDiscRobot(q=(6.0, 5.0), radius=2.0, limits=((0, 10), (0, 9)), figsize=(4.2, 3.9)):
    """Disc robot A(q): the configuration q = (x, y) is the center of the disc."""
    fig, ax = plt.subplots(figsize=figsize)
    styleAxes(ax, limits, None, "x", "y", grid=False)
    ax.add_patch(Circle(q, radius, facecolor=to_rgba(FREE, 0.18), edgecolor=FREE, linewidth=2))
    ax.plot([q[0], q[0]], [limits[1][0], q[1]], color=INK_MUTED, linestyle=(0, (3, 3)), linewidth=1)
    ax.plot([limits[0][0], q[0]], [q[1], q[1]], color=INK_MUTED, linestyle=(0, (3, 3)), linewidth=1)
    ax.annotate("", xy=(q[0] + radius, q[1]), xytext=q, arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.2))
    ax.text(q[0] + radius / 2, q[1] + 0.15, "r", ha="center", va="bottom", fontsize=11)
    ax.scatter([q[0]], [q[1]], s=30, color=INK, zorder=5)
    ax.text(q[0] - 0.2, q[1] + 0.25, r"$q = (x, y)$", ha="right", va="bottom", fontsize=10)
    ax.text(q[0], q[1] - radius * 0.6, r"$A(q)$", ha="center", va="center", fontsize=12, color=INK)
    ax.set_xticks([q[0]])
    ax.set_xticklabels(["x"])
    ax.set_yticks([q[1]])
    ax.set_yticklabels(["y"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(colors=INK, labelsize=11, length=0)
    fig.tight_layout()
    plt.show()


def plotMinkowskiSum(obstacle, radius=1.0, limits=((-0.5, 8.5), (0, 9)), figsize=(12.5, 4.4)):
    """W-space obstacle B_i + disc robot A -> C-space obstacle CB_i (obstacle grown by the radius)."""
    cObstacle = obstacle.buffer(radius)
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    styleAxes(axs[0], limits, r"W-space: obstacle $B_i$", "x", "y")
    drawGeometry(axs[0], obstacle, OBSTACLE, zorder=2)

    styleAxes(axs[1], limits, r"W-space: disc robot $A$", "x", "y")
    center = (np.mean(limits[0]), np.mean(limits[1]))
    axs[1].add_patch(Circle(center, radius, facecolor=to_rgba(FREE, 0.18), edgecolor=FREE, linewidth=2))
    axs[1].scatter([center[0]], [center[1]], s=25, color=INK, zorder=5)
    axs[1].annotate("", xy=(center[0] + radius, center[1]), xytext=center,
                    arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.2))
    axs[1].text(center[0] + radius / 2, center[1] + 0.12, "r", ha="center", va="bottom", fontsize=11)

    styleAxes(axs[2], limits, r"C-space: obstacle $CB_i$", r"$q_1 = x$", r"$q_2 = y$")
    drawGeometry(axs[2], cObstacle, C_OBSTACLE, alpha=0.85, zorder=2)
    drawGeometry(axs[2], obstacle, OBSTACLE, zorder=3)
    # robot placed so that it touches the obstacle: its center lies on the boundary of CB_i
    boundary = cObstacle.exterior
    for s in np.linspace(0, 1, 9)[:-1]:
        p = boundary.interpolate(s, normalized=True)
        axs[2].add_patch(Circle((p.x, p.y), radius, facecolor="none", edgecolor=FREE, linewidth=0.9, zorder=4))
        axs[2].scatter([p.x], [p.y], s=12, color=INK, zorder=5)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.45)
    addArrowBetweenAxes(fig, axs[1], axs[2], "", y=0.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    toFigure = fig.transFigure.inverted()
    left = toFigure.transform_bbox(axs[0].get_tightbbox(renderer)).x1
    right = toFigure.transform_bbox(axs[1].get_tightbbox(renderer)).x0
    fig.text((left + right) / 2, 0.5, "+", ha="center", va="center", fontsize=22, color=INK)
    plt.show()


def plotDiscRobotCSpace(scene, radius, start, goal, limits=((0, 22), (0, 22)), figsize=(11, 5.6)):
    """Disc robot in W-space and the corresponding point robot problem in C-space."""
    obstacles = unary_union(list(scene.values()))
    cObstacles = obstacles.buffer(radius)

    # is there a path at all? -> start and goal must lie in the same connected part of the free C-space
    freeSpace = box(limits[0][0], limits[1][0], limits[0][1], limits[1][1]).difference(cObstacles)
    parts = getattr(freeSpace, "geoms", [freeSpace])
    startPart = [i for i, part in enumerate(parts) if part.contains(Point(start))]
    goalPart = [i for i, part in enumerate(parts) if part.contains(Point(goal))]
    if not startPart or not goalPart:
        verdict = "start or goal in collision"
    elif startPart == goalPart:
        verdict = "a path exists"
    else:
        verdict = "no path: start and goal are separated"

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    styleAxes(axs[0], limits, f"W-space: disc robot (r = {radius})", "x", "y")
    drawGeometry(axs[0], obstacles, OBSTACLE, zorder=2)
    for q, color, label in ((start, START, r"$q_s$"), (goal, GOAL, r"$q_g$")):
        axs[0].add_patch(Circle(q, radius, facecolor=to_rgba(color, 0.25), edgecolor=color, linewidth=2, zorder=4))
        axs[0].scatter([q[0]], [q[1]], s=15, color=INK, zorder=5)
        axs[0].annotate(label, q, xytext=(q[0] + radius + 0.2, q[1] - 0.2), fontsize=12, zorder=5)

    styleAxes(axs[1], limits, f"C-space: point robot  ({verdict})", r"$q_1 = x$", r"$q_2 = y$")
    drawGeometry(axs[1], cObstacles, C_OBSTACLE, alpha=0.85, zorder=2)
    drawGeometry(axs[1], obstacles, OBSTACLE, zorder=3)
    drawStartGoal(axs[1], start, goal, size=60, labelOffset=(0.5, 0.4))

    handles = [Patch(color=OBSTACLE, label=r"obstacle $B_i$ (W-space)"),
               Patch(color=C_OBSTACLE, alpha=0.85, label=r"added by the radius: $CB_i$ = obstacle + orange band")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.subplots_adjust(wspace=0.4)
    addArrowBetweenAxes(fig, axs[0], axs[1], "grow\nobstacles", y=0.55)
    plt.show()


# ---------------------------------------------------------------------------
# planar robots
# ---------------------------------------------------------------------------

def drawPlanarRobot(ax, robot, q, color=INK, alpha=1.0, linewidth=4, zorder=6):
    """Links as thick lines, joints as small circles, base as a square."""
    robot.move(q)
    points = np.array(robot.get_transforms())
    ax.plot(points[:, 0], points[:, 1], color=color, alpha=alpha, linewidth=linewidth,
            solid_capstyle="round", zorder=zorder)
    ax.scatter(points[1:-1, 0], points[1:-1, 1], s=22, color="white", edgecolor=color, linewidth=1.3,
               alpha=alpha, zorder=zorder + 1)
    ax.scatter([points[0, 0]], [points[0, 1]], s=90, marker="s", color=INK_SECONDARY, zorder=zorder + 1)


def drawObstaclesColored(ax, obstacles, alpha=1.0, zorder=2):
    """Every obstacle gets its own color (the same color is used in the C-space)."""
    for (name, obstacle), color in zip(obstacles.items(), OBSTACLE_COLORS):
        drawGeometry(ax, obstacle, color, alpha=alpha, zorder=zorder)


def obstacleLegendHandles(obstacles):
    return [Patch(color=color, label=name) for name, color in zip(obstacles, OBSTACLE_COLORS)]


def drawCSpace2D(ax, axes, cspace, alpha=1.0):
    """C-space grid: 0 = free (white), i = configuration collides with obstacle i."""
    colors = ["white"] + OBSTACLE_COLORS[:int(cspace.max())]
    step = [(a[1] - a[0]) / 2 for a in axes]
    extent = [axes[0][0] - step[0], axes[0][-1] + step[0], axes[1][0] - step[1], axes[1][-1] + step[1]]
    ax.imshow(np.ma.masked_equal(cspace.T, 0), origin="lower", extent=extent, interpolation="nearest",
              cmap=ListedColormap(colors[1:]), vmin=1, vmax=max(1, len(colors) - 1), alpha=alpha, zorder=1)


def gridPath(axes, cspace, start, goal):
    """Shortest collision-free path on the sampled C-space grid (8-neighbourhood)."""
    def nearest(q):
        return tuple(int(np.argmin(np.abs(a - v))) for a, v in zip(axes, q))

    graph = nx.Graph()
    free = np.argwhere(cspace == 0)
    freeSet = set(map(tuple, free))
    moves = [m for m in itertools.product((-1, 0, 1), repeat=cspace.ndim) if any(m)]
    for cell in freeSet:
        for m in moves:
            neighbour = tuple(np.add(cell, m))
            if neighbour in freeSet:
                graph.add_edge(cell, neighbour, weight=float(np.linalg.norm(m)))
    s, g = nearest(start), nearest(goal)
    if s not in freeSet or g not in freeSet:
        raise ValueError("start or goal configuration is in collision")
    cells = nx.shortest_path(graph, s, g, weight="weight")
    return [list(start)] + [[axes[d][i] for d, i in enumerate(cell)] for cell in cells[1:-1]] + [list(goal)]


def plotPlanarRobotCSpace(robot, obstacles, axes, cspace, path, numberOfPoses=6,
                          workspaceLimits=((-3.5, 3.5), (-3.5, 3.5)), figsize=(11.5, 5.8)):
    """2-DoF planar robot: W-space with robot poses along the path, C-space with the sampled obstacles."""
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    shown = np.linspace(0, len(path) - 1, numberOfPoses + 2).round().astype(int)
    grays = plt.get_cmap("Greys")(np.linspace(0.35, 0.75, len(shown) - 2))

    styleAxes(axs[0], workspaceLimits, "W-space: robot with two rotational joints", "x", "y")
    drawObstaclesColored(axs[0], obstacles)
    for i, color in zip(shown[1:-1], grays):
        drawPlanarRobot(axs[0], robot, path[i], color=color, linewidth=2.5, zorder=4)
    drawPlanarRobot(axs[0], robot, path[0], color=START)
    drawPlanarRobot(axs[0], robot, path[-1], color=GOAL)
    for q, label in ((path[0], r"$q_s$"), (path[-1], r"$q_g$")):
        robot.move(q)
        tip = robot.get_transforms()[-1]
        axs[0].annotate(label, tip, xytext=(tip[0] + 0.15, tip[1] + 0.15), fontsize=12, zorder=10)

    styleAxes(axs[1], ((-np.pi, np.pi), (-np.pi, np.pi)), "C-space: sampled and checked for collision",
              r"$j_1$ [rad]", r"$j_2$ [rad]", grid=False)
    setJointAngleTicks(axs[1])
    drawCSpace2D(axs[1], axes, cspace)
    p = np.array(path)
    axs[1].plot(p[:, 0], p[:, 1], color=INK, linewidth=1.5, zorder=5)
    axs[1].scatter(p[shown[1:-1], 0], p[shown[1:-1], 1], s=30, color=grays, edgecolor=INK, linewidth=0.6, zorder=6)
    drawStartGoal(axs[1], path[0], path[-1], labelOffset=(0.12, 0.12))

    handles = obstacleLegendHandles(obstacles) + [
        Line2D([], [], color=INK, linewidth=1.5, label="path (point robot in C-space)")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plt.show()


def plotConfiguration(robot, obstacles, axes, cspace, q, workspaceLimits=((-3.5, 3.5), (-3.5, 3.5)),
                      figsize=(10, 5)):
    """One configuration q, shown in the W-space and in the C-space (e.g. for interactive sliders)."""
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    styleAxes(axs[0], workspaceLimits, "W-space", "x", "y")
    drawObstaclesColored(axs[0], obstacles)
    drawPlanarRobot(axs[0], robot, q, color=INK)
    styleAxes(axs[1], ((-np.pi, np.pi), (-np.pi, np.pi)), "C-space", r"$j_1$ [rad]", r"$j_2$ [rad]", grid=False)
    setJointAngleTicks(axs[1])
    drawCSpace2D(axs[1], axes, cspace)
    axs[1].scatter([q[0]], [q[1]], s=80, color=INK, edgecolor="white", linewidth=1.5, zorder=6)
    fig.legend(handles=obstacleLegendHandles(obstacles), loc="lower center", ncol=len(obstacles),
               frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plt.show()


def plotPlanarRobotCSpace3D(robot, obstacles, axes, cspace, start, goal,
                            workspaceLimits=((-5, 5), (-5, 5)), figsize=(12.5, 6)):
    """3-DoF planar robot: W-space (2D) and the C-space with the sampled obstacles (3D)."""
    fig = plt.figure(figsize=figsize)
    ax0 = fig.add_axes((0.06, 0.13, 0.37, 0.8))
    styleAxes(ax0, workspaceLimits, "W-space: robot with three rotational joints", "x", "y")
    drawObstaclesColored(ax0, obstacles)
    for q, color, label in ((start, START, r"$q_s$"), (goal, GOAL, r"$q_g$")):
        drawPlanarRobot(ax0, robot, q, color=color)
        tip = robot.get_transforms()[-1]
        ax0.annotate(label, tip, xytext=(tip[0] + 0.2, tip[1] + 0.2), fontsize=12, zorder=10)

    ax1 = fig.add_axes((0.47, 0.08, 0.5, 0.86), projection="3d")
    ax1.computed_zorder = False   # draw start and goal on top of the voxels
    edges = [np.concatenate(([a[0] - (a[1] - a[0]) / 2], (a[:-1] + a[1:]) / 2, [a[-1] + (a[1] - a[0]) / 2]))
             for a in axes]
    X, Y, Z = np.meshgrid(*edges, indexing="ij")
    facecolors = np.empty(cspace.shape, dtype=object)
    for k, color in enumerate(OBSTACLE_COLORS, start=1):
        facecolors[cspace == k] = color
    ax1.voxels(X, Y, Z, cspace > 0, facecolors=facecolors, edgecolor=None, alpha=0.55, shade=True, zorder=1)
    for q, color, label in ((start, START, r"$q_s$"), (goal, GOAL, r"$q_g$")):
        ax1.scatter([q[0]], [q[1]], [q[2]], s=80, color=color, edgecolor="white", linewidth=1.5,
                    depthshade=False, zorder=10)
        ax1.text(q[0], q[1], q[2] + 0.35, label, fontsize=12, color=INK, zorder=11)
    ax1.set_xlabel(r"$j_1$", color=INK_SECONDARY)
    ax1.set_ylabel(r"$j_2$", color=INK_SECONDARY)
    ax1.set_zlabel(r"$j_3$", color=INK_SECONDARY)
    setJointAngleTicks(ax1, dim3=True)
    ax1.tick_params(colors=INK_MUTED, labelsize=8)
    ax1.set_title("C-space (3D): configurations in collision", fontsize=12, color=INK, loc="left")
    ax1.view_init(elev=22, azim=-58)
    fig.legend(handles=obstacleLegendHandles(obstacles), loc="lower center", ncol=len(obstacles),
               frameon=False, fontsize=10)
    plt.show()


# ---------------------------------------------------------------------------
# implicit C-space
# ---------------------------------------------------------------------------

def implicitSearch(collisionChecker, start, goal, stepSize=0.2, maxChecks=250):
    """Greedy best-first search on a lattice in C-space. The obstacles are unknown:
    every configuration the search generates is checked for collision in W-space."""
    start, goal = np.array(start, dtype=float), np.array(goal, dtype=float)
    limits = collisionChecker.getEnvironmentLimits()
    moves = [m for m in itertools.product((-1, 0, 1), repeat=len(start)) if any(m)]

    def config(key):
        return start + stepSize * np.array(key)

    startKey = (0,) * len(start)
    parent = {startKey: None}
    openList = [(np.linalg.norm(goal - start), startKey)]
    checks = []
    current = startKey
    while openList and len(checks) < maxChecks:
        _, current = heapq.heappop(openList)
        if np.linalg.norm(config(current) - goal) <= stepSize:
            break
        for m in moves:
            key = tuple(np.add(current, m))
            q = config(key)
            if key in parent or not all(lo <= v <= hi for v, (lo, hi) in zip(q, limits)):
                continue
            inCollision = collisionChecker.pointInCollision(list(q))  # collision check in W-space
            checks.append((q, inCollision))
            parent[key] = current
            if not inCollision:
                heapq.heappush(openList, (np.linalg.norm(goal - q), key))
            if len(checks) >= maxChecks:
                break

    path, key = [], current
    while key is not None:
        path.append(config(key))
        key = parent[key]
    return {"checks": checks, "path": path[::-1], "goal": goal}


def plotImplicitCSpace(robot, obstacles, axes, cspace, result,
                       workspaceLimits=((-3.5, 3.5), (-3.5, 3.5)), figsize=(12.5, 5.8)):
    """Left: what the planner knows about the C-space. Right: the collision checks in W-space."""
    checks = result["checks"]
    free = np.array([q for q, c in checks if not c])
    colliding = np.array([q for q, c in checks if c])
    path = np.array(result["path"])

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    styleAxes(axs[0], ((-np.pi, np.pi), (-np.pi, np.pi)), "C-space: planning (point robot)",
              r"$j_1$ [rad]", r"$j_2$ [rad]", grid=False)
    setJointAngleTicks(axs[0])
    axs[0].set_facecolor("#f0efec")
    axs[0].contour(axes[0], axes[1], (cspace > 0).T.astype(float), levels=[0.5], colors=INK_MUTED,
                             linewidths=0.8, linestyles="dashed", zorder=2)
    if len(free):
        axs[0].scatter(free[:, 0], free[:, 1], s=14, color=FREE, zorder=4)
    if len(colliding):
        axs[0].scatter(colliding[:, 0], colliding[:, 1], s=22, marker="x", color=COLLISION, linewidth=1.5, zorder=4)
    axs[0].plot(path[:, 0], path[:, 1], color=INK, linewidth=1.5, zorder=5)
    drawStartGoal(axs[0], path[0], result["goal"], labelOffset=(0.12, 0.12))

    styleAxes(axs[1], workspaceLimits, "W-space: collision checking", "x", "y")
    for obstacle in obstacles.values():
        drawGeometry(axs[1], obstacle, OBSTACLE, zorder=2)
    lastColliding = [q for q, c in checks if c]
    if lastColliding:
        drawPlanarRobot(axs[1], robot, lastColliding[-1], color=COLLISION, linewidth=3)
        axs[0].scatter([lastColliding[-1][0]], [lastColliding[-1][1]], s=160, facecolor="none",
                       edgecolor=COLLISION, linewidth=2, zorder=6)
    drawPlanarRobot(axs[1], robot, path[-1], color=FREE, linewidth=3)
    axs[0].scatter([path[-1][0]], [path[-1][1]], s=160, facecolor="none", edgecolor=FREE, linewidth=2, zorder=6)

    handles = [Line2D([], [], marker="o", linestyle="", color=FREE, label="checked: free"),
               Line2D([], [], marker="x", linestyle="", color=COLLISION, markeredgewidth=1.5,
                      label="checked: collision"),
               Line2D([], [], color=INK, linewidth=1.5, label="current path"),
               Line2D([], [], color=INK_MUTED, linestyle="dashed", linewidth=0.8,
                      label="C-obstacles (unknown to the planner)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.subplots_adjust(wspace=0.5)
    addArrowBetweenAxes(fig, axs[0], axs[1], "configuration q", y=0.62)
    addArrowBetweenAxes(fig, axs[1], axs[0], "free / collision", y=0.42)
    plt.show()
    print(f"{len(checks)} collision checks: {len(free)} free, {len(colliding)} in collision")
