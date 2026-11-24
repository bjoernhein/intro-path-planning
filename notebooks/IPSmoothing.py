# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

Smoothing of a collision-free path with shortcuts, the generic approach of the lecture (slides "Smoothing"):

    Smooth(p): (l_A, l_B) <- ChooseShortcut(p, 0, 1, true)
               while l_A != l_B:
                   success <- OptimizationStrategy(l_A, l_B, p)
                   (l_A, l_B) <- ChooseShortcut(p, l_A, l_B, success)

A position l in [0, 1] on the path is its length from the start, normalized by the length of the path.
OptimizationStrategy checks the segment from p(l_A) to p(l_B) and replaces the part of the path between them, if the
segment is collision-free. ChooseShortcut is one of three strategies:

    "random":        pure probabilistic, l_A and l_B uniformly distributed
    "latombe":       extended probabilistic, a random position l_rand, the interval around it is halved
                     until the shortcut is collision-free or shorter than deltaLMin
    "deterministic": Berchtold & Glavina, the worst path point (rating Q) is removed by connecting its predecessor
                     and its successor, if this fails the interval is halved towards it (DelTree)

Attention: the smoothing needs a collision-free path and a reliable check of the shortcuts. lineInCollision of the
collision checkers samples a segment (CollisionChecker: 40 configurations, KinChainCollisionChecker: fk_resolution).
Shortcuts are long and run close along the obstacles - exactly where sampled checks miss collisions. Therefore
config["lineCheck"] (e.g. lineInCollisionExact for a point robot) or config["resolution"] choose the check.

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import math
import random
import statistics

import networkx as nx

from IPPerfMonitor import IPPerfMonitor

STRATEGIES = ("random", "latombe", "deterministic")
EPS = 1e-9


# ------------------------------------------------------------------------------------------------------------------
# paths
# ------------------------------------------------------------------------------------------------------------------

def pathFromPlanner(planner, solution):
    """Configurations of a solution (list of nodes) of any planner of the course, which provides graph with 'pos'."""
    return [tuple(float(c) for c in planner.graph.nodes[node]["pos"]) for node in solution]


def pathLength(path):
    """Length of a path given as list of configurations."""
    return sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:]))


def positions(path):
    """Position l_i in [0, 1] of every path point: length from the start, normalized by the path length."""
    total = pathLength(path)
    ls, length = [0.0], 0.0
    for a, b in zip(path[:-1], path[1:]):
        length += math.dist(a, b)
        ls.append(length / total if total > 0 else 0.0)
    return ls


def pointAt(path, l, ls=None):
    """Configuration p(l) and index k of the segment (path[k], path[k+1]) it lies on."""
    ls = ls or positions(path)
    for k in range(len(path) - 1):
        if l <= ls[k + 1] or k == len(path) - 2:
            span = ls[k + 1] - ls[k]
            t = 0.0 if span <= 0 else min(max((l - ls[k]) / span, 0.0), 1.0)
            return tuple(a + t * (b - a) for a, b in zip(path[k], path[k + 1])), k
    raise ValueError("path with less than two points")


def quality(path, i):
    """Rating Q of the inner path point i: (|p_i-1 p_i| + |p_i p_i+1|) / |p_i-1 p_i+1| >= 1.
    Q = 1: the point lies on the straight line, the larger Q, the sharper the spike or the longer the detour."""
    base = math.dist(path[i - 1], path[i + 1])
    detour = math.dist(path[i - 1], path[i]) + math.dist(path[i], path[i + 1])
    return math.inf if base < EPS else detour / base


def removeCollinear(path, tolerance=1e-9):
    """Basic operation "removing collinear points": inner points, which lie on the straight line between their
    neighbours (Q = 1 within the tolerance). No collision check is needed, the path itself does not change."""
    result = [path[0]]
    for i in range(1, len(path) - 1):
        if quality([result[-1], path[i], path[i + 1]], 1) > 1 + tolerance:
            result.append(path[i])
    result.append(path[-1])
    return result


def sampledLineCheck(collisionChecker, resolution):
    """Check of a segment for any collision checker: configurations with a maximal distance of resolution, both ends
    included. Returns a function (qa, qb) -> True, if the segment is in collision."""
    def lineInCollision(qa, qb):
        n = max(1, math.ceil(math.dist(qa, qb) / resolution))
        return any(collisionChecker.pointInCollision([a + k / n * (b - a) for a, b in zip(qa, qb)])
                   for k in range(n + 1))
    return lineInCollision


def collidingSegments(path, lineInCollision):
    """Indices k of the segments (path[k], path[k+1]) in collision, checked with the given function (qa, qb)."""
    return [k for k, (a, b) in enumerate(zip(path[:-1], path[1:])) if lineInCollision(list(a), list(b))]


def historySeries(history, budget):
    """Path length and number of path points after 0, 1, ..., budget checks, from the history of a Smoother
    (after the last check the values are kept). Returns (lengths, points)."""
    lengths, points, k = [], [], 0
    for check in range(budget + 1):
        while k + 1 < len(history) and history[k + 1][0] <= check:
            k += 1
        lengths.append(history[k][1])
        points.append(history[k][2])
    return lengths, points


class Terminate(object):
    """Terminate(p) of the slides: the lengths of the path after the last steps in a queue; as soon as it holds
    sMax values, the oldest one is removed and the smoothing stops, if the variance is smaller than vMin."""

    def __init__(self, sMax, vMin):
        self.sMax, self.vMin = sMax, vMin
        self.queue = []

    def __call__(self, path):
        self.queue.append(pathLength(path))
        if len(self.queue) < self.sMax:
            return False
        self.queue.pop(0)
        return statistics.pvariance(self.queue) < self.vMin


# ------------------------------------------------------------------------------------------------------------------
# smoothing
# ------------------------------------------------------------------------------------------------------------------

class Smoother(object):
    """Smoothing of a path with one of the three strategies, interface similar to the planners:
    smoothPath(path, config) returns the smoothed path as list of nodes of self.graph (attribute 'pos'),
    so the visualizations and animateSolution of IPEnvironmentKin work as for the planners."""

    def __init__(self, collisionChecker=None):
        """collisionChecker: the collision checker interface (lineInCollision); smoothSolution takes the one of the
        planner, if none is given"""
        self._collisionChecker = collisionChecker
        self.graph = nx.Graph()   # the smoothed path, nodes "start", 1, 2, ..., "goal" with attribute 'pos'
        self.inputPath = []       # the path given to smoothPath
        self.path = []            # the smoothed path as list of configurations
        self.steps = []           # protocol of every tested shortcut, see _optimize()
        self.history = []         # (checks, path length, number of points) at the start and after every check
        self.checks = 0           # checked segments (shortcuts)
        self.reason = None        # why the smoothing stopped: "done", "terminated", "maxChecks", "maxIterations"

    def smoothSolution(self, planner, solution, config):
        """Smooth the solution (list of nodes) of a previous planning with any planner of the course: the path is
        taken from planner.graph, the collision checker from the planner, if the smoother has none."""
        if self._collisionChecker is None:
            self._collisionChecker = planner._collisionChecker
        return self.smoothPath(pathFromPlanner(planner, solution), config)

    @IPPerfMonitor
    def smoothPath(self, path, config):
        """
        Args:
            path: collision-free path, list of configurations
            config (dict):
                config["strategy"] = "deterministic"   # "random", "latombe" or "deterministic"
                config["maxChecks"] = 100              # maximal number of segment collision checks
                config["deltaLMin"] = 0.02             # smallest interval l_B - l_A (normalized length)
                config["sMax"], config["vMin"]         # optional: Terminate with this queue size and variance
                config["lineCheck"]                    # optional: function (qa, qb) -> True if in collision,
                                                       #   e.g. collisionChecker.lineInCollisionExact (point robot)
                config["resolution"]                   # optional: sample the segments with this maximal distance
                                                       #   (sampledLineCheck), for any collision checker
            Without lineCheck and resolution the shortcuts are checked with collisionChecker.lineInCollision.

        Random numbers come from the module random (seed it with random.seed for reproducible runs).
        """
        strategy = config.get("strategy", "deterministic")
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        self._choose = getattr(self, "_choose" + strategy.capitalize())
        self._deltaLMin = config.get("deltaLMin", 0.02)
        if "lineCheck" in config:
            self._lineInCollision = config["lineCheck"]
        elif "resolution" in config:
            self._lineInCollision = sampledLineCheck(self._collisionChecker, config["resolution"])
        else:
            self._lineInCollision = self._collisionChecker.lineInCollision
        terminate = Terminate(config["sMax"], config["vMin"]) if "sMax" in config else None
        maxChecks = config.get("maxChecks", 100)

        # 0. reset
        self.inputPath = [tuple(float(c) for c in q) for q in path]
        p = list(self.inputPath)
        self.steps, self.checks = [], 0
        self.history = [(0, pathLength(p), len(p))]
        self._lRand, self._worst, self._failed, self._shortcut = None, None, set(), None

        # 1. Smooth(p)
        lA, lB = self._choose(p, 0.0, 1.0, True)
        iterations = 0
        self.reason = "done"
        while lA != lB:
            if self.checks >= maxChecks:
                self.reason = "maxChecks"
                break
            iterations += 1
            if iterations > 100 * maxChecks:
                self.reason = "maxIterations"
                break
            checks = self.checks
            success, p = self._optimize(p, lA, lB, strategy)
            if terminate is not None and self.checks > checks and terminate(p):
                self.reason = "terminated"
                break
            lA, lB = self._choose(p, lA, lB, success)

        # 2. store the path as graph
        self.path = [list(q) for q in p]
        nodes = ["start"] + list(range(1, len(p) - 1)) + ["goal"]
        self.graph.clear()
        for node, q in zip(nodes, self.path):
            self.graph.add_node(node, pos=q)
        self.graph.add_edges_from(zip(nodes[:-1], nodes[1:]))
        return nodes

    # ------------------------------------------------------------------------------------------------------------
    # OptimizationStrategy
    # ------------------------------------------------------------------------------------------------------------

    def _optimize(self, p, lA, lB, strategy):
        """Shortcut from p(l_A) to p(l_B): if it is collision-free, all path points between are removed and
        p(l_A), p(l_B) are inserted. Without a path point between l_A and l_B there is nothing to shorten: no check.
        Returns (success, path). Every check is recorded in self.steps as dict:

        check: number of the check, lA, lB, qa, qb: p(l_A), p(l_B), free, previous: path before, path: path after,
        removed: removed path points, chosen: "new" or "halve" (how l_A, l_B were chosen), lRand (latombe),
        worst: p(l_w) (deterministic)
        """
        ls = positions(p)
        inner = [i for i in range(1, len(p) - 1) if lA < ls[i] < lB]
        if not inner:
            return False, p
        qa, _ = pointAt(p, lA, ls)
        qb, _ = pointAt(p, lB, ls)
        self.checks += 1
        free = not self._lineInCollision(list(qa), list(qb))
        newPath = p
        if free:
            head, tail = p[:inner[0]], p[inner[-1] + 1:]
            newPath = head + ([qa] if math.dist(qa, head[-1]) > EPS else []) + \
                ([qb] if math.dist(qb, tail[0]) > EPS else []) + tail
        self.steps.append(dict(check=self.checks, lA=lA, lB=lB, qa=qa, qb=qb, free=free, previous=list(p),
                               path=list(newPath), removed=[p[i] for i in inner], chosen=self._chosen,
                               strategy=strategy, lRand=self._lRand, worst=self._worst))
        self.history.append((self.checks, pathLength(newPath), len(newPath)))
        return free, newPath

    # ------------------------------------------------------------------------------------------------------------
    # ChooseShortcut
    # ------------------------------------------------------------------------------------------------------------

    def _chooseRandom(self, p, lA, lB, success):
        """pure probabilistic: two random positions (the arguments are not used)"""
        self._chosen = "new"
        x, y = random.random(), random.random()
        return min(x, y), max(x, y)

    def _chooseLatombe(self, p, lA, lB, success):
        """extended probabilistic: l_rand is kept between the calls, the interval around it is halved"""
        self._chosen = "halve"
        if success or self._lRand is None or lB - lA < self._deltaLMin:
            self._lRand = random.random()
            lA, lB = 0.0, 1.0
            self._chosen = "new"
        return (lA + self._lRand) / 2, (lB + self._lRand) / 2

    def _chooseDeterministic(self, p, lA, lB, success):
        """deterministic (Berchtold & Glavina): worst path point first; a point, whose bisection failed, gets a new
        chance only when one of its neighbours has changed. No candidate left: l_A = l_B, Smooth ends."""
        if success and self._shortcut is not None:
            for q in self._shortcut:                      # p(l_A), p(l_B) of the successful shortcut
                i = min(range(len(p)), key=lambda k: math.dist(p[k], q))
                for j in (i - 1, i, i + 1):
                    if 0 <= j < len(p):
                        self._failed.discard(p[j])
        ls = positions(p)
        if success or self._worst not in p or lB - lA < self._deltaLMin:
            if not success and self._worst in p:
                self._failed.add(self._worst)
            candidates = [i for i in range(1, len(p) - 1) if p[i] not in self._failed and quality(p, i) > 1 + EPS]
            if not candidates:
                self._worst = None
                return 0.0, 0.0
            i = max(candidates, key=lambda k: quality(p, k))
            self._worst, self._chosen = p[i], "new"
            lA, lB = ls[i - 1], ls[i + 1]
        else:
            lw = ls[p.index(self._worst)]
            self._chosen = "halve"
            lA, lB = (lA + lw) / 2, (lB + lw) / 2
        self._shortcut = (pointAt(p, lA, ls)[0], pointAt(p, lB, ls)[0])
        return lA, lB
