# coding: utf-8
"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

It is based on the slides given during the course, so please **read the information in theses slides first**

Remark: The code is a slightly modified version of AStar, whithout reopening of nodes, when once in the closed list.
With config["reopening"] = True it handles the successors as the A* of the textbooks: if a cheaper path to a node
already in the graph is found, the node gets the new father and the lower costs g and goes (back) into the open list -
also when it is already closed (reopening).

Discretisation of the C-space as on the slides "A* - Algorithm for path planning":

* dimension i: limits Theta_min,i ... Theta_max,i (getEnvironmentLimits() of the collision checker),
  step size DeltaTheta_i (config["discretization"])
* grid point -> configuration: phi_i = Theta_min,i + DeltaTheta_i * x_i
* configuration -> grid point: x_i = floor((phi_i - Theta_min,i) / DeltaTheta_i + 1/2)

A* takes place on the grid: the successors of a node are its neighbours x_i +/- 1, every edge costs 1 (g counts
the edges) and the heuristic measures the distance in grid coordinates. The configurations are needed for the
collision check and the solution path. Start and goal belong to the grid point of their field, their nodes keep
the exact configuration, so that the path starts in S and ends in G.

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import copy
import heapq
import itertools
import math
import numbers

import networkx as nx
from scipy.spatial.distance import euclidean, cityblock
from IPPlanerBase import PlanerBase

from IPPerfMonitor import IPPerfMonitor


class AStar(PlanerBase):
    def __init__(self, collChecker = 0):
        """Contructor:

        Initialize all necessary members"""

        super(AStar,self).__init__(collChecker)
        self.graph = nx.DiGraph() # = CloseList
        self.openList = [] # (<value>, <node>)

        self.goal      =  []
        self.goalGrid  =  []   # grid coordinates of the goal
        self.goalName  =  None
        self.goalFound = False
        self.solutionPath = []

        self.limits = self._collisionChecker.getEnvironmentLimits()
        self.dim = self._collisionChecker.getDim()

        self.w = 0.5
        self.discretization = [1] * self.dim   # step size DeltaTheta_i per dimension
        self.maxIterations = 1000
        self.reopening = False
        return

    def _getNodeID(self, x):
        """Compute a unique identifier based on the grid coordinates x of a node.

        The integer grid coordinates are exact, the configurations are floating point numbers (0.1 + 0.2 != 0.3)."""

        nodeId = "-"
        for i in x:
            nodeId += str(i)+"-"
        return nodeId

    def _toConfiguration(self, x):
        """Grid point -> configuration: phi_i = Theta_min,i + DeltaTheta_i * x_i"""
        return [limit[0] + step * xi for xi, limit, step in zip(x, self.limits, self.discretization)]

    def _toGrid(self, phi):
        """Configuration -> grid point: x_i = floor((phi_i - Theta_min,i) / DeltaTheta_i + 1/2)

        (not round(): Python rounds 6.5 to 6, half to even)"""
        return [math.floor((p - limit[0]) / step + 0.5) for p, limit, step in zip(phi, self.limits, self.discretization)]

    def _setDiscretization(self, discretization):
        """One step size for all dimensions or a list with the step size DeltaTheta_i of every dimension"""
        if isinstance(discretization, numbers.Real):
            discretization = [discretization] * self.dim
        discretization = list(discretization)
        if len(discretization) != self.dim or not all(step > 0 for step in discretization):
            raise ValueError("discretization needs {} positive step sizes, got {}".format(self.dim, discretization))
        self.discretization = discretization

    @IPPerfMonitor
    def planPath(self, startList, goalList, config):
        """

        Args:
            start (array): start position in planning space
            goal (array) : goal position in planning space
            config (dict): dictionary with the needed information about the configuration options

        Example:

            config["w"] = 0.5
            config["heuristic"] = "euclid"
            config["discretization"] = [1.0, 0.5]   # optional: step size DeltaTheta_i per dimension
                                                    #           (one number: the same for all dimensions), default 1
            config["maxIterations"] = 1000          # optional: maximal number of expanded nodes, default 1000
            config["reopening"] = False             # optional: True: cheaper paths to nodes in the graph are taken
                                                    #           over, closed nodes are reopened, default False

        """
        # 0. reset
        self.graph.clear()
        self.openList = []
        self.goalFound = False
        self.solutionPath = []

        self._setDiscretization(config.get("discretization", 1))
        self.maxIterations = config.get("maxIterations", 1000)
        self.reopening = config.get("reopening", False)

        try:
            # 1. check start and goal whether collision free (s. BaseClass)
            checkedStartList, checkedGoalList = self._checkStartGoal(startList,goalList)

            # 2.
            self.w = config["w"]
            self.heuristic = config["heuristic"]

            start = list(checkedStartList[0])
            self.goal = list(checkedGoalList[0])
            startGrid = self._toGrid(start)
            self.goalGrid = self._toGrid(self.goal)
            if self.goalGrid == startGrid and self.goal != start:
                # start and goal in the same field: the goal takes the neighbouring grid point in its direction
                offset = [(g - s) / step for g, s, step in zip(self.goal, start, self.discretization)]
                i = max(range(self.dim), key=lambda j: abs(offset[j]))
                self.goalGrid[i] += 1 if offset[i] > 0 else -1
            self.goalName = self._getNodeID(self.goalGrid)
            self._addGraphNode(startGrid, start)

            currentBestName = self._getBestNodeName()
            breakNumber = 0
            while currentBestName:
              if breakNumber > self.maxIterations:
                break

              breakNumber += 1

              currentBest = self.graph.nodes[currentBestName]

              if currentBestName == self.goalName:
                self.solutionPath = []
                self._collectPath( currentBestName, self.solutionPath )
                self.goalFound = True
                break

              currentBest["status"]= 'closed'
              if self._collisionChecker.pointInCollision(currentBest["pos"]):
                currentBest['collision']= 1
                currentBestName = self._getBestNodeName()
                continue
              self.graph.nodes[currentBestName]['collision'] = 0

              # handleNode merges with former expandNode
              self._handleNode(currentBestName)
              currentBestName = self._getBestNodeName()

            if self.goalFound:
                return self.solutionPath
            else:
                return None
        except Exception as e:
            print("Planning failed:", e)
            return None

    def _insertNodeNameInOpenList(self, nodeName):
        """Get an existing node stored in graph and put it in the OpenList"""
        value = self._evaluateNode(nodeName)
        self.graph.nodes[nodeName]["f"] = value
        heapq.heappush(self.openList,(value,nodeName))

    @IPPerfMonitor
    def _addGraphNode(self, x, pos, fatherName=None):
        """Add a node with its grid coordinates x and its configuration pos into the graph.
        Attention: Existing node is overwritten!"""
        nodeName = self._getNodeID(x)
        self.graph.add_node(nodeName, pos=pos, grid=x, status='open', g=0)

        if fatherName != None:
            self.graph.add_edge(nodeName, fatherName)
            self.graph.nodes[nodeName]["g"] = self.graph.nodes[fatherName]["g"] + 1

        self._insertNodeNameInOpenList(nodeName)


    def _setLimits(self, lowLimit, highLimit):
        """ Sets the limits of the investigated search space """
        assert(len(lowLimit)==len(highLimit)==self.dim)
        self.limits = list()
        for i in range(self.dim):
          self.limits.append([lowLimit[i],highLimit[i]])
        return

    def _getBestNodeName(self):
        """ Returns the best name of best node (None, if the open list is empty)

        With reopening a node can be in the heap several times: outdated entries (node closed meanwhile or
        a lower f) are skipped."""
        while self.openList:
            value, nodeName = heapq.heappop(self.openList)
            node = self.graph.nodes[nodeName]
            if node["status"] == 'open' and node["f"] == value:
                return nodeName
        return None

    def _addSuccessor(self, fatherName, x):
        """Add the grid point x as successor of fatherName, if it lies inside the limits and is not yet in the graph.
        The grid point of the goal gets the exact goal configuration."""
        pos = list(self.goal) if x == self.goalGrid else self._toConfiguration(x)
        if not self._inLimits(pos):
            return
        nodeName = self._getNodeID(x)
        if nodeName in self.graph:
            # Without reopening: if node already in graph do not touch it... Concequences?
            if self.reopening:
                self._reopenNode(nodeName, fatherName)
            return
        self._addGraphNode(x, pos, fatherName)

    def _reopenNode(self, nodeName, fatherName):
        """A node already in the graph is reached again from fatherName. If this path is cheaper, the node gets
        fatherName as new father and the lower g and goes into the open list again: a node in OPEN gets a better
        position, a closed node is reopened (its successors are corrected, when it is expanded again)."""
        node = self.graph.nodes[nodeName]
        g = self.graph.nodes[fatherName]["g"] + 1
        if g >= node["g"] or node.get("collision") == 1:
            return
        self.graph.remove_edges_from(list(self.graph.out_edges(nodeName)))
        self.graph.add_edge(nodeName, fatherName)
        node["g"] = g
        if node["status"] == 'closed':
            node["status"] = 'open'
            node["reopened"] = node.get("reopened", 0) + 1
        else:
            node["updated"] = node.get("updated", 0) + 1
        self._insertNodeNameInOpenList(nodeName)

    @IPPerfMonitor
    def _handleNode(self, nodeName):
        """Generats possible successor positions in all dimensions"""
        result =  []
        node = self.graph.nodes[nodeName]
        for i in range(len(node["grid"])):
            for u in [-1,1]:
                    x = copy.copy(node["grid"])
                    x[i] += u
                    self._addSuccessor(nodeName, x)

        return result

    @IPPerfMonitor
    def _handleNode9(self, nodeName):
        """Generats possible successor positions also in diagonal direction: every x_i changes by -1, 0 or +1,
        but not all by 0 (2D: the 3x3 = 9 fields around the node without the node itself, nD: 3^n - 1 neighbours).

        Only the grid points are checked for collision, not the edges in between: a successor further away than
        one field could jump over a thin obstacle."""
        result =  []
        node = self.graph.nodes[nodeName]
        for offset in itertools.product([-1, 0, 1], repeat=len(node["grid"])):
            if not any(offset):
                continue
            x = [xi + oi for xi, oi in zip(node["grid"], offset)]
            self._addSuccessor(nodeName, x)

        return result


    @IPPerfMonitor
    def _computeHeuristicValue(self, nodeName):
        """ Computes Heuristic Value in grid coordinates: Euclidean or Manhattan Distance """

        result = 0
        node = self.graph.nodes[nodeName]
        if self.heuristic == "euclidean":
            return euclidean(self.goalGrid, node["grid"])
        else:
            return cityblock(self.goalGrid, node["grid"])

    @IPPerfMonitor
    def _evaluateNode(self, nodeName):
        node = self.graph.nodes[nodeName]
        return  self.w * self._computeHeuristicValue(nodeName) + (1 - self.w) * node["g"]


    def _collectPath(self, nodeName, solutionPath):

      fathers = list(self.graph.successors(nodeName))
      #print len(fathers)
      if len(fathers) == 1:
        self._collectPath( fathers[0], solutionPath )
      elif len(fathers) == 0:
        solutionPath.append(nodeName)
        return
      else:
        raise Exception("not suitable numbers of fathers = {}.... please check".format(len(fathers)))
      solutionPath.append(nodeName)
      return

    @IPPerfMonitor
    def _inLimits(self, pos):
        result = True
        for i, limit in enumerate(self.limits):
          if pos[i] < limit[0] or pos[i] > limit[1]:
            result = False
            break;
        return result
