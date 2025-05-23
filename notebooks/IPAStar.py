# coding: utf-8
"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

It is based on the slides given during the course, so please **read the information in theses slides first**

Remark: The code is a slightly modified version of AStar, whithout reopening of nodes, when once in the closed list.

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import copy
import networkx as nx
import heapq
import math
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
        self.goalFound = False

        self.limits = self._collisionChecker.getEnvironmentLimits()

        self.w = 0.5  
        return

    def _getNodeID(self,pos):
        """Compute a unique identifier based on the position"""

        nodeId = "-"
        for i in pos:
            nodeId += str(i)+"-"
        return nodeId

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

        """
        # 0. reset
        self.graph.clear()

        try:
            # 1. check start and goal whether collision free (s. BaseClass)
            checkedStartList, checkedGoalList = self._checkStartGoal(startList,goalList)

            # 2.
            self.w = config["w"]
            self.heuristic = config["heuristic"]

            self.goal = checkedGoalList[0]
            self._addGraphNode(checkedStartList[0])

            currentBestName = self._getBestNodeName()
            breakNumber = 0
            while currentBestName:
              if breakNumber > 1000:
                break

              breakNumber += 1

              currentBest = self.graph.nodes[currentBestName]

              if currentBest["pos"] == self.goal:
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
        except:
            print("Planning failed")
            return None

    def _insertNodeNameInOpenList(self, nodeName):
        """Get an existing node stored in graph and put it in the OpenList"""
        heapq.heappush(self.openList,(self._evaluateNode(nodeName),nodeName))

    @IPPerfMonitor
    def _addGraphNode(self, pos, fatherName=None):
        """Add a node based on the position into the graph. Attention: Existing node is overwritten!"""
        self.graph.add_node(self._getNodeID(pos), pos=pos, status='open', g=0)

        if fatherName != None:
            self.graph.add_edge(self._getNodeID(pos), fatherName)
            self.graph.nodes[self._getNodeID(pos)]["g"] = self.graph.nodes[fatherName]["g"] + 1

        self._insertNodeNameInOpenList(self._getNodeID(pos))

   
    def _setLimits(self, lowLimit, highLimit):
        """ Sets the limits of the investigated search space """
        assert(len(lowLimit)==len(highLimit)==self.dim)
        self.limits = list()
        for i in range(self.dim):
          self.limits.append([lowLimit[i],highLimit[i]])
        return
  
    def _getBestNodeName(self):
        """ Returns the best name of best node """
        return heapq.heappop(self.openList)[1]

    @IPPerfMonitor
    def _handleNode(self, nodeName):
        """Generats possible successor positions in all dimensions"""
        result =  []
        node = self.graph.nodes[nodeName]
        for i in range(len(node["pos"])):
            for u in [-1,1]:
                    newPos = copy.copy(node["pos"])
                    newPos[i] += u
                    if not self._inLimits(newPos):
                        continue
                    try:
                        # Do not do reopening! If node already in graph do not add it... Concequences?
                        self.graph.nodes[self._getNodeID(newPos)]
                        continue
                    except:
                        pass

                    self._addGraphNode(newPos,nodeName)

        return result

    @IPPerfMonitor
    def _handleNode9(self, nodeName):
        """Generats possible successor positions also in diagonal direction"""
        result =  []
        node = self.graph.nodes[nodeName]
        for i in range(len(node["pos"])):
          for j in range(len(node["pos"])): 
              for u in [-1,1]:
                  for v in [-1,0,1]:
                        newPos = copy.copy(node["pos"])
                        newPos[i] += u
                        newPos[j] += v
                        if not self._inLimits(newPos):
                            continue
                        try:
                            # Do not do reopening! If node already in graph do not add it... Concequences?
                            self.graph.nodes[self._getNodeID(newPos)]
                            continue
                        except:
                            pass

                        self._addGraphNode(newPos,nodeName)

        return result


    @IPPerfMonitor
    def _computeHeuristicValue(self, nodeName):
        """ Computes Heuristic Value: Manhattan Distance """

        result = 0
        node = self.graph.nodes[nodeName]
        if self.heuristic == "euclidean":
            return euclidean(self.goal, node["pos"])
        else:
            return cityblock(self.goal, node["pos"])

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


