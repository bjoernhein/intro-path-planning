# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import IPPRMBase
from IPPerfMonitor import IPPerfMonitor
import networkx as nx
import random
import numpy as np
import math

# reduce coding effort by using function provided by scipy
from scipy.spatial.distance import euclidean, cityblock

class BasicPRM(IPPRMBase.PRMBase):

    def __init__(self, _collChecker):
        super(BasicPRM, self).__init__(_collChecker)
        self.graph = nx.Graph()

    
    @IPPerfMonitor
    def _inSameConnectedComponent(self, node1, node2):
        """ Check whether two nodes are part of the same connected component using
            functionality from NetworkX: this is the case, if there is a path between them.
            (has_path searches from both nodes and stops early, much faster than
            enumerating all connected components)
        """
        return nx.has_path(self.graph, node1, node2)

    
    @IPPerfMonitor
    def _nearestNeighbours(self, pos, radius):
        """ Brute Force method to find all nodes of a
        graph near the given position **pos** with in the distance of
        **radius**, sorted by increasing distance """

        result = list()
        for node in self.graph.nodes(data=True):
            if euclidean(node[1]['pos'],pos) <= radius:
                result.append(node)

        result.sort(key=lambda node: euclidean(node[1]['pos'], pos))
        return result
    
    @IPPerfMonitor
    def _learnRoadmapNearestNeighbour(self, radius, numNodes, collisionCheckingSteps=40):
        """ Generate a roadmap by given number of nodes and radius, that should be tested for connection."""
        # nodeID is used for uniquely enumerating all nodes and as their name
        nodeID = 1
        while nodeID <= numNodes:
        
            # Generate a 'randomly chosen, free configuration'
            newNodePos = self._getRandomFreePosition()

            # Find set of candidates to connect to sorted by distance
            result = self._nearestNeighbours(newNodePos, radius)
            self.graph.add_node(nodeID, pos=newNodePos)

            # for all nearest neighbours check whether a connection is possible;
            # neighbours already in the same connected component are skipped
            for data in result:
                if self._inSameConnectedComponent(nodeID,data[0]):
                    continue

                neighbourPos = data[1]['pos']
                if not self._collisionChecker.lineInCollision(
                    newNodePos, neighbourPos, steps=collisionCheckingSteps
                ):
                    self.graph.add_edge(nodeID, data[0], weight=euclidean(newNodePos, neighbourPos))
            
            nodeID += 1
            
    @IPPerfMonitor
    def planPath(self, startList, goalList, config):
        """
        
        Args:
            start (array): start position in planning space
            goal (array) : goal position in planning space
            config (dict): dictionary with the needed information about the configuration options
            
        Example:
        
            config["radius"]   = 5.0
            config["numNodes"] = 300
            config["collisionCheckingSteps"] = 40
            config["useKDTree"] = True
            
            startList = [[1,1]]
            goalList  = [[10,1]]
            
            instance.planPath(startList,goalList,config)
        
        """
        # 0. reset
        self.graph.clear()
        collisionCheckingSteps = config.get("collisionCheckingSteps", 40)
        
        # 1. check start and goal whether collision free (s. BaseClass)
        checkedStartList, checkedGoalList = self._checkStartGoal(startList,goalList)
        
        # 2. learn Roadmap
        self._learnRoadmapNearestNeighbour(
            config["radius"],
            config["numNodes"],
            collisionCheckingSteps=collisionCheckingSteps,
        )

        # 3. find connection of start and goal to roadmap
        # find nearest, collision-free connection between node on graph and start
        result = self._nearestNeighbours(checkedStartList[0], config["radius"])
        for node in result:
            neighbourPos = node[1]['pos']
            if not self._collisionChecker.lineInCollision(
                checkedStartList[0], neighbourPos, steps=collisionCheckingSteps
            ):
                self.graph.add_node("start", pos=checkedStartList[0], color='lightgreen')
                self.graph.add_edge(
                    "start", node[0], weight=euclidean(checkedStartList[0], neighbourPos)
                )
                break

        result = self._nearestNeighbours(checkedGoalList[0], config["radius"])
        for node in result:
            neighbourPos = node[1]['pos']
            if not self._collisionChecker.lineInCollision(
                checkedGoalList[0], neighbourPos, steps=collisionCheckingSteps
            ):
                self.graph.add_node("goal", pos=checkedGoalList[0], color='lightgreen')
                self.graph.add_edge(
                    "goal", node[0], weight=euclidean(checkedGoalList[0], neighbourPos)
                )
                break

        try:
            path = nx.shortest_path(self.graph, "start", "goal", weight="weight")
        except:
            return []
        return path
