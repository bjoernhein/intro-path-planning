# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).
It gathers all visualizations of the investigated and explained planning algorithms.
License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from IPBenchmark import Benchmark 
from IPEnvironment import CollisionChecker
from shapely.geometry import Point, Polygon, LineString
import shapely.affinity
import math
import numpy as np


benchList = list()

# all benchmarks use the same environment limits
limits = [[0.0, 22.0], [0.0, 22.0]]

# The level is derived from the clearance c of the benchmark: the largest radius of a disc robot,
# which could still move from start to goal (= half the width of the narrowest passage), see IP-4-0.
# c >= 2: 1 ("easy"), 0.75 <= c < 2: 2 ("normal"), 0.4 <= c < 0.75: 3 ("hard"), c < 0.4: 4 ("insane")

# -----------------------------------------
trapField = dict()
trapField["obs1"] =   LineString([(6, 18), (6, 8), (16, 8), (16,18)]).buffer(1.0)
description = "Following the direct connection from goal to start would lead the algorithm into a trap."
benchList.append(Benchmark("Trap", CollisionChecker(trapField, limits), [[10,15]], [[10,1]], description, 1))  # c = 3.0

# -----------------------------------------
bottleNeckField = dict()
bottleNeckField["obs1"] = LineString([(0, 13), (11, 13)]).buffer(.5)
bottleNeckField["obs2"] = LineString([(13, 13), (23,13)]).buffer(.5)
description = "Planner has to find a narrow passage."
benchList.append(Benchmark("Bottleneck", CollisionChecker(bottleNeckField, limits), [[4,15]], [[18,1]], description, 3))  # c = 0.5

# -----------------------------------------
fatBottleNeckField = dict()
fatBottleNeckField["obs1"] = Polygon([(0, 8), (11, 8),(11, 15), (0, 15)]).buffer(.5)
fatBottleNeckField["obs2"] = Polygon([(13, 8), (24, 8),(24, 15), (13, 15)]).buffer(.5)
description = "Planner has to find a narrow passage with a significant extent."
benchList.append(Benchmark("Fat bottleneck", CollisionChecker(fatBottleNeckField, limits), [[4,21]], [[18,1]], description, 3))  # c = 0.5

# -----------------------------------------

myField = dict()
myField["L"] = Polygon([(10, 16), (10, 11), (13, 11), (13,12), (11,12), (11,16)])
myField["T"] = Polygon([(14,16), (14, 15), (15, 15),(15,11), (16,11), (16,15), (17, 15), (17, 16)])
myField["C"] = Polygon([(19, 16), (19, 11), (22, 11), (22, 12), (20, 12), (20, 15), (22, 15), (22, 16)])

myField["Antenna_L"] = Polygon([(3, 12), (1, 16), (2, 16), (4, 12)])
myField["Antenna_Head_L"] = Point(1.5, 16).buffer(1)

myField["Antenna_R"] = Polygon([(7, 12), (9, 16), (8, 16), (6, 12)])
myField["Antenna_Head_R"] = Point(8.5, 16).buffer(1)

myField["Rob_Head"] = Polygon([(2, 13), (2, 8), (8, 8), (8, 13)])
description = "Planner has to find a passage past a robot head and the print of the LTC."
benchList.append(Benchmark("MyField", CollisionChecker(myField, limits), [[4,21]], [[18,1]], description, 2))  # c = 1.0
