#!/usr/bin/env python3
'''
Author: Haoran Peng
Email: gavinsweden@gmail.com
'''
from typing import Dict
import numpy as np

from MAPF_CBS.agent import Agent
from MAPF_CBS.constraints import Constraints

class CTNode:

    def __init__(self, constraints: Constraints,
                       solution: Dict[Agent, np.ndarray]):

        self.constraints = constraints
        self.solution = solution
        self.cost = self.sic(solution)

    # Sum-of-Individual-Costs heuristics
    @staticmethod
    def sic(solution):
        # print(sum(len(sol) for sol in solution.items()))
        return sum(len(sol) for sol in solution.items())

    def __lt__(self, other):
        return self.cost < other.cost

    def __str__(self):
        return str(self.constraints.agent_constraints)

