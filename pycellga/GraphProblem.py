from typing import List

from mpmath import power as pw
from networkx import Graph
from pycellga.common import GeneType
from pycellga.problems.abstract_problem import AbstractProblem


class GraphProblem(AbstractProblem):
    def __init__(self, n_var, graph: Graph):
        super().__init__(gen_type=GeneType.REAL, n_var=n_var, xl=-100, xu=100)
        self.graph = graph

    def f(self, x: List[float]) -> float:
        return round(sum(pw(xi, 2) for xi in x), 3)
