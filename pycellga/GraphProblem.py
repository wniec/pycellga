import numpy as np
from networkx import Graph
from pycellga.common import GeneType
from pycellga.problems.abstract_problem import AbstractProblem


class GraphProblem(AbstractProblem):
    def __init__(self, n_var, graph: Graph):
        super().__init__(gen_type=GeneType.REAL, n_var=n_var, xl=-100, xu=100)
        self.graph = graph

    def f(self, x: np.ndarray) -> float:
        return np.round(np.sum(x**2), 3)


class GraphRastriginProblem(AbstractProblem):
    def __init__(self, n_var, graph: Graph, offset: float, boudaries=5):
        super().__init__(
            gen_type=GeneType.REAL, n_var=n_var, xl=-5, xu=5
        )  #  Typical optimization interval for rastrigin function
        self.graph = graph
        self.n_var = n_var
        self.offset = offset
        self.xl = [- boudaries] * n_var
        self.xu = [boudaries] * n_var


    def f(self, x: np.ndarray) -> float:
        # ręczna implementacja funkcji rastrigina w n wymiarach
        return (
            10
            * self.n_var
            * np.sum(
                (x - self.offset) ** 2 - 10 * np.cos(2 * np.pi * (x - self.offset))
            )
        )
