import numpy as np
import math
from mpmath import power as pw
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
    def __init__(self, n_var, graph: Graph, offset: float, boundaries=5):
        super().__init__(
            gen_type=GeneType.REAL, n_var=n_var, xl=-5, xu=5
        )  #  Typical optimization interval for rastrigin function
        self.graph = graph
        self.n_var = n_var
        self.offset = offset
        self.xl = [- boundaries] * n_var
        self.xu = [boundaries] * n_var


    def f(self, x: np.ndarray) -> float:
        # ręczna implementacja funkcji rastrigina w n wymiarach
        return (
            10
            * self.n_var
            * np.sum(
                (x - self.offset) ** 2 - 10 * np.cos(2 * np.pi * (x - self.offset))
            )
        )


    def __str__(self):
        return f"Rastrigin problem"

class GraphRosenbrockProblem(AbstractProblem):
    def __init__(self, n_var, graph: Graph, offset: float, boundaries=5):
        super().__init__(
            gen_type=GeneType.REAL, n_var=n_var, xl=-boundaries, xu=boundaries
        )
        self.graph = graph
        self.n_var = n_var
        self.offset = offset
        self.xl = [-boundaries] * n_var
        self.xu = [boundaries] * n_var

    def f(self, x: np.ndarray) -> float:
        x = x - self.offset
        

        fitness = sum(
            [
                (100 * np.power((x[i + 1] - np.power(x[i], 2)), 2)) + np.power((1 - x[i]), 2)
                for i in range(self.n_var - 1)
            ]
        )
        return fitness


    def __str__(self):
        return f"Rosenbrock problem"
    

class GraphAckleyProblem(AbstractProblem):
    def __init__(self, n_var, graph: Graph, offset: float, boundaries=32.768):
        super().__init__(
            gen_type=GeneType.REAL, n_var=n_var, xl=-boundaries, xu=boundaries
        )
        self.graph = graph
        self.n_var = n_var
        self.offset = offset
        self.xl = [-boundaries] * n_var
        self.xu = [boundaries] * n_var

    def f(self, x: np.ndarray) -> float:
        x = x - self.offset        

        sum1 = sum(gene ** 2 for gene in x)
        sum2 = sum(np.cos(2 * np.pi * gene) for gene in x)

        fitness = (
            -20.0 * np.exp(-0.2 * np.sqrt(sum1 / self.n_var))
            - np.exp(sum2 / self.n_var)
            + 20.0
            + np.e
        )
        return fitness

    def __str__(self):
        return f"Ackley problem"

class GraphGriewankProblem(AbstractProblem):
    def __init__(self, n_var, graph: Graph, offset: float, boundaries=600):
        super().__init__(
            gen_type=GeneType.REAL, n_var=n_var, xl=-boundaries, xu=boundaries
        )
        self.graph = graph
        self.n_var = n_var
        self.offset = offset
        self.xl = [-boundaries] * n_var
        self.xu = [boundaries] * n_var

    def f(self, x: np.ndarray) -> float:
        x = x - self.offset 
        sum_sq = sum(xi**2 for xi in x)
        prod_cos = math.prod(math.cos(xi / math.sqrt(i + 1)) for i, xi in enumerate(x))
        fitness = 1 + sum_sq / 4000 - prod_cos
        return fitness

    
    def __str__(self):
        return f"Griewank problem"