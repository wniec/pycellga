from typing import List
import numpy as np
from networkx import Graph
from pycellga.grid import Grid
from pycellga.individual import Individual
from pycellga.population import Population, OptimizationMethod
from pycellga.problems.abstract_problem import AbstractProblem


class GraphPopulation(Population):
    def __init__(
        self,
        graph: Graph,
        method_name: OptimizationMethod = OptimizationMethod.CGA,
        ch_size: int = 0,
        n_rows: int = 0,
        n_cols: int = 0,
        gen_type: str = "",
        problem: AbstractProblem = None,
        vector: list = [],
        mins: list[float] = [],
        maxs: list[float] = [],
    ):
        super().__init__(
            method_name, ch_size, n_rows, n_cols, gen_type, problem, vector, mins, maxs
        )
        self.graph = graph

    def initial_population(self) -> List[Individual]:
        """
        Generate the initial population of individuals.

        Returns
        -------
        List[Individual]
            A list of initialized `Individual` objects with their respective chromosomes, fitness values, positions, and neighbors.
        """
        pop_size = self.n_rows * self.n_cols
        pop_list = []

        grid = Grid(self.n_rows, self.n_cols).make_2d_grid()
        neighbors = [self.graph.neighbors(i) for i in self.graph.nodes]
        for i in range(pop_size):
            ind = Individual(
                gen_type=self.gen_type,
                ch_size=self.ch_size,
                mins=self.mins,
                maxs=self.maxs,
            )

            # Initialize chromosome and evaluate fitness for cga, syn_cga and alpha_cga
            ind.chromosome = ind.randomize()
            ind.fitness_value = self.problem.f(np.array(ind.chromosome))

            ind.position = grid[i]
            ind.neighbors = self.graph.neighbors(i)
            pop_list.append(ind)
        for i in range(pop_size):
            pop_list[i].neighbors_positions = [
                pop_list[v].position for v in neighbors[i]
            ]
        return pop_list
