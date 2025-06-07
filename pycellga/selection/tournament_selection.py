import numpy as np
from typing import List

from pycellga.individual import Individual
from pycellga.selection.selection_operator import SelectionOperator

import random
class TournamentSelection(SelectionOperator):
    """
    TournamentSelection performs a tournament selection on a population of individuals
    to select parent individuals for crossover.

    Parameters
    ----------
    pop_list : list of Individual
        The population of individuals to select from.
    c : int
        The index of the individual to start selection from.
    K : int
        The number of individuals to be chosen at random from neighbors.
    """

    def __init__(self, pop_list: List[Individual] = [], c: int = 0, K: int = 2):
        """
        Initialize the TournamentSelection object.

        Parameters
        ----------
        pop_list : list of Individual
            The population of individuals to select from.
        c : int
            The index of the individual to start selection from.
        K : int
            The number of individuals to be chosen at random from neighbors.
        """
        self.pop_list = pop_list
        self.c = c
        self.K = K

    def get_parents(self) -> List[Individual]:
        """
        Perform the tournament selection to get parent individuals.
        Returns
        -------
        list of Individual
            A list containing the selected parent individuals.
        """
        parents = []

        # Always include self as p1
        p1 = self.pop_list[self.c - 1]
        parents.append(p1)

        # Retrieve neighbors
        neighbors_positions = getattr(p1, "neighbors_positions", [])
        neighbors = [
            ind for ind in self.pop_list if getattr(ind, "position", None) in neighbors_positions
        ]

        # Fallback if no neighbors found
        if not neighbors:
            print(f"[WARN] Individual {self.c} has no neighbors. Using self-mating.")
            parents.append(p1)
            return parents

        # Reduce K if fewer neighbors than K
        k = min(self.K, len(neighbors))
        tournament_selection_pool = random.sample(neighbors, k)

        # Sort the tournament selection pool by fitness value in descending order
        tournament_selection_pool_ordered = sorted(
            tournament_selection_pool, key=lambda x: x.fitness_value, reverse=True
        )

        # Select the best from tournament as p2
        p2 = tournament_selection_pool_ordered[0]
        parents.append(p2)

        return parents
