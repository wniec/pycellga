from networkx import erdos_renyi_graph
from pycellga.mutation.byte_mutation_random import ByteMutationRandom
from pycellga.recombination.byte_one_point_crossover import ByteOnePointCrossover
from pycellga.selection.tournament_selection import TournamentSelection

from GraphProblem import GraphRastriginProblem
from optimizer import graph_cga

if __name__ == "__main__":
    graph = erdos_renyi_graph(n=100, p=0.5)
    result = graph_cga(
        n_rows=10,
        n_cols=10,
        n_gen=100,
        ch_size=10,
        p_crossover=0.9,
        p_mutation=0.2,
        problem=GraphRastriginProblem(n_var=10, graph=graph, offset=2),
        selection=TournamentSelection,
        recombination=ByteOnePointCrossover,
        mutation=ByteMutationRandom,
        seed_par=100,
    )

    # Print the results
    print("Best solution chromosome:", result.chromosome)
    print("Best fitness value:", result.fitness_value)

    # Expected Output:
    # Best solution chromosome: [2.0 for i in range(10)]
    # Best fitness value: -10_000
