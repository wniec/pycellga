from networkx import erdos_renyi_graph
from pycellga.mutation.byte_mutation_random import ByteMutationRandom
from pycellga.recombination.byte_one_point_crossover import ByteOnePointCrossover
from pycellga.selection.tournament_selection import TournamentSelection

from GraphProblem import GraphProblem
from optimizer import graph_cga

if __name__ == "__main__":
    graph = erdos_renyi_graph(n=25, p=0.5)
    result = graph_cga(
        n_rows=5,
        n_cols=5,
        n_gen=100,
        ch_size=5,
        p_crossover=0.9,
        p_mutation=0.2,
        problem=GraphProblem(n_var=5, graph=graph),
        selection=TournamentSelection,
        recombination=ByteOnePointCrossover,
        mutation=ByteMutationRandom,
        seed_par=100,
    )

    # Print the results
    print("Best solution chromosome:", result.chromosome)
    print("Best fitness value:", result.fitness_value)

# Expected Output:
# Best solution chromosome: [0.0, 0.0, 0.0, 0.0, 0.0]
# Best fitness value: 0.0
