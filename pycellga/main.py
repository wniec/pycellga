import networkx as nx
from networkx import erdos_renyi_graph, grid_graph
from pycellga.mutation.byte_mutation_random import ByteMutationRandom
from pycellga.recombination.byte_one_point_crossover import ByteOnePointCrossover
from pycellga.selection.tournament_selection import TournamentSelection

from GraphProblem import GraphRastriginProblem
from optimizer import graph_cga

if __name__ == "__main__":
    graph = erdos_renyi_graph(n=100, p=0.5)
    torus_graph = grid_graph(dim=(10, 10), periodic=True)
    torus_graph_nodes_flattened = {node: i for i, node in enumerate(torus_graph.nodes)}
    empty_graph = nx.Graph()
    for edge in torus_graph.edges:
        empty_graph.add_edge(torus_graph_nodes_flattened[edge[0]], torus_graph_nodes_flattened[edge[1]])
    # Wierzchołki w grafie torus mają 2 indeksy, więc muszę im pozamieniać nazwy
    torus_graph = empty_graph
    n_rows, n_cols = 10, 10
    n_generations = 100
    result = graph_cga(
        n_rows=n_rows,
        n_cols=n_cols,
        n_gen=n_generations,
        ch_size=10,
        p_crossover=0.9,
        p_mutation=0.2,
        problem=GraphRastriginProblem(n_var=10, graph=torus_graph, offset=2),
        selection=TournamentSelection,
        recombination=ByteOnePointCrossover,
        mutation=ByteMutationRandom,
        seed_par=100,
    )

    # Print the results
    print("Best solution chromosome:", result.chromosome)
    print("Best fitness value:", result.fitness_value)
    print("Function evaluations:", n_generations * n_rows * n_cols)

    # Expected Output:
    # Best solution chromosome: [2.0 for i in range(10)]
    # Best fitness value: -10_000
