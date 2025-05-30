import numpy as np
import cma
import networkx as nx
import matplotlib.pyplot as plt

from networkx import erdos_renyi_graph, grid_graph
from pycellga.mutation.byte_mutation_random import ByteMutationRandom
from pycellga.recombination.byte_one_point_crossover import ByteOnePointCrossover
from pycellga.selection.tournament_selection import TournamentSelection

from pycellga.GraphProblem import GraphRastriginProblem, GraphRosenbrockProblem, GraphAckleyProblem, GraphGriewankProblem
from pycellga.optimizer import graph_cga
from typing import Tuple, Union, Literal

from pyswarms.single.global_best import GlobalBestPSO
from scipy.optimize import differential_evolution


PROBLEM_TYPING = Union[
    GraphRastriginProblem,
    GraphRosenbrockProblem,
    GraphAckleyProblem,
    GraphGriewankProblem
]


SOLVER_TYPE = Literal[ "cma", "pso", "de"]


def set_up_problem(n_rows, n_cols, dims, offset, problem_type: PROBLEM_TYPING, boundaries: int = 5) -> PROBLEM_TYPING: # type: ignore
    torus_graph = grid_graph(dim=(n_rows, n_cols), periodic=True)
    torus_graph_nodes_flattened = {node: i for i, node in enumerate(torus_graph.nodes)}
    empty_graph = nx.Graph()
    for edge in torus_graph.edges:
        empty_graph.add_edge(torus_graph_nodes_flattened[edge[0]], torus_graph_nodes_flattened[edge[1]])
    torus_graph = empty_graph

    #setting up random offset to see 
    return problem_type(n_var=dims, graph=torus_graph, offset=offset, boundaries=boundaries)


def run_cga(dims, n_generations, n_rows, n_cols, problem):
    result = graph_cga(
        n_rows=n_rows,
        n_cols=n_cols,
        n_gen=n_generations,
        ch_size=dims,
        p_crossover=0.9,
        p_mutation=0.2,
        problem=problem,
        selection=TournamentSelection,
        recombination=ByteOnePointCrossover,
        mutation=ByteMutationRandom,
        seed_par=100,

    )
    return (result.chromosome, result.fitness_value)



def run_cma(problem, dims, max_evals, boundaries):
    x0 = np.random.random(dims)    
    sigma = 0.5

    es = cma.CMAEvolutionStrategy(x0, 
        sigma, 
        {
        'maxfeval': max_evals,
        'verb_disp': 0,
        'bounds': [[-boundaries, -boundaries], [boundaries, boundaries]]  
        }
        )

    while not es.stop():
        solutions = es.ask()                  
        function_value = [problem.f(x) for x in solutions]  
        es.tell(solutions, function_value) 
    return (es.result.xbest, es.result.fbest)                          


def run_pso(problem, dims, max_evals, boundaries):
    """
    Particle Swarm Optimization using PySwarms
    """
    # Konfiguracja PSO
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}  # cognitive, social, inertia parameters
    n_particles = 30  # liczba cząstek w rojku
    
    # Bounds w formacie PySwarms: (min_bounds, max_bounds)
    bounds = (np.full(dims, -boundaries), np.full(dims, boundaries))
    
    # Wrapper funkcji - PySwarms wymaga funkcji przyjmującej macierz (n_particles x dims)
    def objective_function(X):
        return np.array([problem.f(x) for x in X])
    
    # Inicjalizacja PSO
    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=dims,
        options=options,
        bounds=bounds
    )
    
    # Uruchomienie optymalizacji
    iters = max_evals // n_particles  # liczba iteracji
    best_cost, best_pos = optimizer.optimize(objective_function, iters=iters)
    
    return (best_pos, best_cost)


def run_differential_evolution(problem, dims, max_evals, boundaries):
    """
    Differential Evolution using SciPy
    """
    # Bounds w formacie SciPy: lista tupli [(min, max), (min, max), ...]
    bounds = [(-boundaries, boundaries) for _ in range(dims)]
    
    # Wrapper funkcji - SciPy DE wymaga funkcji przyjmującej pojedynczy wektor
    def objective_function(x):
        return problem.f(x)
    
    # Uruchomienie Differential Evolution
    result = differential_evolution(
        objective_function,
        bounds=bounds,
        maxiter=max_evals // 15,  # maxiter * popsize ≈ max_evals (domyślny popsize=15)
        seed=42,  # dla powtarzalności
        disp=False  # wyłącz wyświetlanie
    )
    
    return (result.x, result.fun)

def run_solver(problem, dims, max_evals, boundaries, solver_type: SOLVER_TYPE):
    if solver_type == "cma":
        return run_cma(problem, dims, max_evals, boundaries)
    elif solver_type == "pso":
        return run_pso(problem, dims, max_evals, boundaries)
    elif solver_type == "de":
        return run_differential_evolution(problem, dims, max_evals, boundaries)
    


from sklearn.decomposition import PCA
import seaborn as sns



def single_test(offset, n_rows, n_cols, n_generations, dims, problem_type: PROBLEM_TYPING, boundaries: int = 5, solver_type: SOLVER_TYPE = "cma", repetitions=10):
    
    solver_results = []
    cga_results = []
    
    problem = set_up_problem(n_rows, n_cols, dims, offset, problem_type, boundaries=boundaries)
    i = 0
    while i < repetitions:
        try:
            cga_result =run_cga(dims, n_generations, n_rows, n_cols, problem)
            solver_result = run_solver(problem, dims, n_generations * n_rows * n_cols, boundaries=boundaries, solver_type=solver_type)
            
        except ValueError as e: # sometimes it doesn't find element in list, just repeat
            print(e)
            continue
        
        cga_results.append(cga_result)
        solver_results.append(solver_result)
        i += 1

    return (cga_results, solver_results)
    
def visualize_results(cga_results, solver_result, parameters, minimum, solver_type: SOLVER_TYPE):
    values = np.array(list(map(lambda x: x[1], cga_results + solver_result)))
    sizes = - values
    
    sizes -= sizes.min()
    sizes = sizes/sizes.max() * 40 + 5


    i = len(solver_result)
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(np.array(list(map(lambda x: x[0], cga_results + solver_result))))
    plt.scatter(x=transformed[:i, 0], y=transformed[:i, 1], c='red', label='cga', s=sizes[:i], alpha=0.7)
    plt.scatter(x=transformed[i:, 0], y=transformed[i:, 1], c='blue', label=solver_type, s=sizes[i:], alpha=0.7)

    pca_min = pca.transform(minimum.reshape(1, -1))
    plt.scatter(x=pca_min[0, 0], y=pca_min[0, 1], c='green', s=40, label='global_minumum', marker='x')
    plt.title(f'dims: {parameters['dims']}   ROWSxCOLS: {parameters['n_rows']} x {parameters['n_cols']}')

    # for point, result in zip(transformed, cga_results + cma_results):
    #     print(point, result[1])
    #     plt.annotate(f"{result[1]:.0f}", point, point - np.array([0.2, 0.3]))
    plt.xlabel('pca1')
    plt.ylabel('pca2')
    plt.legend()
    plt.show()


def test_parameters(parameter_dicts, problem_type: PROBLEM_TYPING, solver_type: SOLVER_TYPE = "cma", repetitions=30, visualize_steps=True):
    
    for parameters in parameter_dicts:

        if problem_type == GraphRastriginProblem:
            minimum = np.ones(parameters['dims']) * parameters['offset']
            minimum_val = - 100 * parameters['dims'] ** 2
            # minimum_val = 0

        elif problem_type == GraphRosenbrockProblem:
            minimum = np.ones(parameters['dims']) + parameters['offset']
            minimum_val = 0

        elif problem_type == GraphAckleyProblem:
            minimum = np.ones(parameters['dims']) * parameters['offset']
            minimum_val = 0

        elif problem_type == GraphGriewankProblem:
            minimum = np.ones(parameters['dims']) * parameters['offset']
            minimum_val = 0

        cga_results, cma_results = single_test(**parameters, problem_type=problem_type,repetitions=repetitions, solver_type=solver_type)
        if visualize_steps:
            visualize_results(cga_results, cma_results, parameters, minimum, solver_type)

        distances_cga = list(map(lambda x:np.linalg.norm(x[0] - minimum), cga_results))
        distances_cma = list(map(lambda x:np.linalg.norm(x[0] - minimum), cma_results))
        errors_cga = list(map(lambda x:(np.array(x[1]) - minimum_val), cga_results))
        errors_cma = list(map(lambda x:(np.array(x[1]) - minimum_val), cma_results))

        print(errors_cma)

        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(10, 5)
        ax[0].set_title('distance to the best vector')
        ax[0].hist(distances_cga, label='cga', color='red', alpha=0.7, rwidth=0.5)
        ax[0].hist(distances_cma, label=solver_type, color='blue', alpha=0.7, rwidth=0.5)
        ax[0].set_xlabel('|vector - vector_min|')
        
        # ax[0].legend()
        ax[1].set_title('difference from optimal solution')
        ax[1].hist(errors_cga, label='cga', color='red', alpha=0.7, rwidth=200)
        ax[1].hist(errors_cma, label=solver_type, color='blue', alpha=0.7, rwidth=200)
        ax[1].legend()
        ax[1].set_xlabel('error (value - minimum)')


        plt.show()