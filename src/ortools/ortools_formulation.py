import numpy as np
from ortools.linear_solver import pywraplp

def ortools_formulation(problem, formulation_type: str = '3D', name: str = 'MultiOD'):
    if formulation_type == '2D':
        X, m = ortools_formulation_2D(problem, name)
    elif formulation_type == '3D':
        X, m = ortools_formulation_3D(problem, name)
    return X, m

def ortools_formulation_3D(p, name: str = 'MultiOD'):
    pass

def ortools_formulation_2D(p, name: str = 'MultiOD'):
    """This formulation is incomplete. Only works for single taxi case."""
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # Variables: Matrix, for element ij, it is interpreted as travels from i to j
    X = np.empty(p.distance_matrix.shape, dtype=object)  # Decision variable matrix
    for i in range(p.distance_matrix.shape[0]):
        for j in range(p.distance_matrix.shape[1]):
            X[i, j] = solver.BoolVar('x[%i,%i]' % (i, j))

    # Constraint 1: dummy to taxi and only to taxi
    con11 = solver.Add(1 <= np.sum(X[0, 1: 1 + p.num_taxi]))
    con12 = solver.Add(np.sum(X[0, 1: 1 + p.num_taxi]) <= p.num_taxi)
    con13 = solver.Add(np.sum(X[0, 1 + p.num_taxi:]) == 0)

    # Constraint 2: no travel from a node to itself
    con2 = solver.Add(np.sum(X.diagonal()) == 0)

    # Constraint 3: Vehicle leaves node that it enters
    for i in range(X.shape[0]):
        solver.Add(np.sum(X[i, :]) == np.sum(X[:, i]))

    # Constraint 4: Every node except dummy is visited once
    for i in range(1, X.shape[1]):
        solver.Add(np.sum(X[:, i]) == 1)

    # Constraint 5: Subtour elimination (MTZ method)
    t = [solver.NumVar(1, len(p.node_index), f't[{i}]') for i in p.node_index]
    for i, _i in enumerate(t[1:], start=1): # The index of t starts from 1
        for j, _j in enumerate(t[1:], start=1):
            if i != j:
                solver.Add(t[j] >= t[i] + 1 - (2 * len(t)) * (1 - X[i, j]))

    # Constraint 6: For every OD pair, O is visited before D and in the same taxi
    for i in range(len(p.O)):
        solver.Add(t[p.O[i]] <= t[p.D[i]])

    # Objective: shortest distance
    solver.Minimize((X * p.distance_matrix).sum())
    solver.Solve()

    X = [[X[i, j].solution_value() for j in range(X.shape[1])] for i in range(X.shape[0])]

    return X, solver




