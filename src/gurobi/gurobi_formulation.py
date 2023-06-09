# solve TSP 
# reference demo on https://github.com/Gurobi/modeling-examples/blob/master/traveling_salesman/tsp.ipynb
import numpy as np 
import gurobipy as gp
from gurobipy import GRB


def formulation(problem, formulation_type: str = '3D', name: str = 'MultiOD'):
    if formulation_type == '2D':
        X, m = formulation_2D(problem, name)
    elif formulation_type == '3D':
        X, m = formulation_3D(problem, name)
    return X, m 

def formulation_3D(p, name: str = 'MultiOD'):
    m = gp.Model(name) 
    # Variables: Matrix, for element ijz, it is interpreted as vehicle z travels from i to j
    X = m.addMVar((1 + 1 + len(p.O) + len(p.D),) * 2 + (p.num_taxi,), vtype=GRB.BINARY, name='x')
    
    # Constraint 1: dummy to taxi and only to taxi
    con11 = m.addConstr(X[0, 1] <= 1)
    con12 = m.addConstr(X[0, 1 + 1:] == 0)
    
    # Constraint 2: no travel from a node to itself
    con2 = m.addConstr(X.diagonal() == 0)

    # Constraint 3: Vehicle leaves node that it enters
    con3 = m.addConstr(X.sum(axis=0) == X.sum(axis=1))

    # Constraint 4: Every node except dummy is visited once
    con4 = m.addConstr(X[:, 2:].sum(axis=(0, 2)) == 1)

    # Constraint 5: Subtour eliminate(MTZ method)
    t = m.addMVar((1 + len(p.O) + len(p.D), p.num_taxi), lb=1, ub=len(p.node_index), vtype=GRB.CONTINUOUS)
    for k, _k in enumerate(range(p.num_taxi)):
        for i, _i in enumerate(t):
            for j, _j in enumerate(t):
                if i != j: m.addConstr(t[j, k] >= t[i, k] + 1 - (2 * t.shape[0]) * (1 - X[i, j, k]))

    # # Constraint 6: For every OD pair, OD are in the same taxi and O is visited before D 
    con61 = m.addConstrs(X[np.asarray(p.O) - p.num_taxi + 1, :, k].sum(axis=1) == X[np.asarray(p.D) - p.num_taxi + 1, :, k].sum(axis=1) for k in range(p.num_taxi))
    con62 = m.addConstr(t[np.asarray(p.O) - p.num_taxi] <= t[np.asarray(p.D) - p.num_taxi])

    # Objective: shortest distance
    obj = 0
    taxis = [*range(1, p.num_taxi + 1)]
    for idx, tid in enumerate(taxis):
        _taxis = taxis[:]
        _taxis.remove(tid)
        dis = np.delete(np.delete(p.distance_matrix, obj=_taxis, axis=0), obj=_taxis, axis=1)
        obj += dis * X[:, :, idx]
    m.setObjective(obj.sum(), sense=GRB.MINIMIZE)
    return X, m 

def formulation_2D(p, name: str = 'MultiOD'):
    """This formulation is incomplete. Only works for single taxi case."""
    m = gp.Model(name)
    # Variables: Matrix, for element ij, it is interpreted as travels from i to j
    X = m.addMVar(p.distance_matrix.shape, vtype=GRB.BINARY, name='x')

    # Constraint 1: dummy to taxi and only to taxi
    con11 = m.addConstr(1 <= (X[0, 1: 1 + p.num_taxi]).sum())
    con12 = m.addConstr((X[0, 1: 1 + p.num_taxi]).sum() <= p.num_taxi)
    con13 = m.addConstr(X[0, 1 + p.num_taxi:] == 0)

    # Constraint 2: no travel from a node to itself
    con2 = m.addConstr(X.diagonal() == 0)

    # Constraint 3: Vehicle leaves node that it enters
    con3 = m.addConstr(X.sum(axis=0) == X.sum(axis=1))

    # Constraint 4: Every node except dummy is visited once
    con4 = m.addConstr(X[:, 1:].sum(axis=0) == 1)

    # Constraint 5: Subtour eliminate(MTZ method)
    t = m.addMVar(len(p.node_index), lb=1, ub=len(p.node_index), vtype=GRB.CONTINUOUS)
    for i, _i in enumerate(t[1:], start=1): # The index of t starts from 1
        for j, _j in enumerate(t[1:], start=1):
            if i != j:
                m.addConstr(t[j] >= t[i] + 1 - (2 * t.size) * (1 - X[i, j]))

    # Constraint 6: For every OD pair, O is visited before D and in the same taxi
    con6 = m.addConstr(t[p.O] <= t[p.D])
    # Objective: shortest distance
    obj = (X * p.distance_matrix).sum()
    m.setObjective(obj, sense=GRB.MINIMIZE)
    return X, m 