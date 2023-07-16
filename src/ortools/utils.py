import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx

def generate_paths_from_ortools_result(X):
    X_res = np.array(X)
    if X_res.ndim == 2:
        G=nx.DiGraph()
        departs, arrivals = X_res.nonzero()
        G.add_nodes_from(set(arrivals) & set(departs))
        G.add_edges_from(zip(departs, arrivals))
        paths = [cycle + [cycle[0]] for cycle in nx.simple_cycles(G)]
    elif X_res.ndim == 3:
        raise NotImplementedError
    return paths

def display_ortools_result(X, solver, problem,
                          figsize: tuple = (8, 6), dpi: float = 80, fig_name: str = None,
                          to_annotate: bool = True, quiver_width: float = 5e-3):
    """Display the path of solved problem."""
    X_res = np.array(X)
    if X_res.ndim == 2:
        departs, arrivals = X_res.nonzero()
        fig = plt.figure(figsize=figsize, dpi=dpi)
        # fig_name = f'{problem}--Obj:{model.ObjVal:.2f}--Gap:{model.MIPGap:.2f}' if fig_name is None else fig_name
        fig_name = f'{problem}--Obj:{solver.Objective().Value():.2f}' if fig_name is None else fig_name
        fig.suptitle(fig_name)
        colors = cm.rainbow(np.linspace(0, 1, 4))  # 4 different color for dummy, taxi, O and D respectively
        x, y = problem.locations[:, 0], problem.locations[:, 1]
        for i in range(len(departs)):
            di, ai = departs[i], arrivals[i]
            # plt.plot([x[di], x[ai]], [y[di], y[ai]], 'k-')
            plt.quiver(x[di], y[di], x[ai] - x[di], y[ai] - y[di], scale_units='xy', angles='xy', scale=1,
                       width=quiver_width)

        plt.plot(x[0], y[0], 'o', color=colors[0], alpha=1)
        plt.plot(x[1: 1 + problem.num_taxi], y[1: 1 + problem.num_taxi], 'o', color=colors[1], alpha=1)
        plt.plot(x[problem.O], y[problem.O], 'o', color=colors[2], alpha=1)
        plt.plot(x[problem.D], y[problem.D], 'o', color=colors[3], alpha=1)
        if to_annotate:
            plt.annotate('dummy', (x[0], y[0]))
            for i in range(1, 1 + problem.num_taxi):
                plt.annotate(f'taxi{i}', (x[i], y[i]))
            for idx, i in enumerate(problem.O, start=1):
                plt.annotate(f'O{idx}', (x[i], y[i]))
            for idx, i in enumerate(problem.D, start=1):
                plt.annotate(f'D{idx}', (x[i], y[i]))
    elif X_res.ndim == 3:
        raise NotImplementedError
    return fig

def display_pd_ortools_result(problem,solv):
    """Prints solution on console."""
    manager = solv[1]
    routing = solv[2]
    solution = solv[3]
    print(f'Objective: {solution.ObjectiveValue()}')
    total_distance = 0
    for vehicle_id in range(problem.num_taxi):
        path = []
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index)+1)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index)+1)
        plan_output += 'Distance of the route: {}m'.format(route_distance)
        print(plan_output)
        total_distance += route_distance
    print('Total Distance of all routes: {}m'.format(total_distance))
