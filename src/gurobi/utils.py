import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import networkx as nx 


def generate_paths_from_gurobi_result(X):
    if X.ndim == 2:
        G=nx.DiGraph()
        X_res = np.vectorize(lambda v: v.x)(X)
        departs, arrivals = X_res.nonzero()
        G.add_nodes_from(set(arrivals) & set(departs))
        G.add_edges_from(zip(departs, arrivals))
        paths = [cycle + [cycle[0]] for cycle in nx.simple_cycles(G)]
    elif X.ndim == 3:
        raise NotImplementedError
    return paths


def display_gurobi_result(X, model, problem, 
                          figsize: tuple = (8, 6), dpi: float = 80, fig_name: str = None, 
                          to_annotate: bool = True, quiver_width: float = 5e-3):
    """Display the path of solved problem."""
    if X.ndim == 2:
        X_res = np.vectorize(lambda v: v.x)(X)
        departs, arrivals = X_res.nonzero()
        fig = plt.figure(figsize=figsize, dpi=dpi)
        fig_name = f'{problem}--Obj:{model.ObjVal:.2f}--Gap:{model.MIPGap:.2f}' if fig_name is None else fig_name
        fig.suptitle(fig_name)
        colors = cm.rainbow(np.linspace(0, 1, 4))  # 4 different color for dummy, taxi, O and D respectively
        x, y = problem.locations[:, 0], problem.locations[:, 1]
        for i in range(len(departs)):
            di, ai = departs[i], arrivals[i]
            # plt.plot([x[di], x[ai]], [y[di], y[ai]], 'k-')
            plt.quiver(x[di], y[di], x[ai] - x[di], y[ai] - y[di], scale_units='xy', angles='xy', scale=1, width=quiver_width)
            
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
    elif X.ndim == 3:
        raise NotImplementedError
    return fig 