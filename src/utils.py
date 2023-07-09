import random
from collections import deque, defaultdict
from itertools import islice

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

class SliceableDeque(deque):
    r"""A class implemented slice for collections.deque
    
    Reference: https://stackoverflow.com/questions/10003143/how-to-slice-a-deque
    """
    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(islice(self, index.start,
                                               index.stop, index.step))
        return deque.__getitem__(self, index)
    

def random_split_dict(d, num_splits):
    """Randomly split the dict into `num_splits` parts."""
    keys = list(d.keys())
    random.shuffle(keys)
    split_points = [0] + sorted(random.sample(range(1, len(keys)), num_splits-1)) + [None]
    
    return [dict((k, d[k]) for k in keys[split_points[i]:split_points[i+1]]) for i in range(num_splits)]


def generate_random_list_from_dict_with_key_before_value(d, shuffled_keys=None):
    """Generates random list from a dict and respects the rule that key appears before value."""
    if shuffled_keys is None:
        keys = list(d.keys())  
        random.shuffle(keys)
    else:
        keys = list(shuffled_keys) if not isinstance(shuffled_keys, list) else shuffled_keys

    result = []
    for key in keys:
        value = d[key]

        # Insert the key at a random index
        key_index = random.randint(0, len(result))
        result.insert(key_index, key)

        # Insert the value at a random index greater than the key's index
        value_index = random.randint(key_index + 1, len(result))
        result.insert(value_index, value)
    
    return result


def display_result(problem, solution,
                   figsize: tuple = (8, 6), dpi: float = 80, fig_name: str = None, 
                   to_annotate: bool = True, quiver_width: float = 5e-3):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    cost = problem.calc_cost(solution)
    fig_name = f'{problem}--Cost:{cost:.2f}' if fig_name is None else fig_name
    fig.suptitle(fig_name)
    colors = cm.rainbow(np.linspace(0, 1, 4))  # 4 different color for dummy, taxi, O and D respectively
    x, y = problem.locations[:, 0], problem.locations[:, 1]
    for path in solution.paths:
        p = [node for node in path]
        for i in range(len(p) - 2): # not displaying back to dummy arrow
            di, ai = p[i], p[i + 1]
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
    return fig


def read_instance_data(instance_path):
    locations = defaultdict(deque)
    with open(instance_path) as f:
        while line := f.readline():
            if line.rstrip() == 'NODE_COORD_SECTION':
                # +0, -0
                locations['dummy'] = np.array(f.readline().rstrip().split()[1:], dtype=float, ndmin=2)
                locations['taxi'] = np.array(f.readline().rstrip().split()[1:], dtype=float, ndmin=2)
                while (loc := f.readline().rstrip()) != 'PRECEDENCE_SECTION':
                    if loc.startswith('+'):
                        locations['O'].append(loc.split()[1:])
                    else:
                        locations['D'].append(loc.split()[1:])
                break 
        f.close()
    locations['O'], locations['D'] = np.asarray(locations['O'], dtype=float), np.asarray(locations['D'], dtype=float)
    return locations 