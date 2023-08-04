import os 
import copy 
import random
from collections import deque, defaultdict
from itertools import islice

# import jax 
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

# jax.tree_util.register_pytree_node(
#   SliceableDeque,
#   flatten_func=lambda sd: (sd, None),
#   unflatten_func=lambda treedef, leaves: SliceableDeque(leaves)
# )    
    

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


def get_lkh3_tour(tour_path):
    with open(tour_path) as f:
        while line := f.readline():
            if line.rstrip() == 'TOUR_SECTION':
                tour_before_mapping = []
                while (node := f.readline().rstrip()) != '-1':
                    tour_before_mapping.append(int(node))
    _node_reorder = [1, 2] + [i for i in range(1, len(tour_before_mapping) + 1) if i > 2 and i % 2 != 0] + [i for i in range(1, len(tour_before_mapping) + 1) if i > 2 and i % 2 == 0]
    node_index_mapping = {old_index: new_index for new_index, old_index in enumerate(_node_reorder)}
    tour_before_reverse = [node_index_mapping[old_index] for old_index in tour_before_mapping]
    subtour_to_reverse = tour_before_reverse[2:]
    tour = tour_before_reverse[:2] + subtour_to_reverse[::-1] + [0]
    return tour 


def get_ortools_tour(tour_path, skip_first_lines: int = 3, num_taxi: int = 1):
    with open(tour_path) as f:
        for _ in range(skip_first_lines):
            next(f)
        tour_before_adding_dummy = list(map(int, f.readline().rstrip().split(' -> ')))
        tour = [0] + tour_before_adding_dummy[:-1] + [0]
    return tour 


def generate_pdtsp_instance(num_O: int, 
                            instance_save_to_dir: str,
                            lkh3_instance_save_to_dir: str = None,
                            *, 
                            random_seed_range: list = [0, 2**15],
                            x_range: list = [0, 1000],
                            y_range: list = [0, 1000],
                            ):
    seed = random.randint(*random_seed_range)
    np.random.seed(seed)
    dim = num_O * 2 + 2
    # Uniformly generate (x,y) coordinate
    x = np.random.uniform(low=x_range[0], high=x_range[1], size=dim).astype(int)
    y = np.random.uniform(low=y_range[0], high=y_range[1], size=dim).astype(int)
    # The first two rows ought to be the same (x,y) coordinate
    x[0] = x[1]
    y[0] = y[1]

    os.makedirs(instance_save_to_dir, exist_ok=True)
    instance_name_header = f'random-{num_O:03}-{seed:05}'
    instance_name = instance_name_header + '.tsp'

    # header
    instance_lines = [
        f'NAME: {instance_name_header}',
        'TYPE: TSP',
        f'COMMENT: size={num_O} seed={seed}',
        f'DIMENSION: {dim}',
        'EDGE_WEIGHT_TYPE: EUC_2D',
        'NODE_COORD_SECTION'
    ]
    # NODE_COORD_SECTION
    instance_lines += [f'+{i // 2} {x[i]} {y[i]}' if i % 2 == 0 else f'-{i // 2} {x[i]} {y[i]}' for i in range(dim) ]
    # PRECEDENCE_SECTION
    instance_lines += ['PRECEDENCE_SECTION']
    instance_lines += [f'+{i} -{i}' for i in range(num_O + 1)]
    instance_lines += ['EOF']

    with open(os.path.join(instance_save_to_dir, instance_name), 'w') as f:
        for line in instance_lines:
            f.write(line + '\n')
    print(f'Instance {instance_name} saved to {instance_save_to_dir}')

    if lkh3_instance_save_to_dir is not None:
        os.makedirs(lkh3_instance_save_to_dir, exist_ok=True)
        lkh3_instance_name_header = f'random-{num_O:03}-{seed:05}'
        lkh3_instance_name = lkh3_instance_name_header + '.pdtsp'
        # lkh3 requires a file that differs from random-uniform
        pd_matrix = np.zeros((dim, 7))
        indexes = np.arange(dim) + 1
        pd_matrix[:, 0] = indexes

        # pickup and delivery
        pickup = copy.deepcopy(indexes)
        pickup[:2] = 0
        pickup_mask = indexes % 2 == 0
        pickup[pickup_mask] = 0
        pickup = np.roll(pickup, 1)
        pd_matrix[:, -1] = pickup

        delivery = copy.deepcopy(indexes)
        delivery[:2] = 0
        delivery_mask = indexes % 2 != 0
        delivery[delivery_mask] = 0
        delivery = np.roll(delivery, -1)
        pd_matrix[:, -2] = delivery

        lkh3_instance_lines = [
            f'NAME: {lkh3_instance_name_header}',
            'TYPE: PDTSP',
            f'COMMENT: size={num_O} seed={seed}',
            f'DIMENSION: {dim}',
            'EDGE_WEIGHT_TYPE: EUC_2D',
            'NODE_COORD_SECTION',    
        ]
        # NODE_COORD_SECTION
        lkh3_instance_lines += [f'{i + 1} {x[i]} {y[i]}' for i in range(dim)]
        # PRECEDENCE_SECTION
        lkh3_instance_lines += ['PICKUP_AND_DELIVERY_SECTION']
        lkh3_instance_lines += [' '.join(map(lambda x: str(int(x)), pd_matrix[i, :])) for i in range(dim)]
        # FIXED_EDGES_SECTION
        lkh3_instance_lines += ['FIXED_EDGES_SECTION', '1 2', '-1', 'EOF']

        with open(os.path.join(lkh3_instance_save_to_dir, lkh3_instance_name), 'w') as f:
            for line in lkh3_instance_lines:
                f.write(line + '\n')
        print(f'Instance {lkh3_instance_name} saved to {lkh3_instance_save_to_dir}')