import os 
import copy 
import random
from collections import deque, defaultdict
from itertools import islice, permutations

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


def random_split_dict_with_capacity(d, num_splits, capacities, capacity):
    """Randomly split the dict into `num_splits` parts, each part will not exceed capacity."""
    keys = list(d.keys())
    gen = feasible_splits_generator(d, keys, capacities, num_splits, max_capacity_per_split=capacity)
    selected_split = next(gen)
    return [{key: d[key] for key in split.keys()} for split in selected_split]


def feasible_splits_generator(d, keys, capacities, num_splits, max_capacity_per_split):
    while True:  # Infinite loop to keep generating feasible splits.
        shuffled_keys = random.sample(keys, len(keys))  # Shuffle keys before generating permutations.
        for perm in permutations(shuffled_keys):
            splits = [{} for _ in range(num_splits)]
            capacities_left = [max_capacity_per_split for _ in range(num_splits)]
            is_feasible = True
            for key in perm:
                value = d[key]
                _item_cap = capacities[key] + capacities[value]
                allocated = False
                for i in range(num_splits):
                    if capacities_left[i] >= _item_cap:
                        splits[i][key] = True  # Placeholder, replace with actual value from original dict.
                        capacities_left[i] -= _item_cap
                        allocated = True
                        break
                if not allocated:
                    is_feasible = False
                    break
            if is_feasible:
                yield splits


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
                   to_annotate: bool = True, 
                   quiver_width: float = 5e-3, 
                   display_back_to_dummy: bool = False,
                   display_title: bool = True,
                   display_legend: bool = False,
                   annotate_font_size: int = 12,
                   node_markersize: int = 10,
                   custom_o_color: str = None,
                   custom_d_color: str = None,
                   annotate_dummy: bool = False,
                   annotate_number: bool = False,
                   display_axis: bool = False):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    if not display_axis:
        plt.axis('off')
    cost = problem.calc_cost(solution)
    if display_title:
        fig_name = f'{problem}--Cost:{cost:.2f}' if fig_name is None else fig_name
        fig.suptitle(fig_name)
    colors = cm.rainbow(np.linspace(0, 1, 4))  # 4 different color for dummy, taxi, O and D respectively
    path_colors = cm.rainbow(np.linspace(0, 1, len(solution.paths))) # different colors for paths
    path_colors[0] = [0, 0, 0, 1] # the first color is set to black
    x, y = problem.locations[:, 0], problem.locations[:, 1]
    for _path_id, path in enumerate(solution.paths):
        p = [node for node in path]
        end = len(p) - 2 if not display_back_to_dummy else len(p) - 1
        for i in range(end): # not displaying back to dummy arrow
            di, ai = p[i], p[i + 1]
            # plt.plot([x[di], x[ai]], [y[di], y[ai]], 'k-')
            plt.quiver(x[di], y[di], x[ai] - x[di], y[ai] - y[di], scale_units='xy', angles='xy', scale=1, width=quiver_width, color=path_colors[_path_id])
        
    plt.plot(x[0], y[0], 'o', color=colors[0], alpha=1, markersize=node_markersize)
    if problem.contains_taxi_node:
        plt.plot(x[1: 1 + problem.num_taxi], y[1: 1 + problem.num_taxi], 'o', color=colors[1], alpha=1, markersize=node_markersize, label='depot')
    plt.plot(x[problem.O], y[problem.O], 'o', markersize=node_markersize, color=colors[2] if custom_o_color is None else custom_o_color, alpha=1, label='pickup')
    plt.plot(x[problem.D], y[problem.D], 'o', markersize=node_markersize, color=colors[3] if custom_d_color is None else custom_d_color, alpha=1, label='delivery')
    if to_annotate:
        if annotate_dummy:
            plt.annotate('dummy', (x[0], y[0]), fontsize=annotate_font_size)
        if problem.contains_taxi_node:
            for i in range(1, 1 + problem.num_taxi):
                plt.annotate(f'taxi{i}' if not annotate_number else f'0', (x[i], y[i]), fontsize=annotate_font_size)
        for idx, i in enumerate(problem.O, start=1):
            plt.annotate(f'O{idx}' if not annotate_number else f'{idx}', (x[i], y[i]), fontsize=annotate_font_size)
        for idx, i in enumerate(problem.D, start=1):
            plt.annotate(f'D{idx}' if not annotate_number else f'{idx + len(problem.O)}', (x[i], y[i]), fontsize=annotate_font_size)
    if display_legend:
        plt.legend()
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


def read_pdptw_instance_data(instance_path, capacity_slack: float = 0.):
    locations = defaultdict(deque)
    capacities = defaultdict(int)
    def _get_data(line: str):
        return line[line.index(":") + 1:].strip()
    
    with open(instance_path) as f:
        while line := f.readline():
            if line.startswith('DIMENSION'):
                _dim = int(_get_data(line))
            elif line.startswith('VEHICLES'):
                num_vehicles = int(_get_data(line))
            elif line.startswith('CAPACITY'):
                capacity = int(_get_data(line))
            elif line.startswith('EDGE_WEIGHT_TYPE'):
                distance_type = _get_data(line)
            elif line.rstrip() == 'NODE_COORD_SECTION':
                locations['depot'] = np.array(f.readline().rstrip().split()[1:], dtype=float, ndmin=2)  # 1
                _end_o = (_dim - 1) // 2
                i = 0
                while (loc := f.readline().rstrip()) != 'PICKUP_AND_DELIVERY_SECTION':
                    i += 1
                    if i <= _end_o:
                        locations['O'].append(loc.split()[1:])
                    else:
                        locations['D'].append(loc.split()[1:])
                else:
                    total_capacity = 0
                    while (loc := f.readline().rstrip()) != 'DEPOT_SECTION':
                        _node, _capacity, *_ = loc.split()
                        _capacity = int(_capacity)
                        # use non-negative capacity
                        _capacity = _capacity if _capacity >= 0 else 0                
                        total_capacity += _capacity
                        capacities[int(_node) - 1] = _capacity
                    break 
        f.close()
    locations['O'], locations['D'] = np.asarray(locations['O'], dtype=float), np.asarray(locations['D'], dtype=float)
    capacity = (total_capacity * (1 + capacity_slack) // num_vehicles) + 1
    return locations, capacities, num_vehicles, capacity, distance_type


def read_pdvrp_instance_data(instance_path):
    locations = defaultdict(deque)
    with open(instance_path) as f:
        while line := f.readline():
            if line.startswith('COMMENT'):
                num_taxi = int(line[line.index('vehicle') + 8: line.index('seed')])
            if line.rstrip() == 'NODE_COORD_SECTION':
                # dummy(+0), taxi(-0)
                locations['dummy'] = np.array(f.readline().rstrip().split()[1:], dtype=float, ndmin=2)
                for _ in range(num_taxi):
                    loc = f.readline().rstrip()
                    locations['taxi'].append(loc.split()[1:]) 
                while (loc := f.readline().rstrip()) != 'PRECEDENCE_SECTION':
                    if loc.startswith('+'):
                        locations['O'].append(loc.split()[1:])
                    else:
                        locations['D'].append(loc.split()[1:])
                break 
        f.close()
    locations['taxi'], locations['O'], locations['D'] = np.asarray(locations['taxi'], dtype=float), np.asarray(locations['O'], dtype=float), np.asarray(locations['D'], dtype=float)
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


def get_lkh3_tour_v2(tour_path):
    with open(tour_path) as f:
        while line := f.readline():
            if line.rstrip() == 'TOUR_SECTION':
                tour_before_mapping = []
                while (node := f.readline().rstrip()) != '-1':
                    tour_before_mapping.append(int(node))
    _node_reorder = [1, 2] + [i for i in range(1, len(tour_before_mapping) + 1) if i > 2 and i % 2 != 0] + [i for i in range(1, len(tour_before_mapping) + 1) if i > 2 and i % 2 == 0]
    node_index_mapping = {old_index: new_index for new_index, old_index in enumerate(_node_reorder, start=-1)}
    tour_before_reverse = [node_index_mapping[old_index] for old_index in tour_before_mapping if old_index > 1]  # two dummy nodes at the same location
    subtour_to_reverse = tour_before_reverse[1:]
    tour = tour_before_reverse[:1] + subtour_to_reverse[::-1] + [0]
    return tour 


def get_ortools_tour(tour_path, skip_first_lines: int = 2, num_taxi: int = 1, include_taxi_node: bool = True):
    with open(tour_path) as f:
        for _ in range(skip_first_lines):
            next(f)
        paths = []
        for _ in range(num_taxi):
            next(f)
            if include_taxi_node:
                tour_before_adding_dummy = list(map(int, f.readline().rstrip().split(' -> ')))
                tour = [0] + tour_before_adding_dummy[:-1] + [0]
            else:
                tour = list(map(int, f.readline().rstrip().split(' -> ')))
            paths.append(tour)
            next(f)
    return paths 


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


def generate_pdvrp_instance(num_O: int, 
                            num_taxi: int, 
                            instance_save_to_dir: str,
                            # lkh3_instance_save_to_dir: str = None,
                            seed: int = None,
                            *, 
                            random_seed_range: list = [0, 2**15],
                            x_range: list = [0, 1000],
                            y_range: list = [0, 1000],
                            ):
    seed = random.randint(*random_seed_range) if seed is None else seed
    np.random.seed(seed)
    dim = num_O * 2 + 1 + num_taxi
    # Uniformly generate (x,y) coordinate
    x = np.random.uniform(low=x_range[0], high=x_range[1], size=dim).astype(int)
    y = np.random.uniform(low=y_range[0], high=y_range[1], size=dim).astype(int)
    # The first num_taxi + 1 rows ought to be the same (x,y) coordinate
    x[1: 1 + num_taxi] = x[0]
    y[1: 1 + num_taxi] = y[0]

    os.makedirs(instance_save_to_dir, exist_ok=True)
    instance_name_header = f'random-{num_O:03}-{num_taxi:05}-{seed:05}'
    instance_name = instance_name_header + '.pdvrp'

    # header
    instance_lines = [
        f'NAME: {instance_name_header}',
        'TYPE: PDVRP',
        f'COMMENT: size={num_O} vehicle={num_taxi} seed={seed}',
        f'DIMENSION: {dim}',
        'EDGE_WEIGHT_TYPE: EUC_2D',
        'NODE_COORD_SECTION'
    ]
    # NODE_COORD_SECTION
    instance_lines += [f'+0 {x[0]} {y[0]}'] + [f'-0 {x[i]} {y[i]}' for i, _ in enumerate(range(num_taxi), start=1)]
    _shift = num_taxi + 1
    instance_lines += [f'+{(i - _shift + 2) // 2} {x[i]} {y[i]}' if i % 2 == 0 else f'-{(i - _shift + 2) // 2} {x[i]} {y[i]}' for i, _ in enumerate(range(num_O * 2), start=_shift) ]
    # PRECEDENCE_SECTION
    instance_lines += ['PRECEDENCE_SECTION']
    instance_lines += [f'+{i} -{i}' for i in range(num_O + 1)]
    instance_lines += ['EOF']

    with open(os.path.join(instance_save_to_dir, instance_name), 'w') as f:
        for line in instance_lines:
            f.write(line + '\n')
    print(f'Instance {instance_name} saved to {instance_save_to_dir}')

    