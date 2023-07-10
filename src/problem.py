import copy 
import random
from itertools import product
import numpy as np

from . import solution
from . import utils 


MultiODSolution = solution.MultiODSolution
random_split_dict = utils.random_split_dict
generate_random_list_from_dict_with_key_before_value = utils.generate_random_list_from_dict_with_key_before_value


class Problem:
    """"""
    def __init__(self, *args, **kwargs):
        pass

    def generate_problem(self, *args, **kwargs):
        """This method generates the problem related information"""
        pass

    def record_solution(self, solution, distance):
        self.num_solutions += 1.0 / distance
        for path in solution:
            if len(path) > 2:
                for to_index in range(1, len(path)):
                    #TODO: change is needed for asymmetric cases.
                    self.num_traversed[path[to_index - 1]][path[to_index]] += 1.0 / distance
                    self.num_traversed[path[to_index]][path[to_index - 1]] += 1.0 / distance
                    # for index_in_the_same_path in range(to_index + 1, len(path)):
                    #     self.num_traversed[path[index_in_the_same_path]][path[to_index]] += 1
                    #     self.num_traversed[path[to_index]][path[index_in_the_same_path]] += 1

    def add_distance_hash(self, distance_hash):
        self.distance_hashes.add(distance_hash)

    def get_location(self, index):
        return self.locations[index]

    def get_distance(self, from_index, to_index):
        return self.distance_matrix[from_index][to_index]
    
    def is_feasible(self, solution: solution.Solution):
        pass

    def calc_cost(self, solution: solution.Solution):
        pass 

    def __repr__(self):
      return f'{type(self).__name__}(nodes={len(self.locations)})'


class MultiODProblem(Problem):
    """
    Parameters
    ------
    locations: dict, the keys should be 'O', 'D', 'taxi' and 'dummy'. The value should be a list or array of shape `(len, 2)`. 
        `O` and `D` must be provided. If `taxi` or `dummy` is not provided, the default value will be applied.
    """

    def __init__(self, num_O: int = 10, num_taxi: int = 1, locations: dict = None, seed: int = 0, 
                 ignore_from_dummy_cost: bool = True, ignore_to_dummy_cost: bool = True):
        self.num_taxi = num_taxi
        self.distance_matrix, self.O, self.D, self.locations, self.node_index = self.generate_problem(num_O=num_O, num_taxi=num_taxi, locations=locations, seed=seed, ignore_from_dummy_cost=ignore_from_dummy_cost, ignore_to_dummy_cost=ignore_to_dummy_cost)
        self.OD_mapping = self.generate_OD_mapping(self.O, self.D)
    
    def generate_problem(self, num_O: int = 10, num_taxi: int = 1, locations: dict = None, seed: int = 0,
                         ignore_from_dummy_cost: bool = True, ignore_to_dummy_cost: bool = True):
        """Method that generates a multi-OD required information.
        
        There will be a total of (1 + `num_O` + `num_taxi`) points. 1 is for a dummy start, 
        which has a distance of 0 to all other points and are always set to the first point.
        Then there will be `num_taxi` points represent the current locations of taxis.
        The locations will be generated uniformly from [0,1] by default. You could also provide the
        locations through `locations`. 
        The `distance_matrix` will be generated in this function. And the index of `O` and `D` will be 
        stored with the same order. For instance, `self.O[0]` and `self.D[0]` will be the first OD-pair.
        
        Parameters
        ------
        locations: dict, the keys should be 'O', 'D', 'taxi' and 'dummy'. The value should be a list or array of shape `(len, 2)`. 
            `O` and `D` must be provided. If `taxi` or `dummy` is not provided, the default value will be applied.
        
        Returns
        ------
        distance_matrix: numpy.array
        O: list
        D: list
        locations: list
        """
        np.random.seed(seed)
        random.seed(seed)
        if locations is not None:
            num_O = len(locations['O'])
            num_taxi = len(locations['taxi'])
            total_num_points = len(locations['O']) + len(locations['D'])
            total_num_points += len(locations['taxi']) if 'taxi' in locations else num_taxi
            total_num_points += 1  # dummy
            
            _locations = []
            _locations.append(locations['dummy'] if 'dummy' in locations else [0,0])
            _locations.append(locations['taxi'] if 'taxi' in locations else np.random.uniform(size=(num_taxi, 2)))
            _locations.append(locations['O'])
            _locations.append(locations['D'])
            _locations = np.vstack(_locations)
        else:
            total_num_points = num_O * 2 + num_taxi + 1  # 1 for the dummy starting point
            _locations = np.random.uniform(size=(total_num_points, 2))  # (x,y)
        locs = copy.deepcopy(_locations)
        node_index = [*range(len(locs))]
        # >=py38 feature
        O = node_index[(_O_start_index := 1 + num_taxi): (_D_start_index := _O_start_index + num_O)]
        D = node_index[_D_start_index: _D_start_index + num_O]
        edge_index = np.array([*zip(*product(node_index, node_index))])
        distance_matrix = np.linalg.norm(locs[edge_index[0]] - locs[edge_index[1]],
                                              ord=2, axis=1).reshape(len(node_index), len(node_index))
        if ignore_from_dummy_cost:
            distance_matrix[0, :] = 0.
        if ignore_to_dummy_cost:
            distance_matrix[:, 0] = 0.
        return distance_matrix, O, D, locs, node_index
    
    def generate_OD_mapping(self, O, D):
        """Gets OD mapping.
        
        Parameters
        ------
        O: list, given in order
        D: list, given in order

        Returns
        ------
        OD_mapping: dict, the keys are O, the values are D

        """
        OD_mapping = {O[i]: D[i] for i in range(len(O))}
        return OD_mapping
    
    def is_feasible(self, solution: solution.MultiODSolution):
        """This function checks whether a solution is feasible to the problem instance.
        
        Parameters
        ------
        solution: Solution, 

        Returns
        ------
        feasible: bool
        """
        if not isinstance(solution, MultiODSolution):
            solution = MultiODSolution(solution, self)
        feasible = solution._is_valid
        if not feasible: return False  
        for path in solution:
             for node in path:
                # constraint 1: the OD pair should appear in the same path
                if node in self.OD_mapping and self.OD_mapping[node] not in path:
                    solution.set_is_valid(False, self)  
                    return False   
                # constraint 2: O should appear before D
                if node in self.OD_mapping and path.indexof(node) >= path.indexof(self.OD_mapping[node]):
                    solution.set_is_valid(False, self)
                    return False 
        return True
    
    def generate_feasible_solution(self, *args):
        """Generates a random feasible solution(for subsequent optimization)"""
        ODs = random_split_dict(self.OD_mapping, self.num_taxi)
        taxi_ids = [*range(1, 1 + self.num_taxi)]
        random.shuffle(taxi_ids)
        paths = [[0, taxi_ids[i]] + generate_random_list_from_dict_with_key_before_value(OD, OD.keys()) + [0] for i, OD in enumerate(ODs)]
        sol = solution.MultiODSolution(paths, self)
        if self.is_feasible(sol):
            return sol 

    def calc_cost(self, solution: MultiODSolution):
        cost = 0.
        for path in solution.paths:
            cost += self.calc_cost_for_single_path(path)
        return cost 
    
    def calc_cost_for_single_path(self, path):
        cost = 0.
        if isinstance(path, solution.MultiODPath):
            p = [node for node in path]
            for idx in range(len(p) - 1):
                cost += path.get_distance_by_node_ids(p[idx], p[idx + 1])
        else:
            for idx in range(len(path) - 1):
                cost += self.distance_matrix[path[idx], path[idx + 1]]
        return cost
