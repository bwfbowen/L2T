from abc import abstractmethod

import time
import numpy as np
import gymnasium as gym 

from . import problem
from . import solution
from . import actions
from . import operators
from . import utils 


EPSILON = 1e-5
MultiODProblem = problem.MultiODProblem
MultiODSolution = solution.MultiODSolution
SliceableDeque = utils.SliceableDeque


def get_default_action_dict(env_instance):
    _actions = [ 
               'actions.InBlockAction({idx}, operator=operators.TwoOptOperator())',
               'actions.PathAction({idx}, operator=operators.SegmentTwoOptOperator())',
               'actions.PathAction({idx}, operator=operators.ExchangeOperator())',
               'actions.PathAction({idx}, operator=operators.InsertOperator())',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=2))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=3))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=4))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=2))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=3))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=4))',
               'actions.PathAction({idx}, operator=operators.ODPairsExchangeOperator())',
            #    'actions.PathRandomAction({idx}, operator=operators.RandomODPairsExchangeOperator(change_percentage=0.1))',
            #    'actions.PathRandomAction({idx}, operator=operators.RandomODPairsExchangeOperator(change_percentage=0.3))',
            #    'actions.PathRandomAction({idx}, operator=operators.RandomODPairsExchangeOperator(change_percentage=0.5))',
            #    'actions.PathRandomAction({idx}, operator=operators.RandomOForwardOperator(change_percentage=0.1))',
            #    'actions.PathRandomAction({idx}, operator=operators.RandomOForwardOperator(change_percentage=0.3))',
            #    'actions.PathRandomAction({idx}, operator=operators.RandomOForwardOperator(change_percentage=0.5))',
            #    'actions.PathRandomAction({idx}, operator=operators.RandomDBackwardOperator(change_percentage=0.1))',
            #    'actions.PathRandomAction({idx}, operator=operators.RandomDBackwardOperator(change_percentage=0.3))',
            #    'actions.PathRandomAction({idx}, operator=operators.RandomDBackwardOperator(change_percentage=0.5))'
               ]
    _action_dict = {idx: eval(_action.format(idx=idx)) for idx, _action in enumerate(_actions, start=1)}
    _action_dict[0] = env_instance._regenerate_feasible_solution_with_random_actions
    return _action_dict


def get_naive_action_dict(env_instance):
    _actions = [ 
               'actions.PathAction({idx}, operator=operators.ExchangeOperator())',
               'actions.PathAction({idx}, operator=operators.InsertOperator())',
               ]
    _action_dict = {idx: eval(_action.format(idx=idx)) for idx, _action in enumerate(_actions, start=1)}
    _action_dict[0] = env_instance._regenerate_feasible_solution
    return _action_dict


def get_default_random_actions():
       _random_actions = ['actions.PathRandomAction({idx}, operator=operators.RandomODPairsExchangeOperator(change_percentage=0.1))',
                          'actions.PathRandomAction({idx}, operator=operators.RandomOForwardOperator(change_percentage=0.1))',
                          'actions.PathRandomAction({idx}, operator=operators.RandomDBackwardOperator(change_percentage=0.1))']
       _random_actions = [eval(a.format(idx=idx)) for idx, a in enumerate(_random_actions)]
       return _random_actions


class MultiODEnv(gym.Env):
    def __init__(self, problem: MultiODProblem = None, action_dict: callable = None,
                 *, 
                 num_O: int = 10, 
                 num_taxi: int = 1, 
                 locations: dict = None, 
                 seed: int = 0, 
                 max_length: int = int(4e4),
                 max_time_length: int = int(1e3),
                 k_recent: int = 1,
                 max_no_improvement: int = 6,
                 best_cost_tolerance: float = 0.01,
                 random_actions: list = None 
                 ):
        super().__init__()
        self.problem = problem if problem is not None else MultiODProblem(num_O=num_O, num_taxi=num_taxi, locations=locations, seed=seed)
        self._action_dict = action_dict(self) if action_dict is not None else get_default_action_dict(self)
        self.action_space = gym.spaces.Discrete(len(self._action_dict))
        self.observation_space = gym.spaces.Dict(
            {'problem': gym.spaces.Box(low=np.array([-np.inf, -np.inf, 0] + [0] * k_recent + [-1] * k_recent), 
                                       high=np.array([np.inf, np.inf, 1] + [len(self._action_dict) - 1] * k_recent + [1] * k_recent),
                                       shape=(3 + k_recent * 2,), 
                                       dtype=np.float32),
            'solution': gym.spaces.Box(low=np.ones((self.problem.num_O * 2, 12)) * np.array([0] * 6 + [0] * 3 + [0] * 3), 
                                       high=np.ones((self.problem.num_O * 2, 12)) * np.array([1e3] * 6 + [1e4] * 3 + [1] * 3), 
                                       shape=(self.problem.num_O * 2, 12),
                                       dtype=np.float32)})
        self._max_length = max_length
        self._max_time_length = max_time_length
        self._k_recent = k_recent
        self._max_no_improvement = max_no_improvement
        self._best_cost_tolerance = best_cost_tolerance
        self._random_actions = random_actions if random_actions is not None else get_default_random_actions()
    
    def step(self, action: int):
        # print(action)
        self._step += 1
        if self._no_improvement >= self._max_no_improvement:
            action = 0
            self._no_improvement = 0
        self.solution, all_delta = self.action_dict[action](self)
        # no improvement:
        if action != 0 and all_delta >= -EPSILON:
            self._no_improvement += 1
        else:
            self._no_improvement = 0
        next_obs = {}
        next_obs['solution'] = self.generate_state(self.solution)
        reward, done = self._calc_reward(all_delta), self._calc_done(self._step)
        self._update_history_buffer(action, all_delta)
        infos = self._calc_infos(all_delta, self._history_action_buffer, self._history_delta_sign)
        if infos['cost'] < self.best_cost:
            self._update_best_solution(self.solution, infos, self._step)
        next_obs['problem'] = self.calc_features_of_problem(infos)
        return next_obs, reward, done, done, infos
        
    def reset(self, seed=None, options=None):
        self.start_time = time.time()
        self._step = 0
        self._no_improvement = 0
        self.solution = self.problem.generate_feasible_solution()
        self._reset_history_buffer()
        self._reset_best_solution(self.solution)
        obs = {}
        obs['solution'], infos = self.generate_state(self.solution), self._calc_infos(k_recent_action=self._history_action_buffer, k_recent_delta_sign=self._history_delta_sign)
        obs['problem'] = self.calc_features_of_problem(infos)
        return obs, infos

    def render(self, mode='human', *, figsize: tuple = (8, 6), dpi: float = 80, fig_name: str = None, to_annotate: bool = True, quiver_width: float = 5e-3):
        if mode == 'human':
            fig = utils.display_result(self.problem, self.solution, figsize=figsize, dpi=dpi, fig_name=fig_name, to_annotate=to_annotate, quiver_width=quiver_width)
            return fig 
    
    def _calc_infos(self, delta: float = 0., k_recent_action=None, k_recent_delta_sign=None):
        infos = {}
        infos['delta'] = delta 
        infos['cost'] = self.problem.calc_cost(self.solution)
        infos['no_improvement'] = self._no_improvement / self._max_no_improvement
        infos['delta_best'] = infos['cost'] - self.best_cost
        infos['k_recent_action'], infos['k_recent_delta_sign'] = k_recent_action, k_recent_delta_sign
        return infos 
    
    def _calc_reward(self, all_delta):
        return -all_delta
    
    def _calc_done(self, step):
        return step >= self._max_length or time.time() - self.start_time >= self._max_time_length
    
    def _regenerate_feasible_solution_with_random_actions(self, *args):
        old_cost = self.problem.calc_cost(self.solution)
        if not old_cost / self.best_cost < 1 + self._best_cost_tolerance:
            _best_solution = MultiODSolution(self.best_solution, self.problem)
            self.solution = _best_solution
        # random disturb
        for ra in self._random_actions:
            self.solution, _ = ra(self)
        new_cost = self.problem.calc_cost(self.solution)
        delta = new_cost - old_cost
        return self.solution, delta
    
    def _regenerate_feasible_solution(self, *args):
        old_cost = self.problem.calc_cost(self.solution)
        self.solution = self.problem.generate_feasible_solution()
        new_cost = self.problem.calc_cost(self.solution)
        delta = new_cost - old_cost
        return self.solution, delta 
    
    def _reset_history_buffer(self):
        self._history_action_buffer = SliceableDeque([0 for _ in range(self._k_recent)], maxlen=self._k_recent)
        self._history_delta_sign = SliceableDeque([0. for _ in range(self._k_recent)], maxlen=self._k_recent)
    
    def _reset_best_solution(self, solution):
        self.best_solution = self.get_repr_of_solution(solution)
        self.best_cost = self.problem.calc_cost(solution)
        self.best_sol_at_step = 0

    def _update_best_solution(self, solution, infos, step):
        self.best_solution = self.get_repr_of_solution(solution)
        self.best_cost = infos['cost']
        self.best_sol_at_step = step
    
    def _update_history_buffer(self, action: int = 0, delta: float = 0.):
        self._history_action_buffer.append(action)
        self._history_delta_sign.append(np.sign(delta))
    
    def generate_state(self, solution):
        return self.calc_features_of_solution(solution)
    
    def get_repr_of_solution(self, solution):
        return [[*iter(path)] for path in solution.paths]
    
    def calc_features_of_solution(self, solution):
        features = np.zeros((self.problem.num_O * 2, 12), dtype=np.float32)
        for path in solution:
            n = len(path) - 1
            for i in range(2, n):
                node = path.get_by_seq_id(i)
                _prev = node.prev_node
                _prev_id, _prev_OD = _prev.node_id, _prev.OD_type if _prev.OD_type is not None else 2
                _next = node.next_node if node.next_node is not None else 0
                if _next != 0:
                    _next_id, _next_OD = _next.node_id, _next.OD_type 
                else:
                    _next_id, _next_OD = 0, 2
                # prev, node, next location, shape: (2*3,)
                f1 = path.locations[[_prev_id, node.node_id, _next_id]].flatten()
                # prev->node, node->next, prev->next distance, shape: (3,)
                f2 = np.array([path.get_distance_by_node_ids(_prev_id, node.node_id), 
                            path.get_distance_by_node_ids(node.node_id, _next_id),
                            path.get_distance_by_node_ids(_prev_id, _next_id)])
                # prev, node, next OD type, shape: (3,)
                f3 = np.array([_prev_OD, node.OD_type, _next_OD])
                features[node.node_id - 1 - self.problem.num_taxi, :6] = f1
                features[node.node_id - 1 - self.problem.num_taxi, 6:9] = f2
                features[node.node_id - 1 - self.problem.num_taxi, 9:12] = f3 
        return features
    
    def calc_features_of_problem(self, infos):
        return np.array([infos['delta'], infos['delta_best'], infos['no_improvement'], *infos['k_recent_action'], *infos['k_recent_delta_sign']], dtype=np.float32)

    @property
    def action_dict(self):
        return self._action_dict
    
    @action_dict.setter
    def set_action_dict(self, value):
        self._action_dict = value 


class SparseMultiODEnv(MultiODEnv):
    def __init__(self, 
                 target_cost: int,
                 problem: MultiODProblem = None, 
                 action_dict: dict = None,
                 *, 
                 num_O: int = 10, 
                 num_taxi: int = 1, 
                 locations: dict = None, 
                 seed: int = 0, 
                 max_length: int = int(4e4),
                 max_time_length: int = int(1e3),
                 k_recent: int = 1
                 ):
        super().__init__(problem=problem, action_dict=action_dict, num_O=num_O, num_taxi=num_taxi, locations=locations, seed=seed, max_length=max_length, max_time_length=max_time_length, k_recent=k_recent)
        self.target_cost = target_cost
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(low=0., high=0.),
            'achieved_goal': gym.spaces.Box(low=0., high=np.inf),
            'desired_goal': gym.spaces.Box(low=0., high=np.inf),
            'problem': gym.spaces.Box(low=np.array([-np.inf, -np.inf] + [0] * k_recent + [-1] * k_recent), 
                                          high=np.array([np.inf, np.inf] + [len(self._action_dict) - 1] * k_recent + [1] * k_recent),
                                          shape=(2 + k_recent * 2,), 
                                          dtype=np.float32),
            'solution': gym.spaces.Box(low=np.ones((self.problem.num_O * 2, 12)) * np.array([0] * 6 + [0] * 3 + [0] * 3), 
                                       high=np.ones((self.problem.num_O * 2, 12)) * np.array([1e3] * 6 + [1e4] * 3 + [1] * 3), 
                                       shape=(self.problem.num_O * 2, 12),
                                       dtype=np.float32)
            })

    def reset(self, seed=None, options=None):
        self.start_time = time.time()
        self._step = 0
        self.solution = self.problem.generate_feasible_solution()
        self._reset_history_buffer()
        self._reset_best_solution(self.solution)
        obs = {'observation': 0., 'achieved_goal': 0., 'desired_goal': self.target_cost}
        obs['solution'], infos = self.generate_state(self.solution), self._calc_infos(k_recent_action=self._history_action_buffer, k_recent_delta_sign=self._history_delta_sign)
        obs['problem'] = np.array([infos['delta'], infos['delta_best'], *infos['k_recent_action'], *infos['k_recent_delta_sign']])
        obs['achieved_goal'] = int(infos['cost'])
        return obs, infos
    
    def step(self, action: int):
        
        self._step += 1
        self.solution, all_delta = self.action_dict[action](self)
        next_obs = {'observation': 0., 'achieved_goal': 0., 'desired_goal': self.target_cost}
        next_obs['solution'] = self.generate_state(self.solution)
        done = self._calc_done(self._step)
        self._update_history_buffer(action, all_delta)
        infos = self._calc_infos(all_delta, self._history_action_buffer, self._history_delta_sign)
        if infos['cost'] < self.best_cost:
            self._update_best_solution(self.solution, infos, self._step)
        next_obs['problem'] = np.array([infos['delta'], infos['delta_best'], *infos['k_recent_action'], *infos['k_recent_delta_sign']], dtype=np.float32)
        next_obs['achieved_goal'] = int(infos['cost'])
        reward = self._calc_reward(achieved_goal=next_obs['achieved_goal'], desired_goal=next_obs['desired_goal'], info=infos)
        return next_obs, reward, done, done, infos

    def _calc_reward(self, achieved_goal, desired_goal, info):
        reward = (np.asarray(achieved_goal) <= np.asarray(desired_goal)).astype(int)
        return reward
    
    @abstractmethod
    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self._calc_reward(achieved_goal=achieved_goal, desired_goal=desired_goal, info=info)
        return reward 