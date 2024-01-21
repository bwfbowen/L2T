from typing import Dict, List, Callable, NamedTuple
from abc import abstractmethod

import time
import copy 
import random 
import numpy as np
import gymnasium as gym 

import dm_env
from acme import specs 

from . import problem as od_problem
from . import solution
from . import utils 
from . import types 
from . import action_dicts as ads

EPSILON = 1e-5
MultiODProblem = od_problem.MultiODProblem


EPSILON = 1e-5
MultiODProblem = od_problem.MultiODProblem
PDP = od_problem.PDP
MultiODSolution = solution.MultiODSolution
SliceableDeque = utils.SliceableDeque


class MultiODEnv(gym.Env):
    """RL Environment for Multi-OD problems.
    
    The main class for RL agents to interact with Multi-OD problems.

    Parameters
    ------
    problem: problem.MultiODProblem, an instance of `problem.MultiODProblem`
    action_dict: callable, specified actions to use
    
    """
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
        self._action_dict = action_dict(self) if action_dict is not None else ads.get_default_action_dict(self)
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
        self._random_actions = random_actions if random_actions is not None else ads.get_default_random_actions()
    
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
                 k_recent: int = 1,
                 max_no_improvement: int = 6,
                 best_cost_tolerance: float = 0.01,
                 random_actions: list = None 
                 ):
        super().__init__(problem=problem, action_dict=action_dict, num_O=num_O, num_taxi=num_taxi, locations=locations, seed=seed, max_length=max_length, max_time_length=max_time_length, k_recent=k_recent, max_no_improvement=max_no_improvement, best_cost_tolerance=best_cost_tolerance, random_actions=random_actions)
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
        self._no_improvement = 0
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
        if self._no_improvement >= self._max_no_improvement:
            action = 0
            self._no_improvement = 0
        self.solution, all_delta = self.action_dict[action](self)
        # no improvement:
        if action != 0 and all_delta >= -EPSILON:
            self._no_improvement += 1
        else:
            self._no_improvement = 0
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
    

class DMultiODEnv(dm_env.Environment):
    def __init__(
        self, 
        problem: od_problem.MultiODProblemV2 = None,
        candidate_actions: List[Callable] = None,
        random_actions: List[Callable] = None,
        initial_strategy: str = '1by1',
        k_recent_action: int = 10,
        max_no_improvement: int = 6,
        max_steps: int = int(4e4),
        num_action_iters: int = 10,
        change_pct: float = 0.1,
        best_cost_tolerance: float = 0.01,
        *,
        num_nodes: int = 21,
        locations: Dict = None,
        seed: int = 0,
        ignore_from_dummy_cost: bool = True,
        ignore_to_dummy_cost: bool = True,
        int_distance: bool = False
    ):
        super().__init__()
        self._problem = problem if problem is not None else od_problem.MultiODProblemV2(num_nodes, locations, seed, ignore_from_dummy_cost, ignore_to_dummy_cost, int_distance)
        self._actions = candidate_actions or ads.get_default_candidate_actions_v2(num_action_iters)
        self._random_actions = random_actions or ads.get_default_random_actions_v2(change_pct)
        self.num_actions = len(self._actions)
        self._k_recent_action = k_recent_action
        self._max_no_improvement = max_no_improvement
        self._best_cost_tolerance = best_cost_tolerance

        self._reset_next_step = True
        self._initial_startegy = initial_strategy
        
        self._env_state = types.EnvState(
            problem=self._problem.info,
            action_history=np.zeros((self._k_recent_action, 2), dtype=np.float32))
        self._best_sol = types.BestSol()
        self._current_cost = 0.
        self._max_steps = max_steps
        self._current_step = 0

        self.observation_space = self.observation_spec()
        self._obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        self._raw_node_obs = np.zeros((len(self._problem.info.locations) - 1, 12), dtype=np.float32)
    
    def _init_solution(self):
        if self._initial_startegy == '1by1':
            info = self._problem.info 
            paths = [[0] + [node for item in info.od_pairing.items() if info.od_type[item[0]] == 0 for node in item] + [0]]
            sol = solution.MultiODSolutionV2(paths, info)
            self._current_cost = self._problem.calc_cost(sol)
            self._init_cost = self._current_cost
            self._best_sol = types.BestSol(
                paths=copy.deepcopy(sol.paths),
                best_cost=self._current_cost,
                best_step=0)
            self._env_state = types.EnvState(
                problem=sol.info,
                action_history=np.zeros((self._k_recent_action, 2), dtype=np.float32),
                improvement=types.Improvement())
            return sol 
        
    def _observe(self, sol: solution.MultiODSolutionV2):
        info = sol.info
        paths = sol.paths
        for i in range(1, len(info.locations)):
            seq_i = info.sequence[i]
            path = paths[seq_i.sequence]
            prev_, next_ = path[seq_i.index - 1], path[seq_i.index + 1]
            _non_i = i - 1
            self._raw_node_obs[_non_i, :6] = info.locations[[prev_, i, next_]].flatten()
            self._raw_node_obs[_non_i, 6:9] = info.distance_matrix[[prev_, i, next_], [i, next_, prev_]].flatten()
            self._raw_node_obs[_non_i, 9:12] = [info.od_type[prev_], info.od_type[i], info.od_type[next_]]
        num_non_depot_nodes = len(info.locations) - 1
        node_dim = num_non_depot_nodes * 12
        action_history_dim = node_dim + 2 * self._k_recent_action
        improvement_dim = action_history_dim + 3
        self._obs[:node_dim] = self._raw_node_obs.flatten()
        self._obs[node_dim: action_history_dim] = self._env_state.action_history.flatten()
        self._obs[action_history_dim: improvement_dim] = [
            self._env_state.improvement.delta, 
            self._env_state.improvement.delta_best,
            self._env_state.improvement.no_improvement]
        return self._obs

    def _regenerate_feasible_solution_with_random_actions(self):
        old_cost = self._current_cost
        if not old_cost / self._best_sol.best_cost < 1 + self._best_cost_tolerance:
            _best_solution = solution.MultiODSolutionV2(
                paths=copy.deepcopy(self._best_sol.paths),
                problem_info=self._problem.info)
            self.solution = _best_solution
        # random disturb
        ra = random.choice(self._random_actions)    
        self.solution, _ = ra(self)
        return self.solution
    
    def reset(self):
        self._reset_next_step = False 
        self.solution = self._init_solution()
        self._current_step = 0
        observation = self._observe(self.solution)
        return dm_env.restart(observation)
    
    def step(self, action):
        if self._reset_next_step:
            return self.reset()
        no_improvement = self._env_state.improvement.no_improvement
        if no_improvement >= self._max_no_improvement:
            self.solution = self._regenerate_feasible_solution_with_random_actions()
            no_improvement = 0
        self.solution, delta = self._actions[action](self)
        # test feasibility
        if not self._problem.is_feasible(self.solution):
            raise ValueError(f'infeasible solution from action {action}: {self._actions[action]}.')
        self._current_step += 1
        self._current_cost = self._problem.calc_cost(self.solution)
        best_cost = self._best_sol.best_cost
        delta_best = self._current_cost - best_cost
        action_history = self._env_state.action_history
        action_history[:-1] = action_history[1:]
        action_history[-1, :] = [action, np.sign(delta)]
        # decision 1: at each step
        # reward = -delta 
        # decision 2: sparse at terminal step
        if self._current_cost < best_cost:
            self._best_sol = types.BestSol(
                paths=self.solution.paths,
                best_cost=self._current_cost,
                best_step=self._current_step)
        done = self._current_step >= self._max_steps
        
        if delta >= -EPSILON:
            no_improvement += 1
        else:
            no_improvement = 0
        
        # update state
        self._env_state = types.EnvState(
            problem=self.solution.info,
            action_history=action_history,
            improvement=types.Improvement(
                delta=delta, 
                delta_best=delta_best,
                no_improvement=no_improvement))
        observation = self._observe(self.solution)
        if done:
            self._reset_next_step = True 
            # normalize improvement
            reward = (self._init_cost - self._best_sol.best_cost) / self._init_cost
            # faster the better
            reward *= 1 + np.log10(self._max_steps / self._best_sol.best_step) 
            return dm_env.termination(
                reward=reward, 
                observation=observation)
        else:
            return dm_env.transition(reward=0., observation=observation)

    def observation_spec(self):
        num_non_depot_nodes = len(self._problem.info.locations) - 1
        total_dim = num_non_depot_nodes * 12 + 2 * self._k_recent_action + 3
        minimum = (
            [0] * num_non_depot_nodes * 9 
            + [-1] * num_non_depot_nodes * 3 
            + [0] * self._k_recent_action 
            + [-1] * self._k_recent_action 
            + [-np.inf] * 2
            + [0])
        maximum = (
            [np.inf] * num_non_depot_nodes * 9
            + [1] * num_non_depot_nodes * 3
            + [self.num_actions] * self._k_recent_action
            + [1] * self._k_recent_action
            + [np.inf] * 2
            + [self._max_no_improvement])
        observation_space = specs.BoundedArray(
            (total_dim,), 
            dtype='float32', 
            minimum=minimum, 
            maximum=maximum)
        return observation_space 

    def action_spec(self):
        return specs.DiscreteArray(num_values=self.num_actions, dtype='int32', name='action')
    
    def reward_spec(self):
        return specs.Array((), dtype='float32')
    
    def discount_spec(self):
        return specs.Array((), dtype='float32')
        


class PDPEnv(MultiODEnv):
    def __init__(
        self, 
        problem: MultiODProblem = None, 
        action_dict: callable = None, 
        *, 
        num_O: int = 10, 
        num_taxi: int = 1, 
        locations: dict = None, 
        capacity: int = None,
        capacities: dict = None,
        capacity_slack: float = .2,
        distance_type: str = 'EXACT_2D',
        ignore_from_dummy_cost: bool = True, 
        ignore_to_dummy_cost: bool = True,
        seed: int = 0, 
        max_length: int = int(40000), 
        max_time_length: int = int(1000), 
        k_recent: int = 1, 
        max_no_improvement: int = 6, 
        best_cost_tolerance: float = 0.01, 
        random_actions: list = None
    ):
        super().__init__(problem, 
                         action_dict, 
                         num_O=num_O, 
                         num_taxi=num_taxi, 
                         locations=locations, 
                         seed=seed, 
                         max_length=max_length, 
                         max_time_length=max_time_length, 
                         k_recent=k_recent, 
                         max_no_improvement=max_no_improvement, 
                         best_cost_tolerance=best_cost_tolerance, 
                         random_actions=random_actions)
        self.problem = problem if problem is not None else PDP(num_O=num_O, num_taxi=num_taxi, locations=locations, capacity=capacity, capacities=capacities, capacity_slack=capacity_slack, distance_type=distance_type, ignore_from_dummy_cost=ignore_from_dummy_cost, ignore_to_dummy_cost=ignore_to_dummy_cost, seed=seed)
        self._action_dict = action_dict(self) if action_dict is not None else ads.get_pdp_default_action_dict(self)
        self.action_space = gym.spaces.Discrete(len(self._action_dict))
        self.observation_space = gym.spaces.Dict(
            {'problem': gym.spaces.Box(low=np.array([-np.inf, -np.inf, 0] + [0] * k_recent + [-1] * k_recent), 
                                       high=np.array([np.inf, np.inf, 1] + [len(self._action_dict) - 1] * k_recent + [1] * k_recent),
                                       shape=(3 + k_recent * 2,), 
                                       dtype=np.float32),
            'solution': gym.spaces.Box(low=np.ones((self.problem.num_O * 2, 15)) * np.array([0] * 6 + [0] * 3 + [0] * 3 + [0] * 3), 
                                       high=np.ones((self.problem.num_O * 2, 15)) * np.array([1e3] * 6 + [1e4] * 3 + [1] * 3 + [1e2] * 3), 
                                       shape=(self.problem.num_O * 2, 15),
                                       dtype=np.float32)})
        self.distance_rescale_factor = 1 / 1000 if distance_type == 'EXACT_2D' else 1
        self._random_actions = random_actions if random_actions is not None else ads.get_pdp_default_random_actions()
    
    def calc_features_of_solution(self, solution):
        features = np.zeros((self.problem.num_O * 2, 15), dtype=np.float32)
        for path_id, path in enumerate(solution):
            n = len(path) - 1
            cumcap = np.sum(path.capacities)
            remain = path.capacity - cumcap 
            for i in range(1, n):
                node = path.get_by_seq_id(i)
                _prev = node.prev_node if node.prev_node is not None else 0
                _next = node.next_node if node.next_node is not None else 0
                if _prev != 0:
                    _prev_id, _prev_OD = _prev.node_id, _prev.OD_type 
                else:
                    _prev_id, _prev_OD = 0, 2
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
                f2 *= self.distance_rescale_factor
                # prev, node, next OD type, shape: (3,)
                f3 = np.array([_prev_OD, node.OD_type, _next_OD])
                # node capacity, path remaining capacity and path index, shape: (3,)
                f4 = np.array([self.problem.capacities[node.node_id], remain, path_id])
                f4 /= self.problem.capacity
                features[node.node_id - 1, :6] = f1
                features[node.node_id - 1, 6:9] = f2
                features[node.node_id - 1, 9:12] = f3 
                features[node.node_id - 1, 12:15] = f4 
        return features
    
    def _calc_reward(self, all_delta):
        return -all_delta * self.distance_rescale_factor
    
    def _calc_infos(self, delta: float = 0., k_recent_action=None, k_recent_delta_sign=None):
        infos = {}
        infos['delta'] = delta * self.distance_rescale_factor 
        infos['cost'] = self.problem.calc_cost(self.solution) * self.distance_rescale_factor
        infos['no_improvement'] = self._no_improvement / self._max_no_improvement
        infos['delta_best'] = (infos['cost'] - self.best_cost) * self.distance_rescale_factor
        infos['k_recent_action'], infos['k_recent_delta_sign'] = k_recent_action, k_recent_delta_sign
        return infos 
