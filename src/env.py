import numpy as np
import gymnasium as gym 

from . import problem
from . import actions
from . import operators
from . import utils 


MultiODProblem = problem.MultiODProblem


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
               'actions.PathRandomAction({idx}, operator=operators.RandomODPairsExchangeOperator(change_percentage=0.1))',
               'actions.PathRandomAction({idx}, operator=operators.RandomODPairsExchangeOperator(change_percentage=0.3))',
               'actions.PathRandomAction({idx}, operator=operators.RandomODPairsExchangeOperator(change_percentage=0.5))',
               'actions.PathRandomAction({idx}, operator=operators.RandomOForwardOperator(change_percentage=0.1))',
               'actions.PathRandomAction({idx}, operator=operators.RandomOForwardOperator(change_percentage=0.3))',
               'actions.PathRandomAction({idx}, operator=operators.RandomOForwardOperator(change_percentage=0.5))',
               'actions.PathRandomAction({idx}, operator=operators.RandomDBackwardOperator(change_percentage=0.1))',
               'actions.PathRandomAction({idx}, operator=operators.RandomDBackwardOperator(change_percentage=0.3))',
               'actions.PathRandomAction({idx}, operator=operators.RandomDBackwardOperator(change_percentage=0.5))'
               ]
    _action_dict = {idx: eval(_action.format(idx=idx)) for idx, _action in enumerate(_actions, start=1)}
    _action_dict[0] = env_instance._regenerate_feasible_solution
    return _action_dict


class MultiODEnv(gym.Env):
    def __init__(self, problem: MultiODProblem = None, action_dict: dict = None,
                 *, 
                 num_O: int = 10, num_taxi: int = 1, locations: dict = None, seed: int = 0, max_length: int = int(4e4)):
        super().__init__()
        self.problem = problem if problem is not None else MultiODProblem(num_O=num_O, num_taxi=num_taxi, locations=locations, seed=seed)
        self._action_dict = action_dict if action_dict is not None else get_default_action_dict(self)
        self._max_length = max_length
    
    def step(self, action: int):
        self._step += 1
        self.solution, all_delta = self.action_dict[action](self)
        next_obs = self.generate_state(self.solution)
        reward, done = self._calc_reward(all_delta), self._calc_done(self._step)
        self._update_history_buffer(action, all_delta)
        infos = self._calc_infos(all_delta, self._history_actions[-self._k_recent:], self._history_delta_sign[-self._k_recent:])
        if infos['cost'] < self.best_cost:
            self._update_best_solution(self.solution, infos, self._step)
        return next_obs, reward, done, infos
        
    def reset(self):
        self._step = 0
        self.solution = self.problem.generate_feasible_solution()
        self._update_history_buffer()
        obs, infos = self.generate_state(self.solution), self._calc_infos()
        self._update_best_solution(self.solution, infos, self._step)
        return obs, infos

    def render(self, mode='human', *, figsize: tuple = (8, 6), dpi: float = 80, fig_name: str = None, to_annotate: bool = True, quiver_width: float = 5e-3):
        if mode == 'human':
            fig = utils.display_result(self.problem, self.solution, figsize=figsize, dpi=dpi, fig_name=fig_name, to_annotate=to_annotate, quiver_width=quiver_width)
            return fig 
    
    def _calc_infos(self, delta: float = 0., k_recent_action=None, k_recent_delta_sign=None):
        infos = {}
        infos['delta'] = delta 
        infos['cost'] = self.problem.calc_cost(self.solution)
        infos['history_actions'], infos['history_delta_sign'] = k_recent_action, k_recent_delta_sign
        return infos 
    
    def _calc_reward(self, all_delta):
        return all_delta
    
    def _calc_done(self, step):
        return step >= self._max_length
    
    def _regenerate_feasible_solution(self, *args):
        old_cost = self.problem.calc_cost(self.solution)
        self.solution = self.problem.generate_feasible_solution()
        new_cost = self.problem.calc_cost(self.solution)
        delta = new_cost - old_cost
        return self.solution, delta
    
    def _update_best_solution(self, solution, infos, step):
        self.best_solution = self.get_np_repr_of_solution(solution)
        self.best_cost = infos['cost']
        self.best_sol_at_step = step
    
    def _update_history_buffer(self, action: int = 0, delta: float = 0.):
        pass 
    
    def generate_state(self, solution):
        return self.get_np_repr_of_solution(solution)
    
    def get_np_repr_of_solution(self, solution):
        return np.asarray([[*iter(path)] for path in solution.paths]) 

    @property
    def action_dict(self):
        return self._action_dict
    
    @action_dict.setter
    def set_action_dict(self, value):
        self._action_dict = value 
