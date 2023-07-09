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
                 num_O: int = 10, num_taxi: int = 1, locations: dict = None, seed: int = 0):
        super().__init__()
        self.problem = problem if problem is not None else MultiODProblem(num_O=num_O, num_taxi=num_taxi, locations=locations, seed=seed)
        self._action_dict = action_dict if action_dict is not None else get_default_action_dict(self)
    
    def step(self, action: int):
        self.solution, all_delta = self.action_dict[action](self)
        next_obs, infos = self.solution, self._calc_infos(all_delta)
        reward, done = self._calc_reward(), self._calc_done()
        return next_obs, reward, done, infos
        
    def reset(self):
        self.solution = self.problem.generate_feasible_solution()
        obs, infos = self.solution, self._calc_infos()
        return obs, infos

    def render(self, mode='human', *, figsize: tuple = (8, 6), dpi: float = 80, fig_name: str = None, to_annotate: bool = True, quiver_width: float = 5e-3):
        if mode == 'human':
            fig = utils.display_result(self.problem, self.solution, figsize=figsize, dpi=dpi, fig_name=fig_name, to_annotate=to_annotate, quiver_width=quiver_width)
            return fig 
    
    def _calc_infos(self, delta: float = 0.):
        return {'delta': delta} 
    
    def _calc_reward(self):
        return 0.
    
    def _calc_done(self):
        return False 
    
    def _regenerate_feasible_solution(self, *args):
        old_cost = self.problem.calc_cost(self.solution)
        self.solution = self.problem.generate_feasible_solution()
        new_cost = self.problem.calc_cost(self.solution)
        delta = new_cost - old_cost
        return self.solution, delta
    
    def generate_state(self):
        return None 
    
    @property
    def action_dict(self):
        return self._action_dict
    
    @action_dict.setter
    def set_action_dict(self, value):
        self._action_dict = value 
