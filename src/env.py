import gymnasium as gym 

from . import problem
from . import actions
from . import operators
from . import utils 


MultiODProblem = problem.MultiODProblem


class MultiODEnv(gym.Env):
    def __init__(self, problem: MultiODProblem = None, action_dict: dict = None,
                 *, 
                 num_O: int = 10, num_taxi: int = 1, locations: dict = None, seed: int = 0):
        super().__init__()
        self.problem = problem if problem is not None else MultiODProblem(num_O=num_O, num_taxi=num_taxi, locations=locations, seed=seed)
        self.action_dict = action_dict if action_dict is not None else self._default_action_dict
    
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
    def _default_action_dict(self):
        _action_dict = {
            0: self._regenerate_feasible_solution,
            1: actions.InBlockAction(1, operator=operators.TwoOptOperator()),
            2: actions.PathAction(2, operator=operators.ExchangeOperator()),
            3: actions.PathAction(3, operator=operators.InsertOperator()),
            4: actions.PathAction(4, operator=operators.OForwardOperator(length=2)),
            5: actions.PathAction(5, operator=operators.OForwardOperator(length=3)),
            6: actions.PathAction(6, operator=operators.OForwardOperator(length=4)),
            7: actions.PathAction(7, operator=operators.DBackwardOperator(length=2)),
            8: actions.PathAction(8, operator=operators.DBackwardOperator(length=3)),
            9: actions.PathAction(9, operator=operators.DBackwardOperator(length=4)),
            10: actions.PathAction(10, operator=operators.ODPairsExchangeOperator()),
            11: actions.PathRandomAction(11, operator=operators.RandomODPairsExchangeOperator(change_percentage=0.1)),
            12: actions.PathRandomAction(12, operator=operators.RandomODPairsExchangeOperator(change_percentage=0.3)),
            13: actions.PathRandomAction(13, operator=operators.RandomODPairsExchangeOperator(change_percentage=0.5)),
            14: actions.PathRandomAction(14, operator=operators.RandomOForwardOperator(change_percentage=0.1)),
            15: actions.PathRandomAction(15, operator=operators.RandomOForwardOperator(change_percentage=0.3)),
            16: actions.PathRandomAction(16, operator=operators.RandomOForwardOperator(change_percentage=0.5)),
            17: actions.PathRandomAction(17, operator=operators.RandomDBackwardOperator(change_percentage=0.1)),
            18: actions.PathRandomAction(18, operator=operators.RandomDBackwardOperator(change_percentage=0.3)),
            19: actions.PathRandomAction(19, operator=operators.RandomDBackwardOperator(change_percentage=0.5)),
            }
        return _action_dict
