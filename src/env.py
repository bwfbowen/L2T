import gymnasium as gym 

from . import problem


MultiODProblem = problem.MultiODProblem


class MultiODEnv(gym.Env):
    def __init__(self, problem: MultiODProblem = None, action_dict: dict = None,
                 *, 
                 num_O: int = 10, num_taxi: int = 1, locations: dict = None, seed: int = 0):
        super().__init__()
        self.problem = problem if problem is not None else MultiODProblem(num_O=num_O, num_taxi=num_taxi, locations=locations, seed=seed)
        self.action_dict = action_dict if action_dict is not None else self._default_action_dict
    
    def step(self, action):
        pass 

    def reset(self):
        pass 
    
    @property
    def _default_action_dict(self):
        return {}
