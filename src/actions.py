import copy 

class Action:
    def __init__(self, index, action_type, operator):
        self.index = index
        self.action_type = action_type 
        self.operator = operator
    
    def __call__(self, env):
        improved_solution = copy.deepcopy(env.solution) 
        all_delta = 0.
        return improved_solution, all_delta