import copy
import gymnasium as gym 

from . import operators
from . import solution

Operator = operators.Operator
MultiODSolution = solution.MultiODSolution


class Action:
    def __init__(self, action_index: int, operator: Operator, action_type: str):
        self.action_index = action_index
        self.action_type = action_type 
        self.operator = operator
    
    def __call__(self, env):
        improved_solution = copy.deepcopy(env.solution) 
        all_delta = 0.
        return improved_solution, all_delta
    
    def __repr__(self) -> str:
        return f'{type(self).__name__}({type(self.operator).__name__})'


class InBlockAction(Action):
    def __init__(self, action_index: int, operator: Operator):
        super().__init__(action_index=action_index, action_type='in-block', operator=operator)
    
    def __call__(self, env: gym.Env):
        improved_solution = env.solution
        all_delta = 0.
        for path_id, path in enumerate(improved_solution.paths):
            blocks = path.O_blocks + path.D_blocks
            for block_id in blocks:
                modified = True 
                while modified:
                    improved_path, delta, modified = self._update(improved_solution, block_id, path_id)
                    if modified:
                        all_delta += delta
            
        return improved_solution, all_delta
    
    def _update(self, solution: MultiODSolution, block_id: int, path_id: int = 0):
        improved_path, delta, label = self.operator(solution=solution, block_id=block_id, path_id=path_id)
        modified = True if label else False
        return improved_path, delta, modified
    

class PathAction(Action):
    def __init__(self, action_index: int, operator: Operator):
        super().__init__(action_index=action_index, action_type='path', operator=operator)

    def __call__(self, env: gym.Env):
        improved_solution = env.solution
        all_delta = 0.
        for path_id, path in enumerate(improved_solution.paths):
            modified = True 
            while modified:
                improved_path, delta, modified = self._update(improved_solution, path_id)
                
                if modified:
                    all_delta += delta 
        return improved_solution, all_delta
    
    def _update(self, solution: MultiODSolution, path_id: int = 0):
        improved_path, delta, label = self.operator(solution=solution, path_id=path_id)
        
        modified = True if label else False
        return improved_path, delta, modified
    

class MultiPathsAction(Action):
    def __init__(self, action_index: int, operator: Operator):
        super().__init__(action_index=action_index, action_type='multi-paths', operator=operator)    
    
    def __call__(self, env: gym.Env):
        improved_solution = env.solution
        all_delta = 0.
        n_paths = len(improved_solution.paths)
        
        for path_id1 in range(n_paths):
            for path_id2 in range(path_id1 + 1, n_paths):
                inner_modified = True 
                while inner_modified:
                    improved_paths, delta1, inner_modified1 = self._update(improved_solution, path_id1, path_id2)
                    improved_paths, delta2, inner_modified2 = self._update(improved_solution, path_id2, path_id1)
                    if inner_modified1:
                        all_delta += delta1 
                    if inner_modified2:
                        all_delta += delta2 
                    inner_modified = inner_modified1 or inner_modified2
        return improved_solution, all_delta

    def _update(self, solution: MultiODSolution, path_id1: int = 0, path_id2: int = 1):
        improved_paths, delta, label = self.operator(solution=solution, path_id1=path_id1, path_id2=path_id2)
        modified = True if label else False 
        return improved_paths, delta, modified
    

class PathRandomAction(Action):
    def __init__(self, action_index: int, operator: Operator):
        super().__init__(action_index=action_index, action_type='path-random', operator=operator)
    
    def __call__(self, env: gym.Env):
        improved_solution = env.solution
        all_delta = 0.
        for path_id, path in enumerate(improved_solution.paths):
            improved_path, delta, _ = self.operator(solution=improved_solution, path_id=path_id)
            
            all_delta += delta 
        return improved_solution, all_delta