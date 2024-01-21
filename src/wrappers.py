from typing import Optional, List

from acme.utils import loggers
from acme.wrappers import EnvironmentWrapper
import dm_env

from src import types
from src import env as od_env
from src import solution as od_solution


class PDTSPLoggingWrapper(EnvironmentWrapper):
    def __init__(self, 
                 environment: od_env.DMultiODEnv,
                 logger: Optional[loggers.Logger] = None,
                 csv_directory: str = '~/acme',
                 target_paths: List[types.Path] = [],
                 logging_interval: int = 100,
    ):
        super().__init__(environment)
        self._logger = logger or loggers.TerminalLogger()
        self._csv_logger = loggers.CSVLogger(directory_or_file=csv_directory, label='env_csv_log') if csv_directory else None 
        target_sol = od_solution.MultiODSolutionV2(target_paths, self._environment._problem.info)
        self._target_cost = self._environment._problem.calc_cost(target_sol)
        self._global_best_cost = float('inf')
        self
        self._action_counter = {str(a): 0 for a in self._environment._actions}
        self._logging_interval = logging_interval
        self._cur_step = 0

    def step(self, action) -> dm_env.TimeStep:
        transition = super().step(action)
        self._cur_step += 1
        self._global_best_cost = min(self._global_best_cost, self._environment._best_sol.best_cost)
        self._action_counter[str(self._environment._actions[action])] += 1
        if self._cur_step % self._logging_interval == 0:
            _log = {'cost': self._environment._current_cost, 
                    'best cost': self._environment._best_sol.best_cost,
                    'step': self._environment._current_step,
                    'best step': self._environment._best_sol.best_step,
                    'target cost': self._target_cost,
                    'episode gap': self._environment._best_sol.best_cost - self._target_cost,
                    'target gap': self._global_best_cost - self._target_cost,
                    'action counts': self._action_counter,}
            self._logger.write(_log)
            if self._csv_logger: self._csv_logger.write(_log)
        if self._environment._best_sol.best_cost <= self._target_cost:
            self._environment._reset_next_step = True 
            return dm_env.termination(reward=transition.reward, observation=transition.observation)
        return transition
    
    def reset(self) -> dm_env.TimeStep:
        _timestep = super().reset()
        self._cur_step = 0
        self._action_counter = {str(a): 0 for a in self._environment._actions}
        return _timestep