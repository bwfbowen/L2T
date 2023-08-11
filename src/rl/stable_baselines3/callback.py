import os 
import time 
import numpy as np
import matplotlib.pyplot as plt 

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure

from .. import utils, solution


display_result = utils.display_result
MultiODSolution = solution.MultiODSolution


class SaveBestSolCallback(BaseCallback):
    """Callback class that records solutions and corresponding statistics.
    
    Records best solutions during training, and the time and steps spent for finding them 
    with tensorboard utilities.

    Parameters
    ------
    log_dir: str, saving path of the tensorboard results.
    instance_name: str, the name of the result directory, also the name displayed in tensorboard.
    verbose: int, whether to display the content. `verbose > 0` to display the content.
    target_cost: float, the `target_cost` is the target of the training, by giving this parameter, 
        `target_gap` will be calculated. If param `target_tour` is given and `target_cost` is not,
        `target_cost` will be calculated from `target_tour`.
    target_tour: MultiODSolution, list[list[int]]. The tour provided as the target. The fig display of 
        the target tour will be recorded.
    early_stop: bool, if a solution reaches the same cost as target cost, then the training will stop.

    """
    def __init__(self, log_dir: str, instance_name: str, verbose: int = 0, target_cost: float = None, target_tour: MultiODSolution = None, early_stop: bool = True):
        super().__init__(verbose=verbose)
        self.cur_best_cost = np.inf 
        self.cur_best_sol = None
        self._rollout_best_sol = None
        self.rollout_best_cost = np.inf
        self.rollout_best_sol_at_step = 0
        self.prev_rollout_best_cost = 0.
        self.log_dir = log_dir
        self.instance_name = instance_name
        self.target_cost = target_cost
        self.target_tour = target_tour
        self.early_stop = early_stop

    def _init_callback(self):
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _on_training_start(self):
        self.start_time = time.time()
        if self.target_tour is not None:
            _problem = self.training_env.get_attr('problem')[0]
            target_figure = display_result(_problem, self.target_tour)
            self.logger.record('target/target_tour', Figure(target_figure, close=True), exclude=('stdout', 'log', 'json', 'csv'))
            plt.close()
            if self.target_cost is None:
                self.target_cost = int(_problem.calc_cost(self.target_tour))
            self.logger.record('target/target_cost', self.target_cost)
        if self.verbose >= 1 and self.target_cost is not None:
            print(f'Target cost: {self.target_cost}')

    def _on_step(self) -> bool:
        best_costs = self.training_env.get_attr('best_cost')
        best_cost_index = np.argmin(best_costs)
        best_cost = int(best_costs[best_cost_index])
        if self.cur_best_cost > best_cost:
            self.cur_best_cost = best_cost
            found_time = time.time()
            best_sol = self.training_env.get_attr('best_solution')[best_cost_index]
            self.cur_best_sol = best_sol 
            best_sol_at_step = self.training_env.get_attr('best_sol_at_step')[best_cost_index]
            _problem = self.training_env.get_attr('problem')[best_cost_index]
            best_sol = MultiODSolution(best_sol, _problem)
            sol_figure = display_result(_problem, best_sol)
            self.logger.record('best/best_sol', Figure(sol_figure, close=True), exclude=('stdout', 'log', 'json', 'csv'))
            plt.close()
            self.logger.record('best/best_cost', best_cost)
            self.logger.record('best/best_sol_at_step', best_sol_at_step)
            self.logger.record('best/best_sol_found_time', found_time - self.start_time)
            if self.verbose >= 1:
                print(f'Best solution cost: {best_cost}, found at {best_sol_at_step} step, {found_time - self.start_time:.2f} seconds used')
        if self.rollout_best_cost > best_cost:
            self.rollout_best_cost = best_cost
            found_time = time.time()
            best_sol = self.training_env.get_attr('best_solution')[best_cost_index]
            self._rollout_best_sol = best_sol
            self._rollout_best_index = best_cost_index
            best_sol_at_step = self.training_env.get_attr('best_sol_at_step')[best_cost_index]
            self.rollout_best_sol_at_step = best_sol_at_step
            self.rollout_found_time = found_time
        if self.target_cost is not None and self.early_stop and best_cost <= self.target_cost:
            self.rollout_end_and_early_stop_logger()
            self.logger.dump(self.num_timesteps)
            return False 
        return True
    
    def _on_rollout_start(self):
        self.rollout_start_time = time.time()
    
    def _on_rollout_end(self):
        self.rollout_end_and_early_stop_logger()
    
    def _on_training_end(self):
        if self.cur_best_sol:
            f = open(os.path.join(self.log_dir, f'{self.instance_name}.{int(self.cur_best_cost)}.tour'), 'w')
            f.write(str(self.cur_best_sol))
            f.close()

    def rollout_end_and_early_stop_logger(self):
        self.logger.record('rollout/rollout_best_cost', self.rollout_best_cost)
        self.logger.record('rollout/rollout_best_sol_at_step', self.rollout_best_sol_at_step)
        self.logger.record('rollout/rollout_best_sol_found_time', self.rollout_found_time - self.rollout_start_time)
        convergence_gap = abs(self.rollout_best_cost - self.prev_rollout_best_cost)
        self.logger.record('rollout/convergence_gap', convergence_gap)

        if self.rollout_best_cost <= self.cur_best_cost:
            self.model.save(os.path.join(self.log_dir, f'{self.instance_name}.{int(self.cur_best_cost)}.model'))
        if self.target_cost is not None:
            target_gap = self.rollout_best_cost - self.target_cost
            self.logger.record('rollout/target_gap', target_gap)
        if self.verbose >= 1:
            logging_str = f'''Rollout best solution cost: {self.rollout_best_cost}, 
                  found at {self.rollout_best_sol_at_step} step, {self.rollout_found_time - self.rollout_start_time:.2f} seconds used. 
                  Convergence gap: {convergence_gap}. ''' + (f'Target gap: {target_gap}' if self.target_cost is not None else '')
            print(logging_str)
        self.prev_rollout_best_cost = self.rollout_best_cost
        self.rollout_best_cost = np.inf 
        self.rollout_best_sol_at_step = 0
        if self.cur_best_sol:
            f = open(os.path.join(self.log_dir, f'{self.instance_name}.{int(self.cur_best_cost)}.tour'), 'w')
            f.write(str(self.cur_best_sol))
            f.close()