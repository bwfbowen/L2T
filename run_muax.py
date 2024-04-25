import os
from typing import Type 
from absl import flags
from muax.frameworks.acme.tf import mcts
from muax.frameworks.acme.tf.mcts.models import simulator, mlp

import acme 
from acme import wrappers
from acme import specs
from acme.tf import networks
from absl import app

import sonnet as snt 
import tensorflow as tf

from src import utils as od_utils
from src import wrappers as od_wrappers
from src import env as od_env
from src import types


RUN_DISTRIBUTED = flags.DEFINE_bool(
    'run_distributed', False, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
INSTANCE_PATH = flags.DEFINE_string(
    'instance_path', '',
    'path to instance.')
USE_SIMULATOR = flags.DEFINE_bool('use_simulator', True, 'To use simulator or not.')
TARGET_PATH = flags.DEFINE_string('target_path', os.path.join(os.path.expanduser('~'), 'l2t_paths', 'U'), 'path to target path.')
TARGET_PATH_TYPE = flags.DEFINE_string('target_path_type', 'lkh3', 'type of path.')
SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')
K_RECENT_ACTION = flags.DEFINE_integer('k_recent_action', 10, 'length of action history')
MAX_NO_IMPROVEMENT = flags.DEFINE_integer('max_no_improvement', 6, 'tolerance of no improvement')
MAX_EPISODE_STEPS = flags.DEFINE_integer('max_episode_steps', 500, 'max iteration steps per instance')
NUM_ACTION_ITERS = flags.DEFINE_integer('num_action_iters', 100, 'maximum number of iteration for each action executing operators per step.')
BEST_TOLERANCE = flags.DEFINE_float('best_tolerance', 0.05, 'best cost tolerance for no improvement.')
CHANGE_PCT = flags.DEFINE_float('change_pct', 0.1, 'change percentage of random actions.')
NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 20_000,
    'Number of environment steps to run the experiment for.')
EVAL_EVERY = flags.DEFINE_integer(
    'eval_every', 50_000,
    'How often (in actor environment steps) to run evaluation episodes.')
EVAL_EPISODES = flags.DEFINE_integer(
    'evaluation_episodes', 0,
    'Number of evaluation episodes to run periodically.')


class NodeOperatorEncoder(snt.Module):
  """Feature extractor for Multi-OD problem. 
    Leverages multi head self attention to extract features for raw solution features and concats with raw problem features.

    Parameters
    ------
    observation_spec: specs.EnvironmentSpec.observations
    hidden_dim: int
    num_heads: int, number of heads in multi head attention.
    """
  def __init__(
      self,
      observation_spec: types.ObservationExtras[specs.BoundedArray],
      hidden_dim: int = 64,
      num_heads: int = 8,
      name='encoder'
    ):
    super().__init__(name=name)

    self.sol_embed_head = snt.Sequential([
            snt.Conv1D(output_channels=hidden_dim, kernel_shape=1),
        ])
    self.sol_norm_before = snt.BatchNorm(True, True)
    
    self.sol_self_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)

    self.sol_residual = snt.Sequential([
            snt.Conv1D(output_channels=hidden_dim, kernel_shape=1),
            tf.keras.layers.ReLU(),
            snt.Conv1D(output_channels=hidden_dim, kernel_shape=1)
        ])
    
    self.sol_norm_after = snt.BatchNorm(True, True)
    self.ln = snt.Linear(output_size=hidden_dim)
    
    self.prob_norm = snt.BatchNorm(True, True)
    self.prob_ln = snt.Linear(output_size=hidden_dim)
  
  def __call__(self, observations: types.ObservationExtras, is_training: bool = True):
    solution_features, problem_features = observations.node_features, observations.operator_features
    solution_features = tf.transpose(solution_features, perm=[0, 2, 1])
    sol_embed = self.sol_embed_head(solution_features)
    sol_embed = self.sol_norm_before(sol_embed, is_training)
    sol_embed = tf.transpose(sol_embed, perm=[0, 2, 1])
    sol_embed = self.sol_self_attn(sol_embed, sol_embed)
    sol_embed = tf.transpose(sol_embed, perm=[0, 2, 1])
    identity = sol_embed
    sol_out = self.sol_residual(sol_embed)
    sol_out += identity
    sol_out = self.sol_norm_after(sol_out, is_training)
    sol_out = tf.reduce_sum(sol_out, axis=2)
    sol_out = self.ln(sol_out)

    prob_out = self.prob_norm(problem_features, is_training)
    prob_out = self.prob_ln(prob_out)
    
    out = tf.concat([sol_out, prob_out], axis=1)
    return out


def make_environment(with_logger: bool = False):
  locations = od_utils.read_instance_data(INSTANCE_PATH.value)
  env = od_env.DMultiODEnvExtras(
    k_recent_action=K_RECENT_ACTION.value,
    max_no_improvement=MAX_NO_IMPROVEMENT.value,
    max_steps=MAX_EPISODE_STEPS.value,
    locations=locations,
    seed=SEED.value,
    num_action_iters=NUM_ACTION_ITERS.value,
    change_pct=CHANGE_PCT.value,
    best_cost_tolerance=BEST_TOLERANCE.value,
    ignore_from_dummy_cost=False,
    ignore_to_dummy_cost=False,
    int_distance=False)
  
  env = wrappers.SinglePrecisionWrapper(env)
  if with_logger:
    if TARGET_PATH_TYPE.value == 'lkh3':
      target_paths = [od_utils.get_lkh3_tour_v2(TARGET_PATH.value)]
      if not env._problem.is_feasible(target_paths):
        raise ValueError('target path is not feasible')
    env = od_wrappers.PDTSPLoggingWrapper(
      env, 
      csv_directory='~/pdtsp-l2t',
      target_paths=target_paths,
      logging_interval=10)
  return env 


def make_model(env):
  if USE_SIMULATOR.value:
    model = simulator.Simulator(env)
  else:
    # TODO
    model = mlp.ReprMLPModel()
  return model 

  
def make_network(env_specs: specs.EnvironmentSpec):
  eval_network = snt.Sequential([
    NodeOperatorEncoder(env_specs.observations, 64, 8),
    snt.nets.MLP([256, 256]),
    networks.PolicyValueHead(env_specs.actions.num_values),
  ])
  return eval_network


def main(_):
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    for gpu in gpus: 
      tf.config.experimental.set_memory_growth(gpu, True)

  if RUN_DISTRIBUTED.value:
    #TODO
    pass 
  else:
    env = make_environment()
    model = make_model(env)
    if TARGET_PATH_TYPE.value == 'lkh3':
      target_paths = [od_utils.get_lkh3_tour_v2(TARGET_PATH.value)]
      if not env._problem.is_feasible(target_paths):
        raise ValueError('target path is not feasible')
    env = od_wrappers.PDTSPLoggingWrapper(
      env,
      csv_directory='~/pdtsp-l2t',
      target_paths=target_paths,
      logging_interval=10)
    env_specs = specs.make_environment_spec(env)
    net = make_network(env_specs)
    optimizer = snt.optimizers.Adam(learning_rate=1e-3)
    agent = mcts.MCTS(
      network=net,
      model=model,
      optimizer=optimizer,
      n_step=10,
      discount=0.99,
      replay_capacity=10000,
      num_simulations=50, 
      environment_spec=env_specs,
      batch_size=512,
    ) 
    loop = acme.EnvironmentLoop(env, agent)
    loop.run(num_steps=NUM_STEPS.value)


if __name__ == '__main__':
  app.run(main)