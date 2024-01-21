from absl import flags
from acme.agents.jax import dqn, ppo, mpo
from acme.agents.jax.mpo import types as mpo_types
from acme import wrappers
from acme import specs
from acme.jax import utils
import haiku as hk
import optax 
from acme.jax import networks as networks_lib
from acme.agents.jax.dqn import losses
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp

from src import utils as od_utils
from src import wrappers as od_wrappers
from src import env as od_env


RUN_DISTRIBUTED = flags.DEFINE_bool(
    'run_distributed', False, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
INSTANCE_PATH = flags.DEFINE_string(
    'instance_path', '',
    'path to instance.')
ALGORITHM = flags.DEFINE_string(
  'algorithm', 'ppo', 'algorithm to use.')
TARGET_PATH = flags.DEFINE_string('target_path', '', 'path to target path.')
TARGET_PATH_TYPE = flags.DEFINE_string('target_path_type', 'lkh3', 'type of path.')
SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')
K_RECENT_ACTION = flags.DEFINE_integer('k_recent_action', 10, 'length of action history')
MAX_NO_IMPROVEMENT = flags.DEFINE_integer('max_no_improvement', 6, 'tolerance of no improvement')
MAX_EPISODE_STEPS = flags.DEFINE_integer('max_episode_steps', int(1e4), 'max iteration steps per instance')
NUM_ACTION_ITERS = flags.DEFINE_integer('num_action_iters', 100, 'maximum number of iteration for each action executing operators per step.')
BEST_TOLERANCE = flags.DEFINE_float('best_tolerance', 0.05, 'best cost tolerance for no improvement.')
CHANGE_PCT = flags.DEFINE_float('change_pct', 0.1, 'change percentage of random actions.')
NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 1_000_000,
    'Number of environment steps to run the experiment for.')
EVAL_EVERY = flags.DEFINE_integer(
    'eval_every', 50_000,
    'How often (in actor environment steps) to run evaluation episodes.')
EVAL_EPISODES = flags.DEFINE_integer(
    'evaluation_episodes', 0,
    'Number of evaluation episodes to run periodically.')


def build_experiment_config():
  """Builds experiment config which can be executed in different ways."""

  def make_environment():
    locations = od_utils.read_instance_data(INSTANCE_PATH.value)
    env = od_env.DMultiODEnv(
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
    if TARGET_PATH_TYPE.value == 'lkh3':
      target_paths = [od_utils.get_lkh3_tour_v2(TARGET_PATH.value)]
      if not env._problem.is_feasible(target_paths):
        raise ValueError('target path is not feasible')
    env = od_wrappers.PDTSPLoggingWrapper(
      env, 
      csv_directory='~/pdtsp-l2t',
      target_paths=target_paths,
      logging_interval=1)
    env = wrappers.SinglePrecisionWrapper(env)
    return env 
  
  if ALGORITHM.value == 'mdqn':
    def make_dqn_network(
        environment_spec: specs.EnvironmentSpec) -> dqn.DQNNetworks:
      """Creates networks for training DQN."""
      def network(inputs):
        model = hk.Sequential([
            hk.nets.MLP([512, environment_spec.actions.num_values]),
        ])
        return model(inputs)
      network_hk = hk.without_apply_rng(hk.transform(network))
      obs = utils.add_batch_dim(utils.zeros_like(environment_spec.observations))
      network = networks_lib.FeedForwardNetwork(
          init=lambda rng: network_hk.init(rng, obs), apply=network_hk.apply)
      typed_network = networks_lib.non_stochastic_network_to_typed(network)
      return dqn.DQNNetworks(policy_network=typed_network)
    # Construct the agent.
    config = dqn.DQNConfig(
        discount=0.99,
        eval_epsilon=0.,
        learning_rate=5e-5,
        n_step=1,
        epsilon=[0.01, 0.05, 0.5],
        target_update_period=2000,
        min_replay_size=20_000,
        max_replay_size=1_000_000,
        samples_per_insert=8,
        batch_size=32)
    loss_fn = losses.MunchausenQLearning(
        discount=config.discount, max_abs_reward=1., huber_loss_parameter=1.,
        entropy_temperature=0.03, munchausen_coefficient=0.9)
    dqn_builder = dqn.DQNBuilder(config, loss_fn=loss_fn)
    return experiments.ExperimentConfig(
        builder=dqn_builder,
        environment_factory=lambda _: make_environment(),
        network_factory=make_dqn_network,
        seed=SEED.value,
        max_num_actor_steps=NUM_STEPS.value)
  
  elif ALGORITHM.value == 'ppo':
    config = ppo.PPOConfig(
        normalize_advantage=True,
        normalize_value=True,
        obs_normalization_fns_factory=ppo.build_mean_std_normalizer)
    ppo_builder = ppo.PPOBuilder(config)
    layer_sizes = (64, 256, 256)
    return experiments.ExperimentConfig(
        builder=ppo_builder,
        environment_factory=lambda _: make_environment(),
        network_factory=lambda spec: ppo.make_discrete_networks(spec, layer_sizes, use_conv=False),
        seed=SEED.value,
        max_num_actor_steps=NUM_STEPS.value)
  
  elif ALGORITHM.value == 'mpo':
    critic_type = mpo.CriticType.NONDISTRIBUTIONAL
    def network_factory(spec: specs.EnvironmentSpec) -> mpo.MPONetworks:
      return mpo.make_control_networks(
          spec,
          policy_layer_sizes=(256, 256, 256),
          critic_layer_sizes=(256, 256, 256),
          policy_init_scale=0.5,
          critic_type=critic_type)
    # Configure and construct the agent builder.
    config = mpo.MPOConfig(
        critic_type=critic_type,
        policy_loss_config=mpo_types.GaussianPolicyLossConfig(epsilon_mean=0.01),
        samples_per_insert=64,
        learning_rate=3e-4,
        experience_type=mpo_types.FromTransitions(n_step=4))
    agent_builder = mpo.MPOBuilder(config, sgd_steps_per_learner_step=1)
    return experiments.ExperimentConfig(
        builder=agent_builder,
        environment_factory=lambda _: make_environment(),
        network_factory=network_factory,
        seed=SEED.value,
        max_num_actor_steps=NUM_STEPS.value)


def main(_):
  config = build_experiment_config()
  if RUN_DISTRIBUTED.value:
    program = experiments.make_distributed_experiment(
        experiment=config, num_actors=4)
    lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
  else:
    experiments.run_experiment(
        experiment=config,
        eval_every=EVAL_EVERY.value,
        num_eval_episodes=EVAL_EPISODES.value)


if __name__ == '__main__':
  app.run(main)