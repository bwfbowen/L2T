from absl import flags
from acme import specs
from acme.agents.jax import mpo
from acme.agents.jax.mpo import types as mpo_types
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp

RUN_DISTRIBUTED = flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
INSTANCE_NAME = flags.DEFINE_string(
    'instance_name', '',
    'path to instance.')
SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')
NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 1_000_000,
    'Number of environment steps to run the experiment for.')
EVAL_EVERY = flags.DEFINE_integer(
    'eval_every', 50_000,
    'How often (in actor environment steps) to run evaluation episodes.')
EVAL_EPISODES = flags.DEFINE_integer(
    'evaluation_episodes', 10,
    'Number of evaluation episodes to run periodically.')


def build_experiment_config():
  """Builds MPO experiment config which can be executed in different ways."""
  suite, task = ENV_NAME.value.split(':', 1)
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
      environment_factory=lambda _: helpers.make_environment(suite, task),
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
