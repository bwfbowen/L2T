import gym
import jax
import coax
import haiku as hk

class MyAgent:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.model = self.build_model()

    def build_model(self):
        num_actions = self.env.action_space.n

        def func(S):
            logits = hk.Linear(num_actions)(S)
            return {'logits': logits}

        model = coax.Policy(func, self.env)
        return model

    def train(self, num_episodes):
        self.model = self.model.train(num_episodes=num_episodes)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = coax.Policy.load(path)

    def evaluate(self, num_episodes=100):
        episode_rewards = []
        for _ in range(num_episodes):
            s = self.env.reset()
            episode_reward = 0
            while True:
                a = self.model(s)
                s, r, done, _ = self.env.step(a)
                episode_reward += r
                if done:
                    break
            episode_rewards.append(episode_reward)
        return episode_rewards
