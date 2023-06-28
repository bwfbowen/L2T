import gym
from stable_baselines3 import PPO

class Agent:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.model = PPO('MlpPolicy', self.env, verbose=1)

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = PPO.load(path)

    def evaluate(self, num_episodes=100):
        episode_rewards = []
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
            episode_rewards.append(episode_reward)
        return episode_rewards