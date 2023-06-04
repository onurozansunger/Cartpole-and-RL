import os
import numpy as np
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Check if the monitor.csv file has been created, else do nothing
            if os.path.isfile(os.path.join(self.log_dir, "monitor.csv")):
                x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                if len(x) > 0:
                    mean_reward = np.mean(y[-100:])
                    if self.verbose > 0:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self.verbose > 0:
                            print(f"Saving new best model at {self.num_timesteps} timesteps")
                        self.model.save(os.path.join(self.log_dir, "best_model"))

                    if mean_reward >= 500:
                        print(f"Stopping at {self.num_timesteps} timesteps with reward {mean_reward}")
                        return False
        return True

log_dir = "/root/ppo_cartpole_tensorboard/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make('CartPole-v1')
env = Monitor(env, log_dir)

# Make sure the env is not wrapped twice
if isinstance(env, VecNormalize):
    raise ValueError("You are not supposed to manually normalize the environment. "
                     "Please remove the 'VecNormalize' wrapper.")

# Add some param noise for exploration
param_noise = None

model = PPO('MlpPolicy', env, verbose=0)
callback = EarlyStoppingCallback(check_freq=1000, log_dir=log_dir)
model.learn(total_timesteps=int(1e5), callback=callback)

# Don't forget to save the VecNormalize statistics when saving the agent
log_path = os.path.join(log_dir, "PPO_1")
model.save(log_path)

if callback.best_mean_reward >= 500:
    print("Training has been finished successfully.")
else:
    print("Training was not successful. Please try again.")

# Test the trained agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

