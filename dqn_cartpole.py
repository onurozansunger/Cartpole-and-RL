import os
import numpy as np
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

class CheckLearningRate(BaseCallback):
    """
    Bir callback ki eğer belirtilen ödül eşiğine ulaşıldıysa eğitimi durdurur.
    """

    def __init__(self, check_freq: int, log_dir: str, threshold: float, verbose=0):
        super(CheckLearningRate, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.threshold = threshold
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        """
        Yeni bir adımın ardından yapılacak işlemi tanımlar.
        """
        if self.n_calls % self.check_freq == 0:

            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward

        return True  # Do not stop training here

# "CartPole-v1" ortamını yaratır.
env = gym.make('CartPole-v1')

# Monitor wrapper'ını kullanarak ortamı sarmalar.
log_dir = "./tmp/gym/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

# Bir DQN modeli oluşturur.
model = DQN('MlpPolicy', env, learning_rate=1e-3, buffer_size=50000, exploration_fraction=0.1, target_update_interval=1000, verbose=1)

callback = CheckLearningRate(check_freq=1000, log_dir=log_dir, threshold=1e-2)

model.learn(total_timesteps=500000, callback=callback)  # training time increased

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

