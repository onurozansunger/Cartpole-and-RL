import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

# CartPole-v1 ortamını oluştur
env = gym.make('CartPole-v1')

# PPO modelini oluştur
model = PPO("MlpPolicy", env, verbose=1)

# Eğitim parametrelerini ayarla
total_timesteps = 500000
eval_freq = 1000
eval_episodes = 10
max_reward = 500

# Model kaydetmek için checkpoint callback'i oluştur
checkpoint_callback = CheckpointCallback(save_freq=eval_freq, save_path='./ppo_cartpole', name_prefix='model')

# Eğitim döngüsü
for timestep in range(total_timesteps):
    # Modeli eğit
    model.learn(total_timesteps=eval_freq, callback=checkpoint_callback)

    # Değerlendirme
    if timestep % eval_freq == 0:
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=eval_episodes)
        print(f"Timestep: {timestep} - Mean Reward: {mean_reward:.2f}")

        # Maksimum ödül elde edildiğinde eğitimi durdur
        if mean_reward >= max_reward:
            print(f"Achieved maximum reward of {max_reward} at timestep {timestep}")
            break

# Eğitim tamamlandığında veya maksimum ödüle ulaşıldığında bilgi mesajı yazdır
if timestep >= total_timesteps - 1:
    print(f"Training completed at timestep {timestep} without achieving maximum reward.")

# Eğitilen modeli kaydet
model.save("ppo_cartpole")

# Ortamı kapat
env.close()

