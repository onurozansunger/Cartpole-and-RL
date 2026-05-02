# Cartpole-and-RL

Reinforcement Learning experiments on the classic **CartPole-v1** environment from OpenAI Gym.

This repo contains from-scratch implementations of several RL algorithms applied to the same task, so the methods can be compared side by side.

## Algorithms

| File | Algorithm | Type |
| --- | --- | --- |
| [`dqn_cartpole.py`](dqn_cartpole.py) | Deep Q-Network (DQN) | Value-based |
| [`a2c_cartpole.py`](a2c_cartpole.py) | Advantage Actor-Critic (A2C) | Actor-Critic |
| [`ddpg_cartpole.py`](ddpg_cartpole.py) | Deep Deterministic Policy Gradient (DDPG) | Actor-Critic (continuous) |
| [`ppo_cartpole.py`](ppo_cartpole.py) | Proximal Policy Optimization (PPO) | Policy gradient |
| [`firsttry.ipynb`](firsttry.ipynb) | Initial Q-Learning experiments | Tabular baseline |

## Environment

- **CartPole-v1** — balance a pole on a moving cart by pushing it left or right.
- **State:** cart position, cart velocity, pole angle, pole angular velocity
- **Actions:** discrete (`left`, `right`) — DDPG version uses a continuous action adaptation.
- **Reward:** +1 per timestep the pole stays upright (max 500).

## Requirements

```bash
pip install gym numpy torch matplotlib
```

## Usage

```bash
python dqn_cartpole.py
python a2c_cartpole.py
python ppo_cartpole.py
python ddpg_cartpole.py
```

Each script trains an agent and prints/plots the episode rewards over time.

## Notes

These were R&D-style studies for self-learning Reinforcement Learning. Hyperparameters and training loops are intentionally minimal to keep the algorithms readable.

## Author

**Onur Ozan Sünger** — MSc Data Science, Sapienza University of Rome
