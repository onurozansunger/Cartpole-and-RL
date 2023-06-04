import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Actor modeli
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        prob = self.softmax(x)
        return prob

# Critic modeli
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# A2C agent sınıfı
class A2CAgent:
    def __init__(self, env, actor, critic, lr_actor=0.001, lr_critic=0.001, gamma=0.99):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        prob = self.actor(state)
        dist = Categorical(prob)
        action = dist.sample()
        return action.item()

    def update_model(self, log_probs, values, rewards):
        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        actor_loss = []
        critic_loss = []
        for log_prob, value, G in zip(log_probs, values, returns):
            advantage = G - value.item()

            actor_loss.append(-log_prob * advantage)
            critic_loss.append(nn.functional.smooth_l1_loss(value, G.unsqueeze(0)))

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        actor_loss = torch.stack(actor_loss).sum()
        critic_loss = torch.stack(critic_loss).sum()
        actor_loss.backward()
        critic_loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            log_probs = []
            values = []
            rewards = []
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                log_prob = torch.log(self.actor(torch.from_numpy(state).float().unsqueeze(0))[0, action])
                value = self.critic(torch.from_numpy(state).float().unsqueeze(0))

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)

                state = next_state

            self.update_model(log_probs, values, rewards)

            if episode % 10 == 0:
                print(f"Episode: {episode}, Reward: {sum(rewards)}")

# CartPole-v1 ortamını oluştur
env = gym.make('CartPole-v1')

# Actor ve Critic modellerini oluştur
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
actor = Actor(input_dim, output_dim)
critic = Critic(input_dim)

# A2C agentını oluştur
agent = A2CAgent(env, actor, critic)

# Eğitim yap
num_episodes = 10000
agent.train(num_episodes)

# Eğitim tamamlandığında modeli kaydet
torch.save(agent.actor.state_dict(), "a2c_cartpole_actor.pth")

# Ortamı kapat
env.close()

