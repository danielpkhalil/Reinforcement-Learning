import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal

# Hyperparameters
env_name = "Humanoid-v5"
num_episodes = 1000
max_timesteps = 1000
gamma = 0.99  # Discount factor
lr = 3e-4     # Learning rate
eps_clip = 0.2  # Clipping for PPO
update_timestep = 4000  # How often to update the policy
k_epochs = 4  # Number of epochs per update

# Create the environment
env = gym.make(env_name)

# Get environment dimensions
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
action_std = 0.5  # Standard deviation of actions for exploration


# Define the Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self):
        raise NotImplementedError  # This is not used in PPO

    def act(self, state):
        mean = self.actor(state)
        cov_mat = torch.diag(action_std**2 * torch.ones(act_dim)).to(state.device)
        dist = MultivariateNormal(mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        mean = self.actor(state)
        cov_mat = torch.diag(action_std**2 * torch.ones(act_dim)).to(state.device)
        dist = MultivariateNormal(mean, cov_mat)
        action_logprob = dist.log_prob(action)
        entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprob, state_value, entropy


# PPO Agent
class PPO:
    def __init__(self, obs_dim, act_dim, lr, gamma, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.policy = ActorCritic(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(obs_dim, act_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.mse_loss = nn.MSELoss()

    def update(self, memory):
        states = torch.stack(memory["states"]).to(device)
        actions = torch.stack(memory["actions"]).to(device)
        logprobs = torch.stack(memory["logprobs"]).to(device)
        rewards = memory["rewards"]
        is_terminals = memory["is_terminals"]

        # Calculate discounted rewards
        discounted_rewards = []
        G = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                G = 0
            G = reward + self.gamma * G
            discounted_rewards.insert(0, G)
        discounted_rewards = torch.tensor(discounted_rewards).to(device)

        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        # PPO update
        for _ in range(k_epochs):
            # Evaluate old actions and values
            logprobs_old, state_values, entropy = self.policy.evaluate(states, actions)

            # Finding the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs_old - logprobs)

            # Surrogate loss
            advantages = discounted_rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, discounted_rewards) - 0.01 * entropy.mean()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())


# Memory for storing transitions
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []


# Training PPO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ppo_agent = PPO(obs_dim, act_dim, lr, gamma, eps_clip)
memory = Memory()

timestep = 0
for episode in range(num_episodes):
    state = env.reset()[0]
    state = torch.tensor(state, dtype=torch.float32).to(device)
    score = 0

    for t in range(max_timesteps):
        timestep += 1

        # Select action
        action, logprob = ppo_agent.policy_old.act(state)
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(logprob)

        # Perform action
        action_env = action.cpu().numpy()
        next_state, reward, terminated, truncated, _ = env.step(action_env)
        done = terminated or truncated

        # Store reward and terminal state
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        state = torch.tensor(next_state, dtype=torch.float32).to(device)
        score += reward

        # Update PPO agent
        if timestep % update_timestep == 0:
            ppo_agent.update(memory)
            memory.clear_memory()
            timestep = 0

        if done:
            break

    print(f"Episode {episode + 1}: Score: {score}")

env.close()
