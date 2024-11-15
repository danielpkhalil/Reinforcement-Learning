import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

wandb.init(project="HumanoidPPO")

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = torch.tanh(self.fc3(x))
        std = torch.exp(self.log_std)
        return mu, std

# Define the value network
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# PPO update function
def ppo_update(policy_net, value_net, optimizer_policy, optimizer_value, trajectories, epsilon, gamma, beta):
    states = torch.tensor(np.array(trajectories['states']), dtype=torch.float32)
    actions = torch.tensor(np.array(trajectories['actions']), dtype=torch.float32)
    rewards = torch.tensor(np.array(trajectories['rewards']), dtype=torch.float32)
    old_log_probs = torch.tensor(np.array(trajectories['log_probs']), dtype=torch.float32)
    
    # Compute discounted rewards and advantages
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    values = value_net(states).detach().squeeze()
    advantages = returns - values

    # Log rewards to wandb
    wandb.log({"reward": rewards.sum().item()})
    
    # PPO update
    for _ in range(4):  # Number of epochs
        mu, std = policy_net(states)
        dist = torch.distributions.Normal(mu, std)
        new_log_probs = dist.log_prob(actions).sum(dim=1)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()
        
        value_loss = ((returns - value_net(states).squeeze()) ** 2).mean()
        
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

# Training loop
def train_ppo(env, policy_net, value_net, optimizer_policy, optimizer_value, num_episodes, gamma, epsilon, beta):
    for episode in range(num_episodes):
        state, _ = env.reset()  # Use env.reset() properly
        done = False
        score = 0
        trajectories = {'states': [], 'actions': [], 'rewards': [], 'log_probs': []}
        
        while not done:
            if isinstance(state, dict):
                state = state['observation']  # Extract the 'observation' part if state is a dictionary
            state = np.array(state).flatten()  # Flatten the state to make it uniform
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Ensure correct shape
            mu, std = policy_net(state_tensor)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample().detach().numpy().flatten()
            next_state, reward, terminated, truncated, info = env.step(action)
            if isinstance(next_state, dict):
                next_state = next_state['observation']  # Ensure next_state is handled correctly
            done = terminated or truncated
            
            log_prob = dist.log_prob(torch.tensor(action)).sum().item()
            trajectories['states'].append(state)
            trajectories['actions'].append(action)
            trajectories['rewards'].append(reward)
            trajectories['log_probs'].append(log_prob)
            
            state = next_state
            score += reward
        
        ppo_update(policy_net, value_net, optimizer_policy, optimizer_value, trajectories, epsilon, gamma, beta)
        print(f"Episode: {episode + 1}, Score: {score}")

if __name__ == "__main__":
    env = gym.make('Humanoid-v5', render_mode='human')
    obs_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    
    policy_net = PolicyNetwork(obs_dim, action_dim)
    value_net = ValueNetwork(obs_dim)
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=1e-4)
    optimizer_value = optim.Adam(value_net.parameters(), lr=1e-3)
    
    num_episodes = 10000
    gamma = 0.99
    epsilon = 0.2
    beta = 0.01
    
    train_ppo(env, policy_net, value_net, optimizer_policy, optimizer_value, num_episodes, gamma, epsilon, beta)
    env.close()
