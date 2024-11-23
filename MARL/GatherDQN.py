import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Experience replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, experience):
        self.memory.append(experience)
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        
        self.batch_size = 64
        self.gamma = 0.99
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
        self.eps = self.eps_start
        self.target_update = 10
        
    def select_action(self, state):
        if random.random() > self.eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_values = self.policy_net(state)
                return q_values.max(0)[1].item()
        else:
            return random.randrange(self.action_size)
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)
        
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1).detach()
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

class GatheringEnvironment:
    def __init__(self, size=8):
        self.size = size
        self.reset()
        
    def reset(self):
        self.grid = np.zeros((self.size, self.size))
        self.agent_pos = [(0, 0), (self.size-1, self.size-1)]
        self.resource_pos = []
        self.spawn_resources(3)
        return self._get_state()
    
    def spawn_resources(self, n):
        for _ in range(n):
            while True:
                x, y = random.randrange(self.size), random.randrange(self.size)
                if (x, y) not in self.agent_pos and (x, y) not in self.resource_pos:
                    self.resource_pos.append((x, y))
                    break
    
    def _get_state(self):
        states = []
        for agent_idx in range(2):
            state = []
            x, y = self.agent_pos[agent_idx]
            # Agent position
            state.extend([x/self.size, y/self.size])
            # Other agent position
            other_x, other_y = self.agent_pos[1-agent_idx]
            state.extend([other_x/self.size, other_y/self.size])
            # Nearest resource
            if self.resource_pos:
                distances = [((x-rx)**2 + (y-ry)**2)**0.5 
                           for rx, ry in self.resource_pos]
                min_dist = min(distances)
                rx, ry = self.resource_pos[distances.index(min_dist)]
                state.extend([rx/self.size, ry/self.size])
            else:
                state.extend([0, 0])
            states.append(state)
        return states
    
    def step(self, actions):
        rewards = [0, 0]
        for agent_idx, action in enumerate(actions):
            x, y = self.agent_pos[agent_idx]
            if action == 0:  # up
                y = max(0, y-1)
            elif action == 1:  # down
                y = min(self.size-1, y+1)
            elif action == 2:  # left
                x = max(0, x-1)
            elif action == 3:  # right
                x = min(self.size-1, x+1)
            self.agent_pos[agent_idx] = (x, y)
        
        for agent_idx in range(2):
            x, y = self.agent_pos[agent_idx]
            if (x, y) in self.resource_pos:
                self.resource_pos.remove((x, y))
                rewards[agent_idx] += 1
                
        if random.random() < 0.1:
            self.spawn_resources(1)
            
        done = len(self.resource_pos) == 0
        return self._get_state(), rewards, done

class PolicyVisualizer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.grid_size = env.size
        self.actions = ['↑', '↓', '←', '→']
        self.action_colors = ['#ff9999', '#99ff99', '#9999ff', '#ffff99']
    
    def get_policy_values(self, agent_idx=0, other_agent_pos=None, resource_pos=None):
        policy_values = np.zeros((self.grid_size, self.grid_size, len(self.actions)))
        
        if other_agent_pos is None:
            other_agent_pos = (self.grid_size-1, self.grid_size-1)
        if resource_pos is None:
            resource_pos = (self.grid_size//2, self.grid_size//2)
            
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                state = [
                    x/self.grid_size, y/self.grid_size,
                    other_agent_pos[0]/self.grid_size, other_agent_pos[1]/self.grid_size,
                    resource_pos[0]/self.grid_size, resource_pos[1]/self.grid_size
                ]
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(self.agent.device)
                    q_values = self.agent.policy_net(state_tensor)
                    policy_values[x, y] = q_values.cpu().numpy()
                
        return policy_values

    def plot_policy_analysis(self, episode):
        """Plot comprehensive policy analysis"""
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Policy heatmap
        ax1 = plt.subplot(121)
        policy_values = self.get_policy_values(
            other_agent_pos=(7, 7),
            resource_pos=(4, 4)
        )
        max_q_values = np.max(policy_values, axis=2)
        sns.heatmap(max_q_values, cmap='viridis', ax=ax1)
        ax1.plot(7.5, 7.5, 'ro', markersize=15, label='Other Agent')
        ax1.plot(4.5, 4.5, 'go', markersize=15, label='Resource')
        ax1.set_title(f'Policy Value Heatmap (Episode {episode})')
        ax1.legend()
        
        # 2. Action preferences
        ax2 = plt.subplot(122)
        preferred_actions = np.argmax(policy_values, axis=2)
        ax2.grid(True)
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                action = preferred_actions[x, y]
                confidence = np.max(policy_values[x, y]) - np.mean(policy_values[x, y])
                
                if action == 0:  # up
                    dx, dy = 0, -0.3
                elif action == 1:  # down
                    dx, dy = 0, 0.3
                elif action == 2:  # left
                    dx, dy = -0.3, 0
                else:  # right
                    dx, dy = 0.3, 0
                
                ax2.arrow(y + 0.5, x + 0.5, dy, dx,
                         head_width=0.1, head_length=0.1,
                         color=self.action_colors[action],
                         alpha=min(1.0, confidence))
        
        ax2.plot(7.5, 7.5, 'ro', markersize=15, label='Other Agent')
        ax2.plot(4.5, 4.5, 'go', markersize=15, label='Resource')
        ax2.set_title(f'Preferred Actions (Episode {episode})')
        ax2.legend()
        ax2.set_xlim(-0.5, self.grid_size + 0.5)
        ax2.set_ylim(-0.5, self.grid_size + 0.5)
        
        plt.tight_layout()
        plt.show()

def train_and_visualize(n_episodes=1000, visualization_interval=100):
    env = GatheringEnvironment()
    agents = [DQNAgent(6, 4) for _ in range(2)]
    visualizer = PolicyVisualizer(env, agents[0])
    
    episode_rewards = []
    cooperation_metrics = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = [0, 0]
        done = False
        
        while not done:
            actions = [agent.select_action(state[i]) for i, agent in enumerate(agents)]
            next_state, rewards, done = env.step(actions)
            
            for i in range(2):
                agents[i].memory.push(Experience(state[i], actions[i], rewards[i], 
                                               next_state[i], done))
                episode_reward[i] += rewards[i]
            
            for agent in agents:
                agent.learn()
            
            state = next_state
        
        if episode % agents[0].target_update == 0:
            for agent in agents:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
        
        episode_rewards.append(episode_reward)
        cooperation_metrics.append(abs(episode_reward[0] - episode_reward[1]))
        
        # Periodic visualization
        if (episode + 1) % visualization_interval == 0:
            print(f"\nEpisode {episode + 1}")
            print(f"Rewards: {episode_reward}")
            print(f"Epsilon: {agents[0].eps:.2f}")
            
            # Plot rewards
            plt.figure(figsize=(15, 5))
            rewards_array = np.array(episode_rewards)
            plt.subplot(121)
            plt.plot(rewards_array[:, 0], label='Agent 1', alpha=0.6)
            plt.plot(rewards_array[:, 1], label='Agent 2', alpha=0.6)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Agent Rewards over Time')
            plt.legend()
            
            plt.subplot(122)
            plt.plot(cooperation_metrics, label='Reward Difference', alpha=0.6)
            plt.xlabel('Episode')
            plt.ylabel('Absolute Reward Difference')
            plt.title('Cooperation Metric over Time')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            # Plot policy analysis
            visualizer.plot_policy_analysis(episode + 1)
    
    return agents, episode_rewards, cooperation_metrics

# Run training with visualizations
if __name__ == "__main__":
    trained_agents, rewards, cooperation = train_and_visualize(n_episodes=1000, visualization_interval=100)