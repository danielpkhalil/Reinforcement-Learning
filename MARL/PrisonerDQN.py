import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

class PrisonersDilemmaEnv:
    def __init__(self, k):
        self.k = k
        self.history = []

    def reset(self):
        self.history = []
        return self.get_state()

    def step(self, action1, action2):
        reward1, reward2 = self._get_rewards(action1, action2)
        self.history.append((action1, action2))
        if len(self.history) > self.k:
            self.history.pop(0)
        return self.get_state(), (reward1, reward2)

    def get_state(self):
        state = np.zeros((self.k, 2))
        for i, (a1, a2) in enumerate(self.history[-self.k:]):
            state[i] = [a1, a2]
        return state.flatten()

    def _get_rewards(self, action1, action2):
        if action1 == 0 and action2 == 0:  # Cooperate, Cooperate
            return 3, 3
        elif action1 == 0 and action2 == 1:  # Cooperate, Defect
            return 0, 5
        elif action1 == 1 and action2 == 0:  # Defect, Cooperate
            return 5, 0
        else:  # Defect, Defect
            return 1, 1

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size)
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.target_model(next_state)[0]).item()
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0))
            target_f[0][action] = target
            self.model.zero_grad()
            criterion = nn.MSELoss()
            loss = criterion(target_f, torch.FloatTensor([target]))
            loss.backward()
            for param in self.model.parameters():
                param.data += param.grad * self.learning_rate

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

def train_dqn_agents(k=5, episodes=1000, batch_size=32):
    env = PrisonersDilemmaEnv(k)
    state_size = env.get_state().shape[0]
    action_size = 2
    agent1 = DQNAgent(state_size, action_size)
    agent2 = DQNAgent(state_size, action_size)

    scores1 = []
    scores2 = []
    choices1 = []
    choices2 = []

    for e in range(episodes):
        state = env.reset()
        total_reward1 = 0
        total_reward2 = 0
        choices1_episode = []
        choices2_episode = []
        for time in range(200):
            action1 = agent1.act(state)
            action2 = agent2.act(state)
            next_state, (reward1, reward2) = env.step(action1, action2)
            done = time == 199
            agent1.remember(state, action1, reward1, next_state, done)
            agent2.remember(state, action2, reward2, next_state, done)
            state = next_state
            total_reward1 += reward1
            total_reward2 += reward2
            choices1_episode.append(action1)
            choices2_episode.append(action2)
            if done:
                agent1.update_target_model()
                agent2.update_target_model()
                break
        scores1.append(total_reward1)
        scores2.append(total_reward2)
        choices1.append(choices1_episode)
        choices2.append(choices2_episode)
        if len(agent1.memory) > batch_size:
            agent1.replay(batch_size)
            agent2.replay(batch_size)
        if e % 100 == 0:
            print(f"Episode: {e}/{episodes}, Score1: {total_reward1}, Score2: {total_reward2}")

    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(scores1, label="Agent 1")
    plt.plot(scores2, label="Agent 2")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()

    plt.subplot(3, 1, 2)
    choices1_flattened = np.array([item for sublist in choices1 for item in sublist]).reshape(-1, 1)
    plt.imshow(choices1_flattened.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
    plt.xlabel("Time Steps")
    plt.ylabel("Agent 1 Choices (0 = Cooperate, 1 = Defect)")
    plt.colorbar(ticks=[0, 1], label='Choice')

    plt.subplot(3, 1, 3)
    choices2_flattened = np.array([item for sublist in choices2 for item in sublist]).reshape(-1, 1)
    plt.imshow(choices2_flattened.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
    plt.xlabel("Time Steps")
    plt.ylabel("Agent 2 Choices (0 = Cooperate, 1 = Defect)")
    plt.colorbar(ticks=[0, 1], label='Choice')

    plt.tight_layout()
    plt.show()

train_dqn_agents()
