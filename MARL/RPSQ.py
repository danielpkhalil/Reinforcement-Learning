import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.classic import rps_v2

# Create the Rock-Paper-Scissors environment
env = rps_v2.env()

# Q-learning parameters
alpha = 0.1    # Learning rate
gamma = 0.9    # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 10000

# Initialize Q-tables for both agents
q_table = {agent: np.zeros((3, 3)) for agent in env.possible_agents}  # 3 states (rock, paper, scissors) and 3 actions (rock, paper, scissors)

# Function to choose an action using epsilon-greedy policy
def choose_action(state, q_table, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, 3)  # Explore: choose a random action
    else:
        return np.argmax(q_table[state])  # Exploit: choose the action with the highest Q-value

# Main Q-learning loop
for episode in range(num_episodes):
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        done = termination or truncation

        if done:
            env.step(None)
        else:
            current_state = np.argmax(observation)
            action = choose_action(current_state, q_table[agent], epsilon)
            env.step(action)
            next_observation, next_reward, next_termination, next_truncation, next_info = env.last()
            next_state = np.argmax(next_observation)
            q_table[agent][current_state, action] += alpha * (reward + gamma * np.max(q_table[agent][next_state]) - q_table[agent][current_state, action])

print("Training complete.")

# Visualize the Q-tables
def plot_q_table(q_table, agent):
    plt.imshow(q_table[agent], cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Q-table for {agent}')
    plt.xlabel('Action (0: Rock, 1: Paper, 2: Scissors)')
    plt.ylabel('State (0: Rock, 1: Paper, 2: Scissors)')
    plt.xticks([0, 1, 2], ['Rock', 'Paper', 'Scissors'])
    plt.yticks([0, 1, 2], ['Rock', 'Paper', 'Scissors'])
    plt.show()

for agent in env.possible_agents:
    plot_q_table(q_table, agent)
