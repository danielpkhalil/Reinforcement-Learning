import numpy as np
import random

# Parameters
actions = ["C", "D"]  # C: Cooperate, D: Defect
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 10000

# Payoff matrix for the Prisoner's Dilemma
payoff_matrix = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}

# Q-tables for both agents
q_table_1 = {action: 0 for action in actions}
q_table_2 = {action: 0 for action in actions}

# Choose an action using an epsilon-greedy policy
def choose_action(q_table, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # Explore
    else:
        return max(q_table, key=q_table.get)  # Exploit

# Q-learning update
def update_q_table(q_table, action, reward, alpha, gamma):
    q_table[action] += alpha * (reward - q_table[action])

# Training loop
for episode in range(episodes):
    # Choose actions independently
    action_1 = choose_action(q_table_1, epsilon)
    action_2 = choose_action(q_table_2, epsilon)

    # Get rewards
    reward_1, reward_2 = payoff_matrix[(action_1, action_2)]

    # Update Q-tables
    update_q_table(q_table_1, action_1, reward_1, alpha, gamma)
    update_q_table(q_table_2, action_2, reward_2, alpha, gamma)

# Display results
print("Final Q-Table for Agent 1:")
print(q_table_1)

print("\nFinal Q-Table for Agent 2:")
print(q_table_2)

# Compute learned policies (softmax probabilities)
def compute_policy(q_table):
    total = sum(np.exp(q) for q in q_table.values())  # Softmax
    return {action: np.exp(q) / total for action, q in q_table.items()}

policy_1 = compute_policy(q_table_1)
policy_2 = compute_policy(q_table_2)

print("\nLearned Policy for Agent 1 (softmax probabilities):")
print(policy_1)

print("\nLearned Policy for Agent 2 (softmax probabilities):")
print(policy_2)

# Theoretical Nash equilibrium (pure strategy: always defect)
nash_policy = {"C": 0.0, "D": 1.0}

print("\nComparison to Theoretical Nash Equilibrium (Always Defect):")
print(f"Agent 1 policy deviation from Nash: {policy_1}")
print(f"Agent 2 policy deviation from Nash: {policy_2}")
