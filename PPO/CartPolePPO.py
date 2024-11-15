import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create the environment with rendering mode
def make_env():
    return gym.make('CartPole-v1', render_mode='human')

env = DummyVecEnv([make_env])

# Create the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_cartpole")

# Load the model (optional)
model = PPO.load("ppo_cartpole")

# Evaluate the trained model with rendering
episodes = 100
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0
    
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        score += reward
        
        # Render the environment
        env.render()
    
    print(f"Episode: {episode}, Score: {score}")

env.close()
