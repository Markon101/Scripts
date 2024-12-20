import gym
import numpy as np

# Create the CartPole environment
env = gym.make("CartPole-v1")

# Set random seed for reproducibility
np.random.seed(42)

# Initialize Q-table (state-action values)
num_bins = 20  # Number of bins for discretization
num_states = [num_bins] * env.observation_space.shape[0]
num_actions = env.action_space.n
Q = np.zeros(num_states + [num_actions])

# Discretization function
def discretize_state(state):
    # Normalize continuous state values to [0, num_bins)
    scaled_state = (state - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
    discretized_state = np.floor(scaled_state * num_bins).astype(int)
    return tuple(discretized_state)

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.99
exploration_prob = 0.1

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()

    for _ in range(200):  # Max episode length
        discretized_state = discretize_state(state)

        # Choose an action using epsilon-greedy policy
        if np.random.rand() < exploration_prob:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[discretized_state])

        next_state, reward, done, _ = env.step(action)

        # Update Q-value using Q-learning update rule
        next_discretized_state = discretize_state(next_state)
        Q[discretized_state + (action,)] += learning_rate * (
            reward + discount_factor * np.max(Q[next_discretized_state]) - Q[discretized_state + (action,)]
        )

        state = next_state

        if done:
            break

    print(f"Episode {episode+1}: Total reward = {np.sum(Q)}")

# Inference loop (optional)
num_inference_episodes = 5
for _ in range(num_inference_episodes):
    state = env.reset()
    done = False

    while not done:
        env.render()  # Display the environment
        discretized_state = discretize_state(state)
        action = np.argmax(Q[discretized_state])
        state, _, done, _ = env.step(action)

# Close the environment
env.close()

