import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MsPacman-v4", render_mode=None)

# q_learning not that compatible due to complex state space, may need to consider alternatives

def q_learning(env, num_episodes, alpha, gamma, initial_epsilon, min_epsilon, decay_rate):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards = []
    epsilon = initial_epsilon

    for episode in range(num_episodes):
        state, _ = env.reset()
        action = None
        done = False
        truncated = False
        total_reward = 0

        while not done or not truncated:
            if np.random.uniform() < epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            q_table[state, action] += alpha * (reward + (gamma * max(q_table[next_state])) - q_table[state, action])
            state = next_state

        rewards.append(total_reward)
        epsilon = max(min_epsilon, epsilon * decay_rate)

    return q_table, rewards


num_episodes = 1000
alpha = 0.1
gamma = 0.99
initial_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.995

q_table, rewards = q_learning(env, num_episodes, alpha, gamma, initial_epsilon, min_epsilon, decay_rate)

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning")
plt.show()
