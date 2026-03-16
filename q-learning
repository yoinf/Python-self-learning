import numpy as np
import random
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.grid = np.zeros((size, size))
        self.goal = (size - 1, size - 1) # Treasure
        self.pit = (2, 2) # Pit
        self.grid[self.goal] = 1 # Reward for treasure
        self.grid[self.pit] = -1 # Penalty for pit
        self.start = (0, 0)
        self.current_state = self.start
        self.actions = ['up', 'down', 'left', 'right']

    def step(self, action):
        x, y = self.current_state
        if action == 'up':
            x = max(0, x - 1)
        elif action == 'down':
            x = min(self.size - 1, x + 1)
        elif action == 'left':
            y = max(0, y - 1)
        elif action == 'right':
            y = min(self.size - 1, y + 1)

        self.current_state = (x, y)
        reward = self.grid[self.current_state]
        done = self.current_state == self.goal or self.current_state == self.pit
        
        return self.current_state, reward, done

    def reset(self):
        self.current_state = self.start
        return self.current_state

    def render(self):
        temp_grid = np.copy(self.grid)
        x, y = self.current_state
        temp_grid[x, y] = 5 # Agent's position
        print(temp_grid)

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.995):
        self.env = env
        self.q_table = np.zeros((env.size, env.size, len(env.actions)))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.min_epsilon = min_exploration_rate
        self.epsilon_decay = exploration_decay
        self.action_space_size = len(env.actions)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            action_index = random.randint(0, self.action_space_size - 1)
        else:
            # Exploit: choose the action with the highest Q-value
            state_index = state[0] * self.env.size + state[1]
            action_index = np.argmax(self.q_table[state])
        return self.env.actions[action_index]
    
    def update_q_table(self, state, action, reward, next_state):
        action_index = self.env.actions.index(action)
        
        # Q-value of the current state-action pair
        current_q_value = self.q_table[state[0], state[1], action_index]
        
        # Max Q-value for the next state
        max_next_q_value = np.max(self.q_table[next_state[0], next_state[1]])
        
        # Q-learning update rule
        new_q_value = (1 - self.lr) * current_q_value + self.lr * (reward + self.gamma * max_next_q_value)
        self.q_table[state[0], state[1], action_index] = new_q_value

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# Create the environment and agent
env = GridWorld(size=5)
agent = QLearningAgent(env)

# Training parameters
num_episodes = 2000
max_steps_per_episode = 100
rewards_per_episode = []

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    for step in range(max_steps_per_episode):
        # Choose action based on epsilon-greedy policy
        action = agent.choose_action(state)

        # Take the action and observe the outcome
        next_state, reward, done = env.step(action)

        # Update the Q-table
        agent.update_q_table(state, action, reward, next_state)

        # Update the state and reward
        state = next_state
        episode_reward += reward

        if done:
            break
    
    # Decay the exploration rate
    agent.decay_epsilon()
    rewards_per_episode.append(step)

    if episode % 100 == 0:
        print(f"Episode: {episode}, Epsilon: {agent.epsilon:.4f}, Total Reward: {episode_reward}")

print("Training finished.")

# Plot the rewards
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward per Episode during Training")
plt.show()
print(agent.q_table)
''''''
# Test the trained agent
print("\nTesting the trained agent:")
state = env.reset()
done = False
env.render()
while not done:
    # Choose action by exploiting (no exploration)
    action = env.actions[np.argmax(agent.q_table[state[0], state[1]])]
    print(f"Agent takes action: {action}")
    next_state, reward, done = env.step(action)
    state = next_state
    env.render()
    if done:
        print("Episode finished.")
''''''
