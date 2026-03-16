import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Create the CartPole environment
env = gym.make("CartPole-v1")

# Neural network model for approximating Q-values
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995  # Slower decay for more stable exploration
batch_size = 64
target_update_freq = 1000
memory_size = 10000
episodes = 5000

# Add gradient clipping threshold
max_grad_norm = 1.0

# Add learning rate scheduler parameters
lr_decay_factor = 0.999
lr_decay_steps = 1000

# Initialize Q-networks
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
# Add learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_steps, gamma=lr_decay_factor)
memory = deque(maxlen=memory_size)

# Function to choose action using epsilon-greedy policy
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = policy_net(state)
        return torch.argmax(q_values).item()  # Exploit

# Function to optimize the model using experience replay
def optimize_model():
    if len(memory) < batch_size:
        return
    
    batch = random.sample(memory, batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

     # Convert lists of numpy arrays to single numpy arrays first for efficiency
    state_batch = np.array(state_batch)
    next_state_batch = np.array(next_state_batch)

    state_batch = torch.FloatTensor(state_batch)
    action_batch = torch.LongTensor(action_batch).unsqueeze(1)
    reward_batch = torch.FloatTensor(reward_batch)
    next_state_batch = torch.FloatTensor(next_state_batch)
    done_batch = torch.FloatTensor(done_batch)

    # Compute Q-values for current states
    q_values = policy_net(state_batch).gather(1, action_batch).squeeze()

    # Compute target Q-values using the target network
    with torch.no_grad():
        max_next_q_values = target_net(next_state_batch).max(1)[0]
        target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)

    loss = nn.MSELoss()(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    
    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
    
    optimizer.step()
    
    # Update learning rate scheduler
    scheduler.step()

# Main training loop
rewards_per_episode = []
steps_done = 0
moving_avg_window = 100
stable_threshold = 300 #195  # CartPole solved threshold
stable_episodes = 0

for episode in range(episodes):
    state, info = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        # Select action
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated # Combine for overall episode termination

        # Store transition in memory with normalized reward
        normalized_reward = reward / 100.0  # Scale rewards to prevent large gradients
        memory.append((state, action, normalized_reward, next_state, done))
        
        # Update state
        state = next_state
        episode_reward += reward
        
        # Optimize model
        optimize_model()

        # Update target network with soft updates for stability
        tau = 0.005  # Soft update parameter
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

        steps_done += 1

    # Decay epsilon with exponential schedule
    epsilon = epsilon_min + (epsilon - epsilon_min) * np.exp(-episode / 1000)
    
    rewards_per_episode.append(episode_reward)

    # Check for stability
    if len(rewards_per_episode) >= moving_avg_window:
        moving_avg = np.mean(rewards_per_episode[-moving_avg_window:])
        if moving_avg >= stable_threshold:
            stable_episodes += 1
            if stable_episodes >= 10:  # Consider stable if average reward stays above threshold for 10 episodes
                print(f"Training stabilized at episode {episode} with average reward {moving_avg:.2f}")
                break
        else:
            stable_episodes = 0

    if episode%10==0:
        print(f"Episode: {episode}, Epsilon: {epsilon:.4f}, Total Reward: {episode_reward}")




# Plotting the rewards per episode
import matplotlib.pyplot as plt
plt.bar(range(len(rewards_per_episode)), rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DQN on CartPole')
plt.show()