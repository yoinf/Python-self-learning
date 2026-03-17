# open the program
import os
import time
import subprocess
from pywinauto import Application, timings
import traceback
# machine learning
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import itertools
# UDP
import socket #, threading, math
# output
import csv
import signal
# plot
import matplotlib.pyplot as plt

# ========== system parameters ==========
SEND_IP = '192.168.40.2'
SEND_PORT = 32361
RECV_PORT = 32360
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.bind(('', RECV_PORT))
recv_sock.settimeout(2.0)

# early stop
early_stop_patience = 300      # Number of episodes to wait for improvement
early_stop_threshold = 80.0    # Average reward threshold for "success"
rolling_window = 100           # How many episodes to average over
best_avg_reward = -float('inf')
early_stop_counter = 0

# ML
class gym():
    def __init__(self):
        # inputs
        self.state = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0) # x,y,heading,lr,la,rr,ra,dx,dy,dh
        self.msg0 = ''
        self.t0 = 0.0
        self.x0 = 0.0
        self.y0 = 0.0
        self.h0 = 0.0
        self.resetstate = self.state[:]
        # outputs
        self.lr0 = 0.0
        self.la0 = 0.0
        self.rr0 = 0.0
        self.ra0 = 0.0
        # actions
        drpm = list(range(-4,6))
        dang = [-1,0,1]
        self.action_list = list(itertools.product(drpm, dang, drpm, dang))
        self.tol = 1.0

    def step(self, action):
        # take action:
        lrpm_act, lang_act, rrpm_act, rang_act = action
        lrpm_act += self.lr0
        lang_act += self.la0
        rrpm_act += self.rr0
        rang_act += self.ra0
        cmd = f"$OBCMD,{int(lrpm_act)},{int(lang_act)},{int(rrpm_act)},{int(rang_act)}*3A"
        send_sock.sendto(cmd.encode(), (SEND_IP, SEND_PORT))
        #print("[Send]", cmd)

        # receive message     
        while True:
            try:
                data, _ = recv_sock.recvfrom(1024)
            except socket.timeout:
                print("[Warning] Timeout waiting for UDP.")
                os._exit(0)
                #return self.state, -10, True  # Penalize + end episode
            msg = data.decode().strip()
            if msg.startswith("$BSPOI"): 
                if msg == self.msg0:
                    continue
                else:
                    self.msg0 = msg
                    break
            
        # message process (normalize)
        _, t, x, y, hdg, lrn, lan, rrn, ran = msg.split(",")
        t = float(t)
        hdg = float(hdg)
        x = float(x)
        y = float(y)
        hdg = float(hdg)
        lrn = float(lrn)
        lan = float(lan)
        rrn = float(rrn)
        ran = float(ran[:-3])
        dt = max(t-self.t0, 1e-6)
        dx = x - self.x0
        dy = y - self.y0
        dh = hdg - self.h0
        if dh > 180.0:
            dh -= 360.0
        elif dh < -180.0:
            dh += 360.0
        # (normalized in state)
        self.state = (x / 25.0,
                      y, 
                      hdg / 180.0, 
                      lrn / 900.0, 
                      lan / 60.0, 
                      rrn / 900.0, 
                      ran / 60.0, 
                      dx / dt / 4, 
                      dy / dt / 4, 
                      dh / dt / 4)
        self.t0, self.x0, self.y0, self.h0 = t, x, y, hdg 
        self.lr0 = lrn
        self.la0 = lan
        self.rr0 = rrn
        self.ra0 = ran

        # reward, hyperparameters
        reward = (x + 0.1) / (1.0 + (x + 0.1) ** 2) ** 0.5 - abs(y) * x / (1.0 + abs(x)) - abs((hdg + 180) % 360 / 180.0 * 0.1)
        done = False
        if abs(y) > 5.0 or (5.0 < hdg and hdg < 355.0):
            reward -= 5.0  # Penalty for going out of bounds
            done = True
        elif x > 25.0 and abs(y) <= 5.0 and (hdg <= 5.0 or hdg >= 355.0):
            reward += 10  # Bonus for successful run
            done = True

        return self.state, reward, done

    def reset(self):
        print('[Reset]',self.state[:3])
        self.state = self.resetstate # xyh, (lrpm, lang, rrpm, rang), dxyh
        self.x0 = 0.0
        self.y0 = 0.0
        self.h0 = 0.0
        self.lr0 = 0.0
        self.la0 = 0.0
        self.rr0 = 0.0
        self.ra0 = 0.0
        #restart_button.click_input()
        c = "$Restart"
        send_sock.sendto(c.encode(), (SEND_IP, SEND_PORT))
        return self.state

    def render(self):
        temp_grid = np.copy(self.grid)
        x, y = self.current_state
        temp_grid[x, y] = 5 # Agent's position
        print(temp_grid)

# Create the environment
env = gym()

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
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
target_update_freq = 1000
memory_size = 10000
episodes = 1000

# Initialize Q-networks
input_dim = len(env.state)
output_dim = len(env.action_list)
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)

# Function to choose action using epsilon-greedy policy
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice(env.action_list) # Explore
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state)
            action_idx = torch.argmax(q_values).item()
            return env.action_list[action_idx]  # Exploit

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
    optimizer.step()

# --- Configuration ---
# Set the full path to your application's executable
APP_PATH = r"D:\\...\.exe"
# Set the application's working directory
APP_DIR = os.path.dirname(APP_PATH)

# 傳送控制封包
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Main
try:
    ###### Open the program
    # 1. Launch the application using subprocess
    print(f"Launching application with subprocess: {APP_PATH}")
    process = subprocess.Popen(APP_PATH, cwd=APP_DIR)
    process_id = process.pid
    print(f"Application launched successfully with PID: {process_id}")
    
    # Give the application time to fully initialize
    print("Waiting for the application to become ready...")
    time.sleep(5) # Adjust this delay if needed

    # 2. Connect to the application's process using pywinauto
    print(f"Connecting to application process with pywinauto...")
    timings.Timings.window_find_timeout = 20
    app_connected = Application(backend="win32").connect(process=process_id)

    # 3. Find the main window of the application
    main_window_title_regex = ""
    print(f"Searching for main window with title matching '{main_window_title_regex}'...")
    main_window = app_connected.window(title_re=main_window_title_regex)
    main_window.wait('ready', timeout=20)
    print(f"Successfully found the main window: '{main_window.wrapper_object().texts()[0]}'")
    # restart_button = main_window.child_window(auto_id="button_Restart", control_type="System.Windows.Forms.Button")

    ######## Training Loop
    # ML
    rewards_per_episode = []
    steps_done = 0
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Select action
            action = select_action(state, epsilon)
            next_state, reward, done = env.step(action)

            # Store transition in memory
            action_idx = env.action_list.index(action)
            memory.append((state, action_idx, reward, next_state, done))
            
            # Update state
            state = next_state
            episode_reward += reward
            
            # Optimize model
            optimize_model()

            # Update target network periodically
            if steps_done % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            steps_done += 1
        print(episode)

        # Decay epsilon
        if episode_reward > 0:
            epsilon = max(epsilon_min, epsilon_decay * epsilon)
        
        rewards_per_episode.append(episode_reward)

        if episode % 100 == 0:
            avg_reward = sum(rewards_per_episode[-100:]) / 100.0
            print(f"[LOG] Episode {episode} | Avg Reward (last 100): {avg_reward:.2f}")

        # Early Stopping Check
        if len(rewards_per_episode) >= rolling_window:
            avg_reward = sum(rewards_per_episode[-rolling_window:]) / rolling_window
            print(f"Episode {episode}: Avg Reward (last {rolling_window}) = {avg_reward:.2f}")
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            if best_avg_reward >= early_stop_threshold and early_stop_counter >= early_stop_patience:
                print(f"\n[Early Stopping] Reached average reward {best_avg_reward:.2f} "
                    f"with no improvement for {early_stop_patience} episodes.")
                break
    
    ###### End of training
    recv_sock.close()
    main_window.close()

except Exception as e:
    print(f"\nAn error occurred during automation: {e}")
    traceback.print_exc()
    # Ensure the subprocess is terminated in case of an error
    try:
        if 'process' in locals():
            os.kill(process.pid, signal.SIGTERM)
            #subprocess.Popen.kill(process) # Use kill() for forceful termination
    except Exception:
        pass

# Save the state_dict
MODEL_SAVE_PATH = "dqn_model.pth" # .pth is a common extension for PyTorch models
torch.save(policy_net.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

CSV_SAVE_PATH = "dqn_rewards.csv"
with open(CSV_SAVE_PATH, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Episode', 'Reward']) # Header row
    for i, reward in enumerate(rewards_per_episode):
        csv_writer.writerow([i + 1, reward]) # Episode numbers start from 1
print(f"Rewards saved to {CSV_SAVE_PATH}")

# Plotting the rewards per episode
if rewards_per_episode:
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Traning Rewards')
    plt.grid(True)
    plt.show()
else:
    print("[Warning] No rewards recorded — skipping plot.")
