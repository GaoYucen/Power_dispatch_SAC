import joblib
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import gym
from torch.distributions import Categorical

import random
from tqdm import tqdm

import config
import os
import argparse
import re

# 检查CUDA和MPS是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")

print('Using device:', device)


# Load the trained prediction model
def load_prediction_model(unit_id=11):
    model_path = f'param/pred_model_{unit_id}.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        print(f"Warning: Prediction model {model_path} not found")
        return None

# Use the prediction model to get power predictions
def get_power_predictions(model, data, unit_id=11):
    if model is None:
        # If no model is available, return zeros or some default value
        return np.zeros(len(data))

    # Prepare input features (excluding target variables and time)
    X = data.drop(['DATATIME', 'PREPOWER', 'YD15', 'ROUND(A.POWER,0)'], axis=1)
    # 替换特征名中的特殊字符
    X.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X.columns]

    # Predict power output
    predicted_power = model.predict(X)
    return predicted_power

#%%
# 定义DQN网络
class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.95, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, learning_rate=3e-4, update_target_freq=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_target_freq = update_target_freq
        self.learning_rate = learning_rate
        self.update_count = 0

        # Main Q-network
        self.q_network = DQNNetwork(state_dim, action_dim).to(device)
        # Target Q-network
        self.target_network = DQNNetwork(state_dim, action_dim).to(device)
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Experience replay buffer (could be implemented, simplified here)
        self.memory = []
        self.batch_size = 64
        self.max_memory_size = 10000

    def select_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def store_transition(self, state, action, reward, next_state, done):
        # Store experience in replay buffer
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)

    def update(self, state=None, action=None, reward=None, next_state=None, done=None):
        # Store transition in memory
        if state is not None:
            self.store_transition(state, action, reward, next_state, done)

        # Skip update if not enough samples
        if len(self.memory) < self.batch_size:
            return

        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Get current Q-values
        current_q_values = self.q_network(states).gather(1, actions)

        # Get target Q-values
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.update_epsilon()

# 假设的成本函数（开机成本+时间*发电成本）
def cost_function(action, time):
    startup_cost = 10
    generation_cost = 0
    return (startup_cost + generation_cost) * action

# def cost_function(action, time, power_level=None):
#     startup_cost = 10 if action == 1 else 0
#     generation_cost = 0.05 * power_level if power_level is not None and action == 1 else 0
#     return startup_cost + generation_cost

# 定义环境（简单示例）
class PowerDispatchEnv(gym.Env):
    def __init__(self, electricity_plan, predicted_power):
        self.electricity_plan_original = electricity_plan
        self.predicted_power = predicted_power

        # 将electricity_plan中随机取0
        np.random.seed(42)  # 设置随机种子以确保结果可复现
        self.electricity_plan = np.array(electricity_plan)
        num_zeros = int(0.1 * len(self.electricity_plan))  # 假设将 10% 的数值设为 0
        zero_indices = np.random.choice(len(self.electricity_plan), num_zeros, replace=False)
        self.electricity_plan[zero_indices] = 0

        self.current_step = 0
        self.action_space = gym.spaces.Discrete(2)  # 动作空间为 0 或 1
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return np.array([
            self.electricity_plan[self.current_step],
            self.predicted_power[self.current_step]
        ])

    def step(self, action):
        demand = self.electricity_plan[self.current_step]
        predicted_power = self.predicted_power[self.current_step]

        cost = cost_function(action, self.current_step)

        generation_power = self.electricity_plan_original[self.current_step] if action == 1 else 0

        reward = -cost if generation_power >= demand else -1000  # 惩罚未满足用电需求
        self.current_step += 1
        done = self.current_step == len(self.electricity_plan)
        if not done:
            next_state = np.array([
                self.electricity_plan[self.current_step],
                self.predicted_power[self.current_step]
            ])
        else:
            next_state = np.array([0, 0])
        return next_state, reward, done, {}

#%% 解析命令行参数
parser = argparse.ArgumentParser(description='SAC power dispatch')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode to run the script in')
parser.add_argument('--unit_id', type=int, default=11, help='Power unit ID (11-20)')
args = parser.parse_args()

mode = args.mode

# 固定随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 如果使用了 CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Load data
unit_id = args.unit_id
data = pd.read_csv(f'data/{unit_id}.csv')
data = data.dropna()

# 取前 num_time_steps 个时间步的数据作为用电计划
electricity_plan = data['ROUND(A.POWER,0)'].values[:config.num_time_steps]
# 将负值替换为 0
electricity_plan = np.where(electricity_plan < 0, 0, electricity_plan)

#%% Load prediction model and get predictions
pred_model = load_prediction_model(unit_id)
predicted_power = get_power_predictions(pred_model, data[:config.num_time_steps], unit_id)

# # 使用前5步的平均值作为当前步的预测值构造predicted_power
# predicted_power = np.zeros_like(electricity_plan)
# predicted_power[:5] = electricity_plan[:5].mean()
# for i in range(5, len(predicted_power)):
#     predicted_power[i] = electricity_plan[i - 5:i].mean()
# print(predicted_power)

#%%
# 训练 SAC 代理
env = PowerDispatchEnv(electricity_plan, predicted_power)
# 统计一下有多少个0
print(np.sum(env.electricity_plan == 0))



#%%
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim, gamma=config.gamma)

if mode == 'train':
    num_episodes = config.num_epochs
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Open a file to write the rewards
    with open('results/DQN.txt', 'w') as f:
        best_reward = -float('inf')
        patience = 100  # Number of episodes to wait for improvement
        patience_counter = 0

        for episode in tqdm(range(num_episodes)):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            # Write the reward to the file
            f.write(f"Episode {episode}: Total Reward = {total_reward}\n")

            # Check for early stopping
            if total_reward > best_reward:
                best_reward = total_reward
                patience_counter = 0
                # Save the best model parameters
                os.makedirs('param', exist_ok=True)
                torch.save(agent.q_network.state_dict(), f'param/best_dqn_model_unit_{unit_id}.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at episode {episode}")
                break

            if episode % 100 == 0:
                print(f"Episode {episode}: Total Reward = {total_reward}")

elif mode == 'test':
    # Load the best model parameters
    agent.q_network.load_state_dict(torch.load(f'param/best_dqn_model_unit_{unit_id}.pth'))
    agent.target_network.load_state_dict(agent.q_network.state_dict())
    agent.epsilon = 0.0  # No exploration during testing

    # Test the agent
    state = env.reset()
    done = False
    total_reward = 0
    actions = []

    while not done:
        action = agent.select_action(state)
        actions.append(action)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

    print(f"Total Reward = {total_reward}")
    print(f"Action distribution: {np.sum(actions)}/{len(actions)} active periods")
