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
# 定义策略网络（Actor）
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_logits = self.fc3(x)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def get_probs(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return torch.softmax(action_logits, dim=-1)

# 定义价值网络（Critic）
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# 定义 SAC 代理类
class SACAgent:
    def __init__(self, state_dim, action_dim, gamma=0.95, tau=0.005, alpha=0.2, lr=3e-4, batch_size=256,
                 buffer_size=10000):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.action_dim = action_dim

        # Networks
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.q_net1 = ValueNetwork(state_dim, action_dim).to(device)
        self.q_net2 = ValueNetwork(state_dim, action_dim).to(device)
        self.target_q_net1 = ValueNetwork(state_dim, action_dim).to(device)
        self.target_q_net2 = ValueNetwork(state_dim, action_dim).to(device)

        # Copy weights to target networks
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            if evaluate:  # During evaluation, use most likely action
                action_probs = self.policy_net.get_probs(state)
                action = torch.argmax(action_probs).item()
            else:  # During training, sample from the distribution
                action, _ = self.policy_net(state)
                action = action.item()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self, state=None, action=None, reward=None, next_state=None, done=None):
        # Store transition if provided
        if state is not None:
            self.store_transition(state, action, reward, next_state, done)

        # Only update if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Update Q-networks
        with torch.no_grad():
            # Get probabilities for next actions
            next_action_probs = self.policy_net.get_probs(next_states)
            next_log_probs = torch.log(next_action_probs + 1e-10)  # Add small epsilon to avoid log(0)

            # Get Q values for next states from target networks
            next_q1_values = self.target_q_net1(next_states)
            next_q2_values = self.target_q_net2(next_states)
            next_q_values = torch.min(next_q1_values, next_q2_values)

            # Calculate expected Q values (soft value)
            expected_q_values = next_action_probs * (next_q_values - self.alpha * next_log_probs)
            expected_q_values = expected_q_values.sum(dim=1, keepdim=True)
            target_q_values = rewards + (1 - dones) * self.gamma * expected_q_values

        # Get current Q estimates
        current_q1_values = self.q_net1(states)
        current_q2_values = self.q_net2(states)

        # Select Q values for actions taken
        current_q1_values = current_q1_values.gather(1, actions.unsqueeze(1))
        current_q2_values = current_q2_values.gather(1, actions.unsqueeze(1))

        # Compute Q network losses
        q1_loss = nn.MSELoss()(current_q1_values, target_q_values.detach())
        q2_loss = nn.MSELoss()(current_q2_values, target_q_values.detach())

        # Update Q networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update policy network
        action_probs = self.policy_net.get_probs(states)
        log_probs = torch.log(action_probs + 1e-10)

        q1_values = self.q_net1(states)
        q2_values = self.q_net2(states)
        q_values = torch.min(q1_values, q2_values)

        # Calculate policy loss (maximize expected return with entropy regularization)
        inside_term = self.alpha * log_probs - q_values
        policy_loss = (action_probs * inside_term).sum(dim=1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update of target networks
        self._soft_update(self.q_net1, self.target_q_net1)
        self._soft_update(self.q_net2, self.target_q_net2)

    def _soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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
agent = SACAgent(state_dim, action_dim, gamma=config.gamma, tau=config.tau, alpha=config.alpha)

if mode == 'train':
    num_episodes = config.num_epochs
    os.makedirs('results', exist_ok=True)

    with open('results/SAC.txt', 'w') as f:
        best_reward = -float('inf')
        patience = 100
        patience_counter = 0

        for episode in tqdm(range(num_episodes)):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                # Update the agent multiple times per step
                for _ in range(1):  # You can adjust this number
                    agent.update()

            # Write the reward to the file
            f.write(f"Episode {episode}: Total Reward = {total_reward}\n")

            # Early stopping logic remains the same

            # Check for early stopping
            if total_reward > best_reward:
                best_reward = total_reward
                patience_counter = 0
                # Save the best model parameters
                os.makedirs('param', exist_ok=True)
                torch.save(agent.policy_net.state_dict(), f'param/best_policy_net_unit_{unit_id}.pth')
                torch.save(agent.q_net1.state_dict(), f'param/best_q_net1_unit_{unit_id}.pth')
                torch.save(agent.q_net2.state_dict(), f'param/best_q_net2_unit_{unit_id}.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at episode {episode}")
                break

            if episode % 100 == 0:
                print(f"Episode {episode}: Total Reward = {total_reward}")

elif mode == 'test':
    #%% 测试
    # Load the best model parameters
    agent.policy_net.load_state_dict(torch.load(f'param/best_policy_net_unit_{unit_id}.pth'))
    agent.q_net1.load_state_dict(torch.load(f'param/best_q_net1_unit_{unit_id}.pth'))
    agent.q_net2.load_state_dict(torch.load(f'param/best_q_net2_unit_{unit_id}.pth'))

    # Test the agent
    state = env.reset()
    done = False
    total_reward = 0
    actions = []

    while not done:
        action = agent.select_action(state, evaluate=True)
        actions.append(action)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

    print(f"Total Reward = {total_reward}")
    print(f"Action distribution: {np.sum(actions)}/{len(actions)} active periods")
