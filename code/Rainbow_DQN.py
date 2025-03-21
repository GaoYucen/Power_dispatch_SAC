import joblib
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
from torch.distributions import Categorical

import random
from tqdm import tqdm

import config
import os
import argparse
import re
import math
from collections import deque

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
# Rainbow DQN Network with Dueling Architecture and Noisy Layers
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(input, weight, bias)


class RainbowDQN(nn.Module):
    def __init__(self, state_dim, action_dim, n_atoms=51, vmin=-10, vmax=10, hidden_dim=256, noisy=True):
        super(RainbowDQN, self).__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.vmin = vmin
        self.vmax = vmax
        self.support = torch.linspace(vmin, vmax, n_atoms).to(device)
        self.delta_z = (vmax - vmin) / (n_atoms - 1)

        # Feature layer
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Noisy layers or standard layers
        if noisy:
            # Value stream
            self.value_hidden = NoisyLinear(hidden_dim, hidden_dim)
            self.value = NoisyLinear(hidden_dim, n_atoms)

            # Advantage stream
            self.advantage_hidden = NoisyLinear(hidden_dim, hidden_dim)
            self.advantage = NoisyLinear(hidden_dim, action_dim * n_atoms)
        else:
            # Value stream
            self.value_hidden = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, n_atoms)

            # Advantage stream
            self.advantage_hidden = nn.Linear(hidden_dim, hidden_dim)
            self.advantage = nn.Linear(hidden_dim, action_dim * n_atoms)

    def forward(self, state):
        batch_size = state.size(0)

        features = self.features(state)

        value_hidden = F.relu(self.value_hidden(features))
        value = self.value(value_hidden).view(batch_size, 1, self.n_atoms)

        advantage_hidden = F.relu(self.advantage_hidden(features))
        advantage = self.advantage(advantage_hidden).view(batch_size, self.action_dim, self.n_atoms)

        # Combine value and advantage using dueling architecture
        q_distr = value + advantage - advantage.mean(dim=1, keepdim=True)

        # Get probabilities
        q_distr = F.softmax(q_distr, dim=2)

        return q_distr

    def reset_noise(self):
        if hasattr(self.value_hidden, 'reset_noise'):
            self.value_hidden.reset_noise()
            self.value.reset_noise()
            self.advantage_hidden.reset_noise()
            self.advantage.reset_noise()

    def get_q_values(self, states):
        """Compute Q values by using the support and probabilities"""
        q_distr = self.forward(states)
        support = self.support.expand_as(q_distr)
        q_values = torch.sum(support * q_distr, dim=2)  # Q(s,a) = sum_i(z_i * p_i)
        return q_values


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = uniform, 1 = full priority)
        self.beta = beta  # Importance sampling correction (0 = no correction, 1 = full correction)
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, n_step=1, gamma=0.95):
        if len(self.buffer) < batch_size:
            indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        else:
            priorities = self.priorities[:len(self.buffer)] ** self.alpha
            probabilities = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()

        # Beta annealing
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Extract samples from buffer
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]

            # Handle n-step returns if enabled
            if n_step > 1:
                reward, next_state, done = self._get_n_step_info(idx, n_step, gamma)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            np.array(weights, dtype=np.float32)
        )

    def _get_n_step_info(self, idx, n_step, gamma):
        """Get n-step return information"""
        reward, next_state, done = self.buffer[idx][2:5]

        for i in range(1, n_step):
            next_idx = (idx + i) % len(self.buffer)

            # If we loop back to the start or hit a done, stop
            if next_idx == self.position or self.buffer[next_idx - 1][4]:  # Check if previous step was done
                break

            reward += (gamma ** i) * self.buffer[next_idx][2]  # Add discounted reward
            next_state = self.buffer[next_idx][3]  # Update next state
            done = self.buffer[next_idx][4]  # Update done

        return reward, next_state, done

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)

class RainbowDQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.95, lr=3e-4,
                 buffer_size=10000, batch_size=64, n_step=3,
                 n_atoms=51, vmin=-10, vmax=10, hidden_dim=256):
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.vmin = vmin
        self.vmax = vmax
        self.support = torch.linspace(vmin, vmax, n_atoms).to(device)
        self.delta_z = (vmax - vmin) / (n_atoms - 1)

        # Initialize networks
        self.q_network = RainbowDQN(state_dim, action_dim, n_atoms, vmin, vmax, hidden_dim, noisy=True).to(device)
        self.target_network = RainbowDQN(state_dim, action_dim, n_atoms, vmin, vmax, hidden_dim, noisy=True).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Initialize replay buffer with prioritization
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=0.6, beta=0.4)

        # Epsilon for exploration (not needed if using noisy nets)
        self.epsilon = 0.05
        self.update_count = 0
        self.target_update_frequency = 1000

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        # During training occasionally pick random action
        if not self.q_network.training or random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.q_network.get_q_values(state)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_dim)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample(self.batch_size, n_step=self.n_step, gamma=self.gamma)

        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device).unsqueeze(1)
        weights = torch.FloatTensor(weights).to(device)

        # Get current Q distribution
        current_q_distribution = self.q_network(states)

        # Get the action-specific distribution
        actions = actions.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.n_atoms)
        current_q_distribution = current_q_distribution.gather(1, actions).squeeze(1)

        with torch.no_grad():
            # Compute greedy actions based on online network
            q_values = self.q_network.get_q_values(next_states)
            next_actions = q_values.max(1)[1]

            # Get next state distribution from target network
            next_q_distribution = self.target_network(next_states)
            next_actions = next_actions.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.n_atoms)
            next_q_distribution = next_q_distribution.gather(1, next_actions).squeeze(1)

            # Compute target distribution of discounted rewards + next state value
            gamma_n = self.gamma ** self.n_step
            rewards = rewards.expand(-1, self.n_atoms)
            dones = dones.expand(-1, self.n_atoms)
            support = self.support.unsqueeze(0).expand(self.batch_size, -1)

            tz = rewards + (1 - dones) * gamma_n * support
            tz = tz.clamp(min=self.vmin, max=self.vmax)

            # Project onto support
            b = (tz - self.vmin) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            # Distribute probability
            target_q_distribution = torch.zeros_like(current_q_distribution)

            # Handle projection for each atom
            for i in range(self.n_atoms):
                # Get batch of indices where the projected value lands
                l_idx = l[:, i]
                u_idx = u[:, i]

                # Get corresponding next state probability
                next_prob = next_q_distribution[:, i].unsqueeze(1)

                # Distribute probability to lower and upper indices
                target_q_distribution.scatter_add_(1, l_idx.unsqueeze(1),
                                                   next_prob * (u_idx.float() - b[:, i]).unsqueeze(1))
                target_q_distribution.scatter_add_(1, u_idx.unsqueeze(1),
                                                   next_prob * (b[:, i] - l_idx.float()).unsqueeze(1))

        # Calculate KL divergence loss with importance sampling weights
        log_probs = torch.log(current_q_distribution + 1e-10)
        loss = -(target_q_distribution * log_probs).sum(1)
        weighted_loss = (weights * loss).mean()

        # Calculate priorities for replay buffer
        with torch.no_grad():
            priorities = loss.abs().cpu().numpy() + 1e-6  # Small constant for stability

        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()

        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, priorities)

        # Reset noise for next forward pass
        self.q_network.reset_noise()
        self.target_network.reset_noise()

        # Periodically update target network
        self.update_count += 1
        if self.update_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

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
# Create the agent
agent = RainbowDQNAgent(state_dim, action_dim,
                        gamma=config.gamma,
                        batch_size=64,
                        buffer_size=10000,
                        n_step=3)

if mode == 'train':
    num_episodes = config.num_epochs
    os.makedirs('results', exist_ok=True)

    with open('results/RainbowDQN.txt', 'w') as f:
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

                # Update network multiple times per step (can adjust)
                agent.update()

                state = next_state
                total_reward += reward

            # Write reward to file
            f.write(f"Episode {episode}: Total Reward = {total_reward}\n")

            # Early stopping logic
            if total_reward > best_reward:
                best_reward = total_reward
                patience_counter = 0
                os.makedirs('param', exist_ok=True)
                torch.save(agent.q_network.state_dict(), f'param/best_rainbow_dqn_model_unit_{unit_id}.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at episode {episode}")
                break

            if episode % 100 == 0:
                print(f"Episode {episode}: Total Reward = {total_reward}")

elif mode == 'test':
    # Load best model
    agent.q_network.load_state_dict(torch.load(f'param/best_rainbow_dqn_model_unit_{unit_id}.pth'))
    agent.target_network.load_state_dict(agent.q_network.state_dict())
    agent.epsilon = 0.0  # No exploration during testing

    # Testing loop
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
