import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from torch.distributions import Categorical

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
        logits = self.fc3(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


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


# 定义 SAC 代理类
class SACAgent:
    def __init__(self, state_dim, action_dim, gamma=0.95, tau=0.01, alpha=0.2):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # 初始化网络
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.q_net1 = ValueNetwork(state_dim, action_dim)
        self.q_net2 = ValueNetwork(state_dim, action_dim)
        self.target_q_net1 = ValueNetwork(state_dim, action_dim)
        self.target_q_net2 = ValueNetwork(state_dim, action_dim)

        # 复制目标网络参数
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        # 定义优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=3e-4)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=3e-4)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _ = self.policy_net(state)
        return action.detach().numpy()[0]

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)  # 确保输入为二维张量
        action = torch.LongTensor([action]).unsqueeze(0)  # 确保输入为二维张量
        reward = torch.FloatTensor([reward]).unsqueeze(0)  # 确保输入为二维张量
        next_state = torch.FloatTensor(next_state).unsqueeze(0)  # 确保输入为二维张量
        done = torch.FloatTensor([done]).unsqueeze(0)  # 确保输入为二维张量

        # 计算目标 Q 值
        with torch.no_grad():
            next_action, next_log_prob = self.policy_net(next_state)
            next_q1 = self.target_q_net1(next_state)
            next_q2 = self.target_q_net2(next_state)
            next_q = torch.min(next_q1, next_q2)
            # 确保 gather 操作的维度正确
            target_q = reward + (1 - done) * self.gamma * (next_q.gather(1, next_action.unsqueeze(1)) - self.alpha * next_log_prob.unsqueeze(1))

        # 更新 Q 网络
        q1 = self.q_net1(state).gather(1, action)
        q2 = self.q_net2(state).gather(1, action)
        q1_loss = nn.MSELoss()(q1, target_q)
        q2_loss = nn.MSELoss()(q2, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward(retain_graph=True)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # 更新策略网络
        new_action, log_prob = self.policy_net(state)
        q1_new = self.q_net1(state)
        q2_new = self.q_net2(state)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_prob.unsqueeze(1) - q_new.gather(1, new_action.unsqueeze(1))).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 软更新目标网络
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# 假设的用电计划
electricity_plan = [100, 200, 150, 300, 250]  # 每个时间步的用电量需求

# 假设的成本函数（开机成本+时间*发电成本）
def cost_function(action, time):
    startup_cost = 10
    generation_cost = 0.5
    return (startup_cost + generation_cost) * action


# 定义环境（简单示例）
class PowerDispatchEnv(gym.Env):
    def __init__(self, electricity_plan):
        self.electricity_plan = electricity_plan
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(2)  # 动作空间为 0 或 1
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return np.array([self.electricity_plan[self.current_step]])

    def step(self, action):
        demand = self.electricity_plan[self.current_step]
        cost = cost_function(action, self.current_step)
        generation_power = 300 if action == 1 else 0
        reward = -cost if generation_power >= demand else -1000  # 惩罚未满足用电需求
        self.current_step += 1
        done = self.current_step == len(self.electricity_plan)
        next_state = np.array([self.electricity_plan[self.current_step]] if not done else [0])
        return next_state, reward, done, {}


# 训练 SAC 代理
env = PowerDispatchEnv(electricity_plan)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = SACAgent(state_dim, action_dim)

num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    if episode % 100 == 0:
        print(f"Episode {episode}: Total Reward = {total_reward}")