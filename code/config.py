# config.py

# 训练的总轮数
num_epochs = 3000

# SAC 代理的其他超参数
gamma = 0.95
tau = 0.01
alpha = 0.2
learning_rate = 3e-4

# 环境相关参数
# 假设从 CSV 文件中取前多少个时间步的数据作为用电计划
num_time_steps = 100