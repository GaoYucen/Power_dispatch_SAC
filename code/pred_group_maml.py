import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# 添加数据标准化
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import torch.nn.functional as F

# 用argparse解析命令行参数
import argparse
parser = argparse.ArgumentParser(description='Train or test the model.')
parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Mode: train or test')
args = parser.parse_args()  # Add this line to parse the arguments

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(64, 1)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 第一层
        identity = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        # 残差块
        identity2 = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = x + identity2  # 残差连接

        # 最后的层
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

# 修改后的元学习训练过程
class MAMLTrainer:
    def __init__(self, input_size):
        self.meta_model = NeuralNetwork(input_size)
        self.meta_lr = 0.001  # 降低元学习率
        self.inner_lr = 0.005  # 降低内循环学习率
        self.meta_optimizer = optim.AdamW(self.meta_model.parameters(), lr=self.meta_lr, weight_decay=0.001)
        self.criterion = nn.MSELoss()

        # 添加学习率调度器
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.meta_optimizer,
            max_lr=0.003,
            epochs=1000,
            steps_per_epoch=3 * meta_batch_size,  # 3个组 × 每组的batch数
            pct_start=0.3,
            anneal_strategy='cos'
        )

    def inner_loop(self, support_x, support_y, steps=10):  # 增加内循环步数
        fast_weights = {}
        for name, param in self.meta_model.named_parameters():
            fast_weights[name] = param.clone().detach().requires_grad_(True)

        # 添加内循环的学习率衰减
        inner_optimizer = optim.SGD([{'params': list(fast_weights.values())}], lr=self.inner_lr)
        # inner_scheduler = optim.lr_scheduler.ExponentialLR(inner_optimizer, gamma=0.95)

        for step in range(steps):
            pred = self.forward_with_weights(support_x, fast_weights)
            loss = self.criterion(pred, support_y)

            grads = torch.autograd.grad(loss, list(fast_weights.values()), create_graph=True)

            # 手动更新权重
            for (name, weight), grad in zip(fast_weights.items(), grads):
                fast_weights[name] = weight - self.inner_lr * grad
            # # 更新
            # inner_optimizer.zero_grad()
            # # 然后更新学习率
            # inner_scheduler.step()

        return fast_weights

    def forward_with_weights(self, x, weights):
        """使用指定权重进行前向传播"""
        # 第一层
        x = F.linear(x, weights['fc1.weight'], weights['fc1.bias'])
        x = F.batch_norm(x,
                         running_mean=None,
                         running_var=None,
                         weight=weights['bn1.weight'],
                         bias=weights['bn1.bias'],
                         training=True)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.meta_model.training)

        # 残差块
        identity = x
        x = F.linear(x, weights['fc2.weight'], weights['fc2.bias'])
        x = F.batch_norm(x,
                         running_mean=None,
                         running_var=None,
                         weight=weights['bn2.weight'],
                         bias=weights['bn2.bias'],
                         training=True)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.meta_model.training)
        x = x + identity

        # 第三层
        x = F.linear(x, weights['fc3.weight'], weights['fc3.bias'])
        x = F.batch_norm(x,
                         running_mean=None,
                         running_var=None,
                         weight=weights['bn3.weight'],
                         bias=weights['bn3.bias'],
                         training=True)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.meta_model.training)

        # 输出层
        x = F.linear(x, weights['fc4.weight'], weights['fc4.bias'])
        return x

# 定义数据文件路径和范围
data_folder = 'data'
file_range = range(12, 21)

# 检查数据文件夹是否存在
if not os.path.exists(data_folder):
    print(f"数据文件夹 {data_folder} 不存在，请检查路径。")
    exit(1)

# 加载所有数据文件
all_data = []
for i in file_range:
    file_path = os.path.join(data_folder, f'{i}.csv')
    if not os.path.exists(file_path):
        print(f"数据文件 {file_path} 不存在，请检查。")
        continue
    data = pd.read_csv(file_path)
    data = data.drop(['YD15'], axis=1)
    data = data.dropna()
    # 仅选取前10000个数据
    data = data.head(10000)
    all_data.append(data)

# 在数据加载后，分组前进行归一化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

for i in range(len(all_data)):
    # 分离特征和目标值
    X = all_data[i].drop(['DATATIME', 'PREPOWER', 'ROUND(A.POWER,0)'], axis=1)
    y = all_data[i]['ROUND(A.POWER,0)'].values.reshape(-1, 1)  # 转为2D数组

    # 分别对X和y进行归一化
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # 将归一化后的数据转回DataFrame
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    all_data[i].loc[:, X.columns] = X_scaled
    all_data[i]['ROUND(A.POWER,0)'] = y_scaled.flatten()

# 在数据加载和预处理后，首先进行分组
num_groups = 3
group_sizes = [len(all_data) // num_groups] * num_groups
remainder = len(all_data) % num_groups
for i in range(remainder):
    group_sizes[i] += 1

groups = []
start_index = 0
for size in group_sizes:
    end_index = start_index + size
    groups.append(all_data[start_index:end_index])
    start_index = end_index

# 训练循环参数优化
num_epochs = 1000
num_inner_steps = 10  # 增加内循环步数
meta_batch_size = 8   # 增加meta batch size
task_batch_size = 512 # 增加任务batch size

input_size = len(groups[0][0].drop(['DATATIME', 'PREPOWER', 'ROUND(A.POWER,0)'], axis=1).columns)
trainer = MAMLTrainer(input_size)
best_meta_loss = float('inf')
patience = 50
no_improve = 0

# 在训练循环前设置训练模式
trainer.meta_model.train()

for epoch in range(num_epochs):
    meta_loss = 0.0
    total_tasks = 0

    # 为每个组创建任务batch
    for group_idx, group in enumerate(groups):
        group_data = pd.concat(group)
        X = group_data.drop(['DATATIME', 'PREPOWER', 'ROUND(A.POWER,0)'], axis=1)
        y = group_data['ROUND(A.POWER,0)']

        # 创建多个支持集和查询集
        for _ in range(meta_batch_size):
            # 随机采样支持集和查询集
            support_indices = np.random.choice(len(X), task_batch_size, replace=False)
            query_indices = np.random.choice(
                list(set(range(len(X))) - set(support_indices)),
                task_batch_size,
                replace=False
            )

            support_x = torch.FloatTensor(X.iloc[support_indices].values)
            support_y = torch.FloatTensor(y.iloc[support_indices].values).reshape(-1, 1)
            query_x = torch.FloatTensor(X.iloc[query_indices].values)
            query_y = torch.FloatTensor(y.iloc[query_indices].values).reshape(-1, 1)

            # 内循环更新获取适应后的权重
            adapted_weights = trainer.inner_loop(support_x, support_y)

            # 使用适应后的权重在查询集上计算损失
            query_pred = trainer.forward_with_weights(query_x, adapted_weights)
            task_loss = trainer.criterion(query_pred, query_y) / task_batch_size
            meta_loss += task_loss
            total_tasks += 1

    # 反向传播和优化
    avg_meta_loss = meta_loss / total_tasks
    trainer.meta_optimizer.zero_grad()
    avg_meta_loss.backward()
    torch.nn.utils.clip_grad_norm_(trainer.meta_model.parameters(), max_norm=0.5)
    trainer.meta_optimizer.step()
    # 每个组结束后更新学习率
    trainer.scheduler.step()

    # 早停检查
    current_meta_loss = meta_loss.item()
    if current_meta_loss < best_meta_loss:
        best_meta_loss = current_meta_loss
        best_model_state = copy.deepcopy(trainer.meta_model.state_dict())
        # 保存最佳模型
        torch.save(best_model_state, 'param/best_meta_model.pt')
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Meta Loss: {current_meta_loss:.6f}")

# 对每个组进行微调和评估
for group_idx, group in enumerate(groups):
    # 加载元模型
    model = NeuralNetwork(input_size)
    model.load_state_dict(best_model_state)

    # 准备组数据
    group_data = pd.concat(group)
    X = group_data.drop(['DATATIME', 'PREPOWER', 'ROUND(A.POWER,0)'], axis=1)
    y = group_data['ROUND(A.POWER,0)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 转换为tensor
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)

    fine_tune_epoches = 3000

    # 微调
    fine_tune_optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.005)
    fine_tune_scheduler = optim.lr_scheduler.OneCycleLR(
        fine_tune_optimizer,
        max_lr=0.003,
        epochs=fine_tune_epoches,  # 增加微调轮数
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    model.train()
    # best_val_loss = float('inf')
    best_loss = float('inf')
    patience = 50
    no_improve = 0

    # # 划分验证集
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    for epoch in range(fine_tune_epoches):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            fine_tune_optimizer.zero_grad()
            pred = model(batch_X)
            loss = trainer.criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            fine_tune_optimizer.step()
            fine_tune_scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            # 保存微调后的模型
            torch.save(model.state_dict(), f'param/best_meta_model_group_{group_idx + 1}.pt')
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping triggered")
            break

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{3000}], Loss: {avg_loss:.6f}")

    print(f"Finished training model for Group {group_idx + 1}")
    print("-" * 50)

        # # 验证
        # model.eval()
        # with torch.no_grad():
        #     X_val_tensor = torch.FloatTensor(X_val.values)
        #     y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1)
        #     val_pred = model(X_val_tensor)
        #     val_loss = trainer.criterion(val_pred, y_val_tensor)
        #
        # # 早停检查
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_state = copy.deepcopy(model.state_dict())
        #     no_improve = 0
        # else:
        #     no_improve += 1
        #     if no_improve >= patience:
        #         break
        #
        # # 打印训练和验证损失
        # if epoch % 10 == 0:
        #     print(f"Group {group_idx + 1}, Epoch {epoch}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
        #
        # model.train()

# 对分组数据调用模型评估
for group_idx, group in enumerate(groups):
    # 准备测试数据
    group_data = pd.concat(group)
    X = group_data.drop(['DATATIME', 'PREPOWER', 'ROUND(A.POWER,0)'], axis=1)
    y = group_data['ROUND(A.POWER,0)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 加载微调后的模型
    input_size = X_test.shape[1]
    model = NeuralNetwork(input_size)
    model.load_state_dict(torch.load(f'param/best_meta_model_group_{group_idx + 1}.pt'))

    # 评估
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test.values)
        y_pred = model(X_test_tensor).numpy()

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\nGroup {group_idx + 1} Results:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²: {r2:.6f}")