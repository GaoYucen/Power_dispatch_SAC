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

if args.mode == 'train':
    # 对每个组分别训练模型
    for group_idx, group_data in enumerate(groups):
        print(f"Training model for Group {group_idx + 1}...")

        # 合并组内数据
        combined_group_data = pd.concat(group_data)

        # 划分特征和目标变量
        X = combined_group_data.drop(['DATATIME', 'PREPOWER', 'ROUND(A.POWER,0)'], axis=1)
        y = combined_group_data['ROUND(A.POWER,0)']

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 转换为tensor
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

        # 初始化模型
        input_size = X_train.shape[1]
        model = NeuralNetwork(input_size)

        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)

        # 优化器和损失函数设置
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.005)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.003,
            epochs=3000,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        criterion = nn.MSELoss()

        # 训练循环
        best_loss = float('inf')
        patience = 50
        no_improve = 0

        model.train()
        for epoch in range(3000):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
                # 为每个组保存单独的模型
                torch.save(model.state_dict(), f'param/best_neural_network_group_{group_idx + 1}.pt')
            else:
                no_improve += 1

            if no_improve >= patience:
                print("Early stopping triggered")
                break

            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{3000}], Loss: {avg_loss:.6f}")

        print(f"Finished training model for Group {group_idx + 1}")
        print("-" * 50)

elif args.mode == 'test':
    # 对每个组分别进行测试
    for group_idx, group_data in enumerate(groups):
        print(f"Testing model for Group {group_idx + 1}...")

        # 合并组内数据
        combined_group_data = pd.concat(group_data)

        # 准备测试数据
        X = combined_group_data.drop(['DATATIME', 'PREPOWER', 'ROUND(A.POWER,0)'], axis=1)
        y = combined_group_data['ROUND(A.POWER,0)']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 加载对应组的模型
        input_size = X_test.shape[1]
        model = NeuralNetwork(input_size)
        model.load_state_dict(torch.load(f'param/best_neural_network_group_{group_idx + 1}.pt'))

        # 评估
        model.eval()
        with torch.no_grad():
            x_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
            y_pred = model(x_test_tensor).numpy().flatten()
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            print(f"Group {group_idx + 1} Results:")
            print(f"  Mean Squared Error (MSE): {mse}")
            print(f"  Root Mean Squared Error (RMSE): {rmse}")
            print(f"  R-squared (R^2): {r2}")
            print("-" * 50)