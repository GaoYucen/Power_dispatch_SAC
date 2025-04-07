#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim

# ensure param file exists
if not os.path.exists('param'):
    os.makedirs('param')

#%% read the csv data 'data/11.csv'
data = pd.read_csv('data/12.csv')

# Removes the line containing NaN
data = data.dropna()
# 仅选取前10000个数据
data = data.head(10000)

#%%
# use different parameters to predict the ROUND
# No DATATIME, PREPOWER, YD15
X = data.drop(['DATATIME', 'PREPOWER', 'YD15', 'ROUND(A.POWER,0)'], axis=1)
y = data['ROUND(A.POWER,0)']

# 替换特征名中的特殊字符
X.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X.columns]

# 数据归一化
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_sacled = scaler_X.fit_transform(X)
y_sacled = scaler_y.fit_transform(y.values.reshape(-1, 1))
# 将归一化后的数据转回DataFrame
X = pd.DataFrame(X_sacled, columns=X.columns, index=X.index)
y = pd.Series(y_sacled.flatten(), index=data.index, name='ROUND(A.POWER,0)')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)


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

# 训练设置
input_size = X_train.shape[1]
model = NeuralNetwork(input_size)
batch_size = 512  # 增大batch size
num_epochs = 3000
base_lr = 0.001
max_lr = 0.003   # 使用更大的学习率范围
weight_decay = 0.005  # 减小L2正则化强度

# 创建数据加载器
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 优化器和学习率调度器
optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,    # 在30%的训练过程中达到最大学习率
    anneal_strategy='cos'
)
criterion = nn.MSELoss()

# 训练循环
best_loss = float('inf')
patience = 50
no_improve = 0

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        # 在训练循环中添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improve = 0
        # 保存最佳模型
        torch.save(model.state_dict(), 'param/best_neural_network_12..pt')
    else:
        no_improve += 1

    if no_improve >= patience:
        print("Early stopping triggered")
        break

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.6f}")

#%% 测试集评估
model.eval()
with torch.no_grad():
    x_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_pred = model(x_test_tensor).numpy().flatten()
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R^2): {r2}")
    # 存储测试集评估结果
    with open('results/pred_one_neural.txt', 'a') as f:
        f.write('neural network:\n')
        f.write(f"Mean Squared Error (MSE): {mse}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
        f.write(f"R-squared (R^2): {r2}\n")
        f.write('-' * 50 + '\n')

