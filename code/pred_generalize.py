import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


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

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        identity2 = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = x + identity2

        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=1000, patience=50):
    best_loss = float('inf')
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.6f}")

    return best_state


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        y_pred = model(X_test_tensor).numpy()

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return mse, rmse, r2


def main():
    # 加载单位11的数据
    data = pd.read_csv('data/11.csv')
    data = data.drop(['YD15'], axis=1)
    data = data.dropna()
    data = data.head(10000)

    # 数据预处理
    X = data.drop(['DATATIME', 'PREPOWER', 'ROUND(A.POWER,0)'], axis=1)
    y = data['ROUND(A.POWER,0)']

    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # 测试不同训练数据量的效果
    train_sizes = [100, 500, 1000, 2000, 5000]
    results = {'meta': [], 'scratch': []}

    for train_size in train_sizes:
        print(f"\n使用 {train_size} 条训练数据进行对比:")

        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled,
            train_size=train_size,
            test_size=0.2,
            random_state=42
        )

        # 准备数据加载器
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=min(512, train_size),
            shuffle=True
        )

        input_size = X_train.shape[1]
        criterion = nn.MSELoss()

        # 1. 使用预训练模型(Meta-learning)进行快速适应
        print("\n1. Meta-learning模型(快速适应)")
        meta_model = NeuralNetwork(input_size)
        meta_model.load_state_dict(torch.load('param/best_meta_model.pt'))

        meta_optimizer = optim.AdamW(meta_model.parameters(), lr=0.001, weight_decay=0.005)
        meta_scheduler = optim.lr_scheduler.OneCycleLR(
            meta_optimizer, max_lr=0.003,
            epochs=100,  # 减少epoch数来测试快速适应能力
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )

        best_meta_state = train_model(
            meta_model, train_loader, criterion,
            meta_optimizer, meta_scheduler,
            num_epochs=100  # 减少训练轮数
        )

        meta_model.load_state_dict(best_meta_state)
        meta_mse, meta_rmse, meta_r2 = evaluate_model(meta_model, X_test, y_test)
        results['meta'].append((meta_mse, meta_rmse, meta_r2))

        # 2. 从零开始训练
        print("\n2. 从零训练模型")
        new_model = NeuralNetwork(input_size)

        new_optimizer = optim.AdamW(new_model.parameters(), lr=0.001, weight_decay=0.005)
        new_scheduler = optim.lr_scheduler.OneCycleLR(
            new_optimizer, max_lr=0.003,
            epochs=1000,  # 保持正常训练轮数
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )

        best_new_state = train_model(
            new_model, train_loader, criterion,
            new_optimizer, new_scheduler,
            num_epochs=100
        )

        new_model.load_state_dict(best_new_state)
        new_mse, new_rmse, new_r2 = evaluate_model(new_model, X_test, y_test)
        results['scratch'].append((new_mse, new_rmse, new_r2))

        print(f"\n训练集大小: {train_size}")
        print("Meta-learning模型结果:")
        print(f"MSE: {meta_mse:.6f}")
        print(f"RMSE: {meta_rmse:.6f}")
        print(f"R²: {meta_r2:.6f}")

        print("\n从零训练模型结果:")
        print(f"MSE: {new_mse:.6f}")
        print(f"RMSE: {new_rmse:.6f}")
        print(f"R²: {new_r2:.6f}")

        mse_improve = (new_mse - meta_mse) / new_mse * 100
        print(f"\nMSE提升: {mse_improve:.2f}%")

    # 绘制对比图
    import matplotlib.pyplot as plt

    import matplotlib as mpl
    mpl.rc('font', family='Arial Unicode MS')

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_sizes, [r[0] for r in results['meta']], 'bo-', label='Meta-learning')
    plt.plot(train_sizes, [r[0] for r in results['scratch']], 'ro-', label='从零训练')
    plt.xlabel('训练数据量')
    plt.ylabel('MSE')
    plt.title('不同训练数据量下的MSE对比')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_sizes, [r[2] for r in results['meta']], 'bo-', label='Meta-learning')
    plt.plot(train_sizes, [r[2] for r in results['scratch']], 'ro-', label='从零训练')
    plt.xlabel('训练数据量')
    plt.ylabel('R²')
    plt.title('不同训练数据量下的R²对比')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figure/比较结果.png')
    plt.close()


if __name__ == '__main__':
    main()