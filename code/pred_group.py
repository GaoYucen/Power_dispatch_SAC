#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import copy
import joblib
import os

#%%
# 定义数据文件路径和范围
data_folder = 'data'
file_range = range(11, 21)

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

#%%
# 数据分组
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

#%%
# MAML 参数
num_inner_updates = 2  # 内循环更新步数
inner_lr = 0.01  # 内循环学习率
outer_lr = 0.001  # 外循环学习率
epoches = 2

# 初始化元模型
meta_model = RandomForestRegressor(random_state=42)

# 用于存储每个分组的内层模型
inner_models = []

#%%
# MAML 训练过程
for epoch in range(epoches):  # 外循环轮数
    meta_gradients = []

    for group_index, group in enumerate(groups):
        print(f"Training and evaluating Group {group_index + 1}...")
        all_X_train = []
        all_y_train = []

        # 合并组内所有数据集的训练集
        for data in group:
            X = data.drop(['DATATIME', 'PREPOWER', 'ROUND(A.POWER,0)'], axis=1)
            y = data['ROUND(A.POWER,0)']
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            all_X_train.append(X_train)
            all_y_train.append(y_train)

        # 合并所有训练数据
        combined_X_train = pd.concat(all_X_train)
        combined_y_train = pd.concat(all_y_train)

        # 复制元模型
        fast_model = copy.deepcopy(meta_model)

        # 内循环更新
        for _ in range(num_inner_updates):
            fast_model.fit(combined_X_train, combined_y_train)

        # 这里简单模拟梯度，实际中对于随机森林无法精确计算梯度，只是概念上的更新
        # 可以考虑使用代理模型或其他方法来近似梯度
        grad = np.random.randn() * outer_lr  # 简单模拟梯度更新

        meta_gradients.append(grad)

        # 保存当前分组的内层模型
        if epoch == epoches - 1:  # 仅在最后一个 epoch 保存内层模型
            inner_models.append(fast_model)

    # 外循环更新元模型（简单平均梯度更新）
    meta_gradient = np.mean(meta_gradients)
    # 这里无法直接更新随机森林的参数，仅作示意
    # 实际中可能需要重新训练或使用更复杂的方法
    # meta_model.update(meta_gradient)

# 存储外层元模型
joblib.dump(meta_model, 'param/meta_model.joblib')

# 存储每个分组的内层模型
for i, inner_model in enumerate(inner_models):
    joblib.dump(inner_model, f'param/inner_model_group_{i + 1}.joblib')

#%%
# 模型评估
# 读取内层模型参数
inner_models = []
for i in range(num_groups):
    inner_model = joblib.load(f'param/inner_model_group_{i + 1}.joblib')
    inner_models.append(inner_model)

for group_index, group in enumerate(groups):
    print(f"Evaluating Group {group_index + 1}...")
    all_X_test = []
    all_y_test = []

    # 合并组内所有数据集的测试集
    for data in group:
        X = data.drop(['DATATIME', 'PREPOWER', 'ROUND(A.POWER,0)'], axis=1)
        y = data['ROUND(A.POWER,0)']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        all_X_test.append(X_test)
        all_y_test.append(y_test)

    # 合并所有测试数据
    combined_X_test = pd.concat(all_X_test)
    combined_y_test = pd.concat(all_y_test)

    # 使用内层模型进行评估
    inner_model = inner_models[group_index]
    y_pred = inner_model.predict(combined_X_test)
    mse = mean_squared_error(combined_y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(combined_y_test, y_pred)

    print(f"  Mean Squared Error (MSE): {mse}")
    print(f"  Root Mean Squared Error (RMSE): {rmse}")
    print(f"  R-squared (R^2): {r2}")
    print("-" * 50)