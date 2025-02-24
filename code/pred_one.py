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

# ensure param file exists
if not os.path.exists('param'):
    os.makedirs('param')

#%% read the csv data 'data/11.csv'
data = pd.read_csv('data/11.csv')

# Removes the line containing NaN
data = data.dropna()

#%%
# use different parameters to predict the ROUND
# No DATATIME, PREPOWER, YD15
X = data.drop(['DATATIME', 'PREPOWER', 'YD15', 'ROUND(A.POWER,0)'], axis=1)
y = data['ROUND(A.POWER,0)']

# 替换特征名中的特殊字符
X.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X.columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% define the model list
models = [
    ('Linear Regression', LinearRegression()),
    ('Decision Tree Regression', DecisionTreeRegressor(random_state=42)),
    ('Random Forest Regression', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('XGBoost Regression', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
    ('LightGBM Regression', None)  # LightGBM 需要单独处理
]

lightgbm_params = {
    'objective': 'regression',
    'metric': 'mse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'random_state': 42
}

#%% train and evaluate each model
# 遍历每个模型进行训练和评估
for name, model in models:
    if name == 'LightGBM Regression':
        # 创建 LightGBM 数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(lightgbm_params, train_data, num_boost_round=100)
        y_pred = model.predict(X_test)
        # 保存 LightGBM 模型
        model.save_model(os.path.join('param', f'{name.replace(" ", "_")}.txt'))
    else:
        # 训练模型
        model.fit(X_train, y_train)
        # 进行预测
        y_pred = model.predict(X_test)
        # 保存模型
        joblib.dump(model, os.path.join('param', f'{name.replace(" ", "_")}.joblib'))

    # 评估模型
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # 输出评估结果
    print(f'{name}:')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R-squared (R^2): {r2}')
    print('-' * 50)

    # 选取前 100 个样本
    y_test_subset = y_test[:100]
    y_pred_subset = y_pred[:100]

    # 绘制预测结果可视化
    plt.scatter(y_test_subset, y_pred_subset)
    plt.xlabel('Actual Power')
    plt.ylabel('Predicted Power')
    plt.title(f'{name}: Actual vs Predicted Power (First 100 samples)')

    # 保存图像并使用 bbox_inches='tight' 去除多余空白
    filename = f'figure/{name.replace(" ", "_")}_prediction_first_100.png'
    plt.savefig(filename, bbox_inches='tight')

    # 显示图像
    plt.show()

