# SAC power dispatch

environment:
- python 3.11
- pytorch 2.11

data: https://gitcode.com/open-source-toolkit/e3850
- Time period: 2021/11/1-2022/6/30
- unit: 11-20

code
- config: parameter setting
- ~~pred_model: prediction model, not applicable current~~
- pred_one: train the prediction model for one unit
- pred_one_neural:
  - 添加了残差连接
  - 使用kaiming初始化 
  - 使用batch训练而不是全量训练 
  - 添加L2正则化 
  - 使用OneCycleLR学习率调度器 
  - 保存最佳模型状态
- pred_group_maml: cluster different groups, and train the prediction models for different groups using the meta-learning MAML, the base model is RandomForestRegressor
- pred_compare_base: use one model for all data
- pred_compare_group: use one model for each group
- pred_generalize: use the meta model for a new unit
- SAC: dispatch the power using SAC RL model

code logic:
- prediction: predict the power output for each unit
- meta-learning: to applied to different units
- SAC: to dispatch power, with predictor

experiment
- pred_one:

Linear Regression:
Mean Squared Error (MSE): 0.18258665944522745
Root Mean Squared Error (RMSE): 0.42730160243699933
R-squared (R^2): 0.8209365163335575
--------------------------------------------------
Decision Tree Regression:
Mean Squared Error (MSE): 0.05715829884822692
Root Mean Squared Error (RMSE): 0.23907801832922015
R-squared (R^2): 0.9439446225517839
--------------------------------------------------
Random Forest Regression:
Mean Squared Error (MSE): 0.025140886071411447
Root Mean Squared Error (RMSE): 0.15855877797022608
R-squared (R^2): 0.9753442302078015
--------------------------------------------------
XGBoost Regression:
Mean Squared Error (MSE): 0.030170369189386724
Root Mean Squared Error (RMSE): 0.17369619797044125
R-squared (R^2): 0.9704117955442692
--------------------------------------------------
LightGBM Regression:
Mean Squared Error (MSE): 0.03852035968269397
Root Mean Squared Error (RMSE): 0.19626604312181456
R-squared (R^2): 0.9622229257174361
--------------------------------------------------
neural network:
Mean Squared Error (MSE): 0.022295890963313567
Root Mean Squared Error (RMSE): 0.14931808652441794
R-squared (R^2): 0.9781343285458611
--------------------------------------------------
- pred_group:
  - 未分组Neural:
Mean Squared Error (MSE): 0.10455876165072131
Root Mean Squared Error (RMSE): 0.32335547258508135
R-squared (R^2): 0.8949083302041475
--------------------------------------------------
Evaluating Group 1...
  Mean Squared Error (MSE): 0.13038383573881077
  Root Mean Squared Error (RMSE): 0.36108701962104756
  R-squared (R^2): 0.8716905719932388
--------------------------------------------------
Evaluating Group 2...
  Mean Squared Error (MSE): 0.04626339767513828
  Root Mean Squared Error (RMSE): 0.21508927838257835
  R-squared (R^2): 0.9529859834990418
--------------------------------------------------
Evaluating Group 3...
  Mean Squared Error (MSE): 0.11267729622920039
  Root Mean Squared Error (RMSE): 0.33567439018966044
  R-squared (R^2): 0.8856566212500309
--------------------------------------------------
  - 分组neural： 
--------------------------------------------------
Testing model for Group 1...
Group 1 Results:
  Mean Squared Error (MSE): 0.08959165083526996
  Root Mean Squared Error (RMSE): 0.2993186443161701
  R-squared (R^2): 0.9101190467855984
--------------------------------------------------
Testing model for Group 2...
Group 2 Results:
  Mean Squared Error (MSE): 0.02219195614221068
  Root Mean Squared Error (RMSE): 0.14896964839258592
  R-squared (R^2): 0.9778131618518973
--------------------------------------------------
Testing model for Group 3...
Group 3 Results:
  Mean Squared Error (MSE): 0.0754087387733355
  Root Mean Squared Error (RMSE): 0.27460651626160565
  R-squared (R^2): 0.9236123357482131
  - 元学习Neural：

Group 1 Results:
MSE: 0.086893
RMSE: 0.294777
R²: 0.912826

Group 2 Results:
MSE: 0.019543
RMSE: 0.139797
R²: 0.980461

Group 3 Results:
MSE: 0.074799
RMSE: 0.273493
R²: 0.924230

- DQN:
  - A2C: -72000
  - SAC: -890
  - DQN: -760
  - Rainbow DQN: -890

  Analyze:
  - DQN performs best compared to other algorithms, this may be because:
    - The problem structure is relatively simple and lends itself to value function approximation methods
    - The state space is small (only 2 dimensions) and the actions are discrete, making simple models more efficient 
    - The reward signal is clear and unambiguous, which is beneficial for Q-learning 
  - SAC and Rainbow DQN performed on par, yielding similar results (-890), indicating that:
    - While they are state of the art algorithms, there is no obvious gain in additional complexity in this task 
    - Advanced features of Rainbow DQN such as distributed learning, noisy networks may be too complex
    - Entropy regularization and stochastic strategies of SAC may not be necessary for this task (SAC is better suited for continuous Spaces, the problem is discrete Spaces)
  - A2C performs the worst, with significantly lower performance than the other algorithms (-72000). Possible reasons:
    - Policy gradient methods may struggle to converge in sparse reward environments 
    - There is no experience replay mechanism to effectively utilize historical data 
    - More hyperparameter tuning and training time may be required