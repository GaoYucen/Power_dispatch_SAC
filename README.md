# SAC电力调度

environment:
python 3.11
pytorch 2.11

data
- Time period: 2021/11/1-2022/6/30
- unit: 11-20

code
- config: parameter setting
- ~~pred_model: prediction model, not applicable current~~
- pred_one: train the prediction model for one unit
- pred_group: cluster different groups, and train the prediction models for different groups using the meta-learning MAML, the base model is RandomForestRegressor

experiment
- baselines:
- ablation study
  - no meta-learning, train one model for all units, report the results w/o meta-learning
