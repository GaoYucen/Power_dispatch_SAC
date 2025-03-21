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
- pred_group: cluster different groups, and train the prediction models for different groups using the meta-learning MAML, the base model is RandomForestRegressor
- SAC: dispatch the power using SAC RL model

code logic:
- prediction: predict the power output for each unit
- meta-learning: to applied to different units
- SAC: to dispatch power, with predictor

experiment
- baselines:
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
- ablation study
  - no meta-learning, train one model for all units, report the results w/o meta-learning