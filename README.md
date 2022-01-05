# Exploring Beyond-Demonstrator via Meta Learning-Based Reward Extrapolation

- Run the following command to get dependencies:
```shell
pip install -r requirements.txt
```

- Train demonstrators:
```shell
sh scripts/train_experts.sh
```

- Generate demonstrations:
```shell
sh scripts/generate_trajs.sh
```

- Learn reward function:
```shell
python LearnAtariReward.py
```

- Learn policies using the inferred reward function:
```shell
python PolicyOptimize.py
```
