# StrengthenLearning - 强化学习算法实现

本项目是用于学习和实践经典强化学习算法的个人项目，在一个可配置的网格世界（Grid World）环境中，实现了包括价值迭代、策略迭代等多种算法。

## 核心算法

本项目当前实现了以下核心的强化学习算法：

- **价值迭代 (Value Iteration)**: 一个通过迭代贝尔曼最优方程来寻找最优价值函数，并最终推导出最优策略的算法。
- **策略迭代 (Policy Iteration)**: 一个在“策略评估”和“策略改进”之间交替进行，直至策略收敛到最优的算法。
- **截断策略迭代 (Truncated Policy Iteration)**: 策略迭代的一个变种，通过限制策略评估步骤的迭代次数来加速计算过程。

## 文件结构

```
.
├── .editorconfig               # 编辑器代码风格配置
├── .gitignore                  # Git 忽略文件配置
├── PolicyIteration.py          # 策略迭代算法实现
├── TruncatedPolicyIteration.py # 截断策略迭代算法实现
├── ValueIteration.py           # 价值迭代算法实现
├── config.json                 # 迷宫环境和算法参数配置
├── config.py                   # 加载和解析配置文件的工具
├── maze_utils.py               # 迷宫环境和MDP数组构建的辅助工具
└── README.md                   # 项目说明文档
```

## 如何运行

你可以直接运行对应的 Python 文件来启动相应的算法。算法会根据 `config.json` 中的配置来构建迷宫环境并执行。

```bash
# 运行价值迭代
python ValueIteration.py

# 运行策略迭代
python PolicyIteration.py

# 运行截断策略迭代
python TruncatedPolicyIteration.py
```

算法执行完毕后，将在控制台打印出最终的价值函数（V-table）和计算出的最优策略。

## 配置说明

所有的环境和算法参数都在 `config.json` 文件中进行配置。

- **`maze`**: 定义了网格世界的大小、起点、终点和障碍物。
- **`algorithms`**:
    - **`shared`**: 定义了所有算法共享的参数，如折扣因子 `gamma`、收敛阈值 `theta` 和最大迭代次数 `max_iterations`。
    - **`policy_iteration`**: 为策略迭代定义的特定参数，如策略评估阶段的最大迭代次数。
    - **`truncated_policy_iteration`**: 为截断策略迭代定义的特定参数，如截断的迭代次数 `truncation_k`。

你可以通过修改此文件来测试算法在不同环境和参数下的表现。