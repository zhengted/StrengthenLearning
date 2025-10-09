# StrengthenLearning - 强化学习算法实现

本项目用于学习和实践经典强化学习算法，在一个可配置的网格世界（Grid World）中实现价值迭代、策略迭代、蒙特卡洛等算法。

## 核心算法

当前包含：
- **价值迭代 (Value Iteration)**
- **策略迭代 (Policy Iteration)**
- **截断策略迭代 (Truncated Policy Iteration)**
- **蒙特卡洛 - 基础版 (MC_Basic)**
- **蒙特卡洛 - ε-贪心 (MC_EpsilonGreedy)**
- **蒙特卡洛 - 探索起点 (MC_ExploringStart)**

## 文件结构

```
.
├── .editorconfig
├── .gitignore
├── README.md
├── Value_vs_Policy_Iteration.md
├── algorithms\
│   ├── __init__.py
│   ├── iteration\
│   │   ├── PolicyIteration.py
│   │   ├── TruncatedPolicyIteration.py
│   │   ├── ValueIteration.py
│   │   └── __init__.py
│   └── mc\
│       ├── MC_Basic.py
│       ├── MC_EpsilonGreedy.py
│       ├── MC_ExploringStart.py
│       └── __init__.py
├── bellman_equation_derivation.md
└── core\
    ├── __init__.py
    ├── config.json              # 迷宫环境与算法参数
    ├── config.py                # 配置加载与MDP数组构建
    └── maze_utils.py            # 迷宫动作与坐标工具
```

## 运行方式

推荐用“模块方式”运行（无需关心 `PYTHONPATH`）：

```bash
# 迭代类
python -m algorithms.iteration.ValueIteration
python -m algorithms.iteration.PolicyIteration
python -m algorithms.iteration.TruncatedPolicyIteration

# 蒙特卡洛类
python -m algorithms.mc.MC_Basic
python -m algorithms.mc.MC_EpsilonGreedy
python -m algorithms.mc.MC_ExploringStart
```

也支持直接脚本运行（已在各脚本顶部注入项目根路径到 `sys.path`）：

```bash
python algorithms/iteration/ValueIteration.py
python algorithms/iteration/PolicyIteration.py
python algorithms/iteration/TruncatedPolicyIteration.py
python algorithms/mc/MC_Basic.py
python algorithms/mc/MC_EpsilonGreedy.py
python algorithms/mc/MC_ExploringStart.py
```

算法执行后会在控制台打印 V 表（价值函数）、Q 表（动作价值）以及最终动作策略。

## 配置说明

- 默认配置位于 `core/config.json`，`core.config.load_config()` 会默认加载该文件。
- 如需自定义配置，可复制并编辑该 JSON，再在调用处传入路径：
  - 模块方式示例（在脚本内）：`load_config("config_custom.json")`（相对路径相对于 `core/`）
- 配置键说明：
  - `maze`: 网格大小、禁区、目标、奖励、动作集、终止吸收等。
  - `algorithms.shared`: `gamma` 折扣因子、`theta` 收敛阈值、`max_iterations` 最大迭代数。
  - `algorithms.policy_iteration`: `evaluation_max_iterations` 策略评估最大迭代数。
  - `algorithms.truncated_policy_iteration`: `truncation_k` 截断步数。
  - `algorithms.mc_basic`: `episode_length` 蒙特卡洛基础版每个 episode 的步数。

## 编码与兼容性

- 源码统一 UTF-8 编码；为避免解码问题，箭头字符改为 Unicode 转义（`\u2191`、`\u2193`、`\u2190`、`\u2192`）。
- 兼容 Python 3.8+：类型注解中的 `Tuple[int, int]` 已统一使用 `typing.Tuple` 写法。

## 开发提示

- 若在 Notebook 中运行，可在首个单元加入：
  ```python
  import sys
  sys.path.insert(0, r'E:\Algorithm\MyStrengthenLearning')
  ```
- 或在 PowerShell 中设置：`$env:PYTHONPATH = "E:\Algorithm\MyStrengthenLearning"`。