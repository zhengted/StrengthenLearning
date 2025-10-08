# -*- coding: utf-8 -*-
from math import gamma
from typing import Tuple

import numpy as np

from config import load_config, build_maze_mdp_arrays
from maze_utils import ACTION_DELTAS_DEFAULT, clamp
import os


ACTION_NAMES = ('up', 'down', 'left', 'right', 'stay')  # 0~4

def idx_to_action(idx: int) -> str:
    if 0 <= idx < len(ACTION_NAMES):
        return ACTION_NAMES[idx]
    raise IndexError(f"idx {idx} out of range")

def action_to_idx(name: str) -> int:
    try:
        return ACTION_NAMES.index(name)
    except ValueError:
        raise ValueError(f"unknown action {name}")

def idx_to_delta(idx: int) -> tuple[int, int]:
    return ACTION_DELTAS_DEFAULT[idx_to_action(idx)]

def prepare_MC_Basic_iteration(cfg_path: str = "config.json") -> Tuple:
    cfg = load_config(cfg_path)
    P, R, state_idx, action_idx, terminal_states = build_maze_mdp_arrays(cfg.maze)
    shared = cfg.algorithms.shared
    pi_extra = cfg.algorithms.policy_iteration
    print(f"Maze MDP loaded: |S|={len(state_idx)}, |A|={len(action_idx)}; terminals={len(terminal_states)}")
    print(
        f"MC_BasicIteration params: gamma={shared.gamma}, theta={shared.theta}, max_iter={shared.max_iterations}, eval_max_iter={pi_extra.evaluation_max_iterations}"
    )
    # Variable notes:
    # P: transition probability array, shape (S, A, S), P[s,a,s'] ∈ [0,1]
    # R: immediate reward array, shape (S, A), R[s,a]
    # state_idx: mapping from (r,c) to state index s
    # action_idx: mapping from action name to index a
    # terminal_states: terminal state indices (np.ndarray[int])
    # shared: shared algorithm params (gamma, theta, max_iterations)
    # cfg: full configuration (maze and algorithms)
    return P, R, state_idx, action_idx, terminal_states, shared, pi_extra, cfg

def rand0_4():
    n = 5
    t = 256 - (256 % n)  # 256 % 5 = 1，所以 t = 255
    while True:
        x = os.urandom(1)[0]
        if x < t:
            return x % n

if __name__ == "__main__":
    import time
    start_time = time.time()
    P, R, state_idx, action_idx, terminal_states, shared, pi_extra, cfg = prepare_MC_Basic_iteration()
    v_table = np.zeros((cfg.maze.height, cfg.maze.width), dtype=float)                          # v_table 5 * 5 存stateValue
    q_table = np.zeros((cfg.maze.height * cfg.maze.width, len(action_idx)), dtype=float)        # q_table 25 * 5 存q_pi(s,a)
    a_table = np.zeros((cfg.maze.height, cfg.maze.width), dtype=int)                            # a_table 5 * 5 记录下旧的动作
    
    W = cfg.maze.width
    terminal_coords = {(int(s) // W, int(s) % W) for s in terminal_states}

    for coord, s in state_idx.items():
        r, c = coord[0], coord[1]
        v_table[r, c] = R[s, 4]

    # PE MC_Basic Evaluation: Iterate until V-function for the current MC_Basic converges
    gamma = shared.gamma
    action_names = {i: name for name, i in action_idx.items()}
    idx_to_coord = {s: coord for coord, s in state_idx.items()}
    # v_old_for_eval = v_table.copy()

    # 针对每个s
    for coord, s in state_idx.items():
        r, c = coord[0], coord[1]
        cur_r, cur_c = r, c
        
        # 针对每个s下的每个a
        for name, a in action_idx.items():
            ar, ac = idx_to_delta(a)
            action_dir_r, action_dir_c = clamp(cur_r + ar, cur_c + ac, cfg.maze.height, cfg.maze.width)
            # 当前s下执行a得到的价值 q_table的初值
            # 迭代若干步
            for eval_iter in range(pi_extra.evaluation_max_iterations):
                episode_length = cfg.algorithms.mc_basic.episode_length - 1
                episodes = []
                episodes.append(a)
                for step in range(episode_length):
                    # 从s,a 之后任意选择一个方向
                    temp_action = rand0_4()
                    # 得到这个方向的delta
                    dr, dc = idx_to_delta(temp_action)
                    # 开始从这个方向叠加价值
                    action_dir_r, action_dir_c = clamp(action_dir_r + dr, action_dir_c + dc, cfg.maze.height, cfg.maze.width)

                    # 把当前的记录记到episodes中
                    episodes.append(temp_action)
                # 这里得到一个样本 episode 针对当前的样本计算q(s, a)
                temp = 0
                current_r, current_c = r, c
                current_s = s
               
                for idx, action in enumerate(episodes):
                    episode_r, episode_c = idx_to_delta(action)
                    current_r, current_c = clamp(current_r + episode_r, current_c + episode_c, cfg.maze.height, cfg.maze.width)
                    temp += (gamma ** idx) * R[current_s, action]
                    if not (current_s in terminal_states):
                        current_s = state_idx[current_r, current_c]
                q_table[s, a] += temp / (pi_extra.evaluation_max_iterations)
    print("After PE")
     # Format and print the final Q-table
    q_table_formatted = [[f"{q:.2f}" for q in row] for row in q_table]
    print("\n Q-table (MC_Basic Iteration):")
    # Create a header for the Q-table
    action_names_header = [name for name, idx in sorted(action_idx.items(), key=lambda item: item[1])]
    print(f"{'State':<6} " + " ".join([f"{name:<5}" for name in action_names_header]))
    for s, row in enumerate(q_table_formatted):
        print(f"{s:<6} [" + " ".join([f"{val:<5}" for val in row]) + "]")

    # 针对初始策略开始迭代
    outer_iteration_count = 0
    while True:
        outer_iteration_count += 1

        # PI MC_Basic Improvement: Greedily update MC_Basic and check for stability
        MC_Basic_stable = True
        for coord, s in state_idx.items():
            r, c = coord[0], coord[1]
            old_action = a_table[r, c]
            # Find the best action by looking one step ahead
            v_s_max = float('-inf')
            best_a = old_action
            for name, a in action_idx.items():
                val = q_table[s, a]  # Store the Q-value
                if val > v_s_max:
                    v_s_max = val
                    best_a = a

            # Update the MC_Basic and V-value for the current state
            a_table[r, c] = best_a
            v_table[r, c] = v_s_max
            
            # Check if the MC_Basic has changed
            if best_a != old_action:
                MC_Basic_stable = False

        # If the MC_Basic is stable, we have found the optimal MC_Basic.
        if MC_Basic_stable:
            print(f"MC_Basic stabilized after {outer_iteration_count} iterations.")
            break
        
        if outer_iteration_count >= shared.max_iterations:
            print(f"Max iterations ({shared.max_iterations}) reached, breaking loop.")
            break

    # Format and print the final V-table
    v_table_formatted = [[f"{v:.2f}" for v in row] for row in v_table]
    print("Final V-table (MC_Basic Iteration):")
    for row in v_table_formatted:
        print(row)

    # Format and print the final Q-table
    q_table_formatted = [[f"{q:.2f}" for q in row] for row in q_table]
    print("\nFinal Q-table (MC_Basic Iteration):")
    # Create a header for the Q-table
    action_names_header = [name for name, idx in sorted(action_idx.items(), key=lambda item: item[1])]
    print(f"{'State':<6} " + " ".join([f"{name:<5}" for name in action_names_header]))
    for s, row in enumerate(q_table_formatted):
        print(f"{s:<6} [" + " ".join([f"{val:<5}" for val in row]) + "]")


    # Format and print the final MC_Basic (a_table)
    action_symbols = {
        action_idx["up"]: "↑",
        action_idx["down"]: "↓",
        action_idx["left"]: "←",
        action_idx["right"]: "→",
        action_idx["stay"]: "・",
    }
    MC_Basic_formatted = [[action_symbols.get(a, "?") for a in row] for row in a_table]
    print("\nFinal MC_Basic (Action Table):")
    for row in MC_Basic_formatted:
        print(" ".join(row))


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nMC_Basic Iteration took {elapsed_time:.4f} seconds to run.")
