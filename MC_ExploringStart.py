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

def prepare_MC_ExploringStart_iteration(cfg_path: str = "config.json") -> Tuple:
    cfg = load_config(cfg_path)
    P, R, state_idx, action_idx, terminal_states = build_maze_mdp_arrays(cfg.maze)
    shared = cfg.algorithms.shared
    pi_extra = cfg.algorithms.policy_iteration
    print(f"Maze MDP loaded: |S|={len(state_idx)}, |A|={len(action_idx)}; terminals={len(terminal_states)}")
    print(
        f"MC_ExploringStartIteration params: gamma={shared.gamma}, theta={shared.theta}, max_iter={shared.max_iterations}, eval_max_iter={pi_extra.evaluation_max_iterations}"
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
    P, R, state_idx, action_idx, terminal_states, shared, pi_extra, cfg = prepare_MC_ExploringStart_iteration()
    v_table = np.zeros((cfg.maze.height, cfg.maze.width), dtype=float)                          # v_table 5 * 5 存stateValue
    q_table = np.zeros((cfg.maze.height * cfg.maze.width, len(action_idx)), dtype=float)        # q_table 25 * 5 存q_pi(s,a)
    a_table = np.zeros((cfg.maze.height, cfg.maze.width), dtype=int)                            # a_table 5 * 5 记录下旧的动作
    
    W = cfg.maze.width
    terminal_coords = {(int(s) // W, int(s) % W) for s in terminal_states}

    for coord, s in state_idx.items():
        r, c = coord[0], coord[1]
        v_table[r, c] = R[s, 4]
    
    gamma = shared.gamma 
    # print("Before MC :")
    # print(v_table)  
    # PE MC_ExploringStart Evaluation: Iterate until V-function for the current MC_ExploringStart converges
    initial_vtable = v_table.copy()
    action_names = {i: name for name, i in action_idx.items()}
    for coord, s in state_idx.items():
        r, c = coord[0], coord[1]
        cur_r, cur_c = r, c
        action_list = []
        episode_terminal = Tuple[int,int]
        v_old_for_eval = v_table.copy()
        for eval_iter in range(pi_extra.evaluation_max_iterations):
            temp_action = rand0_4()
            action_list.append(temp_action)
            dr, dc = idx_to_delta(temp_action)
            cur_r, cur_c = clamp(cur_r + dr, cur_c + dc, cfg.maze.height, cfg.maze.width)
            if (cur_r, cur_c) in terminal_coords:
                action_list.append(4)
                break
        # 一个坐标的 episode 完成
        episode_terminal = [cur_r, cur_c]
        need_add_val = 0
        # print(f"current coord: {r},{c}")
        action_strs = [ACTION_NAMES[a] for a in action_list]
        # print(f"action_list:{action_strs}")
        g = 0
        for i in range(len(action_list) - 1, -1, -1): 
            x = action_list[i]
            dr,dc = 0,0
            if x == 0:
                dr = 1
            if x == 1:
                dr = -1
            if x == 2:
                dc = 1
            if x == 3:
                dc = -1
            temp_r, temp_c = cur_r, cur_c
            cur_r, cur_c = clamp(cur_r + dr, cur_c + dc, cfg.maze.height, cfg.maze.width)
            before_g = g
            g = gamma * g + initial_vtable[temp_r, temp_c]
            # print(f"current i:{i} x:{x} g:{g} temp_r_c:{temp_r, temp_c} cur_r_c:{cur_r, cur_c} before_g:{before_g} v_table[cur_r, cur_c]:{initial_vtable[cur_r, cur_c]}")
        # print(f"need add val:{g / len(action_list)}")
        v_table[r, c] = g / len(action_list)
    print("After MC :")
    print(v_table)  

    # 针对初始策略开始迭代
    outer_iteration_count = 0
    while True:
        outer_iteration_count += 1

        # PI MC_ExploringStart Improvement: Greedily update MC_ExploringStart and check for stability
        MC_ExploringStart_stable = True
        for coord, s in state_idx.items():
            r, c = coord[0], coord[1]
            old_action = a_table[r, c]

            # Find the best action by looking one step ahead
            v_s_max = float('-inf')
            best_a = old_action
            for name, a in action_idx.items():
                dr, dc = ACTION_DELTAS_DEFAULT[name]
                next_r, next_c = r + dr, c + dc
                next_r, next_c = clamp(next_r, next_c, cfg.maze.height, cfg.maze.width)
                # CORRECTED: Use the just-evaluated v_table, not a stale v_old
                val = R[s, a] + shared.gamma * v_table[next_r, next_c]
                q_table[s, a] = val  # Store the Q-value
                if val > v_s_max:
                    v_s_max = val
                    best_a = a

            # Update the MC_ExploringStart and V-value for the current state
            a_table[r, c] = best_a
            v_table[r, c] = v_s_max
            
            # Check if the MC_ExploringStart has changed
            if best_a != old_action:
                MC_ExploringStart_stable = False

        # If the MC_ExploringStart is stable, we have found the optimal MC_ExploringStart.
        if MC_ExploringStart_stable:
            print(f"MC_ExploringStart stabilized after {outer_iteration_count} iterations.")
            break
        
        if outer_iteration_count >= shared.max_iterations:
            print(f"Max iterations ({shared.max_iterations}) reached, breaking loop.")
            break

    # Format and print the final V-table
    v_table_formatted = [[f"{v:.2f}" for v in row] for row in v_table]
    print("Final V-table (MC_ExploringStart Iteration):")
    for row in v_table_formatted:
        print(row)

    # Format and print the final Q-table
    q_table_formatted = [[f"{q:.2f}" for q in row] for row in q_table]
    print("\nFinal Q-table (MC_ExploringStart Iteration):")
    # Create a header for the Q-table
    action_names_header = [name for name, idx in sorted(action_idx.items(), key=lambda item: item[1])]
    print(f"{'State':<6} " + " ".join([f"{name:<5}" for name in action_names_header]))
    for s, row in enumerate(q_table_formatted):
        print(f"{s:<6} [" + " ".join([f"{val:<5}" for val in row]) + "]")


    # Format and print the final MC_ExploringStart (a_table)
    action_symbols = {
        action_idx["up"]: "↑",
        action_idx["down"]: "↓",
        action_idx["left"]: "←",
        action_idx["right"]: "→",
        action_idx["stay"]: "・",
    }
    MC_ExploringStart_formatted = [[action_symbols.get(a, "?") for a in row] for row in a_table]
    print("\nFinal MC_ExploringStart (Action Table):")
    for row in MC_ExploringStart_formatted:
        print(" ".join(row))


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nMC_ExploringStart Iteration took {elapsed_time:.4f} seconds to run.")
