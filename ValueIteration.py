# -*- coding: utf-8 -*-
from pydoc import doc
from typing import Tuple
from dataclasses import asdict
import json

from numpy import dtype, int32

from config import load_config, build_maze_mdp_arrays
from maze_utils import ACTION_DELTAS_DEFAULT, clamp


def prepare_value_iteration(cfg_path: str = "config.json") -> Tuple:
    cfg = load_config(cfg_path)
    P, R, state_idx, action_idx, terminal_states = build_maze_mdp_arrays(cfg.maze)
    shared = cfg.algorithms.shared
    print(f"Maze MDP loaded: |S|={len(state_idx)}, |A|={len(action_idx)}; terminals={len(terminal_states)}")
    print(
        f"ValueIteration params: gamma={shared.gamma}, theta={shared.theta}, max_iter={shared.max_iterations}"
    )
    # Variable notes:
    # P: transition probability array, shape (S, A, S), P[s,a,s'] ∈ [0,1]
    # R: immediate reward array, shape (S, A), R[s,a]
    # state_idx: mapping from (r,c) to state index s
    # action_idx: mapping from action name to index a
    # terminal_states: terminal state indices (np.ndarray[int])
    # shared: shared algorithm params (gamma, theta, max_iterations)
    # cfg: full configuration (maze and algorithms)
    return P, R, state_idx, action_idx, terminal_states, shared, cfg
        

def build_value_and_action_grids(v_table, q_table, action_idx, cfg, terminal_set=None):
    H = cfg.maze.height
    W = cfg.maze.width
    v_grid = [[float(v_table[r * W + c]) for c in range(W)] for r in range(H)]
    print(f"\nV ({H}x{W}):")
    for row in v_grid:
        print([f"{val:.2f}" for val in row])

    idx_to_action = {idx: name for name, idx in action_idx.items()}
    actions_grid = [[None for _ in range(W)] for _ in range(H)]
    for s in range(H * W):
        r, c = divmod(s, W)
        row = q_table[s]
        if terminal_set is not None and s in terminal_set:
            actions_grid[r][c] = "-"
            continue
        best_a = max(range(len(row)), key=lambda a: row[a])
        actions_grid[r][c] = idx_to_action.get(best_a, str(best_a))

    return H, W, v_grid, actions_grid


if __name__ == "__main__":
    import numpy as np
    import time

    start_time = time.time()
    # 读配置
    P, R, state_idx, action_idx, terminal_states, shared, cfg = prepare_value_iteration()

    # q_table stores action values v(s,a)
    q_table = np.zeros((cfg.maze.height * cfg.maze.width, len(action_idx)), dtype=float)
    v_table = np.zeros((cfg.maze.height * cfg.maze.width), dtype=float)

    # Maze target states: fixed action 'stay'; reward handled accordingly
    W = cfg.maze.width
    H = cfg.maze.height
    terminal_coords = [(int(s) // W, int(s) % W) for s in terminal_states]

    # For terminal states, compute V(s) with 'stay' action
    gamma = shared.gamma

    # 遍历状态
    # Precompute terminal set to avoid ambiguous ndarray equality
    # 这个作用就是把终点所在的索引放进一个set中
    terminal_set = set(int(x) for x in terminal_states.flatten() if hasattr(terminal_states, "flatten")) if hasattr(terminal_states, "__iter__") else {int(terminal_states)}

    iteration_count = 0
    while iteration_count < shared.max_iterations:
        iteration_count += 1
        delta = 0.0
        for coord, s in state_idx.items():

            v_s_max = float('-inf')
            best_a = None

            for name, a in action_idx.items():
                r, c = coord[0], coord[1]
                dr, dc = ACTION_DELTAS_DEFAULT[name]
                next_r, next_c = clamp(r + dr, c + dc, H, W)
                next_s = state_idx[(next_r, next_c)]
                # Use immediate reward R[s,a] (not R[next_s,a])
                val = R[s, a] + gamma * v_table[next_s]
                if val > v_s_max:
                    v_s_max = val
                    best_a = a

            # update V(s) and compute delta without storing v_old
            change = abs(v_s_max - v_table[s])
            v_table[s] = v_s_max
            # keep only optimal action value in Q table
            if best_a is not None:
                q_table[s, :] = 0
                q_table[s, best_a] = v_s_max
            delta = max(delta, change)

        # convergence check using theta
        if delta < getattr(shared, "theta", 0.0):
            print(f"Value Iteration Stop, spend {iteration_count} count")
            break
        print(f"current iteration count : {iteration_count}")
        build_value_and_action_grids(v_table, q_table, action_idx, cfg, terminal_set)

    # print(v_table)
    H, W, v_grid, actions_grid = build_value_and_action_grids(v_table, q_table, action_idx, cfg, terminal_set)
    print(f"\nActions (optimal per state, {H}x{W}):")
    for r in range(H):
        print([actions_grid[r][c] for c in range(W)])
    # print(q_table)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nValue Iteration took {elapsed_time:.4f} seconds to run.")

