# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np

from config import load_config, build_maze_mdp_arrays
from maze_utils import ACTION_DELTAS_DEFAULT, clamp


def prepare_policy_iteration(cfg_path: str = "config.json") -> Tuple:
    cfg = load_config(cfg_path)
    P, R, state_idx, action_idx, terminal_states = build_maze_mdp_arrays(cfg.maze)
    shared = cfg.algorithms.shared
    pi_extra = cfg.algorithms.policy_iteration
    print(f"Maze MDP loaded: |S|={len(state_idx)}, |A|={len(action_idx)}; terminals={len(terminal_states)}")
    print(
        f"PolicyIteration params: gamma={shared.gamma}, theta={shared.theta}, max_iter={shared.max_iterations}, eval_max_iter={pi_extra.evaluation_max_iterations}"
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


if __name__ == "__main__":
    import time
    start_time = time.time()
    P, R, state_idx, action_idx, terminal_states, shared, pi_extra, cfg = prepare_policy_iteration()
    v_table = np.zeros((cfg.maze.height, cfg.maze.width), dtype=float)                          # v_table 5 * 5 存stateValue
    q_table = np.zeros((cfg.maze.height * cfg.maze.width, len(action_idx)), dtype=float)        # q_table 25 * 5 存q_pi(s,a)
    a_table = np.zeros((cfg.maze.height, cfg.maze.width), dtype=int)                            # a_table 5 * 5 记录下旧的动作

    # 制定一个初始策略，所有状态采用动作stay
    for coord, s in state_idx.items():
        r, c = coord[0], coord[1]
        v_table[r, c] = R[s, action_idx["stay"]] + shared.gamma * v_table[r, c]
        a_table[r, c] = action_idx["stay"]
    
    # 针对初始策略开始迭代
    outer_iteration_count = 0
    while True:
        outer_iteration_count += 1

        # PE Policy Evaluation: Iterate until V-function for the current policy converges
        action_names = {i: name for name, i in action_idx.items()}
        for eval_iter in range(pi_extra.evaluation_max_iterations):
            delta = 0
            v_old_for_eval = v_table.copy()

            for coord, s in state_idx.items():
                r, c = coord[0], coord[1]

                # Get action from the fixed policy
                action_idx_current = a_table[r, c]
                # Skip terminal states if they have a policy assigned
                if action_idx_current == -1: continue
                action_name = action_names[action_idx_current]

                # Find next state
                dr, dc = ACTION_DELTAS_DEFAULT[action_name]
                next_r, next_c = clamp(r + dr, c + dc, cfg.maze.height, cfg.maze.width)

                # Correct Bellman expectation update using values from the previous sweep
                new_v = R[s, action_idx_current] + shared.gamma * v_old_for_eval[next_r, next_c]

                delta = max(delta, abs(new_v - v_old_for_eval[r, c]))
                v_table[r, c] = new_v

            # Check for convergence
            if delta < shared.theta:
                break

        # PI Policy Improvement: Greedily update policy and check for stability
        policy_stable = True
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

            # Update the policy and V-value for the current state
            a_table[r, c] = best_a
            v_table[r, c] = v_s_max
            
            # Check if the policy has changed
            if best_a != old_action:
                policy_stable = False

        # If the policy is stable, we have found the optimal policy.
        if policy_stable:
            print(f"Policy stabilized after {outer_iteration_count} iterations.")
            break
        
        if outer_iteration_count >= shared.max_iterations:
            print(f"Max iterations ({shared.max_iterations}) reached, breaking loop.")
            break

    # Format and print the final V-table
    v_table_formatted = [[f"{v:.2f}" for v in row] for row in v_table]
    print("Final V-table (Policy Iteration):")
    for row in v_table_formatted:
        print(row)

    # Format and print the final Q-table
    q_table_formatted = [[f"{q:.2f}" for q in row] for row in q_table]
    print("\nFinal Q-table (Policy Iteration):")
    # Create a header for the Q-table
    action_names_header = [name for name, idx in sorted(action_idx.items(), key=lambda item: item[1])]
    print(f"{'State':<6} " + " ".join([f"{name:<5}" for name in action_names_header]))
    for s, row in enumerate(q_table_formatted):
        print(f"{s:<6} [" + " ".join([f"{val:<5}" for val in row]) + "]")


    # Format and print the final policy (a_table)
    action_symbols = {
        action_idx["up"]: "↑",
        action_idx["down"]: "↓",
        action_idx["left"]: "←",
        action_idx["right"]: "→",
        action_idx["stay"]: "·",
    }
    policy_formatted = [[action_symbols.get(a, "?") for a in row] for row in a_table]
    print("\nFinal Policy (Action Table):")
    for row in policy_formatted:
        print(" ".join(row))


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nPolicy Iteration took {elapsed_time:.4f} seconds to run.")
