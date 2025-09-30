# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import json
import MC_Basic
from maze_utils import build_action_deltas, clamp


# Maze-shaped MDP configuration
@dataclass
class MazeConfig:
    width: int
    height: int
    forbidden: List[Tuple[int, int]]
    targets: List[Tuple[int, int]]
    rewards: Dict[str, float]  # keys: other, forbidden, target
    actions: Optional[List[str]] = None  # default: [up, down, left, right, stay]
    terminal_absorbing: bool = True  # whether target is absorbing state


@dataclass
class AlgoShared:
    gamma: float
    theta: float
    max_iterations: int


@dataclass
class AlgoPolicyIterationExtra:
    evaluation_max_iterations: int


@dataclass
class AlgoTruncatedPolicyIterationExtra:
    truncation_k: int

@dataclass
class AlgoMCBasicExtra:
    episode_length: int

@dataclass
class Algorithms:
    shared: AlgoShared
    policy_iteration: AlgoPolicyIterationExtra
    truncated_policy_iteration: AlgoTruncatedPolicyIterationExtra
    mc_basic: AlgoMCBasicExtra


@dataclass
class RootConfig:
    maze: MazeConfig
    algorithms: Algorithms


def _load_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    lower = path.lower()
    if lower.endswith(".json"):
        return json.loads(text)
    raise ValueError(f"Unsupported config format (expect .json): {path}")


def load_config(path: str = "config.json") -> RootConfig:
    data = _load_file(path)

    maze_data = data["maze"]
    maze = MazeConfig(
        width=int(maze_data["width"]),
        height=int(maze_data["height"]),
        forbidden=[tuple(p) for p in maze_data.get("forbidden", [])],
        targets=[tuple(p) for p in maze_data.get("targets", [])],
        rewards={
            "other": float(maze_data["rewards"]["other"]),
            "forbidden": float(maze_data["rewards"]["forbidden"]),
            "target": float(maze_data["rewards"]["target"]),
        },
        actions=maze_data.get("actions") or ["up", "down", "left", "right", "stay"],
        terminal_absorbing=bool(maze_data.get("terminal_absorbing", True)),
    )

    algo_data = data["algorithms"]
    shared = AlgoShared(
        gamma=float(algo_data["gamma"]),
        theta=float(algo_data["theta"]),
        max_iterations=int(algo_data["max_iterations"]),
    )
    pi_extra = AlgoPolicyIterationExtra(
        evaluation_max_iterations=int(algo_data["policy_iteration"]["evaluation_max_iterations"]),
    )
    tpi_extra = AlgoTruncatedPolicyIterationExtra(
        truncation_k=int(algo_data["truncated_policy_iteration"]["truncation_k"]),
    )
    mc_basic_extra = AlgoMCBasicExtra(
        episode_length=int(algo_data["mc_basic"]["episode_length"]),
    )

    return RootConfig(
        maze=maze,
        algorithms=Algorithms(shared=shared, policy_iteration=pi_extra, truncated_policy_iteration=tpi_extra, mc_basic=mc_basic_extra),
    )


def build_maze_mdp_arrays(maze: MazeConfig) -> Tuple["np.ndarray", "np.ndarray", Dict[Tuple[int, int], int], Dict[str, int], "np.ndarray"]:
    import numpy as np

    H, W = maze.height, maze.width
    actions = maze.actions or ["up", "down", "left", "right", "stay"]
    A = len(actions)

    # Every grid cell is a state
    coords: List[Tuple[int, int]] = [(r, c) for r in range(H) for c in range(W)]
    S = len(coords)
    state_idx: Dict[Tuple[int, int], int] = {coord: i for i, coord in enumerate(coords)}
    action_idx: Dict[str, int] = {a: i for i, a in enumerate(actions)}

    # Cell type mapping
    cell_type: Dict[Tuple[int, int], str] = {}
    forb_set = set(maze.forbidden)
    targ_set = set(maze.targets)
    for coord in coords:
        if coord in targ_set:
            cell_type[coord] = "target"
        elif coord in forb_set:
            cell_type[coord] = "forbidden"
        else:
            cell_type[coord] = "other"

    # Initialize P and R arrays
    P = np.zeros((S, A, S), dtype=float)
    R = np.zeros((S, A), dtype=float)

    # Terminal state indices (NumPy array)
    terminal_states = np.array([state_idx[coord] for coord in coords if cell_type[coord] == "target"], dtype=int)

    # Action direction deltas (from shared utils)
    delta = build_action_deltas(actions)

    for coord in coords:
        si = state_idx[coord]
        for a in actions:
            ai = action_idx[a]

            if maze.terminal_absorbing and si in terminal_states:
                P[si, ai, si] = 1.0
                R[si, ai] = maze.rewards["target"]
                continue

            dr, dc = delta[a]
            nr, nc = clamp(coord[0] + dr, coord[1] + dc, H, W)
            next_coord = (nr, nc)
            spi = state_idx[next_coord]

            P[si, ai, spi] = 1.0
            R[si, ai] = maze.rewards[cell_type[next_coord]]

    return P, R, state_idx, action_idx, terminal_states