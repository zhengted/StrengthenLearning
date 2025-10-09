# -*- coding: utf-8 -*-
from typing import Dict, Tuple, List

# Default action direction deltas
ACTION_DELTAS_DEFAULT: Dict[str, Tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "stay": (0, 0),
}


def build_action_deltas(actions: List[str]) -> Dict[str, Tuple[int, int]]:
    """Return direction deltas for provided actions.
    Actions not in default map are mapped to (0, 0).
    """
    return {a: ACTION_DELTAS_DEFAULT.get(a, (0, 0)) for a in actions}


def clamp(r: int, c: int, H: int, W: int) -> Tuple[int, int]:
    """Clamp coordinates into the grid range [0, H-1] x [0, W-1]."""
    if r < 0 or r >= H or c < 0 or c >= W:
        return max(0, min(r, H - 1)), max(0, min(c, W - 1))
    return r, c