import numpy as np


def build_action_axis(action_grid_size: int) -> np.ndarray:
    if action_grid_size < 3:
        raise ValueError("action_grid_size must be at least 3.")
    return np.linspace(-1.0, 1.0, action_grid_size, dtype=np.float32)


def build_action_table_np(action_grid_size: int) -> np.ndarray:
    axis = build_action_axis(action_grid_size)
    return np.array([[ux, uy] for ux in axis for uy in axis], dtype=np.float32)


def decode_action_index(action_index: int, action_grid_size: int) -> np.ndarray:
    table = build_action_table_np(action_grid_size)
    return table[int(action_index)]
