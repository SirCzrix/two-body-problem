import numpy as np

from src import constants as const


def calc_derivative(
    t: float, state: np.ndarray
):  ##TODO here add check for config flags
    if state.shape != (6,):
        raise ValueError(f"Expected 6-element vector, given vector was {state.shape}")
    a_vector = calc_gravitation_acceleration(state)
    state_d = np.zeros((6,))
    state_d[0:3] = state[3:6]
    state_d[3:6] = a_vector
    return state_d


def calc_gravitation_acceleration(state: np.ndarray):
    r_val_cube = np.linalg.norm(state[0:3]) ** 3
    a_vector = -(const.EARTH_MU / r_val_cube) * (state[0:3])  # G*M/r^3 *-r_vector
    return a_vector
