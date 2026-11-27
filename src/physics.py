from abc import ABC, abstractmethod

import numpy as np

from src import constants as const


class ForceModel(ABC):
    @abstractmethod
    def calc_acceleration(self, t: float, state: np.ndarray) -> np.ndarray:
        pass

    def calc_derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        return np.concatenate((state[3:6], self.calc_acceleration(t, state)))


class CentralGravity(ForceModel):
    def __init__(self, mu: float = const.EARTH_MU):
        self.mu = mu

    def calc_acceleration(self, t: float, state: np.ndarray) -> np.ndarray:
        r_val_cube = np.linalg.norm(state[0:3]) ** 3
        a_vector = -(self.mu / r_val_cube) * (state[0:3])  # G*M/r^3 *-r_vector
        return a_vector
