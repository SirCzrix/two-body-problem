import collections.abc
from abc import ABC, abstractmethod

import numpy as np


class Integrator(ABC):
    @abstractmethod
    def step(
        self,
        function: collections.abc.Callable[[float, np.ndarray], np.ndarray],
        t: float,
        state: np.ndarray,
        h: float,
    ) -> np.ndarray:
        pass


class RK4(Integrator):
    def step(
        self,
        function: collections.abc.Callable[[float, np.ndarray], np.ndarray],
        t: float,
        state: np.ndarray,
        h: float,
    ) -> np.ndarray:
        if state.ndim != 1:
            raise ValueError(f"Expected 1D vector, given vector was {state.shape}")
        k1 = function(t, state)

        state2 = state + (h * 0.5) * k1
        k2 = function(t + h * 0.5, state2)

        state3 = state + (h * 0.5) * k2
        k3 = function(t + h * 0.5, state3)

        state4 = state + h * k3
        k4 = function(t + h, state4)

        k = (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return state + k * h
