import numpy as np
import pytest

from src.integrators import RK4


def test_RK4_exp():
    t = 0.0
    h = 0.01
    state = np.array([1.0])
    for _ in range(100):
        state = RK4(exp_derivative, t, state, h)
        t += h
    assert state[0] == pytest.approx(np.e, rel=1e-7)


def test_RK4_x():
    t = 0.0
    h = 0.01
    state = np.array([1.0])
    for _ in range(100):
        state = RK4(x_derivative, t, state, h)
        t += h
    assert state[0] == pytest.approx(2, rel=1e-7)


def exp_derivative(t, state):
    return state


def x_derivative(t, state):
    return np.array([1])
