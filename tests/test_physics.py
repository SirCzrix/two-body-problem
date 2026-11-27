import numpy as np
import pytest

import src.constants as const
from src.integrators import RK4
from src.physics import calc_derivative


def test_radius():
    r = 6778137.0
    t = 0.0
    v1 = np.sqrt(const.EARTH_MU / r)
    T = 2 * np.pi * np.sqrt(r**3 / const.EARTH_MU)
    h = T / 5554
    state = np.array([r, 0.0, 0.0, 0.0, v1, 0.0])
    for _ in range(5554):
        state = RK4(calc_derivative, t, state, h)
        t += h
    assert np.linalg.norm(state[0:3]) == pytest.approx(r, rel=1e-7)


def test_low_orbit():  # ISS
    r = 6778137.0
    t = 0.0
    v1 = np.sqrt(const.EARTH_MU / r)
    T = 2 * np.pi * np.sqrt(r**3 / const.EARTH_MU)
    h = T / 5554
    state = np.array([r, 0.0, 0.0, 0.0, v1, 0.0])
    for _ in range(5554):
        state = RK4(calc_derivative, t, state, h)
        t += h
    assert state == pytest.approx(
        np.array([r, 0.0, 0.0, 0.0, v1, 0.0]), rel=1e-7, abs=1e-5
    )


def test_energy_conservation():
    r = 6778137.0
    t = 0.0
    v1 = np.sqrt(const.EARTH_MU / r)
    T = 2 * np.pi * np.sqrt(r**3 / const.EARTH_MU)
    h = T / 5554
    state = np.array([r, 0.0, 0.0, 0.0, v1, 0.0])
    e_start = (v1**2) / 2 - const.EARTH_MU / r
    for _ in range(5554):
        state = RK4(calc_derivative, t, state, h)
        t += h
    e_end = (np.linalg.norm(state[3:6]) ** 2) / 2 - const.EARTH_MU / np.linalg.norm(
        state[0:3]
    )
    assert e_start == pytest.approx(e_end, rel=1e-10)
