import numpy as np
import pytest

import src.constants as const
from src.integrators import RK4
from src.models import AstralBody
from src.physics import CentralGravity


@pytest.fixture
def iss_satelite() -> AstralBody:
    r = 6778137.0
    v1 = np.sqrt(const.EARTH_MU / r)
    state = np.array([r, 0.0, 0.0, 0.0, v1, 0.0])
    return AstralBody.from_state(state, CentralGravity(), RK4())


def test_radius(iss_satelite):
    r = 6778137.0
    t = 0.0
    T = 2 * np.pi * np.sqrt(r**3 / const.EARTH_MU)
    h = T / 5554
    for _ in range(5554):
        iss_satelite.step(t, h)
        t += h
    assert np.linalg.norm(iss_satelite.radius) == pytest.approx(r, rel=1e-7)


def test_low_orbit(iss_satelite):  # ISS
    r = 6778137.0
    t = 0.0
    T = 2 * np.pi * np.sqrt(r**3 / const.EARTH_MU)
    h = T / 5554
    v1 = np.sqrt(const.EARTH_MU / r)
    for _ in range(5554):
        iss_satelite.step(t, h)
        t += h
    assert iss_satelite.get_state() == pytest.approx(
        np.array([r, 0.0, 0.0, 0.0, v1, 0.0]), rel=1e-7, abs=1e-5
    )


def test_energy_conservation(iss_satelite):
    r = 6778137.0
    t = 0.0
    T = 2 * np.pi * np.sqrt(r**3 / const.EARTH_MU)
    h = T / 5554
    e_start = (iss_satelite.speed**2) / 2 - const.EARTH_MU / iss_satelite.radius
    for _ in range(5554):
        iss_satelite.step(t, h)
        t += h
    e_end = (iss_satelite.speed**2) / 2 - const.EARTH_MU / iss_satelite.radius
    assert e_start == pytest.approx(e_end, rel=1e-10)
