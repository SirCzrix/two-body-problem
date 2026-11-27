import numpy as np

from src import integrators, physics


class AstralBody:
    def __init__(
        self,
        start_pos: np.ndarray,
        start_vel: np.ndarray,
        force_model: physics.ForceModel,
        integrator: integrators.Integrator,
    ):
        if start_pos.shape != (3,):
            raise ValueError(
                f"Expected position vector with size (3,), given vector was {start_pos.shape}"
            )
        if start_vel.shape != (3,):
            raise ValueError(
                f"Expected velocity vector with size (3,), given vector was {start_vel.shape}"
            )
        self.pos = np.array(start_pos, dtype=float, copy=True)
        self.vel = np.array(start_vel, dtype=float, copy=True)
        self.force_model = force_model
        self.integrator = integrator
        self.trajectory: list[np.ndarray] = [self.pos.copy()]

    @classmethod
    def from_state(
        cls,
        state: np.ndarray,
        force_model: physics.ForceModel,
        integrator: integrators.Integrator,
    ):
        if state.shape != (6,):
            raise ValueError(
                f"Expected state vector with size (6,), given vector was {state.shape}"
            )
        pos = state[0:3]
        vel = state[3:6]
        return cls(pos, vel, force_model, integrator)

    def get_state(self) -> np.ndarray:
        return np.concatenate((self.pos, self.vel))

    @property
    def radius(self) -> float:
        return float(np.linalg.norm(self.pos))

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))

    def step(self, t: float, dt: float):
        state = self.get_state()
        tmp = self.integrator.step(self.force_model.calc_derivative, t, state, dt)
        self.pos = tmp[0:3]
        self.vel = tmp[3:6]
        self.trajectory.append(self.pos.copy())
