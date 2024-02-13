import numpy as np
import torch
from torch import cos, sin, cross, einsum

from control.models.controlled_dynamical_systems import ControlledDynamicalSystem
from torchdyn.numerics.odeint import odeint


class Quadcopter(ControlledDynamicalSystem):
    """
    Quadcopter state space model compatible with batch inputs adapted from: https://github.com/DiffEqML/torchcontrol.
    """
    STATE_DIM = 12
    CONTROL_DIM = 4
    G = 9.81
    RAD2DEG = 180 / np.pi
    DEG2RAD = np.pi / 180
    M = 0.027
    L = 0.0397
    THRUST2WEIGHT_RATIO = 2.25
    J = torch.diag(torch.Tensor([1.4e-5, 1.4e-5, 2.17e-5]))
    J_INV = torch.linalg.inv(J)
    KF = 3.16e-10
    KM = 7.94e-12
    GRAVITY = G * M
    HOVER_RPM = np.sqrt(GRAVITY / (4 * KF))
    MAX_RPM = np.sqrt((THRUST2WEIGHT_RATIO * GRAVITY) / (4 * KF))
    MAX_THRUST = (4 * KF * MAX_RPM ** 2)
    # DroneModel.CF2X:
    MAX_XY_TORQUE = (2 * L * KF * MAX_RPM ** 2) / np.sqrt(2)
    MAX_Z_TORQUE = (2 * KM * MAX_RPM ** 2)

    def __init__(self, controller, target_state: torch.Tensor, target_state_cost_coeffs: torch.Tensor = None,
                 time_res: float = 0.02, solver='euler', **odeint_kwargs):
        super(Quadcopter, self).__init__()
        self.controller = controller
        self.target_state = target_state
        self.target_state_cost_coeffs = target_state_cost_coeffs.unsqueeze(dim=0) if target_state_cost_coeffs is not None else torch.ones(1, 12)
        self.time_res = time_res
        self.solver = solver
        self.nfe = 0  # count number of function evaluations of the vector field
        self.controls = []
        self.odeint_kwargs = odeint_kwargs

    def reset(self):
        """Reset number of dynamics evaluations counter and controls retention"""
        self.nfe = 0
        self.controls = []

    def __dynamics(self, t, state):
        self.nfe += 1  # increment number of function evaluations

        # Control input evaluation
        control = self.controller(state)
        self.controls.append(control)

        pos = state[..., 0:3]
        rpy = state[..., 3:6]
        vel = state[..., 6:9]
        rpy_rates = state[..., 9:12]

        next_state = self.__compute_next_state(pos, rpy, rpy_rates, vel, control)
        return next_state

    def __compute_next_state(self, pos, rpy, rpy_rates, vel, rpm):
        # Compute forces and torques
        forces = rpm ** 2 * self.KF
        thrust_z = torch.sum(forces, dim=-1)
        thrust = torch.zeros(pos.shape, device=pos.device)
        thrust[..., 2] = thrust_z
        rotation = euler_matrix(rpy[..., 0], rpy[..., 1], rpy[..., 2]).to(pos.device)
        thrust_world_frame = einsum('...ij, ...j-> ...i', rotation, thrust)
        force_world_frame = thrust_world_frame - torch.tensor([0, 0, self.GRAVITY], device=pos.device)
        z_torques = rpm ** 2 * self.KM
        z_torque = (-z_torques[..., 0] + z_torques[..., 1] - z_torques[..., 2] + z_torques[..., 3])

        # DroneModel.CF2X:
        x_torque = (forces[..., 0] + forces[..., 1] - forces[..., 2] - forces[..., 3]) * (self.L / np.sqrt(2))
        y_torque = (- forces[..., 0] + forces[..., 1] + forces[..., 2] - forces[..., 3]) * (self.L / np.sqrt(2))
        torques = torch.cat([x_torque[..., None], y_torque[..., None], z_torque[..., None]], -1)
        torques = torques - cross(rpy_rates, einsum('ij,...i->...j', self.J.to(rpy_rates.device), rpy_rates))
        rpy_rates_deriv = einsum('ij,...i->...j', self.J_INV.to(rpy_rates.device), torques)
        no_pybullet_dyn_accs = force_world_frame / self.M

        return torch.cat([vel, rpy_rates, no_pybullet_dyn_accs, rpy_rates_deriv], -1)

    def __compute_costs(self, states, custom_target_state: torch.Tensor = None):
        target_state_coeffs = self.target_state_cost_coeffs.to(states.device)
        target_state = self.target_state.to(states.device) if custom_target_state is None else custom_target_state.to(states.device)
        if len(target_state.shape) == 1:
            target_state_unsqueezed = target_state.unsqueeze(dim=0).unsqueeze(dim=0)
        else:
            target_state_unsqueezed = target_state.unsqueeze(dim=1)

        weighted_state_diff = (states - target_state_unsqueezed) * target_state_coeffs.unsqueeze(dim=0)
        return torch.sum(weighted_state_diff ** 2, dim=-1)

    def forward(self, state: torch.Tensor, steps: int, return_controls: bool = False, custom_target_state: torch.Tensor = None, **kwargs):
        t_span = torch.linspace(0, self.time_res * steps, steps + 1)
        states = odeint(self.__dynamics, state, t_span, solver=self.solver, **self.odeint_kwargs)[1].permute(1, 0, 2)
        costs = self.__compute_costs(states, custom_target_state=custom_target_state)
        controls = torch.stack(self.controls).permute(1, 0, 2)
        self.reset()

        if not return_controls:
            return states, costs
        return states, costs, controls


def euler_matrix(ai, aj, ak, repetition=True):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    Readapted for Pytorch: some tricks going on
    """
    si, sj, sk = sin(ai), sin(aj), sin(ak)
    ci, cj, ck = cos(ai), cos(aj), cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    i = 0
    j = 1
    k = 2  # indexing

    # Tricks to create batched matrix [...,3,3]
    M = torch.cat(3 * [torch.cat(3 * [torch.zeros(ai.shape)[..., None, None]], -1)], -2)
    if repetition:
        M[..., i, i] = cj
        M[..., i, j] = sj * si
        M[..., i, k] = sj * ci
        M[..., j, i] = sj * sk
        M[..., j, j] = -cj * ss + cc
        M[..., j, k] = -cj * cs - sc
        M[..., k, i] = -sj * ck
        M[..., k, j] = cj * sc + cs
        M[..., k, k] = cj * cc - ss
    else:
        M[..., i, i] = cj * ck
        M[..., i, j] = sj * sc - cs
        M[..., i, k] = sj * cc + ss
        M[..., j, i] = cj * sk
        M[..., j, j] = sj * ss + cc
        M[..., j, k] = sj * cs - sc
        M[..., k, i] = -sj
        M[..., k, j] = cj * si
        M[..., k, k] = cj * ci
    return M
