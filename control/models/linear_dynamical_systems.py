import math

import torch
import torch.nn.functional as F

from control.models.controlled_dynamical_systems import DynamicalSystem


class LinearTimeInvariantSystem(DynamicalSystem):

    def __init__(self, state_dim: int, control_dim: int, observation_dim: int = -1, perturbation_std: float = 0.0,
                 state_cost_mat_rank: int = -1, control_cost_mat_rank: int = -1, diag_cost_matrices: bool = False,
                 mats_init_std: float = -1, init_transition_mats_near_identity: bool = False, random_seed: int = -1):
        super(LinearTimeInvariantSystem, self).__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.observation_dim = observation_dim
        self.perturbation_std = perturbation_std
        self.state_cost_mat_rank = state_cost_mat_rank
        self.control_cost_mat_rank = control_cost_mat_rank
        self.diag_cost_matrices = diag_cost_matrices
        self.mats_init_std = mats_init_std if mats_init_std > 0 else 1 / math.sqrt(self.state_dim)
        self.init_transition_mats_near_identity = init_transition_mats_near_identity
        self.random_seed = random_seed

        self.__initialize_system()

    def __initialize_system(self):
        if self.random_seed >= 0:
            curr_rng_state = torch.random.get_rng_state()
            torch.random.manual_seed(self.random_seed)

        A = torch.randn(self.state_dim, self.state_dim) * self.mats_init_std
        if self.init_transition_mats_near_identity:
            A += torch.eye(self.state_dim)
        self.register_buffer("A", A)

        B = torch.randn(self.control_dim, self.state_dim) * self.mats_init_std
        if self.init_transition_mats_near_identity:
            B += torch.eye(self.control_dim, self.state_dim)
        self.register_buffer("B", B)

        state_cost_mat_rank = self.state_cost_mat_rank if self.state_cost_mat_rank >= 0 else self.state_dim
        if state_cost_mat_rank == 0:
            Q = torch.zeros(self.state_dim, self.state_dim)
            self.register_buffer("Q", Q)
        elif self.diag_cost_matrices:
            Q = torch.zeros(self.state_dim, self.state_dim)
            indices = torch.randperm(self.state_dim)[:state_cost_mat_rank]
            Q[indices, indices] = 1
            self.register_buffer("Q", Q)
        else:
            Q = torch.randn(self.state_dim, state_cost_mat_rank) * self.mats_init_std
            self.register_buffer("Q", torch.matmul(Q, Q.t()))

        control_cost_mat_rank = self.control_cost_mat_rank if self.control_cost_mat_rank >= 0 else self.control_dim
        if control_cost_mat_rank == 0:
            R = torch.zeros(self.control_dim, self.control_dim)
            self.register_buffer("R", R)
        elif self.diag_cost_matrices:
            R = torch.zeros(self.control_dim, self.control_dim)
            indices = torch.randperm(self.control_dim)[:control_cost_mat_rank]
            R[indices, indices] = 1
            self.register_buffer("R", R)
        else:
            R = torch.randn(self.control_dim, control_cost_mat_rank) * self.mats_init_std
            self.register_buffer("R", torch.matmul(R, R.t()))

        if self.observation_dim > 0:
            C = torch.randn(self.state_dim, self.observation_dim) * self.mats_init_std
            if self.init_transition_mats_near_identity:
                C += torch.eye(self.state_dim, self.observation_dim)
            self.register_buffer("C", C)

            D = torch.randn(self.control_dim, self.observation_dim) * self.mats_init_std
            if self.init_transition_mats_near_identity:
                D += torch.eye(self.control_dim, self.observation_dim)
            self.register_buffer("D", D)

        if self.random_seed >= 0:
            torch.random.set_rng_state(curr_rng_state)

    def set_custom_system(self, A: torch.Tensor, B: torch.Tensor, Q: torch.Tensor, R: torch.Tensor,
                          C: torch.Tensor = None, D: torch.Tensor = None):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.C = C
        self.D = D

    def get_state_dim(self):
        return self.state_dim

    def get_control_dim(self):
        return self.control_dim

    def compute_time_step_cost(self, state: torch.Tensor, control: torch.Tensor):
        return F.bilinear(state, state, self.Q.unsqueeze(dim=0)) + F.bilinear(control, control, self.R.unsqueeze(dim=0))

    def forward(self, state: torch.Tensor, control: torch.Tensor):
        new_state = torch.matmul(state, self.A) + torch.matmul(control, self.B) + torch.randn_like(state) * self.perturbation_std
        cost = self.compute_time_step_cost(state, control)
        return new_state, cost
