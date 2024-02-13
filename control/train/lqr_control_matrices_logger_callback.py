import logging

import torch

from common.train.callbacks import Callback
from common.train.fit_output import FitOutput
from common.train.trainer import Trainer
from control.models.controllers import LinearStateController, LinearNNStateController
from control.models.controlled_dynamical_systems import ControlledDynamicalSystemImpl


class LQRControlMatricesLogger(Callback):

    def __init__(self, logger: logging.Logger, model: ControlledDynamicalSystemImpl, epoch_log_interval: int = 1):
        self.logger = logger
        self.model = model
        self.epoch_log_interval = epoch_log_interval

    def on_fit_start(self, trainer: Trainer, num_epochs: int):
        self.__log_control_matrices()
        self.__log_dynamics_matrices()

    def on_epoch_end(self, trainer: Trainer):
        if self.epoch_log_interval > 0 and (trainer.epoch + 1) % self.epoch_log_interval == 0:
            self.__log_control_matrices()

    def on_fit_end(self, trainer: Trainer, num_epochs_ran: int, fit_output: FitOutput):
        self.__log_control_matrices()

    def __log_dynamics_matrices(self):
        self.logger.info(f"A:\n{self.model.system.A}")
        self.logger.info(f"B:\n{self.model.system.B}")
        self.logger.info(f"Q:\n{self.model.system.Q}")
        self.logger.info(f"R:\n{self.model.system.R}")

    def __log_control_matrices(self):
        with torch.no_grad():
            if isinstance(self.model.controller, LinearStateController) or isinstance(self.model.controller, LinearNNStateController):
                self.__log_control_matrices_for_linear_controllers()
            else:
                self.__log_control_standard_basis_transitions_for_nonlinear_controller()

    def __log_control_matrices_for_linear_controllers(self):
        K = self.model.controller.compute_control_matrix()
        state_update_matrix = self.model.system.A + torch.matmul(K, self.model.system.B)

        self.logger.info(f"Controller matrix K:\n{K.data}")
        self.logger.info(f"State update matrix A + KB (states are row vectors):\n{state_update_matrix}")
        self.logger.info(f"(A + KB)Q (states are row vectors):\n{torch.matmul(state_update_matrix, self.model.system.Q)}")

    def __log_control_standard_basis_transitions_for_nonlinear_controller(self):
        standard_basis_controls = self.model.controller(torch.eye(self.model.system.state_dim, device=self.model.system.A.device))
        state_updates = self.model.system.A + torch.matmul(standard_basis_controls, self.model.system.B)

        self.logger.info(f"Standard basis controls K(e_1,...,e_n):\n{standard_basis_controls}")
        self.logger.info(f"Standard basis state updates matrix A + K(e_1,...,e_n)B (states are row vectors):\n{state_updates}")
        self.logger.info(f"(A + K(e_1,...,e_n)B)Q (states are row vectors):\n{torch.matmul(state_updates, self.model.system.Q)}")
