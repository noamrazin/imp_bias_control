import logging

import torch

from common.train.callbacks import Callback
from common.train.fit_output import FitOutput
from common.train.trainer import Trainer
from control.models.controllers import LinearStateController, LinearNNStateController
from control.models.controlled_dynamical_systems import ControlledDynamicalSystemImpl


class ControllerLogger(Callback):

    def __init__(self, logger: logging.Logger, model: ControlledDynamicalSystemImpl, states: torch.Tensor, epoch_log_interval: int = 1,
                 device=torch.device("cpu")):
        self.logger = logger
        self.model = model
        self.states = states
        self.epoch_log_interval = epoch_log_interval
        self.device = device

    def on_fit_start(self, trainer: Trainer, num_epochs: int):
        self.__log_controller()

    def on_epoch_end(self, trainer: Trainer):
        if self.epoch_log_interval > 0 and (trainer.epoch + 1) % self.epoch_log_interval == 0:
            self.__log_controller()

    def on_fit_end(self, trainer: Trainer, num_epochs_ran: int, fit_output: FitOutput):
        self.__log_controller()

    def __log_controller(self):
        with torch.no_grad():
            if isinstance(self.model.controller, LinearStateController) or isinstance(self.model.controller, LinearNNStateController):
                self.__log_control_matrices_for_linear_controllers()
            else:
                self.__log_control_over_states_for_nonlinear_controller()

    def __log_control_matrices_for_linear_controllers(self):
        self.logger.info(f"Controller matrix K:\n{self.model.controller.compute_control_matrix().data}")

        if hasattr(self.model.controller, "bias"):
            self.logger.info(f"Controller bias:\n{self.model.controller.bias.data}")

    def __log_control_over_states_for_nonlinear_controller(self):
        self.model.to(self.device)
        self.states = self.states.to(self.device)

        controls = self.model.controller(self.states)
        self.logger.info(f"States:\n{self.states}\nControls:\n{controls}")
