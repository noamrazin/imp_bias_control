import os

import torch

from common.train.callbacks import Callback
from common.train.trainer import Trainer
from control.models.controlled_dynamical_systems import ControlledDynamicalSystemImpl
from control.models.pendulum import create_pendulum_dynamics_animation


class PendulumAnimationPlotter(Callback):
    ANIMATION_FILE_NAME_TEMPLATE = "{phase}_epoch_{epoch}_init_angle_{angle:.3f}_pendulum.gif"
    PENDULUM_ANIMATION_DIR_NAME = "animations"

    def __init__(self, model: ControlledDynamicalSystemImpl, train_initial_states: torch.Tensor, train_time_horizon: int,
                 test_initial_states: torch.Tensor, test_time_horizon: int, save_dir: str, epoch_save_interval: int = -1,
                 device=torch.device("cpu"), logger=None):
        self.model = model
        self.train_initial_states = train_initial_states
        self.train_time_horizon = train_time_horizon
        self.test_initial_states = test_initial_states
        self.test_time_horizon = test_time_horizon
        self.save_dir = save_dir
        self.epoch_save_interval = epoch_save_interval
        self.device = device
        self.logger = logger

    def on_fit_start(self, trainer: Trainer, num_epochs: int):
        self.__save_pendulum_animation(trainer.epoch)

    def on_epoch_end(self, trainer: Trainer):
        if self.epoch_save_interval > 0 and (trainer.epoch + 1) % self.epoch_save_interval == 0:
            self.__save_pendulum_animation(trainer.epoch)

    def on_fit_termination(self, trainer: Trainer):
        self.__save_pendulum_animation(trainer.epoch)

    def __save_pendulum_animation(self, epoch: int):
        with torch.no_grad():
            self.model.eval()
            self.model.to(self.device)

            self.__save_pendulum_animation_for_initial_states(self.train_initial_states, self.train_time_horizon, "train", epoch)
            self.__save_pendulum_animation_for_initial_states(self.test_initial_states, self.test_time_horizon, "test", epoch)

    def __save_pendulum_animation_for_initial_states(self, initial_states: torch.Tensor, time_horizon: int, phase: str, epoch: int):
        initial_states = initial_states.to(self.device)
        all_states, _, all_controls = self.model(initial_states, time_horizon, return_controls=True)

        if self.logger is not None:
            self.__log_final_states(phase, initial_states, all_states[:, -1, :])

        for i in range(initial_states.shape[0]):
            states = all_states[i]
            controls = all_controls[i]

            dir = os.path.join(self.save_dir, self.PENDULUM_ANIMATION_DIR_NAME)
            if not os.path.exists(dir):
                os.makedirs(dir)

            file_name = self.ANIMATION_FILE_NAME_TEMPLATE.format(phase=phase, epoch=epoch, angle=states[0, 0].item())
            output_path = os.path.join(dir, file_name)
            create_pendulum_dynamics_animation(states, controls, save_path=output_path, rod_length=self.model.system.length)

    def __log_final_states(self, phase: str, initial_states: torch.Tensor, final_states: torch.Tensor):
        self.logger.info(f"{phase} initial states:\n{initial_states.detach().cpu()}\n{phase} final states:\n{final_states.detach().cpu()}")
