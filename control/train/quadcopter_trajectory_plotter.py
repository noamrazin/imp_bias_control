import os

from common.train.callbacks import Callback
from common.train.trainer import Trainer
from control.models.quadcopter import Quadcopter
from control.utils.quadcoptor_plotting import *


class QuadcopterTrajectoryPlotter(Callback):
    TRAJECTORY_3D_PLOT_FILE_NAME_TEMPLATE = "{phase}_epoch_{epoch}_init_pos_{x:.2f}_{y:.2f}_{z:.2f}_traj3d.png"
    ANIMATION_3D_PLOT_FILE_NAME_TEMPLATE = "{phase}_epoch_{epoch}_init_pos_{x:.2f}_{y:.2f}_{z:.2f}_animation.gif"
    TRAJECTORY_INFO_PLOT_FILE_NAME_TEMPLATE = "{phase}_epoch_{epoch}_init_pos_{x:.2f}_{y:.2f}_{z:.2f}_traj_info.png"
    CONTROL_INFO_PLOT_FILE_NAME_TEMPLATE = "{phase}_epoch_{epoch}_init_pos_{x:.2f}_{y:.2f}_{z:.2f}_traj_control.png"
    QUAD_PLOTS_DIR = "plots"
    QUAD_ANIMATION_DIR_NAME = "animations"

    def __init__(self, model: Quadcopter, train_initial_states: torch.Tensor, train_time_horizon: int, test_initial_states: torch.Tensor,
                 test_time_horizon: int, target_state_pos: torch.Tensor, save_dir: str, quad_plot_lim: float = 1.2, save_quad_animation: bool = False,
                 epoch_save_interval: int = -1, device=torch.device("cpu"), logger=None):
        self.model = model
        self.train_initial_states = train_initial_states
        self.train_time_horizon = train_time_horizon
        self.test_initial_states = test_initial_states
        self.test_time_horizon = test_time_horizon
        self.target_state_pos = target_state_pos
        self.save_dir = save_dir
        self.quad_plot_lim = quad_plot_lim
        self.save_quad_animation = save_quad_animation
        self.epoch_save_interval = epoch_save_interval
        self.device = device
        self.logger = logger

    def on_fit_start(self, trainer: Trainer, num_epochs: int):
        self.__save_quadcopter_plots(trainer.epoch, save_quad_animation=self.save_quad_animation)

    def on_epoch_end(self, trainer: Trainer):
        if self.epoch_save_interval > 0 and (trainer.epoch + 1) % self.epoch_save_interval == 0:
            self.__save_quadcopter_plots(trainer.epoch, save_quad_animation=False)

    def on_fit_termination(self, trainer: Trainer):
        self.__save_quadcopter_plots(trainer.epoch, save_quad_animation=self.save_quad_animation)

    def __save_quadcopter_plots(self, epoch: int, save_quad_animation: bool = False):
        with torch.no_grad():
            self.model.eval()
            self.model.to(self.device)

            self.__save_quadcopter_plots_for_initial_states(self.train_initial_states, self.train_time_horizon, "train", epoch,
                                                            save_quad_animation=save_quad_animation)
            self.__save_quadcopter_plots_for_initial_states(self.test_initial_states, self.test_time_horizon, "test", epoch,
                                                            save_quad_animation=save_quad_animation)

    def __save_quadcopter_plots_for_initial_states(self, initial_states: torch.Tensor, time_horizon: int, phase: str, epoch: int,
                                                   save_quad_animation: bool = False):
        initial_states = initial_states.to(self.device)
        all_states, _, all_controls = self.model(initial_states, time_horizon, return_controls=True)

        if self.logger is not None:
            self.__log_final_states(phase, initial_states, all_states[:, -1, :])

        for i in range(initial_states.shape[0]):
            states = all_states[i]
            controls = all_controls[i]

            dir = os.path.join(self.save_dir, self.QUAD_PLOTS_DIR)
            if not os.path.exists(dir):
                os.makedirs(dir)

            traj_info_plot_file_name = self.TRAJECTORY_INFO_PLOT_FILE_NAME_TEMPLATE.format(phase=phase, epoch=epoch, x=states[0, 0].item(),
                                                                                           y=states[0, 1].item(), z=states[0, 2].item())
            save_quadcopter_trajectory_state_info_plot(states, save_path=os.path.join(dir, traj_info_plot_file_name))

            control_plot_file_name = self.CONTROL_INFO_PLOT_FILE_NAME_TEMPLATE.format(phase=phase, epoch=epoch, x=states[0, 0].item(),
                                                                                      y=states[0, 1].item(), z=states[0, 2].item())
            save_quadcopter_controls_plot(controls, save_path=os.path.join(dir, control_plot_file_name))

            traj_3d_plot_file_name = self.TRAJECTORY_3D_PLOT_FILE_NAME_TEMPLATE.format(phase=phase, epoch=epoch, x=states[0, 0].item(),
                                                                                       y=states[0, 1].item(), z=states[0, 2].item())
            save_quadcopter_trajectory_3d(states, target_state_pos=self.target_state_pos, save_path=os.path.join(dir, traj_3d_plot_file_name),
                                          min_lim=-self.quad_plot_lim, max_lim=self.quad_plot_lim)

            if save_quad_animation:
                anim_dir = os.path.join(self.save_dir, self.QUAD_ANIMATION_DIR_NAME)
                if not os.path.exists(anim_dir):
                    os.makedirs(anim_dir)

                animation_3d_plot_file_name = self.ANIMATION_3D_PLOT_FILE_NAME_TEMPLATE.format(phase=phase, epoch=epoch, x=states[0, 0].item(),
                                                                                               y=states[0, 1].item(), z=states[0, 2].item())
                save_quadcopter_trajectory_3d_animation(states, target_state_pos=self.target_state_pos,
                                                        save_path=os.path.join(anim_dir, animation_3d_plot_file_name),
                                                        min_lim=-self.quad_plot_lim, max_lim=self.quad_plot_lim)

    def __log_final_states(self, phase: str, initial_states: torch.Tensor, final_states: torch.Tensor):
        self.logger.info(f"{phase} initial states:\n{initial_states.detach().cpu()}\n{phase} final states:\n{final_states.detach().cpu()}")
