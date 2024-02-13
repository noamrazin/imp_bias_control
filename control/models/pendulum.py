import matplotlib as mpl
import numpy as np
import torch

mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from control.models.controlled_dynamical_systems import DynamicalSystem


class Pendulum(DynamicalSystem):
    """
    Controlled pendulum swing up dynamical system.
    """

    def __init__(self, mass: float = 1, length: float = 1, gravity: float = 10, time_res: float = 0.05, control_norm_cost_coeff: float = 0):
        super().__init__()
        self.mass = mass
        self.length = length
        self.gravity = gravity
        self.time_res = time_res
        self.control_norm_cost_coeff = control_norm_cost_coeff

    def get_state_dim(self):
        return 2

    def get_control_dim(self):
        return 1

    def compute_time_step_cost(self, state: torch.Tensor, control: torch.Tensor, custom_target_state: torch.Tensor = None):
        if custom_target_state is None:
            target_state = torch.zeros_like(state)
            target_state[:, 0] = torch.pi
        else:
            target_state = custom_target_state

        return torch.norm(state - target_state, dim=1) ** 2 + self.control_norm_cost_coeff * control ** 2

    def _compute_angular_acceleration(self, angle: torch.Tensor, control: torch.Tensor):
        squared_length = self.length ** 2
        return (control.squeeze(dim=1) - self.mass * self.gravity * torch.sin(angle)) / (self.mass * squared_length)

    def forward(self, state: torch.Tensor, control: torch.Tensor, custom_target_state: torch.Tensor = None):
        angle, angular_velocity = state[:, 0], state[:, 1]

        new_angular_velocity = angular_velocity + self.time_res * self._compute_angular_acceleration(angle, control)
        new_angle = angle + self.time_res * angular_velocity

        new_state = torch.stack([new_angle, new_angular_velocity], dim=1)
        cost = self.compute_time_step_cost(state, control, custom_target_state=custom_target_state)
        return new_state, cost


def create_pendulum_dynamics_animation(states: torch.Tensor, controls: torch.Tensor, save_path: str, rod_length: int = 1):
    """
    Creates a pendulum dynamics animation.
    """
    states = states.cpu().detach().numpy()
    angle = states[:, 0]
    angle_velocity = states[:, 1]
    controls = controls.cpu().detach().numpy()

    # Calculate position of pendulum
    x = rod_length * np.sin(angle)
    y = - rod_length * np.cos(angle)

    # Set up figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))

    # Set up pendulum subplot
    ax1.set_xlim(-1.2 * rod_length, 1.2 * rod_length)
    ax1.set_ylim(-1.2 * rod_length, 1.2 * rod_length)
    ax1.set_aspect('equal')

    line, = ax1.plot([], [], '-k', lw=3)
    tip, = ax1.plot([], [], 'o', markersize=5, color='red')

    # Set up control subplot
    ax2.set_xlim(0, len(controls))
    ax2.set_ylim(np.min(controls) - 0.1, np.max(controls) + 0.1)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Control")

    control_line, = ax2.plot([], [], lw=2)
    fig.subplots_adjust(wspace=0.2)

    # Initialize function for animation
    def init():
        line.set_data([], [])
        control_line.set_data([], [])
        return line, control_line

    # Animate function
    def animate(i):
        # Update position of pendulum
        line.set_data([0, x[i]], [0, y[i]])
        tip.set_data([x[i]], [y[i]])
        # Update control input
        control_line.set_data(np.arange(i + 1), controls[:i + 1])
        # Update title with current system state
        ax1.set_title(rf"$\theta (t) = {angle[i]:.3f}$ , " + r"$\frac{d}{dt} \theta (t) =" + f" {angle_velocity[i]:.3f}$", pad=8)
        return line, tip, control_line

    # Create animation object and save as GIF
    anim = FuncAnimation(fig, animate, frames=len(states), interval=100, blit=True, init_func=init)
    anim.save(save_path, dpi=80, writer='imagemagick')
    plt.close(fig)
