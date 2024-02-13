import matplotlib as mpl
import numpy as np
import torch

from control.models.quadcopter import euler_matrix

mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

"""
Adapted from: https://github.com/DiffEqML/torchcontrol.
"""


# Cube util function
def cuboid_data2(pos, size=(1, 1, 1), rotation=None):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:, :, i] *= size[i]
    if rotation is not None:
        for i in range(4):
            X[:, i, :] = np.dot(rotation, X[:, i, :].T).T
    X += pos
    return X


# Plot cube for drone body
def plot_cube(position, size=None, rotation=None, color=None, **kwargs):
    if not isinstance(color, (list, np.ndarray)):
        color = ["C0"] * len(position)
    if not isinstance(size, (list, np.ndarray)):
        size = [(1, 1, 1)] * len(position)

    g = cuboid_data2(position, size=size, rotation=rotation)
    return Poly3DCollection(g, facecolor=np.repeat(color, 6), **kwargs)


def save_quadcopter_trajectory_3d(traj, target_state_pos, save_path: str, end_time_step: int = -1, min_lim: float = -1, max_lim: float = 1):
    '''
    Plot trajectory of the drone up to the i-th element
    Args
        traj: drone trajectory
        target_state_pos: target state
        end_time_step: plot until this frame. If -1 will plot all frames.
    '''
    end_time_step = end_time_step if end_time_step >= 0 else traj.shape[0]
    fig = plt.figure(figsize=(4, 4))
    ax = plt.axes(projection='3d')

    if isinstance(traj, torch.Tensor):
        traj = traj.detach().cpu().numpy()

    # For visualization
    scale = 1.5
    s = 50
    dxm = scale * 0.16  # arm length (m)
    dym = scale * 0.16  # arm length (m)
    dzm = scale * 0.05  # motor height (m)
    lw = scale
    drone_size = [dxm / 2, dym / 2, dzm]

    ax.set_xlim3d(min_lim, max_lim)
    ax.set_ylim3d(min_lim, max_lim)
    ax.set_zlim3d(min_lim, max_lim)

    l1, = ax.plot([], [], [], lw=lw, color='red')
    l2, = ax.plot([], [], [], lw=lw, color='green')

    initial = traj[0]

    init = ax.scatter(initial[0], initial[1], initial[2], marker='^', color='green', label='Initial Position', s=s)
    fin = ax.scatter(target_state_pos[0], target_state_pos[1], target_state_pos[2], marker='*', color="#ffbf00", label='Target',
                     s=s, edgecolor="black")  # set linestyle to none

    ax.plot(traj[:end_time_step, 0], traj[:end_time_step, 1], traj[:end_time_step, 2], alpha=1, linestyle='-')
    pos = traj[end_time_step - 1]
    x = pos[0]
    y = pos[1]
    z = pos[2]

    # Trick to reuse the same function
    R = euler_matrix(torch.Tensor([pos[3]]), torch.Tensor([pos[4]]), torch.Tensor([pos[5]])).numpy().squeeze(0)
    motorPoints = np.array([[dxm, -dym, dzm], [0, 0, 0], [dxm, dym, dzm], [-dxm, dym, dzm], [0, 0, 0], [-dxm, -dym, dzm], [-dxm, -dym, -dzm]])
    motorPoints = np.dot(R, np.transpose(motorPoints))
    motorPoints[0, :] += x
    motorPoints[1, :] += y
    motorPoints[2, :] += z

    # Motors
    l1.set_data(motorPoints[0, 0:3], motorPoints[1, 0:3])
    l1.set_3d_properties(motorPoints[2, 0:3])
    l2.set_data(motorPoints[0, 3:6], motorPoints[1, 3:6])
    l2.set_3d_properties(motorPoints[2, 3:6])

    # Body
    pos = ((motorPoints[:, 6] + 2 * motorPoints[:, 1]) / 3)
    body = plot_cube(pos, drone_size, rotation=R, edgecolor="k")
    ax.add_collection3d(body)

    ax.legend()
    ax.set_xlabel(f'$x$')
    ax.set_ylabel(f'$y$')
    ax.set_zlabel(f'$z$')

    ax.set_title(f'Final Position ({x:.2f},{y:.2f},{z:.2f})', pad=8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.52, -0.05), fancybox=True, shadow=False, ncol=3)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def save_quadcopter_trajectory_3d_animation(traj, target_state_pos, save_path: str, min_lim: float = -1, max_lim: float = 1):
    '''
    Animate drone and save gif
    Args
        traj: drone trajectory
        target_state_pos: target state position (tensor of shape (3,))
        path: save path for
    '''

    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes(projection='3d')

    if isinstance(traj, torch.Tensor):
        traj = traj.detach().cpu().numpy()

    # For visualization
    scale = 1.5
    s = 50
    dxm = scale * 0.16  # arm length (m)
    dym = scale * 0.16  # arm length (m)
    dzm = scale * 0.05  # motor height (m)
    drone_size = [dxm / 2, dym / 2, dzm]

    ax.set_xlim3d(min_lim, max_lim)
    ax.set_ylim3d(min_lim, max_lim)
    ax.set_zlim3d(min_lim, max_lim)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    l1, = ax.plot([], [], [], lw=2, color='red')
    l2, = ax.plot([], [], [], lw=2, color='green')

    initial = traj[0]
    tr = traj

    init = ax.scatter(initial[0], initial[1], initial[2], marker='^', color="green", label='Initial Position', s=s)
    fin = ax.scatter(target_state_pos[0], target_state_pos[1], target_state_pos[2], marker='*', color="#ffbf00", label='Target',
                     s=s, edgecolor="black")  # set linestyle to none

    ax.legend()
    ax.legend(loc='upper center', bbox_to_anchor=(0.52, -0.05), fancybox=True, shadow=False, ncol=3)

    # Single frame plotting
    def get_frame(i):
        del ax.collections[:]  # remove previous 3D elements
        init = ax.scatter(initial[0], initial[1], initial[2], marker='^', color='blue', label='Initial Position', s=s)
        fin = ax.scatter(target_state_pos[0], target_state_pos[1], target_state_pos[2], marker='*', color="ffbf00", label='Target',
                         s=s, edgecolor="black")  # set linestyle to none

        ax.plot(tr[:i, 0], tr[:i, 1], tr[:i, 2], alpha=0.1, linestyle='-', color='tab:blue')
        pos = tr[i]
        x = pos[0]
        y = pos[1]
        z = pos[2]

        # Trick to reuse the same function
        R = euler_matrix(torch.Tensor([pos[3]]), torch.Tensor([pos[4]]), torch.Tensor([pos[5]])).numpy().squeeze(0)
        motorPoints = np.array([[dxm, -dym, dzm], [0, 0, 0], [dxm, dym, dzm], [-dxm, dym, dzm], [0, 0, 0], [-dxm, -dym, dzm], [-dxm, -dym, -dzm]])
        motorPoints = np.dot(R, np.transpose(motorPoints))
        motorPoints[0, :] += x
        motorPoints[1, :] += y
        motorPoints[2, :] += z

        # Motors
        l1.set_data(motorPoints[0, 0:3], motorPoints[1, 0:3])
        l1.set_3d_properties(motorPoints[2, 0:3])
        l2.set_data(motorPoints[0, 3:6], motorPoints[1, 3:6])
        l2.set_3d_properties(motorPoints[2, 3:6])

        # Body
        pos = ((motorPoints[:, 6] + 2 * motorPoints[:, 1]) / 3)
        body = plot_cube(pos, drone_size, rotation=R, edgecolor="k")
        ax.add_collection3d(body)
        ax.set_title(f'Quadcopter Trajectory: Position ({x:.2f},{y:.2f},{z:.2f}) Time Step {i}', pad=8)

    an = FuncAnimation(fig, get_frame, init_func=None, frames=traj.shape[0], interval=20, blit=False)
    an.save(save_path, dpi=80, writer='imagemagick', fps=20)
    plt.close(fig)


def save_quadcopter_trajectory_state_info_plot(traj, save_path: str):
    '''
    Simple plot with all variables in time.
    '''

    fig, axs = plt.subplots(12, 1, figsize=(10, 10))

    axis_labels = ['$x$', '$y$', '$z$', '$\phi$', r'$\theta$', '$\psi$', '$\dot x$', '$\dot y$',
                   '$\dot z$', '$\dot \phi$', r'$\dot \theta$', '$\dot \psi$']

    for ax, i, axis_label in zip(axs, range(len(axs)), axis_labels):
        ax.plot(traj[:, i].cpu().detach(), color='tab:red')
        ax.label_outer()
        ax.set_ylabel(axis_label)

    fig.suptitle('Trajectory States', y=0.92)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def save_quadcopter_controls_plot(controls, save_path: str):
    '''
    Simple plot with all variables in time
    '''

    fig, axs = plt.subplots(4, 1, figsize=(10, 5))

    axis_labels = ['$u_0$ RPM', '$u_1$ RPM', '$u_2$ RPM', '$u_3$ RPM']

    for ax, i, axis_label in zip(axs, range(len(axs)), axis_labels):
        ax.plot(controls[:, i].cpu().detach(), color='tab:red')
        ax.label_outer()
        ax.set_ylabel(axis_label)

    fig.suptitle('Trajectory Controls', y=0.94)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
