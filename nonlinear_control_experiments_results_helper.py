import json
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

PENDULUM_TASK = "pendulum"
QUADCOPTER_TASK = "quadcopter"
MARKERS = ("o", "s", "X", "^", "P", "v", "<", ">", "D", "H", "p")
RESULTS_SUMMARY_PATH = "summary.json"


def __load_json_file_from_dir(dir_path: Path, file_name: str):
    config_file_path = dir_path.joinpath(file_name)
    if not config_file_path.exists():
        return None

    with open(config_file_path.as_posix()) as f:
        return json.load(f)


def __load_experiments_summaries(dir_exps: str, verbose: bool = False):
    summaries = []
    experiment_paths = [path for path in Path(dir_exps).iterdir() if path.is_dir()]
    for path in experiment_paths:
        summary = __load_json_file_from_dir(path, RESULTS_SUMMARY_PATH)
        if summary is None:
            continue
        summaries.append(summary)

    if verbose:
        print(f"Loaded {len(summaries)} experiment summaries from: {dir_exps}")
    return summaries


def __load_train_and_test_states_from_summary_at_time_step_percentage(summary: dict, time_step_percentage: int = 0):
    if time_step_percentage == 0:
        return np.array(summary["train initial states"]), np.array(summary["test initial states"])
    elif time_step_percentage == 100:
        return np.array(summary["train final states"]), np.array(summary["test final states"])
    else:
        return np.array(summary[f"train {time_step_percentage}% steps states"]), np.array(summary[f"test {time_step_percentage}% steps states"])


def __plot_pendulum(train_angles: np.ndarray, test_angles: np.ndarray,
                    train_color: str = "#0242A1", test_color: str = "#CB3535", target_color: str = "#EEB609",
                    target_angle: float = np.pi, save_plot_to: str = ""):
    fig, ax = plt.subplots()

    circle = plt.Circle((0, 0), 0.1, color='gray', linestyle='--', linewidth=2, fill=False, alpha=0.8)
    ax.add_patch(circle)

    # Reduce angles by pi / 2 so that 0 angle means aligned with x axis
    train_angles = train_angles - np.pi / 2
    test_angles = test_angles - np.pi / 2

    # Plotting train states
    train_x = 0.1 * np.cos(train_angles)
    train_y = 0.1 * np.sin(train_angles)
    for i in range(len(train_angles)):
        ax.plot([0, train_x[i]], [0, train_y[i]], "k-", alpha=0.7, linewidth=5, zorder=1)  # Pendulum rods
        ax.scatter(train_x[i], train_y[i], marker=MARKERS[- i % len(MARKERS) - 1], color=train_color, s=600, zorder=2)

    # Plotting test states
    test_x = 0.1 * np.cos(test_angles)
    test_y = 0.1 * np.sin(test_angles)
    for i in range(len(test_angles)):
        ax.plot([0, test_x[i]], [0, test_y[i]], "k-", alpha=0.7, linewidth=5, zorder=1)  # Pendulum rods
        ax.scatter(test_x[i], test_y[i], marker=MARKERS[i % len(MARKERS)], color=test_color, s=600, zorder=2)

    # Plot target state, reduce angle by pi / 2 so that 0 angle means aligned with x axis
    target_x = 0.1 * np.cos(target_angle - np.pi / 2)
    target_y = 0.1 * np.sin(target_angle - np.pi / 2)
    ax.scatter([target_x], [target_y], color=target_color, s=850, marker="*", edgecolor="black", zorder=2)

    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_xlim(-0.11, 0.11)
    ax.set_ylim(-0.11, 0.11)

    plt.tight_layout()
    if save_plot_to:
        os.makedirs(os.path.dirname(save_plot_to), exist_ok=True)
        plt.savefig(save_plot_to, bbox_inches='tight', pad_inches=0.1, transparent=True, dpi=300)
        plt.clf()
    else:
        plt.show()


def __plot_quadcopter(train_positions: np.ndarray, test_positions: np.ndarray,
                      train_color: str = "#0242A1", test_color: str = "#CB3535", target_color: str = "#EEB609",
                      target_position: Tuple[float] = (0, 0, 1), view_elev: float = 10, view_azim: float = 45,
                      x_lims: Tuple[float, float] = (-0.5, 0.5), y_lims: Tuple[float, float] = (-0.5, 0.5), z_lims: Tuple[float, float] = (-0.1, 1.2),
                      save_plot_to: str = ""):
    ax = plt.figure().add_subplot(projection='3d')
    ax.view_init(view_elev, view_azim)

    for i in range(train_positions.shape[0]):
        ax.scatter(train_positions[i, 0], train_positions[i, 1], train_positions[i, 2], marker=MARKERS[- i % len(MARKERS) - 1], color=train_color,
                   s=120)

    for i in range(test_positions.shape[0]):
        ax.scatter(test_positions[i, 0], test_positions[i, 1], test_positions[i, 2], marker=MARKERS[i % len(MARKERS)], color=test_color, s=120)

    ax.scatter(target_position[0], target_position[1], target_position[2], color=target_color, s=350, marker="*", edgecolor="black")

    ax.set_xlabel(f'$x$', fontsize=13, labelpad=5)
    ax.set_ylabel(f'$y$', fontsize=13, labelpad=5)
    ax.set_zlabel(f'$z$', fontsize=13, labelpad=5)
    ax.tick_params(labelsize=11)

    if x_lims is not None:
        ax.set_xlim3d(x_lims[0], x_lims[1])
    if y_lims is not None:
        ax.set_ylim3d(y_lims[0], y_lims[1])
    if z_lims is not None:
        ax.set_zlim3d(z_lims[0], z_lims[1])

    plt.tight_layout()
    if save_plot_to:
        os.makedirs(os.path.dirname(save_plot_to), exist_ok=True)
        plt.savefig(save_plot_to, bbox_inches='tight', pad_inches=0.1, transparent=True, dpi=300)
        plt.clf()
    else:
        plt.show()


def __plot_pendulum_states_for_time_step_percentage(summary: dict, time_step_percentage: int = 0,
                                                    file_name: str = "", save_plot_to_dir: str = ""):
    train_states, test_states = __load_train_and_test_states_from_summary_at_time_step_percentage(summary, time_step_percentage)
    __plot_pendulum(train_states[:, 0], test_states[:, 0],
                    save_plot_to=os.path.join(save_plot_to_dir, f"{file_name}.pdf") if save_plot_to_dir else "")


def __plot_quadcopter_states_for_time_step_percentage(summary: dict, time_step_percentage: int = 0, file_name: str = "", save_plot_to_dir: str = "",
                                                      view_elev: float = 10, view_azim: float = 45, x_lims: Tuple[float, float] = (-0.5, 0.5),
                                                      y_lims: Tuple[float, float] = (-0.5, 0.5), z_lims: Tuple[float, float] = (-0.1, 1.2)):
    train_states, test_states = __load_train_and_test_states_from_summary_at_time_step_percentage(summary, time_step_percentage)
    __plot_quadcopter(train_states[:, 0:3], test_states[:, 0:3],
                      save_plot_to=os.path.join(save_plot_to_dir, f"{file_name}.pdf") if save_plot_to_dir else "",
                      view_elev=view_elev, view_azim=view_azim, x_lims=x_lims, y_lims=y_lims, z_lims=z_lims)


def __plot_states(dir_exps: str, task: str = PENDULUM_TASK, save_plot_to_dir: str = "", time_step_percentage_interval: int = 20, **kwargs):
    exp_summaries = __load_experiments_summaries(dir_exps)
    exp_test_costs = np.array([summary["best_score_epoch_tracked_values"]["test cost"]["value"] for summary in exp_summaries])
    summary_of_median_test_cost_exp = exp_summaries[np.argsort(exp_test_costs)[len(exp_test_costs) // 2]]

    for h in range(0, 100 + time_step_percentage_interval, time_step_percentage_interval):
        if task == PENDULUM_TASK:
            __plot_pendulum_states_for_time_step_percentage(summary_of_median_test_cost_exp,
                                                            time_step_percentage=h,
                                                            file_name=f"{task}_median_{h}_percent_steps_states",
                                                            save_plot_to_dir=save_plot_to_dir)
        elif task == QUADCOPTER_TASK:
            __plot_quadcopter_states_for_time_step_percentage(summary_of_median_test_cost_exp,
                                                              time_step_percentage=h,
                                                              file_name=f"{task}_median_{h}_percent_steps_states",
                                                              save_plot_to_dir=save_plot_to_dir,
                                                              **kwargs)
        else:
            raise ValueError(f"Unknown task {task}")


def __plot_baseline_states(dir_baseline_exps: str, task: str = PENDULUM_TASK, save_plot_to_dir: str = "", time_step_percentage_interval: int = 20,
                           **kwargs):
    baseline_exp_summaries = __load_experiments_summaries(dir_baseline_exps)
    baseline_exp_train_costs = np.array([summary["best_score_epoch_tracked_values"]["train cost"]["value"] for summary in baseline_exp_summaries])
    baseline_exp_summary = baseline_exp_summaries[baseline_exp_train_costs.argmin()]

    for h in range(0, 100 + time_step_percentage_interval, time_step_percentage_interval):
        if task == PENDULUM_TASK:
            __plot_pendulum_states_for_time_step_percentage(baseline_exp_summary,
                                                            time_step_percentage=h,
                                                            file_name=f"{task}_baseline_{h}_percent_steps_states",
                                                            save_plot_to_dir=save_plot_to_dir)
        elif task == QUADCOPTER_TASK:
            __plot_quadcopter_states_for_time_step_percentage(baseline_exp_summary,
                                                              time_step_percentage=h,
                                                              file_name=f"{task}_baseline_{h}_percent_steps_states",
                                                              save_plot_to_dir=save_plot_to_dir,
                                                              **kwargs)
        else:
            raise ValueError(f"Unknown task {task}")


def plot_pendulum_states_figures(dir_exps: str, dir_baseline_exps: str, save_plot_to_dir: str = ""):
    __plot_states(dir_exps, task=PENDULUM_TASK, save_plot_to_dir=save_plot_to_dir, time_step_percentage_interval=20)
    __plot_baseline_states(dir_baseline_exps, task=PENDULUM_TASK, save_plot_to_dir=save_plot_to_dir, time_step_percentage_interval=20)


def plot_quadcopter_states_figures(dir_exps: str, dir_baseline_exps: str, save_plot_to_dir: str = "",
                                   view_elev: float = 10, view_azim: float = 45, x_lims: Tuple[float, float] = (-0.5, 0.5),
                                   y_lims: Tuple[float, float] = (-0.5, 0.5), z_lims: Tuple[float, float] = (-0.1, 1.2)):
    __plot_states(dir_exps, task=QUADCOPTER_TASK, save_plot_to_dir=save_plot_to_dir, time_step_percentage_interval=20,
                  view_elev=view_elev, view_azim=view_azim, x_lims=x_lims, y_lims=y_lims, z_lims=z_lims)
    __plot_baseline_states(dir_baseline_exps, task=QUADCOPTER_TASK, save_plot_to_dir=save_plot_to_dir, time_step_percentage_interval=20,
                           view_elev=view_elev, view_azim=view_azim, x_lims=x_lims, y_lims=y_lims, z_lims=z_lims)


def print_cost_metrics(dir_exps: str, dir_baseline_exps: str, dir_min_test_cost_exps: str):
    print("============================================================")
    print(f"dir_exps: {dir_exps}\ndir_baseline_exps: {dir_baseline_exps}\ndir_min_test_cost_exps: {dir_min_test_cost_exps}")
    print("============================================================")

    exp_summaries = __load_experiments_summaries(dir_exps, verbose=True)
    baseline_exp_summaries = __load_experiments_summaries(dir_baseline_exps, verbose=True)
    min_test_cost_exp_summaries = __load_experiments_summaries(dir_min_test_cost_exps, verbose=True)

    exp_train_costs = np.array([summary["best_score_epoch_tracked_values"]["train cost"]["value"] for summary in exp_summaries])
    exp_test_costs = np.array([summary["best_score_epoch_tracked_values"]["test cost"]["value"] for summary in exp_summaries])
    baseline_exp_train_costs = np.array([summary["best_score_epoch_tracked_values"]["train cost"]["value"] for summary in baseline_exp_summaries])
    baseline_exp_test_costs = np.array([summary["best_score_epoch_tracked_values"]["test cost"]["value"] for summary in baseline_exp_summaries])
    min_test_cost_exp_test_costs = np.array(
        [summary["best_score_epoch_tracked_values"]["test cost"]["value"] for summary in min_test_cost_exp_summaries])

    print("============================================================")
    test_cost_sorting_indices = np.argsort(exp_test_costs)
    median_test_cost_index = test_cost_sorting_indices[len(test_cost_sorting_indices) // 2]
    lower_quartile_test_cost_index = test_cost_sorting_indices[len(test_cost_sorting_indices) // 4]
    upper_quartile_test_cost_index = test_cost_sorting_indices[3 * len(test_cost_sorting_indices) // 4]

    print(f"Minimal test cost: {exp_test_costs.min()} , "
          f"Median test cost: {exp_test_costs[median_test_cost_index]} , "
          f"Lower quartile test cost {exp_test_costs[lower_quartile_test_cost_index]} , "
          f"Upper quartile test cost {exp_test_costs[upper_quartile_test_cost_index]}")
    print(f"Minimal train cost: {exp_train_costs.min()} , "
          f"Train cost of median test cost: {exp_train_costs[median_test_cost_index]} , "
          f"Train cost of lower quartile test cost {exp_train_costs[lower_quartile_test_cost_index]} , "
          f"Train cost of upper quartile test cost {exp_train_costs[upper_quartile_test_cost_index]}")

    print("============================================================")
    min_test_cost = min_test_cost_exp_test_costs.min()
    print(f"Minimal test cost (as measured when fitting only test states): {min_test_cost}")
    print(f"Minimal baseline train cost: {baseline_exp_train_costs.min()}")
    print(f"Test cost of minimal train cost baseline: {baseline_exp_test_costs[baseline_exp_train_costs.argmin()]}")

    print("============================================================")
    print(f"Normalized cost measure of extrapolation based on median test cost: "
          f"{(np.median(exp_test_costs) - min_test_cost) / (baseline_exp_test_costs[baseline_exp_train_costs.argmin()] - min_test_cost)}")
    print(f"Normalized cost measure of extrapolation based on upper quartile test cost: "
          f"{(np.percentile(exp_test_costs, q=75) - min_test_cost) / (baseline_exp_test_costs[baseline_exp_train_costs.argmin()] - min_test_cost)}")
    print(f"Normalized cost measure of extrapolation based on lower quartile test cost: "
          f"{(np.percentile(exp_test_costs, q=25) - min_test_cost) / (baseline_exp_test_costs[baseline_exp_train_costs.argmin()] - min_test_cost)}")
    print("============================================================")


def main():
    # Print metrics and plot states for pendulum experiments corresponding to those reported in Figure 2 of the paper
    print("***************************************************************************************************")
    print("Printing metrics for pendulum experiments")
    print_cost_metrics(dir_exps="outputs/pend/ext", dir_baseline_exps="outputs/pend/no_ext", dir_min_test_cost_exps="outputs/pend/fit_test")
    plot_pendulum_states_figures(dir_exps="outputs/pend/ext", dir_baseline_exps="outputs/pend/no_ext", save_plot_to_dir="outputs/plots/pend")
    print("***************************************************************************************************")

    # Print metrics and plot states for quadcopter experiments corresponding to those reported in Figure 2 of the paper
    print("\n\n***************************************************************************************************")
    print("Printing metrics for quadcopter extrapolation to initial states of lower height experiments")
    print_cost_metrics(dir_exps="outputs/quad/ext", dir_baseline_exps="outputs/quad/no_ext", dir_min_test_cost_exps="outputs/quad/fit_test")
    plot_quadcopter_states_figures(dir_exps="outputs/quad/ext", dir_baseline_exps="outputs/quad/no_ext", save_plot_to_dir="outputs/plots/quad_below")
    print("***************************************************************************************************")

    # Print metrics and plot states for quadcopter experiments corresponding to those reported in Figure 7 of the paper
    print("\n\n***************************************************************************************************")
    print("Printing metrics for quadcopter extrapolation to initial states of lower height with additional unseen initial states experiments")
    print_cost_metrics(dir_exps="outputs/quad_add/ext", dir_baseline_exps="outputs/quad_add/no_ext",
                       dir_min_test_cost_exps="outputs/quad_add/fit_test")
    plot_quadcopter_states_figures(dir_exps="outputs/quad_add/ext", dir_baseline_exps="outputs/quad_add/no_ext",
                                   save_plot_to_dir="outputs/plots/quad_add_test")
    print("***************************************************************************************************")

    # Print metrics and plot states for quadcopter experiments corresponding to those reported in Figure 9 of the paper
    print("\n\n***************************************************************************************************")
    print("Printing metrics for quadcopter extrapolation to initial states of different horizontal distances experiments")
    print_cost_metrics(dir_exps="outputs/quad_dist/ext", dir_baseline_exps="outputs/quad_dist/no_ext",
                       dir_min_test_cost_exps="outputs/quad_dist/fit_test")
    plot_quadcopter_states_figures(dir_exps="outputs/quad_dist/ext", dir_baseline_exps="outputs/quad_dist/no_ext",
                                   save_plot_to_dir="outputs/plots/quad_dist",
                                   view_elev=25, x_lims=(-0.5, 1.25), y_lims=(-0.5, 0.5), z_lims=(-0.1, 1.2))
    print("***************************************************************************************************")


if __name__ == "__main__":
    main()
