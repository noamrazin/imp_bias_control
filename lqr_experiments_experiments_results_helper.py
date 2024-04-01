import json
import os
from collections import defaultdict
from pathlib import Path
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['hatch.linewidth'] = 0.7

CONFIG_FILE_PATH = "config.json"
RESULTS_SUMMARY_PATH = "summary.json"

METRIC_NAME_TO_DISPLAY_NAME = {
    "normalized optimality extrapolation measure": "Normalized Optimality\nMeasure of Extrapolation",
    "normalized cost extrapolation measure": "Normalized Cost\nMeasure of Extrapolation"
}

KEY_NAME_TO_DISPLAY_NAME = {
    "num_train_initial_states": "Number of Initial States Seen in Training",
    "train_time_horizon": "H",
    "custom_rnd": "random",
    "shift": "shift",
    "identity": "identity (no extrapolation)",
    "rnd": "random A,B,Q"
}

COLORS = ["#EC5809", "#006AA3", "#5B8F53", "#926381", "#D9AF30", "#720E07"]
COLORS_WITH_RND = ["#EC5809", "#006AA3", "#926381", "#5B8F53", "#D9AF30", "#720E07"]
MARKERS = ["X", "s", "o", "^", "v", "d", "h", "1", "<", "*"]
LINESTYLES = ["solid", "dotted", "dashed"]
BARSTYLES = [None, "//", "+", "--"]


def __load_json_file_from_dir(dir_path: Path, file_name: str):
    config_file_path = dir_path.joinpath(file_name)
    if not config_file_path.exists():
        return None

    with open(config_file_path.as_posix()) as f:
        return json.load(f)


def __should_exclude_experiment(config: dict, keys_to_exclude_values_dict: dict = None):
    for key in keys_to_exclude_values_dict.keys():
        if config[key] in keys_to_exclude_values_dict[key]:
            return True
    return False


def __extract_per_key_metrics_values(dir: List[str], aggr_keys: List[str], metric_names: List[str], keys_to_exclude_values_dict: dict = None):
    per_key_metrics_values = defaultdict(lambda: defaultdict(list))

    experiment_paths = [path for path in Path(dir).iterdir() if path.is_dir()]
    for path in experiment_paths:
        config = __load_json_file_from_dir(path, CONFIG_FILE_PATH)
        summary = __load_json_file_from_dir(path, RESULTS_SUMMARY_PATH)
        if config is None or summary is None:
            continue

        key = tuple([config[key] for key in aggr_keys])
        if keys_to_exclude_values_dict is not None and __should_exclude_experiment(config, keys_to_exclude_values_dict):
            continue

        for metric_name in metric_names:
            metric_value = summary["last_tracked_values"][metric_name]["value"]
            per_key_metrics_values[key][metric_name].append(metric_value)

    return per_key_metrics_values


def __create_and_save_bar_plot_for_two_group_keys(per_first_group_per_second_group_x_metrics: dict, xlabel: str = "",
                                                  ylabel: str = "", title: str = "", bar_width: float = 0.2,
                                                  bar_x_values_spacing: float = 2, bar_within_x_group_spacing: float = 0.15,
                                                  first_group_name_suffix: str = "", y_bottom_lim: float = 0, y_top_lim: float = None,
                                                  figure_height: float = 2.6, figure_width: float = 4.5,
                                                  save_plot_to: str = ""):
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    num_first_groups = len(per_first_group_per_second_group_x_metrics)

    for i, first_group in enumerate(sorted(list(per_first_group_per_second_group_x_metrics.keys()))):
        second_per_group_per_x_metrics = per_first_group_per_second_group_x_metrics[first_group]
        sorted_second_group_names = sorted(list(second_per_group_per_x_metrics.keys()), key=lambda group_name: KEY_NAME_TO_DISPLAY_NAME[group_name])
        for j, second_group in enumerate(sorted_second_group_names):
            per_x_metrics = second_per_group_per_x_metrics[second_group]
            sorted_x_values = sorted(list(per_x_metrics.keys()))
            y_median = np.array([np.median(np.array(per_x_metrics[x])) for x in sorted_x_values])
            y_upper_quartile = np.array([np.percentile(np.array(per_x_metrics[x]), q=75) for x in sorted_x_values])
            y_lower_quartile = np.array([np.percentile(np.array(per_x_metrics[x]), q=25) for x in sorted_x_values])

            bar_positions = np.arange(len(sorted_x_values)) * bar_width * (
                    num_first_groups * len(sorted_second_group_names) + bar_x_values_spacing) + (
                                    j * (num_first_groups + bar_within_x_group_spacing) + i) * bar_width
            ax.bar(bar_positions, y_median, bar_width,
                   label=KEY_NAME_TO_DISPLAY_NAME[second_group] + " " + first_group_name_suffix + f" = {first_group}",
                   color=COLORS[j], alpha=0.7 if i == 0 else 0.55, hatch=BARSTYLES[i])
            ax.errorbar(bar_positions, y_median, yerr=[y_median - y_lower_quartile, y_upper_quartile - y_median],
                        fmt='none', ecolor=COLORS[j], elinewidth=1.2, capsize=2, alpha=1)

    ax.set_ylim(bottom=y_bottom_lim, top=y_top_lim)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    first_bar_positions = np.arange(len(sorted_x_values)) * bar_width * (num_first_groups * len(sorted_second_group_names) + bar_x_values_spacing)
    last_bar_positions = np.arange(len(sorted_x_values)) * bar_width * (num_first_groups * len(sorted_second_group_names) + bar_x_values_spacing) + (
            (len(sorted_second_group_names) - 1) * (num_first_groups + bar_within_x_group_spacing) + num_first_groups - 1) * bar_width

    ax.set_xticks(first_bar_positions + (last_bar_positions - first_bar_positions) / 2)

    ax.set_xticklabels(sorted_x_values)
    ax.tick_params(labelsize=9)
    if title:
        ax.set_title(title, fontsize=13, pad=8)

    plt.legend()
    plt.tight_layout()
    if save_plot_to:
        os.makedirs(os.path.dirname(save_plot_to), exist_ok=True)
        plt.savefig(save_plot_to, bbox_inches='tight', pad_inches=0.1)
        plt.clf()
    else:
        plt.show()


def __create_and_save_bar_plot(per_group_per_x_metrics: dict, xlabel: str = "", ylabel: str = "", title: str = "",
                               bar_width: float = 0.2, y_bottom_lim: float = 0, y_top_lim: float = None,
                               bar_x_values_spacing: float = 2, bar_within_x_group_spacing: float = 0.15,
                               figure_height: float = 2.6, figure_width: float = 4.5, custom_colors: List = None, save_plot_to: str = ""):
    colors = custom_colors if custom_colors else COLORS
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    sorted_group_names = sorted(list(per_group_per_x_metrics.keys()), key=lambda group_name: KEY_NAME_TO_DISPLAY_NAME[group_name])

    for i, group in enumerate(sorted_group_names):
        per_x_metrics = per_group_per_x_metrics[group]
        sorted_x_values = sorted(list(per_x_metrics.keys()))
        y_median = np.array([np.median(np.array(per_x_metrics[x])) for x in sorted_x_values])
        y_upper_quartile = np.array([np.percentile(np.array(per_x_metrics[x]), q=75) for x in sorted_x_values])
        y_lower_quartile = np.array([np.percentile(np.array(per_x_metrics[x]), q=25) for x in sorted_x_values])

        bar_positions = (np.arange(len(sorted_x_values)) * bar_width * (len(sorted_group_names) + bar_x_values_spacing) +
                         i * bar_width * (1 + bar_within_x_group_spacing))
        ax.bar(bar_positions, y_median, bar_width, label=KEY_NAME_TO_DISPLAY_NAME[group], color=colors[i], alpha=0.7)
        ax.errorbar(bar_positions, y_median, yerr=[y_median - y_lower_quartile, y_upper_quartile - y_median],
                    fmt='none', ecolor=colors[i], elinewidth=1.2, capsize=3, alpha=1)

    ax.set_ylim(bottom=y_bottom_lim, top=y_top_lim)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    first_bar_positions = np.arange(len(sorted_x_values)) * bar_width * (len(sorted_group_names) + bar_x_values_spacing)
    last_bar_positions = (np.arange(len(sorted_x_values)) * bar_width * (len(sorted_group_names) + bar_x_values_spacing) +
                          (len(sorted_group_names) - 1) * bar_width * (1 + bar_within_x_group_spacing))

    ax.set_xticks(first_bar_positions + (last_bar_positions - first_bar_positions) / 2)

    ax.set_xticklabels(sorted_x_values)
    ax.tick_params(labelsize=9)

    if title:
        ax.set_title(title, fontsize=13, pad=8)

    plt.legend()
    plt.tight_layout()
    if save_plot_to:
        os.makedirs(os.path.dirname(save_plot_to), exist_ok=True)
        plt.savefig(save_plot_to, bbox_inches='tight', pad_inches=0.1)
        plt.clf()
    else:
        plt.show()


def plot_metric_for_two_group_keys(experiments_dir: str, first_group_key: str, second_group_key: str, x_axis_key: str, metric_name: str,
                                   keys_to_exclude_values_dict: dict = None, save_plot_to: str = ""):
    per_key_metrics_values = __extract_per_key_metrics_values(experiments_dir, [first_group_key, second_group_key, x_axis_key], [metric_name],
                                                              keys_to_exclude_values_dict)

    per_first_group_per_second_group_x_metrics = defaultdict(lambda: defaultdict(dict))
    for key, metric_values_dict in per_key_metrics_values.items():
        per_first_group_per_second_group_x_metrics[key[0]][key[1]][key[2]] = metric_values_dict[metric_name]

    print("====================================")
    print(f"Creating plot for data:")
    print(f"First group key: {first_group_key}")
    for first_group, per_second_group_metrics in per_first_group_per_second_group_x_metrics.items():
        for second_group, per_x_metrics in per_second_group_metrics.items():
            print(f"First group: {first_group} , Second group: {second_group}")
            print(f"x values: {sorted(list(per_x_metrics.keys()))}")
            for x, metric_values in per_x_metrics.items():
                print(f"Number of y values for x value {x} is: {len(metric_values)}")

    __create_and_save_bar_plot_for_two_group_keys(per_first_group_per_second_group_x_metrics,
                                                  xlabel=KEY_NAME_TO_DISPLAY_NAME[x_axis_key],
                                                  ylabel=METRIC_NAME_TO_DISPLAY_NAME[metric_name],
                                                  first_group_name_suffix=KEY_NAME_TO_DISPLAY_NAME[first_group_key],
                                                  save_plot_to=save_plot_to)


def plot_metric(experiments_dir: str, group_key: str, x_axis_key: str, metric_name: str, keys_to_exclude_values_dict: dict = None,
                custom_colors: List = None, save_plot_to: str = ""):
    per_key_metrics_values = __extract_per_key_metrics_values(experiments_dir, [group_key, x_axis_key], [metric_name],
                                                              keys_to_exclude_values_dict)

    per_group_per_x_metrics = defaultdict(dict)
    for key, metric_values_dict in per_key_metrics_values.items():
        per_group_per_x_metrics[key[0]][key[1]] = metric_values_dict[metric_name]

    print("====================================")
    print(f"Creating plot for data:")
    for group, per_x_metrics in per_group_per_x_metrics.items():
        print(f"Group: {group}")
        print(f"x values: {sorted(list(per_x_metrics.keys()))}")
        for x, metric_values in per_x_metrics.items():
            print(f"Number of y values for x value {x} is: {len(metric_values)}")
    print("====================================")

    __create_and_save_bar_plot(per_group_per_x_metrics, xlabel=KEY_NAME_TO_DISPLAY_NAME[x_axis_key], ylabel=METRIC_NAME_TO_DISPLAY_NAME[metric_name],
                               custom_colors=custom_colors, save_plot_to=save_plot_to)


def print_metrics(experiments_dir: str, aggr_keys: List[str], metric_names: List[str], keys_to_exclude_values_dict: dict = None):
    per_key_metrics_values = __extract_per_key_metrics_values(experiments_dir, aggr_keys, metric_names, keys_to_exclude_values_dict)

    print(f"Experiments directory: {experiments_dir}\n"
          f"Printing metrics: {metric_names}\n"
          f"Grouping by keys: {aggr_keys}\n")

    sorted_keys = sorted(per_key_metrics_values.keys())
    for key in sorted_keys:
        metric_values_dict = per_key_metrics_values[key]
        for metric_name, metric_values in metric_values_dict.items():
            np_metric_values = np.array(metric_values)
            print("------------------------------------------------------------------------")
            print(f"Key: {key} | Metric: {metric_name} | Num values: {np_metric_values.shape[0]} | median {np.median(np_metric_values)} , min {np_metric_values.min()} , "
                  f"max {np_metric_values.max()} , lower quartile {np.percentile(np_metric_values, q=25)} , "
                  f"upper quartile {np.percentile(np_metric_values, q=75)}")


def main():
    print_metrics(experiments_dir="outputs/lqr",
                  aggr_keys=["linear_system", "num_train_initial_states", "train_time_horizon"],
                  metric_names=["normalized cost extrapolation measure", "normalized optimality extrapolation measure", "excess train cost",
                                "train cost"])

    # Plot optimality and cost measures of extrapolation for experiments corresponding to those reported in Figure 1 of the paper
    plot_metric(experiments_dir="outputs/lqr",
                group_key="linear_system",
                x_axis_key="num_train_initial_states",
                metric_name="normalized optimality extrapolation measure",
                keys_to_exclude_values_dict={
                    "linear_system": ["rnd"],
                    "train_time_horizon": [8]
                },
                save_plot_to="outputs/plots/lqr/e_opt_h5.pdf")
    plot_metric(experiments_dir="outputs/lqr",
                group_key="linear_system",
                x_axis_key="num_train_initial_states",
                metric_name="normalized cost extrapolation measure",
                keys_to_exclude_values_dict={
                    "linear_system": ["rnd"],
                    "train_time_horizon": [8]
                },
                save_plot_to="outputs/plots/lqr/e_cost_h5.pdf")

    # Plot optimality and cost measures of extrapolation for experiments corresponding to those reported in Figure 3 of the paper
    plot_metric_for_two_group_keys(experiments_dir="outputs/lqr",
                                   first_group_key="train_time_horizon",
                                   second_group_key="linear_system",
                                   x_axis_key="num_train_initial_states",
                                   metric_name="normalized optimality extrapolation measure",
                                   keys_to_exclude_values_dict={
                                       "linear_system": ["rnd"]
                                   },
                                   save_plot_to="outputs/plots/lqr/e_opt_h8.pdf")
    plot_metric_for_two_group_keys("outputs/lqr",
                                   first_group_key="train_time_horizon",
                                   second_group_key="linear_system",
                                   x_axis_key="num_train_initial_states",
                                   metric_name="normalized cost extrapolation measure",
                                   keys_to_exclude_values_dict={
                                       "linear_system": ["rnd"]
                                   },
                                   save_plot_to="outputs/plots/lqr/e_cost_h8.pdf")

    # Plot optimality and cost measures of extrapolation for experiments corresponding to those reported in Figure 4 of the paper
    plot_metric("outputs/lqr",
                group_key="linear_system",
                x_axis_key="num_train_initial_states",
                metric_name="normalized optimality extrapolation measure",
                keys_to_exclude_values_dict={
                    "train_time_horizon": [8]
                },
                custom_colors=COLORS_WITH_RND,
                save_plot_to="outputs/plots/lqr/e_opt_h5_rnd_B_Q.pdf")

    plot_metric("outputs/lqr",
                group_key="linear_system",
                x_axis_key="num_train_initial_states",
                metric_name="normalized cost extrapolation measure",
                keys_to_exclude_values_dict={
                    "train_time_horizon": [8]
                },
                custom_colors=COLORS_WITH_RND,
                save_plot_to="outputs/plots/lqr/e_cost_h5_rnd_B_Q.pdf")


if __name__ == "__main__":
    main()
