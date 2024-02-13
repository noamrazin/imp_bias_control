from typing import List

import torch
from torch.utils.data import DataLoader

import common.evaluation.metrics as metrics
from common.evaluation.evaluators import MetricsEvaluator, Evaluator
from control.models.controlled_dynamical_systems import ControlledDynamicalSystemImpl


class ControllerValidationEvaluator(Evaluator):
    TEST_COST_METRIC_NAME = "test cost"
    TEST_COST_RED_ABS_METRIC_NAME = "test cost reduction absolute"
    TEST_COST_RED_PERCENT_METRIC_NAME = "test cost reduction percent"

    def __init__(self, controlled_system: ControlledDynamicalSystemImpl, test_time_horizon: int,
                 data_loader: DataLoader, initial_test_cost: float, device=torch.device("cpu"), average_costs: bool = False):
        self.controlled_system = controlled_system
        self.test_time_horizon = test_time_horizon
        self.data_loader = data_loader
        self.initial_test_cost = initial_test_cost
        self.device = device
        self.average_costs = average_costs

        self.metric_infos = {
            self.TEST_COST_METRIC_NAME: metrics.MetricInfo(self.TEST_COST_METRIC_NAME, metrics.DummyAveragedMetric()),
            self.TEST_COST_RED_ABS_METRIC_NAME: metrics.MetricInfo(self.TEST_COST_RED_ABS_METRIC_NAME, metrics.DummyAveragedMetric()),
            self.TEST_COST_RED_PERCENT_METRIC_NAME: metrics.MetricInfo(self.TEST_COST_RED_PERCENT_METRIC_NAME, metrics.DummyAveragedMetric())
        }
        self.metrics = {name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

    def get_metric_infos(self):
        return self.metric_infos

    def get_metrics(self):
        return self.metrics

    def get_tracked_values(self):
        return self.tracked_values

    def evaluate(self):
        with torch.no_grad():
            self.controlled_system.to(self.device)
            for state in self.data_loader:
                state = state[0]
                state = state.to(self.device)
                states, costs = self.controlled_system(state, self.test_time_horizon)

                if isinstance(costs, List):
                    costs = torch.concat(costs, dim=1)

                cost = costs.sum(dim=1).mean().item() if not self.average_costs else costs.mean().item()

                cost_metric = self.metrics[self.TEST_COST_METRIC_NAME]
                cost_tracked_value = self.tracked_values[self.TEST_COST_METRIC_NAME]
                cost_metric(cost)
                cost_tracked_value.add_batch_value(cost)

                self.__evaluate_test_cost_reduction_metrics(cost)

            eval_metric_values = {name: metric.current_value() for name, metric in self.metrics.items()}
            return eval_metric_values

    def __evaluate_test_cost_reduction_metrics(self, cost):
        test_cost_red_abs = self.initial_test_cost - cost
        test_cost_red_abs_metric = self.metrics[self.TEST_COST_RED_ABS_METRIC_NAME]
        test_cost_red_abs_tracked_value = self.tracked_values[self.TEST_COST_RED_ABS_METRIC_NAME]
        test_cost_red_abs_metric(test_cost_red_abs)
        test_cost_red_abs_tracked_value.add_batch_value(test_cost_red_abs)

        test_cost_red_percent = (test_cost_red_abs / self.initial_test_cost) * 100
        test_cost_red_per_metric = self.metrics[self.TEST_COST_RED_PERCENT_METRIC_NAME]
        test_cost_red_per_tracked_value = self.tracked_values[self.TEST_COST_RED_PERCENT_METRIC_NAME]
        test_cost_red_per_metric(test_cost_red_percent)
        test_cost_red_per_tracked_value.add_batch_value(test_cost_red_percent)
