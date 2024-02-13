import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import common.evaluation.metrics as metrics
from common.evaluation.evaluators import MetricsEvaluator, Evaluator
from control.models.controlled_dynamical_systems import ControlledDynamicalSystemImpl


class LQRValidationEvaluator(Evaluator):
    TEST_COST_METRIC_NAME = "test cost"
    EXCESS_TEST_COST_METRIC_NAME = "excess test cost"
    TEST_COST_RED_ABS_METRIC_NAME = "test cost reduction absolute"

    MIN_L2_CONTROLLER_TEST_COST_RED_ABS_METRIC_NAME = "min L2 controller test cost reduction absolute"
    NORMALIZED_COST_EXTRAPOLATION_MEASURE_METRIC_NAME = "normalized cost extrapolation measure"
    NORMALIZED_OPT_EXTRAPOLATION_MEASURE_METRIC_NAME = "normalized optimality extrapolation measure"

    def __init__(self, controlled_system: ControlledDynamicalSystemImpl, test_time_horizon: int,
                 data_loader: DataLoader, initial_test_cost: float, min_L2_controller_test_cost: float = None,
                 device=torch.device("cpu")):
        self.controlled_system = controlled_system
        self.test_time_horizon = test_time_horizon
        self.data_loader = data_loader
        self.initial_test_cost = initial_test_cost
        self.min_L2_controller_test_cost = min_L2_controller_test_cost
        self.device = device

        self.metric_infos = self.__create_metric_infos()
        self.metrics = {name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

    def __create_metric_infos(self):
        metric_infos = {
            self.TEST_COST_METRIC_NAME: metrics.MetricInfo(self.TEST_COST_METRIC_NAME, metrics.DummyAveragedMetric()),
        }

        if torch.allclose(self.controlled_system.system.R, torch.zeros_like(self.controlled_system.system.R)):
            metric_infos.update({
                self.EXCESS_TEST_COST_METRIC_NAME: metrics.MetricInfo(self.EXCESS_TEST_COST_METRIC_NAME, metrics.DummyAveragedMetric()),
                self.NORMALIZED_OPT_EXTRAPOLATION_MEASURE_METRIC_NAME: metrics.MetricInfo(self.NORMALIZED_OPT_EXTRAPOLATION_MEASURE_METRIC_NAME,
                                                                                          metrics.DummyAveragedMetric())
            })
            if self.min_L2_controller_test_cost is not None:
                metric_infos.update({
                    self.NORMALIZED_COST_EXTRAPOLATION_MEASURE_METRIC_NAME: metrics.MetricInfo(
                        self.NORMALIZED_COST_EXTRAPOLATION_MEASURE_METRIC_NAME,
                        metrics.DummyAveragedMetric())
                })
        else:
            metric_infos.update({
                self.TEST_COST_RED_ABS_METRIC_NAME: metrics.MetricInfo(self.TEST_COST_RED_ABS_METRIC_NAME, metrics.DummyAveragedMetric()),
            })
            if self.min_L2_controller_test_cost is not None:
                metric_infos.update({
                    self.MIN_L2_CONTROLLER_TEST_COST_RED_ABS_METRIC_NAME: metrics.MetricInfo(self.MIN_L2_CONTROLLER_TEST_COST_RED_ABS_METRIC_NAME,
                                                                                             metrics.DummyAveragedMetric()),
                })

        return metric_infos

    def get_metric_infos(self):
        return self.metric_infos

    def get_metrics(self):
        return self.metrics

    def get_tracked_values(self):
        return self.tracked_values

    def __compute_metrics(self):
        for state in self.data_loader:
            state = state[0]
            state = state.to(self.device)
            states, costs = self.controlled_system(state, self.test_time_horizon)
            cost = costs.sum(dim=1).mean().item()

            cost_metric = self.metrics[self.TEST_COST_METRIC_NAME]
            cost_tracked_value = self.tracked_values[self.TEST_COST_METRIC_NAME]
            cost_metric(cost)
            cost_tracked_value.add_batch_value(cost)

            if torch.allclose(self.controlled_system.system.R, torch.zeros_like(self.controlled_system.system.R)):
                self.__evaluate_excess_test_cost_metrics(cost, costs[:, 0].mean().item())
                self.__evaluate_normalized_opt_measure(state)
            else:
                self.__evaluate_test_cost_reduction_metrics(cost)

    def __evaluate_excess_test_cost_metrics(self, cost, minimal_test_cost):
        excess_test_cost = cost - minimal_test_cost
        excess_cost_metric = self.metrics[self.EXCESS_TEST_COST_METRIC_NAME]
        excess_cost_tracked_value = self.tracked_values[self.EXCESS_TEST_COST_METRIC_NAME]
        excess_cost_metric(excess_test_cost)
        excess_cost_tracked_value.add_batch_value(excess_test_cost)

        if self.min_L2_controller_test_cost is not None:
            min_L2_controller_excess_test_cost = self.min_L2_controller_test_cost - minimal_test_cost
            min_L2_excess_test_ratio = excess_test_cost / min_L2_controller_excess_test_cost
            min_L2_excess_test_cost_ratio_metric = self.metrics[self.NORMALIZED_COST_EXTRAPOLATION_MEASURE_METRIC_NAME]
            min_L2_excess_test_cost_ratio_tracked_value = self.tracked_values[self.NORMALIZED_COST_EXTRAPOLATION_MEASURE_METRIC_NAME]
            min_L2_excess_test_cost_ratio_metric(min_L2_excess_test_ratio)
            min_L2_excess_test_cost_ratio_tracked_value.add_batch_value(min_L2_excess_test_ratio)

    def __evaluate_normalized_opt_measure(self, state: torch.Tensor):
        normalized_opt_measure = LQRValidationEvaluator.compute_normalized_optimality_measure(self.controlled_system, state)
        normalized_opt_measure_metric = self.metrics[self.NORMALIZED_OPT_EXTRAPOLATION_MEASURE_METRIC_NAME]
        normalized_opt_measure_tracked_value = self.tracked_values[self.NORMALIZED_OPT_EXTRAPOLATION_MEASURE_METRIC_NAME]
        normalized_opt_measure_metric(normalized_opt_measure)
        normalized_opt_measure_tracked_value.add_batch_value(normalized_opt_measure)

    def __evaluate_test_cost_reduction_metrics(self, cost):
        test_cost_red_abs = self.initial_test_cost - cost
        test_cost_red_abs_metric = self.metrics[self.TEST_COST_RED_ABS_METRIC_NAME]
        test_cost_red_abs_tracked_value = self.tracked_values[self.TEST_COST_RED_ABS_METRIC_NAME]
        test_cost_red_abs_metric(test_cost_red_abs)
        test_cost_red_abs_tracked_value.add_batch_value(test_cost_red_abs)

        if self.min_L2_controller_test_cost is not None:
            min_L2_test_cost_red_abs = self.min_L2_controller_test_cost - cost
            min_L2_test_cost_red_abs_metric = self.metrics[self.MIN_L2_CONTROLLER_TEST_COST_RED_ABS_METRIC_NAME]
            min_L2_test_cost_red_abs_tracked_value = self.tracked_values[self.MIN_L2_CONTROLLER_TEST_COST_RED_ABS_METRIC_NAME]
            min_L2_test_cost_red_abs_metric(min_L2_test_cost_red_abs)
            min_L2_test_cost_red_abs_tracked_value.add_batch_value(min_L2_test_cost_red_abs)

    @staticmethod
    def compute_normalized_optimality_measure(model: ControlledDynamicalSystemImpl, initial_states: torch.Tensor):
        with torch.no_grad():
            horizon_one_states = torch.matmul(initial_states, model.system.A) + torch.matmul(model.controller(initial_states), model.system.B)
            second_step_cost = F.bilinear(horizon_one_states, horizon_one_states, model.system.Q.unsqueeze(dim=0))
            nominator = second_step_cost.mean().item()

            horizon_one_zero_K_states = torch.matmul(initial_states, model.system.A)
            zero_controller_second_step_cost = F.bilinear(horizon_one_zero_K_states, horizon_one_zero_K_states, model.system.Q.unsqueeze(dim=0))
            denominator = zero_controller_second_step_cost.mean().item()
            return nominator / denominator

    def evaluate(self):
        with torch.no_grad():
            self.controlled_system.to(self.device)
            self.__compute_metrics()
            eval_metric_values = {name: metric.current_value() for name, metric in self.metrics.items()}
            return eval_metric_values
