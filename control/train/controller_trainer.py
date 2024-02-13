from typing import List

import torch

from common.evaluation.evaluators.evaluator import VoidEvaluator
from common.train.trainer import Trainer
from control.models.controlled_dynamical_systems import ControlledDynamicalSystem


class ControllerTrainer(Trainer):

    def __init__(self, controlled_system: ControlledDynamicalSystem, optimizer, train_time_horizon: int,
                 train_evaluator=VoidEvaluator(), val_evaluator=VoidEvaluator(), callback=None,
                 device=torch.device("cpu"), average_costs: bool = True, adv_initial_states: torch.Tensor = None,
                 adversarial_initial_states_cost_coeff: float = 0.1,
                 adv_initial_states_kwargs: dict = None):
        super().__init__(controlled_system, optimizer, train_evaluator, val_evaluator, callback, device)
        self.controlled_system = controlled_system
        self.train_time_horizon = train_time_horizon
        self.average_costs = average_costs
        self.adv_initial_states = adv_initial_states
        self.adversarial_initial_states_cost_coeff = adversarial_initial_states_cost_coeff
        self.adv_initial_states_kwargs = adv_initial_states_kwargs

    def batch_update(self, batch_num, state, total_num_batches):
        state = state[0]
        state = state.to(self.device)
        states, costs = self.controlled_system(state, self.train_time_horizon)

        if isinstance(costs, List):
            costs = torch.concat(costs, dim=1)

        cost = costs.sum(dim=1).mean() if not self.average_costs else costs.mean()

        if self.adv_initial_states is not None:
            self.__optimization_step_with_adv_cost(cost)
        else:
            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()

        return {
            "train cost": cost.item()
        }

    def __optimization_step_with_adv_cost(self, cost):
        kwargs = self.adv_initial_states_kwargs if self.adv_initial_states_kwargs else {}
        _, adv_costs, controls = self.controlled_system(self.adv_initial_states.to(self.device), self.train_time_horizon,
                                                        return_controls=True, **kwargs)

        adv_cost = adv_costs.sum(dim=1).mean() if not self.average_costs else adv_costs.mean()
        self.optimizer.zero_grad()
        (cost + self.adversarial_initial_states_cost_coeff * adv_cost).backward()
        self.optimizer.step()
