import torch

from common.evaluation.evaluators.evaluator import VoidEvaluator
from common.train.trainer import Trainer
from control.models.controlled_dynamical_systems import ControlledDynamicalSystemImpl


class LQRTrainer(Trainer):

    def __init__(self, controlled_system: ControlledDynamicalSystemImpl, optimizer, train_time_horizon: int,
                 train_evaluator=VoidEvaluator(), val_evaluator=VoidEvaluator(), callback=None,
                 device=torch.device("cpu"), average_costs: bool = False):
        super().__init__(controlled_system, optimizer, train_evaluator, val_evaluator, callback, device)
        self.controlled_system = controlled_system
        self.train_time_horizon = train_time_horizon
        self.average_costs = average_costs

    def batch_update(self, batch_num, state, total_num_batches):
        state = state[0]
        state = state.to(self.device)
        states, costs = self.controlled_system(state, self.train_time_horizon)

        cost = costs.sum(dim=1).mean() if not self.average_costs else costs.mean()

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        excess_cost = cost - costs[:, 0].mean()

        return {
            "train cost": cost.item(),
            "excess train cost": excess_cost.item()
        }
