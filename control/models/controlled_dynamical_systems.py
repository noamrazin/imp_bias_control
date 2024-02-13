from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class DynamicalSystem(nn.Module, ABC):

    @abstractmethod
    def get_state_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_control_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def compute_time_step_cost(self, state: torch.Tensor, control: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute the cost for a single time step.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, state: torch.Tensor, control: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the dynamical system.
        :param state: state of the system.
        :param control: control input.
        :return: next state, cost for the transition.
        """
        raise NotImplementedError


class ControlledDynamicalSystem(nn.Module, ABC):

    @abstractmethod
    def forward(self, state: torch.Tensor, steps: int, return_controls: bool = False, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the controlled dynamical system.
        @param state: initial state of the system.
        @param steps: number of steps to simulate.
        @param return_controls: whether to return the controls.
        @return: states, costs, controls (if return_controls is True).
        """
        raise NotImplementedError


class ControlledDynamicalSystemImpl(ControlledDynamicalSystem):

    def __init__(self, system: DynamicalSystem, controller: nn.Module):
        super(ControlledDynamicalSystemImpl, self).__init__()
        self.system = system
        self.controller = controller

    def forward(self, state: torch.Tensor, steps: int, return_controls: bool = False, custom_controller=None, **kwargs):
        states = []
        costs = []
        controls = []

        controller = self.controller if custom_controller is None else custom_controller

        curr_state = state
        for i in range(steps):
            control = controller(curr_state)
            new_state, cost = self.system(curr_state, control, **kwargs)

            states.append(curr_state)
            costs.append(cost)
            curr_state = new_state

            if return_controls:
                controls.append(control)

        states.append(curr_state)
        last_control = controller(curr_state)
        last_cost = self.system.compute_time_step_cost(curr_state, last_control, **kwargs)
        costs.append(last_cost)
        controls.append(last_control)

        states = torch.stack(states).permute(1, 0, 2)
        costs = torch.concat(costs, dim=1)

        if return_controls:
            controls = torch.stack(controls).permute(1, 0, 2)
            return states, costs, controls

        return states, costs
