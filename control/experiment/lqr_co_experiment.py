import logging
import re

import torch.utils.data

from common.data.modules import DataModule
from common.evaluation.evaluators import Evaluator, TrainEvaluator, TrainBatchOutputEvaluator
from common.experiment import ExperimentResult
from common.experiment import FitExperimentBase
from common.experiment.fit_experiment_base import ScoreInfo
from common.train.callbacks import Callback
from common.train.fit_output import FitOutput
from common.train.trainer import Trainer
from control.datasets.initial_states_datamodule import InitialStatesDataModule
from control.evaluation.lqr_validation_evaluator import LQRValidationEvaluator
from control.models.controlled_dynamical_systems import ControlledDynamicalSystemImpl
from control.models.controllers import *
from control.models.linear_dynamical_systems import *
from control.train.lqr_trainer import LQRTrainer

STANDARD_BASIS_VECTOR_INIT_STATE_REGEX = re.compile("e([0-9]+)")


def create_custom_random_linear_system(state_dim: int, control_dim: int, config):
    if config["system_rnd_seed"] >= 0:
        curr_rng_state = torch.random.get_rng_state()
        torch.random.manual_seed(config["system_rnd_seed"])

    linear_system = LinearTimeInvariantSystem(state_dim, control_dim)
    mats_init_std = config["mats_init_std"] if config["mats_init_std"] > 0 else 1 / math.sqrt(state_dim)
    A = torch.randn(state_dim, state_dim) * mats_init_std

    Q = torch.eye(state_dim)
    B = torch.eye(control_dim, state_dim)
    R = torch.zeros(control_dim, control_dim)

    linear_system.set_custom_system(A, B, Q, R)

    if config["system_rnd_seed"] >= 0:
        torch.random.set_rng_state(curr_rng_state)

    return linear_system


def create_shift_linear_system(state_dim: int, control_dim: int, config):
    linear_system = LinearTimeInvariantSystem(state_dim, control_dim)
    A = torch.eye(state_dim)
    A = torch.roll(A, 1, dims=1)
    B = torch.eye(state_dim, control_dim)
    Q = torch.eye(state_dim)
    R = torch.zeros(control_dim, control_dim)

    linear_system.set_custom_system(A, B, Q, R)
    return linear_system


def create_identity_linear_system(state_dim: int, control_dim: int, config):
    linear_system = LinearTimeInvariantSystem(state_dim, control_dim)
    A = torch.eye(state_dim)
    B = torch.eye(state_dim, control_dim)
    Q = torch.eye(state_dim)
    R = torch.zeros(control_dim, control_dim)

    linear_system.set_custom_system(A, B, Q, R)
    return linear_system


class LQRControllerOptimizationExperiment(FitExperimentBase):

    @staticmethod
    def add_experiment_specific_args(p):
        FitExperimentBase.add_experiment_base_specific_args(p)

        p.add_argument("--train_time_horizon", type=int, default=5, help="Time horizon of the LQR problem used to train the controller")
        p.add_argument("--test_time_horizon", type=int, default=-1, help="Time horizon of the LQR problem used to evaluate the controller. If < 0, "
                                                                         "will use the train time horizon")
        p.add_argument("--state_dim", type=int, default=5, help="State dimension")
        p.add_argument("--control_dim", type=int, default=5, help="Control dimension")

        p.add_argument("--controller_init_std", type=float, default=0.1, help="Standard deviation of the linear controller initialization (only "
                                                                              "used if 'controller_zero_init' is False)")
        p.add_argument("--controller_zero_init", action="store_true", help="If True will initialize the linear controller to zero")

        p.add_argument("--mats_init_std", type=float, default=-1, help="Standard deviation of initialization for system matrices. If <= 0, will use"
                                                                       "1 / sqrt(state_dim) as the standard deviation")

        p.add_argument("--linear_system", type=str, default="rnd",
                       help="Which linear dynamical system to use. Supports 'custom_rnd', 'rnd', 'shift', and 'identity'")
        p.add_argument("--system_rnd_seed", type=int, default=-1, help="Random seed for generating the linear dynamical system")
        p.add_argument("--train_initial_states_type", type=str, default="rnd",
                       help="Which initial states to train on. Use 'rnd' for Gaussian, 'rnd_orth' for random orthonormal, "
                            "'e' for standard basis vectors, and 'e<i>' for a single standard basis vector, replacing '<i>' with the non-zero coordinate index")
        p.add_argument("--num_train_initial_states", type=int, default=1, help="Number of train initial states to use.")

        p.add_argument("--lr", type=float, default=0.001, help="Gradient descent learning rate")

    def __create_train_initial_states(self, config: dict):
        if config["train_initial_states_type"] == "rnd":
            if config["system_rnd_seed"] >= 0:
                curr_rng_state = torch.random.get_rng_state()
                torch.random.manual_seed(config["system_rnd_seed"] + 1)

            initial_state = torch.randn(config["num_train_initial_states"], config["state_dim"])

            if config["system_rnd_seed"] >= 0:
                torch.random.set_rng_state(curr_rng_state)

            return initial_state / torch.norm(initial_state, dim=1, keepdim=True)
        elif config["train_initial_states_type"] == "e":
            return torch.eye(config["num_train_initial_states"], config["state_dim"])

        standard_basis_vector_match = STANDARD_BASIS_VECTOR_INIT_STATE_REGEX.match(config["train_initial_states_type"])
        if standard_basis_vector_match:
            initial_state = torch.zeros(1, config["state_dim"])
            initial_state[0, int(standard_basis_vector_match.group(1)) - 1] = 1.0
            return initial_state
        else:
            raise ValueError(f"Unknown train initial states type: {config['train_initial_states_type']}")

    def __create_test_initial_states(self, initial_states, config: dict):
        # If number of training initial states is greater than the state dimension, than use train initial states as test initial states as well
        if config["state_dim"] <= config["num_train_initial_states"]:
            return initial_states

        if config["train_initial_states_type"] == "rnd":
            if config["system_rnd_seed"] >= 0:
                curr_rng_state = torch.random.get_rng_state()
                torch.random.manual_seed(config["system_rnd_seed"] + 2)

            initial_states = torch.randn(config["state_dim"] - config["num_train_initial_states"], config["state_dim"])

            if config["system_rnd_seed"] >= 0:
                torch.random.set_rng_state(curr_rng_state)

            return initial_states / torch.norm(initial_states, dim=1, keepdim=True)
        elif config["train_initial_states_type"] == "e":
            return torch.eye(config["state_dim"], config["state_dim"])[config["num_train_initial_states"]:, :]

        standard_basis_vector_match = STANDARD_BASIS_VECTOR_INIT_STATE_REGEX.match(config["train_initial_states_type"])
        if standard_basis_vector_match:
            train_initial_state_index = int(standard_basis_vector_match.group(1)) - 1
            initial_states = torch.eye(config["state_dim"], config["state_dim"])
            initial_states = torch.cat([initial_states[:train_initial_state_index, :], initial_states[train_initial_state_index + 1:, :]], dim=0)
            return initial_states
        else:
            raise ValueError(f"Unknown train initial states type: {config['train_initial_states_type']}")

    def __create_rnd_orth_initial_states(self, config: dict):
        if config["system_rnd_seed"] >= 0:
            curr_rng_state = torch.random.get_rng_state()
            torch.random.manual_seed(config["system_rnd_seed"] + 1)

        mat = torch.randn(1, config["state_dim"], config["state_dim"])
        initial_states = torch.linalg.qr(mat)[0][0]

        if config["system_rnd_seed"] >= 0:
            torch.random.set_rng_state(curr_rng_state)

        # If number of training initial states is greater than the state dimension, than use train initial states as test initial states as well
        if config["state_dim"] <= config["num_train_initial_states"]:
            return initial_states, initial_states

        return initial_states[:config["num_train_initial_states"]], initial_states[config["num_train_initial_states"]:]

    def create_datamodule(self, config: dict, state: dict, logger: logging.Logger) -> DataModule:
        load_dataset_to_device = state["device"] if config["load_dataset_to_gpu"] and len(state["gpu_ids"]) > 0 else None

        if config["train_initial_states_type"] != "rnd_orth":
            train_initial_states = self.__create_train_initial_states(config)
            test_initial_states = self.__create_test_initial_states(train_initial_states, config)
        else:
            train_initial_states, test_initial_states = self.__create_rnd_orth_initial_states(config)

        datamodule = InitialStatesDataModule(train_initial_states, test_initial_states, load_dataset_to_device=load_dataset_to_device)
        datamodule.setup()
        return datamodule

    def create_model(self, datamodule: InitialStatesDataModule, config: dict, state: dict, logger: logging.Logger) -> nn.Module:
        if config["linear_system"] == "custom_rnd":
            linear_system = create_custom_random_linear_system(config["state_dim"], config["control_dim"], config)
        elif config["linear_system"] == "shift":
            linear_system = create_shift_linear_system(config["state_dim"], config["control_dim"], config)
        elif config["linear_system"] == "identity":
            linear_system = create_identity_linear_system(config["state_dim"], config["control_dim"], config)
        elif config["linear_system"] == "rnd":
            linear_system = LinearTimeInvariantSystem(config["state_dim"], config["control_dim"],
                                                      mats_init_std=config["mats_init_std"],
                                                      random_seed=config["system_rnd_seed"])
            linear_system.R = torch.zeros_like(linear_system.R)
        else:
            raise ValueError(f"Unsupported linear system type: {config['system']}")

        controller = LinearNNStateController(state_dim=config["state_dim"], control_dim=config["control_dim"],
                                             depth=1, init_std=config["controller_init_std"], zero_init=config["controller_zero_init"])

        return ControlledDynamicalSystemImpl(linear_system, controller)

    def create_train_and_validation_evaluators(self, model: ControlledDynamicalSystemImpl, datamodule: InitialStatesDataModule, device,
                                               config: dict, state: dict, logger: logging.Logger) -> Tuple[TrainEvaluator, Evaluator]:
        train_eval = TrainBatchOutputEvaluator(metric_names=["train cost", "excess train cost"])

        test_time_horizon = config["test_time_horizon"] if config["test_time_horizon"] >= 0 else config["train_time_horizon"]
        state["initial train cost"] = self.__compute_cost(model, datamodule.train_initial_states, config["train_time_horizon"])
        state["initial test cost"] = self.__compute_cost(model, datamodule.test_initial_states, test_time_horizon)

        if config["state_dim"] <= config["control_dim"]:
            state["min L2 controller test cost"] = self.__compute_test_cost_for_min_L2_norm_train_controller(model,
                                                                                                             datamodule.train_initial_states,
                                                                                                             datamodule.test_initial_states,
                                                                                                             test_time_horizon)

        if torch.allclose(model.system.R, torch.zeros_like(model.system.R)):
            state["minimal train cost"] = self.__compute_first_state_cost(model, datamodule.train_initial_states)
            state["minimal test cost"] = self.__compute_first_state_cost(model, datamodule.test_initial_states)
            state["initial normalized optimality extrapolation measure"] = LQRValidationEvaluator.compute_normalized_optimality_measure(model,
                                                                                                                                        datamodule.test_initial_states)

        val_evaluator = LQRValidationEvaluator(model, test_time_horizon, datamodule.test_dataloader(),
                                               initial_test_cost=state["initial test cost"],
                                               min_L2_controller_test_cost=state.get("min L2 controller test cost"),
                                               device=device)

        return train_eval, val_evaluator

    def get_default_score_info(self, config: dict, state: dict) -> ScoreInfo:
        return ScoreInfo(metric_name="train cost", is_train_metric=False, largest=False, return_best_score=False)

    def create_additional_metadata_to_log(self, model: ControlledDynamicalSystemImpl, datamodule: InitialStatesDataModule,
                                          config: dict, state: dict,
                                          logger: logging.Logger) -> dict:
        additional_metadata = super().create_additional_metadata_to_log(model, datamodule, config, state, logger)
        additional_metadata["num_train_initial_states"] = datamodule.train_initial_states.shape[0]
        additional_metadata["train_initial_states"] = datamodule.train_initial_states.tolist()
        additional_metadata["num_test_initial_states"] = datamodule.test_initial_states.shape[0]
        additional_metadata["test_initial_states"] = datamodule.test_initial_states.tolist()
        additional_metadata["initial train cost"] = state["initial train cost"]
        additional_metadata["initial test cost"] = state["initial test cost"]

        if torch.allclose(model.system.R, torch.zeros_like(model.system.R)):
            additional_metadata["minimal train cost"] = state["minimal train cost"]
            additional_metadata["minimal test cost"] = state["minimal test cost"]
            additional_metadata["initial normalized optimality extrapolation measure"] = state["initial normalized optimality extrapolation measure"]

        return additional_metadata

    def __compute_cost(self, model: ControlledDynamicalSystemImpl, initial_states: torch.Tensor, time_horizon: int,
                       custom_controller=None):
        with torch.no_grad():
            states, costs = model(initial_states, time_horizon, custom_controller=custom_controller)
            return costs.sum(dim=1).mean().item()

    def __compute_first_state_cost(self, model: ControlledDynamicalSystemImpl, initial_states: torch.Tensor):
        with torch.no_grad():
            cost = F.bilinear(initial_states, initial_states, model.system.Q.unsqueeze(dim=0))
            return cost.mean().item()

    def __compute_test_cost_for_min_L2_norm_train_controller(self, model: ControlledDynamicalSystemImpl, train_initial_states: torch.Tensor,
                                                             test_initial_states: torch.Tensor, time_horizon: int):
        with torch.no_grad():
            Q, R = torch.linalg.qr(train_initial_states.t().unsqueeze(dim=0))
            proj_to_train_initial_states_mat = torch.matmul(Q[0], Q[0].t())

            min_L2_norm_train_controller_mat = - torch.matmul(proj_to_train_initial_states_mat,
                                                              torch.matmul(model.system.A, torch.linalg.pinv(model.system.B)))
            controller = lambda x: torch.matmul(x, min_L2_norm_train_controller_mat)
            return self.__compute_cost(model, test_initial_states, time_horizon, custom_controller=controller)

    def create_trainer(self, model: ControlledDynamicalSystemImpl, datamodule: InitialStatesDataModule, train_evaluator: TrainEvaluator,
                       val_evaluator: Evaluator, callback: Callback, device, config: dict, state: dict, logger: logging.Logger) -> Trainer:
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
        return LQRTrainer(model, optimizer, train_time_horizon=config["train_time_horizon"],
                          train_evaluator=train_evaluator, val_evaluator=val_evaluator, callback=callback, device=device)

    def on_experiment_end(self, model: ControlledDynamicalSystemImpl, datamodule: InitialStatesDataModule, trainer: Trainer, fit_output: FitOutput,
                          experiment_result: ExperimentResult, config: dict, state: dict, logger: logging.Logger):
        experiment_result.summary["initial train cost"] = state["initial train cost"]
        experiment_result.summary["initial test cost"] = state["initial test cost"]

        if torch.allclose(model.system.R, torch.zeros_like(model.system.R)):
            experiment_result.summary["minimal train cost"] = state["minimal train cost"]
            experiment_result.summary["minimal test cost"] = state["minimal test cost"]
            experiment_result.summary["initial excess train cost"] = state["initial train cost"] - state["minimal train cost"]
            experiment_result.summary["initial excess test cost"] = state["initial test cost"] - state["minimal test cost"]
            experiment_result.summary["initial normalized optimality extrapolation measure"] = state[
                "initial normalized optimality extrapolation measure"]

        if state.get("min L2 controller test cost") is not None:
            experiment_result.summary["min L2 controller test cost"] = state["min L2 controller test cost"]
            if torch.allclose(model.system.R, torch.zeros_like(model.system.R)):
                experiment_result.summary["excess min L2 controller test cost"] = state["min L2 controller test cost"] - state["minimal test cost"]
