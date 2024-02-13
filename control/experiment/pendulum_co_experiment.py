import logging
import math
from collections import OrderedDict

import torch.utils.data

import common.train.callbacks as callbacks
from common.data.modules import DataModule
from common.evaluation.evaluators import Evaluator, TrainEvaluator, TrainBatchOutputEvaluator
from common.experiment import ExperimentResult
from common.experiment import FitExperimentBase
from common.experiment.fit_experiment_base import ScoreInfo
from common.train.callbacks import Callback
from common.train.fit_output import FitOutput
from common.train.trainer import Trainer
from control.datasets.initial_states_datamodule import InitialStatesDataModule
from control.evaluation.controller_validation_evaluator import ControllerValidationEvaluator
from control.models.controlled_dynamical_systems import ControlledDynamicalSystemImpl
from control.models.controllers import *
from control.models.pendulum import Pendulum
from control.train.controller_trainer import ControllerTrainer
from control.train.pendulum_animation_plotter import PendulumAnimationPlotter


def create_custom_initial_states(angles):
    return torch.tensor([[angle, 0.0] for angle in angles])


class PendulumControllerOptimizationExperiment(FitExperimentBase):

    @staticmethod
    def add_experiment_specific_args(p):
        FitExperimentBase.add_experiment_base_specific_args(p)

        p.add_argument("--train_time_horizon", type=int, default=50, help="Time horizon used to train the controller")
        p.add_argument("--test_time_horizon", type=int, default=50, help="Time horizon used to evaluate the controller")

        p.add_argument("--controller", type=str, default="mlp", help="Which controller to use. Supports: 'linear_nn' and 'mlp'")
        p.add_argument("--controller_hidden_dim", type=int, default=10, help="Hidden dimension of the controller when using linear NN or MLP")
        p.add_argument("--controller_depth", type=int, default=2, help="Depth of the controller when using linear NN or MLP")
        p.add_argument("--controller_init_std", type=float, default=0.01, help="Standard deviation of the linear controller initialization")
        p.add_argument("--controller_zero_init", action="store_true", help="If True will initialize the linear controller to zero")
        p.add_argument("--controller_identity_init", action="store_true", help="If True will initialize the linear controller to identity")

        p.add_argument("--initial_states_rnd_seed", type=int, default=-1, help="Random seed for generating the initial_states")
        p.add_argument("--train_initial_states_type", type=str, default="rnd",
                       help="Which initial states to train on. Use 'rnd' for random uniform angles, "
                            "'custom' for custom initial states with angles specified in 'custom_initial_states_angles'")
        p.add_argument("--custom_initial_states_angles", type=float, nargs="+", default=[math.pi - 0.5, math.pi + 0.5],
                       help="Angles of the custom initial train states to use.")
        p.add_argument("--adversarial_initial_states_angles", type=float, nargs="+", default=[],
                       help='Angles of "adversarial" initial train states whose target state is [0, 0] as opposed to the real [pi, 0] goal.')
        p.add_argument("--adversarial_initial_states_cost_coeff", type=float, default=0.1, help="Coefficient of adversarial initial states cost.")
        p.add_argument("--num_train_initial_states", type=int, default=1, help="Number of train initial states to use.")
        p.add_argument("--test_initial_angles", type=float, nargs="+", default=[0,
                                                                                0.7853981852531433,
                                                                                1.5707963705062866,
                                                                                2.356194496154785,
                                                                                3.1415927410125732,
                                                                                3.9269909858703613,
                                                                                4.71238899230957,
                                                                                5.4977874755859375], help="Initial angles to use for test states.")

        p.add_argument("--optimizer", type=str, default="sgd", help="optimizer to use. Supports: 'sgd' and 'adam'.")
        p.add_argument("--lr", type=float, default=0.01, help="Gradient descent learning rate")

    def __create_train_initial_states(self, config: dict):
        if config["train_initial_states_type"] == "rnd":
            if config["initial_states_rnd_seed"] >= 0:
                curr_rng_state = torch.random.get_rng_state()
                torch.random.manual_seed(config["initial_states_rnd_seed"])

            initial_angles = torch.distributions.uniform.Uniform(0, 2 * torch.pi).sample([config["num_train_initial_states"]])
            initial_states = torch.stack([initial_angles, torch.zeros_like(initial_angles)], dim=1)

            if config["initial_states_rnd_seed"] >= 0:
                torch.random.set_rng_state(curr_rng_state)

            return initial_states
        elif config["train_initial_states_type"] == "custom":
            return create_custom_initial_states(config["custom_initial_states_angles"])
        else:
            raise ValueError(f"Unknown train initial states type: {config['train_initial_states_type']}")

    def __create_test_initial_states(self, config: dict):
        angles = torch.tensor(config["test_initial_angles"])
        return torch.stack([angles, torch.zeros_like(angles)], dim=1)

    def create_datamodule(self, config: dict, state: dict, logger: logging.Logger) -> DataModule:
        load_dataset_to_device = state["device"] if config["load_dataset_to_gpu"] and len(state["gpu_ids"]) > 0 else None
        train_initial_states = self.__create_train_initial_states(config)
        test_initial_states = self.__create_test_initial_states(config)

        datamodule = InitialStatesDataModule(train_initial_states, test_initial_states, load_dataset_to_device=load_dataset_to_device)
        datamodule.setup()
        return datamodule

    def create_model(self, datamodule: InitialStatesDataModule, config: dict, state: dict, logger: logging.Logger) -> nn.Module:
        system = Pendulum(control_norm_cost_coeff=config["control_norm_cost_coeff"])

        if config["controller"] == "linear_nn":
            controller = LinearNNStateController(state_dim=system.get_state_dim(), control_dim=system.get_control_dim(),
                                                 hidden_dim=config["controller_hidden_dim"], depth=config["controller_depth"],
                                                 init_std=config["controller_init_std"], zero_init=config["controller_zero_init"],
                                                 identity_init=config["controller_identity_init"], use_bias=True)
        elif config["controller"] == "mlp":
            hidden_layer_sizes = [config["controller_hidden_dim"]] * (config["controller_depth"] - 1)
            controller = MultiLayerPerceptronStateController(input_size=system.get_state_dim(), output_size=system.get_control_dim(),
                                                             hidden_layer_sizes=hidden_layer_sizes)
        else:
            raise ValueError(f"Unsupported controller: {config['controller']}")

        return ControlledDynamicalSystemImpl(system, controller)

    def create_train_and_validation_evaluators(self, model: ControlledDynamicalSystemImpl, datamodule: InitialStatesDataModule, device,
                                               config: dict, state: dict,
                                               logger: logging.Logger) -> Tuple[TrainEvaluator, Evaluator]:
        train_eval = TrainBatchOutputEvaluator(metric_names=["train cost"])

        state["initial train cost"] = self.__compute_cost(model, datamodule.train_initial_states, config["train_time_horizon"], state["device"])
        state["initial test cost"] = self.__compute_cost(model, datamodule.test_initial_states, config["test_time_horizon"], state["device"])

        val_evaluator = ControllerValidationEvaluator(model, config["test_time_horizon"], datamodule.test_dataloader(),
                                                      initial_test_cost=state["initial test cost"], device=device, average_costs=True)

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
        return additional_metadata

    def __compute_cost(self, model: ControlledDynamicalSystemImpl, initial_states: torch.Tensor, time_horizon: int, device,
                       average_costs: bool = True, no_agg: bool = False):
        with torch.no_grad():
            model.to(device)
            initial_states = initial_states.to(device)
            states, costs = model(initial_states, time_horizon)
            if no_agg:
                return costs.mean(dim=1).detach() if average_costs else costs.sum(dim=1).detach()

            return costs.mean().item() if average_costs else costs.sum(dim=1).mean().item()

    def customize_callbacks(self, callbacks_dict: OrderedDict, model: ControlledDynamicalSystemImpl, datamodule: InitialStatesDataModule,
                            config: dict, state: dict, logger: logging.Logger):
        callbacks_dict["stop_on_nan"] = callbacks.TerminateOnNaN(verify_batches=False)
        callbacks_dict["pendulum_animation_logger"] = PendulumAnimationPlotter(model, train_initial_states=datamodule.train_initial_states,
                                                                               train_time_horizon=config["train_time_horizon"],
                                                                               test_initial_states=datamodule.test_initial_states,
                                                                               test_time_horizon=config["test_time_horizon"],
                                                                               save_dir=state["experiment_dir"],
                                                                               epoch_save_interval=-1, device=state["device"], logger=logger)

    def create_trainer(self, model: ControlledDynamicalSystemImpl, datamodule: InitialStatesDataModule, train_evaluator: TrainEvaluator,
                       val_evaluator: Evaluator, callback: Callback, device, config: dict, state: dict, logger: logging.Logger) -> Trainer:
        if config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        elif config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
        else:
            raise ValueError(f"Unsupported optimizer type: '{config['optimizer']}'")

        if config["adversarial_initial_states_angles"]:
            adv_train_initial_states = create_custom_initial_states(config["adversarial_initial_states_angles"]).to(device)
            adv_train_target_states = torch.zeros_like(adv_train_initial_states).to(device)
            adv_train_target_states[adv_train_initial_states[:, 0] > torch.pi, 0] = 2 * torch.pi
            adv_train_initial_states_args = {"custom_target_state": adv_train_target_states}
        else:
            adv_train_initial_states = None
            adv_train_initial_states_args = None

        return ControllerTrainer(model, optimizer, train_time_horizon=config["train_time_horizon"],
                                 train_evaluator=train_evaluator, val_evaluator=val_evaluator, callback=callback, device=device,
                                 average_costs=True, adv_initial_states=adv_train_initial_states,
                                 adversarial_initial_states_cost_coeff=config["adversarial_initial_states_cost_coeff"],
                                 adv_initial_states_kwargs=adv_train_initial_states_args)

    def on_experiment_end(self, model: ControlledDynamicalSystemImpl, datamodule: InitialStatesDataModule, trainer: Trainer, fit_output: FitOutput,
                          experiment_result: ExperimentResult, config: dict, state: dict, logger: logging.Logger):
        experiment_result.summary["initial train cost"] = state["initial train cost"]
        experiment_result.summary["initial test cost"] = state["initial test cost"]

        # Add train and test initial and final states info to summary
        experiment_result.summary["train initial states"] = datamodule.train_initial_states.tolist()
        experiment_result.summary["test initial states"] = datamodule.test_initial_states.tolist()

        with torch.no_grad():
            train_states = model(datamodule.train_initial_states.to(state["device"]), config["train_time_horizon"])[0]
            experiment_result.summary["train final states"] = train_states[:, -1, :].tolist()

            test_states = model(datamodule.test_initial_states.to(state["device"]), config["test_time_horizon"])[0]
            experiment_result.summary["test final states"] = test_states[:, -1, :].tolist()

            for i in range(1, 10):
                experiment_result.summary[f"train {10 * i}% steps states"] = train_states[:, int((i / 10) * config["train_time_horizon"]), :].tolist()
                experiment_result.summary[f"test {10 * i}% steps states"] = test_states[:, int((i / 10) * config["test_time_horizon"]), :].tolist()
