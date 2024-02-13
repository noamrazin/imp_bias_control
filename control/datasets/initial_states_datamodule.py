import torch

from common.data.loaders.fast_tensor_dataloader import FastTensorDataLoader
from common.data.modules.datamodule import DataModule


class InitialStatesDataModule(DataModule):

    def __init__(self, train_initial_states: torch.Tensor, test_initial_states: torch.Tensor,
                 dataloader_num_workers: int = 0, pin_memory: bool = False, load_dataset_to_device=None):
        """
        @param train_initial_states: tensor of shape (num_train_samples, state_dim) with initial states to train on
        @param test_initial_states: tensor of shape (num_test_samples, state_dim) with initial states to test on
        @param dataloader_num_workers: num workers to use for data loading. When using multiprocessing, must use 0
        @param pin_memory: pin_memory argument of Dataloader
        @param load_dataset_to_device: device to load dataset to (default is CPU)
        """
        self.train_initial_states = train_initial_states
        self.test_initial_states = test_initial_states
        self.dataloader_num_workers = dataloader_num_workers
        self.pin_memory = pin_memory
        self.load_dataset_to_device = load_dataset_to_device

        if self.load_dataset_to_device is not None:
            self.train_initial_states = self.train_initial_states.to(load_dataset_to_device)
            self.test_initial_states = self.test_initial_states.to(load_dataset_to_device)

    def setup(self):
        pass

    def train_dataloader(self):
        return FastTensorDataLoader(self.train_initial_states, shuffle=False)

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        return FastTensorDataLoader(self.test_initial_states, shuffle=False)
