from typing import Sequence, Callable, Tuple

import torch
import torch.nn as nn


class LinearStateController(nn.Module):

    def __init__(self, state_dim: int, control_dim: int, init_std: float = 0.01, zero_init: bool = False, identity_init: bool = False,
                 use_bias: bool = False):
        super(LinearStateController, self).__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.init_std = init_std
        self.zero_init = zero_init
        self.identity_init = identity_init
        self.use_bias = use_bias

        if zero_init:
            K = torch.zeros(state_dim, control_dim)
        elif identity_init:
            K = torch.eye(state_dim, control_dim)
        else:
            K = torch.randn(state_dim, control_dim) * init_std

        self.K = nn.Parameter(K)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1, control_dim))

    def compute_control_matrix(self):
        return self.K

    def forward(self, state: torch.Tensor):
        lin_out = torch.matmul(state, self.K)
        if self.use_bias:
            lin_out += self.bias

        return lin_out


class LinearNNStateController(nn.Module):

    def __init__(self, state_dim: int, control_dim: int, depth: int, hidden_dim: int = 10, init_std: float = 0.01, zero_init: bool = False,
                 identity_init: bool = False, use_bias: bool = False):
        super(LinearNNStateController, self).__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.init_std = init_std
        self.zero_init = zero_init
        self.identity_init = identity_init
        self.use_bias = use_bias

        self.layers = nn.ParameterList(self.__create_layers())
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1, control_dim))

    def __create_layers(self):
        layers = []
        in_dims = [self.state_dim] + [self.hidden_dim] * (self.depth - 1)
        out_dims = [self.hidden_dim] * (self.depth - 1) + [self.control_dim]

        for i in range(self.depth):
            layers.append(self.__initialize_new_layer(in_dims[i], out_dims[i]))

        return layers

    def __initialize_new_layer(self, in_dim: int, out_dim: int):
        if self.zero_init:
            K = torch.zeros(in_dim, out_dim)
        elif self.identity_init:
            K = torch.eye(in_dim, out_dim)
        else:
            K = torch.randn(in_dim, out_dim) * self.init_std

        return nn.Parameter(K)

    def compute_control_matrix(self):
        curr = self.layers[0]
        for i in range(1, len(self.layers)):
            curr = torch.matmul(curr, self.layers[i])

        return curr

    def forward(self, state: torch.Tensor):
        lin_out = torch.matmul(state, self.compute_control_matrix())
        if self.use_bias:
            lin_out += self.bias

        return lin_out


class MultiLayerPerceptronStateController(nn.Module):
    """
    Simple MultiLayer Perceptron state controller.
    """

    def __init__(self, input_size: int, output_size: int, hidden_layer_sizes: Sequence[int] = None, bias: bool = True,
                 hidden_layer_activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(inplace=True),
                 output_scaling: Tuple = None, flatten_inputs: bool = True):
        """
        :param input_size: input size.
        :param output_size: output size.
        :param hidden_layer_sizes: sequence of hidden dimension sizes.
        :param bias: if set to False, the linear layers will not use biases.
        :param hidden_layer_activation: activation for the hidden layers.
        :param output_scaling: A pair of min and max values to scale outputs to. Does so by applying Tanh non-linearity, followed by scaling.
        :param flatten_inputs: if True (default), will flatten all inputs to 2 dimensional tensors (i.e. flattens all non-batch dimensions) in the
        forward function.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.hidden_layer_sizes = hidden_layer_sizes if hidden_layer_sizes is not None else []
        self.depth = len(self.hidden_layer_sizes) + 1
        self.output_scaling = output_scaling
        self.flatten_inputs = flatten_inputs

        self.hidden_layers_sequential = self.__create_hidden_layers_sequential_model(hidden_layer_activation)
        self.output_layer = nn.Linear(self.hidden_layer_sizes[-1] if self.hidden_layer_sizes else input_size, output_size, bias=self.bias)

    def __create_hidden_layers_sequential_model(self, activation):
        layers = []

        prev_size = self.input_size
        for hidden_layer_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_layer_size, bias=self.bias))
            layers.append(activation)
            prev_size = hidden_layer_size

        return nn.Sequential(*layers)

    def get_linear_layers(self) -> Sequence[nn.Linear]:
        linear_layers = []
        for child_module in self.hidden_layers_sequential.children():
            if isinstance(child_module, nn.Linear):
                linear_layers.append(child_module)

        linear_layers.append(self.output_layer)
        return linear_layers

    def get_layer_dims(self, layer_index: int) -> Tuple[int, int]:
        num_rows = self.hidden_layer_sizes[layer_index - 1] if layer_index != 0 else self.input_size
        num_cols = self.hidden_layer_sizes[layer_index] if layer_index != self.depth - 1 else self.output_size
        return num_rows, num_cols

    def __scale_output(self, x):
        min_val = self.output_scaling[0]
        max_val = self.output_scaling[1]

        x = torch.tanh(x)
        return 0.5 * (x + 1) * (max_val - min_val) + min_val

    def forward(self, state: torch.Tensor):
        if self.flatten_inputs:
            state = state.view(state.size(0), -1)

        state = self.hidden_layers_sequential(state)
        output = self.output_layer(state)
        return output if self.output_scaling is None else self.__scale_output(output)
