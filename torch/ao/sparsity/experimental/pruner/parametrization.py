import torch
from torch import nn


class PruningParametrization(nn.Module):
    def __init__(self, original_outputs):
        super().__init__()
        self.original_outputs = set(range(original_outputs.item()))
        self.pruned_outputs = set()  # Will contain indicies of outputs to prune

    def forward(self, x):
        valid_outputs = self.original_outputs - self.pruned_outputs
        return x[list(valid_outputs)]


class ActivationReconstruction:
    def __init__(self, parametrization):
        self.param = parametrization

    def __call__(self, module, input, output):
        max_outputs = self.param.original_outputs
        pruned_outputs = self.param.pruned_outputs
        reconstructed_tensor = torch.zeros((output.shape[0], len(max_outputs)))
        valid_columns = list(max_outputs - pruned_outputs)
        reconstructed_tensor[:, valid_columns] = output
        return reconstructed_tensor


class BiasHook:
    def __init__(self, has_bias):
        self.has_bias = has_bias

    def __call__(self, module, input, output):
        if self.has_bias:
            return output + module._bias
        return output
