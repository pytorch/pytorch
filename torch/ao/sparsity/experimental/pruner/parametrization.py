import torch
from torch import nn
from typing import Any, List


class PruningParametrization(nn.Module):
    def __init__(self, original_outputs, axis=0):
        super().__init__()
        self.original_outputs = set(range(original_outputs.item()))
        self.pruned_outputs = set()  # Will contain indicies of outputs to prune
        self.axis = axis

    def forward(self, x):
        valid_outputs = list(self.original_outputs - self.pruned_outputs)
        if self.axis == 0:
            return x[valid_outputs]
        elif self.axis == 1:
            return x[:, valid_outputs]
        else:
            # TODO: Need to figure out how to get over specific axis
            raise NotImplementedError("Only 0 or 1 is supported as axis")


class ZeroesParametrization(nn.Module):
    r"""Zero out pruned channels instead of removing.
    E.g. used for Batch Norm pruning, which should match previous Conv2d layer."""
    def __init__(self, original_outputs):
        super().__init__()
        self.original_outputs = set(range(original_outputs.item()))
        self.pruned_outputs = set()  # Will contain indicies of outputs to prune

    def forward(self, x):
        """ TODO: Potential bug: if we are assigning to the .data(), this might
        break to the parametrizations.
        """
        x.data[list(self.pruned_outputs)] = 0
        return x


class ActivationReconstruction:
    def __init__(self, parametrization):
        self.param = parametrization

    def __call__(self, module, input, output):
        max_outputs = self.param.original_outputs
        pruned_outputs = self.param.pruned_outputs
        valid_columns = list(max_outputs - pruned_outputs)

        # get size of reconstructed output
        sizes = list(output.shape)
        sizes[1] = len(max_outputs)

        # get valid indices of reconstructed output
        indices: List[Any] = []
        for size in output.shape:
            indices.append(slice(0, size, 1))
        indices[1] = valid_columns

        reconstructed_tensor = torch.zeros(sizes,
                                           dtype=output.dtype,
                                           device=output.device,
                                           layout=output.layout)
        reconstructed_tensor[indices] = output
        return reconstructed_tensor


class BiasHook:
    def __init__(self, parametrization, prune_bias):
        self.param = parametrization
        self.prune_bias = prune_bias

    def __call__(self, module, input, output):
        pruned_outputs = self.param.pruned_outputs

        if getattr(module, '_bias', None) is not None:
            bias = module._bias.data
            if self.prune_bias:
                bias[list(pruned_outputs)] = 0

            # reshape bias to broadcast over output dimensions
            idx = [1] * len(output.shape)
            idx[1] = -1
            bias = bias.reshape(idx)

            output += bias
        return output
