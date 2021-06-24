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
        original_out = output.shape[1] + len(pruned_outputs)
        zeros = torch.zeros((output.shape[0], 1))
        cols = []
        jdx = 0
        for idx in range(original_out):
            if idx in pruned_outputs:
                cols.append(zeros)
            else:
                cols.append(output[:, jdx].reshape((-1, 1)))
                jdx += 1
        return torch.hstack(cols)
