from torch import nn


class PruningParametrization(nn.Module):
    def __init__(self, original_rows):
        super().__init__()
        self.original_rows = set(range(original_rows))
        self.pruned_rows = set()  # Will contain indicies of rows to prune

    def forward(self, x):
        valid_rows = self.original_rows - self.pruned_rows
        return x[list(valid_rows)]
