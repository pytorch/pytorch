import torch
from .module import Module


class PairwiseDistance(Module):
    r"""Computes the distance between two vectors using the p-norm.
    """

    def __init__(self, p=2):
        super(PairwiseDistance, self).__init__()
        assert p % 1 == 0
        self.norm = p

    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)


# TODO: Cosine
# TODO: CosineDistance - make sure lua's CosineDistance isn't actually cosine similarity
# TODO: Euclidean
# TODO: WeightedEuclidean
