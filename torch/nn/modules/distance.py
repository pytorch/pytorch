import torch
from .module import Module


class PairwiseDistance(Module):
    r"""
    Computes the batchwise pairwise distance between vectors v1,v2:

        .. math ::
            \Vert x \Vert _p := \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}

        Args:
            x (Tensor): input tensor containing the two input batches

        Attributes:
            p (real): the norm degree. Default: 2

        Shape:
            - Input: :math:`(2, N, D)` where `D = vector dimension`
            - Output: :math:`(N, 1)

        >>> pdist = nn.PairwiseDistance(2)
        >>> input = autograd.Variable(torch.randn(2, 100, 128))
        >>> output = pdist(input)
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
