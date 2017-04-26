import torch
from .module import Module
from .. import functional as F


class PairwiseDistance(Module):
    r"""
    Computes the batchwise pairwise distance between vectors v1,v2:

        .. math ::
            \Vert x \Vert _p := \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}

        Args:
            x (Tensor): input tensor containing the two input batches
            p (real): the norm degree. Default: 2

        Shape:
            - Input: :math:`(N, D)` where `D = vector dimension`
            - Output: :math:`(N, 1)`

        >>> pdist = nn.PairwiseDistance(2)
        >>> input1 = autograd.Variable(torch.randn(100, 128))
        >>> input2 = autograd.Variable(torch.randn(100, 128))
        >>> output = pdist(input1, input2)
    """
    def __init__(self, p=2, eps=1e-6):
        super(PairwiseDistance, self).__init__()
        self.norm = p
        self.eps = eps

    def forward(self, x1, x2):
        return F.pairwise_distance(x1, x2, self.norm, self.eps)

# TODO: Cosine
# TODO: CosineDistance - make sure lua's CosineDistance isn't actually cosine similarity
# TODO: Euclidean
# TODO: WeightedEuclidean
