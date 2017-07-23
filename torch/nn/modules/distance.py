import torch
from .module import Module
from .. import functional as F


class PairwiseDistance(Module):
    r"""
    Computes the batchwise pairwise distance between vectors v1,v2:

    .. math ::
        \Vert x \Vert _p := \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}

    Args:
        p (real): the norm degree. Default: 2
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-6

    Shape:
        - Input1: :math:`(N, D)` where `D = vector dimension`
        - Input2: :math:`(N, D), same shape as the Input1
        - Output: :math:`(N, 1)`

    Examples::

    >>> pdist = nn.PairwiseDistance(p=2)
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


class CosineSimilarity(Module):
    r"""Returns cosine similarity between x1 and x2, computed along dim.

    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}

    Args:
        dim (int, optional): Dimension where cosine similarity is computed. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8

    Shape:
        - Input1: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`
        - Input2: :math:`(\ast_1, D, \ast_2)`, same shape as the Input1
        - Output: :math:`(\ast_1, \ast_2)`

    Examples::

    >>> input1 = autograd.Variable(torch.randn(100, 128))
    >>> input2 = autograd.Variable(torch.randn(100, 128))
    >>> cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    >>> output = cos(input1, input2)
    >>> print(output)
    """
    def __init__(self, dim=1, eps=1e-8):
        super(CosineSimilarity, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        return F.cosine_similarity(x1, x2, self.dim, self.eps)


# TODO: Cosine
# TODO: Euclidean
# TODO: WeightedEuclidean
