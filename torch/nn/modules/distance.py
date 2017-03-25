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
            - Output: :math:`(N, 1)

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

class CosineSimilarity(Module):
    """
    Computes the batchwise cosine similarity between vectors v1,v2:

        .. math ::
            cossim(v_1,v_2) = \frac{v_1 * v_2}{\left| v_1 \right| * \left| v_2 \right|}

        Args: 
            - **x** of size (2, batch, size): Tensor containing the two input batches x1,x2 = x
    """

    def __init__(self):
        super(CosineSimilarity, self).__init__()
        self.eps = 1e-12

    def forward(self, x1, x2):
        return F.cosine_similarity(x1, x2, self.eps)

# TODO: CosineDistance - make sure lua's CosineDistance isn't actually cosine similarity
# TODO: Euclidean
# TODO: WeightedEuclidean
# TODO: PairwiseDistance
