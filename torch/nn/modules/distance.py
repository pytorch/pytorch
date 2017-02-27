import torch
from .module import Module
# TODO: Cosine


class CosineDistance(Module):
    """
    Computes cosine similarity between two vectors x1,x2:

          x1*x2
        ---------
        |x1|*|x2|

    """

    def __init__(self):
        super(CosineDistance, self).__init__()
        self.eps = 1e-12

    def forward(self, x):
        assert len(x) == 2, "Input needs to be two vectors"
        x1,x2 = x
        if x1.dim() == 1 or x2.dim() == 1:
            x1 = x1.view(1, -1)
            x2 = x2.view(1, -1)
        #
        #	 x1 * x2	 w12
        #	--------- = -----
        #	|x1|*|x2|	w1*w2
        #
        # Nominator
        w12 = torch.sum(x1 * x2, 1)
        w1 = 1. / (torch.norm(x1, 2, 1) + self.eps)
        w2 = 1. / (torch.norm(x2, 2, 1) + self.eps)
        return (w12 * (w1 * w2)).squeeze()

# TODO: Euclidean
# TODO: WeightedEuclidean
# TODO: PairwiseDistance
