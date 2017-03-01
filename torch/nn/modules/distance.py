
# TODO: Cosine
# TODO: CosineDistance - make sure lua's CosineDistance isn't actually cosine similarity
# TODO: Euclidean
# TODO: WeightedEuclidean

from .module import Module
from torch.autograd.variable import Norm

class PairwiseDistance(Module):
    def __init__(self, norm_type, dim=1):
        super(PairwiseDistance, self).__init__()
        self.norm_type = norm_type
        self.dim = dim

    def forward(self, input):
        return Norm(self.norm_type, self.dim)(input[1]-input[0])

