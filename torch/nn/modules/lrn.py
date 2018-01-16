from .. import functional as F
from .module import Module

class SpatialCrossMapLRN(Module):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        super(SpatialCrossMapLRN, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        return F.spatial_cross_map_lrn(self.size, self.alpha, self.beta, self.k)(input)