import torch
from .BatchNormalization import BatchNormalization


class SpatialBatchNormalization(BatchNormalization):
    """
       This class implements Batch Normalization as described in the paper:
       "Batch Normalization: Accelerating Deep Network Training
                             by Reducing Internal Covariate Shift"
                   by Sergey Ioffe, Christian Szegedy

       This implementation is useful for inputs coming from convolution layers.
       For non-convolutional layers, see BatchNormalization.lua

       The operation implemented is:
               (x - mean(x))
       y = --------------------- * gamma + beta
           standard-deviation(x)
       where gamma and beta are learnable parameters.

       The learning of gamma and beta is optional.

       Usage:
       with    learnable parameters: nn.SpatialBatchNormalization(N [, eps] [, momentum])
                                   where N = dimensionality of input
       without learnable parameters: nn.SpatialBatchNormalization(N [, eps] [, momentum], False)

       eps is a small value added to the variance to avoid divide-by-zero.
           Defaults to 1e-5

       In training time, this layer keeps a running estimate of it's computed mean and std.
       The running sum is kept with a default momentum of 0.1 (unless over-ridden)
       In test time, this running mean/std is used to normalize.
    """

    # expected dimension of input
    nDim = 4
