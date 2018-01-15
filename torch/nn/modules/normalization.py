from .module import Module
from .. import functional as F


class LocalResponseNorm2d(Module):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        r"""Applies local response normalization:

        .. math::

            `b_{c,i,j} = a_{c,i,j}\left(k + \alpha\sum_{c'=\max(0, c-n/2)}^{\min(N-1,c+n/2)}
            a_{c',i,j}^2\right)^{-\beta}`

        Args:
            size: amount of neighbouring channels used for normalization
            alpha: multiplicative factor
            beta: root factor
            k: additive factor

        Shape:
            - Input: :math:`(N, C, H, W)`
            - Output: :math:`(N, C, H, W)`
        Examples::
            >>> lrn = nn.LocalResponseNorm2d(3)
            >>> input = autograd.Variable(torch.randn(32, 5, 24, 24))
            >>> output = lrn(input)
        """
        super(LocalResponseNorm2d, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        return F.local_response_norm(input, self.size, self.alpha, self.beta,
                                     self.k)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.size) \
            + ', alpha=' + str(self.alpha) \
            + ', beta=' + str(self.beta) \
            + ', k=' + str(self.k) + ')'


class CrossMapLRN2d(Module):

    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        super(CrossMapLRN2d, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        return self._backend.CrossMapLRN2d(self.size, self.alpha, self.beta,
                                           self.k)(input)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.size) \
            + ', alpha=' + str(self.alpha) \
            + ', beta=' + str(self.beta) \
            + ', k=' + str(self.k) + ')'


# TODO: ContrastiveNorm2d
# TODO: DivisiveNorm2d
# TODO: SubtractiveNorm2d
