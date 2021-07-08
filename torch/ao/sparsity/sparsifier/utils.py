from torch import nn

# Parametrizations
class FakeSparsity(nn.Module):
    r"""Parametrization for the weights. Should be attached to the 'weight' or
    any other parmeter that requires a mask applied to it.

    Note::

        Once the mask is passed, the variable should not change the id. The
        contents of the mask can change, but the mask reference itself should
        not.
    """
    def __init__(self, mask):
        super().__init__()
        self.register_buffer('mask', mask)

    def forward(self, x):
        assert self.mask.shape == x.shape
        return self.mask * x
