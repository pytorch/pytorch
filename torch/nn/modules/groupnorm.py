import torch
from .module import Module


class GroupNorm2D(Module):
    r"""Applies Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization`_ .
    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x]} + \epsilon} * \gamma + \beta
    The mean and standard-deviation are calculated separately per mini-batch.
    :math:`\gamma` and :math:`\beta` are hyper-parameters
    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)
    Examples:
        >>> input = torch.randn(32, 64, 32, 32)
        >>> # With Default Hyper-Parameters
        >>> m = nn.GroupNorm2D(gamma=1, beta=0.5, group=32, eps=1e-5)
        >>> # Activating the module
        >>> output = m(input)
    .. _`Group Normalization`: https://arxiv.org/abs/1803.08494
    """

    def __init__(self, gamma=1, beta=0.5, group=32, eps=1e-5):
        super(GroupNorm2D, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.group = group
        self.eps = eps

    def _check_input_dim(self, inputs):
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(inputs.dim()))

    def forward(self, inputs):
        self._check_input_dim(inputs)

        N, C, H, W = inputs.size()
        inputs = inputs.view(N, self.group, C // self.group, H, W)

        mean, var = inputs.mean(1, keepdim=True), inputs.var(1, keepdim=True)
        inputs = (inputs - mean) / torch.sqrt(var + self.eps)
        inputs = inputs.view(N, C, H, W)

        return inputs * self.gamma + self.beta
