import torch
from .mlc_optimizer import MLCOptimizer
from ..optim.rmsprop import RMSprop


class MLCRMSprop(MLCOptimizer):
    r"""Implements RMSprop algorithm.

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where :math:`\alpha`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        max_gradient_clipping(float, optional):  maximum gradient to be clipped(default: inf)
        min_gradient_clipping(float, optional):  minimum gradient to be clipped(default: -inf)

    """

    def __init__(self,
                 params,
                 lr=1e-2,
                 alpha=0.99,
                 eps=1e-8,
                 weight_decay=0,
                 momentum=0,
                 centered=False,
                 max_gradient_clipping=float("inf"), min_gradient_clipping=float("-inf")):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        use_gradient_clipping = True if (max_gradient_clipping is not float("inf") or
                                         min_gradient_clipping is not float("-inf")) else False
        defaults = dict(lr=lr,
                        momentum=momentum,
                        alpha=alpha,
                        eps=eps,
                        centered=centered,
                        weight_decay=weight_decay,
                        use_gradient_clipping=use_gradient_clipping,
                        max_gradient_clipping=max_gradient_clipping,
                        min_gradient_clipping=min_gradient_clipping)
        params = list(params)
        super(MLCRMSprop, self).__init__(params, defaults)
        self.mlcopt = torch._C.MLCOptimizerRMSProp(lr,
                                                   momentum,
                                                   alpha,
                                                   eps,
                                                   centered,
                                                   weight_decay,
                                                   use_gradient_clipping,
                                                   max_gradient_clipping,
                                                   min_gradient_clipping)
        self.torchopt = RMSprop(params,
                                lr=lr,
                                alpha=alpha,
                                eps=eps,
                                weight_decay=weight_decay,
                                momentum=momentum,
                                centered=centered)

    def __setstate__(self, state):
        super(MLCRMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)


    def _same_parameters(self, group):
        keys = ['alpha', 'eps', 'weight_decay', 'momentum', 'centered']
        return all(self.defaults[k] == group[k] for k in keys)
