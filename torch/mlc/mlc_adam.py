import torch
from .mlc_optimizer import MLCOptimizer
from ..optim.adam import Adam


class MLCAdam(MLCOptimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_.
            This is currently not supported with MLC backend. (default: False)
        max_gradient_clipping(float, optional):  maximum gradient to be clipped(default: inf)
        min_gradient_clipping(float, optional):  minimum gradient to be clipped(default: -inf)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    """

    def __init__(self,
                 params,
                 lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 max_gradient_clipping=float("inf"),
                 min_gradient_clipping=float("-inf")):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if amsgrad:
            raise RuntimeError("amsgrad is not supported with MLC backend.")
        use_gradient_clipping = True if (max_gradient_clipping is not float("inf") or
                                         min_gradient_clipping is not float("-inf")) else False
        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        amsgrad=amsgrad,
                        use_gradient_clipping=use_gradient_clipping,
                        max_gradient_clipping=max_gradient_clipping,
                        min_gradient_clipping=min_gradient_clipping)
        params = list(params)
        super(MLCAdam, self).__init__(params, defaults)
        self.mlcopt = torch._C.MLCOptimizerAdam(lr,
                                                betas[0],
                                                betas[1],
                                                eps,
                                                weight_decay,
                                                use_gradient_clipping,
                                                max_gradient_clipping,
                                                min_gradient_clipping)
        self.torchopt = Adam(params,
                             lr=lr,
                             betas=betas,
                             eps=eps,
                             weight_decay=weight_decay,
                             amsgrad=amsgrad)



    def __setstate__(self, state):
        super(MLCAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)


    def _same_parameters(self, group):
        keys = ['betas', 'eps', 'weight_decay']
        return all(self.defaults[k] == group[k] for k in keys)
