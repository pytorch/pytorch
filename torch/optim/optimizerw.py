import torch
from .optimizer import Optimizer
from .adam import Adam
from .sgd import SGD


class OptimizerW(Optimizer):
    r"""Base class for decoupled weight decay optimizers.

    The distinction between weight decay and L2 regularisation is discussed in
    `Decoupled Weight Decay Regularization`_.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Arguments:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).

    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    """

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step (parameter update).

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. Optional for most optimizers.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group.pop('weight_decay')
            for p in group['params']:
                grad = p.grad
                if grad is None:
                    continue

                if grad.is_sparse:
                    if weight_decay > 0:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")

                    update = self.get_sparse_update(p, **group)
                else:
                    update = self.get_update(p, **group)

                p.mul_(1 - weight_decay * lr).add_(-lr, update)

        return loss


class SGDW(OptimizerW, SGD):
    r"""Implements SGDW algorithm.

    This variant of SGD was proposed in `Decoupled Weight Decay Regularization`_.

    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    """

    pass


class AdamW(OptimizerW, Adam):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    This variant of Adam was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    pass
