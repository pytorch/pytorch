import torch
from .optimizer import Optimizer, required


class QHAdam(Optimizer):
    r"""Implements the QHAdam optimization algorithm `(Ma and Yarats, 2018)`_.

    Note that the NAdam optimizer is accessible via a specific parameterization of
    of QHAdam. See :func:`~torch.optim.QHAdam.from_nadam()` for details.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (:math:`\alpha` from the paper) (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of the gradient and its square (default: (0.9, 0.999))
        nus (Tuple[float, float], optional): immediate discount factors used to
            estimate the gradient and its square (default: (1.0, 1.0))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 regularlization coefficient, times two)
            (default: 0)

    Example:
        >>> optimizer = torch.optim.QHAdam(
        ...     model.parameters(), lr=3e-4, nus=(0.8, 1.0), betas=(0.99, 0.999))
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    .. _`(Ma and Yarats, 2018)`: https://arxiv.org/abs/1810.06801
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), nus=(1.0, 1.0),
                 weight_decay=0, eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, nus=nus,
                        weight_decay=weight_decay, eps=eps)
        super(QHAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            nu1, nu2 = group["nus"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if d_p.is_sparse:
                    raise RuntimeError("QHAdam does not support sparse gradients")

                param_state = self.state[p]

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                d_p_sq = d_p.mul(d_p)

                if len(param_state) == 0:
                    param_state["step"] = 0
                    param_state["exp_avg"] = torch.zeros_like(p.data)
                    param_state["exp_avg_sq"] = torch.zeros_like(p.data)

                param_state['step'] += 1
                step = param_state['step']
                exp_avg = param_state['exp_avg']
                exp_avg_sq = param_state['exp_avg_sq']

                # Mathematical trick to remove the need for explicit bias correction
                short_ratio = (1.0 - beta1) / (1.0 - (beta1 ** step))
                long_ratio = (1.0 - beta2) / (1.0 - (beta2 ** step))
                exp_avg.mul_(1.0 - short_ratio).add_(short_ratio, d_p)
                exp_avg_sq.mul_(1.0 - long_ratio).add_(long_ratio, d_p_sq)

                avg_grad = exp_avg.mul(nu1)
                if nu1 != 1.0:
                    avg_grad.add_(1.0 - nu1, d_p)

                avg_grad_rms = exp_avg_sq.mul(nu2)
                if nu2 != 1.0:
                    avg_grad_rms.add_(1.0 - nu2, d_p_sq)
                avg_grad_rms.sqrt_()
                if eps != 0.0:
                    avg_grad_rms.add_(eps)

                p.data.addcdiv_(-lr, avg_grad, avg_grad_rms)

        return loss

    @classmethod
    def from_nadam(cls, lr=1e-3, betas=(0.9, 0.999)):
        r"""Calculates the QHAdam hyperparameters required to recover the NAdam optimizer
        `(Dozat, 2016)`_.

        This is *not* an identical recovery of the formulation in the paper, due to subtle
        differences in the application of the bias correction in the first moment estimator.
        However, in practice, this difference is almost certainly irrelevant.

        Args:
            lr (float, optional): learning rate (:math:`\alpha` from the paper) (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of the gradient and its square (default: (0.9, 0.999))

        Returns:
            Three-element ``dict`` containing ``lr``, ``betas``, and ``nus`` to use in QHAdam.

        Example:
            >>> optimizer = torch.optim.QHAdam(
            ...     model.parameters(),
            ...     weight_decay=1e-4,
            ...     **torch.optim.QHAdam.from_nadam(lr=1e-3, betas=(0.9, 0.999)))

        .. _`(Dozat, 2016)`: https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
        """
        nus = (betas[0], 1.0)
        return {"lr": lr, "nus": nus, "betas": tuple(betas)}
