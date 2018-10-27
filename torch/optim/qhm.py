import math

import torch
from .optimizer import Optimizer, required


class QHM(Optimizer):
    r"""Implements the quasi-hyperbolic momentum (QHM) optimization algorithm
    `(Ma and Yarats, 2018)`_.

    Note that many other optimization algorithms are accessible via specific
    parameterizations of QHM. See :func:`~torch.optim.QHM.from_accsgd()`,
    :func:`~torch.optim.QHM.from_robust_momentum()`, etc. for details.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (:math:`\alpha` from the paper)
        momentum (float, optional): momentum factor (:math:`\beta` from the paper) (default: 0.0)
        nu (float, optional): immediate discount factor (:math:`\nu` from the paper) (default: 1.0)
        weight_decay (float, optional): weight decay (L2 regularization coefficient, times two)
            (default: 0)
        weight_decay_type (str, optional): method of applying the weight decay:
            ``"grad"`` for accumulation in the gradient (same as :class:`torch.optim.SGD`) or
            ``"direct"`` for direct application to the parameters
            (default: ``"grad"``)

    Example:
        >>> optimizer = torch.optim.QHM(model.parameters(), lr=1.0, nu=0.7, momentum=0.999)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    .. _`(Ma and Yarats, 2018)`: https://arxiv.org/abs/1810.06801

    .. note::

        Mathematically, QHM is a simple interpolation between plain SGD and momentum:

        .. math::

            g_{t + 1} \leftarrow \beta \cdot g_t + (1 - \beta) \cdot \nabla_t \\
            \theta_{t + 1} \leftarrow \theta_t + \alpha \left[ (1 - \nu) \cdot \nabla_t + \nu \cdot g_{t + 1} \right]

        Here, :math:`\alpha` is the learning rate, :math:`\beta` is the momentum factor, and
        :math:`\nu` is the "immediate discount" factor which controls the interpolation
        between plain SGD and momentum.
        :math:`g_t` is the momentum buffer, :math:`\theta_t` is the parameter vector, and
        :math:`\nabla_t` is the gradient with respect to :math:`\theta_t`.

    .. note::

        QHM uses **dampened** momentum. This means that when converting from plain momentum to QHM,
        the learning rate must be scaled by :math:`\frac{1}{1 - \beta}`. For example, momentum with
        learning rate :math:`\alpha = 0.1` and momentum :math:`\beta = 0.9` should be converted to
        QHM with learning rate :math:`\alpha = 1.0`.
    """

    def __init__(self, params, lr=required, momentum=0.0, nu=1.0,
                 weight_decay=0, weight_decay_type="grad"):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if weight_decay_type not in ("grad", "direct"):
            raise ValueError("Invalid weight_decay_type value: {}".format(weight_decay_type))

        defaults = dict(lr=lr, momentum=momentum, nu=nu,
                        weight_decay=weight_decay,
                        weight_decay_type=weight_decay_type)
        super(QHM, self).__init__(params, defaults)

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
            lr, nu, momentum = group["lr"], group["nu"], group["momentum"]
            weight_decay, weight_decay_type = group["weight_decay"], group["weight_decay_type"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]

                if weight_decay != 0:
                    if weight_decay_type == "grad":
                        d_p.add_(weight_decay, p.data)
                    elif weight_decay_type == "direct":
                        p.data.mul_(1.0 - lr * weight_decay)
                    else:
                        raise ValueError("Invalid weight decay type provided")

                if len(param_state) == 0:
                    param_state["momentum_buffer"] = torch.zeros_like(p.data)

                momentum_buffer = param_state["momentum_buffer"]
                momentum_buffer.mul_(momentum).add_(1.0 - momentum, d_p)

                p.data.add_(-lr * nu, momentum_buffer)
                p.data.add_(-lr * (1.0 - nu), d_p)

        return loss

    @classmethod
    def from_pid(cls, k_p, k_i, k_d):
        r"""Calculates the QHM hyperparameters required to recover a PID optimizer
        as described in `Recht (2018)`_.

        Args:
            k_p (float): proportional gain (see reference)
            k_i (float): integral gain (see reference)
            k_d (float): derivative gain (see reference)

        Returns:
            Three-element ``dict`` containing ``lr``, ``momentum``, and ``nu`` to use in QHM.

        Example:
            >>> optimizer = torch.optim.QHM(
            ...     model.parameters(),
            ...     weight_decay=1e-4,
            ...     **torch.optim.QHM.from_pid(k_p=-0.1, k_i=1.0, k_d=3.0))

        .. _`Recht (2018)`: https://web.archive.org/web/20181027184056/http://www.argmin.net/2018/04/19/pid/
        """
        lr = k_i
        nu = k_p * k_p / (k_i * k_d)
        momentum = k_d / (k_d - k_p)
        return {"lr": lr, "nu": nu, "momentum": momentum}

    @classmethod
    def from_synthesized_nesterov(cls, alpha, beta1, beta2):
        r"""Calculates the QHM hyperparameters required to recover the synthesized
        Nesterov optimizer (Section 6 of `Lessard et al. (2016)`_).

        Args:
            alpha (float): learning rate
            beta1 (float): first momentum (see reference)
            beta2 (float): second momentum (see reference)

        Returns:
            Three-element ``dict`` containing ``lr``, ``momentum``, and ``nu`` to use in QHM.

        Example:
            >>> optimizer = torch.optim.QHM(
            ...     model.parameters(),
            ...     weight_decay=1e-4,
            ...     **torch.optim.QHM.from_synthesized_nesterov(alpha=0.1, beta1=0.9, beta2=0.6))

        .. _`Lessard et al. (2016)`: https://arxiv.org/abs/1408.3595
        """
        lr = alpha / (1.0 - beta1)
        nu = 1.0 - ((1.0 - beta1) / beta1) * beta2
        momentum = beta1
        return {"lr": lr, "nu": nu, "momentum": momentum}

    @classmethod
    def from_robust_momentum(cls, l, kappa, rho=None):
        r"""Calculates the QHM hyperparameters required to recover the Robust Momentum
        `(Cyrus et al., 2018)`_ or Triple Momentum `(Scoy et al., 2018)`_ optimizers.

        Args:
            l (float): Lipschitz constant of gradient (see reference)
            kappa (float): condition ratio (see reference)
            rho (float, optional): noise-free convergence rate. If None, will return
                the parameters for the Triple Momentum optimizer.

        Returns:
            Three-element ``dict`` containing ``lr``, ``momentum``, and ``nu`` to use in QHM.

        Example:
            >>> optimizer = torch.optim.QHM(
            ...     model.parameters(),
            ...     weight_decay=1e-4,
            ...     **torch.optim.QHM.from_robust_momentum(l=5.0, kappa=15.0))

        .. _`(Cyrus et al., 2018)`: https://arxiv.org/abs/1710.04753

        .. _`(Scoy et al., 2018)`: http://www.optimization-online.org/DB_FILE/2017/03/5908.pdf
        """
        if rho is None:
            rho = 1.0 - 1.0 / math.sqrt(kappa)

        alpha = kappa * ((1.0 - rho) ** 2) * (1.0 + rho) / l
        beta1 = kappa * (rho ** 3) / (kappa - 1.0)
        beta2 = (rho ** 3) / ((kappa - 1.0) * ((1.0 - rho) ** 2) * (1.0 + rho))
        return cls.from_synthesized_nesterov(alpha, beta1, beta2)

    @classmethod
    def from_accsgd(cls, delta, kappa, xi, eps=0.7):
        r"""Calculates the QHM hyperparameters required to recover the AccSGD optimizer
        `(Kidambi et al., 2018)`_.

        Args:
            delta (float): short step (see reference)
            kappa (float): long step parameter (see reference)
            xi (float): statistical advantage parameter (see reference)
            eps (float): arbitrary value, between 0 and 1 exclusive (see reference) (default: 0.7)

        Returns:
            Three-element ``dict`` containing ``lr``, ``momentum``, and ``nu`` to use in QHM.

        Example:
            >>> optimizer = torch.optim.QHM(
            ...     model.parameters(),
            ...     weight_decay=1e-4,
            ...     **torch.optim.QHM.from_accsgd(delta=0.1, kappa=1000.0, xi=10.0))

        .. _`(Kidambi et al., 2018)`: https://arxiv.org/abs/1803.05591
        """
        lr = (delta * eps * (1.0 + xi)) / (1.0 + eps)
        nu = (eps * xi - 1.0) / (eps * (1.0 + xi))
        momentum = (kappa - (eps * eps) * xi) / (kappa + eps * xi)
        return {"lr": lr, "nu": nu, "momentum": momentum}

    @classmethod
    def from_two_state_optimizer(cls, h, k, l, m, q, z):
        r"""Calculates the QHM hyperparameters required to recover the following optimizer
        (named "TSO" in `Ma and Yarats (2018)`_):

        .. math::

            a_{t + 1} \leftarrow h \cdot a_t + k \cdot \theta_t + l \cdot \nabla_t \\
            \theta_{t + 1} \leftarrow m \cdot a_t + q \cdot \theta_t + z \cdot \nabla_t

        Here, :math:`a_t` and :math:`\theta_t` are the two states and :math:`\nabla_t` is
        the gradient with respect to :math:`\theta_t`.

        Be careful that your coefficients satisfy the regularity conditions from the reference.

        Args:
            h (float): see description
            k (float): see description
            l (float): see description
            m (float): see description
            q (float): see description
            z (float): see description

        Returns:
            Three-element ``dict`` containing ``lr``, ``momentum``, and ``nu`` to use in QHM.

        Example:
            >>> optimizer = torch.optim.QHM(
            ...     model.parameters(),
            ...     weight_decay=1e-4,
            ...     **torch.optim.QHM.from_two_state_optimizer(
            ...         h=0.9, k=0.0, l=0.1, m=-0.09, q=1.0, z=-0.01))

        .. _`Ma and Yarats (2018)`: https://arxiv.org/abs/1810.06801
        """
        phi = math.sqrt((h - q) * (h - q) + 4.0 * k * m)
        psi = k * m - h * q
        xi = (h - q - phi) * (l * m - h * z) + 2.0 * m * (l * q - k * z)

        lr = 0.5 * xi / (phi * psi)
        nu = 2.0 * m * (l * q - k * z) / xi
        momentum = 0.5 * (h + q - phi)
        return {"lr": lr, "nu": nu, "momentum": momentum}
