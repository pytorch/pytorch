r"""Functional interface"""
from torch import Tensor
from typing import List, Optional

def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool):
    r"""Differentiable Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = d_p
                momentum_buffer_list[i] = buf
            else:
                buf = buf * momentum + d_p * (1 - dampening)
                momentum_buffer_list[i] = buf

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        params[i] = params[i] - d_p * lr
