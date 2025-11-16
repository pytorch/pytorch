import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from .module import Module
from .. import init


class StateSpaceModel(Module):
    r"""
    A small, generic state space layer for sequence data.

    This is intentionally simple and written in terms of existing PyTorch ops so
    that it can act as a reference implementation for more optimized kernels.

    The layer implements a discrete-time linear state space system

        x_{t+1} = A x_t + B u_t
        y_t     = C x_t + D u_t

    where ``u_t`` is the input at time step ``t`` and ``y_t`` is the output.

    Args:
        input_size:   Size of the input features ``u_t``.
        state_size:   Size of the hidden state ``x_t``.
        output_size:  Size of the output features ``y_t``.
        bias:         If ``True``, adds a learnable bias to the output.
        init_stable:  If ``True``, initialize ``A`` with eigenvalues inside the unit circle.

    Shape:
        - Input:  ``(batch, time, input_size)``
        - Output: ``(batch, time, output_size)``

    Notes:
        This version uses a Python loop over the time dimension instead of a fused
        scan, on purpose, to keep the reference easy to follow and to avoid
        depending on any new low-level kernels.
    """

    def __init__(
        self,
        input_size: int,
        state_size: int,
        output_size: int,
        *,
        bias: bool = True,
        init_stable: bool = True,
    ) -> None:
        super().__init__()
        if input_size <= 0 or state_size <= 0 or output_size <= 0:
            raise ValueError("input_size, state_size and output_size must be positive")

        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size

        # State transition and readout matrices
        self.A = torch.nn.Parameter(torch.empty(state_size, state_size))
        self.B = torch.nn.Parameter(torch.empty(state_size, input_size))
        self.C = torch.nn.Parameter(torch.empty(output_size, state_size))
        self.D = torch.nn.Parameter(torch.empty(output_size, input_size))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters(init_stable=init_stable)

    def _reset_parameters(self, *, init_stable: bool) -> None:
        # Keep this deliberately simple; we want something numerically tame rather
        # than clever. As long as gradients flow and the layer behaves sensibly,
        # more advanced initialization can be explored in follow-up work.
        if init_stable:
            # Start from a small random matrix and push eigenvalues inward by
            # scaling down the spectral norm. We avoid depending on any LAPACK
            # calls here to keep it robust on all backends.
            init.orthogonal_(self.A)
            with torch.no_grad():
                self.A.mul_(0.1)
        else:
            init.xavier_uniform_(self.A)

        init.xavier_uniform_(self.B)
        init.xavier_uniform_(self.C)
        init.xavier_uniform_(self.D)

        if self.bias is not None:
            fan_in = self.output_size
            bound = 1.0 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        input: Tensor,
        state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input: Tensor of shape ``(batch, time, input_size)``.
            state: Optional initial state of shape ``(batch, state_size)``.
                   If omitted, it is initialized to zeros.

        Returns:
            output: Tensor of shape ``(batch, time, output_size)``.
            final_state: Tensor of shape ``(batch, state_size)``.
        """
        if input.dim() != 3:
            raise ValueError(
                f"StateSpaceModel expects input of shape (batch, time, input_size), got {tuple(input.shape)}"
            )

        batch, time, features = input.shape
        if features != self.input_size:
            raise ValueError(
                f"Expected input_size {self.input_size}, got {features}"
            )

        if state is None:
            # Keep the state dtype and device in sync with the input.
            state = input.new_zeros(batch, self.state_size)
        else:
            if state.shape != (batch, self.state_size):
                raise ValueError(
                    f"Expected state of shape (batch, {self.state_size}), got {tuple(state.shape)}"
                )

        A = self.A
        B = self.B
        C = self.C
        D = self.D
        bias = self.bias

        outputs = []
        x = state

        # The plain Python loop is not ideal for very long sequences, but it
        # makes the semantics obvious and easy to test. Fused scan kernels can
        # later drop in behind the same interface.
        for t in range(time):
            u_t = input[:, t, :]  # (batch, input_size)
            x = torch.matmul(x, A.t()) + torch.matmul(u_t, B.t())
            y = torch.matmul(x, C.t()) + torch.matmul(u_t, D.t())
            if bias is not None:
                y = y + bias
            outputs.append(y)

        output = torch.stack(outputs, dim=1)
        return output, x


__all__ = ["StateSpaceModel"]


