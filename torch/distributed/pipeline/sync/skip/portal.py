# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""Portal keeps a tensor in the pocket plane. The tensor becomes hidden to the
autograd engine. The shared context of three functions (:class:`PortalBlue`,
:class:`PortalOrange`, and :class:`PortalCopy`) out of the computation graph is
one of the most important feature of :mod:`torchpipe.skip`.

The metaphor is inspired by Portal™ from Valve.

"""
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from ..copy import Context as CopyContext
from ..copy import Copy
from ..phony import get_phony
from ..stream import AbstractStream, get_device

__all__: List[str] = []


class Portal:
    """A portal for a tensor."""

    def __init__(self, tensor: Optional[Tensor], tensor_life: int) -> None:
        self.put_tensor(tensor, tensor_life)
        self.grad: Optional[Tensor] = None

    def blue(self) -> Tensor:
        """Creates a :class:`PortalBlue` which hides the underlying tensor from
        the autograd engine.

        Join the returning phony to the main lane of the autograd graph to
        assure the correct backpropagation::

            PortalBlue --+
                         |
            ---------- Join --

        """
        tensor = self.use_tensor()

        if tensor is None:
            return get_phony(torch.device("cpu"), requires_grad=False)

        return PortalBlue.apply(self, tensor)

    def orange(self, phony: Tensor) -> Optional[Tensor]:
        """Creates a :class:`PortalOrange` which retrieves the hidden tensor
        without losing ability of backpropagation.

        Give a phony forked from the main lane of an autograd graph::

                +-- PortalOrange --+
                |                  |
            -- Fork --------- f(a, b) --

        """
        self.check_tensor_life()

        if self.tensor is None:
            return self.use_tensor()

        return PortalOrange.apply(self, phony)

    def copy(self, prev_stream: AbstractStream, next_stream: AbstractStream, phony: Tensor,) -> Tensor:
        """Copies the hidden tensor by a :class:`PortalCopy`.

        Give a phony and use the returning phony to keep backpropagation::

                +-- PortalCopy --+
                |                |
            -- Fork ---------- Join --

        """
        if self.tensor is None:
            return get_phony(torch.device("cpu"), requires_grad=False)

        return PortalCopy.apply(self, prev_stream, next_stream, phony)

    def check_tensor_life(self) -> None:
        if self.tensor_life <= 0:
            raise RuntimeError("tensor in portal has been removed")

    def put_tensor(self, tensor: Optional[Tensor], tensor_life: int) -> None:
        """Stores a tensor into this portal."""
        # [Life of Tensor through Portal]
        #
        # The tensor can be retrieved by use_tensor() up to 'tensor_life'
        # times. When the life becomes 0, the tensor will be deleted for
        # deallocation in CUDA memory.
        #
        # The below events participate in a tensor through a portal.
        # Note that [x] denotes the events which call use_tensor():
        #
        #  1. [x] blue()
        #  2. [ ]   PortalBlue.forward
        #  3. [ ] copy()
        #  4. [ ]   PortalCopy.forward
        #  5. [ ] orange()
        #  6. [x]   PortalOrange.forward
        # - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #  7. [ ] orange() (recomputed)
        #  8. [x]   PortalOrange.forward (recomputed)
        #  9. [ ]   PortalOrange.backward
        # 10. [ ] PortalCopy.backward
        # 11. [x] blue() (recomputed)
        # 12. [ ]   PortalBlue.forward (recomputed)
        # 13. [ ]   PortalBlue.backward
        #
        self.tensor_life = tensor_life

        if tensor_life > 0:
            self.tensor = tensor
        else:
            self.tensor = None

    def use_tensor(self) -> Optional[Tensor]:
        """Retrieves the underlying tensor and decreases the tensor  life. When
        the life becomes 0, it the tensor will be removed.
        """
        self.check_tensor_life()

        tensor = self.tensor

        self.tensor_life -= 1

        if self.tensor_life <= 0:
            self.tensor = None

        return tensor

    def put_grad(self, grad: Tensor) -> None:
        """Stores a gradient into this portal."""
        self.grad = grad

    def use_grad(self) -> Tensor:
        """Retrieves and removes the underlying gradient. The gradient is
        always ephemeral.
        """
        if self.grad is None:
            raise RuntimeError("grad in portal has been removed or never set")

        grad = self.grad
        self.grad = None
        return grad


# Common interface between :class:`PortalBlue`, :class:`PortalOrange`, and
# :class:`PortalCopy`.
class Context(CopyContext):
    portal: Portal


class PortalBlue(torch.autograd.Function):
    """Hides a tensor from the autograd engine by a :class:`Portal`."""

    @staticmethod
    # type: ignore
    def forward(
        ctx: Context,
        portal: Portal,
        # This tensor must be retrieved by portal.use_tensor().
        tensor: Tensor,
    ) -> Tensor:
        ctx.portal = portal

        phony = get_phony(tensor.device, requires_grad=False)
        return phony.detach()

    @staticmethod
    # type: ignore
    def backward(ctx: Context, grad_phony: Tensor,) -> Tuple[None, Tensor]:
        # The paired PortalOrange should keep the gradient.
        grad = ctx.portal.use_grad()
        return None, grad


class PortalOrange(torch.autograd.Function):
    """Retrieves the hidden tensor from a :class:`Portal`."""

    @staticmethod
    # type: ignore
    def forward(ctx: Context, portal: Portal, phony: Tensor) -> Tensor:
        ctx.portal = portal

        tensor = portal.use_tensor()
        assert tensor is not None

        return tensor.detach()

    @staticmethod
    def backward(ctx: Context, grad: Tensor) -> Tuple[None, None]:  # type: ignore
        # The paired PortalBlue will use the gradient.
        ctx.portal.put_grad(grad)
        return None, None


class PortalCopy(torch.autograd.Function):
    """Copies the hidden tensor in a :class:`Portal`. It replaces the hidden
    tensor with copied one.
    """

    @staticmethod
    # type: ignore
    def forward(
        ctx: Context, portal: Portal, prev_stream: AbstractStream, next_stream: AbstractStream, phony: Tensor,
    ) -> Tensor:
        ctx.portal = portal

        assert portal.tensor is not None
        (portal.tensor,) = Copy.forward(ctx, prev_stream, next_stream, portal.tensor)

        phony = get_phony(get_device(next_stream), requires_grad=False)
        return phony.detach()

    @staticmethod
    # type: ignore
    def backward(ctx: Context, grad_phony: Tensor,) -> Tuple[None, None, None, None]:
        portal = ctx.portal

        assert portal.grad is not None
        _, _, portal.grad = Copy.backward(ctx, portal.grad)

        return None, None, None, None
