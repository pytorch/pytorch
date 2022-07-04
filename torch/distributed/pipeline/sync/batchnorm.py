# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""Tracks the running statistics per mini-batch instead of micro-batch."""
from typing import TypeVar, cast

import torch
from torch import Tensor, nn
from torch.nn.functional import batch_norm
from torch.nn.modules.batchnorm import _BatchNorm

from .checkpoint import is_recomputing

__all__ = ["DeferredBatchNorm"]


TModule = TypeVar("TModule", bound=nn.Module)


class DeferredBatchNorm(_BatchNorm):
    """A BatchNorm layer tracks multiple micro-batches to update running
    statistics per mini-batch.
    """

    sum: Tensor
    sum_squares: Tensor
    running_mean: Tensor
    running_var: Tensor
    num_batches_tracked: Tensor

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        chunks: int = 1,
    ) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats=True)

        self.register_buffer("sum", torch.zeros_like(self.running_mean))
        self.register_buffer("sum_squares", torch.zeros_like(self.running_var))

        self.counter = 0
        self.tracked = 0
        self.chunks = chunks

    def _check_input_dim(self, input: Tensor) -> None:
        # It's the typical _check_input_dim() implementation in PyTorch.
        if input.dim() <= 2:
            raise ValueError("expected at least 3D input (got %dD input)" % input.dim())

    def _track(self, input: Tensor) -> bool:
        """Tracks statistics of a micro-batch."""
        # Dimensions except channel. For example, (0, 2, 3) is for BatchNorm2d.
        dim = [0]
        dim.extend(range(2, input.dim()))

        with torch.no_grad():
            self.sum += input.sum(dim)
            self.sum_squares += (input ** 2).sum(dim)

        size = input.size().numel() // input.size(1)
        self.counter += size
        self.tracked += 1

        return self.tracked == self.chunks

    def _commit(self) -> None:
        """Updates the running statistics of a mini-batch."""
        exponential_average_factor = 0.0
        self.num_batches_tracked += 1
        if self.momentum is None:  # use cumulative moving average
            exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        else:  # use exponential moving average
            exponential_average_factor = self.momentum

        mean = self.sum / self.counter
        var = self.sum_squares / self.counter - mean ** 2

        # Calculate the exponential moving average here.
        m = exponential_average_factor

        self.running_mean *= 1 - m
        self.running_mean += mean * m

        self.running_var *= 1 - m
        self.running_var += var * m

        self.sum.zero_()
        self.sum_squares.zero_()
        self.counter = 0
        self.tracked = 0

    def forward(self, input: Tensor) -> Tensor:
        if not self.training:
            # Don't train parameters on the evaluation mode.
            return batch_norm(
                input,
                running_mean=self.running_mean,
                running_var=self.running_var,
                weight=self.weight,
                bias=self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )

        if not is_recomputing():
            # Track a micro-batch on the training mode
            # but not under a recomputation.
            tracked_enough = self._track(input)

            # Update the running statistics for a mini-batch
            # if it has tracked enough micro-batches.
            if tracked_enough:
                self._commit()

        # Normalize a micro-batch and train the parameters.
        return batch_norm(
            input,
            running_mean=None,
            running_var=None,
            weight=self.weight,
            bias=self.bias,
            training=True,
            momentum=0.0,
            eps=self.eps,
        )

    @classmethod
    def convert_deferred_batch_norm(cls, module: TModule, chunks: int = 1) -> TModule:
        """Converts a :class:`nn.BatchNorm` or underlying
        :class:`nn.BatchNorm`s into :class:`DeferredBatchNorm`::

            from torchvision.models.resnet import resnet101
            from torchpipe.batchnorm import DeferredBatchNorm
            model = resnet101()
            model = DeferredBatchNorm.convert_deferred_batch_norm(model)

        """
        if isinstance(module, DeferredBatchNorm) and module.chunks is chunks:
            return cast(TModule, module)

        module_output: nn.Module = module

        if isinstance(module, _BatchNorm) and module.track_running_stats:
            module_output = DeferredBatchNorm(module.num_features, module.eps, module.momentum, module.affine, chunks)
            if module.affine:
                module_output.register_parameter("weight", module.weight)
                module_output.register_parameter("bias", module.bias)
            module_output.register_buffer("running_mean", module.running_mean)
            module_output.register_buffer("running_var", module.running_var)
            module_output.register_buffer("num_batches_tracked", module.num_batches_tracked)

        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_deferred_batch_norm(child, chunks))

        return cast(TModule, module_output)
