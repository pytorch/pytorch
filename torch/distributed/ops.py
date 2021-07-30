# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
The ``torch.distributed.ops`` module contains a set of autograd-aware collective
functions.

These functions are synchronous and will be inserted in the autograd graph. You
need to ensure that all ranks participating in the operation perform a backward
pass for the backward communication to effectively happen.
"""

import torch
import torch.distributed as dist

from torch.autograd import Function
from typing import Any, Optional, Tuple


def sum(input: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    """Computes the element-wise sum across ranks.

    Arguments:
        input (Tensor):
            The input of this rank.
        group (ProcessGroup, optional):
            The process group to work on. If ``None``, the default process group
            will be used.

    Returns:
        The element-wise sum of all inputs.
    """
    return _DistributedSum.apply(input, group)


class _DistributedSum(Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, input: torch.Tensor, group: dist.ProcessGroup
    ) -> torch.Tensor:
        ctx.group = group

        # We have to clone `input` to prevent in-place writes by `all_reduce`.
        output = input.clone()

        dist.all_reduce(output, dist.ReduceOp.SUM, ctx.group)

        return output

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        grad_output = sum(grad_output, ctx.group)

        return (grad_output, None)


def sum_on_rank(
    dst: int, input: torch.Tensor, group: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    """Computes the element-wise sum across ranks and returns it on ``dst``.

    Arguments:
        dst (int):
            The destination rank on which to return the result.
        input (Tensor):
            The input of this rank.
        group (ProcessGroup, optional):
            The process group to work on. If ``None``, the default process group
            will be used.

    Returns:
        The element-wise sum of all inputs on ``dst``, and a zero tensor
        on other ranks.
    """
    return _DistributedSumOnRank.apply(dst, input, group)


class _DistributedSumOnRank(Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, dst: int, input: torch.Tensor, group: dist.ProcessGroup
    ) -> torch.Tensor:
        ctx.dst = dst

        ctx.group = group

        # We have to clone `input` to prevent in-place writes by `reduce`.
        if dist.get_rank(ctx.group) == ctx.dst:
            input = input.clone()

        dist.reduce(input, ctx.dst, dist.ReduceOp.SUM, ctx.group)

        # Since there is no logical output on ranks other than `dst`, we just
        # return a zero tensor.
        if dist.get_rank(ctx.group) != ctx.dst:
            input = torch.zeros_like(input)

        return input

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[None, torch.Tensor, None]:
        grad_output = copy(ctx.dst, grad_output, ctx.group)

        return (None, grad_output, None)


def prod(input: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    """Computes the element-wise product across ranks.

    Arguments:
        input (Tensor):
            The input of this rank.
        group (ProcessGroup, optional):
            The process group to work on. If ``None``, the default process group
            will be used.

    Returns:
        The element-wise (a.k.a. Hadamard) product of all inputs.
    """
    return _DistributedProd.apply(input, group)


class _DistributedProd(Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, input: torch.Tensor, group: dist.ProcessGroup
    ) -> torch.Tensor:
        ctx.group = group

        # We have to clone `input` to prevent in-place writes by `all_reduce`.
        output = input.clone()

        dist.all_reduce(output, dist.ReduceOp.PRODUCT, ctx.group)

        # We effectively revert back our contribution to the product and use the
        # result as the coefficient of our input.
        ctx.coefficient = output / input

        return output

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        grad_output = sum(grad_output, ctx.group)

        return (grad_output * ctx.coefficient, None)


def minimum(input: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    """Computes the element-wise minimum across ranks.

    Arguments:
        input (Tensor):
            The input of this rank.
        group (ProcessGroup, optional):
            The process group to work on. If ``None``, the default process group
            will be used.

    Returns:
        The element-wise minimum of all inputs.
    """
    return _DistributedMinMax.apply(input, group, dist.ReduceOp.MIN)


def maximum(input: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    """Computes the element-wise maximum across ranks.

    Arguments:
        input (int):
            The input of this rank.
        group (ProcessGroup, optional):
            The process group to work on. If ``None``, the default process group
            will be used.

    Returns:
        The element-wise maximum of all inputs.
    """
    return _DistributedMinMax.apply(input, group, dist.ReduceOp.MAX)


class _DistributedMinMax(Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, input: torch.Tensor, group: dist.ProcessGroup, op: dist.ReduceOp
    ) -> torch.Tensor:
        ctx.group = group

        group_size = dist.get_world_size(ctx.group)

        # Gather all inputs to compute the minimum or maximum. Note that we need
        # the individual inputs in the backward pass; therefore, we can't use
        # `all_reduce` here.
        inputs = [torch.zeros_like(input) for _ in range(group_size)]

        dist.all_gather(inputs, input, group)

        # Compute the element-wise minimum or maximum of all inputs.
        output = inputs[0].clone()
        for i in inputs[1:]:
            if op == dist.ReduceOp.MIN:
                output = torch.minimum(output, i)
            else:
                output = torch.maximum(output, i)

        rank = dist.get_rank(ctx.group)

        # We save the inputs of other ranks in the context since we need them to
        # compute the gradients in the backward pass.
        del inputs[rank]

        ctx.other_inputs = inputs

        ctx.save_for_backward(input, output)

        return output

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None]:
        input: torch.Tensor
        output: torch.Tensor

        input, output = ctx.saved_tensors

        grad_output = sum(grad_output, ctx.group)

        # If the inputs of other ranks have the same minima or maxima, we should
        # scale down the corresponding input gradients.
        scale = torch.ones_like(grad_output)
        for i in ctx.other_inputs:
            scale += torch.where(input == i, 1, 0)

        grad_input = grad_output / scale

        # Zero out all input gradients for which our input is not the minimum or
        # maximum.
        grad_input.masked_fill_(input != output, 0)

        return (grad_input, None, None)


def copy(src: int, input: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    """Copies ``input`` from the source rank to other ranks.

    Arguments:
        src (int):
            The source rank which holds the actual input.
        input (Tensor):
            On the source rank ``input`` represents the tensor to be copied to
            other ranks; on other ranks it is a tensor with the same shape and
            data type used to save the received data.

    Returns:
        On the source rank a view of ``input`` with a custom autograd function;
        on other ranks ``input`` itself.

    Note:
        This function is differentiable, meaning gradients will flow back from
        the result of this operation to ``input``.

        Make sure to use the return value of this function instead of ``input``
        for further autograd operations; otherwise, the gradients will not be
        calculated correctly.
    """
    return _DistributedCopy.apply(src, input, group)


class _DistributedCopy(Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, src: int, input: torch.Tensor, group: dist.ProcessGroup
    ) -> torch.Tensor:
        ctx.src = src

        ctx.group = group

        dist.broadcast(input, ctx.src, ctx.group)

        # For ranks other than `src`, the data of `input` will be overwritten
        # by `broadcast`; therefore, we have to explicitly update the version
        # counter of `input`'s storage.
        if dist.get_rank(ctx.group) != ctx.src:
            ctx.mark_dirty(input)

        # Note that on `src` the autograd engine will internally create a view
        # of `input`, attach a new `grad_fn` to the view, and return that view
        # instead of `input` to the caller.
        return input

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[None, torch.Tensor, None]:
        grad_output = sum_on_rank(ctx.src, grad_output, ctx.group)

        return (None, grad_output, None)
