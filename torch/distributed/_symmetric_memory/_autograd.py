from typing import Optional

import torch
import torch.distributed._symmetric_memory as symm_mem


class AllToAllVDev2d(torch.autograd.Function):
    """
    Autograd function for `all_to_all_vdev_2d`
    Usage:
        >>> # xdoctest: +SKIP("undefined vars")
        >>> # Create input and output tensors
        >>> input = symm_mem.empty(...)
        >>> out = symm_mem.empty(...)
        >>> # Create input splits and output splits/offsets
        >>> input_splits = symm_mem.empty(...)
        >>> out_splits_offsets = symm_mem.empty(...)
        >>> # Forward
        >>> output = AllToAllVDev2d.apply(
        ...     input, out, input_splits, out_splits_offsets, group_name, major_align
        ... )
        >>> # Backward
        >>> s = output.sum()
        >>> s.backward()
        >>> print(input.grad)
    """

    initialized: bool = False

    @staticmethod
    def forward(  # type: ignore[no-untyped-def]
        ctx,
        input: torch.Tensor,
        out: torch.Tensor,
        in_splits: torch.Tensor,
        out_splits_offsets: torch.Tensor,
        group_name: str,
        major_align: int,
        # Buffers needed for backward pass
        grad_out_buf: torch.Tensor,
        grad_in_buf: torch.Tensor,
        grad_in_splits_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Functionality is the same as `all_to_all_vdev_2d` but with autograd support.
        """
        # Shuffle input to output
        torch.ops.symm_mem.all_to_all_vdev_2d(
            input, out, in_splits, out_splits_offsets, group_name, major_align
        )

        # Output splits in forward is the input splits in backward
        ctx.save_for_backward(out_splits_offsets, grad_out_buf, grad_in_buf, grad_in_splits_offsets)
        ctx.group_name = group_name
        return out

    @staticmethod
    def backward(  # type: ignore[no-untyped-def]
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None, None, None]:
        """
        Backward pass of `all_to_all_vdev_2d` is `all_to_all_vdev_2d_offset`.

        Args:
            `grad_output`: gradients of output passed back from the downstream.

        Returns:
            `grad_input`: gradients of input.
        """
        # Splits info
        # Splits/offsets of grad_out is the same as out splits/offsets in forward
        (grad_out_splits_offsets, grad_out_buf, grad_in_buf, grad_in_splits_offsets) = ctx.saved_tensors

        # TODO: is there a way to tell autograd to feed grad_output directly to
        # our symm_mem buffer?
        grad_out_buf.narrow(0, 0, grad_output.shape[0]).copy_(
            grad_output
        )

        # Shuffle gradients back to the input
        torch.ops.symm_mem.all_to_all_vdev_2d_offset(
            grad_out_buf,
            grad_in_buf,
            grad_out_splits_offsets,
            grad_in_splits_offsets,
            group_name=ctx.group_name,
        )
        return grad_in_buf, None, None, None, None, None, None, None, None
