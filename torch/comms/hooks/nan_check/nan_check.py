# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""
NanCheckHook for detecting NaN values in tensors before collective operations.

Catches numerical instability early before it propagates across ranks.

Example:
    >>> # xdoctest: +SKIP("requires a CUDA device and a configured communicator")
    >>> from torch.comms.hooks import NanCheckHook
    >>> import torch.comms
    >>> comm = torch.comms.new_comm("nccl", device, "world")
    >>> nan_check = NanCheckHook()
    >>> nan_check.register_with_comm(comm)
    >>> # NaN in tensors will now raise RuntimeError before collective runs
"""

from __future__ import annotations

from typing import Any

import torch
from torch._C._comms import OpName


__all__ = ["NanCheckHook"]


# Lazily resolved reference to torch.ops.c10d.check_for_nan.
# The op is registered when torch.distributed is imported (which happens
# before any communicator is created), so we resolve on first use to
# avoid a circular import at module load time.
# pyre-ignore[5]: Global has no type annotation.
_check_for_nan = None


def _get_check_for_nan() -> Any:
    global _check_for_nan
    if _check_for_nan is None:
        _check_for_nan = torch.ops.c10d.check_for_nan
    return _check_for_nan


def _check_tensor(tensor: Any, label: str, op_name: str, comm_name: str) -> None:
    """Check a single tensor for NaN values using c10d::check_for_nan."""
    try:
        _get_check_for_nan()(tensor)
    except RuntimeError as e:
        raise RuntimeError(
            f"NaN detected in {label} tensor for '{op_name}' on comm '{comm_name}': {e}"
        ) from None


def _check_single(tensor: Any, label: str, op_name: str, comm_name: str) -> None:
    if tensor.is_floating_point():
        _check_tensor(tensor, label, op_name, comm_name)


def _check_list(tensors: Any, label: str, op_name: str, comm_name: str) -> None:
    for tensor in tensors:
        if tensor.is_floating_point():
            _check_tensor(tensor, label, op_name, comm_name)


# Ops with a single .tensor field that is both input and output (in-place)
_INPLACE_OPS: frozenset[OpName] = frozenset(
    {
        OpName.broadcast,
        OpName.all_reduce,
        OpName.reduce,
    }
)

# Ops with .input (single tensor) and .output (single tensor)
_SINGLE_IO_OPS: frozenset[OpName] = frozenset(
    {
        OpName.all_gather_single,
        OpName.reduce_scatter_single,
        OpName.all_to_all_single,
        OpName.all_to_all_v_single,
        OpName.gather_single,
    }
)

# Ops with .input (single tensor) and .output (tensor list)
_SINGLE_IN_LIST_OUT_OPS: frozenset[OpName] = frozenset(
    {
        OpName.all_gather,
        OpName.all_gather_v,
        OpName.gather,
    }
)

# Ops with .input (tensor list) and .output (single tensor)
_LIST_IN_SINGLE_OUT_OPS: frozenset[OpName] = frozenset(
    {
        OpName.reduce_scatter,
        OpName.reduce_scatter_v,
        OpName.scatter,
    }
)

# Ops with .input (tensor list) and .output (tensor list)
_LIST_IO_OPS: frozenset[OpName] = frozenset(
    {
        OpName.all_to_all,
    }
)


class NanCheckHook:
    """Hook that checks for NaN values in tensors before collective operations.

    Registers a pre-hook on communicators that inspects input and/or output
    tensors for NaN values using the dispatched ``c10d::check_for_nan``
    op (works on both CPU and CUDA). If detected, raises a ``RuntimeError``
    with context about which operation and communicator triggered the check.

    Args:
        check_inputs: Whether to check input tensors. Default: True.
        check_outputs: Whether to check output tensors. Default: False.
    """

    def __init__(
        self,
        check_inputs: bool = True,
        check_outputs: bool = False,
    ) -> None:
        self._check_inputs = check_inputs
        self._check_outputs = check_outputs
        self._registered_count: int = 0

    def register_with_comm(self, comm: Any) -> None:
        """Register the NaN check hook with a communicator.

        Args:
            comm: A TorchComm communicator instance.
        """
        comm_name: str = comm.get_name()

        def _pre_hook(name: OpName, op_id: int, args: Any) -> None:
            op_str = str(name).rsplit(".", 1)[-1]

            if name == OpName.send:
                if self._check_inputs:
                    _check_single(args.tensor, "input", op_str, comm_name)

            elif name == OpName.recv:
                if self._check_outputs:
                    _check_single(args.tensor, "output", op_str, comm_name)

            elif name in _INPLACE_OPS:
                if self._check_inputs or self._check_outputs:
                    _check_single(args.tensor, "input", op_str, comm_name)

            elif name in _SINGLE_IO_OPS:
                if self._check_inputs:
                    _check_single(args.input, "input", op_str, comm_name)
                if self._check_outputs:
                    _check_single(args.output, "output", op_str, comm_name)

            elif name in _SINGLE_IN_LIST_OUT_OPS:
                if self._check_inputs:
                    _check_single(args.input, "input", op_str, comm_name)
                if self._check_outputs:
                    _check_list(args.output, "output", op_str, comm_name)

            elif name in _LIST_IN_SINGLE_OUT_OPS:
                if self._check_inputs:
                    _check_list(args.input, "input", op_str, comm_name)
                if self._check_outputs:
                    _check_single(args.output, "output", op_str, comm_name)

            elif name in _LIST_IO_OPS:
                if self._check_inputs:
                    _check_list(args.input, "input", op_str, comm_name)
                if self._check_outputs:
                    _check_list(args.output, "output", op_str, comm_name)

            # barrier, split, new_window, finalize: no tensors to check

        comm.register_pre_hook(_pre_hook)
        self._registered_count += 1

    def is_enabled(self) -> bool:
        """Return whether any communicators are registered."""
        return self._registered_count > 0
