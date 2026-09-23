# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import torch

from ._utils import _make_tensor_from_meta, PipeliningMetadataError, TensorMeta


_INCOMPLETE_RECV_BUFFER_ERROR = (
    "Receive buffer for {input_name!r} is still owned by an incomplete pipeline "
    "step; in-flight P2P work may still write to it. Call "
    "dist.destroy_process_group(), then reconstruct the stages and schedule; "
    "recreating the schedule alone does not make the storage safe to reuse"
)


class _RecvInfo:
    """Input tensor descriptor for a pipeline stage.

    Handles both received activations from a previous stage
    (``is_root_arg=False``) and root-level model inputs provided
    by the user (``is_root_arg=True``).
    """

    def __init__(
        self,
        input_name: str,
        source: int | None,
        tensor_meta: TensorMeta | None,
        *,
        is_root_arg: bool = False,
    ):
        # Name of this input
        self.input_name = input_name
        # Stage index of the source of this input (None for root args)
        self.source = source
        # Allocated immediately before recv and consumed by microbatch compute
        self.buffer: torch.Tensor | None = None
        # Tensor metadata for validation and DTensor reconstruction
        self.tensor_meta = tensor_meta
        # Whether this is a root-level model input (no recv needed)
        self.is_root_arg = is_root_arg

    def allocate_buffer(self, device: torch.device | str) -> torch.Tensor:
        """Allocate and retain this receive's runtime buffer."""
        if self.tensor_meta is None:
            raise PipeliningMetadataError(
                f"Receive '{self.input_name}' has no tensor metadata to allocate from"
            )
        if self.buffer is not None:
            raise PipeliningMetadataError(
                _INCOMPLETE_RECV_BUFFER_ERROR.format(input_name=self.input_name)
            )
        self.buffer = _make_tensor_from_meta(self.tensor_meta, device)
        return self.buffer

    def set_buffer(self, buffer: torch.Tensor) -> None:
        """Retain a receive buffer supplied by the runtime or a local stage."""
        if self.tensor_meta is None:
            raise PipeliningMetadataError(
                f"Receive '{self.input_name}' expects no gradient because its "
                "tensor metadata is None, but a gradient tensor was provided"
            )
        if self.buffer is not None:
            raise PipeliningMetadataError(
                _INCOMPLETE_RECV_BUFFER_ERROR.format(input_name=self.input_name)
            )
        self.buffer = buffer

    def take_buffer(self) -> torch.Tensor:
        """Transfer this descriptor's buffer reference to stage computation."""
        if self.buffer is None:
            raise PipeliningMetadataError(
                f"Receive buffer for '{self.input_name}' has not been set"
            )
        buffer = self.buffer
        # The schedule waits for recv completion before this transfer, so
        # clearing the descriptor remains safe if later reconstruction fails.
        self.buffer = None
        return buffer

    def __repr__(self):
        if self.is_root_arg:
            return f"_RecvInfo(input={self.input_name}, root_arg=True)"
        meta_type = type(self.tensor_meta).__name__ if self.tensor_meta else "None"
        shape = self.tensor_meta.shape if self.tensor_meta is not None else "None"
        buffer_state = "allocated" if self.buffer is not None else "unallocated"
        return (
            f"_RecvInfo(input={self.input_name}, source={self.source}, "
            f"shape={shape}, meta={meta_type}, buffer={buffer_state})"
        )


def _ensure_recv_infos_drained(
    recv_info_by_microbatch: dict[int, tuple[_RecvInfo, ...]],
) -> None:
    """Reject descriptor replacement while an interrupted step owns storage."""
    for recv_infos in recv_info_by_microbatch.values():
        for info in recv_infos:
            if info.buffer is not None:
                raise PipeliningMetadataError(
                    _INCOMPLETE_RECV_BUFFER_ERROR.format(input_name=info.input_name)
                )


def _clear_unlaunched_recv_infos(recv_infos: tuple[_RecvInfo, ...]) -> None:
    """Drop buffers from a receive batch that was never submitted to P2P."""
    for info in recv_infos:
        info.buffer = None


def _assign_recv_info_buffers(
    assignments: tuple[tuple[_RecvInfo, torch.Tensor], ...],
) -> None:
    """Assign a complete same-rank receive batch transactionally.

    Callers prepare every tensor before entering this helper. Validating every
    descriptor before mutation ensures that a failed local handoff leaves no
    partial ownership behind.

    Args:
        assignments: Receive descriptors paired with their prepared buffers.
    """
    for info, _ in assignments:
        if info.tensor_meta is None:
            raise PipeliningMetadataError(
                f"Receive '{info.input_name}' has no tensor metadata"
            )
        if info.buffer is not None:
            raise PipeliningMetadataError(
                _INCOMPLETE_RECV_BUFFER_ERROR.format(input_name=info.input_name)
            )
    for info, buffer in assignments:
        info.buffer = buffer
