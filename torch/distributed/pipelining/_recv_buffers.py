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
        # Assigned immediately before recv and consumed by microbatch compute.
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
        state = "set" if self.buffer is not None else "unset"
        return (
            f"_RecvInfo(input={self.input_name}, source={self.source}, "
            f"shape={shape}, meta={meta_type}, buffer={state})"
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


class _RecvBufferPool:
    """Fixed receive buffers with explicit per-slot ownership.

    The pool retains ordinary, autograd-neutral base tensors for stable storage
    addresses. Each acquisition gives the receive descriptor a fresh detached
    alias so per-lease autograd state cannot survive slot reuse.

    A slot is released only after its schedule-derived consumers finish.
    ProcessGroupNCCL then orders a later receive after work already enqueued on
    the issuing stream; callers that issue receives from another stream must
    provide equivalent ordering before reusing a slot.
    """

    def __init__(self, direction: str) -> None:
        self._direction = direction
        self._buffers: tuple[tuple[torch.Tensor | None, ...], ...] = ()
        self._allocation_signature: tuple[
            tuple[torch.Size, tuple[int, ...], torch.dtype] | None, ...
        ] = ()
        self._owners: dict[int, int] = {}

    def prepare(
        self,
        num_slots: int,
        recv_infos: tuple[_RecvInfo, ...],
        device: torch.device | str,
    ) -> None:
        """Grow the stable pool to ``num_slots`` for compatible metadata."""
        if self._owners:
            raise PipeliningMetadataError(
                f"The {self._direction} receive buffer pool still has storage "
                "owned by an incomplete pipeline step; destroy the process "
                "group, then reconstruct the stages and schedule before retrying"
            )

        metas: tuple[TensorMeta | None, ...] = tuple(
            None if info.is_root_arg else info.tensor_meta for info in recv_infos
        )
        allocation_signature = tuple(
            None if meta is None else (meta.shape, meta.stride, meta.dtype)
            for meta in metas
        )
        if self._allocation_signature != allocation_signature:
            self._allocation_signature = allocation_signature
            self._buffers = ()
        if len(self._buffers) >= num_slots:
            return
        with torch.inference_mode(False):
            self._buffers += tuple(
                tuple(
                    _make_tensor_from_meta(meta, device) if meta is not None else None
                    for meta in metas
                )
                for _ in range(num_slots - len(self._buffers))
            )

    @property
    def is_idle(self) -> bool:
        """Return whether no microbatch owns a slot."""
        return not self._owners

    @property
    def num_slots(self) -> int:
        """Return the number of allocated slots."""
        return len(self._buffers)

    def reset(self) -> None:
        """Release an idle pool's retained allocation and metadata."""
        if self._owners:
            raise PipeliningMetadataError(
                f"The {self._direction} receive buffer pool still has storage "
                "owned by an incomplete pipeline step; destroy the process "
                "group, then reconstruct the stages and schedule before retrying"
            )
        self._buffers = ()
        self._allocation_signature = ()

    def acquire(
        self,
        slot: int,
        microbatch_index: int,
        recv_infos: tuple[_RecvInfo, ...],
    ) -> None:
        """Assign fresh tensor aliases from one exclusively owned pool slot."""
        if not 0 <= slot < len(self._buffers):
            raise PipeliningMetadataError(
                f"{self._direction} receive buffer slot {slot} is out of range"
            )
        if slot in self._owners:
            raise PipeliningMetadataError(
                f"{self._direction} receive buffer slot {slot} is still owned by "
                f"microbatch {self._owners[slot]}"
            )

        buffers = self._buffers[slot]
        if len(buffers) != len(recv_infos):
            raise PipeliningMetadataError(
                f"{self._direction} receive buffer slot {slot} has "
                f"{len(buffers)} tensors, expected {len(recv_infos)}"
            )
        for info in recv_infos:
            if info.buffer is not None:
                raise PipeliningMetadataError(
                    _INCOMPLETE_RECV_BUFFER_ERROR.format(input_name=info.input_name)
                )
        for info, buffer in zip(recv_infos, buffers, strict=True):
            if buffer is not None:
                info.set_buffer(buffer.detach())
        self._owners[slot] = microbatch_index

    def release(self, slot: int, microbatch_index: int) -> None:
        """Release a slot after its schedule-derived lifetime completes."""
        owner = self._owners.get(slot)
        if owner is None:
            raise PipeliningMetadataError(
                f"{self._direction} receive buffer slot {slot} is not active"
            )
        if owner != microbatch_index:
            raise PipeliningMetadataError(
                f"{self._direction} receive buffer slot {slot} is owned by "
                f"microbatch {owner}, not {microbatch_index}"
            )

        del self._owners[slot]

    def aliases(self, tensor: torch.Tensor) -> bool:
        """Return whether ``tensor`` shares storage with any pooled buffer."""
        return any(
            buffer is not None and torch._C._is_alias_of(tensor, buffer)
            for slot_buffers in self._buffers
            for buffer in slot_buffers
        )

    def release_all(self) -> None:
        """Release every active slot after outstanding communication completes."""
        for slot, microbatch_index in tuple(self._owners.items()):
            self.release(slot, microbatch_index)
