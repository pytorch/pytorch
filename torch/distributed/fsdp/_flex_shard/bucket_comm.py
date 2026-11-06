# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, ClassVar, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import _get_device_handle

from .placement_contract import PlacementPreparedUnshard, PlacementUnshardResult


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.distributed.device_mesh import DeviceMesh

    from .bucket_storage import ParamInfo
    from .placement_contract import GradientReduction, Placement


class UnshardHandle:
    """Handle for a FlexShard bucket unshard operation."""

    def finish(self) -> PlacementUnshardResult:
        """Wait for the unshard and return its full params (once)."""
        raise NotImplementedError

    def wait(self) -> None:
        """Wait until the unshard result is usable on the current stream."""
        raise NotImplementedError

    def release_buffers(self) -> None:
        """Release buffers owned by an unconsumed unshard operation."""
        raise NotImplementedError


class ReduceGradHandle:
    """Handle for a FlexShard bucket gradient reduction operation."""

    # Where the reduced grads are ready (None: the reduce-grad stream), and
    # whether the reduction only accumulated a partial output, with no grads.
    post_stream: torch.Stream | None = None
    partial: bool = False
    # The streams that allocated the reduction's buffers, which wait for it
    # before their memory is released
    allocation_streams: tuple[torch.Stream, ...]

    def finish(self) -> list[torch.Tensor]:
        """Wait for the reduction and return local gradient shards."""
        raise NotImplementedError

    def wait(self) -> None:
        """Wait until the reduce-grad result is usable on the current stream."""
        raise NotImplementedError

    def synchronize(self) -> None:
        """Block the host until the reduce-grad result is complete."""
        raise NotImplementedError

    def release_buffers(self, release_sharded_grads: bool) -> None:
        """Release temporary buffers owned by the reduce-grad operation."""
        raise NotImplementedError

    def record_sharded_grads(
        self,
        sharded_grads: list[torch.Tensor],
        stream: torch.Stream,
    ) -> None:
        """Track sharded grads until queued work on stream is complete."""
        raise NotImplementedError


def _stream_device(stream: torch.Stream) -> torch.device | None:
    # The device to wait on: an offloaded bucket's buffers can all be on CPU.
    # torch.cpu's Stream stand-in has no device.
    return stream.device if isinstance(stream, torch.Stream) else None


def _first_tensor_device(*tensor_groups: list[torch.Tensor]) -> torch.device | None:
    """The device of the first tensor, preferring an accelerator one: an
    offloaded bucket's buffers mix CPU and accelerator tensors."""
    devices = [tensor.device for tensors in tensor_groups for tensor in tensors]
    return next((d for d in devices if d.type != "cpu"), devices[0] if devices else None)


def _get_bucket_placement(
    infos: list[ParamInfo],
    operation: str,
) -> Placement:
    placement = infos[0].placement
    compatibility_key = placement.bucket_compatibility_key()
    for info in infos[1:]:
        if info.placement.bucket_compatibility_key() != compatibility_key:
            raise ValueError(
                f"FlexShard bucket {operation} requires all parameters in a "
                "bucket to use compatible placements, but "
                f"{infos[0].fqn!r} uses {placement!r} and {info.fqn!r} uses "
                f"{info.placement!r}."
            )
    return placement


def begin_bucket_unshard(
    tensors: list[torch.Tensor],
    infos: list[ParamInfo],
    mesh: DeviceMesh,
    unshard_stream: torch.Stream,
    debug_fqn: str | None = None,
    persistent_buffers: list[torch.Tensor] | None = None,
    **prepare_kwargs: Any,
) -> UnshardHandle:
    """Begin a bucket unshard and return a handle for full params.

    Eager unshards are persistent (see ``PlacementPreparedUnshard``); during
    graph capture the traced graph owns buffer lifetimes. Given the
    ``persistent_buffers`` of an earlier unshard, the unshard refills them, in
    run or in finish as the placement chooses. Their storage is re-allocated on
    the current stream already, or, if the placement refills them in finish,
    is re-allocated before finish.
    """
    placement = _get_bucket_placement(infos, "unshard")
    prepare_kwargs = {k: v for k, v in prepare_kwargs.items() if v is not None}

    if torch.compiler.is_compiling():
        prepared = placement.prepare_unshard_bucket(
            tensors, infos, mesh, debug_fqn, **prepare_kwargs
        )
        result = _run_and_finish_unshard(prepared)
        return SyncUnshardResult(result.full_params)

    device = prepare_kwargs.get("device", tensors[0].device)
    device_handle = _get_device_handle(device.type)
    copy_in_done = device_handle.Event()
    copy_in_done.record(device_handle.current_stream(device))
    with device_handle.stream(unshard_stream):
        unshard_stream.wait_event(copy_in_done)
        prepared = placement.prepare_unshard_bucket(
            tensors, infos, mesh, debug_fqn, **prepare_kwargs
        )
        prepared.persistent = True
        prepared.persistent_buffers = persistent_buffers
        if persistent_buffers is None:
            prepared.placement.run_prepared_unshard(prepared)
        else:
            # Run may write the persistent buffers, which the unsharded params
            # view; autograd may have saved those before a reshard.
            with (
                torch.inference_mode(False),
                torch.no_grad(),
                torch.autograd._unsafe_preserve_version_counter(
                    tuple(persistent_buffers)
                ),
            ):
                prepared.placement.run_prepared_unshard(prepared)
        event = device_handle.Event()
        event.record(unshard_stream)
    return AsyncUnshardResult(
        prepared=prepared,
        event=event,
        unshard_stream=unshard_stream,
        device_handle=device_handle,
    )


@dataclass
class PostReduceScatter:
    """The steps between a bucket's reduce-scatter and its grad accumulation,
    as FSDP2 runs them: the all-reduce over the replicas of the mesh (HSDP,
    ``replicate()``) and the user's all-reduce hook, on ``stream`` so that the
    next bucket's reduce-scatter does not wait for them. Without
    ``requires_all_reduce``, the reduce-scatter output accumulates in
    ``partial_output`` instead and the all-reduce of a later backward adds it.
    """

    stream: torch.Stream
    replicate_group: dist.ProcessGroup | None = None
    requires_all_reduce: bool = True
    partial_output: torch.Tensor | None = None
    hook: Callable[[torch.Tensor], None] | None = None


def begin_reduce_grad(
    tensors: list[torch.Tensor],
    infos: list[ParamInfo],
    mesh: DeviceMesh,
    reduction: GradientReduction,
    reduce_grad_stream: torch.Stream,
    debug_fqn: str | None = None,
    post_reduce_scatter: PostReduceScatter | None = None,
    **prepare_kwargs: Any,
) -> ReduceGradHandle:
    """Begin a bucket reduce-grad and return a handle for local grad shards.

    The handle's ``post_stream`` is where the reduced grads are ready; it holds
    no grads if the reduce-scatter output only accumulated into
    ``post_reduce_scatter.partial_output``."""
    placement = _get_bucket_placement(infos, "reduce-grad")
    prepare_kwargs = {k: v for k, v in prepare_kwargs.items() if v is not None}
    prepared = placement.prepare_reduce_grad(
        tensors, infos, mesh, debug_fqn, **prepare_kwargs
    )
    if torch.compiler.is_compiling():
        if post_reduce_scatter is not None:
            raise NotImplementedError(
                "FlexShard: torch.compile does not support the HSDP all-reduce or "
                "an all-reduce hook yet."
            )
        result = prepared.placement.reduce_prepared_grad(prepared, reduction)
        return SyncReduceGradResult(result.sharded_grads)

    device = prepared.buffers[0].device
    device_handle = _get_device_handle(device.type)
    copy_in_stream = device_handle.current_stream(device)
    copy_in_done = device_handle.Event()
    copy_in_done.record(copy_in_stream)
    post = post_reduce_scatter
    all_reduce = post is not None and post.replicate_group is not None
    with device_handle.stream(reduce_grad_stream):
        reduce_grad_stream.wait_event(copy_in_done)
        result = prepared.placement.reduce_prepared_grad(prepared, reduction)
        output = result.flat_output
        if post is not None and output is None:
            raise NotImplementedError(
                f"FlexShard: {type(placement).__name__} does not support the HSDP "
                "all-reduce or an all-reduce hook."
            )
        partial = False
        if all_reduce and post is not None and output is not None:
            if not post.requires_all_reduce:
                if post.partial_output is None:
                    post.partial_output = output
                elif post.partial_output.dtype != output.dtype:
                    # A wider reduce output widens the pending one
                    post.partial_output = post.partial_output.to(output.dtype) + output
                else:
                    post.partial_output += output
                partial = True
            elif post.partial_output is not None:
                output += post.partial_output
                post.partial_output = None
        event = device_handle.Event()
        event.record(reduce_grad_stream)
    post_stream = reduce_grad_stream
    if post is not None and output is not None and not partial:
        from .placements.utils import _gradient_reduce_ops

        _, _, _, all_reduce_op, post_factor = _gradient_reduce_ops(
            reduction, mesh.size(), output.dtype
        )
        post_stream = post.stream
        post_stream.wait_event(event)
        with device_handle.stream(post_stream):
            if all_reduce:
                dist.all_reduce(output, group=post.replicate_group, op=all_reduce_op)
            if post.hook is not None:
                post.hook(output)
            if post_factor is not None:
                output.div_(post_factor)
            event = device_handle.Event()
            event.record(post_stream)
    # The reduce output is read on another stream only by the post-reduce
    # steps; otherwise the sharded grads, which alias or copy it on its own
    # stream, alone decide its lifetime, as in FSDP2 (e.g. freed once
    # offloaded to CPU).
    output_buffers = result.buffers if post_stream is not reduce_grad_stream else []
    return AsyncReduceGradResult(
        sharded_grads=[] if partial else result.sharded_grads,
        event=event,
        buffers=[*prepared.buffers, *output_buffers],
        allocation_streams=(copy_in_stream, reduce_grad_stream, post_stream),
        device_handle=device_handle,
        post_stream=post_stream,
        partial=partial,
    )


def begin_explicit_bucket_unshard(
    tensors: list[torch.Tensor],
    infos: list[ParamInfo],
    mesh: DeviceMesh,
    *,
    async_op: bool,
    debug_fqn: str | None = None,
    persistent_buffers: list[torch.Tensor] | None = None,
    **prepare_kwargs: Any,
) -> UnshardHandle:
    """Begin a bucket unshard for ``FlexShardModule.unshard()`` on the current
    stream, as FSDP2's ``FSDPModule.unshard()`` all-gathers on it
    (``FSDPCommContext.get_all_gather_streams``).

    The copy-in, the all-gather buffers and the copy-out all belong to the
    current stream, so its memory pool gets the buffers back after the
    copy-out. With ``async_op``, the collective is asynchronous, so the current
    stream runs other work until the handle is waited on or finished.
    Otherwise this returns after the collective is queued, as a synchronous
    collective does. See ``begin_bucket_unshard`` for ``persistent_buffers``.
    """
    placement = _get_bucket_placement(infos, "unshard")
    prepare_kwargs = {k: v for k, v in prepare_kwargs.items() if v is not None}
    prepared = placement.prepare_unshard_bucket(
        tensors, infos, mesh, debug_fqn, **prepare_kwargs
    )
    prepared.persistent = True
    prepared.persistent_buffers = persistent_buffers
    if persistent_buffers is None:
        work = prepared.placement.run_prepared_unshard(prepared, async_op=async_op)
    else:
        with (
            torch.inference_mode(False),
            torch.no_grad(),
            torch.autograd._unsafe_preserve_version_counter(tuple(persistent_buffers)),
        ):
            work = prepared.placement.run_prepared_unshard(prepared, async_op=async_op)
    return ExplicitUnshardResult(prepared=prepared, work=work)


@dataclass
class SyncUnshardResult(UnshardHandle):
    """Already-finished unshard result used during graph capture."""

    full_params: list[torch.Tensor]

    def finish(self) -> PlacementUnshardResult:
        return PlacementUnshardResult(self.full_params)

    def wait(self) -> None:
        return

    def release_buffers(self) -> None:
        return


@dataclass
class AsyncUnshardResult(UnshardHandle):
    """State needed to finish an async unshard launched on a side stream."""

    prepared: PlacementPreparedUnshard
    event: torch.Event
    unshard_stream: torch.Stream
    device_handle: ModuleType
    _device: torch.device | None = field(default=None, init=False)
    _finished: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._device = _stream_device(self.unshard_stream) or _first_tensor_device(
            self.prepared.buffers
        )

    def finish(self) -> PlacementUnshardResult:
        if self._finished:
            raise RuntimeError("An unshard may only be finished once.")
        self._finished = True
        self.wait()
        result = self.prepared.placement.finish_prepared_unshard(self.prepared)
        # Only work queued so far on the current stream (the copy-out) reads
        # the prepare and finish buffers.
        self._release([*self.prepared.buffers, *result.buffers])
        self.prepared.buffers.clear()
        result.buffers.clear()
        return result

    def wait(self) -> None:
        if self._device is not None:
            self.device_handle.current_stream(self._device).wait_event(self.event)

    def release_buffers(self) -> None:
        """Release an unconsumed unshard's buffers after current-stream work."""
        self._release(list(self.prepared.buffers))
        self.prepared.buffers.clear()

    def _release(self, tensors: list[torch.Tensor]) -> None:
        if not tensors:
            return
        current_stream = self.device_handle.current_stream(self._device)
        StreamHandoff(
            tensors,
            (self.unshard_stream, current_stream),
            self.device_handle,
        ).release_after_current_stream()


@dataclass
class PostForwardUnshardResult(UnshardHandle):
    """An unshard from the chunks of the unsharded parameters kept after forward
    (FSDP2's ``reshard_after_forward=int``): an all-gather over the smaller
    post-forward group, then a copy into the persistent buffers."""

    gathered: torch.Tensor
    chunk_nbytes: list[int]
    persistent_buffers: list[torch.Tensor]
    world_size: int
    event: torch.Event
    stream: torch.Stream
    device_handle: ModuleType

    def finish(self) -> PlacementUnshardResult:
        self.wait()
        keep = [i for i, nbytes in enumerate(self.chunk_nbytes) if nbytes]
        outputs = [self.persistent_buffers[i].view(-1).view(torch.uint8) for i in keep]
        torch.ops.fsdp._all_gather_copy_out_(
            outputs,
            self.gathered,
            [self.chunk_nbytes[i] for i in keep],
            [1] * len(keep),
            self.world_size,
        )
        self.release_buffers()
        return PlacementUnshardResult(
            full_params=[], persistent_buffers=self.persistent_buffers
        )

    def wait(self) -> None:
        self.device_handle.current_stream(self.gathered.device).wait_event(self.event)

    def release_buffers(self) -> None:
        current_stream = self.device_handle.current_stream(self.gathered.device)
        StreamHandoff(
            [self.gathered], (self.stream, current_stream), self.device_handle
        ).release_after_current_stream()


def begin_post_forward_unshard(
    post_forward_storage: torch.Tensor,
    chunk_nbytes: list[int],
    persistent_buffers: list[torch.Tensor],
    group: dist.ProcessGroup,
    stream: torch.Stream,
) -> PostForwardUnshardResult:
    """All-gather the chunks kept after forward over ``group`` on ``stream``."""
    device = post_forward_storage.device
    device_handle = _get_device_handle(device.type)
    ready = device_handle.Event()
    ready.record(device_handle.current_stream(device))
    world_size = group.size()
    with device_handle.stream(stream):
        stream.wait_event(ready)
        gathered = torch.empty(
            world_size * post_forward_storage.numel(), dtype=torch.uint8, device=device
        )
        dist.all_gather_into_tensor(gathered, post_forward_storage, group=group)
        event = device_handle.Event()
        event.record(stream)
    return PostForwardUnshardResult(
        gathered=gathered,
        chunk_nbytes=chunk_nbytes,
        persistent_buffers=persistent_buffers,
        world_size=world_size,
        event=event,
        stream=stream,
        device_handle=device_handle,
    )


@dataclass
class ExplicitUnshardResult(UnshardHandle):
    """An unshard begun on the current stream by ``FlexShardModule.unshard()``.

    ``work`` is the asynchronous collective's work, as in FSDP2's
    ``AllGatherResult``, or ``None`` if the collective was synchronous.
    """

    prepared: PlacementPreparedUnshard
    work: dist.Work | None
    _finished: bool = field(default=False, init=False)

    def finish(self) -> PlacementUnshardResult:
        if self._finished:
            raise RuntimeError("An unshard may only be finished once.")
        self._finished = True
        self.wait()
        result = self.prepared.placement.finish_prepared_unshard(self.prepared)
        # The current stream queued the all-gather and the copy-out, so its
        # pool can reuse the buffers once these references drop.
        self.prepared.buffers.clear()
        result.buffers.clear()
        return result

    def wait(self) -> None:
        if self.work is not None:
            self.work.wait()
            self.work = None

    def release_buffers(self) -> None:
        self.wait()
        self.prepared.buffers.clear()


@dataclass
class SyncReduceGradResult(ReduceGradHandle):
    """Already-finished reduce-grad result used during graph capture."""

    allocation_streams: ClassVar[tuple[torch.Stream, ...]] = ()
    sharded_grads: list[torch.Tensor]

    def finish(self) -> list[torch.Tensor]:
        return self.sharded_grads

    def wait(self) -> None:
        return

    def synchronize(self) -> None:
        return

    def release_buffers(self, release_sharded_grads: bool) -> None:
        if release_sharded_grads:
            self.sharded_grads.clear()

    def record_sharded_grads(
        self,
        sharded_grads: list[torch.Tensor],
        stream: torch.Stream,
    ) -> None:
        self.sharded_grads = sharded_grads


@dataclass
class AsyncReduceGradResult(ReduceGradHandle):
    """State needed to finish an async reduce-grad launched on a side stream."""

    sharded_grads: list[torch.Tensor]
    event: torch.Event | None
    buffers: list[torch.Tensor]
    allocation_streams: tuple[torch.Stream, ...]
    device_handle: ModuleType
    post_stream: torch.Stream | None = None
    partial: bool = False
    _device: torch.device | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._device = _stream_device(
            self.allocation_streams[0]
        ) or _first_tensor_device(self.sharded_grads, self.buffers)

    def finish(self) -> list[torch.Tensor]:
        self.wait()
        return self.sharded_grads

    def wait(self) -> None:
        if self._device is None:
            return
        if self.event is not None:
            self.device_handle.current_stream(self._device).wait_event(self.event)

    def synchronize(self) -> None:
        if self.event is not None:
            self.event.synchronize()

    def release_buffers(self, release_sharded_grads: bool) -> None:
        """Release pending reduce-grad buffers after its completion wait."""
        tensors = list(self.buffers)
        self.buffers.clear()
        if release_sharded_grads:
            tensors.extend(self.sharded_grads)
            self.sharded_grads.clear()
        if not tensors:
            return
        StreamHandoff(
            tensors,
            self.allocation_streams,
            self.device_handle,
        ).release_after_current_stream()

    def record_sharded_grads(
        self,
        sharded_grads: list[torch.Tensor],
        stream: torch.Stream,
    ) -> None:
        self.sharded_grads = sharded_grads
        self._device = (
            _stream_device(stream)
            or _first_tensor_device(self.sharded_grads, self.buffers)
            or self._device
        )
        self.event = self.device_handle.Event()
        self.event.record(stream)


def _run_and_finish_unshard(prepared: PlacementPreparedUnshard):
    prepared.placement.run_prepared_unshard(prepared)
    return prepared.placement.finish_prepared_unshard(prepared)


class StreamHandoff:
    """Keep tensors alive until their allocation streams own the final wait."""

    __slots__ = (
        "_tensors",
        "_allocation_streams",
        "_device_handle",
        "_released",
    )

    def __init__(
        self,
        tensors: list[torch.Tensor],
        allocation_streams: tuple[torch.Stream, ...],
        device_handle: ModuleType | None = None,
    ) -> None:
        if device_handle is None:
            device_handle = _get_device_handle(_first_tensor_device(tensors).type)
        self._tensors = tensors
        self._allocation_streams = allocation_streams
        self._device_handle = device_handle
        self._released = False

    def release_after(self, event: torch.Event) -> None:
        """Order every allocation stream after event, then drop keepalive refs."""
        if self._released:
            return
        seen_streams: set[int] = set()
        for stream in self._allocation_streams:
            stream_id = id(stream)
            if stream_id in seen_streams:
                continue
            seen_streams.add(stream_id)
            stream.wait_event(event)
        self._released = True
        self._tensors.clear()

    def release_after_current_stream(self) -> None:
        """Record the final consumer point and release after that event."""
        if self._released:
            return
        if not self._tensors:
            self._released = True
            return
        streams = self._allocation_streams
        device = (streams and _stream_device(streams[0])) or _first_tensor_device(
            self._tensors
        )
        event = self._device_handle.Event()
        event.record(self._device_handle.current_stream(device))
        self.release_after(event)
