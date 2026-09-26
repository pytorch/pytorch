from __future__ import annotations

import functools
import sys
import warnings
from collections.abc import Sequence
from typing import Any
from typing_extensions import deprecated

import torch
from torch.cuda._utils import (
    _check_cuda_bindings,
    _cuda_bindings_driver as _drv,
    _cuda_bindings_runtime as _rt,
    _ensure_cuda_bindings_version,
    _HAS_CUDA_BINDINGS,
)


__all__ = [
    "GreenContext",
    "SMPartition",
]

_STREAMS_PER_GREEN_CONTEXT_POOL = 32

_WORKQUEUE_SCOPE_VALUES = {
    "device_ctx": 0,
    "balanced": 1,
}

_CONTEXT_STACK_DEPRECATION = (
    "`GreenContext.set_context` and `GreenContext.pop_context` are deprecated. "
    "Please create a stream with `GreenContext.Stream()` and use "
    "`torch.cuda.stream(stream)` instead."
)


# note: this can safely be cached in a process/thread because
# the driver version cannot change during the lifetime of a process
@functools.cache
def _get_driver_version() -> int:
    # pyrefly: ignore [missing-attribute]
    return _check_cuda_bindings(_drv.cuDriverGetVersion())


def _ensure_cuda_version(version: int, message: str) -> None:
    if _get_driver_version() < version:
        raise RuntimeError(message)
    _ensure_cuda_bindings_version(version, message)


def _ensure_supported() -> None:
    if torch.version.cuda is None or torch.version.hip is not None:
        raise RuntimeError("Green Context is only supported on Nvidia CUDA")
    if sys.platform == "win32":
        raise RuntimeError("Green Context is not supported on Windows")
    if not _HAS_CUDA_BINDINGS:
        raise RuntimeError("GreenContext requires the cuda.bindings package")
    _ensure_cuda_version(
        12080, "Green Context requires CUDA driver and cuda.bindings 12.8+"
    )


def _ensure_workqueue_supported() -> None:
    _ensure_cuda_version(
        13010, "Workqueue configuration requires CUDA driver and cuda.bindings 13.1+"
    )


def _parse_workqueue_scope(workqueue_scope: str | None) -> int | None:
    if workqueue_scope is None:
        return None
    if workqueue_scope not in _WORKQUEUE_SCOPE_VALUES:
        raise ValueError(
            "workqueue_scope must be 'device_ctx' or 'balanced', "
            f"got '{workqueue_scope}'"
        )
    return _WORKQUEUE_SCOPE_VALUES[workqueue_scope]


class SMPartition:
    r"""An SM resource selected by CUDA, with its device and allocation metadata.

    Obtain resources with :meth:`from_device`, :meth:`split`, or
    :attr:`GreenContext.sm_partition`. Construct a :class:`GreenContext` with
    ``sm_partition=partition`` to run work on the selected SMs.

    A partition describes a set of SMs; it does not reserve them against other
    contexts. Reusing a partition for multiple contexts shares those SMs.
    """

    def __init__(
        self,
        _resource: Any,
        _device_id: int,
        _owner: SMPartition | GreenContext | None = None,
    ) -> None:
        self._resource = _resource
        self._device_id = _device_id
        self._owner = _owner

    @classmethod
    def from_device(cls, device_id: int | None = None) -> SMPartition:
        r"""Return the full device SM resource.

        Initializes the CUDA driver. If ``device_id`` is omitted, uses the
        current PyTorch device, initializing PyTorch CUDA state if necessary.
        """
        _ensure_supported()
        if device_id is None:
            device_id = torch.cuda.current_device()
        _check_cuda_bindings(_drv.cuInit(0))  # pyrefly: ignore [missing-attribute]
        # pyrefly: ignore [missing-attribute]
        device = _check_cuda_bindings(_drv.cuDeviceGet(device_id))
        resource = _check_cuda_bindings(
            _drv.cuDeviceGetDevResource(  # pyrefly: ignore [missing-attribute]
                device,
                _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,  # pyrefly: ignore [missing-attribute]
            )
        )
        return cls(resource, device_id)

    @property
    def device_id(self) -> int:
        r"""The device index of this SM resource."""
        return self._device_id

    @property
    def sm_count(self) -> int:
        r"""The actual number of SMs in this resource."""
        return self._resource.sm.smCount

    @property
    def coscheduled_sm_count(self) -> int:
        r"""The co-scheduled SM alignment reported by CUDA for this resource."""
        return self._resource.sm.smCoscheduledAlignment

    def split(
        self,
        *,
        num_sms: int | Sequence[int] = 0,
        coscheduled_sm_count: int | Sequence[int] = 0,
        preferred_coscheduled_sm_count: int | Sequence[int] = 0,
        backfill: bool | Sequence[bool] = False,
    ) -> tuple[tuple[SMPartition, ...], SMPartition | None]:
        r"""Split this resource into disjoint groups and an optional remainder.

        Requires CUDA driver and bindings 13.1+. CUDA checks the requested
        counts and hardware constraints; counts are not automatically rounded.
        A count of zero requests discovery of the largest remaining group
        satisfying its constraints. Groups are processed in order.

        Args:
            num_sms (int or sequence of int, optional): SM count for each group.
                Zero requests discovery. Default: ``0``.
            coscheduled_sm_count (int or sequence of int, optional): Co-scheduled
                SM grouping size for thread-block clusters. Zero lets CUDA
                determine cluster capabilities from the selected resources.
                Default: ``0``.
            preferred_coscheduled_sm_count (int or sequence of int, optional):
                Preferred larger grouping size, when CUDA can combine groups.
                Zero selects the CUDA default. Default: ``0``.
            backfill (bool or sequence of bool, optional): Allow CUDA to fill
                groups with SMs outside complete co-scheduled groupings.
                Default: ``False``.

        Each option can be a scalar or a sequence. All sequences must have the
        same nonzero length; scalars are broadcast to that length. If every
        option is scalar, the split has one group. An early discovery group can
        exhaust the SMs needed by later groups.
        A group with both ``num_sms=0`` and ``backfill=True`` consumes all
        remaining SMs and must be the last group.
        Returns ``(partitions, remainder)``, with ``None`` for an empty remainder.
        The remainder does not inherit the requested alignment.

        To subdivide a returned partition or remainder, create a
        :class:`GreenContext` from it and split the context's queried
        :attr:`~GreenContext.sm_partition`. CUDA drivers can reject raw split
        outputs as already partitioned resources. Context creation is explicit.

        Children are subsets of this resource and overlap it. Siblings from
        this operation, including the remainder, are disjoint. Results from
        separate splits on the same or overlapping input resources may overlap.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> sms = SMPartition.from_device(device_id=0)
            >>> (first,), rest = sms.split(num_sms=4, coscheduled_sm_count=2)
            >>> rest_ctx = GreenContext(sm_partition=rest)
            >>> (second,), rest = rest_ctx.sm_partition.split(
            ...     num_sms=4, coscheduled_sm_count=2
            ... )
        """
        message = "SM splitting requires CUDA driver and cuda.bindings 13.1+"
        _ensure_cuda_version(13010, message)
        options = {
            "num_sms": num_sms,
            "coscheduled_sm_count": coscheduled_sm_count,
            "preferred_coscheduled_sm_count": preferred_coscheduled_sm_count,
            "backfill": backfill,
        }
        sequences = {
            name: tuple(value)
            for name, value in options.items()
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        }
        lengths = {len(value) for value in sequences.values()}
        if len(lengths) > 1:
            raise ValueError("All sequence options must have the same length")
        count = lengths.pop() if lengths else 1
        if count == 0:
            raise ValueError("A split must contain at least one group")
        values = {
            name: sequences[name] if name in sequences else (value,) * count
            for name, value in options.items()
        }
        counts = values["num_sms"]
        co_counts = values["coscheduled_sm_count"]
        preferred = values["preferred_coscheduled_sm_count"]
        backfills = values["backfill"]
        for name, values in (
            ("num_sms", counts),
            ("coscheduled_sm_count", co_counts),
            ("preferred_coscheduled_sm_count", preferred),
        ):
            for value in values:
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    msg = f"{name} entries must be nonnegative integers, got {value!r}"
                    raise ValueError(msg)
        if any(not isinstance(value, bool) for value in backfills):
            raise ValueError("backfill entries must be bool values")
        params = []
        for index, count in enumerate(counts):
            if count == 0 and backfills[index] and index < len(counts) - 1:
                raise ValueError(
                    f"Split group {index} has num_sms=0 and backfill=True, which "
                    "consumes all remaining SMs. Only the last group may use this "
                    "combination; move it last or specify a positive num_sms."
                )
            # pyrefly: ignore [missing-attribute]
            param = _drv.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
            param.smCount = count
            param.coscheduledSmCount = co_counts[index]
            param.preferredCoscheduledSmCount = preferred[index]
            if backfills[index]:
                param.flags = (
                    # pyrefly: ignore [missing-attribute]
                    _drv.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_BACKFILL
                )
            params.append(param)
        resources, remainder = _check_cuda_bindings(
            # pyrefly: ignore [missing-attribute]
            _drv.cuDevSmResourceSplit(len(params), self._resource, 0, params)
        )
        children = tuple(SMPartition(r, self.device_id, self) for r in resources)
        remaining = None
        is_sm_resource = (
            # pyrefly: ignore [missing-attribute]
            remainder.type == _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        )
        if is_sm_resource and remainder.sm.smCount:
            remaining = SMPartition(remainder, self.device_id, self)
        return children, remaining


class GreenContext:
    r"""Wrapper around a CUDA green context.

    .. warning::
       This API is in beta and may change in future releases.

    CUDA work should be placed on streams created from the green context:

    .. code-block:: python

        ctx = GreenContext(...)
        stream = ctx.Stream()
        with torch.cuda.stream(stream):
            # torch operations here are using resources from `ctx`
            pass

    Green-context streams are custom CUDA streams. Synchronization with other
    streams is the user's responsibility and should be handled with CUDA events,
    as with any other custom stream.
    """

    def __init__(
        self,
        *,
        num_sms: int | None = None,
        sm_partition: SMPartition | None = None,
        workqueue_scope: str | None = None,
        workqueue_concurrency_limit: int | None = None,
        device_id: int | None = None,
    ) -> None:
        r"""Create a CUDA green context.

        Specify ``num_sms``, ``sm_partition``, or ``workqueue_scope``.
        ``num_sms`` and ``sm_partition`` are mutually exclusive. Either can be
        combined with workqueue configuration.

        Arguments:
            num_sms (int, optional): The number of SMs to use in the green
                context. When ``None``, uses ``sm_partition`` if provided,
                otherwise all SMs.
            sm_partition (SMPartition, optional): An existing SM resource to use.
                The context is created on the partition's device. Reusing a
                partition creates contexts sharing the same SMs.
            workqueue_scope (str, optional): Workqueue sharing scope. One of
                ``"device_ctx"`` (shared across all contexts, default driver
                behavior) or ``"balanced"`` (non-overlapping workqueues with
                other balanced green contexts). When ``None``, no workqueue
                configuration is applied.
            workqueue_concurrency_limit (int, optional): Maximum number of
                concurrent stream-ordered workloads for the workqueue. Requires
                ``workqueue_scope`` to be set.
            device_id (int, optional): The device index of green context.
                When ``None``, the current device is used.
        """
        self._device_id = None
        self._green_ctx = None
        self._context = None
        _ensure_supported()

        scope_value = _parse_workqueue_scope(workqueue_scope)
        if scope_value is not None:
            _ensure_workqueue_supported()

        if sm_partition is not None and not isinstance(sm_partition, SMPartition):
            raise TypeError("sm_partition must be an SMPartition")
        if num_sms is not None and sm_partition is not None:
            raise ValueError("num_sms and sm_partition are mutually exclusive")
        if num_sms is None and sm_partition is None and scope_value is None:
            raise RuntimeError(
                "At least one of num_sms, sm_partition, or workqueue_scope must be specified"
            )
        if workqueue_concurrency_limit is not None and scope_value is None:
            raise RuntimeError(
                "workqueue_concurrency_limit requires workqueue_scope to be set"
            )

        if sm_partition is not None:
            if device_id is not None and device_id != sm_partition.device_id:
                raise ValueError("device_id must match the SM partition's device")
            device_id = sm_partition.device_id
        elif device_id is None:
            device_id = torch.cuda.current_device()

        # pyrefly: ignore [missing-attribute]
        current_ctx = _check_cuda_bindings(_drv.cuCtxGetCurrent())

        if int(current_ctx) == 0:
            warnings.warn(
                "Attempted to create a green context but there was no primary "
                "context! Creating a primary context...",
                stacklevel=2,
            )
            # pyrefly: ignore [missing-attribute]
            _check_cuda_bindings(_rt.cudaFree(0))

        # pyrefly: ignore [missing-attribute]
        device = _check_cuda_bindings(_drv.cuDeviceGet(device_id))
        resources = []
        if sm_partition is not None:
            resources.append(sm_partition._resource)

        if num_sms is not None:
            sm_resource = _check_cuda_bindings(
                _drv.cuDeviceGetDevResource(  # pyrefly: ignore [missing-attribute]
                    device,
                    _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,  # pyrefly: ignore [missing-attribute]
                )
            )
            if num_sms <= 0 or num_sms > sm_resource.sm.smCount:
                raise RuntimeError(
                    "Invalid number of SMs requested for green context: "
                    f"{num_sms} (device has {sm_resource.sm.smCount} SMs)"
                )
            split_result, nb_groups, _remaining = _check_cuda_bindings(
                _drv.cuDevSmResourceSplitByCount(  # pyrefly: ignore [missing-attribute]
                    1, sm_resource, 0, num_sms
                )
            )
            if nb_groups != 1:
                raise RuntimeError("Failed to create single SM resource group")
            resources.append(split_result[0])

        if scope_value is not None:
            wq_resource = _check_cuda_bindings(
                _drv.cuDeviceGetDevResource(  # pyrefly: ignore [missing-attribute]
                    device,
                    _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG,  # pyrefly: ignore [missing-attribute]
                )
            )
            wq_resource.wqConfig.sharingScope = scope_value
            if workqueue_concurrency_limit is not None:
                wq_resource.wqConfig.wqConcurrencyLimit = workqueue_concurrency_limit
            resources.append(wq_resource)

        desc = _check_cuda_bindings(
            _drv.cuDevResourceGenerateDesc(  # pyrefly: ignore [missing-attribute]
                resources, len(resources)
            )
        )
        green_ctx = _check_cuda_bindings(
            _drv.cuGreenCtxCreate(  # pyrefly: ignore [missing-attribute]
                desc,
                device,
                _drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,  # pyrefly: ignore [missing-attribute]
            )
        )
        try:
            # pyrefly: ignore [missing-attribute]
            context = _check_cuda_bindings(_drv.cuCtxFromGreenCtx(green_ctx))
            if int(context) == 0:
                raise RuntimeError("Green ctx conversion to regular ctx failed!")
            self._init_from_cuda_objects(device_id, green_ctx, context)
        except Exception:
            # pyrefly: ignore [missing-attribute]
            _check_cuda_bindings(_drv.cuGreenCtxDestroy(green_ctx))
            raise

    def __del__(self) -> None:
        green_ctx = getattr(self, "_green_ctx", None)
        if green_ctx is None:
            return

        # attempt to destroy streams related to this green context
        # we ignore errors to avoid leaking exceptions during __del__
        end = min(_STREAMS_PER_GREEN_CONTEXT_POOL, self._curr_stream_idx + 1)
        for idx in reversed(range(end)):
            try:
                # pyrefly: ignore [missing-attribute]
                _check_cuda_bindings(_drv.cuStreamDestroy(self._green_ctx_streams[idx]))
            except RuntimeError as e:
                warnings.warn(
                    f"Error while destroying green context stream at idx {idx} "
                    f"for green context {green_ctx}: {e}"
                )
        self._green_ctx = None
        try:
            # pyrefly: ignore [missing-attribute]
            _check_cuda_bindings(_drv.cuGreenCtxDestroy(green_ctx))
        except RuntimeError as e:
            warnings.warn(f"Error while destroying green context {green_ctx}: {e}")

    def _init_from_cuda_objects(self, device_id: int, green_ctx, context) -> None:
        self._device_id = device_id
        self._context = context
        self._sm_count: int | None = None
        self._parent_stream: torch.cuda.Stream | None = None
        self._green_ctx_streams: list[Any | None] = [
            None
        ] * _STREAMS_PER_GREEN_CONTEXT_POOL
        self._curr_stream_idx = -1
        self._green_ctx = green_ctx

    @staticmethod
    def split(
        *,
        num_sms: int | Sequence[int] = 0,
        coscheduled_sm_count: int | Sequence[int] = 0,
        preferred_coscheduled_sm_count: int | Sequence[int] = 0,
        backfill: bool | Sequence[bool] = False,
        workqueue_scope: str | None = None,
        workqueue_concurrency_limit: int | None = None,
        device_id: int | None = None,
    ) -> tuple[GreenContext, ...]:
        r"""Create contexts backed by disjoint SM partitions of a device.

        Partition options are those of :meth:`SMPartition.split`. Workqueue
        options are applied to each context. Unassigned SMs are unused; use
        :meth:`SMPartition.split` to retain the remainder for later use.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> a, b = GreenContext.split(
            ...     num_sms=(24, 40), coscheduled_sm_count=8, device_id=0
            ... )
        """
        source = SMPartition.from_device(device_id)
        partitions, _ = source.split(
            num_sms=num_sms,
            coscheduled_sm_count=coscheduled_sm_count,
            preferred_coscheduled_sm_count=preferred_coscheduled_sm_count,
            backfill=backfill,
        )
        return tuple(
            GreenContext(
                sm_partition=partition,
                workqueue_scope=workqueue_scope,
                workqueue_concurrency_limit=workqueue_concurrency_limit,
            )
            for partition in partitions
        )

    @property
    def device_id(self) -> int:
        r"""The device index of this green context."""
        self._ensure_alive()
        if self._device_id is None:
            raise RuntimeError("GreenContext has been destroyed")
        return self._device_id

    @property
    def sm_partition(self) -> SMPartition:
        r"""The context's actual SM resource, which can be subdivided.

        The returned resource keeps this context alive while it is in use.
        """
        self._ensure_alive()
        resource = _check_cuda_bindings(
            _drv.cuGreenCtxGetDevResource(  # pyrefly: ignore [missing-attribute]
                self._green_ctx,
                _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,  # pyrefly: ignore [missing-attribute]
            )
        )
        return SMPartition(resource, self.device_id, self)

    @property
    def sm_count(self) -> int:
        r"""The actual number of SMs available to this context."""
        self._ensure_alive()
        if self._sm_count is None:
            self._sm_count = self.sm_partition.sm_count
        return self._sm_count

    @staticmethod
    def create(
        *,
        num_sms: int | None = None,
        sm_partition: SMPartition | None = None,
        workqueue_scope: str | None = None,
        workqueue_concurrency_limit: int | None = None,
        device_id: int | None = None,
    ) -> GreenContext:
        r"""Create a CUDA green context.

        Kept for compatibility, see `GreenContext` constructor.
        """
        return GreenContext(
            num_sms=num_sms,
            sm_partition=sm_partition,
            workqueue_scope=workqueue_scope,
            workqueue_concurrency_limit=workqueue_concurrency_limit,
            device_id=device_id,
        )

    @staticmethod
    def max_workqueue_concurrency(device_id: int | None = None) -> int:
        r"""Return the maximum workqueue concurrency limit for the device.

        This queries the device for the default number of concurrent
        stream-ordered workloads supported by workqueue configuration
        resources.

        Arguments:
            device_id (int, optional): The device index to query. When
                ``None``, the current device is used.
        """
        _ensure_supported()
        _ensure_workqueue_supported()
        if device_id is None:
            device_id = torch.cuda.current_device()

        # pyrefly: ignore [missing-attribute]
        device = _check_cuda_bindings(_drv.cuDeviceGet(device_id))
        wq_resource = _check_cuda_bindings(
            _drv.cuDeviceGetDevResource(  # pyrefly: ignore [missing-attribute]
                device,
                _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG,  # pyrefly: ignore [missing-attribute]
            )
        )
        return wq_resource.wqConfig.wqConcurrencyLimit

    def _ensure_alive(self) -> None:
        if self._green_ctx is None or self._context is None:
            raise RuntimeError("GreenContext has been destroyed")

    @deprecated(_CONTEXT_STACK_DEPRECATION, category=FutureWarning)
    def set_context(self) -> None:
        r"""Make the green context the current context.

        Deprecated. Create streams with :meth:`Stream` and use
        :func:`torch.cuda.stream` instead.
        """
        self._ensure_alive()
        if self._parent_stream is not None:
            raise RuntimeError("set_context called twice before pop_context")
        current_stream = torch.cuda.current_stream()
        self._parent_stream = current_stream

        event = torch.cuda.Event()
        event.record(current_stream)

        # pyrefly: ignore [missing-attribute]
        current_ctx = _check_cuda_bindings(_drv.cuCtxGetCurrent())
        if int(current_ctx) == 0:
            # pyrefly: ignore [missing-attribute]
            _check_cuda_bindings(_drv.cuCtxSetCurrent(self._context))
        else:
            # pyrefly: ignore [missing-attribute]
            _check_cuda_bindings(_drv.cuCtxPushCurrent(self._context))

        green_ctx_stream = torch.cuda.default_stream(self._device_id)
        event.wait(green_ctx_stream)
        torch.cuda.set_stream(green_ctx_stream)

    @deprecated(_CONTEXT_STACK_DEPRECATION, category=FutureWarning)
    def pop_context(self) -> None:
        r"""Assuming the green context is the current context, pop it from the
        context stack and restore the previous context.

        Deprecated. Create streams with :meth:`Stream` and use
        :func:`torch.cuda.stream` instead.
        """
        try:
            self._ensure_alive()
            if self._parent_stream is None:
                raise RuntimeError("pop_context called without matching set_context")

            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())

            # pyrefly: ignore [missing-attribute]
            popped = _check_cuda_bindings(_drv.cuCtxPopCurrent())
            # pyrefly: ignore [bad-argument-type]
            if int(popped) != int(self._context):
                raise RuntimeError("expected popped context to be the current ctx")

            event.wait(self._parent_stream)
            torch.cuda.set_stream(self._parent_stream)
        finally:
            self._parent_stream = None

    def Stream(self) -> torch.cuda.Stream:
        r"""Return a CUDA stream associated with this green context.

        Use the returned stream with :func:`torch.cuda.stream` to run work on
        the green context. Synchronization with other streams is not automatic;
        use CUDA events as with any other custom stream.
        """
        self._ensure_alive()
        curr_idx = self._curr_stream_idx + 1
        idx = curr_idx % _STREAMS_PER_GREEN_CONTEXT_POOL
        if curr_idx < _STREAMS_PER_GREEN_CONTEXT_POOL:
            green_ctx_stream = _check_cuda_bindings(
                _drv.cuGreenCtxStreamCreate(  # pyrefly: ignore [missing-attribute]
                    self._green_ctx,
                    _drv.CUstream_flags.CU_STREAM_NON_BLOCKING,  # pyrefly: ignore [missing-attribute]
                    0,
                )
            )
            self._green_ctx_streams[idx] = green_ctx_stream
        else:
            green_ctx_stream = self._green_ctx_streams[idx]
        self._curr_stream_idx = curr_idx
        # pyrefly: ignore [bad-argument-type]
        return torch.cuda.ExternalStream(int(green_ctx_stream), self._device_id)
