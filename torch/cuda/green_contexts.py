from __future__ import annotations

import functools
import sys
import threading
import warnings
import weakref
from typing import Any, TYPE_CHECKING
from typing_extensions import deprecated

import torch
from torch._vendor.packaging.version import Version
from torch.cuda._utils import (
    _check_cuda_bindings,
    _cuda_bindings_driver as _drv,
    _cuda_bindings_runtime as _rt,
    _cuda_bindings_version,
    _get_device_index,
    _HAS_CUDA_BINDINGS,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.types import Device

__all__ = [
    "GreenContext",
    "execute_in_green_contexts",
    "get_green_context_from_stream",
    "get_num_locality_domains",
    "is_localization_supported",
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

_STREAM_TO_GREEN_CTX: weakref.WeakValueDictionary[int, GreenContext] = (
    weakref.WeakValueDictionary()
)
_STREAM_TO_GREEN_CTX_LOCK = threading.RLock()


# note: this can safely be cached in a process/thread because
# the driver version cannot change during the lifetime of a process
@functools.cache
def _get_driver_version() -> int:
    try:
        # pyrefly: ignore [missing-attribute]
        return _check_cuda_bindings(_drv.cuDriverGetVersion())
    except RuntimeError as e:
        warnings.warn(f"Error while querying CUDA driver version: {e}")
        return -1


def _ensure_driver_version(version: int, message: str) -> None:
    drv_version = _get_driver_version()
    if drv_version < 0 or drv_version < version:
        raise RuntimeError(message)
    try:
        # Prereleases compare as their target release, e.g. 13.4.0b1 as 13.4.0.
        vs = Version(str(_cuda_bindings_version)).release
        cb_version = vs[0] * 1000 + vs[1] * 10
        if len(vs) > 2:
            cb_version += vs[2]
    except Exception:
        raise RuntimeError(
            f"Invalid cuda.bindings version: '{_cuda_bindings_version}'"
        ) from None
    if cb_version < 0 or cb_version < version:
        raise RuntimeError(message)


def _ensure_supported() -> None:
    if not torch.backends.cuda.is_built() or torch.version.hip is not None:
        raise RuntimeError("Green Context is only supported on Nvidia CUDA")
    if sys.platform == "win32":
        raise RuntimeError("Green Context is not supported on Windows")
    if not _HAS_CUDA_BINDINGS:
        raise RuntimeError("GreenContext requires the cuda.bindings package")
    _ensure_driver_version(
        12080, "Green Context requires user mode driver and cuda.bindings package 12.8+"
    )


def _ensure_workqueue_supported() -> None:
    _ensure_driver_version(
        13010,
        "Green Context workqueue configuration requires user mode driver and "
        "cuda.bindings package 13.1+",
    )


def _ensure_localization_supported() -> None:
    _ensure_driver_version(
        13040,
        "Green Context localization requires user mode driver and "
        "cuda.bindings package 13.4+",
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


# note: the following functions can be cached as well as the return values
# cannot change during the lifetime of a process
@functools.cache
def _get_drv_device(device_id: int) -> Any:
    # pyrefly: ignore [missing-attribute]
    return _check_cuda_bindings(_drv.cuDeviceGet(device_id))


@functools.cache
def _get_num_locality_domains(device_id: int) -> int:
    _ensure_supported()
    _ensure_localization_supported()
    # pyrefly: ignore [missing-attribute]
    device_result = _drv.cuDeviceGet(device_id)
    # pyrefly: ignore [missing-attribute]
    if device_result[0] == _drv.CUresult.CUDA_ERROR_NOT_INITIALIZED:
        # note: all devices with SM 10.X have exactly 2 locality domains, and
        # any device with SM < 10.0 or SM == 12.X has 1 locality domain, so
        # we can safely provide the result without initializing the CUDA context
        # and poisoning the fork. In the future, this might need to be updated.
        capability = torch.cuda._raw_device_capability_nvml(device_id)
        return 2 if capability is not None and capability[0] == 10 else 1
    device = _check_cuda_bindings(device_result)
    return _check_cuda_bindings(
        # pyrefly: ignore [missing-attribute]
        _drv.cuDeviceGetAttribute(
            # pyrefly: ignore [missing-attribute]
            _drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT,
            device,
        )
    )


@functools.cache
def _get_dev_major(device_id: int) -> int:
    device = _get_drv_device(device_id)
    return _check_cuda_bindings(
        # pyrefly: ignore [missing-attribute]
        _drv.cuDeviceGetAttribute(
            # pyrefly: ignore [missing-attribute]
            _drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        )
    )


def _validate_coscheduled_sm_count(device_id: int, coscheduled_sm_count: int) -> None:
    # Non-zero values must be multiples of 2; the maximum is 32 on CC 9.0+
    # and 2 on earlier architectures.
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html
    max_coscheduled_count = 2 if _get_dev_major(device_id) < 9 else 32
    if (
        coscheduled_sm_count < 0
        or coscheduled_sm_count > max_coscheduled_count
        or coscheduled_sm_count % 2 != 0
    ):
        raise RuntimeError(
            "coscheduled_sm_count must be 0 or a multiple of 2 between 2 and "
            f"{max_coscheduled_count}, inclusive"
        )


@functools.cache
def _get_localized_sm_resources(
    device_id: int,
    locality_domain_backfill: bool = False,
    coscheduled_sm_count: int = 0,
) -> tuple[Any, ...]:
    _ensure_localization_supported()
    num_domains = get_num_locality_domains(device_id)
    if num_domains <= 1:
        raise RuntimeError(f"Localization is not supported on device {device_id}")

    device = _get_drv_device(device_id)
    sm_resource = _check_cuda_bindings(
        # pyrefly: ignore [missing-attribute]
        _drv.cuDeviceGetDevResource(
            device,
            # pyrefly: ignore [missing-attribute]
            _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
        )
    )
    group_flag = (
        # pyrefly: ignore [missing-attribute]
        _drv.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID
    )
    sm_count = 0  # discovery mode
    if locality_domain_backfill:
        # pyrefly: ignore [missing-attribute]
        group_flag |= _drv.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_BACKFILL
        total_sm_count = sm_resource.sm.smCount
        if total_sm_count % num_domains != 0:
            raise RuntimeError(
                "Locality-domain backfill requires the device SM resource "
                f"count ({total_sm_count}) to be evenly divisible by the "
                f"number of locality domains ({num_domains})"
            )
        sm_count = total_sm_count // num_domains
    params = []
    for domain_id in range(num_domains):
        # pyrefly: ignore [missing-attribute]
        param = _drv.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
        param.smCount = sm_count
        param.coscheduledSmCount = coscheduled_sm_count
        param.flags = group_flag
        param.localityDomainId = domain_id
        params.append(param)

    # Discovery mode may leave SMs outside the locality domains in the
    # remainder. Backfill requests equal groups whose total consumes the full
    # device SM resource, as enforced by the divisibility check above.
    split_result, _remaining = _check_cuda_bindings(
        # pyrefly: ignore [missing-attribute]
        _drv.cuDevSmResourceSplit(
            num_domains,
            sm_resource,
            0,
            params,
        )
    )
    return tuple(split_result)


def get_num_locality_domains(device: Device = None) -> int:
    r"""Return the number of CUDA locality domains for a device."""
    if torch.cuda.is_initialized():
        device_id = _get_device_index(device, optional=True)
    elif device is None:
        device_id = 0
    else:
        parsed_device = torch.device(device) if isinstance(device, str) else device
        if (
            isinstance(parsed_device, torch.device)
            and parsed_device.type == "cuda"
            and parsed_device.index is None
        ):
            device_id = 0
        else:
            device_id = _get_device_index(device)
    return _get_num_locality_domains(device_id)


def is_localization_supported(device: Device = None) -> bool:
    r"""Return whether CUDA green context localization is available."""
    try:
        return get_num_locality_domains(device) > 1
    except RuntimeError:
        return False


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

    # __del__ may run while Python is clearing module globals during interpreter
    # shutdown. Keep stable references on the class for cleanup.
    _check_cuda_bindings_ = _check_cuda_bindings
    _drv_ = _drv

    def __init__(
        self,
        *,
        num_sms: int | None = None,
        workqueue_scope: str | None = None,
        workqueue_concurrency_limit: int | None = None,
        locality_domain_id: int | None = None,
        locality_domain_backfill: bool | None = None,
        coscheduled_sm_count: int | None = None,
        device_id: int | None = None,
        green_context_obj: Any | None = None,
    ) -> None:
        r"""Create a CUDA green context.

        At least one of ``num_sms``, ``workqueue_scope``,
        ``locality_domain_id`` must be specified.
        ``num_sms`` and ``locality_domain_id`` cannot be specified together.
        ``locality_domain_backfill`` may only be specified together with
        ``locality_domain_id``.
        ``coscheduled_sm_count`` may only be specified together with
        ``locality_domain_id``.

        If ``green_context_obj`` is used, the context will wrap the given
        green context. In this case, no other argument can be specified
        at the same time.

        Arguments:
            num_sms (int, optional): The number of SMs to use in the green
                context. When ``None``, SMs are not partitioned by count.
            workqueue_scope (str, optional): Workqueue sharing scope. One of
                ``"device_ctx"`` (shared across all contexts, default driver
                behavior) or ``"balanced"`` (non-overlapping workqueues with
                other balanced green contexts). When ``None``, no workqueue
                configuration is applied.
            workqueue_concurrency_limit (int, optional): Maximum number of
                concurrent stream-ordered workloads for the workqueue. Requires
                ``workqueue_scope`` to be set.
            locality_domain_id (int, optional): Locality domain whose SMs
                should back this green context. When ``None``, no locality
                domain is selected.
            locality_domain_backfill (bool, optional): If ``True``, supplement
                the selected locality domain with additional SMs so all
                locality domains receive an equal share of the device's SMs.
                May only be specified with ``locality_domain_id``. Defaults to
                ``None``.
            coscheduled_sm_count (int, optional): The minimum number of SMs
                guaranteed to be co-scheduled for a thread block cluster.
                This determines the cluster capability of the green context
                if ``locality_domain_id`` is provided. When ``None`` or zero,
                it is determined automatically by the system. Defaults to
                ``None``.
            device_id (int, optional): The device index used.
                When ``None``, the current device is used.
            green_context_obj (optional): Wrap this cuda-bindings green context.
        """
        self._device_id = None
        self._green_ctx = None
        self._context = None
        # we don't own the green context object until it has been set up
        self._is_owning = False
        _ensure_supported()

        # if green_context_obj is provided, we just check that nothing else
        # is provided, and that the context is valid.
        if green_context_obj is not None:
            # we require latest support here, to be able to query everything
            # about the given green context + be able to use cuCtxGetDevice_v2
            _ensure_localization_supported()
            other_values = [
                num_sms,
                workqueue_scope,
                workqueue_concurrency_limit,
                locality_domain_id,
                locality_domain_backfill,
                coscheduled_sm_count,
                device_id,
            ]
            if any(v is not None for v in other_values):
                raise RuntimeError(
                    "If green_context_obj is provided, no other argument must be provided to GreenContext()"
                )
            # this also checks whether the green context is valid
            # pyrefly: ignore [missing-attribute]
            context = _check_cuda_bindings(_drv.cuCtxFromGreenCtx(green_context_obj))
            if int(context) == 0:
                raise RuntimeError("Green ctx conversion to regular ctx failed!")
            # pyrefly: ignore [missing-attribute]
            device_id = int(_check_cuda_bindings(_drv.cuCtxGetDevice_v2(context)))
            num_locality_domains = _get_num_locality_domains(device_id)
            sm_res = _check_cuda_bindings(
                # pyrefly: ignore [missing-attribute]
                _drv.cuGreenCtxGetDevResource(
                    # pyrefly: ignore [missing-attribute]
                    green_context_obj,
                    # pyrefly: ignore [missing-attribute]
                    _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
                )
            )
            if (
                sm_res.sm.flags
                # pyrefly: ignore [missing-attribute]
                & _drv.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID
            ) != 0:
                locality_domain_id = sm_res.sm.localityDomainId
            self._init_from_cuda_objects(
                device_id,
                green_context_obj,
                context,
                locality_domain_id,
                num_locality_domains,
            )
            return

        scope_value = _parse_workqueue_scope(workqueue_scope)
        if scope_value is not None:
            _ensure_workqueue_supported()
        if locality_domain_id is not None:
            _ensure_localization_supported()

        if locality_domain_backfill is not None and locality_domain_id is None:
            raise RuntimeError("locality_domain_backfill requires locality_domain_id")
        if coscheduled_sm_count is not None and locality_domain_id is None:
            raise RuntimeError("coscheduled_sm_count requires locality_domain_id")
        if num_sms is None and scope_value is None and locality_domain_id is None:
            raise RuntimeError(
                "At least one of num_sms, workqueue_scope, or "
                "locality_domain_id must be specified"
            )
        if locality_domain_id is not None and num_sms is not None:
            raise RuntimeError(
                "locality_domain_id and num_sms cannot be specified together"
            )
        if workqueue_concurrency_limit is not None and scope_value is None:
            raise RuntimeError(
                "workqueue_concurrency_limit requires workqueue_scope to be set"
            )

        if device_id is None:
            device_id = torch.cuda.current_device()
        if coscheduled_sm_count is None:
            coscheduled_sm_count = 0
        if locality_domain_id is not None:
            _validate_coscheduled_sm_count(device_id, coscheduled_sm_count)

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
        drv_device = _check_cuda_bindings(_drv.cuDeviceGet(device_id))
        resources = []
        num_locality_domains = 0

        if num_sms is not None:
            sm_resource = _check_cuda_bindings(
                # pyrefly: ignore [missing-attribute]
                _drv.cuDeviceGetDevResource(
                    drv_device,
                    # pyrefly: ignore [missing-attribute]
                    _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
                )
            )
            if num_sms <= 0 or num_sms > sm_resource.sm.smCount:
                raise RuntimeError(
                    "Invalid number of SMs requested for green context: "
                    f"{num_sms} (device has {sm_resource.sm.smCount} SMs)"
                )
            split_result, nb_groups, _remaining = _check_cuda_bindings(
                # pyrefly: ignore [missing-attribute]
                _drv.cuDevSmResourceSplitByCount(1, sm_resource, 0, num_sms)
            )
            if nb_groups != 1:
                raise RuntimeError("Failed to create single SM resource group")
            resources.append(split_result[0])

        if locality_domain_id is not None:
            localized_sms = _get_localized_sm_resources(
                device_id, bool(locality_domain_backfill), coscheduled_sm_count
            )
            num_locality_domains = len(localized_sms)
            if locality_domain_id < 0 or locality_domain_id >= num_locality_domains:
                raise RuntimeError(
                    "Invalid locality domain ID: "
                    f"{locality_domain_id} (device has {num_locality_domains})"
                )
            resources.append(localized_sms[locality_domain_id])

        if scope_value is not None:
            wq_resource = _check_cuda_bindings(
                # pyrefly: ignore [missing-attribute]
                _drv.cuDeviceGetDevResource(
                    drv_device,
                    # pyrefly: ignore [missing-attribute]
                    _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG,
                )
            )
            wq_resource.wqConfig.sharingScope = scope_value
            if workqueue_concurrency_limit is not None:
                wq_resource.wqConfig.wqConcurrencyLimit = workqueue_concurrency_limit
            resources.append(wq_resource)

        desc = _check_cuda_bindings(
            # pyrefly: ignore [missing-attribute]
            _drv.cuDevResourceGenerateDesc(resources, len(resources))
        )
        green_ctx = _check_cuda_bindings(
            # pyrefly: ignore [missing-attribute]
            _drv.cuGreenCtxCreate(
                desc,
                drv_device,
                # pyrefly: ignore [missing-attribute]
                _drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
            )
        )
        try:
            # pyrefly: ignore [missing-attribute]
            context = _check_cuda_bindings(_drv.cuCtxFromGreenCtx(green_ctx))
            if int(context) == 0:
                raise RuntimeError("Green ctx conversion to regular ctx failed!")
        except Exception:
            # pyrefly: ignore [missing-attribute]
            _check_cuda_bindings(_drv.cuGreenCtxDestroy(green_ctx))
            raise

        self._init_from_cuda_objects(
            device_id,
            green_ctx,
            context,
            locality_domain_id,
            num_locality_domains,
        )
        # at this point, we are set up, and we own the green context
        self._is_owning = True

    def __del__(self) -> None:
        green_ctx = getattr(self, "_green_ctx", None)
        if green_ctx is None:
            return

        cls = type(self)
        # retrieve these symbols from the class type instance to avoid
        # dangling references during interpreter shutdown
        _check_cuda_bindings = cls._check_cuda_bindings_
        _drv = cls._drv_
        if _check_cuda_bindings is None or _drv is None:
            self._green_ctx = None
            return

        # attempt to destroy streams related to this green context
        # we ignore errors to avoid leaking exceptions during __del__
        end = min(len(self._green_ctx_streams), self._curr_stream_idx + 1)
        for idx in reversed(range(end)):
            green_ctx_stream = self._green_ctx_streams[idx]
            if green_ctx_stream is None:
                continue
            try:
                # pyrefly: ignore [missing-attribute]
                _check_cuda_bindings(_drv.cuStreamDestroy(green_ctx_stream))
            except RuntimeError as e:
                warnings.warn(
                    f"Error while destroying green context stream at idx {idx} "
                    f"for green context {green_ctx}: {e}"
                )
        self._green_ctx = None
        if not self._is_owning:
            return
        try:
            # pyrefly: ignore [missing-attribute]
            _check_cuda_bindings(_drv.cuGreenCtxDestroy(green_ctx))
        except RuntimeError as e:
            warnings.warn(f"Error while destroying green context {green_ctx}: {e}")

    def _init_from_cuda_objects(
        self,
        device_id: int,
        green_ctx: Any,
        context: Any,
        locality_domain_id: int | None,
        num_locality_domains: int,
    ) -> None:
        self._device_id = device_id
        self._green_ctx = green_ctx
        self._context = context
        self._locality_domain_id = locality_domain_id
        self._num_locality_domains = num_locality_domains
        self._parent_stream: torch.cuda.Stream | None = None
        self._green_ctx_streams: list[Any | None] = [
            None
        ] * _STREAMS_PER_GREEN_CONTEXT_POOL
        self._curr_stream_idx = -1

    @staticmethod
    def create(
        *,
        num_sms: int | None = None,
        workqueue_scope: str | None = None,
        workqueue_concurrency_limit: int | None = None,
        locality_domain_id: int | None = None,
        locality_domain_backfill: bool | None = None,
        coscheduled_sm_count: int | None = None,
        device_id: int | None = None,
    ) -> GreenContext:
        r"""Create a CUDA green context.

        Kept for compatibility, see `GreenContext` constructor.
        """
        return GreenContext(
            num_sms=num_sms,
            workqueue_scope=workqueue_scope,
            workqueue_concurrency_limit=workqueue_concurrency_limit,
            locality_domain_id=locality_domain_id,
            locality_domain_backfill=locality_domain_backfill,
            coscheduled_sm_count=coscheduled_sm_count,
            device_id=device_id,
        )

    @staticmethod
    def max_workqueue_concurrency(device_id: int | None = None) -> int:
        r"""Return the maximum workqueue concurrency limit for the device.

        This queries the device for the default number of concurrent
        stream-ordered workloads supported by workqueue configuration
        resources.

        Arguments:
            device_id (int, optional): The device index to query.
                When ``None``, the current device is used.
        """
        _ensure_supported()
        _ensure_workqueue_supported()
        if device_id is None:
            device_id = torch.cuda.current_device()

        # pyrefly: ignore [missing-attribute]
        drv_device = _check_cuda_bindings(_drv.cuDeviceGet(device_id))
        wq_resource = _check_cuda_bindings(
            # pyrefly: ignore [missing-attribute]
            _drv.cuDeviceGetDevResource(
                drv_device,
                # pyrefly: ignore [missing-attribute]
                _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG,
            )
        )
        return wq_resource.wqConfig.wqConcurrencyLimit

    @property
    def device_id(self) -> int:
        self._ensure_alive()
        device_id = self._device_id
        if device_id is None:
            raise RuntimeError("GreenContext has been destroyed")
        return device_id

    @property
    def locality_domain_id(self) -> int | None:
        self._ensure_alive()
        return self._locality_domain_id

    @property
    def has_locality_domain(self) -> bool:
        self._ensure_alive()
        return self._locality_domain_id is not None and self._locality_domain_id >= 0

    @property
    def num_locality_domains(self) -> int:
        self._ensure_alive()
        return self._num_locality_domains

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
        self._curr_stream_idx += 1
        curr_idx = self._curr_stream_idx
        idx = curr_idx % _STREAMS_PER_GREEN_CONTEXT_POOL
        if curr_idx < _STREAMS_PER_GREEN_CONTEXT_POOL:
            green_ctx_stream = _check_cuda_bindings(
                # pyrefly: ignore [missing-attribute]
                _drv.cuGreenCtxStreamCreate(
                    self._green_ctx,
                    # pyrefly: ignore [missing-attribute]
                    _drv.CUstream_flags.CU_STREAM_NON_BLOCKING,
                    0,
                )
            )
            self._green_ctx_streams[idx] = green_ctx_stream
            with _STREAM_TO_GREEN_CTX_LOCK:
                _STREAM_TO_GREEN_CTX.setdefault(int(green_ctx_stream), self)
        else:
            green_ctx_stream = self._green_ctx_streams[idx]
        # pyrefly: ignore [bad-argument-type]
        return torch.cuda.ExternalStream(int(green_ctx_stream), self._device_id)


def _get_green_ctx_from_stream(stream: int) -> GreenContext | None:
    with _STREAM_TO_GREEN_CTX_LOCK:
        ctx = _STREAM_TO_GREEN_CTX.get(stream)
    if ctx is not None:
        return ctx
    try:
        # pyrefly: ignore [missing-attribute]
        drv_ctx = _check_cuda_bindings(_drv.cuStreamGetGreenCtx(stream))
        if int(drv_ctx) == 0:
            return None
        ctx = GreenContext(green_context_obj=drv_ctx)
        with _STREAM_TO_GREEN_CTX_LOCK:
            _STREAM_TO_GREEN_CTX.setdefault(stream, ctx)
        return ctx
    except RuntimeError:
        return None


def get_green_context_from_stream(stream: torch.cuda.Stream) -> GreenContext | None:
    r"""Return the green context associated with a CUDA stream, if any.

    If the association is not already registered, it is queried from CUDA and
    returned as a non-owning :class:`GreenContext` wrapper. The wrapper does not
    destroy the underlying CUDA green context. Returns ``None`` if ``stream`` is
    not associated with a green context.
    """
    return _get_green_ctx_from_stream(stream.cuda_stream)


def execute_in_green_contexts(
    green_ctx_streams: list[torch.cuda.Stream],
    fn: Callable[[int, GreenContext | None], None],
) -> None:
    r"""Execute a function in a list of green context streams in parallel."""
    if not green_ctx_streams:
        raise ValueError("Need at least one green context to execute in!")
    if len(green_ctx_streams) == 1:
        with torch.cuda.stream(green_ctx_streams[0]):
            fn(0, get_green_context_from_stream(green_ctx_streams[0]))
        return

    green_contexts = [
        get_green_context_from_stream(stream) for stream in green_ctx_streams
    ]
    green_events = [torch.cuda.Event() for _ in green_ctx_streams]
    main_event = torch.cuda.Event()
    main_stream = torch.cuda.current_stream()
    main_event.record(main_stream)

    if all(stream.device == main_stream.device for stream in green_ctx_streams):
        # Avoid restoring the main stream between callbacks. The old sequence
        # main -> green[0] -> main -> green[1] -> main adds enough host latency
        # to prevent short kernels from overlapping.
        try:
            for i, (green_ctx_stream, green_context, green_event) in enumerate(
                zip(green_ctx_streams, green_contexts, green_events)
            ):
                torch.cuda.set_stream(green_ctx_stream)
                green_ctx_stream.wait_event(main_event)
                fn(i, green_context)
                green_event.record(green_ctx_stream)
        finally:
            torch.cuda.set_stream(main_stream)
    else:
        # Preserve StreamContext device-switching behavior when a stream is on
        # a device other than the caller's current device.
        for i, (green_ctx_stream, green_context, green_event) in enumerate(
            zip(green_ctx_streams, green_contexts, green_events)
        ):
            with torch.cuda.stream(green_ctx_stream):
                green_ctx_stream.wait_event(main_event)
                fn(i, green_context)
                green_event.record(green_ctx_stream)

    for green_event in green_events:
        main_stream.wait_event(green_event)
