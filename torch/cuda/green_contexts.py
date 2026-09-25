from __future__ import annotations

import functools
import sys
import threading
import warnings
import weakref
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING
from typing_extensions import deprecated

import torch
from torch.cuda._utils import (
    _check_cuda_bindings,
    _cuda_bindings_driver as _drv,
    _cuda_bindings_runtime as _rt,
    _ensure_cuda_bindings_version,
    _get_device_index,
    _HAS_CUDA_BINDINGS,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.types import Device

__all__ = [
    "GreenContext",
    "SMPartition",
    "execute_in_green_contexts",
    "get_green_context_from_stream",
    "get_num_locality_domains",
    "is_localization_supported",
]

_STREAMS_PER_GREEN_CONTEXT_POOL = 32
_NVML_ERROR_NOT_SUPPORTED = 3
_NVML_DEVICE_MIG_ENABLE = 1
_NVML_GPU_VIRTUALIZATION_MODE_VGPU = 2

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


def _ensure_cuda_version(version: int, message: str) -> None:
    drv_version = _get_driver_version()
    if drv_version < 0 or drv_version < version:
        raise RuntimeError(message)
    _ensure_cuda_bindings_version(version, message)


def _ensure_supported() -> None:
    if not torch.backends.cuda.is_built() or torch.version.hip is not None:
        raise RuntimeError("Green Context is only supported on Nvidia CUDA")
    if sys.platform == "win32":
        raise RuntimeError("Green Context is not supported on Windows")
    if not _HAS_CUDA_BINDINGS:
        raise RuntimeError("GreenContext requires the cuda.bindings package")
    _ensure_cuda_version(
        12080, "Green Context requires user mode driver and cuda.bindings package 12.8+"
    )


def _ensure_workqueue_supported() -> None:
    _ensure_cuda_version(
        13010,
        "Green Context workqueue configuration requires user mode driver and "
        "cuda.bindings package 13.1+",
    )


def _ensure_localization_supported() -> None:
    _ensure_cuda_version(
        13040,
        "Green Context localization requires user mode driver and "
        "cuda.bindings package 13.4+",
    )


def _ensure_disjoint_sm_split_supported() -> None:
    _ensure_cuda_version(
        13010,
        "Green Context disjoint SM splits require user mode driver and "
        "cuda.bindings package 13.1+",
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


def _uses_single_locality_domain_nvml(device: Device) -> bool | None:
    from ctypes import byref, c_int, c_uint, c_void_p, CDLL

    try:
        nvml_h = CDLL("libnvidia-ml.so.1")
        if nvml_h.nvmlInit() != 0:
            return None
        device_index = torch.cuda._get_nvml_device_index(device)
        device_handle = c_void_p()
        if (
            nvml_h.nvmlDeviceGetHandleByIndex_v2(device_index, byref(device_handle))
            != 0
        ):
            return None

        vgpu_mode = c_int()
        status = nvml_h.nvmlDeviceGetVirtualizationMode(device_handle, byref(vgpu_mode))
        if status not in (0, _NVML_ERROR_NOT_SUPPORTED):
            return None
        if status == 0 and vgpu_mode.value == _NVML_GPU_VIRTUALIZATION_MODE_VGPU:
            return True

        is_mig_device = c_uint()
        status = nvml_h.nvmlDeviceIsMigDeviceHandle(device_handle, byref(is_mig_device))
        if status not in (0, _NVML_ERROR_NOT_SUPPORTED):
            return None
        if status == 0 and is_mig_device.value != 0:
            return True

        current_mig_mode = c_uint()
        pending_mig_mode = c_uint()
        status = nvml_h.nvmlDeviceGetMigMode(
            device_handle, byref(current_mig_mode), byref(pending_mig_mode)
        )
        if status not in (0, _NVML_ERROR_NOT_SUPPORTED):
            return None
        if status == 0 and current_mig_mode.value == _NVML_DEVICE_MIG_ENABLE:
            return True

        numa_node = c_uint()
        status = nvml_h.nvmlDeviceGetNumaNodeId(device_handle, byref(numa_node))
        if status not in (0, _NVML_ERROR_NOT_SUPPORTED):
            return None
        return status == 0
    except (AttributeError, IndexError, OSError):
        return None


def _get_num_locality_domains_nvml() -> int | None:
    try:
        # Without initializing the CUDA driver/context, we can only determine
        # locality domain support under certain conditions:
        # all NVML devices visible through `CUDA_VISIBLE_DEVICES` must agree;
        # we assume that `device_id` must be one of these devices.
        # Locality domains are not supported on Windows, but this is already
        # checked in `_ensure_supported` above.
        # We check for vGPU/MIG devices or devices exposed as NUMA nodes:
        # these don't support locality domains either, so it must be 1.
        # Finally, all devices with SM 10.X have exactly 2 locality domains,
        # and devices with SM < 10.0 or SM == 12.X have 1 locality domain.
        # These conditions might evolve in the future.

        visible_devices = torch.cuda._parse_visible_devices()
        # UUID parsing stops at the first identifier with a different prefix,
        # so a returned string list beginning with MIG- contains only MIG UUIDs.
        if (
            visible_devices
            and isinstance(visible_devices[0], str)
            and visible_devices[0].startswith("MIG-")
        ):
            return 1
        num_nvml_devices = torch.cuda._device_count_nvml()
        if num_nvml_devices <= 0:
            reason = "NVML device count is not available"
        else:
            loc_domains = set()
            for nvml_id in range(num_nvml_devices):
                capability = torch.cuda._raw_device_capability_nvml(nvml_id)
                single_domain = _uses_single_locality_domain_nvml(nvml_id)
                if single_domain is None or capability is None:
                    reason = f"NVML device queries failed for visible device {nvml_id}"
                    break
                loc_domains.add(2 if not single_domain and capability[0] == 10 else 1)
            else:
                if len(loc_domains) == 1:
                    return loc_domains.pop()
                reason = (
                    "visible NVML devices have different inferred locality domain "
                    f"counts: {sorted(loc_domains)}"
                )
    except (AttributeError, IndexError, OSError, RuntimeError, ValueError) as error:
        reason = str(error)
    warnings.warn(
        f"Cannot determine the locality domain count using NVML: {reason}. "
        "Falling back to CUDA driver initialization, which may poison subsequent "
        "forks that use CUDA.",
        stacklevel=3,
    )
    return None


@functools.cache
def _get_num_locality_domains(device_id: int) -> int:
    _ensure_supported()
    _ensure_localization_supported()
    # pyrefly: ignore [missing-attribute]
    device_result = _drv.cuDeviceGet(device_id)
    # pyrefly: ignore [missing-attribute]
    if device_result[0] == _drv.CUresult.CUDA_ERROR_NOT_INITIALIZED:
        count = _get_num_locality_domains_nvml()
        if count is not None:
            return count
        _check_cuda_bindings(_drv.cuInit(0))  # pyrefly: ignore [missing-attribute]
        # pyrefly: ignore [missing-attribute]
        device_result = _drv.cuDeviceGet(device_id)
    device = _check_cuda_bindings(device_result)
    return _check_cuda_bindings(
        # pyrefly: ignore [missing-attribute]
        _drv.cuDeviceGetAttribute(
            # pyrefly: ignore [missing-attribute]
            _drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT,
            device,
        )
    )


def _ensure_primary_context() -> None:
    # pyrefly: ignore [missing-attribute]
    current_ctx = _check_cuda_bindings(_drv.cuCtxGetCurrent())
    if int(current_ctx) != 0:
        return
    warnings.warn(
        "Attempted to create a green context but there was no primary "
        "context! Creating a primary context...",
        stacklevel=3,
    )
    # pyrefly: ignore [missing-attribute]
    _check_cuda_bindings(_rt.cudaFree(0))


def get_num_locality_domains(device: Device = None) -> int:
    r"""Return the number of CUDA locality domains, or one if unavailable.

    Accepts a device index, CUDA device string, or :class:`torch.device`.
    Before CUDA initialization, first tries NVML without initializing the driver.
    If NVML cannot determine the count, warns and initializes the driver to query
    it directly. This fallback may poison subsequent forks that use CUDA.
    """
    try:
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
    except (RuntimeError, ValueError, TypeError):
        return 1


def is_localization_supported(device: Device = None) -> bool:
    r"""Return whether CUDA green context localization is available."""
    return get_num_locality_domains(device) > 1


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

    @property
    def locality_domain_id(self) -> int | None:
        r"""The locality domain reported by CUDA, or ``None`` if unspecified.

        Returns ``None`` if locality queries require newer CUDA software.
        This reads metadata, not the physical locality of every selected SM.
        """
        try:
            _ensure_localization_supported()
        except RuntimeError:
            return None
        # Nested splits need not inherit locality metadata. Backfill may also
        # include SMs outside this domain; an ID is not a containment guarantee.
        flag = (
            # pyrefly: ignore [missing-attribute]
            _drv.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID
        )
        if self._resource.sm.flags & flag:
            return self._resource.sm.localityDomainId
        return None

    def split(
        self,
        *,
        num_sms: int | Sequence[int] = 0,
        coscheduled_sm_count: int | Sequence[int] = 0,
        preferred_coscheduled_sm_count: int | Sequence[int] = 0,
        backfill: bool | Sequence[bool] = False,
        locality_domain_ids: int | None | Sequence[int | None] = None,
    ) -> tuple[tuple[SMPartition, ...], SMPartition | None]:
        r"""Split this resource into disjoint groups and an optional remainder.

        Requires CUDA driver and bindings 13.1+. CUDA checks the requested
        counts and hardware constraints; counts are not automatically rounded.
        A count of zero requests discovery of the largest remaining group
        satisfying its constraints. Groups are processed in order.

        Args:
            num_sms (int or sequence of int): SM count for each requested group.
                Zero enables discovery mode. Default: ``0``.
            coscheduled_sm_count (int or sequence of int, optional): Co-scheduled
                SM grouping size for thread-block clusters. Zero lets CUDA
                determine cluster capabilities from the selected resources.
                Default: ``0``.
            preferred_coscheduled_sm_count (int or sequence of int, optional):
                Preferred larger grouping size, when CUDA can combine groups.
                Zero selects the CUDA default. Default: ``0``.
            backfill (bool or sequence of bool, optional): Allow CUDA to fill
                groups with SMs outside the co-scheduling or locality constraints.
                Default: ``False``.
            locality_domain_ids (int, None, or sequence of int or None, optional):
                Select SMs from these locality domains during splitting. ``None``
                leaves locality unconstrained. Requires CUDA driver and bindings
                13.4+ when any domain is specified. Default: ``None``.

        Each option can be a scalar or a sequence. All sequences must have the
        same nonzero length; scalars are broadcast to that length. If every
        option is scalar, the split has one group.

        An early discovery group, especially with backfill, may leave no SMs
        for later groups. CUDA rejects groups resolving to zero SMs. Backfill
        relaxes constraints to reach a requested count; it does not otherwise
        consume the remainder.
        Returns ``(partitions, remainder)``, with ``None`` for an empty remainder.
        The remainder does not inherit the requested alignment.

        To subdivide a returned partition or remainder, create a
        :class:`GreenContext` from it and split the context's queried
        :attr:`~GreenContext.sm_partition`. CUDA drivers can reject raw split
        outputs as already partitioned resources. Context creation is explicit.

        Children are subsets of this resource and overlap it. Siblings from
        this operation, including the remainder, are disjoint. Results from
        separate split operations may overlap.

        Example::

            >>> sms = SMPartition.from_device(device_id=0)
            >>> (first,), rest = sms.split(num_sms=4, coscheduled_sm_count=2)
            >>> rest_ctx = GreenContext(sm_partition=rest)
            >>> (second,), rest = rest_ctx.sm_partition.split(
            ...     num_sms=4, coscheduled_sm_count=2
            ... )
        """
        _ensure_disjoint_sm_split_supported()
        fields = {
            "num_sms": num_sms,
            "coscheduled_sm_count": coscheduled_sm_count,
            "preferred_coscheduled_sm_count": preferred_coscheduled_sm_count,
            "backfill": backfill,
            "locality_domain_ids": locality_domain_ids,
        }
        sequences = {
            name: tuple(value)
            for name, value in fields.items()
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
            for name, value in fields.items()
        }
        counts = values["num_sms"]
        co_counts = values["coscheduled_sm_count"]
        preferred = values["preferred_coscheduled_sm_count"]
        backfills = values["backfill"]
        domains = values["locality_domain_ids"]
        if any(domain is not None for domain in domains):
            _ensure_localization_supported()
        for domain in domains:
            if domain is not None and (
                not isinstance(domain, int) or isinstance(domain, bool) or domain < 0
            ):
                raise ValueError(
                    "locality_domain_ids entries must be nonnegative integers or None"
                )
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
            if domains[index] is not None:
                param.flags |= (
                    # pyrefly: ignore [missing-attribute]
                    _drv.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID
                )
                param.localityDomainId = domains[index]
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
        sm_partition: SMPartition | None = None,
        device_id: int | None = None,
        green_context_obj: Any | None = None,
    ) -> None:
        r"""Create a CUDA green context.

        At least one of ``num_sms``, ``workqueue_scope``, or
        ``sm_partition`` must be specified. ``num_sms`` and
        ``sm_partition`` cannot be specified together.

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
            sm_partition (SMPartition, optional): Existing SM resources to use.
                The context uses the partition's device. Reusing a partition
                creates contexts sharing the same SMs.
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
        locality_domain_id = None

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
                sm_partition,
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
            num_locality_domains = get_num_locality_domains(device_id)
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
        if sm_partition is not None and not isinstance(sm_partition, SMPartition):
            raise TypeError("sm_partition must be an SMPartition")
        if num_sms is None and scope_value is None and sm_partition is None:
            raise RuntimeError(
                "At least one of num_sms, workqueue_scope, or sm_partition "
                "must be specified"
            )
        if num_sms is not None and sm_partition is not None:
            raise RuntimeError("num_sms and sm_partition cannot be specified together")
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

        _ensure_primary_context()

        # pyrefly: ignore [missing-attribute]
        drv_device = _check_cuda_bindings(_drv.cuDeviceGet(device_id))
        resources = []
        num_locality_domains = 1

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

        if sm_partition is not None:
            resources.append(sm_partition._resource)
            locality_domain_id = sm_partition.locality_domain_id
            if locality_domain_id is not None:
                num_locality_domains = get_num_locality_domains(device_id)

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
        self._sm_count: int | None = None
        self._parent_stream: torch.cuda.Stream | None = None
        self._green_ctx_streams: list[Any | None] = [
            None
        ] * _STREAMS_PER_GREEN_CONTEXT_POOL
        self._curr_stream_idx = -1

    @staticmethod
    def split(
        *,
        num_sms: int | Sequence[int] = 0,
        coscheduled_sm_count: int | Sequence[int] = 0,
        preferred_coscheduled_sm_count: int | Sequence[int] = 0,
        backfill: bool | Sequence[bool] = False,
        locality_domain_ids: int | None | Sequence[int | None] = None,
        workqueue_scope: str | None = None,
        workqueue_concurrency_limit: int | None = None,
        device_id: int | None = None,
    ) -> tuple[GreenContext, ...]:
        r"""Create contexts backed by disjoint SM partitions of a device.

        Partition options are those of :meth:`SMPartition.split`. Workqueue
        options are applied to each context. Unassigned SMs are unused; use
        :meth:`SMPartition.split` to retain the remainder for later use.

        Example::

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
            locality_domain_ids=locality_domain_ids,
        )
        return tuple(
            GreenContext(
                sm_partition=partition,
                workqueue_scope=workqueue_scope,
                workqueue_concurrency_limit=workqueue_concurrency_limit,
            )
            for partition in partitions
        )

    @staticmethod
    def create(
        *,
        num_sms: int | None = None,
        workqueue_scope: str | None = None,
        workqueue_concurrency_limit: int | None = None,
        sm_partition: SMPartition | None = None,
        device_id: int | None = None,
    ) -> GreenContext:
        r"""Create a CUDA green context.

        Kept for compatibility, see `GreenContext` constructor.
        """
        return GreenContext(
            num_sms=num_sms,
            workqueue_scope=workqueue_scope,
            workqueue_concurrency_limit=workqueue_concurrency_limit,
            sm_partition=sm_partition,
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
        r"""Return the current number of SMs available to this green context."""
        self._ensure_alive()
        if self._sm_count is not None:
            return self._sm_count
        sm_resource = _check_cuda_bindings(
            # pyrefly: ignore [missing-attribute]
            _drv.cuGreenCtxGetDevResource(
                self._green_ctx,
                # pyrefly: ignore [missing-attribute]
                _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
            )
        )
        self._sm_count = sm_resource.sm.smCount
        return self._sm_count

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
    fn: Callable[[int], None],
) -> None:
    r"""Execute a function in a list of green context streams in parallel.

    ``fn`` is invoked for each stream with its index in ``green_ctx_streams``.
    During the invocation, that stream is the default stream for CUDA work,
    activating execution in its green context. The associated green context can
    be obtained with :func:`get_green_context_from_stream`, using either the
    default stream inside ``fn`` or ``green_ctx_streams[index]``.
    """
    if not green_ctx_streams:
        raise ValueError("Need at least one green context to execute in!")
    if len(green_ctx_streams) == 1:
        with torch.cuda.stream(green_ctx_streams[0]):
            fn(0)
        return

    green_events = [torch.cuda.Event() for _ in green_ctx_streams]
    main_event = torch.cuda.Event()
    main_stream = torch.cuda.current_stream()
    main_event.record(main_stream)

    if all(stream.device == main_stream.device for stream in green_ctx_streams):
        # Avoid restoring the main stream between callbacks. The old sequence
        # main -> green[0] -> main -> green[1] -> main adds enough host latency
        # to prevent short kernels from overlapping.
        try:
            for i, (green_ctx_stream, green_event) in enumerate(
                zip(green_ctx_streams, green_events)
            ):
                torch.cuda.set_stream(green_ctx_stream)
                green_ctx_stream.wait_event(main_event)
                fn(i)
                green_event.record(green_ctx_stream)
        finally:
            torch.cuda.set_stream(main_stream)
    else:
        # Preserve StreamContext device-switching behavior when a stream is on
        # a device other than the caller's current device.
        for i, (green_ctx_stream, green_event) in enumerate(
            zip(green_ctx_streams, green_events)
        ):
            with torch.cuda.stream(green_ctx_stream):
                green_ctx_stream.wait_event(main_event)
                fn(i)
                green_event.record(green_ctx_stream)

    for green_event in green_events:
        main_stream.wait_event(green_event)
