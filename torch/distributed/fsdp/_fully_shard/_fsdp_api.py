# mypy: allow-untyped-defs
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist


_ReduceOp = dist.ReduceOp | dist.ReduceOp.RedOpType


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    """
    This configures FSDP's mixed precision. Unlike autocast, this applies mixed
    precision at the module level, not op level, which means low-precision
    activations are saved for backward and high-to-low-precision casts are
    incurred only at module boundaries.

    FSDP works well with module-level mixed precision since it keeps the
    high-precision sharded parameters in memory anyway. In other words, FSDP
    does not require any extra memory to keep a high-precision copy of the
    parameters for the optimizer step.

    Attributes:
        param_dtype (Optional[torch.dtype]): This specifies the dtype for
            the unsharded parameter and hence the dtype for forward/backward
            computation and the parameter all-gather. If this is ``None``, then
            the unsharded parameter uses the original dtype. The optimizer step
            uses the sharded parameter in the original dtype. (Default:
            ``None``)
        reduce_dtype (Optional[torch.dtype]): This specifies the dtype for
            gradient reduction (i.e. reduce-scatter or all-reduce). If this is
            ``None`` but ``param_dtype`` is not ``None``, then the reduction
            uses the compute dtype. This can be used to run gradient reduction
            in full precision while using low precision for compute. If also
            gradient reduction is disabled via :meth:`set_requires_gradient_sync`,
            then FSDP will accumulate gradients using ``reduce_dtype``.
            (Default: ``None``)
        output_dtype (Optional[torch.dtype]): This specifies the dtype for
            casting floating-point forward outputs. This can be used to
            help implement cases where different modules have different mixed
            precision policies. (Default: ``None``)
        cast_forward_inputs (bool): This specifies whether FSDP should cast the
            forward's floating-point input tensors to ``param_dtype`` or not.
            For grouped ``fully_shard([a, b, ...])``, the cast is applied per
            module, before each module's forward.
    """

    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    cast_forward_inputs: bool = True


class Comm(ABC):
    """
    Interface for communication primitives.
    A primitive primarily needs to handle 3 tasks, namely:

    1. How to allocate memory for communication
       Depending on the goal, an implementation can choose to:
       a. associate each call to a temporary buffer
          (best for flexibility and simplicity)
       b. reuse a persistent buffer for efficiency reasons

    2. Where to allocate memory
       (e.g. NCCL mem pool or regular cuda caching allocator)

    3. What to do/call upon the comm is called
       (see `AllGather` interface as an example)
    """

    @abstractmethod
    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        This handles the "how to allocate memory" part.

        A default implementation could be simply:

        .. code-block:: python
            with self.mem_pool:
                torch.empty(...)

        Args:
            size (Sequence[Union[int, torch.SymInt]]): size of the tensor buffer
            dtype (torch.dtype): dtype of the tensor buffer
            device (torch.device): which device to allocate the tensor onto
        """
        ...


class AllGather(Comm):
    """
    Interface for all_gather comm primitive
    """

    @abstractmethod
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> dist.Work | None: ...


class ReduceScatter(Comm):
    """
    Interface for reduce_scatter comm primitive
    """

    @abstractmethod
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = False,
    ) -> dist.Work | None: ...


@dataclass
class DataParallelMeshDims:
    """
    Specifies which dimensions of a full SPMD :class:`DeviceMesh` correspond to
    data parallelism when using :func:`fully_shard` whose parameters are already
    DTensors on that mesh.

    Attributes:
        shard (Optional[Union[str, tuple[str, ...]]]): Mesh dimension name(s)
            that FSDP shards parameters on. If a tuple of names, those dims
            are flattened into a single shard dimension. At least one of
            ``shard`` and ``replicate`` must be set.
        replicate (Optional[Union[str, tuple[str, ...]]]): Mesh dimension
            name(s) for HSDP or DDP replication. If a tuple of names, those
            dims are flattened into a single replicate dimension.
    """

    shard: str | tuple[str, ...] | None = None
    replicate: str | tuple[str, ...] | None = None

    def __post_init__(self):
        if self.shard is None and self.replicate is None:
            raise ValueError(
                "At least one of shard or replicate must be set in DataParallelMeshDims"
            )

    @property
    def shard_names(self) -> tuple[str, ...]:
        if self.shard is None:
            return ()
        if isinstance(self.shard, str):
            return (self.shard,)
        return tuple(self.shard)

    @property
    def replicate_names(self) -> tuple[str, ...]:
        if self.replicate is None:
            return ()
        if isinstance(self.replicate, str):
            return (self.replicate,)
        return tuple(self.replicate)


@dataclass
class OffloadPolicy:
    """
    This base class represents the policy of no offloading and is only used as
    the default value for the ``offload_policy`` arg.
    """


@dataclass
class CPUOffloadPolicy(OffloadPolicy):
    """
    This offload policy offloads parameters, gradients, and optimizer states to
    CPU. Sharded parameters are copied host-to-device before all-gather. The
    all-gathered parameters are freed according to ``reshard_after_forward``.
    Sharded gradients are copied device-to-host in backward, and the optimizer
    step runs on CPU with CPU optimizer states.

    Attributes:
        pin_memory (bool): Whether to pin sharded parameter and gradient
            memory. Pinning memory allows both more efficient H2D/D2H copies
            and for the copies to overlap with compute. However, the pinned
            memory cannot be used by other processes. Set this to ``False`` if
            you have insufficient CPU memory. (Default: ``True``)
    """

    pin_memory: bool = True


def _select_largest_first(
    fqn_numbytes: list[tuple[str, int]], budget_bytes: int
) -> list[str]:
    """Select FQNs largest-first until the CPU offload budget is exhausted.

    Parameters are sorted by byte size descending, with a stable tie-break on
    FQN, and the longest prefix whose cumulative size does not exceed
    ``budget_bytes`` is returned, stopping at the first parameter that would
    exceed it. The selection is deterministic, never overshoots the budget, and
    is monotonic in the budget: raising ``budget_bytes`` only extends the
    prefix, so the selected set grows as a superset. It takes no torch
    dependency so it can be unit tested on plain Python data.
    """
    ordered = sorted(fqn_numbytes, key=lambda kv: (-kv[1], kv[0]))
    selected: list[str] = []
    used = 0
    for fqn, numbytes in ordered:
        if used + numbytes > budget_bytes:
            break
        selected.append(fqn)
        used += numbytes
    return selected


def cpu_offload_by_budget(
    module: torch.nn.Module, budget_bytes: int, *, pin_memory: bool = True
) -> dict[str, OffloadPolicy]:
    """Build a per-parameter CPU offload map under a memory budget.

    Returns a mapping from parameter FQN (relative to ``module``) to
    :class:`OffloadPolicy`, suitable to pass as ``fully_shard``'s
    ``offload_policy``. The largest parameters are offloaded first until
    ``budget_bytes`` of parameter memory has been placed on CPU; the remaining
    parameters stay on device and are omitted from the map. ``budget_bytes`` is
    measured against full (unsharded) parameter sizes. A budget of ``0``
    offloads nothing and a budget at or above the total parameter size offloads
    everything, reproducing :class:`OffloadPolicy` and :class:`CPUOffloadPolicy`
    respectively.

    Note:
        When only some parameters are offloaded, a parameter group spans both
        CPU and device shards. Adam/AdamW stay correct on such a group, but the
        default ``foreach``/``fused`` auto-selection does not engage a
        multi-tensor kernel across mixed devices; pass ``fused=True`` (or
        ``foreach=True``) to keep the per-device multi-tensor step.
    """
    if budget_bytes < 0:
        raise ValueError(f"budget_bytes must be non-negative, but got {budget_bytes}")
    fqn_numbytes = [
        (fqn, param.numel() * param.element_size())
        for fqn, param in module.named_parameters()
    ]
    policy = CPUOffloadPolicy(pin_memory=pin_memory)
    return {fqn: policy for fqn in _select_largest_first(fqn_numbytes, budget_bytes)}
