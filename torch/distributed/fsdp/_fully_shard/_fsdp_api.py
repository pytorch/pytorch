# mypy: allow-untyped-defs
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, KW_ONLY

import torch
import torch.distributed as dist

from ._all_gather_layout import AllGatherLayout, DEFAULT_ALL_GATHER_LAYOUT


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
        reduce_dtype (Optional[torch.dtype]): The dtype for gradient reduction
            (reduce-scatter or all-reduce) and for accumulating gradients while
            reduction is disabled via :meth:`FSDPModule.set_requires_gradient_sync`.
            If ``None``, follows the parameter's ``grad_dtype`` configured before
            lazy initialization (the first forward or :meth:`FSDPModule.unshard`):
            unset uses the original parameter dtype;
            explicit ``None`` accepts any incoming gradient dtype. This fallback
            is independent of ``param_dtype``. Gradients with different dtypes
            in one communication group are reduced in their promoted dtype
            (e.g. fp32 for bf16 and fp32). Reduced shards retain the input
            ``grad_dtype`` policy. (Default: ``None``)

            .. versionchanged:: 2.15
                With ``reduce_dtype=None``, gradients were previously reduced
                in ``param_dtype`` when it was set. They now follow the
                parameter's ``grad_dtype`` as described above, e.g. fp32 for fp32
                parameters with ``param_dtype=torch.bfloat16``. Set
                ``reduce_dtype=torch.bfloat16`` to keep the previous behavior.
                FSDP1 still reduces in ``param_dtype``.
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

    ``layout`` is installed with the comm by ``set_custom_all_gather`` and
    selects input packing and output handling; a backend whose collective
    produces a custom layout provides its matching ``AllGatherLayout``.
    """

    layout: AllGatherLayout = DEFAULT_ALL_GATHER_LAYOUT
    # Preserve version counters when outputs may alias saved parameter views
    reuses_output_storage: bool = False

    def release_output(self) -> None:
        """Release this group's output lease on the current stream.

        Called after reshard, after waiting for a discarded unused prefetch, and
        after a failed input preparation or collective setup. It must be
        idempotent when no output is active. Previously returned parameter views
        must remain valid objects; a backend sharing their storage must restore
        the same regions on the next gather and order overwrites after all local
        and remote consumers, since FSDP does not synchronize backend-owned
        output reuse.
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


@dataclass(frozen=True)
class AllGatherInput:
    r"""Describe one payload returned by an FSDP all-gather extension.

    Return these records in the inputs of ``(inputs, metadata)`` from
    ``fsdp_pre_all_gather``. Each rank's payload is concatenated along ``dim``
    using its own shape, independently of the parameter's shard dimension.
    For example, a payload of shape ``(2, F, D)`` with ``dim=1`` produces
    ``(2, world_size * F, D)``. Scalar payloads are treated as shape ``(1,)``.
    The gathered payload is optionally reshaped to ``output_size`` before
    being passed to the unchanged ``fsdp_post_all_gather`` hook.

    Payloads must be flattenable with ``view(-1)``. Each rank must return the
    same payload shapes, dtypes, and layouts; extensions own any padding.
    The number, element counts, and dtypes of payloads must stay fixed across
    calls so FSDP can reuse their output buffers.

    Attributes:
        tensor (Tensor): Local payload to communicate.
        dim (int): Payload dimension to concatenate across ranks. Negative
            dimensions are supported. Defaults to 0.
        output_size (torch.Size, optional): Shape passed to the post hook, with
            the same number of elements as the gathered payload. Defaults to
            the concatenated shape.
    """

    tensor: torch.Tensor
    _: KW_ONLY
    dim: int = 0
    output_size: torch.Size | None = None
