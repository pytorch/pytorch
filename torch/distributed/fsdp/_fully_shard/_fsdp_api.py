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
    Interface for all_gather comm primitive.

    By default FSDP all-gathers into a rank-major ``[rank][param]`` buffer and
    then copies each parameter out into its own contiguous storage. A backend
    that can instead produce a parameter-major ``[param][rank]`` output may set
    :attr:`supports_param_contiguous_output` to ``True`` and implement
    :meth:`prepare_param_contiguous_output`. FSDP then views the unsharded
    parameters directly on top of the backend output, skipping the copy-out and
    its extra allocation.
    """

    # Whether the backend can write a parameter-contiguous ``[param][rank]``
    # all-gather output that FSDP can use in place (i.e. without a copy-out).
    supports_param_contiguous_output: bool = False

    @abstractmethod
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> dist.Work | None: ...

    def prepare_param_contiguous_output(
        self,
        all_gather_input_split_sizes: list[int],
        all_gather_input_numel: int,
        world_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> object | None:
        """
        Prepare backend state for a parameter-contiguous all-gather output.

        Only called when :attr:`supports_param_contiguous_output` is ``True``.
        ``all_gather_input_split_sizes`` gives the per-parameter input numels
        (summing to ``all_gather_input_numel``). The returned value, if any, is
        opaque backend metadata carried through the all-gather and is otherwise
        unused by FSDP.
        """
        return None

    def clear_param_contiguous_output(self) -> None:
        """Release any state created by :meth:`prepare_param_contiguous_output`."""


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
