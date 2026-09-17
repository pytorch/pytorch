# mypy: allow-untyped-defs
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import torch
import torch.distributed as dist
import torch.nn as nn


_ReduceOp = dist.ReduceOp | dist.ReduceOp.RedOpType


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    r"""
    This configures FSDP's mixed precision. Unlike autocast, parameter casting
    happens when parameters are all-gathered, while optional input and output
    casting happens at module boundaries. This means low-precision activations
    are saved for backward and high-to-low-precision casts are incurred only at
    those boundaries.

    FSDP works well with module-level mixed precision since it keeps the
    high-precision sharded parameters in memory anyway. In other words, FSDP
    does not require any extra memory to keep a high-precision copy of the
    parameters for the optimizer step.

    .. warning::
        ``param_dtype_override_fn`` must return the same result for each logical parameter
        on every rank. Rank-dependent results may cause ranks to build incompatible
        collective buffers, which can fail or hang.

    Attributes:
        param_dtype (Optional[torch.dtype]): This specifies the default dtype
            for the unsharded parameters and hence the dtype for
            forward/backward computation and the parameter all-gather. Forward
            input casting also uses this dtype. If this is ``None``, then the
            unsharded parameters use their original dtype. The optimizer step
            uses the sharded parameters in the original dtype. (Default:
            ``None``)
        reduce_dtype (Optional[torch.dtype]): The dtype for unsharded gradients
            and gradient reduction (reduce-scatter or all-reduce).
            FSDP sets the unsharded parameter's ``grad_dtype`` to this dtype, so
            autograd produces and accumulates gradients in this dtype regardless
            of whether gradient synchronization is enabled. FSDP packs these
            gradients without casting before reduction. If ``None``, this uses
            the compute dtype. Reduced sharded gradients use each parameter's
            ``grad_dtype`` as specified before calling :func:`fully_shard`.
            (Default: ``None``)
        output_dtype (Optional[torch.dtype]): This specifies the dtype for
            casting floating-point forward outputs. This can be used to
            help implement cases where different modules have different mixed
            precision policies. (Default: ``None``)
        cast_forward_inputs (bool): This specifies whether FSDP should cast the
            forward's floating-point input tensors to ``param_dtype`` or not.
            For grouped ``fully_shard([a, b, ...])``, the cast is applied per
            module, before each module's forward.
        param_dtype_override_fn (Optional[Callable[[nn.Parameter], Optional[torch.dtype]]]):
            Optional per-parameter override for ``param_dtype``. The callable
            is evaluated once for each managed parameter when FSDP is applied.
            Returning the parameter's original dtype preserves that parameter
            in its original dtype; returning ``None`` or ``param_dtype`` uses
            the default ``param_dtype``. Other dtypes are not supported.
            Forward input casting continues to use ``param_dtype``. If
            parameters within one group resolve to multiple compute dtypes,
            configure one common effective reduction dtype for the group.
            (Default: ``None``)
    """

    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    cast_forward_inputs: bool = True
    param_dtype_override_fn: Callable[[nn.Parameter], torch.dtype | None] | None = (
        field(default=None, kw_only=True)
    )

    def _resolve_for_param(self, param: nn.Parameter) -> "MixedPrecisionPolicy":
        if self.param_dtype_override_fn is None:
            return self
        param_dtype = self.param_dtype
        if self.param_dtype_override_fn is not None:
            param_dtype_override = self.param_dtype_override_fn(param)
            if param_dtype_override is not None:
                if not isinstance(param_dtype_override, torch.dtype):
                    raise ValueError(
                        "param_dtype_override_fn must return a torch.dtype or None but got "
                        f"{type(param_dtype_override)}"
                    )
                if param_dtype_override not in (self.param_dtype, param.dtype):
                    raise ValueError(
                        "param_dtype_override_fn must return None, param_dtype, or the "
                        "parameter's original dtype but got "
                        f"{param_dtype_override} for a parameter with dtype "
                        f"{param.dtype} and param_dtype {self.param_dtype}"
                    )
                param_dtype = param_dtype_override
        return replace(
            self,
            param_dtype=param_dtype,
            param_dtype_override_fn=None,
        )


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
