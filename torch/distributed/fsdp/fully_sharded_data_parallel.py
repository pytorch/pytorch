import contextlib
import functools
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from math import inf
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Generator,
    NamedTuple,
    Set,
    Tuple,
    Union,
    cast,
)

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributed import ProcessGroup
from torch.distributed._sharded_tensor import (
    init_from_local_shards,
    Shard,
    ShardedTensor,
)
from torch.distributed.distributed_c10d import _get_default_group
from torch.nn.parameter import Parameter

from .flatten_params_wrapper import FlatParameter, FlattenParamsWrapper, FLAT_PARAM
from .utils import (
    _apply_to_tensors,
    _replace_by_prefix,
)
from .wrap import _recursive_wrap

if TYPE_CHECKING:
    from collections import OrderedDict  # noqa: F401


FSDP_WRAPPED_MODULE = "_fsdp_wrapped_module"


@dataclass
class CPUOffload:
    """
    CPU offlaoding config. Currently, only parameter and gradient CPU
    offload are supported.
    offload_params: Offloading parameters to CPUs when these parameters are
                    not used for computation on GPUs. This implicitly enables
                    gradient offloading to CPUs in order for parameters and
                    gradients to be on the same device to work with optimizer.
    """

    offload_params: bool = False
    # TODO: state dict offloading
    # https://github.com/pytorch/pytorch/issues/67224


class BackwardPrefetch(Enum):
    """
    Specify where to prefetch next layer's full parameters
    during backward pass.
    BACKWARD_PRE: prefetch right before current layer's backward computation
                  starts, this approach will increase backward communication
                  and computation overalpping and potentialy improve training
                  performance, but it may increase the peak memory usage as
                  the prefetched full parameters will be kept in the GPU memory
                  until next layer's backward computation is done.
    BACKWARD_POST: prefetch right after current layer's backward computation finishes,
                   this approach will not increase peak memory as prefetching happens
                   after current layer's full parameters are freed.
                   It could potentially improve backward communication and computation
                   overlapping as it avoids all_gather and reduce_scatter are blocked
                   each other in the single NCCL stream. However, based on our experiments,
                   for some models, the backward post backward hook fire order is not always
                   the reversed forward computation order, so this
                   approach may prefetch full parameters for layers ahead of next layer,
                   this 'ahead' all_gather could delay next layer's all_gather in the
                   single NCCL stream and cause the next layer's computation delay. So it may
                   cause some performance regession for some models.
    """

    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    # TODO, BACKWARD_PRE_CPU, prefetch full parameters and keep them in the CPU memory


class TrainingState_(Enum):
    """
    Simple enum to indicate what state FSDP is in. Used for asserting
    to make sure APIs are called in the correct state.
    ..note::
        ``BACKWARD_PRE`` and ``BACKWARD_POST`` states are used to ensure we
        receives backward hooks in the correct order. It is used to catch
        unexpected order of hooks being called (likely due to our
        hook registration logic or autograd engine logic changes).
    """

    IDLE = auto()
    FORWARD = auto()
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    SUMMON_FULL_PARAMS = auto()


class StateDictType(Enum):
    """
    This enum indicates that which type of ``state_dict`` the FSDP module is
    currently processing (returning or loading).
    The default value is FULL_STATE_DICT to comply the PyTorch convention.
    ..note::
        FSDP currently supports two types of ``state_dict``:
            1. ``state_dict/load_state_dict`: this pair of APIs return and load
               the non-sharded, unflattened parameters. The semantics is the
               same as using DDP.
            2. ``_local_state_dict/_load_local_state_dict``: this pair of APIs return
               and load local sharded, flattened parameters. The values returned
               by ``_local_state_dict`` can be directly used by FSDP and is only
               meaningful to FSDP (because parameters are flattened). Note that
               these APIs are meant for use via the :func:`state_dict_type`
               context manager as follows:
                   >>> with fsdp.state_dict_type(StateDictType.LOCAL_STATE_DICT):
                   >>>     state = fsdp.state_dict()  # loads local state dict
            3. [Planned for future support] ``sharded_state_dict/load_sharded_state_dict``: this pair of APIs
               return and load sharded, unflattened parameters. The ``state_dict``
               return by ``sharded_state_dict`` can be used by all other parallel
               schemes (resharding may be required).
    """

    FULL_STATE_DICT = auto()
    LOCAL_STATE_DICT = auto()
    SHARDED_STATE_DICT = auto()


class FullyShardedDataParallel(nn.Module):
    """
    A wrapper for sharding Module parameters across data parallel workers. This
    is inspired by `Xu et al.`_ as well as the ZeRO Stage 3 from DeepSpeed_.
    FullyShardedDataParallel is commonly shorten to FSDP.

    .. _`Xu et al.`: https://arxiv.org/abs/2004.13336
    .. _DeepSpeed: https://www.deepspeed.ai/

    Example::

        >>> import torch
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> torch.cuda.set_device(device_id)
        >>> sharded_module = FSDP(my_module)
        >>> optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
        >>> x = sharded_module(x, y=3, z=torch.Tensor([1]))
        >>> loss = x.sum()
        >>> loss.backward()
        >>> optim.step()

    .. warning::
        The optimizer must be initialized *after* the module has been wrapped,
        since FSDP will shard parameters in-place and this will break any
        previously initialized optimizers.

    .. warning::
        Module should be already placed on the destination device or
        device is set properly using torch.cuda.set_device(device_id).
        FSDP will get compute device from module first, if module device
        is CPU, FSDP will then get compute device from current device.

    .. warning::
        FSDP currently does not support gradient accumulation outside
        `no_sync()` when using CPU offloading. Trying to do so yields incorrect
        results since FSDP will use the newly-reduced gradient instead of
        accumulating with any existing gradient.

    Args:
        module (nn.Module):
            module to be wrapped with FSDP.
        process_group (Optional[ProcessGroup]):
            process group for sharding
        cpu_offload (Optional [CPUOffload]):
            CPU offloading config. Currently, only parameter and gradient CPU
            offload is supported. It can be enabled via passing in
            ``cpu_offload=CPUOffload(offload_params=True)``. Note that this
            currently implicitly enables gradient offloading to CPU in order for
            params and grads to be on same device to work with optimizer. This
            API is subject to change. Default is ``None`` in which case there
            will be no offloading.
        fsdp_auto_wrap_policy: (Optional [callable]):
            A callable specifying a policy to recursively wrap layers with FSDP.
            Note that this policy currently will only apply to child modules of
            the passed in module. The remainder modules are always wrapped in
            the returned FSDP root instance.
            ``default_auto_wrap_policy`` written in ``torch.distributed.fsdp.wrap`` is
            an example of ``fsdp_auto_wrap_policy`` callable, this policy wraps layers
            with parameter sizes larger than 100M. Users can supply the customized
            ``fsdp_auto_wrap_policy`` callable that should accept following arguments:
            ``module: nn.Module``, ``recurse: bool``, ``unwrapped_params: int``,
            extra customized arguments could be added to the customized
            ``fsdp_auto_wrap_policy`` callable as well.

            Example::

                >>> def custom_auto_wrap_policy(
                >>>     module: nn.Module,
                >>>     recurse: bool,
                >>>     unwrapped_params: int,
                >>>     # These are customizable for this policy function.
                >>>     min_num_params: int = int(1e8),
                >>> ) -> bool:
                >>>     return unwrapped_params >= min_num_params

        backward_prefetch: (Optional[BackwardPrefetch]):
            This is an experimental feature that is subject to change in the
            the near future. It allows users to enable two different backward_prefetch
            algorithms to help backward communication and computation overlapping.
            Pros and cons of each algorithm is explained in the class ``BackwardPrefetch``.
    """

    def __init__(
        self,
        module: nn.Module,
        process_group: Optional[ProcessGroup] = None,
        cpu_offload: Optional[CPUOffload] = None,
        fsdp_auto_wrap_policy: Optional[Callable] = None,
        backward_prefetch: Optional[BackwardPrefetch] = None,
    ):
        torch._C._log_api_usage_once("torch.distributed.fsdp")
        super().__init__()
        # if fsdp_auto_wrap_policy is specified, submodules should not be
        # already wrapped, otherwise we'd attempt to double wrap them resulting
        # in errors.
        if fsdp_auto_wrap_policy is not None:
            self._check_wrapped(
                module,
                check_fn=lambda mod: not isinstance(mod, FullyShardedDataParallel),
                err_fn=lambda mod: f"Expected {mod} to NOT be FullyShardedDataParallel if auto_wrap is enabled.",
            )
            _recursive_wrap(
                module,
                auto_wrap_policy=fsdp_auto_wrap_policy,
                wrapper_cls=FullyShardedDataParallel,
                # Note that we have the recursive_wrap skip wrapping for
                # the outermost (this) module otherwise it will result in a
                # double-wrap causing issues.
                only_wrap_children=True,
                # FSDP arguments follow.
                process_group=process_group,
                cpu_offload=cpu_offload,
                backward_prefetch=backward_prefetch,
                # Note that recursive_wap should not call FSDP with wrapping
                # enabled, as this recursive call handles all wrapping,
                # including for nested children.
                fsdp_auto_wrap_policy=None,
            )

        self.process_group = process_group or _get_default_group()
        self.rank = self.process_group.rank()
        self.world_size = self.process_group.size()
        # device for computation, if module is on GPU, use module.device;
        # if module is on CPU, use current device;
        self.compute_device = _get_default_cuda_device(module)

        # Free full params and keep shard only after forward
        self.reshard_after_forward = True

        # setting two factors to avoid underflow and overflow
        self.gradient_predivide_factor: float = self._get_gradient_predivide_factor(
            self.world_size
        )
        self.gradient_postdivide_factor: float = (
            self.world_size / self.gradient_predivide_factor
        )

        self.numel_padded_per_param: List[int] = []
        self.cpu_offload = cpu_offload or CPUOffload()
        self.backward_prefetch = backward_prefetch

        # Only handle params which are not already sharded. This enables
        # sharding individual layers of a Module, with an outer wrapper to
        # shard any leftover parameters.
        params = []
        for param_name, param in module.named_parameters():
            if not isinstance(param, FlatParameter):
                params.append(param)

        self._fsdp_wrapped_module: FlattenParamsWrapper = FlattenParamsWrapper(
            module, param_list=params
        )
        assert getattr(self, FSDP_WRAPPED_MODULE) is self._fsdp_wrapped_module
        del module  # free original module in case it helps garbage collection
        if self._fsdp_wrapped_module.flat_param is not None:
            self.params = [self._fsdp_wrapped_module.flat_param]
        else:
            self.params = []

        # Shard module parameters in place
        self._shard_parameters()

        # Make sure all parameters are sharded.
        for n, p in self.named_parameters():
            if not isinstance(p, FlatParameter):
                raise RuntimeError(
                    f"found unsharded parameter: {n} ; {p.size()} {p.__class__}"
                )
        self._reset_lazy_init()

        # Flag indicating if we require gradient reduction in the backward
        # pass (set to `False` in the `no_sync()` context manager)
        self._require_backward_grad_sync: bool = True

        # Enum to indicate if we're in the forward/backward pass, idle, etc.
        self.training_state = TrainingState_.IDLE

        self._state_dict_type = StateDictType.FULL_STATE_DICT

        # FSDP currently provides three different state_dicts. The actual
        # state_dict that will be saved/loaded is decided by
        # self._state_dict_type. And the main logic of each state_dict is
        # implemented in the hook. Therefore, for each hook (post-save and
        # pre-load), there is a dispatcher dictionary to dispatch the execution
        # flow to the correct implementation.
        self._register_state_dict_hook(self._post_state_dict_hook)
        self._post_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: self._full_post_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: self._local_post_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: self._sharded_post_state_dict_hook,
        }
        self._register_load_state_dict_pre_hook(
            self._pre_load_state_dict_hook, with_module=True
        )
        self._pre_load_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: self._full_pre_load_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: self._local_pre_load_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: self._sharded_pre_load_state_dict_hook,
        }

        # Flag to guard against preparing gradients multiple times per backward pass.
        self._pre_backward_hook_has_run = False
        # Used for prefetching all gather full params in post backward hook
        self._need_rebuild_full_params = False

        # If specified, offload parameter shard to CPU.
        if self.cpu_offload.offload_params:
            for p in self.params:
                self._offload_to_cpu(p)

    @classmethod
    def _check_wrapped(cls, begin_module, check_fn, err_fn):
        for _, mod in begin_module.named_modules():
            if not check_fn(mod):
                raise ValueError(err_fn(mod))

    @property
    def module(self) -> FlattenParamsWrapper:
        """make model.module accessible, just like DDP."""
        assert isinstance(self._fsdp_wrapped_module, FlattenParamsWrapper)
        return self._fsdp_wrapped_module

    def check_is_root(self) -> bool:
        self._lazy_init()
        assert self._is_root is not None
        return self._is_root

    @staticmethod
    def fsdp_modules(
        module: nn.Module, root_only: bool = False
    ) -> List["FullyShardedDataParallel"]:
        """
        Helper function to return all nested FSDP instances, including self.

        Args:
            module: the root module. This module does not have to be a FSDP module.
            root_only: whether to return only root FSDP modules (default: False).

        Returns:
            fsdp_modules: the FSDP modules that are nested in the input module.
        """
        fsdp_modules = []
        for sub_module in module.modules():
            if isinstance(sub_module, FullyShardedDataParallel):
                if not root_only or sub_module.check_is_root():
                    fsdp_modules.append(sub_module)

        return fsdp_modules

    def apply(self, fn: Callable[[nn.Module], None]) -> "FullyShardedDataParallel":
        r"""Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
        as well as self. Typical use includes initializing the parameters of a model
        (see also :ref:`nn-init-doc`).

        Compared to ``torch.nn.Module.apply``, this version additionally gathers
        the full parameters before applying ``fn``. It should not be called from
        within another ``summon_full_params`` context.

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self
        """
        uninitialized = self._is_root is None
        self._assert_state(TrainingState_.IDLE)
        with self.summon_full_params(recurse=False, writeback=True):
            ret = super().apply(fn)

        # Reset lazy init that might be called by summon_full_params, since
        # it could have set is_root incorrectly for non-root FSDP instances.
        if uninitialized and self._is_root:
            for module in self.fsdp_modules(self):
                module._reset_lazy_init()

        return ret

    # setting two factors 'self.gradient_predivide_factor'
    # and 'self.gradient_postdivide_factor' to avoid underflow and overflow
    def _get_gradient_predivide_factor(self, world_size: int) -> float:
        factor: int = 1
        while world_size % factor == 0 and world_size / factor > factor:
            factor *= 2
        return float(factor)

    def _offload_to_cpu(self, p):
        """
        Offloads parameter to CPU from self.compute_device. If the parameter is
        already on CPU then this is a noop.
        """
        cpu_device = torch.device("cpu")
        if p.device == cpu_device:
            return
        with torch.no_grad():
            p.data = p.to(cpu_device)

    def _cast_buffers(
        self, device: Optional[torch.device] = None, memo: Optional[Set] = None
    ) -> None:
        """Move all buffers to the given *device*.
        If *device* is not given, then it will default to
        ``self.compute_device``. In the
        case of nested FSDP instances, we will respect the child instance's
        ``compute_device`` configuration.
        Args:
            device (torch.device, Optional):
                device to cast buffers to (defaults to compute_device)
            memo (Set, Optional):
                set of modules that have already been processed
        """
        if memo is None:
            memo = set()
        for module in self.modules():
            if module is not self and isinstance(module, FullyShardedDataParallel):
                # Allow any child FSDP instances to handle their own buffers.
                module._cast_buffers(device=device, memo=memo)
            elif module not in memo:
                memo.add(module)
                for name, buf in module.named_buffers(recurse=False):
                    if buf is None:
                        continue
                    buf = buf.to(device=device or self.compute_device)
                    setattr(module, name, buf)

    @torch.no_grad()
    def _shard_parameters(self) -> None:
        """
        At initialization we wrap a module with full parameters and shard the
        parameters in-place. Sharding is implemented by viewing each parameter
        as a 1D Tensor and retaining only a single slice, where the slice size
        is determined by the number of data parallel workers.
        After this initial sharding is complete, the user can initialize a
        ``torch.optim.Optimizer`` in the usual way, i.e.::
        .. code-block:: python
            optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
        The optimizer will see only a single slice of parameters and will thus
        allocate less memory for optimizer state, avoiding redundancy across
        data parallel workers.
        """
        for p in self.params:
            assert not p._is_sharded, "Param should have not been sharded yet."
            assert (
                p.is_floating_point()
            ), "Autograd does not support operations for integer type."

            # Sharding is done only when world_size is larger than 1.
            p._is_sharded = self.world_size > 1  # type: ignore[attr-defined]
            p._orig_size = p.size()  # type: ignore[attr-defined]

            if not p._is_sharded:  # type: ignore[attr-defined]
                self.numel_padded_per_param.append(0)
                continue

            # Save the original storage and free it later on.
            # Since we're modifying the tensor's storage directly,
            # make sure the tensor is the sole occupant of the storage.
            assert (
                p.storage_offset() == 0
            ), "The tensor is not the sole occupant of the storage."
            orig_storage = p.storage()

            # Replace p with the relevant shard.
            local_shard, num_padded = self._get_shard(p)
            p.set_(local_shard)  # type: ignore[call-overload]
            p.shard_by_offsets(
                self.rank * local_shard.numel(),
                (self.rank + 1) * local_shard.numel() - 1,
                num_padded,
            )
            self.numel_padded_per_param.append(num_padded)

            # Free storage that contains the original full data.
            if orig_storage.size() > 0:
                orig_storage.resize_(0)  # type: ignore[attr-defined]

        assert len(self.numel_padded_per_param) == len(
            self.params
        ), "numel_padded_per_param is not populated correctly."

    def _get_shard(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Return the local shard of a full tensor."""
        # Shard using torch.chunk to match all-gather/reduce-scatter.
        chunks = torch.flatten(tensor).chunk(self.world_size)
        if len(chunks) < (self.rank + 1):
            # If there are not enough chunks to shard across ranks, create an
            # empty chunk that will just be padded with zeros to be the
            # appropriate size.
            chunk = chunks[0].new_empty(0)
        else:
            chunk = chunks[self.rank]
        # Determine number of padding elements.
        num_to_pad = chunks[0].numel() - chunk.numel()
        assert (
            num_to_pad >= 0
        ), "Chunk's size should be equal or smaller than \
            the first chunk's size."

        # We always need to clone here, because regardless of padding the
        # original parameter, of which this chunk is a view of, is deallocated
        # after _get_shard.
        shard = chunk.clone()
        if num_to_pad > 0:
            shard = F.pad(shard, [0, num_to_pad])
        return shard, num_to_pad

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.module, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
        return self.module.__getitem__(key)  # type: ignore[operator]

    def _reset_lazy_init(self) -> None:
        """
        Reset instance so :func:`_lazy_init` will run on the next forward.
        Currently this is only called in __init__
        """
        self._is_root: Optional[bool] = None
        self._streams: Dict[str, torch.cuda.Stream] = {}
        self._fsdp_graph_order: List[nn.Module] = []
        self._my_fsdp_idx_in_graph: Optional[int] = None
        for p in self.params:
            if hasattr(p, "_local_shard"):
                # reset attributes that are added in _init_param_attributes, as
                # part of _lazy_init
                del p._local_shard  # type: ignore[attr-defined]

    def _lazy_init(self) -> None:
        """Initialization steps that should happen lazily, typically right
        before the first forward pass.
        """
        # Initialize param attributes lazily, in case the param's dtype or
        # device changes after __init__.
        for p in self.params:
            self._init_param_attributes(p)

        # Initialize _is_root and setup streams. These steps would ideally
        # happen in __init__, but _is_root can only be determined after the
        # entire model hierarchy is setup, thus we run it lazily.
        if self._is_root is None:
            # _is_root means that we are in the outermost module's forward.
            self._set_is_root()
            self._setup_streams()

        if self._is_root:
            # Buffers stay on GPU, and don't get sharded. Since _cast_buffers
            # applies recursively, we only call this from the root instance.
            self._cast_buffers()

            # Don't free the full params for the outer-most (root) instance,
            # In most cases, root instance contains params in the last layers
            # or has no params. In these cases, those params will be needed
            # immediately after for the backward pass. Note that this only
            # applies currently when freeing parameters at end of layer's
            # forward pass.
            self.reshard_after_forward = False

            # Due to the use of streams, we need to make sure the previous
            # ``optim.step()`` is done before we all-gather parameters.
            self._wait_for_previous_optim_step()

    @torch.no_grad()
    def _init_param_attributes(self, p: Parameter) -> None:
        """
        We manage several attributes on each Parameter instance. The first two
        are set by :func:`_shard_parameters`:
            ``_is_sharded``: ``True`` if the Parameter is sharded or ``False``
                if the Parameter is intentionally not sharded (in which case we
                will all-reduce grads for this param). Currently the only way
                `_is_sharded = False` is if world_size = 1.
            ``_orig_size``: the size of the original Parameter (before sharding)
        A few attributes are set here:
            ``_local_shard``: a single shard of the parameter. This is needed to
                recover the shard after rebuilding full parameter in forward
                and backward.
            ``_full_param_padded``: the full weight (padded to be evenly
                divisible by ``world_size``), used for computation in the
                forward and backward pass. It is initialized with the
                appropriate size and then has its storage freed. This will be
                resized in place and only materialized (via all-gather) as needed.
        Another attribute is set by :func:`_register_post_backward_hooks`:
            ``_shard_bwd_hook``: it holds the parameter's AccumulateGrad object
                and the registered post hook handle.
        """
        assert hasattr(p, "_is_sharded") and hasattr(
            p, "_orig_size"
        ), "Parameters should have been sharded during construction."
        # If _local_shard has been set in the first lazy init and
        # current parameter is pointed to _local_shard, no need to
        # set the _local_shard again.
        if hasattr(p, "_local_shard"):
            # If CPU offloading, p._local_shard should have been placed on CPU
            # during its first lazy construction.
            if self.cpu_offload.offload_params:
                assert p._local_shard.device == torch.device(  # type: ignore[attr-defined]
                    "cpu"
                ), (
                    "Expected p._local_shard to be on CPU, "  # type: ignore[attr-defined]
                    f"but it's on {p._local_shard.device}"  # type: ignore[attr-defined]
                )
            return

        # A single shard of the parameters. Also makes p._local_shard to be on
        # CPU if we are CPU offloading, since p.data would be on CPU during
        # init.
        if self.cpu_offload.offload_params:
            assert p.device == torch.device("cpu"), (
                "Expected param to be on CPU when cpu_offloading is enabled. "
                "If CPU offloading is enabled correctly, you may be "
                "accidentally moving the model to CUDA after FSDP initialization."
            )
        p._local_shard = p.data  # type: ignore[attr-defined]
        # If CPU offloading, pin the memory to enable faster CPU -> GPU device
        # transfer.
        if self.cpu_offload.offload_params:
            assert p._local_shard.device == torch.device("cpu")  # type: ignore[attr-defined]
            p._local_shard.pin_memory()  # type: ignore[attr-defined]
            # When offloading parameters, also move the grad shard to CPU during
            # backward pass. In this case, it's important to pre-allocate the
            # CPU grad shard in pinned memory so that we can do a non-blocking
            # transfer.
            p._cpu_grad = torch.zeros_like(  # type: ignore[attr-defined]
                p, device=torch.device("cpu")
            ).pin_memory()

        # We also maintain a full-sized parameter of type self.compute_dtype.
        # We resize the storage to size 0 at init (here) and only materialize
        # as needed. The storage may contain padding elements so that it is
        # evenly divisible by world_size, although these padding elements will
        # be removed before the relevant computation.
        if p._is_sharded:  # type: ignore[attr-defined]
            p._full_param_padded = torch.zeros(  # type: ignore[attr-defined]
                p.numel() * self.world_size,
                device=self.compute_device,
                dtype=p.dtype,
            )
            _free_storage(p._full_param_padded)  # type: ignore[attr-defined]

    def _set_is_root(self) -> None:
        """If ``True``, implies that no other :class:`FullyShardedDataParallel`
        instance wraps this one. Called once by :func:`_lazy_init`.
        """
        if self._is_root is not None:
            return
        # No FSDP instance wraps this, else _is_root would be set to False.
        self._is_root = True
        # If final backward callback is never been queued, state should be IDLE.
        # If final backward callback is queued, the callback should be finished
        # and the state was reset to be IDLE.
        # This should be asserted at the beginning of forward pass in the root instance only.
        # For children instances, if they are checkpointed, state will not be reset to
        # IDLE after each inner forward/backward.
        self._assert_state(TrainingState_.IDLE)
        for n, m in self.named_modules():
            # `n != ""` excludes self.
            if n != "" and isinstance(m, FullyShardedDataParallel):
                # We relax the assert for non-root instance, when the nested initialized module is wrapped
                # again in FSDP later, for example after training to run inference.
                assert (
                    m._is_root is None or not m._is_root
                ), "Non-root instance's _is_root flag should have not been set yet \
                    or has already been set as False."
                if m._is_root is None:
                    m._is_root = False

    def _setup_streams(self) -> None:
        """Create streams to overlap data transfer and computation."""
        if len(self._streams) > 0 or not self._is_root:
            return

        if torch.cuda.is_available():
            # Stream for all-gathering parameters.
            self._streams["all_gather"] = torch.cuda.Stream()
            # Stream for overlapping grad reduction with the backward pass.
            self._streams["post_backward"] = torch.cuda.Stream()

        # We share streams with all children instances, which allows them to
        # overlap transfers across the forward pass without synchronizing with
        # the default stream.
        for n, m in self.named_modules():
            if n != "" and isinstance(m, FullyShardedDataParallel):
                m._streams = self._streams
                m._fsdp_graph_order = self._fsdp_graph_order

    def _wait_for_previous_optim_step(self) -> None:
        """
        The outer-most :class:`FullyShardedDataParallel` instance (i.e., the root
        instance) needs to synchronize with the default stream to ensure the
        previous optimizer step is done.
        """
        if not torch.cuda.is_available():
            return
        self._streams["all_gather"].wait_stream(torch.cuda.current_stream())

    def _need_prefetch_pre_backward_hook(self) -> bool:
        if (
            self.backward_prefetch == BackwardPrefetch.BACKWARD_PRE
            and self._fsdp_graph_order is not None
            and self._my_fsdp_idx_in_graph is not None
            and self._my_fsdp_idx_in_graph > 0
            and self._fsdp_graph_order[self._my_fsdp_idx_in_graph - 1].training_state
            != TrainingState_.BACKWARD_POST
        ):
            return True
        else:
            return False

    def _need_prefetch_post_backward_hook(self) -> bool:
        if (
            self.backward_prefetch == BackwardPrefetch.BACKWARD_POST
            and self._fsdp_graph_order is not None
            and self._my_fsdp_idx_in_graph is not None
            and self._my_fsdp_idx_in_graph > 0
            and self._fsdp_graph_order[self._my_fsdp_idx_in_graph - 1].training_state
            != TrainingState_.BACKWARD_POST
            and self._fsdp_graph_order[
                self._my_fsdp_idx_in_graph - 1
            ]._need_rebuild_full_params
        ):
            return True
        else:
            return False

    @staticmethod
    @contextlib.contextmanager
    def state_dict_type(module: nn.Module, state_dict_type: StateDictType) -> Generator:
        """
        A context manager to set the state_dict_type of all the descendant FSDP
        modules of the target module. The target module does not have to be a FSDP
        module. If the target module is a FSDP module, its state_dict_type will
        also be changed.
        .. note:: This API should be called for only the top-level (root) module.
        .. note:: The default state_dict_type is StateDictTyp.FULL_STATE_DICT.
        .. note:: This API enables users to transparently use the conventional
        ``state_dict`` API to take model checkpoints in cases where the root
        FSDP module is wrapped by another ``nn.Module``. For example, the
        following will ensure `state_dict`  is called on all non-FSDP instances,
        while dispatching into `local_state_dict` implementation for FSDP:
        >>> model = DDP(FSDP(...))
        >> fsdp_root = model.module
        >>> with fsdp_root.state_dict_type(StateDictType.LOCAL_STATE_DICT):
        >>>     checkpoint = model.state_dict()
        Args:
            state_dict_type (StateDictType): the desired state_dict_type to set.
        """
        prev_state_dict_type = None
        for module in FullyShardedDataParallel.fsdp_modules(module):
            if prev_state_dict_type is None:
                prev_state_dict_type = module._state_dict_type
            if prev_state_dict_type != module._state_dict_type:
                raise RuntimeError(
                    "All FSDP module should the same state_dict_type."
                )
            module._state_dict_type = state_dict_type
        try:
            yield
        finally:
            assert prev_state_dict_type is not None  # Avoid mypy warning
            for module in FullyShardedDataParallel.fsdp_modules(module):
                module._state_dict_type = prev_state_dict_type

    def _full_post_state_dict_hook(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        prefix: str,
    ) -> "OrderedDict[str, torch.Tensor]":
        """
        Hook that runs after model.state_dict() is called before returning result to
        user. For FSDP, we may have to clone the tensors in state_dict as params go
        back to sharded version after summon_full_params ends, and also remove
        "_fsdp_wrapped_module" prefix.
        """
        self._assert_state([TrainingState_.SUMMON_FULL_PARAMS])
        for key in state_dict.keys():
            # Due to recursive call of summon_full_params, avoid unnecessasry
            # reclone of tensors in case they have already been cloned.
            if (
                not getattr(state_dict[key], "_has_been_cloned", False)
            ):
                state_dict[key] = state_dict[key].clone().detach()
                state_dict[key]._has_been_cloned = True  # type: ignore[attr-defined]

        _replace_by_prefix(state_dict, prefix + f"{FSDP_WRAPPED_MODULE}.", prefix)
        return state_dict

    def _local_post_state_dict_hook(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        prefix: str,
    ) -> "OrderedDict[str, torch.Tensor]":
        """
        This hook create a ShardedTensor from the local flat_param and replace
        the state_dict[f"{prefix}{FLAT_PARAM}] with the ShardedTensor. No copy
        will happen. The underlying storage is the same.
        """
        _replace_by_prefix(state_dict, f"{prefix}{FSDP_WRAPPED_MODULE}.", prefix)
        # state_dict[f"{prefix}{FLAT_PARAM}"] exists and has the same tensor
        # value as the flat_param but it is a pure Tensor because
        # nn.Module.state_dict() will detach the parameter. Therefore, we need
        # to get flat_param from the FlattenParamsWrapper to get the metadata.
        flat_param = getattr(self.module, FLAT_PARAM, None)
        assert (
            flat_param is not None
        ), "flat_param cannot be None when doing local_state_dict."

        # Construct a ShardedTensor from the flat_param.
        full_numel = flat_param.full_numel
        shard_offset = flat_param.numel() * self.rank
        valid_data_size = flat_param.numel() - flat_param.num_padded
        if valid_data_size > 0 and flat_param.num_padded > 0:
            flat_param = flat_param.narrow(0, 0, valid_data_size)
        local_shards = [
            Shard.from_tensor_and_offsets(flat_param, [shard_offset], self.rank)
        ]
        state_dict[f"{prefix}{FLAT_PARAM}"] = init_from_local_shards(
            local_shards, full_numel, process_group=self.process_group
        )  # type: ignore[assignment]

        return state_dict

    def _sharded_post_state_dict_hook(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        prefix: str,
    ) -> "OrderedDict[str, torch.Tensor]":
        raise NotImplementedError("Will be implemented as part of https://github.com/pytorch/pytorch/issues/73518")

    @staticmethod
    def _post_state_dict_hook(
        module: nn.Module,
        state_dict: "OrderedDict[str, torch.Tensor]",
        prefix: str,
        *args: Any,
    ) -> "OrderedDict[str, torch.Tensor]":
        """
        _post_state_dict_hook() is called after the state_dict() of this
        FSDP module is executed. ``self._state_dict_type`` is used to decide
        what postprocessing will be done.
        """
        self = cast(FullyShardedDataParallel, module)
        return self._post_state_dict_hook_fn[self._state_dict_type](state_dict, prefix)

    def state_dict(self, *args, **kwargs):
        """
        The entry point of all three FSDP state_dict APIs. By default, calling
        ``state_dict`` on an FSDP module will result in FSDP attempting to bring
        the entire (nested) model into memory and taking the local model's
        ``state_dict`` on every rank, which could result in OOM if the model
        cannot fit on a single GPU. As a result, :func:`state_dict_type` API is
        available to configure between `state_dict` implementations. User can
        thus use `with self.state_dict_type(self, StateDictType.LOCAL_STATE_DICT)`
        context manager to perform a local checkpoint that will store only local
        shards of the module. Currently, the only supported implementations are
        ``StateDictType.LOCAL_STATE_DICT`` and ``StateDictType.FULL_STATE_DICT``
        (default).

        Example::

        >>> import torch
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> from torch.distributed.fsdp import StateDictType
        >>> torch.cuda.set_device(device_id)
        >>> my_module = nn.Linear(...)
        >>> sharded_module = FSDP(my_module)
        >>> with sharded_module.state_dict_type(StateDictType.FULL_STATE_DICT):
        >>>     full_dict = sharded_module.state_dict()
        >>> full_dict.keys()
        >>> odict_keys(['weight', 'bias'])
        >>> # using local state dict
        >>> with sharded_module.state_dict_type(StateDictType.LOCAL_STATE_DICT):
        >>>     local_dict = sharded_module.state_dict()
        >>> local_dict.keys()
        >>> odict_keys(['flat_param', 'inner.flat_param'])
        >>> {hi}

        .. warning:: This needs to be called on all ranks, since synchronization
            primitives may be used.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._lazy_init()
        if self._state_dict_type == StateDictType.FULL_STATE_DICT:
            if self.training_state != TrainingState_.SUMMON_FULL_PARAMS:
                with self.summon_full_params(recurse=False, writeback=False):
                    state_dict = super().state_dict(*args, **kwargs)
            else:
                state_dict = super().state_dict(*args, **kwargs)

            # TODO: support offload to CPU in post state dict hook.
            return state_dict

        elif self._state_dict_type == StateDictType.LOCAL_STATE_DICT:
            assert getattr(self.module, FLAT_PARAM, None) is not None
            assert isinstance(self.module.flat_param, FlatParameter)
            return super().state_dict(*args, **kwargs)
        elif self._state_dict_type == StateDictType.SHARDED_STATE_DICT:
            raise NotImplementedError("Will be implemented as part of https://github.com/pytorch/pytorch/issues/73518.")
        else:
            raise ValueError(f"Unknown StateDictType {self._state_dict_type}.")

    def _local_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        """
        Returns the local state of the module. Parameters are flattened and
        sharded, so the resulting state_dict can only be loaded after the module
        has been wrapped with FSDP.
        """
        with self.state_dict_type(self, StateDictType.LOCAL_STATE_DICT):
            return self.state_dict(*args, **kwargs)

    def _full_pre_load_state_dict_hook(
        self,
        state_dict: Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"],
        prefix: str,
    ) -> None:
        _replace_by_prefix(state_dict, prefix, prefix + f"{FSDP_WRAPPED_MODULE}.")

    def _local_pre_load_state_dict_hook(
        self,
        state_dict: Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"],
        prefix: str,
    ) -> None:
        """
        This hook finds the local flat_param for this FSDP module from the
        state_dict. The flat_param should be a ShardedTensor. This hook converts
        the ShardedTensor to a tensor. No copy happen unless padding is required.
        """
        _replace_by_prefix(state_dict, prefix, f"{prefix}{FSDP_WRAPPED_MODULE}.")
        key = f"{prefix}{FSDP_WRAPPED_MODULE}.{FLAT_PARAM}"
        load_tensor = state_dict[key]
        assert isinstance(
            load_tensor, ShardedTensor
        ), "Tensors in local_state_dict should be ShardedTensor."

        # Convert the ShardedTensor to a Tensor.
        shards = load_tensor.local_shards()
        assert len(shards), "load_local_state_dict assume one shard per ShardedTensor."
        load_tensor = cast(torch.Tensor, shards[0].tensor)

        # Get the metada of the flat_param to decide whether to pad the loaded
        # tensor.
        flat_param = self.module.flat_param
        assert flat_param is not None
        if flat_param.num_padded not in (0, flat_param.numel()):
            assert load_tensor.numel() < flat_param.numel(), (
                f"Local shard size = {flat_param.numel()} and the tensor in "
                f"the state_dict is {load_tensor.numel()}."
            )
            load_tensor = F.pad(load_tensor, [0, flat_param.num_padded])
        state_dict[key] = load_tensor

    def _sharded_pre_load_state_dict_hook(
        self,
        state_dict: Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"],
        prefix: str,
    ) -> None:
        raise NotImplementedError("Will be implemented as part of https://github.com/pytorch/pytorch/issues/73518.")

    @staticmethod
    def _pre_load_state_dict_hook(
        module: nn.Module,
        state_dict: Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"],
        prefix: str,
        *args: Any,
    ) -> None:
        """
        ``_pre_state_dict_hook` is called before ``self._load_from_state_dict()``
        is called. ``self._state_dict_type`` is used to decide what preprocessing
        will be done.
        """
        self = cast(FullyShardedDataParallel, module)
        self._pre_load_state_dict_hook_fn[self._state_dict_type](state_dict, prefix)

    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        *args,
    ) -> NamedTuple:
        """
        The entry point of all three FSDP load_state_dict APIs. By default,
        calling ``load_state_dict`` on an FSDP module will result in FSDP
        attempting to load a "full" state_dict, i.e. a state_dict consisting of
        full, unsharded, unflattened original module parameters. This requires
        FSDP to load the full parameter context on each rank which could result
        in GPU OOM. As a result, :func:`state_dict_type` API is available to
        configure between `load_state_dict` implementations. User can thus use
        ``with self.state_dict_type(self, StateDictType.LOCAL_STATE_DICT)`` context
        manager to load a local state dict checkpoint that will restore only
        local shards of the module. Currently, the only supported
        implementations are ``StateDictType.LOCAL_STATE_DICT`` and
        ``StateDictType.FULL_STATE_DICT`` (default). Please see :func:`state_dict`
        for documentation around creating an FSDP checkpoint.

        Example::

        >>> import torch
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> from torch.distributed.fsdp import StateDictType
        >>> torch.cuda.set_device(device_id)
        >>> my_module = nn.Linear(...)
        >>> sharded_module = FSDP(my_module)
        >>> checkpoint = torch.load(PATH)
        >>> full_state_dict = checkpoint['full_state_dict']
        >>> with sharded_module.state_dict_type(StateDictType.FULL_STATE_DICT):
        >>>     sharded_module.load_state_dict(full_state_dict)
        >>> full_dict.keys()
        >>> odict_keys(['weight', 'bias'])
        >>> # using local state dict
        >>> local_state_dict = checkpoint['local_state_dict]
        >>> with sharded_module.state_dict_type(StateDictType.LOCAL_STATE_DICT):
        >>>     sharded_module.load_state_dict(local_state_dict)
        >>> local_dict.keys()
        >>> odict_keys(['flat_param', 'inner.flat_param'])

        .. warning:: This needs to be called on all ranks, since synchronization
            primitives may be used.
        """
        torch.cuda.synchronize()
        if self._state_dict_type == StateDictType.FULL_STATE_DICT:
            # Note that it needs writeback=True to persist
            with self.summon_full_params(writeback=True):
                return super().load_state_dict(state_dict, *args)

        elif self._state_dict_type == StateDictType.LOCAL_STATE_DICT:
            return super().load_state_dict(state_dict, *args)
        elif self._state_dict_type == StateDictType.SHARDED_STATE_DICT:
            raise NotImplementedError("Will be implemented as part of https://github.com/pytorch/pytorch/issues/73518.")
        else:
            raise ValueError(f"Unknown StateDictType {self._state_dict_type}.")

    def _load_local_state_dict(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        *args,
    ) -> NamedTuple:
        """
        Load states from a flatten, sharded state dictionary.
        """
        with self.state_dict_type(self, StateDictType.LOCAL_STATE_DICT):
            return self.load_state_dict(state_dict, *args)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self._lazy_init()

        # Start of a forward pass.
        self.training_state = TrainingState_.FORWARD

        # All-gather full parameters, moving them to compute_device if
        # necessary.
        self._rebuild_full_params()
        # Wait for all_gather full parameters to finish before computation
        torch.cuda.current_stream().wait_stream(self._streams["all_gather"])

        # Register backward hooks to reshard params and reduce-scatter grads.
        # These need to be re-registered every forward pass in some cases where grad_fn
        # is mutated.
        self._register_post_backward_hooks()

        outputs = self.module(*args, **kwargs)

        if self not in self._fsdp_graph_order:
            self._my_fsdp_idx_in_graph = len(self._fsdp_graph_order)
            self._fsdp_graph_order.append(self)

        if self.reshard_after_forward:
            self._free_full_params()
        # Switch to original local shards of params. We maintain this invariant throughout
        # the code, i.e., ``p.data == p._local_shard`` after each function. This
        # also ensures that after the first forward, the optimizer state will be
        # initialized with the correct dtype and (sharded) size, since optimizer
        # state is typically initialized lazily in ``optim.step()``.
        self._use_param_local_shard()

        # Register pre-backward hooks to all-gather the params for the backward
        # pass (if output's grad was needed). This won't register anything if
        # we are in eval mode.
        outputs = self._register_pre_backward_hooks(outputs)

        # Done with a forward pass.
        self.training_state = TrainingState_.IDLE

        return outputs

    @torch.no_grad()
    def _write_back_current_shard(self):
        for p in self.params:
            if not p._is_sharded:  # type: ignore[attr-defined]
                continue  # Already copied because no sharding.
            chunks = p._full_param_padded.chunk(self.world_size)  # type: ignore[attr-defined]
            assert len(chunks) > self.rank
            chunk = chunks[self.rank]
            p._local_shard.copy_(chunk)  # type: ignore[attr-defined]

    def _collect_local_params(self):
        def _is_full_param_in_use(p: Parameter):
            return p._is_sharded and p._full_param_padded.storage().size() > 0  # type: ignore[attr-defined]

        return [p for p in self.params if not _is_full_param_in_use(p)]

    @contextlib.contextmanager
    def summon_full_params(
        self, recurse: bool = True, writeback: bool = True
    ) -> Generator:
        r""" A context manager to expose full params for the current FSDP
        instance. Can be useful *after* forward/backward for a model to get
        the params for additional processing or checking.

        .. note:: This can be used on inner FSDPs.
        .. note:: This can *not* be used within a forward or backward pass. Nor
            can forward and backward be started from within this context.
        .. note:: Parameters will revert to their local shards after the context
            manager exits, storage behavior is the same as forward.
        .. note:: The full parameters can be modified, but only the portion
            corresponding to the local param shard will persist after the
            context manager exits (unless ``writeback=False``, in which case
            changes will be discarded). In the case where FSDP does not shard
            the parameters, currently only when world_size == 1, the
            modification is persisted regardless of ``writeback``.

        Args:
            recurse (bool, Optional): recursively summon all params for nested
                FSDP instances (default: True)
            writeback (bool, Optional): if ``False``, modifications to params are
                discarded after the context manager exists;
                disabling this can be slightly more efficient (default: True)
        """
        if recurse:
            with contextlib.ExitStack() as stack:
                # Summon all params for any nested FSDP instances.
                for module in self.fsdp_modules(self):
                    stack.enter_context(
                        module.summon_full_params(
                            recurse=False, writeback=writeback
                        )
                    )
                # Yield to the caller, with full params in all nested instances.
                yield
            # Exiting from the ExitStack will re-shard params.
            return
        else:
            torch.cuda.synchronize()
            self._lazy_init()
            self._assert_state([TrainingState_.IDLE])
            # Set the state so that we assert when trying to go into
            # forward/backward.
            self.training_state = TrainingState_.SUMMON_FULL_PARAMS

            currently_local_params = self._collect_local_params()
            self._rebuild_full_params()
            # Wait for all_gather to finish before computation
            torch.cuda.current_stream().wait_stream(self._streams["all_gather"])

            # FSDP now has the full flattened parameter. Unflatten it to get the
            # full parameters.
            with contextlib.ExitStack() as stack:
                stack.enter_context(self.module.unflatten_params())
                try:
                    yield
                finally:
                    if writeback:
                        self._write_back_current_shard()
                    stack.close()
                    self._free_full_params(currently_local_params)
                    self._use_param_local_shard()
                    self.training_state = TrainingState_.IDLE

    def _register_pre_backward_hooks(self, outputs: Any) -> Any:
        """Register pre-backward hook to run before the wrapped module's
        backward. Hooks should be attached to all outputs from the forward.
        Returns:
            outputs: new outputs with hooks registered if they requires gradient.
        """
        # Reset before each backward pass
        self._need_rebuild_full_params = False

        if not torch.is_grad_enabled():
            return outputs  # don't register hooks if grad isn't enabled

        if self._is_root:
            # This actually means that only root instance has
            # _post_backward_callback_queued defined. Accidentally accessing this field
            # will assert on all other instances, giving us a nice bug checker.
            self._post_backward_callback_queued = False

        # Reset before each backward pass
        self._pre_backward_hook_has_run = False

        def _pre_backward_hook(*unused: Any) -> None:
            # Run ``_pre_backward_hook`` only once per backward pass
            if self._pre_backward_hook_has_run:
                return
            # try to queue final backward callback only once for root, so
            # that final backward callback is attached to the outer most
            # backward graph task and called after all the backward
            # calls are completed.
            if self._is_root:
                self._queue_wait_for_post_backward()

            if self._need_prefetch_pre_backward_hook():
                # Always wait for all_gather before rebuilding full params, just
                # in case full params have already been prefetched in previous layer's
                # pre-backward hook.
                torch.cuda.current_stream().wait_stream(self._streams["all_gather"])

            # All-gather full parameters, moving them to compute device if
            # necessary.
            self._rebuild_full_params()
            # Wait for all_gather to finish before computation
            torch.cuda.current_stream().wait_stream(self._streams["all_gather"])

            # Prefetch next layer's full params in backward pass,
            # since it is prefetching, no need to wait for all_gather stream.
            if self._need_prefetch_pre_backward_hook():
                self._fsdp_graph_order[self._my_fsdp_idx_in_graph - 1]._rebuild_full_params()  # type: ignore[operator]

            self._pre_backward_hook_has_run = True
            # Prepare p.grad so that it is in the right shape, device, accumulated values, etc.
            self._prep_grads_for_backward()
            # Start of a backward pass for the first time in an backward pass.
            self._assert_state([TrainingState_.IDLE])
            self.training_state = TrainingState_.BACKWARD_PRE

        def _register_hook(t: torch.Tensor) -> torch.Tensor:
            if t.requires_grad:
                t.register_hook(_pre_backward_hook)
                self._need_rebuild_full_params = True
            return t

        # Attach hooks to Tensor outputs.
        outputs = _apply_to_tensors(_register_hook, outputs)

        return outputs

    def _register_post_backward_hooks(self) -> None:
        """
        Register backward hooks to reshard params and reduce-scatter grads.
        This is called during forward pass. The goal is to attach a hook
        on each of the parameter's gradient generating function (``grad_acc``
        below) so that the hook is called *after* all gradients for that
        param are computed.
        Goals:
        1. We want the hook to fire once and only once *after* all gradients
        are accumulated for a param.
        2. If it fires more than once, we end up incorrectly shard the grad
        multiple times. (could lead to dimension too small)
        3. If it fires once but too early or doesn't fire, we leave gradients
        unsharded. (could lead to dimension too large)
        Due to multiple-pass forward, this function can be called on
        the same parameter multiple times in a single forward pass. If we register
        the hook multiple time, we end up getting called multiple times. We
        could try to get a new hook every time and delete the previous one
        registered. However, due to *unknown reason* (I have debugged it for
        a long time!), in mixed precision mode, we get two different ``grad_acc``
        objects below during different calls of this function (in the same
        forward pass). If we keep the last one, the hook end up firing too
        early. In full precision mode, we luckily get the *same* ``grad_acc``
        object, so deleting and re-registering still ensured the hook fire
        once after all gradients are generated.
        Empirically, keep the first hook register per forward pass seems to
        work the best. We do need to remove the hook at the end of the
        backward pass. Otherwise, the next forward pass will not register
        a new hook, which is needed for a new forward pass.
        """
        if not torch.is_grad_enabled():
            return  # don't register grad hooks if grad isn't enabled
        for p in self.params:
            if p.requires_grad:
                if hasattr(p, "_shard_bwd_hook"):
                    continue
                # Register a hook on the first call, empirically, autograd
                # fires it at the end for this param, which makes sense.
                p_tmp = p.expand_as(p)  # Get a grad_fn on p_tmp.
                assert (
                    p_tmp.grad_fn is not None
                ), "p_tmp grad_fn should not be None, it is used to access \
                    p's AccumulateGrad object and register post hook on it."
                grad_acc = p_tmp.grad_fn.next_functions[0][
                    0
                ]  # Gets its AccumulateGrad object.
                handle = grad_acc.register_hook(
                    functools.partial(self._post_backward_hook, p)
                )
                p._shard_bwd_hook = (grad_acc, handle)  # type: ignore[attr-defined]

    @torch.no_grad()
    def _post_backward_hook(self, param: Parameter, *unused: Any) -> None:
        """
        At the start of :func:`_post_backward_hook`, ``param.grad`` contains the
        full gradient for the local batch. The reduce-scatter op will replace
        ``param.grad`` with a single shard of the summed gradient across all
        GPUs. This shard will align with the current GPU rank. For example::
            before reduce_scatter:
                param.grad (GPU #0): [1, 2, 3, 4]
                param.grad (GPU #1): [5, 6, 7, 8]
            after reduce_scatter:
                param.grad (GPU #0): [6, 8]    # 1+5, 2+6
                param.grad (GPU #1): [10, 12]  # 3+7, 4+8
        The local GPU's ``optim.step`` is responsible for updating a single
        shard of params, also corresponding to the current GPU's rank. This
        alignment is created by :func:`_shard_parameters`, which ensures that
        the local optimizer only sees the relevant parameter shard.
        """
        # First hook callback will see PRE state. If we have multiple params,
        # then subsequent hook callbacks will see POST state.
        self._assert_state([TrainingState_.BACKWARD_PRE, TrainingState_.BACKWARD_POST])
        self.training_state = TrainingState_.BACKWARD_POST
        if param.grad is None:
            return

        if param.grad.requires_grad:
            raise RuntimeError(
                "FSDP only works with gradients that don't require gradients"
            )

        self._free_full_params(cast(List[FlatParameter], [param]))

        # Switch to local shard after backward.
        self._use_param_local_shard(cast(List[FlatParameter], [param]))

        # Prefetch previous layer's full params in backward pass post backward hook,
        # If next layer's backward computation is done and full params are freed,
        # no need to prefetch the full params again.
        # Only prefetch full params if any of the next layer's outputs requires grad
        if self._need_prefetch_post_backward_hook():
            self._fsdp_graph_order[self._my_fsdp_idx_in_graph - 1]._rebuild_full_params()  # type: ignore[operator]
            # Next layer's computation will start right after this all_gather,
            # Wait for all_gather to finish before computation.
            torch.cuda.current_stream().wait_stream(self._streams["all_gather"])

        if not self._require_backward_grad_sync:
            return

        # Wait for all work in the current stream to finish, then start the
        # reductions in post_backward stream.
        self._streams["post_backward"].wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self._streams["post_backward"]):
            orig_grad_data = param.grad.data

            if self.gradient_predivide_factor > 1:
                # Average grad by world_size for consistency with PyTorch DDP.
                param.grad.div_(self.gradient_predivide_factor)

            grad = param.grad.data
            if param._is_sharded:  # type: ignore[attr-defined]
                # We clear `param.grad` to permit repeated gradient
                # computations when this FSDP module is called multiple times.
                # This is to avoid a race among multiple re-entrant backward
                # passes. For example, the second backward pass computation
                # precedes ahead of the first backward pass reduction, which is
                # possible since the reduction is in a different stream and is
                # async. Then, the first backward pass may be incorrectly
                # reducing the second backward pass's `param.grad`.
                # The reduced gradients are accumulated in
                # `param._saved_grad_shard`, and the gradient reductions can
                # happen in arbitrary order, though we tolerate this due to the
                # (approximate) commutativity of floating-point addition.
                param.grad = None
                grad_flatten = torch.flatten(grad)
                chunks = list(grad_flatten.chunk(self.world_size))
                num_pad = self.world_size * chunks[0].numel() - grad.numel()
                input_flattened = F.pad(grad_flatten, [0, num_pad])
                output = torch.zeros_like(chunks[0])
                dist._reduce_scatter_base(
                    output, input_flattened, group=self.process_group
                )
                if self.gradient_postdivide_factor > 1:
                    # Average grad by world_size for consistency with PyTorch DDP.
                    output.div_(self.gradient_postdivide_factor)
                # To support gradient accumulation outside `no_sync()`, we save
                # the gradient data to `param._saved_grad_shard` before the
                # backward pass, accumulate gradients into it here, and set
                # `param.grad` with the accumulated value at the end of the
                # backward pass in preparation for the optimizer step.
                accumulate_grad = hasattr(param, "_saved_grad_shard")
                if accumulate_grad:
                    p_assert(
                        param._saved_grad_shard.shape == output.shape,  # type: ignore[attr-defined]
                        "Shape mismatch when accumulating gradients: "  # type: ignore[attr-defined]
                        f"existing grad shape={param._saved_grad_shard.shape} "
                        f"new grad shape={output.shape}"  # type: ignore[attr-defined]
                    )
                    p_assert(
                        param._saved_grad_shard.device == output.device,  # type: ignore[attr-defined]
                        "Device mismatch when accumulating gradients: "  # type: ignore[attr-defined]
                        f"existing grad device={param._saved_grad_shard.device} "
                        f"new grad device={output.device}"  # type: ignore[attr-defined]
                    )
                    param._saved_grad_shard += output  # type: ignore[attr-defined]
                else:
                    param._saved_grad_shard = output  # type: ignore[attr-defined]
                grad = param._saved_grad_shard  # type: ignore[attr-defined]
            else:
                # Currently the only way for _is_sharded to be False is if
                # world_size == 1. This could be relaxed in the future, e.g,
                # no sharding like PyTorch DDP, in which case grads should be
                # all-reduced here.
                assert (
                    self.world_size == 1
                ), "Currently the only way for _is_sharded to be False is \
                    world_size == 1"

            # Regardless of sharding or not, offload the grad to CPU if we are
            # offloading params. This is so param and grad reside on same device
            # which is needed for the optimizer step.
            if self.cpu_offload.offload_params:
                # We specify non_blocking=True
                # and ensure the appropriate synchronization is done by waiting
                # streams in _wait_for_post_backward.
                param._cpu_grad.copy_(  # type: ignore[attr-defined]
                    grad.detach(), non_blocking=True
                )
                # Don't let this memory get reused until after the transfer.
                grad.data.record_stream(torch.cuda.current_stream())

            # After _post_backward_hook returns, orig_grad_data will eventually
            # go out of scope, at which point it could otherwise be freed for
            # further reuse by the main stream while the div/reduce_scatter/copy
            # are underway in the post_backward stream. See:
            # github.com/NVIDIA/apex/blob/master/apex/parallel/distributed.py
            orig_grad_data.record_stream(self._streams["post_backward"])

    def _queue_wait_for_post_backward(self) -> None:
        """Try to queue a `wait_for_post_backward` callback.
        Only called on root and only queue one callback at the beginning of
        outer most backward.
        """
        assert (
            self._is_root
        ), "_queue_wait_for_post_backward can only be called on root."
        if not self._post_backward_callback_queued:
            self._assert_state([TrainingState_.IDLE])
            self._post_backward_callback_queued = True
            Variable._execution_engine.queue_callback(self._wait_for_post_backward)

    @torch.no_grad()
    def _wait_for_post_backward(self) -> None:
        """Wait for post-backward to finish. Only called on root instance."""
        assert self._is_root, "_wait_for_post_backward can only be called on root."
        # Check if the root module has params and if any of them has
        # the `requires_grad` field set. If `requires_grad=False` for
        # all the params, the post_backward hook will not fire and the
        # state will remain in `TrainingState_.BACKWARD_PRE`.
        if any([p.requires_grad for p in self.params]):
            self._assert_state(TrainingState_.BACKWARD_POST)
        else:
            self._assert_state(TrainingState_.BACKWARD_PRE)

        if self._require_backward_grad_sync:
            torch.cuda.current_stream().wait_stream(self._streams["post_backward"])
            if self.cpu_offload.offload_params:
                # We need to wait for the non-blocking GPU ->
                # CPU grad transfers to finish. We need to do this for GPU -> CPU
                # copies because when grad is on CPU, it won't wait for any CUDA
                # stream to finish GPU -> CPU copies unless we explicitly block the
                # host-side with synchronize().
                torch.cuda.current_stream().synchronize()

        # A backward pass is done, clean up below.

        def _finalize_params(fsdp_module: FullyShardedDataParallel) -> None:
            """Helper used below on all fsdp modules."""
            for p in fsdp_module.params:
                if p.requires_grad:
                    if hasattr(p, "_shard_bwd_hook"):
                        assert len(p._shard_bwd_hook) == 2 and len(  # type: ignore[attr-defined]
                            p._shard_bwd_hook  # type: ignore[attr-defined]
                        ), (  # type: ignore[attr-defined]
                            "p._shard_bwd_hook fields are not valid."
                        )
                        p._shard_bwd_hook[1].remove()  # type: ignore[attr-defined]
                        delattr(p, "_shard_bwd_hook")
                    # Preserve the gradient accumulation state if not
                    # synchronizing: `p.grad` remains the unsharded gradient
                    # accumulated from prior `no_sync()` iterations, and
                    # `p._saved_grad_shard` remains the sharded gradient from
                    # the last synchronized iteration
                    if not self._require_backward_grad_sync:
                        continue
                    # Set `p.grad` as needed to ensure optimizer correctness
                    # since optimizers operate on the `grad` attribute
                    if hasattr(p, "_cpu_grad"):
                        p_assert(
                            p.device == torch.device("cpu"),
                            f"Device mismatch: p={p.device} "  # type: ignore[attr-defined]
                            f"p._cpu_grad={p._cpu_grad}"
                        )
                        p.grad = p._cpu_grad  # type: ignore[attr-defined]
                    elif hasattr(p, "_saved_grad_shard"):
                        p_assert(
                            p.device == p._saved_grad_shard.device,  # type: ignore[attr-defined]
                            f"Device mismatch: p={p.device} "  # type: ignore[attr-defined]
                            f"p._saved_grad_shard={p._saved_grad_shard.device}"
                        )
                        p.grad = p._saved_grad_shard  # type: ignore[attr-defined]
                    else:
                        p_assert(
                            not p._is_sharded, "All sharded parameters should "
                            "use `_saved_grad_shard`"
                        )
                    if hasattr(p, "_saved_grad_shard"):
                        delattr(p, "_saved_grad_shard")

        # Update root and nested FSDP's hooks and flags.
        for m in self.modules():  # includes self
            if isinstance(m, FullyShardedDataParallel):
                _finalize_params(m)
                m._pre_backward_hook_has_run = False
                if any(p.requires_grad for p in m.parameters()):
                    # Check if the module has params and if any of them has
                    # the `requires_grad` field set. If `requires_grad=False` for
                    # all the params, the post_backward hook will not fire and the
                    # state will remain in `TrainingState_.BACKWARD_PRE`.
                    if any([p.requires_grad for p in m.params]):
                        m._assert_state(TrainingState_.BACKWARD_POST)
                    else:
                        m._assert_state(TrainingState_.BACKWARD_PRE)
                else:
                    # When `m` and its children has no params or has params but
                    # none with `requires_grad==True`, there are two cases:
                    # 1. output tensors are `requires_grad==True`. In this case,
                    # pre-backward hook is still registered, so it is in BACKWARD_PRE state.
                    # 2. output tensors are `requires_grad==False`. In this case,
                    # pre-backward hook is not registered, so it is in IDLE state.
                    m._assert_state([TrainingState_.BACKWARD_PRE, TrainingState_.IDLE])
                m.training_state = TrainingState_.IDLE

                if m._is_root:
                    # reset this flag for cases like "one forward pass + multiple backward passes"
                    self._post_backward_callback_queued = False

    @torch.no_grad()
    def _rebuild_full_params(self) -> None:
        """
        Gather all shards of params.
        """
        self._lazy_init()

        def update_p_data(output_tensor: torch.Tensor) -> None:
            """
            Helper function to update p.data pointer.
            Args:
                output_tensor (torch.Tensor): this tensor contains the data we just gathered.
            """
            p.data = output_tensor
            # Trim any padding and reshape to match original size.
            p.data = p.data[: p._orig_size.numel()].view(p._orig_size)  # type: ignore[attr-defined]

        with torch.cuda.stream(self._streams["all_gather"]):
            for p in self.params:
                if self.cpu_offload.offload_params:
                    # Move params to GPU if needed. Note that we don't use
                    # self._full_param_padded.device here because the attr is
                    # not set always, i.e. when world_size=1 and
                    # p._is_sharded = False. However when it is set, the
                    # device is always self.compute_device.
                    p.data = p.data.to(self.compute_device, non_blocking=True)
                # e.g., when world_size == 1
                if not p._is_sharded:  # type: ignore[attr-defined]
                    continue
                # If full param has been rebuilt or has not been freed, no need to call all gather
                elif (
                    p._full_param_padded.storage().size()  # type: ignore[attr-defined]
                    == p._full_param_padded.size().numel()  # type: ignore[attr-defined]
                ):
                    update_p_data(p._full_param_padded)  # type: ignore[attr-defined]
                    continue
                else:
                    # If full param has not been rebuilt or has been freed, call all gather
                    p_data = p.data  # type: ignore[attr-defined]
                    p_full_size = p._full_param_padded.size()  # type: ignore[attr-defined]
                    assert (
                        p_full_size.numel() == p_data.numel() * self.world_size
                    ), "Param full size should be equal to its shard size multiply world_size."
                    assert (
                        p._full_param_padded.storage().size() == 0  # type: ignore[attr-defined]
                    ), "Full param's storage should have been freed before if all gather is needed."  # type: ignore[attr-defined]
                    # Allocate based on full size from all shards.
                    _alloc_storage(p._full_param_padded, size=p_full_size)  # type: ignore[attr-defined]
                    output_tensor = p._full_param_padded  # type: ignore[attr-defined]

                    # Fill output_tensor with (p.data for each shard in self.world_size)
                    dist._all_gather_base(
                        output_tensor, p_data, group=self.process_group
                    )

                    # Set p.data = output_tensor (with padding trimmed)
                    update_p_data(output_tensor)

    @torch.no_grad()
    def _prep_grads_for_backward(self) -> None:
        """Make sure p.grad has the correct size/device, otherwise set it to None."""
        for p in self.params:
            if p.grad is not None and (
                p.grad.size() != p._orig_size  # type: ignore[attr-defined]
                or p.grad.device != p.device
            ):
                offloaded: bool = p.grad.device != p.device
                if offloaded:
                    assert self.cpu_offload.offload_params, \
                        "`p.grad.device` and `p.device` should be the same " \
                        "if not offloading parameters to CPU"
                prev_iter_outside_no_sync: bool = \
                    p.grad.size() == p._local_shard.shape  # type: ignore[attr-defined]
                # As long as the previous iteration was outside `no_sync()`,
                # then we must save the gradient in `_saved_grad_shard`, even
                # if the current iteration is inside `no_sync()`. This is to
                # prepare for the next iteration outside `no_sync()`, which may
                # try to accumulate gradients. FSDP accumulates gradients in
                # the separate variable `p._saved_grad_shard` to leave `p.grad`
                # for the per-iteration gradient.
                if prev_iter_outside_no_sync:
                    # FSDP currently does not support gradient accumulation
                    # outside `no_sync()` when using CPU offloading (see the
                    # warning in the class's docstring).
                    if not offloaded:
                        p._saved_grad_shard = p.grad.data  # type: ignore[attr-defined]
                p.grad = None

    @torch.no_grad()
    def _free_full_params(self, params: Optional[List[FlatParameter]] = None) -> None:
        """
        Free up storage for full parameters.
        """
        if params is None:
            params = self.params
        current_stream = torch.cuda.current_stream()
        for p in params:
            # e.g., world_size == 1
            if not p._is_sharded:  # type: ignore[attr-defined]
                continue
            # Don't let PyTorch reuse this memory until all work in the current
            # stream is complete.
            p._full_param_padded.record_stream(current_stream)  # type: ignore[attr-defined]
            # There may be external references to the Tensor Storage that we
            # can't modify, such as references that are created by
            # ctx.save_for_backward in the forward pass. Thus when we
            # unshard parameters, we should reuse the original Tensor
            # Storage object and unshard it in-place. For now, just resize
            # the Storage to 0 to save memory.
            _free_storage(p._full_param_padded)  # type: ignore[attr-defined]

    @torch.no_grad()
    def _use_param_local_shard(
        self, params: Optional[List[FlatParameter]] = None
    ) -> None:
        """Use local shard for a list of params. Also implicitly offloads
        parameters back to CPU if we are CPU offloading."""
        if params is None:
            params = self.params
        for p in params:
            if self.cpu_offload.offload_params:
                # Ensure local_shard resides in CPU if we are offloading params.
                assert p._local_shard.device == torch.device(  # type: ignore[attr-defined]
                    "cpu"
                ), "Expected p._local_shard to be on CPU"
            p.data = p._local_shard  # type: ignore[attr-defined]

    def _assert_state(self, state: Union[TrainingState_, List[TrainingState_]]) -> None:
        """Assert we are in the given state."""
        # Since assert can be turned off and this error checking
        # is really important, we use explicit error checking
        # and raise a ValueError if needed.
        if isinstance(state, TrainingState_):
            state = [state]
        if self.training_state not in state:
            msg = (
                f"expected to be in states {state} but current state "
                f"is {self.training_state}"
            )
            # In case we are failing in the context of autograd hook, asserting
            # may not generate useful msg. So, let's print it to be sure.
            if self.rank == 0:
                print(f"Asserting FSDP instance is: {self}")
                print(f"ERROR: {msg}")
                traceback.print_stack()
            raise ValueError(msg)

    @contextmanager
    def no_sync(self) -> Generator:
        """
        A context manager to disable gradient synchronizations across FSDP
        instances. Within this context, gradients will be accumulated in module
        variables, which will later be synchronized in the first
        forward-backward pass after exiting the context. This should only be
        used on the root FSDP instance and will recursively apply to all
        children FSDP instances.

        .. note:: This likely results in higher memory usage because FSDP will
            accumulate the full model gradients (instead of gradient shards)
            until the eventual sync.

        .. note:: When used with CPU offloading, the gradients will not be
            offloaded to CPU when inside the context manager. Instead, they
            will only be offloaded right after the eventual sync.
        """
        self._lazy_init()
        assert self._is_root, "`no_sync()` on inner FSDP instances is not supported"
        self._assert_state(TrainingState_.IDLE)
        old_flags = []
        for m in self.modules():
            if isinstance(m, FullyShardedDataParallel):
                old_flags.append((m, m._require_backward_grad_sync))
                m._require_backward_grad_sync = False
        try:
            yield
        finally:
            for m, old_flag in old_flags:
                assert not m._require_backward_grad_sync, (
                    "`_require_backward_grad_sync` was incorrectly set to "
                    "`True` while in the `no_sync()` context manager"
                )
                m._require_backward_grad_sync = old_flag

    @property
    def params_with_grad(self) -> List[Parameter]:
        """[p for p in self.parameters() if p.grad is not None]"""
        return [p for p in self.parameters() if p.grad is not None]

    @torch.no_grad()
    def clip_grad_norm_(
        self, max_norm: Union[float, int], norm_type: Union[float, int] = 2.0
    ) -> None:
        """
        Clip all gradients at this point in time. The norm is computed over all
        gradients together, as if they were concatenated into a single vector.
        Gradients are modified in-place.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'``
                for infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).

        .. note:: This is analogous to `torch.nn.utils.clip_grad_norm_` but
            handles the partitioning and multiple devices per rank under the
            hood. The default torch util is not applicable here, because each
            rank only has a partial view of all the grads in the model, so
            calling it for FSDP models would lead to different scaling being
            applied per subset of model parameters.

        .. warning:: This needs to be called on all ranks, since synchronization
            primitives will be used.
        """
        # Call `_lazy_init` to ensure the stream synchronization is done appropriately.
        self._lazy_init()
        assert self._is_root, "clip_grad_norm should only be called on the root (parent) instance"
        self._assert_state(TrainingState_.IDLE)

        max_norm = float(max_norm)
        norm_type = float(norm_type)
        # Computes the max norm for this shard's gradients and sync's across workers
        local_norm = _calc_grad_norm(self.params_with_grad, norm_type).cuda()  # type: ignore[arg-type]
        if norm_type == inf:
            total_norm = local_norm
            dist.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=self.process_group)
        else:
            total_norm = local_norm ** norm_type
            dist.all_reduce(total_norm, group=self.process_group)
            total_norm = total_norm ** (1.0 / norm_type)

        if self.cpu_offload:
            total_norm = total_norm.cpu()

        clip_coef = torch.tensor(max_norm, dtype=total_norm.dtype, device=total_norm.device) / (total_norm + 1e-6)
        if clip_coef < 1:
            # multiply by clip_coef, aka, (max_norm/total_norm).
            for p in self.params_with_grad:
                assert p.grad is not None
                p.grad.detach().mul_(clip_coef.to(p.grad.device))


def _get_default_cuda_device(module: nn.Module) -> torch.device:
    """Try to infer CUDA device from module parameters."""
    try:
        compute_device = next(module.parameters()).device
        if compute_device.type == "cuda":
            return compute_device
    # e.g., if module does not have parameters, it will throw StopIteration,
    # in this case, instead of raising exception, return cuda device.
    except StopIteration:
        pass
    # Fall back to current CUDA device
    return torch.device("cuda")


def _free_storage(data: torch.Tensor) -> None:
    """Free underlying storage of a Tensor."""
    if data.storage().size() > 0:
        # Since we're modifying the Tensor's Storage directly, make sure the Tensor
        # is the sole occupant of the Storage.
        assert (
            data.storage_offset() == 0
        ), "The tensor is not the sole occupant of the storage."
        data.storage().resize_(0)  # type: ignore[attr-defined]


@torch.no_grad()
def _alloc_storage(data: torch.Tensor, size: torch.Size) -> None:
    """Allocate storage for a tensor."""
    if data.storage().size() == size.numel():  # no need to reallocate
        return
    assert (
        data.storage().size() == 0
    ), "Then tensor storage should have been resized to be 0."
    data.storage().resize_(size.numel())  # type: ignore[attr-defined]

def p_assert(cond: Any, s: Any) -> None:
    """This is used as an alternate to ``assert`` when in the backward context
    to print the error message ``s`` since otherwise, it is swallowed."""
    if not cond:
        print(s)
        raise AssertionError

def _calc_grad_norm(parameters: List[torch.nn.Parameter], p: float) -> torch.Tensor:
    r"""Calculate gradient norm of an iterable of parameters.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return torch.tensor(0.0)
    if p == inf:
        local_norm = torch.tensor(max(par.grad.detach().abs().max() for par in parameters))
    else:
        # Compute the norm in full precision no matter what
        local_norm = torch.linalg.norm(
            torch.stack(
                [
                    torch.linalg.norm(par.grad.detach(), p, dtype=torch.float32)
                    for par in parameters
                ]
            ),
            p,
        )
    local_norm.to(dtype=parameters[0].dtype)
    return local_norm
