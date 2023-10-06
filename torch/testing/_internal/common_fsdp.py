# Owner(s): ["oncall: distributed"]

import itertools
import os
import re
import sys
from abc import ABC, abstractmethod
from contextlib import nullcontext
from copy import deepcopy
from enum import auto, Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import TrainingState
from torch.distributed.fsdp._init_utils import NO_RESHARD_AFTER_FORWARD_STRATEGIES
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import always_wrap_policy, ModuleWrapPolicy, wrap
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    MultiThreadedTestCase,
    TEST_SKIPS,
)
from torch.testing._internal.common_utils import FILE_SCHEMA, get_cycles_per_ms


class FSDPInitMode(Enum):
    # No FSDP wrapping
    NO_FSDP = auto()
    # FSDP recursive wrapping
    RECURSIVE = auto()
    # TODO: FSDP non-recursive wrapping
    # NONRECURSIVE = auto()


class CUDAInitMode(Enum):
    # Move model to CUDA before passing to the FSDP constructor
    CUDA_BEFORE = auto()
    # Move model to CUDA after passing to the FSDP constructor
    CUDA_AFTER = auto()
    # Keep on CPU
    CUDA_NEVER = auto()


class FSDPTestModel(nn.Module, ABC):
    """This defines the interface expected from all models used commonly for
    FSDP unit tests."""

    @abstractmethod
    def get_input(self, device) -> Tuple[torch.Tensor, ...]:
        """Returns an input for the model as as tuple."""
        ...

    @abstractmethod
    def get_loss(self, input, output) -> torch.Tensor:
        """Returns the loss given the input and output."""
        ...

    @abstractmethod
    def run_backward(self, loss) -> None:
        """Runs the backward pass (e.g. including ``loss.backward()``)."""
        ...

    @staticmethod
    @abstractmethod
    def init(
        group: dist.ProcessGroup,
        fsdp_init_mode: FSDPInitMode,
        *init_args: Any,
        cuda_init_mode: CUDAInitMode,
        fsdp_kwargs: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
        **init_kwargs: Any,
    ) -> nn.Module:
        """Initializes an instance of this model."""
        ...


def _assert_module_states(
    model: nn.Module,
    process_group: dist.ProcessGroup,
    assert_fn: Callable,
):
    """
    All-gathers module states across ranks and calls ``assert_fn`` on each pair
    of corresponding states from rank 0 and a nonzero rank. For example, if
    ``assert_fn`` is ``self.assertEqual()``, then this checks that all module
    states are equal across ranks.
    """
    # Include names for debugging convenience
    named_module_states = [
        (param_name, param.detach().cpu())
        for param_name, param in model.named_parameters()
    ]
    named_module_states += [
        (buffer_name, buffer.detach().cpu())
        for buffer_name, buffer in model.named_buffers()
    ]
    world_size = dist.get_world_size(process_group)
    olist = [None for _ in range(world_size)]
    dist.all_gather_object(olist, named_module_states, group=process_group)
    rank0_states = olist[0]
    for state in olist[1:]:
        for (_, p1), (_, p2) in zip(rank0_states, state):
            assert_fn(p1, p2)


def _zero_model(
    model: nn.Module,
    zero_buffers: bool = False,
    summon_full=True,
):
    """Zeros the parameters and optionally buffers of ``model`` in place."""
    ctx = FSDP.summon_full_params(model) if summon_full else nullcontext()
    with ctx:
        for param in model.parameters():
            with torch.no_grad():
                param.zero_()
        if zero_buffers:
            for buffer in model.buffers():
                with torch.no_grad():
                    buffer.zero_()


def _get_state_dict(model, cpu_offload=False, half=False):
    if not cpu_offload:
        model = model.cuda()
    if half:
        model.half()

    return model.state_dict()


def subtest_name(test_name_mapping, *args):
    return "_".join(
        [test_name_mapping[str(s)] if s is not None else "none" for s in args]
    )


def _broadcast_state_dict(rank, state_dict):
    # For non-FSDP roots, some parts of the model state on rank 0 may
    # not be on CPU, so we move everything to CPU to avoid issues like:
    # https://github.com/pytorch/pytorch/issues/77113.
    for param_name, param in state_dict.items():
        if param.device != torch.device("cpu"):
            state_dict[param_name] = param.cpu()

    olist = [state_dict if rank == 0 else None]
    dist.broadcast_object_list(olist)
    state_dict = olist[0]
    # Ensure that the state is on CUDA
    for param_name in state_dict.keys():
        state_dict[param_name] = state_dict[param_name].cuda()
    return state_dict


def get_full_params(model: nn.Module, recurse: bool = True):
    """
    Returns the full unsharded parameters of ``model``. Any FSDP-managed
    parameters offloaded to CPU are moved to GPU in the returned list.

    Args:
        recurse (bool): If ``False``, only unshards the parameters immediate to
            ``model``; if ``True``, recurses through the module hierarchy
            rooted at ``model``.
    """
    with FSDP.summon_full_params(model, recurse=recurse):
        return deepcopy(list(model.parameters()))


def _maybe_cuda(model: nn.Module, move_to_cuda: bool):
    return model.cuda() if move_to_cuda else model


def _maybe_wrap_fsdp(model: nn.Module, wrap_fsdp: bool, *args, **kwargs):
    return model if not wrap_fsdp else FSDP(model, *args, **kwargs)


class DummyProcessGroup:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size

    def allreduce(self, *args, **kwargs):
        dist_wait = mock.Mock()

        def get_future():
            future = torch.futures.Future()
            future.set_result(1)
            return future

        dist_wait.get_future = get_future
        return dist_wait


class TransformerWithSharedParams(FSDPTestModel):
    def __init__(
        self,
        group: dist.ProcessGroup,
        cuda_init_mode: CUDAInitMode,
        add_bn: bool,
        deterministic: bool,
    ):
        super().__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        if deterministic:
            torch.manual_seed(0)
        d_vocab = 23
        d_model = 16

        self.embed_tokens = nn.Embedding(d_vocab, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=8,
            dropout=0.1,
        )
        self.output_proj = nn.Linear(d_model, d_vocab)

        # share the embedding and output projection weights
        self.output_proj.weight = self.embed_tokens.weight
        self.register_buffer(
            "vocab_bias", self.embed_tokens.weight.new_ones((d_model,))
        )
        self.register_buffer(
            "long_buffer",
            torch.zeros_like(self.vocab_bias, dtype=torch.long),
        )  # type: ignore[arg-type]

        self.bs = 2
        self.bn = torch.nn.BatchNorm1d(self.bs) if add_bn else torch.nn.Identity()
        if cuda_init_mode == CUDAInitMode.CUDA_BEFORE:
            self = self.cuda()
        if deterministic:
            self.eval()

    def get_input(self, device):
        torch.manual_seed(1 + self.rank)  # keep everything deterministic
        src = torch.arange(12, device=device).view(6, self.bs)  # T x B
        tgt = torch.arange(self.bs * 4, device=device).view(4, self.bs)  # T x B
        return (src, tgt)

    def forward(self, src_ids, tgt_ids):
        src = self.embed_tokens(src_ids)
        src = src + self.vocab_bias + self.long_buffer.type_as(src)  # type: ignore[operator]
        tgt = self.embed_tokens(tgt_ids)
        tgt = self.bn(tgt)
        x = self.transformer(src, tgt)
        return self.output_proj(x)

    def get_loss(self, input, output):
        _, tgt = input
        return nn.functional.cross_entropy(
            output.view(-1, output.size(-1)), tgt.view(-1), reduction="sum"
        )

    def run_backward(self, loss):
        loss.backward()

    @staticmethod
    def init(
        group: dist.ProcessGroup,
        fsdp_init_mode: FSDPInitMode,
        cuda_init_mode: CUDAInitMode,
        fsdp_kwargs: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
        add_bn: bool = True,
    ) -> Union[nn.Module, FSDP]:
        """
        Initializes a :class:`TransformerWithSharedParams` instance.

        Args:
            fsdp_init_mode (FSDPInitMode): If ``NO_FSDP``, then does not wrap
                any modules with FSDP. If ``RECURSIVE``, then wraps with
                top-level FSDP. By default, the top-level FSDP uses the
                ``ModuleWrapPolicy`` for encoder and decoder layers, but a
                different auto wrap policy may be specified via
                ``fsdp_kwargs``.
            cuda_init_mode (CUDAInitMode): Determines model movement to CUDA.
            fsdp_kwargs (Optional[Dict[str, Any]]): Optional keyword arguments
                forwarded to the FSDP constructor.
            deterministic (bool): Whether to make the model deterministic
                across constructions.
            add_bn (bool): Whether to include batch norm in the model.
        """

        if fsdp_kwargs is None:
            fsdp_kwargs = {}
        if fsdp_init_mode == FSDPInitMode.NO_FSDP:
            if isinstance(group, tuple):
                pg = group[0]
            else:
                pg = group
            return TransformerWithSharedParams(
                pg, cuda_init_mode, add_bn, deterministic
            )
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            # Default to the `ModuleWrapPolicy`
            if "auto_wrap_policy" not in fsdp_kwargs:
                auto_wrap_policy = ModuleWrapPolicy(
                    {
                        TransformerEncoderLayer,
                        TransformerDecoderLayer,
                    }
                )
            else:
                auto_wrap_policy = fsdp_kwargs.pop("auto_wrap_policy")

            if (
                "sharding_strategy" in fsdp_kwargs
                and fsdp_kwargs["sharding_strategy"]
                in {ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2}
                and not isinstance(group, tuple)
            ):
                fsdp_pg = None
            else:
                fsdp_pg = group

            if isinstance(group, tuple):
                tformer_pg = group[0]
            else:
                tformer_pg = group

            m = TransformerWithSharedParams(
                tformer_pg, cuda_init_mode, add_bn, deterministic
            )
            fsdp_model = FSDP(
                m,
                fsdp_pg,
                auto_wrap_policy=auto_wrap_policy,
                **fsdp_kwargs,
            )
            if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
                fsdp_model = fsdp_model.cuda()
            return fsdp_model
        raise ValueError(f"Unsupported FSDP init mode: {fsdp_init_mode}")

    def get_ignored_modules(self):
        return [self.transformer]


class NestedWrappedModule(FSDPTestModel):
    def __init__(
        self,
        group: dist.ProcessGroup,
        wrap_fsdp: bool,
        cuda_init_mode: CUDAInitMode,
        deterministic: bool,
        **fsdp_kwargs,
    ):
        super().__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        move_to_cuda = cuda_init_mode == CUDAInitMode.CUDA_BEFORE

        def _maybe_wrap(layer):
            if wrap_fsdp:
                return FSDP(layer, group, **fsdp_kwargs)
            return layer

        if deterministic:
            torch.manual_seed(0)
        self.module = nn.Sequential(
            _maybe_cuda(nn.Linear(8, 4), move_to_cuda),
            _maybe_wrap(
                nn.Sequential(
                    _maybe_wrap(_maybe_cuda(nn.Linear(4, 16), move_to_cuda)),
                    _maybe_cuda(nn.Linear(16, 16), move_to_cuda),
                ),
            ),
            _maybe_wrap(_maybe_cuda(nn.Linear(16, 4), move_to_cuda)),
            _maybe_cuda(nn.Linear(4, 8), move_to_cuda),
        )

    def get_input(self, device):
        torch.manual_seed(1 + self.rank)  # keep everything deterministic
        return (torch.rand(4, 8, device=device),)

    def forward(self, x):
        return self.module(x)

    def get_loss(self, input, output):
        loss = output.sum()
        return loss

    def run_backward(self, loss):
        loss.backward()

    @staticmethod
    def init(
        group: dist.ProcessGroup,
        fsdp_init_mode: FSDPInitMode,
        cuda_init_mode: CUDAInitMode,
        fsdp_kwargs: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
    ) -> nn.Module:
        """
        Initializes a :class:`NestedWrappedModule` instance.

        Args:
            fsdp_init_mode (FSDPInitMode): If ``NO_FSDP``, then does not wrap
                any modules with FSDP. If ``RECURSIVE``, then wraps some nested
                modules with FSDP but not the top-level module. The model may
                later be wrapped with a top-level FSDP external to this method
                if desired.
            cuda_init_mode (CUDAInitMode): Determines model movement to CUDA.
            fsdp_kwargs (Optional[Dict[str, Any]]): Optional keyword arguments
                forwarded to the FSDP constructor.
            deterministic (bool): Whether to make the model deterministic
                across constructions.
        """
        if fsdp_kwargs is None:
            fsdp_kwargs = {}
        if fsdp_init_mode == FSDPInitMode.NO_FSDP:
            return NestedWrappedModule(
                group,
                wrap_fsdp=False,
                cuda_init_mode=cuda_init_mode,
                deterministic=deterministic,
            )
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            # Does not wrap with top-level FSDP
            fsdp_model = NestedWrappedModule(
                group,
                wrap_fsdp=True,
                cuda_init_mode=cuda_init_mode,
                deterministic=deterministic,
                **fsdp_kwargs,
            )
            if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
                fsdp_model = fsdp_model.cuda()
            return fsdp_model
        raise ValueError(f"Unsupported FSDP init mode: {fsdp_init_mode}")


class AlwaysWrapNestedWrappedModule(NestedWrappedModule):
    @staticmethod
    def init(
        group: dist.ProcessGroup,
        fsdp_init_mode: FSDPInitMode,
        cuda_init_mode: CUDAInitMode,
        fsdp_kwargs: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
    ):
        """
        Initializes a :class:`NestedWrappedModule` instance, but unlike
        :meth:`NestedWrappedModule.init`, for the ``RECURSIVE`` init mode, this
        wraps with top-level FSDP and the ``always_wrap_policy()`` auto wrap
        policy.
        """
        super_ = super(AlwaysWrapNestedWrappedModule, AlwaysWrapNestedWrappedModule)
        model = super_.init(
            group=group,
            fsdp_init_mode=FSDPInitMode.NO_FSDP,
            cuda_init_mode=cuda_init_mode,
            fsdp_kwargs=fsdp_kwargs,
            deterministic=deterministic,
        )
        if fsdp_init_mode == FSDPInitMode.NO_FSDP:
            return model
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            fsdp_model = FSDP(model, auto_wrap_policy=always_wrap_policy, **fsdp_kwargs)
            if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
                fsdp_model = fsdp_model.cuda()
            return fsdp_model


class NonUniformReqGradNWM(NestedWrappedModule):
    def __init__(
        self,
        group: dist.ProcessGroup,
        wrap_fsdp: bool,
        cuda_init_mode: CUDAInitMode,
        deterministic: bool,
        **fsdp_kwargs,
    ):
        super(NestedWrappedModule, self).__init__()
        # This `__init__` only differs from `NestedWrappedModule.__init__` in that
        # the last two `nn.Linear` layers are FSDP wrapped in a `nn.Sequential`
        # container. This arrangement results in all elements of the last two parameters
        # residing on a single rank. Freezing all parameters except those two allows us
        # to verify that `ShardedGradScaler` accommodates situations where some ranks
        # have no (non-zero sized) parameter shards.
        self.rank = group.rank()
        self.world_size = group.size()
        move_to_cuda = cuda_init_mode == CUDAInitMode.CUDA_BEFORE

        def _maybe_wrap(layer):
            if wrap_fsdp:
                return FSDP(layer, group, **fsdp_kwargs)
            return layer

        if deterministic:
            torch.manual_seed(0)
        self.module = nn.Sequential(
            _maybe_cuda(nn.Linear(8, 4), move_to_cuda),
            _maybe_wrap(
                nn.Sequential(
                    _maybe_wrap(_maybe_cuda(nn.Linear(4, 16), move_to_cuda)),
                    _maybe_cuda(nn.Linear(16, 16), move_to_cuda),
                ),
            ),
            _maybe_wrap(
                nn.Sequential(
                    _maybe_cuda(nn.Linear(16, 4), move_to_cuda),
                    _maybe_cuda(nn.Linear(4, 8), move_to_cuda),
                ),
            ),
        )

    @staticmethod
    def _set_nonuniform_req_grad(model, req_grad_mask) -> None:
        for n, p in model.named_parameters():
            if not re.match(req_grad_mask, n):
                p.requires_grad_(False)

    @staticmethod
    def init(
        group: dist.ProcessGroup,
        fsdp_init_mode: FSDPInitMode,
        cuda_init_mode: CUDAInitMode,
        fsdp_kwargs: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
    ):
        """
        Initializes a :class:`NestedWrappedModule` instance, but unlike
        :meth:`NestedWrappedModule.init`, it wraps a second :class:`torch.nn.Sequential`
        container to enable the desired non-uniform ``requires_grad``
        ``use_orig_params=True`` tests. For both ``RECURSIVE`` and ``NO_FSDP``
        init modes, freezes all parameters except the last two to validate
        ``ShardedGradScaler`` support for ranks with no (non-zero sized) local shards in
        FSDP ``use_orig_params=True`` mode.
        """
        # The parameters that should remain unfrozen are in `module.2.1`. The regex
        # pattern below matches the relevant parameter names both with and without
        # an interstitial FSDP module indicator (`_fsdp_wrapped_module`) present.
        req_grad_pattern = re.compile(r"module\.2.*\.1.*")
        if fsdp_init_mode == FSDPInitMode.NO_FSDP:
            ddp_model = NonUniformReqGradNWM(
                group,
                wrap_fsdp=False,
                cuda_init_mode=cuda_init_mode,
                deterministic=deterministic,
            )
            NonUniformReqGradNWM._set_nonuniform_req_grad(ddp_model, req_grad_pattern)
            return ddp_model
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            if fsdp_kwargs is None:
                fsdp_kwargs = {}
            fsdp_model = NonUniformReqGradNWM(
                group,
                wrap_fsdp=True,
                cuda_init_mode=cuda_init_mode,
                deterministic=deterministic,
                **fsdp_kwargs,
            )
            if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
                fsdp_model = fsdp_model.cuda()
            NonUniformReqGradNWM._set_nonuniform_req_grad(fsdp_model, req_grad_pattern)
            return fsdp_model
        raise ValueError(f"Unsupported FSDP init mode: {fsdp_init_mode}")


class ModuleWithDelay(FSDPTestModel):
    """This class wraps a :class:`FSDPTestModel` to optionally add a delay
    after computing the loss and/or before the gradient reduction."""

    def __init__(
        self,
        module: nn.Module,
        delay_after_loss_ms: int,
        delay_before_reduction_ms: int,
    ):
        super().__init__()
        self.delay_after_loss_ms = delay_after_loss_ms
        self.delay_before_reduction_ms = delay_before_reduction_ms
        self.module = module

    def get_input(self, device):
        return self.module.get_input(device)

    def forward(self, x):
        return self.module(x)

    def get_loss(self, input, output):
        loss = self.module.get_loss(input, output)
        if self.delay_after_loss_ms > 0:
            torch.cuda._sleep(int(self.delay_after_loss_ms * get_cycles_per_ms()))
        return loss

    def run_backward(self, loss):
        orig_reduce_scatter = torch.distributed.reduce_scatter_tensor

        def _delayed_reduce_scatter(*args, **kwargs):
            if self.delay_before_reduction_ms > 0:
                torch.cuda._sleep(
                    int(self.delay_before_reduction_ms * get_cycles_per_ms())
                )
            return orig_reduce_scatter(*args, **kwargs)

        with mock.patch(
            "torch.distributed.reduce_scatter_tensor", _delayed_reduce_scatter
        ):
            self.module.run_backward(loss)

    @staticmethod
    def init(
        module_class: Type[FSDPTestModel],
        *model_args: Any,
        delay_after_loss_ms: int,
        delay_before_reduction_ms: int,
        **model_kwargs: Any,
    ):
        """
        Args:
            module_class (Type[FSDPTestModel]): Wrapped module class to which
                to add delays.
            model_args: Positional arguments forwarded to the ``module_class``
                ``init()``.
            delay_after_loss_ms (int): Delay after computing the loss/before
                the optimizer step (in ms).
            delay_before_reduction_ms (int): Delay before reduce-scattering
                gradients (in ms).
            model_kwargs: Keyword arguments forwarded to the ``module_class``
                ``init()``.
        """
        return ModuleWithDelay(
            module_class.init(*model_args, **model_kwargs),
            delay_after_loss_ms,
            delay_before_reduction_ms,
        )


class NestedWrappedModuleWithDelay(ModuleWithDelay):
    @staticmethod
    def init(
        group: dist.ProcessGroup,
        fsdp_init_mode: FSDPInitMode,
        cuda_init_mode: CUDAInitMode = CUDAInitMode.CUDA_AFTER,
        fsdp_kwargs: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
        delay_after_loss_ms: int = 0,
        delay_before_reduction_ms: int = 0,
    ):
        return super(NestedWrappedModuleWithDelay, NestedWrappedModuleWithDelay).init(
            NestedWrappedModule,
            group=group,
            fsdp_init_mode=fsdp_init_mode,
            cuda_init_mode=cuda_init_mode,
            fsdp_kwargs=fsdp_kwargs,
            deterministic=deterministic,
            delay_after_loss_ms=delay_after_loss_ms,
            delay_before_reduction_ms=delay_before_reduction_ms,
        )


class DummyDDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class MixtureOfExperts(NestedWrappedModule):
    def __init__(
        self,
        group: dist.ProcessGroup,
        wrap_fsdp: bool,
        cuda_init_mode: CUDAInitMode,
        delay_before_free_ms: int,
        deterministic: bool,
        **fsdp_kwargs,
    ):
        super().__init__(
            group=group,
            wrap_fsdp=wrap_fsdp,
            cuda_init_mode=cuda_init_mode,
            deterministic=deterministic,
        )
        self.group = group
        self.delay_before_free_ms = delay_before_free_ms
        self.wrap_fsdp = wrap_fsdp
        self.move_to_cuda = cuda_init_mode == CUDAInitMode.CUDA_BEFORE
        if deterministic:
            # Give each rank different expert parameters
            torch.manual_seed(42 + self.rank)
        d_expert = 23
        d_shared = 12
        d_input = 8
        expert = _maybe_cuda(nn.Linear(d_expert, d_shared), self.move_to_cuda)

        self.num_expert_params = sum([p.numel() for p in expert.parameters()])
        for p in expert.parameters():
            p.expert = True  # type: ignore[attr-defined]

        if deterministic:
            # Keep all other parameters the same across ranks
            torch.manual_seed(0)

        shared = _maybe_cuda(nn.Linear(d_shared, d_expert), self.move_to_cuda)

        if wrap_fsdp:
            # we create a process group of size 1 for the expert params
            expert_group = torch.distributed.new_group(
                [group.rank()]
            )  # world size 1 means no shard
            expert = FSDP(expert, expert_group, **fsdp_kwargs)  # type: ignore[assignment]
            shared = FSDP(shared, group, **fsdp_kwargs)  # type: ignore[assignment]

        self.module = nn.Sequential(
            _maybe_cuda(nn.Linear(d_input, d_shared), self.move_to_cuda),
            shared,
            expert,
            _maybe_cuda(nn.Linear(d_shared, d_input), self.move_to_cuda),
        )

    def forward(self, x):
        if self.delay_before_free_ms > 0:
            expert = self.module[2]
            if isinstance(expert, FSDP):
                orig_reshard = torch.distributed.fsdp._runtime_utils._reshard

                def _delayed_reshard(*args, **kwargs):
                    torch.cuda._sleep(
                        int(self.delay_before_free_ms * get_cycles_per_ms())
                    )
                    return orig_reshard(*args, **kwargs)

                # This patch covers any `import torch..._reshard` uses.
                with mock.patch(
                    "torch.distributed.fsdp._runtime_utils._reshard", _delayed_reshard
                ):
                    return self.module(x)

        return self.module(x)

    def run_backward(self, loss):
        loss.backward()
        # Manually reduce gradients if not wrapped in FullyShardedDataParallel
        if not self.wrap_fsdp:
            with torch.no_grad():
                for p in self.parameters():
                    if hasattr(p, "expert"):
                        continue  # these params don't need grad reduction
                    p.grad.div_(self.world_size)
                    torch.distributed.all_reduce(p.grad, group=self.group)

    @staticmethod
    def init(
        group: dist.ProcessGroup,
        fsdp_init_mode: FSDPInitMode,
        cuda_init_mode: CUDAInitMode,
        fsdp_kwargs: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
        delay_before_free_ms: int = 0,
    ):
        """
        Initializes a :class:`MixtureOfExperts` instance.

        Args:
            fsdp_init_mode (FSDPInitMode): If ``NO_FSDP``, then does not wrap
                any modules with FSDP. If ``RECURSIVE``, then wraps some nested
                modules with FSDP, including the expert and shared layers, but
                not the top-level module. The model may later be wrapped with a
                top-level FSDP external to this method if desired.
            cuda_init_mode (CUDAInitMode): Determines model movement to CUDA.
            fsdp_kwargs (Optional[Dict[str, Any]]): Optional keyword arguments
                forwarded to the FSDP constructor.
            deterministic (bool): Whether to make the model deterministic
                across constructions.
            delay_before_free_ms (int): Delay before resharding expert
                parameters in the forward pass (in ms).
        """
        if fsdp_kwargs is None:
            fsdp_kwargs = {}
        if fsdp_init_mode == FSDPInitMode.NO_FSDP:
            return MixtureOfExperts(
                group,
                wrap_fsdp=False,
                cuda_init_mode=cuda_init_mode,
                delay_before_free_ms=delay_before_free_ms,
                deterministic=deterministic,
            )
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            # Does not wrap with top-level FSDP
            fsdp_model = MixtureOfExperts(
                group,
                wrap_fsdp=True,
                cuda_init_mode=cuda_init_mode,
                delay_before_free_ms=delay_before_free_ms,
                deterministic=deterministic,
                **fsdp_kwargs,
            )
            if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
                fsdp_model = fsdp_model.cuda()
            return fsdp_model
        raise ValueError(f"Unsupported FSDP init mode: {fsdp_init_mode}")


def run_subtests(
    cls_inst,
    subtest_config: Dict[str, List[Any]],
    test_fn: Callable,
    *test_args,
    **test_kwargs: Any,
):
    """
    Runs a test function given by ``test_fn`` as a subtest according to the
    configurations specified by ``subtest_config``. This amortizes the
    costly setup overhead (including process spawn and initializing the
    process group) over the subtests.

    Args:
        subtest_config (Dict[str, List[Any]]): A mapping from subtest
            keyword argument name to a list of its possible values.
        test_fn (Callable): A callable that runs the actual test.
        test_args: Positional arguments to pass to ``test_fn``.
        test_kwargs: Keyword arguments to pass to ``test_fn``.
    """
    # Convert the config mapping to a list to have a fixed order
    subtest_config_items: List[Tuple[str, List[Any]]] = list(subtest_config.items())
    subtest_config_keys: List[str] = [item[0] for item in subtest_config_items]
    subtest_config_values: List[List[Any]] = [item[1] for item in subtest_config_items]
    for values in itertools.product(*subtest_config_values):
        # Map keyword to chosen value
        subtest_kwargs = dict(zip(subtest_config_keys, values))
        with cls_inst.subTest(**subtest_kwargs):
            test_fn(*test_args, **test_kwargs, **subtest_kwargs)
        dist.barrier()


class FSDPTestMultiThread(MultiThreadedTestCase):
    @property
    def world_size(self):
        return torch.cuda.device_count() if torch.cuda.is_available() else 4

    def setUp(self):
        super().setUp()
        self._spawn_threads()

    def run_subtests(self, *args, **kwargs):
        return run_subtests(self, *args, **kwargs)


class FSDPTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        # Set NCCL_DESYNC_DEBUG=0 to disable the NCCL `workCleanupLoop()`,
        # which can cause unit test flakiness:
        # https://github.com/pytorch/pytorch/issues/90848
        os.environ["NCCL_DESYNC_DEBUG"] = "0"
        self._spawn_processes()

    @property
    def world_size(self):
        return torch.cuda.device_count() if torch.cuda.is_available() else 4

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def init_method(self):
        return f"{FILE_SCHEMA}{self.file_name}"

    def _check_cpu_offload(self, fsdp_model, cpu_offload):
        self.assertEqual(cpu_offload, fsdp_model.cpu_offload)

    def _check_backward_prefetch(self, fsdp_model, backward_prefetch):
        self.assertEqual(backward_prefetch, fsdp_model.backward_prefetch)

    def _check_forward_prefetch(self, fsdp_model, forward_prefetch):
        self.assertEqual(forward_prefetch, fsdp_model.forward_prefetch)

    def run_subtests(self, *args, **kwargs):
        return run_subtests(self, *args, **kwargs)

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe):
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        print(f"dist init r={self.rank}, world={self.world_size}")

        # Specify gloo backend to make 'init_process_group()' succeed,
        # Actual tests will be skipped if there is no enough GPUs.
        backend = "nccl" if torch.cuda.is_available() else "gloo"

        try:
            dist.init_process_group(
                init_method=self.init_method,
                backend=backend,
                world_size=int(self.world_size),
                rank=self.rank,
            )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        if torch.cuda.is_available() and torch.cuda.device_count():
            torch.cuda.set_device(self.rank % torch.cuda.device_count())

        # Execute barrier prior to running test to ensure that every process
        # has finished initialization and that the following test
        # immediately exiting due to a skip doesn't cause flakiness.
        dist.barrier()

        self.run_test(test_name, pipe)

        dist.barrier()

        dist.destroy_process_group()

    def _train_for_several_steps(
        self,
        model: nn.Module,
        num_steps: int,
        autocast: bool,
        lr: float = 0.01,
        fsdp_cpu_offload: Optional[CPUOffload] = None,
        save_model: bool = False,
        mixed_precision: Optional[MixedPrecision] = None,
        enable_sharded_grad_scaler: bool = False,
        use_pure_fp16: bool = False,
        sharded_grad_scaler_kwargs: Optional[Dict[str, Any]] = None,
    ):
        cpu_offload_params = fsdp_cpu_offload and fsdp_cpu_offload.offload_params

        model_device = next(model.parameters()).device
        if sharded_grad_scaler_kwargs is None:
            sharded_grad_scaler_kwargs = {}
        sharded_grad_scaler = ShardedGradScaler(
            enabled=enable_sharded_grad_scaler, **sharded_grad_scaler_kwargs
        )
        # use SGD with momentum instead of Adam, since Adam is scale invariant
        # and this makes it bad for tests
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        for _ in range(num_steps):
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=autocast):
                # Inputs always cuda regardless of cpu offloading, or model.device
                input = model.module.get_input(torch.device("cuda"))
                if use_pure_fp16 or (mixed_precision and not isinstance(model, FSDP)):
                    if isinstance(input, torch.Tensor):
                        input = input.half()
                    else:
                        input = tuple(x.half() for x in input)
                output = model(*input)
                # Post-forward, if CPU offloading model param should be on CPU.
                if (
                    cpu_offload_params
                    and isinstance(model, FSDP)
                    # If not resharding after forward, the parameters are still
                    # exposed as unsharded views into the GPU flat parameter
                    and model.sharding_strategy
                    not in NO_RESHARD_AFTER_FORWARD_STRATEGIES
                ):
                    for p in model.parameters():
                        # Params should always be on CPU
                        self.assertEqual(p.device, torch.device("cpu"))

                loss = model.module.get_loss(input, output).to(model_device)
            loss = sharded_grad_scaler.scale(loss)

            if not mixed_precision and not use_pure_fp16:
                assert (
                    loss.dtype == torch.float32
                ), "loss data type should be float32, as the original \
                    parameter data type is float32."
            else:
                if use_pure_fp16:
                    self.assertEqual(loss.dtype, torch.float16)
                # FSDP loss is fp16, DDP AMP loss is fp32
                elif isinstance(model, FSDP):
                    self.assertEqual(loss.dtype, mixed_precision.param_dtype)
                else:
                    self.assertEqual(loss.dtype, torch.float32)
            model.module.run_backward(loss)
            # Post-backward, if CPU offloading model params should be on CPU.
            if cpu_offload_params and isinstance(model, FSDP):
                for p in model.parameters():
                    # Params should always be on CPU
                    self.assertEqual(p.device, torch.device("cpu"))
            # Unscale the gradients and step
            sharded_grad_scaler.step(optim)
            # Update the scale factor
            sharded_grad_scaler.update()
            # if save_model, simulate save + load.
            if save_model:
                state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                # Zero params, if save/load state_dict did not work properly, this
                # would break the parity test with DDP.
                _zero_model(model)
                model.load_state_dict(state_dict)

        if isinstance(model, FSDP):
            model._assert_state(TrainingState.IDLE)
        return loss.detach()

    def _test_fsdp_parity(
        self,
        model_class: Type[FSDPTestModel],
        fsdp_init_mode: FSDPInitMode,
        cuda_init_mode: CUDAInitMode,
        ref_init_fn: Optional[Callable] = None,
        num_iters: int = 2,
        save_model: bool = True,
        cpu_offload: CPUOffload = CPUOffload(),
        backward_prefetch: Optional[BackwardPrefetch] = None,
        sharding_strategy: Optional[ShardingStrategy] = None,
        mixed_precision: Optional[MixedPrecision] = None,
        forward_prefetch: bool = False,
        use_orig_params: bool = False,
        enable_sharded_grad_scaler: bool = False,
        use_pure_fp16: bool = False,
        init_kwargs: Optional[Dict[str, Any]] = None,
        sharded_grad_scaler_kwargs: Optional[Dict[str, Any]] = None,
        **fsdp_kwargs,
    ):
        """
        Tests FSDP training against a reference, which defaults to DDP but
        may be customized with ``ref_init_fn``.

        Args:
            model_class (Type[FSDPTestModel]): A model class that inherits from
                ``FSDPTestModel``, which defines the expected interface.
            fsdp_init_mode (FSDPInitMode): The mode to initialize the
                FSDP-wrapped model. This should not be ``NO_FSDP``.
            ref_init_fn (Optional[Callable]): A callable to invoke that wraps a
                non-wrapped model to construct the reference model, where this
                wrapper should provide data parallel semantics. If ``None``,
                then the callable defaults to the DDP constructor.
        """
        assert (
            fsdp_init_mode != FSDPInitMode.NO_FSDP
        ), "Expects an FSDP init mode that wraps with FSDP"
        if init_kwargs is None:
            init_kwargs = {}
        lr = 1e-2
        rank = self.process_group.rank()
        # Establish reference behavior with DDP
        model = model_class.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
            **init_kwargs,
        )
        if ref_init_fn is None:
            ref_model = DDP(model, device_ids=[rank], output_device=rank)
        else:
            ref_model = ref_init_fn(model)
        if use_pure_fp16:
            ref_model = ref_model.half()
        ref_loss = self._train_for_several_steps(
            ref_model,
            num_iters,
            autocast=mixed_precision is not None,
            lr=lr,
            fsdp_cpu_offload=cpu_offload,
            mixed_precision=mixed_precision,
            enable_sharded_grad_scaler=enable_sharded_grad_scaler,
            use_pure_fp16=use_pure_fp16,
            sharded_grad_scaler_kwargs=sharded_grad_scaler_kwargs,
        )
        ddp_params = list(ref_model.parameters())
        # Check against FSDP behavior
        fsdp_kwargs.update(
            {
                "cpu_offload": cpu_offload,
                "backward_prefetch": backward_prefetch,
                "sharding_strategy": sharding_strategy,
                "mixed_precision": mixed_precision,
                "forward_prefetch": forward_prefetch,
                "use_orig_params": use_orig_params,
            }
        )
        try:
            fsdp_model = model_class.init(
                self.process_group,
                fsdp_init_mode,
                cuda_init_mode,
                fsdp_kwargs,
                deterministic=True,
                **init_kwargs,
            )
        except Exception as e:
            raise ValueError(f"Initializing {model_class} raised error {str(e)}") from e
        if not isinstance(fsdp_model, FSDP):
            # Enforce that we wrap with top-level FSDP since we are comparing
            # assuming a data parallel reference and some test models may not
            # do so in their `init()` method
            fsdp_model = FSDP(fsdp_model, self.process_group, **fsdp_kwargs)
        if use_pure_fp16:
            # Change the model parameter dtype after FSDP initialization
            fsdp_model = fsdp_model.half()
        if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
            fsdp_model = fsdp_model.cuda()
        offload_params = cpu_offload is not None and cpu_offload.offload_params
        # Offloading parameters with `CUDA_AFTER` should raise an error during
        # lazy initialization due to the parameter devices not being CPU;
        # otherwise, all parameter devices should be CPU
        expects_device_error = (
            offload_params and cuda_init_mode == CUDAInitMode.CUDA_AFTER
        )
        expects_cpu_device = (
            offload_params and cuda_init_mode != CUDAInitMode.CUDA_AFTER
        )
        if expects_cpu_device:
            cpu_device = torch.device("cpu")
            for param in fsdp_model.parameters():
                self.assertEqual(param.device, cpu_device)
        context = (
            self.assertRaisesRegex(
                RuntimeError,
                "An FSDP-managed module with parameter CPU offloading enabled "
                "has parameters on cuda",
            )
            if expects_device_error
            else nullcontext()
        )
        with context:
            fsdp_loss = self._train_for_several_steps(
                fsdp_model,
                num_iters,
                autocast=False,
                lr=lr,
                fsdp_cpu_offload=cpu_offload,
                save_model=save_model,
                mixed_precision=mixed_precision,
                enable_sharded_grad_scaler=enable_sharded_grad_scaler,
                use_pure_fp16=use_pure_fp16,
                sharded_grad_scaler_kwargs=sharded_grad_scaler_kwargs,
            )
        # No need to check for parameter and loss parity if expecting an error
        if expects_device_error:
            return
        # Check parameter devices are CPU if offloading to CPU before calling
        # `get_full_params()`, which will cast the parameters to FP32
        if offload_params:
            for param in fsdp_model.parameters():
                self.assertEqual(param.device, cpu_device)
            fsdp_loss = fsdp_loss.cuda()
        fsdp_unsharded_params = get_full_params(fsdp_model)
        # Do not check dtype since the reference DDP loss may not be the same
        # dtype as the FSDP loss in the case of mixed precision
        torch.testing.assert_close(ref_loss, fsdp_loss, check_dtype=False)
        # Do not check for parameter parity if using mixed precision since (1)
        # the DDP parameters are in FP16 (from `half()`) while the FSDP
        # parameters are in FP32 (from `summon_full_params()`) and (2) DDP runs
        # the optimizer in FP16 while FSDP runs it in FP32
        # TODO: Disable checking the parameters for pure FP16 due to floating
        # point inaccuracy. Note that this means that the backward pass is not
        # checked: https://github.com/pytorch/pytorch/issues/90784
        if mixed_precision is None and not use_pure_fp16:
            self.assertEqual(
                ddp_params,
                fsdp_unsharded_params,
                exact_device=True,
                msg="FSDP did not match DDP",
            )


class SkipModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        return self.lin(x)


class NestedLinear(nn.Module):
    def __init__(self, fsdp_wrap):
        super().__init__()
        if fsdp_wrap:
            self.nested_linear = wrap(nn.Linear(10, 10, bias=False).cuda())
        else:
            self.nested_linear = nn.Linear(10, 10, bias=False).cuda()

    def forward(self, x):
        return self.nested_linear(x)


class SkipModel(nn.Module):
    def __init__(self, double_nest):
        super().__init__()
        self.linear = nn.Linear(10, 10, bias=False).cuda()
        self.linear_skip = SkipModule().cuda()
        self.nested_linear = wrap(NestedLinear(fsdp_wrap=double_nest))

    def forward(self, x):
        x = self.linear(x)
        x = self.linear_skip(x)
        x = self.nested_linear(x)
        return x
