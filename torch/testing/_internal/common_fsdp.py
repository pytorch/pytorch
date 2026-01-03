# mypy: allow-untyped-defs
# Owner(s): ["oncall: distributed"]

import contextlib
import os
import re
import sys
import time
import unittest
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from enum import auto, Enum
from functools import wraps
from typing import Any, cast, no_type_check, Optional, Union
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable import checkpoint
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffload,
    fully_shard,
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp._common_utils import TrainingState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
    FSDPParamGroup,
    RegisterPostBackwardFunction,
)
from torch.distributed.fsdp._init_utils import NO_RESHARD_AFTER_FORWARD_STRATEGIES
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import always_wrap_policy, ModuleWrapPolicy, wrap
from torch.distributed.tensor import distribute_tensor, DTensor, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
    SequenceParallel,
)
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    MultiThreadedTestCase,
    run_subtests,
    TEST_SKIPS,
)
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    get_cycles_per_ms,
    set_rng_seed,
    TEST_CUDA,
    TEST_HPU,
    TEST_XPU,
)
from torch.utils._triton import has_triton


DEVICE_COUNT = 4  # default

if TEST_CUDA:
    DEVICE_TYPE = "cuda"
    DISTRIBUTED_BACKEND = "nccl"
    DEVICE_COUNT = torch.cuda.device_count()
elif TEST_HPU:
    DEVICE_TYPE = "hpu:0"
    DISTRIBUTED_BACKEND = "hccl"
elif TEST_XPU:
    DEVICE_TYPE = "xpu"
    DISTRIBUTED_BACKEND = "xccl"
    DEVICE_COUNT = torch.xpu.device_count()
else:
    DEVICE_TYPE = "cpu"
    DISTRIBUTED_BACKEND = "gloo"
    DEVICE_COUNT = 1


class FSDPInitMode(Enum):
    # No FSDP wrapping
    NO_FSDP = auto()
    # FSDP recursive wrapping
    RECURSIVE = auto()
    # TODO: FSDP non-recursive wrapping
    # NONRECURSIVE = auto()


class DEVICEInitMode(Enum):
    # Move model to DEVICE before passing to the FSDP constructor
    DEVICE_BEFORE = auto()
    # Move model to DEVICE after passing to the FSDP constructor
    DEVICE_AFTER = auto()
    # Keep on CPU
    DEVICE_NEVER = auto()


class FSDPTestModel(nn.Module, ABC):
    """This defines the interface expected from all models used commonly for
    FSDP unit tests."""

    @abstractmethod
    def get_input(self, device) -> tuple[torch.Tensor, ...]:
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
    def init(*args: Any, **kwargs: Any) -> nn.Module:
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
    assert rank0_states is not None  # mypy
    for state in olist[1:]:
        assert state is not None  # mypy
        for (_, p1), (_, p2) in zip(rank0_states, state, strict=True):
            assert_fn(p1, p2)


def get_devtype():
    return torch.device(DEVICE_TYPE)


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
        model = model.to(DEVICE_TYPE)
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
    state_dict = cast(dict[str, torch.Tensor], olist[0])
    # Ensure that the state is on DEVICE
    for param_name in state_dict:
        state_dict[param_name] = state_dict[param_name].to(DEVICE_TYPE)
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


def _move_to_device(model: nn.Module, move_to_device: bool):
    return model.to(DEVICE_TYPE) if move_to_device else model


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
            future: torch.futures.Future = torch.futures.Future()
            future.set_result(1)
            return future

        dist_wait.get_future = get_future
        return dist_wait


class TransformerWithSharedParams(FSDPTestModel):
    def __init__(
        self,
        group: dist.ProcessGroup,
        device_init_mode: DEVICEInitMode,
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
            torch.zeros_like(self.vocab_bias, dtype=torch.long),  # type: ignore[arg-type]
        )  # type: ignore[arg-type]

        self.bs = 2
        self.bn = torch.nn.BatchNorm1d(self.bs) if add_bn else torch.nn.Identity()
        if device_init_mode == DEVICEInitMode.DEVICE_BEFORE:
            self = self.to(DEVICE_TYPE)
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
        device_init_mode: DEVICEInitMode,
        fsdp_kwargs: Optional[dict[str, Any]] = None,
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
            device_init_mode (DEVICEInitMode): Determines model movement to DEVICE.
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
                pg, device_init_mode, add_bn, deterministic
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
                tformer_pg, device_init_mode, add_bn, deterministic
            )
            fsdp_model = FSDP(
                m,
                fsdp_pg,
                auto_wrap_policy=auto_wrap_policy,
                **fsdp_kwargs,
            )
            if device_init_mode == DEVICEInitMode.DEVICE_AFTER:
                fsdp_model = fsdp_model.to(DEVICE_TYPE)
            return fsdp_model
        raise ValueError(f"Unsupported FSDP init mode: {fsdp_init_mode}")

    def get_ignored_modules(self):
        return [self.transformer]


class NestedWrappedModule(FSDPTestModel):
    def __init__(
        self,
        group: dist.ProcessGroup,
        wrap_fsdp: bool,
        device_init_mode: DEVICEInitMode,
        deterministic: bool,
        **fsdp_kwargs,
    ):
        super().__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        move_to_device = device_init_mode == DEVICEInitMode.DEVICE_BEFORE

        def _maybe_wrap(layer):
            if wrap_fsdp:
                return FSDP(layer, group, **fsdp_kwargs)
            return layer

        if deterministic:
            torch.manual_seed(0)
        self.module = nn.Sequential(
            _move_to_device(nn.Linear(8, 4), move_to_device),
            _maybe_wrap(
                nn.Sequential(
                    _maybe_wrap(_move_to_device(nn.Linear(4, 16), move_to_device)),
                    _move_to_device(nn.Linear(16, 16), move_to_device),
                ),
            ),
            _maybe_wrap(_move_to_device(nn.Linear(16, 4), move_to_device)),
            _move_to_device(nn.Linear(4, 8), move_to_device),
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
        device_init_mode: DEVICEInitMode,
        fsdp_kwargs: Optional[dict[str, Any]] = None,
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
            device_init_mode (DEVICEInitMode): Determines model movement to DEVICE.
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
                device_init_mode=device_init_mode,
                deterministic=deterministic,
            )
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            # Does not wrap with top-level FSDP
            fsdp_model = NestedWrappedModule(
                group,
                wrap_fsdp=True,
                device_init_mode=device_init_mode,
                deterministic=deterministic,
                **fsdp_kwargs,
            )
            if device_init_mode == DEVICEInitMode.DEVICE_AFTER:
                fsdp_model = fsdp_model.to(DEVICE_TYPE)
            return fsdp_model
        raise ValueError(f"Unsupported FSDP init mode: {fsdp_init_mode}")


class AlwaysWrapNestedWrappedModule(NestedWrappedModule):
    @staticmethod
    def init(
        group: dist.ProcessGroup,
        fsdp_init_mode: FSDPInitMode,
        device_init_mode: DEVICEInitMode,
        fsdp_kwargs: Optional[dict[str, Any]] = None,
        deterministic: bool = False,
    ):
        """
        Initializes a :class:`NestedWrappedModule` instance, but unlike
        :meth:`NestedWrappedModule.init`, for the ``RECURSIVE`` init mode, this
        wraps with top-level FSDP and the ``always_wrap_policy()`` auto wrap
        policy.
        """
        model = super(
            AlwaysWrapNestedWrappedModule, AlwaysWrapNestedWrappedModule
        ).init(
            group=group,
            fsdp_init_mode=FSDPInitMode.NO_FSDP,
            device_init_mode=device_init_mode,
            fsdp_kwargs=fsdp_kwargs,
            deterministic=deterministic,
        )
        if fsdp_init_mode == FSDPInitMode.NO_FSDP:
            return model
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            fsdp_kwargs = fsdp_kwargs or {}
            fsdp_model = FSDP(model, auto_wrap_policy=always_wrap_policy, **fsdp_kwargs)
            if device_init_mode == DEVICEInitMode.DEVICE_AFTER:
                fsdp_model = fsdp_model.to(DEVICE_TYPE)
            return fsdp_model


class NonUniformReqGradNWM(NestedWrappedModule):
    def __init__(
        self,
        group: dist.ProcessGroup,
        wrap_fsdp: bool,
        device_init_mode: DEVICEInitMode,
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
        move_to_device = device_init_mode == DEVICEInitMode.DEVICE_BEFORE

        def _maybe_wrap(layer):
            if wrap_fsdp:
                return FSDP(layer, group, **fsdp_kwargs)
            return layer

        if deterministic:
            torch.manual_seed(0)
        self.module = nn.Sequential(
            _move_to_device(nn.Linear(8, 4), move_to_device),
            _maybe_wrap(
                nn.Sequential(
                    _maybe_wrap(_move_to_device(nn.Linear(4, 16), move_to_device)),
                    _move_to_device(nn.Linear(16, 16), move_to_device),
                ),
            ),
            _maybe_wrap(
                nn.Sequential(
                    _move_to_device(nn.Linear(16, 4), move_to_device),
                    _move_to_device(nn.Linear(4, 8), move_to_device),
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
        device_init_mode: DEVICEInitMode,
        fsdp_kwargs: Optional[dict[str, Any]] = None,
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
                device_init_mode=device_init_mode,
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
                device_init_mode=device_init_mode,
                deterministic=deterministic,
                **fsdp_kwargs,
            )
            if device_init_mode == DEVICEInitMode.DEVICE_AFTER:
                fsdp_model = fsdp_model.to(DEVICE_TYPE)
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
        return self.module.get_input(device)  # type: ignore[operator]

    def forward(self, x):
        return self.module(x)

    def get_loss(self, input, output):
        loss = self.module.get_loss(input, output)  # type: ignore[operator]
        if self.delay_after_loss_ms > 0:
            if TEST_HPU or TEST_XPU:
                time.sleep(self.delay_after_loss_ms / 1000)
            elif TEST_CUDA:
                torch.cuda._sleep(int(self.delay_after_loss_ms * get_cycles_per_ms()))

        return loss

    def run_backward(self, loss):
        orig_reduce_scatter = torch.distributed.reduce_scatter_tensor

        def _delayed_reduce_scatter(*args, **kwargs):
            if self.delay_before_reduction_ms > 0:
                if TEST_CUDA:
                    torch.cuda._sleep(
                        int(self.delay_before_reduction_ms * get_cycles_per_ms())
                    )
                elif TEST_HPU or TEST_XPU:
                    time.sleep(self.delay_before_reduction_ms / 1000)
            return orig_reduce_scatter(*args, **kwargs)

        with mock.patch(
            "torch.distributed.reduce_scatter_tensor", _delayed_reduce_scatter
        ):
            self.module.run_backward(loss)  # type: ignore[operator]

    @staticmethod
    def init(
        module_class: type[FSDPTestModel],
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
    def init(  # type: ignore[override]
        group: dist.ProcessGroup,
        fsdp_init_mode: FSDPInitMode,
        device_init_mode: DEVICEInitMode = DEVICEInitMode.DEVICE_AFTER,
        fsdp_kwargs: Optional[dict[str, Any]] = None,
        deterministic: bool = False,
        delay_after_loss_ms: int = 0,
        delay_before_reduction_ms: int = 0,
    ):
        return ModuleWithDelay.init(
            NestedWrappedModule,
            group=group,
            fsdp_init_mode=fsdp_init_mode,
            device_init_mode=device_init_mode,
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
        device_init_mode: DEVICEInitMode,
        delay_before_free_ms: int,
        deterministic: bool,
        **fsdp_kwargs,
    ):
        super().__init__(
            group=group,
            wrap_fsdp=wrap_fsdp,
            device_init_mode=device_init_mode,
            deterministic=deterministic,
        )
        self.group = group
        self.delay_before_free_ms = delay_before_free_ms
        self.wrap_fsdp = wrap_fsdp
        self.move_to_device = device_init_mode == DEVICEInitMode.DEVICE_BEFORE
        if deterministic:
            # Give each rank different expert parameters
            torch.manual_seed(42 + self.rank)
        d_expert = 23
        d_shared = 12
        d_input = 8
        expert = _move_to_device(nn.Linear(d_expert, d_shared), self.move_to_device)

        self.num_expert_params = sum(p.numel() for p in expert.parameters())
        for p in expert.parameters():
            p.expert = True  # type: ignore[attr-defined]

        if deterministic:
            # Keep all other parameters the same across ranks
            torch.manual_seed(0)

        shared = _move_to_device(nn.Linear(d_shared, d_expert), self.move_to_device)

        if wrap_fsdp:
            # we create a process group of size 1 for the expert params
            expert_group = torch.distributed.new_group(
                [group.rank()]
            )  # world size 1 means no shard
            expert = FSDP(expert, expert_group, **fsdp_kwargs)  # type: ignore[assignment]
            shared = FSDP(shared, group, **fsdp_kwargs)  # type: ignore[assignment]

        self.module = nn.Sequential(
            _move_to_device(nn.Linear(d_input, d_shared), self.move_to_device),
            shared,
            expert,
            _move_to_device(nn.Linear(d_shared, d_input), self.move_to_device),
        )

    def forward(self, x):
        if self.delay_before_free_ms > 0:
            expert = self.module[2]
            if isinstance(expert, FSDP):
                orig_reshard = torch.distributed.fsdp._runtime_utils._reshard

                def _delayed_reshard(*args, **kwargs):
                    if TEST_CUDA:
                        torch.cuda._sleep(
                            int(self.delay_before_free_ms * get_cycles_per_ms())
                        )
                    elif TEST_HPU or TEST_XPU:
                        time.sleep(self.delay_before_free_ms / 1000)

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
                    if p.grad is not None:
                        p.grad.div_(self.world_size)
                        torch.distributed.all_reduce(p.grad, group=self.group)

    @staticmethod
    def init(
        group: dist.ProcessGroup,
        fsdp_init_mode: FSDPInitMode,
        device_init_mode: DEVICEInitMode,
        fsdp_kwargs: Optional[dict[str, Any]] = None,
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
            device_init_mode (DEVICEInitMode): Determines model movement to DEVICE.
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
                device_init_mode=device_init_mode,
                delay_before_free_ms=delay_before_free_ms,
                deterministic=deterministic,
            )
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            # Does not wrap with top-level FSDP
            fsdp_model = MixtureOfExperts(
                group,
                wrap_fsdp=True,
                device_init_mode=device_init_mode,
                delay_before_free_ms=delay_before_free_ms,
                deterministic=deterministic,
                **fsdp_kwargs,
            )
            if device_init_mode == DEVICEInitMode.DEVICE_AFTER:
                fsdp_model = fsdp_model.to(DEVICE_TYPE)
            return fsdp_model
        raise ValueError(f"Unsupported FSDP init mode: {fsdp_init_mode}")


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        device: Optional[torch.device] = None,
        *,
        bias: bool = True,
        with_buffer: bool = False,
        dim_multiplier: int = 4,
    ):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim_multiplier * dim, device=device, bias=bias)
        self.out_proj = nn.Linear(dim_multiplier * dim, dim, device=device, bias=bias)
        if with_buffer:
            self.register_buffer("buffer", torch.randn((dim,), device=device))
        else:
            self.buffer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        z = F.relu(z)
        z = self.out_proj(z)
        z = F.relu(z)
        if self.buffer is not None:
            z = z + self.buffer
        return z

    def reset_parameters(self):
        if self.buffer is not None:
            torch.nn.init.normal_(self.buffer)


class MLPStack(nn.Sequential):
    def __init__(self, mlp_dim: int, *, with_seq_parallel: bool = False):
        modules: list[nn.Module] = [
            # Use multiplier of 3 to exercise uneven case
            MLP(mlp_dim, dim_multiplier=3),
            MLP(mlp_dim),
            MLP(mlp_dim, dim_multiplier=3),
        ]
        if with_seq_parallel:
            modules.append(nn.LayerNorm(mlp_dim, bias=False))
        super().__init__(*modules)
        self.with_seq_parallel = with_seq_parallel

    def parallelize(
        self,
        tp_mesh: DeviceMesh,
        dp_mesh: DeviceMesh,
        use_activation_checkpointing: bool,
        **fsdp_kwargs,
    ) -> "MLPStack":
        parallelize_plan = {
            # Pass `use_local_output=False` to keep as DTensor to preserve
            # uneven activation dims
            "0.in_proj": ColwiseParallel(use_local_output=False),
            "0.out_proj": RowwiseParallel(use_local_output=False),
            "1.in_proj": ColwiseParallel(use_local_output=False),
            "1.out_proj": RowwiseParallel(use_local_output=False),
            "2.in_proj": ColwiseParallel(use_local_output=False),
            "2.out_proj": RowwiseParallel(output_layouts=Shard(1))
            if self.with_seq_parallel
            else RowwiseParallel(),
        }
        if self.with_seq_parallel:
            parallelize_plan["3"] = SequenceParallel(sequence_dim=1)
        parallelize_module(self, device_mesh=tp_mesh, parallelize_plan=parallelize_plan)
        for module in self:
            if isinstance(module, nn.LayerNorm):
                continue
            if use_activation_checkpointing:
                checkpoint(module)
            fully_shard(module, mesh=dp_mesh, **fsdp_kwargs)
        fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)
        return self


class DoubleLinear(nn.Module):
    """
    This can be used for returning multiple outputs from a module
    (``use_second_linear=True``) or for having an unused module (``False``).
    """

    def __init__(self, dim: int, use_second_linear: bool = True):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.use_second_linear = use_second_linear

    def forward(
        self, x: torch.Tensor
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.use_second_linear:
            return self.relu(self.lin1(x)), self.relu(self.lin2(x))
        return self.relu(self.lin1(x))


# NOTE: For these patch methods, if we want safety under multi-threading (e.g.
# when using multi-threaded process group), then we want:
# (1) a barrier immediately after reading the original value to ensure that all
# threads see the same original value
# (2) a barrier immediately before restoring the original value to ensure that
# all threads use the patched value inside the context
@contextlib.contextmanager
def patch_all_gather(new_all_gather_into_tensor: Callable):
    orig_all_gather = dist.all_gather_into_tensor
    dist.barrier()
    dist.all_gather_into_tensor = new_all_gather_into_tensor
    try:
        yield
    finally:
        dist.barrier()
        dist.all_gather_into_tensor = orig_all_gather


@contextlib.contextmanager
def patch_foreach_all_gather(new_foreach_all_gather: Callable):
    orig_foreach_all_gather = (
        torch.distributed.fsdp._fully_shard._fsdp_param_group.foreach_all_gather
    )
    dist.barrier()
    torch.distributed.fsdp._fully_shard._fsdp_param_group.foreach_all_gather = (
        new_foreach_all_gather
    )
    try:
        yield
    finally:
        dist.barrier()
        torch.distributed.fsdp._fully_shard._fsdp_param_group.foreach_all_gather = (
            orig_foreach_all_gather
        )


@contextlib.contextmanager
def patch_foreach_reduce(new_foreach_reduce: Callable):
    orig_foreach_foreach_reduce = (
        torch.distributed.fsdp._fully_shard._fsdp_param_group.foreach_reduce
    )
    dist.barrier()
    torch.distributed.fsdp._fully_shard._fsdp_param_group.foreach_reduce = (
        new_foreach_reduce
    )
    try:
        yield
    finally:
        dist.barrier()
        torch.distributed.fsdp._fully_shard._fsdp_param_group.foreach_reduce = (
            orig_foreach_foreach_reduce
        )


@contextlib.contextmanager
def patch_reduce_scatter(new_reduce_scatter_tensor: Callable):
    orig_reduce_scatter = dist.reduce_scatter_tensor
    dist.barrier()
    dist.reduce_scatter_tensor = new_reduce_scatter_tensor
    try:
        yield
    finally:
        dist.barrier()
        dist.reduce_scatter_tensor = orig_reduce_scatter


@contextlib.contextmanager
def patch_all_reduce(new_all_reduce: Callable):
    orig_all_reduce = dist.all_reduce
    dist.barrier()
    dist.all_reduce = new_all_reduce
    try:
        yield
    finally:
        dist.barrier()
        dist.all_reduce = orig_all_reduce


@no_type_check
@contextlib.contextmanager
def patch_unshard(new_unshard: Callable):
    orig_unshard = FSDPParamGroup.unshard
    dist.barrier()
    FSDPParamGroup.unshard = new_unshard
    try:
        yield
    finally:
        dist.barrier()
        FSDPParamGroup.unshard = orig_unshard


@no_type_check
@contextlib.contextmanager
def patch_reshard(new_reshard: Callable):
    orig_reshard = FSDPParamGroup.reshard
    dist.barrier()
    FSDPParamGroup.reshard = new_reshard
    try:
        yield
    finally:
        dist.barrier()
        FSDPParamGroup.reshard = orig_reshard


@no_type_check
@contextlib.contextmanager
def patch_post_backward(new_post_backward: Callable):
    orig_post_backward = FSDPParamGroup.post_backward
    dist.barrier()
    FSDPParamGroup.post_backward = new_post_backward
    try:
        yield
    finally:
        dist.barrier()
        FSDPParamGroup.post_backward = orig_post_backward


@no_type_check
@contextlib.contextmanager
def patch_register_post_backward_hook_backward(new_backward: Callable):
    orig_backward = RegisterPostBackwardFunction.backward
    dist.barrier()
    RegisterPostBackwardFunction.backward = new_backward
    try:
        yield
    finally:
        dist.barrier()
        RegisterPostBackwardFunction.backward = orig_backward


def reduce_scatter_with_assert(
    cls,
    orig_reduce_scatter: Callable,
    assert_fn: Callable,  # `assert_fn(output: Tensor)`
    *args: Any,
    **kwargs: Any,
):
    if len(args) > 0:
        output = args[0]
    elif "output" in kwargs:
        output = kwargs["output"]
    else:
        raise AssertionError(
            f"Cannot get reduce-scatter output from\nargs: {args}\nkwargs: {kwargs}"
        )
    assert_fn(output)
    return orig_reduce_scatter(*args, **kwargs)


def check_sharded_parity(
    cls,  # unit test class
    replicated_module: nn.Module,
    sharded_module: nn.Module,
    prefixes_to_ignore: tuple[str, ...] = (),
):
    for (replicated_name, replicated_param), (sharded_name, sharded_param) in zip(
        replicated_module.named_parameters(),
        sharded_module.named_parameters(),
        strict=True,
    ):
        clean_sharded_name = sharded_name
        for prefix in prefixes_to_ignore:
            clean_sharded_name = clean_sharded_name.replace(prefix, "")
        cls.assertEqual(replicated_name, clean_sharded_name)
        cls.assertIsInstance(sharded_param, DTensor)
        assert isinstance(sharded_param, DTensor)  # mypy
        mesh, placements = sharded_param.device_mesh, sharded_param.placements
        if tuple(placements) == (Shard(0), Shard(0)):
            raise AssertionError(
                "FSDP's (Shard(0), Shard(0)) layout differs from distribute_tensor(), "
                "so we cannot check for equality using it"
            )
        sharded_ref_param = distribute_tensor(replicated_param, mesh, placements)
        cls.assertEqual(sharded_param.to_local(), sharded_ref_param.to_local())
        if replicated_param.grad is None:
            cls.assertIsNone(sharded_param.grad)
            continue
        cls.assertIsNotNone(sharded_param.grad)
        sharded_ref_grad = distribute_tensor(replicated_param.grad, mesh, placements)
        cls.assertIsInstance(sharded_param.grad, DTensor)
        assert isinstance(sharded_param.grad, DTensor)  # mypy
        cls.assertEqual(sharded_param.grad.to_local(), sharded_ref_grad.to_local())


@unittest.skipIf(TEST_XPU, "not-support-multithread")
class FSDPTestMultiThread(MultiThreadedTestCase):
    @property
    def world_size(self):
        return DEVICE_COUNT

    def setUp(self):
        super().setUp()
        self._spawn_threads()

    def run_subtests(self, *args, **kwargs):
        return run_subtests(self, *args, **kwargs)

    def perThreadSetUp(self):
        torch._dynamo.reset()

    def perThreadTearDown(self):
        torch._dynamo.reset()


class FSDPTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        # Set TORCH_NCCL_DESYNC_DEBUG=0 to disable the NCCL `workCleanupLoop()`,
        # which can cause unit test flakiness:
        # https://github.com/pytorch/pytorch/issues/90848
        os.environ["TORCH_NCCL_DESYNC_DEBUG"] = "0"
        self._spawn_processes()

    @property
    def world_size(self):
        return DEVICE_COUNT

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def destroy_pg_upon_exit(self) -> bool:
        # Overriding base test class: do not auto destroy PG upon exit.
        return False

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
    def _run(cls, rank, test_name, file_name, pipe, **kwargs):  # type: ignore[override]
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        fake_pg = kwargs.get("fake_pg", False)

        print(f"dist init r={self.rank}, world={self.world_size}")
        if torch.accelerator.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        # Specify gloo backend to make 'init_process_group()' succeed,
        # Actual tests will be skipped if there is no enough GPUs.
        try:
            if fake_pg:
                store = torch.testing._internal.distributed.fake_pg.FakeStore()
                dist.init_process_group(
                    backend="fake",
                    world_size=self.world_size,
                    rank=rank,
                    store=store,
                )
            else:
                dist.init_process_group(
                    init_method=self.init_method,
                    backend=DISTRIBUTED_BACKEND,
                    world_size=int(self.world_size),
                    rank=self.rank,
                )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        device_ids = None
        device_id = self.rank % DEVICE_COUNT
        if TEST_CUDA or TEST_XPU:
            torch.accelerator.set_device_index(device_id)
        device_ids = [device_id]

        # Execute barrier prior to running test to ensure that every process
        # has finished initialization and that the following test
        # immediately exiting due to a skip doesn't cause flakiness.
        dist.barrier(device_ids=device_ids)

        torch._dynamo.reset()
        set_rng_seed()
        self.run_test(test_name, pipe)
        torch._dynamo.reset()

        dist.barrier(device_ids=device_ids)

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
        sharded_grad_scaler_kwargs: Optional[dict[str, Any]] = None,
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
            with torch.amp.autocast(DEVICE_TYPE, enabled=autocast):
                # Inputs always cuda regardless of cpu offloading, or model.device
                input = model.module.get_input(torch.device(DEVICE_TYPE))  # type: ignore[operator, union-attr]
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

                loss = model.module.get_loss(input, output).to(model_device)  # type: ignore[operator, union-attr]
            loss = sharded_grad_scaler.scale(loss)

            if not mixed_precision and not use_pure_fp16:
                assert loss.dtype == torch.float32, (
                    "loss data type should be float32, as the original \
                    parameter data type is float32."
                )
            else:
                if use_pure_fp16:
                    self.assertEqual(loss.dtype, torch.float16)
                # FSDP loss is fp16, DDP AMP loss is fp32
                elif isinstance(model, FSDP):
                    assert mixed_precision is not None  # mypy
                    self.assertEqual(loss.dtype, mixed_precision.param_dtype)
                else:
                    self.assertEqual(loss.dtype, torch.float32)
            model.module.run_backward(loss)  # type: ignore[operator, union-attr]
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
        return loss.detach()  # type: ignore[possibly-undefined]

    def _test_fsdp_parity(
        self,
        model_class: type[FSDPTestModel],
        fsdp_init_mode: FSDPInitMode,
        device_init_mode: DEVICEInitMode,
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
        init_kwargs: Optional[dict[str, Any]] = None,
        sharded_grad_scaler_kwargs: Optional[dict[str, Any]] = None,
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
        assert fsdp_init_mode != FSDPInitMode.NO_FSDP, (
            "Expects an FSDP init mode that wraps with FSDP"
        )
        if init_kwargs is None:
            init_kwargs = {}
        lr = 1e-2
        rank = self.process_group.rank()
        # Establish reference behavior with DDP
        model = model_class.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            DEVICEInitMode.DEVICE_BEFORE,
            deterministic=True,
            **init_kwargs,
        )
        if ref_init_fn is None:
            if TEST_HPU:
                ref_model = DDP(
                    model, device_ids=[DEVICE_TYPE], output_device=DEVICE_TYPE
                )
            else:
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
                device_init_mode,
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
        if device_init_mode == DEVICEInitMode.DEVICE_AFTER:
            fsdp_model = fsdp_model.to(DEVICE_TYPE)
        offload_params = cpu_offload is not None and cpu_offload.offload_params
        # Offloading parameters with `DEVICE_AFTER` should raise an error during
        # lazy initialization due to the parameter devices not being CPU;
        # otherwise, all parameter devices should be CPU
        expects_device_error = (
            offload_params and device_init_mode == DEVICEInitMode.DEVICE_AFTER
        )
        expects_cpu_device = (
            offload_params and device_init_mode != DEVICEInitMode.DEVICE_AFTER
        )
        if expects_cpu_device:
            cpu_device = torch.device("cpu")
            for param in fsdp_model.parameters():
                self.assertEqual(param.device, cpu_device)
        context = (
            self.assertRaisesRegex(
                RuntimeError,
                "An FSDP-managed module with parameter CPU offloading enabled "
                f"has parameters on {DEVICE_TYPE}",
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
            cpu_device = torch.device("cpu")
            for param in fsdp_model.parameters():
                self.assertEqual(param.device, cpu_device)
            fsdp_loss = fsdp_loss.to(DEVICE_TYPE)
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


def compiled_fsdp_test(compile_compute_on_module: Optional[type] = None):
    def fully_shard_with_compiled_compute(*args, **kwargs):
        torch.distributed.fsdp.fully_shard(*args, **kwargs)  # type: ignore[operator]
        if compile_compute_on_module is None or isinstance(
            args[0], compile_compute_on_module
        ):
            args[0].compile()

    class FullyShardMode(Enum):
        EAGER = auto()
        COMPILED_COMPUTE = auto()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_fully_shard: Any = torch.distributed.fsdp.fully_shard
            for mode in FullyShardMode:
                if mode != FullyShardMode.EAGER and not has_triton():
                    warnings.warn(
                        "Inductor on GPU needs Triton and recent GPU arch", stacklevel=2
                    )
                    continue
                # barrier to ensure thread reading the same value
                original_skip_fsdp_hooks = torch._dynamo.config.skip_fsdp_hooks
                original_compile_threads = torch._inductor.config.compile_threads
                torch.distributed.barrier()

                if mode == FullyShardMode.EAGER:
                    fully_shard_patch = original_fully_shard
                elif mode == FullyShardMode.COMPILED_COMPUTE:
                    torch._dynamo.config.skip_fsdp_hooks = True
                    torch._inductor.config.compile_threads = 1
                    fully_shard_patch = fully_shard_with_compiled_compute  # type: ignore[assignment]
                else:
                    raise NotImplementedError(
                        f"Need to implement FullyShardMode={mode}"
                    )

                # fully_shard is imported as a global
                # through `from ... import fully_shard`
                func.__globals__[original_fully_shard.__name__] = fully_shard_patch
                func(*args, **kwargs)
                # other threads use patched func before this thread restores
                torch.distributed.barrier()
                func.__globals__[original_fully_shard.__name__] = original_fully_shard
                torch._dynamo.config.skip_fsdp_hooks = original_skip_fsdp_hooks
                torch._inductor.config.compile_threads = original_compile_threads

        return wrapper

    return decorator


class SkipModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        return self.lin(x)


class NestedLinear(nn.Module):
    def __init__(self, fsdp_wrap):
        super().__init__()
        if fsdp_wrap:
            self.nested_linear = wrap(nn.Linear(10, 10, bias=False).to(DEVICE_TYPE))
        else:
            self.nested_linear = nn.Linear(10, 10, bias=False).to(DEVICE_TYPE)

    def forward(self, x):
        return self.nested_linear(x)


class SkipModel(nn.Module):
    def __init__(self, double_nest):
        super().__init__()
        self.linear = nn.Linear(10, 10, bias=False).to(DEVICE_TYPE)
        self.linear_skip = SkipModule().to(DEVICE_TYPE)
        self.nested_linear = wrap(
            NestedLinear(fsdp_wrap=double_nest), device_id=DEVICE_TYPE
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.linear_skip(x)
        x = self.nested_linear(x)
        return x
