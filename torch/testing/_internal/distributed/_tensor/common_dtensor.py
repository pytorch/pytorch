# mypy: allow-untyped-defs

# Copyright (c) Meta Platforms, Inc. and affiliates

import contextlib
import copy
import functools
import itertools
import sys
import types
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, cast, Optional, TypeVar, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._local_tensor import (
    LocalIntNode,
    LocalTensor,
    LocalTensorMode,
    maybe_disable_local_tensor_mode,
    maybe_run_for_local_tensor,
)
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor.placement_utils import ShardOrderEntry
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    MultiProcessTestCase,
    MultiThreadedTestCase,
    run_subtests,
    skip_if_lt_x_gpu,
    TEST_SKIPS,
)
from torch.testing._internal.common_utils import (
    TEST_CUDA,
    TEST_HPU,
    TEST_PRIVATEUSE1,
    TEST_XPU,
)
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec


DEVICE_COUNT: int

if TEST_CUDA or TEST_XPU or TEST_HPU or TEST_PRIVATEUSE1:
    DEVICE_TYPE = torch.accelerator.current_accelerator().type
    DEVICE_COUNT = torch.accelerator.device_count()
    PG_BACKEND = dist.Backend.default_device_backend_map[DEVICE_TYPE]
else:
    DEVICE_TYPE = "cpu"
    PG_BACKEND = "gloo"

NUM_DEVICES = 4

# We use this as a proxy for "multiple GPUs exist"
if (TEST_CUDA or TEST_XPU or TEST_HPU or TEST_PRIVATEUSE1) and DEVICE_COUNT > 1:
    # when we actually have multiple GPUs, relax the requirement to smaller counts.
    NUM_DEVICES = min(NUM_DEVICES, DEVICE_COUNT)

T = TypeVar("T")


# simple RMSNorm layer for testing
class RMSNormPython(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x)
        return output * self.weight


class MLPModule(nn.Module):
    def __init__(self, device, bias: bool = True):
        super().__init__()
        torch.manual_seed(5)
        self.net1 = nn.Linear(10, 16, bias=bias, device=device)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 10, bias=bias, device=device)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

    def reset_parameters(self):
        self.net1.reset_parameters()
        self.net2.reset_parameters()


class MLPStacked(nn.Module):
    def __init__(self, device, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([MLPModule(device) for i in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass
class ModelArgs:
    n_layers: int = 2
    vocab_size: int = 8
    max_seq_len: int = 16
    dim: int = 16
    n_heads: int = 4
    dropout_p: float = 0.1
    use_attn_mask: bool = True
    weight_tying: bool = True
    checkpoint_activations: bool = False


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.dim % args.n_heads == 0
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads
        self.dropout_p = args.dropout_p
        self.resid_dropout = nn.Dropout(args.dropout_p)
        self.use_attn_mask = args.use_attn_mask

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

    def forward(self, x):
        bsz, seq_len, _ = x.size()
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
        queries = queries.view(bsz, seq_len, self.n_heads, self.head_dim)
        keys = keys.view(bsz, seq_len, self.n_heads, self.head_dim)
        values = values.view(bsz, seq_len, self.n_heads, self.head_dim)

        queries = queries.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        values = values.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)

        output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            None,
            self.dropout_p if self.training else 0,
            self.use_attn_mask,
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.resid_dropout(self.wo(output))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_p):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.w2 = nn.Linear(hidden_dim, dim)
        self.resid_dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.resid_dropout(self.w2(self.gelu(self.w1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention_norm = nn.LayerNorm(args.dim)
        self.attention = Attention(args)
        self.ffn_norm = nn.LayerNorm(args.dim)
        self.feed_forward = FeedForward(
            args.dim, hidden_dim=4 * args.dim, dropout_p=args.dropout_p
        )

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# A toy transformer model, partly inspired by the nanoGPT model:
# https://github.com/karpathy/nanoGPT.
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size is not None
        assert args.max_seq_len is not None
        self.model_args = args
        self.max_seq_len = args.max_seq_len
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.pos_embeddings = nn.Embedding(args.max_seq_len, args.dim)
        self.dropout = nn.Dropout(args.dropout_p)
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))
        self.norm = nn.LayerNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        if args.weight_tying:
            self.output.weight = self.tok_embeddings.weight
        self.checkpoint_activations = args.checkpoint_activations

    def forward(self, tokens):
        _bsz, seq_len = tokens.size()
        assert seq_len <= self.max_seq_len
        h = self.tok_embeddings(tokens)
        pos = torch.arange(0, seq_len, device=tokens.device)
        p = self.pos_embeddings(pos)  # positional embeddings of shape (seq_len, dim)
        h = h + p
        h = self.dropout(h)
        for layer in self.layers:
            if self.checkpoint_activations:
                h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        h = self.norm(h)
        output = self.output(h).float()
        return output

    @staticmethod
    def parallelize(
        module: "Transformer",
        device_mesh: DeviceMesh,
        use_seq_parallel: bool,
        local_output_for_attn: bool = False,
    ) -> nn.Module:
        assert isinstance(module, Transformer), f"Requires Transformer but got {module}"
        # Parallelize the root submodules.
        if use_seq_parallel:
            root_plan = {
                "tok_embeddings": RowwiseParallel(
                    input_layouts=Replicate(), output_layouts=Shard(1)
                ),
                "pos_embeddings": RowwiseParallel(
                    input_layouts=Replicate(), output_layouts=Shard(0)
                ),
                "norm": SequenceParallel(),
            }
        else:
            root_plan = {
                "tok_embeddings": RowwiseParallel(
                    input_layouts=Replicate(), output_layouts=Replicate()
                ),
                "pos_embeddings": RowwiseParallel(
                    input_layouts=Replicate(), output_layouts=Replicate()
                ),
            }

        module_tp = parallelize_module(module, device_mesh, root_plan)
        # Parallelize the attention and feed forward submodules.
        for layer in module_tp.layers:
            layer_parallelize_plan = {}
            if use_seq_parallel:
                layer_parallelize_plan["attention"] = PrepareModuleInput(
                    input_layouts=Shard(1),
                    desired_input_layouts=Replicate(),
                )
                # shard the RMSNorms
                layer_parallelize_plan["attention_norm"] = SequenceParallel()
                layer_parallelize_plan["ffn_norm"] = SequenceParallel()
            layer_parallelize_plan["attention.wq"] = ColwiseParallel(
                use_local_output=local_output_for_attn
            )
            layer_parallelize_plan["attention.wk"] = ColwiseParallel(
                use_local_output=local_output_for_attn
            )
            layer_parallelize_plan["attention.wv"] = ColwiseParallel(
                use_local_output=local_output_for_attn
            )
            layer_parallelize_plan["attention.wo"] = (
                RowwiseParallel(output_layouts=Shard(1))
                if use_seq_parallel
                else RowwiseParallel()
            )

            layer_parallelize_plan["feed_forward.w1"] = (
                ColwiseParallel(input_layouts=Shard(1))
                if use_seq_parallel
                else ColwiseParallel()
            )
            layer_parallelize_plan["feed_forward.w2"] = (
                RowwiseParallel(output_layouts=Shard(1))
                if use_seq_parallel
                else RowwiseParallel()
            )

            parallelize_module(layer, device_mesh, layer_parallelize_plan)

        # Parallelize the output submodule. If weight tying is enabled, we need to
        # make sure output.weight is sharded consistently as tok_embeddings.weight,
        # at the cost of the all_reduce operation using RowwiseParallel.
        output_parallelize_plan = (
            ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),
            )
            if use_seq_parallel
            else ColwiseParallel(output_layouts=Replicate())
        )
        parallelize_module(module_tp.output, device_mesh, output_parallelize_plan)

        if local_output_for_attn:
            for layer in module_tp.layers:
                layer.attention.n_heads = (
                    module_tp.model_args.n_heads // device_mesh.size()
                )

        # Manually set output.weight so that parameters and gradients are shared.
        if module_tp.model_args.weight_tying:
            module_tp.output.weight = module_tp.tok_embeddings.weight

        return module_tp


def skip_unless_torch_gpu(method: T) -> T:
    """
    Test decorator which skips the test unless there's a GPU available to torch.

    >>> # xdoctest: +SKIP
    >>> @skip_unless_torch_gpu
    >>> def test_some_method(self) -> None:
    >>>   ...
    """
    # The builtin @skip_if_no_gpu relies on os.environ['WORLD_SIZE'] being set.
    return cast(T, skip_if_lt_x_gpu(NUM_DEVICES)(method))


class DTensorContinuousTestBase(MultiProcContinuousTest):
    @classmethod
    def device_type(cls) -> str:
        # if enough GPU/XPU/HPU we can use those devices, otherwise we fallback to CPU
        if (
            not (TEST_CUDA or TEST_XPU or TEST_HPU or TEST_PRIVATEUSE1)
            or DEVICE_COUNT < cls.world_size
        ):
            return "cpu"
        else:
            return DEVICE_TYPE

    @classmethod
    def backend_str(cls) -> str:
        backend = dist.get_default_backend_for_device(DEVICE_TYPE)
        return backend


class DTensorTestBase(MultiProcessTestCase):
    @property
    def is_local_tensor_enabled(self) -> bool:
        return False

    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    @property
    def device_type(self) -> str:
        # if enough GPU/XPU/HPU we can use those devices, otherwise we fallback to CPU
        if (
            not (TEST_CUDA or TEST_XPU or TEST_HPU or TEST_PRIVATEUSE1)
            or DEVICE_COUNT < self.world_size
        ):
            return "cpu"
        else:
            return DEVICE_TYPE

    @property
    def backend(self) -> str:
        backend = dist.get_default_backend_for_device(self.device_type)
        return backend

    def init_manual_seed_for_rank(self) -> None:
        torch.manual_seed(self.rank)

    def build_device_mesh(self) -> DeviceMesh:
        return init_device_mesh(self.device_type, (self.world_size,))

    def init_pg(self, eager_init, backend: Optional[str] = None) -> None:
        if "nccl" in self.backend and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        curr_backend = dist.get_default_backend_for_device(self.device_type)

        if backend is None:
            backend = self.backend

        if backend not in [
            "nccl",
            "gloo",
            "mpi",
            f"cpu:gloo,{self.device_type}:{curr_backend}",
            "hccl",
            "xccl",
            "fake",
            "cpu:gloo,xpu:xccl",
        ]:
            raise RuntimeError(f"Backend {backend} not supported!")

        device_id = None
        if "nccl" in backend or "xccl" in backend:
            # set device for nccl pg for collectives
            # TODO: if users want to enable testing across hosts, we may need
            # to change this part.
            torch.accelerator.set_device_index(self.rank)
            # we only need to set device_id for nccl backend with eager init
            device_id = (
                torch.device(f"{self.device_type}:{self.rank}") if eager_init else None
            )

        # For nccl backend, bind the device to the process if device_id is not None
        # so the nccl communicator is immediately formed and we can use `ncclCommSplit`
        # for form subgroup to avoid unnecessary overhead.
        dist.init_process_group(
            backend=backend,
            world_size=self.world_size,
            rank=self.rank,  # pyre-ignore[16]
            init_method=f"file://{self.file_name}",  # pyre-ignore[16]
            device_id=device_id,
        )

    def destroy_pg(self, device_id: Optional[int] = None) -> None:
        # Wait for all ranks to reach here before starting shutdown.
        # FIXME dist.barrier deadlocks with multiple threads and NCCL: https://github.com/pytorch/pytorch/issues/95895
        # dist.all_reduce(torch.zeros((1,), device="cuda" if TEST_CUDA else "cpu"))
        # FIXME can't use the above all_reduce as it causes hangs on bionic and focal. It hangs:
        #  test_dtensor.py  -- DTensorMeshTest.test_dtensor_device_mesh_device_conversion
        if device_id is None:
            device_id = (
                torch.cuda.current_device() if self.device_type == "cuda" else self.rank
            )

        if self.device_type == "cpu":
            # NOTE: when `device_id` is not None, barrier() will choose the accelerator
            # of the most pripority, which means if the test specifies to use CPU for
            # testing while CUDA is available on the host, the barrier() will use CUDA.
            # To avoid this and better respect `self.device_type`, we add this branch to
            # enforce barrier() to use CPU when `self.device_type` is CPU and other
            # accelerator is also available.
            dist.barrier()
        else:
            dist.barrier(device_ids=[device_id])

        dist.destroy_process_group()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def _test_op_on_dtensor(self, op_call, *args, **kwargs) -> None:
        """
        This function checks ``op_call(dtensor).full_tensor() == op_call(dtensor.full_tensor())``.
        Unlike _test_op where the DTensor sharding is generated by DTensorConverter,
        this function takes in DTensor object directly as argument and test the equality
        of calling op on full_tensor() and DTensor.
        """
        # call full_tensor() on DTensor args/kwargs
        args_flattened, args_spec = tree_flatten(args)
        full_tensor_args_flattened = tuple(
            arg.full_tensor().detach().clone() if isinstance(arg, DTensor) else arg
            for arg in args_flattened
        )
        full_tensor_args = tree_unflatten(full_tensor_args_flattened, args_spec)
        full_tensor_kwargs = {
            k: v.full_tensor() if isinstance(v, DTensor) else v
            for k, v in kwargs.items()
        }

        out_flattened, _ = tree_flatten(
            op_call(*full_tensor_args, **full_tensor_kwargs)
        )
        d_out_flattened, _ = tree_flatten(op_call(*args, **kwargs))
        d_out_full_tensor_flattened = [dt.full_tensor() for dt in d_out_flattened]
        self.assertEqual(out_flattened, d_out_full_tensor_flattened)

    # pyre-ignore[2]:
    def _test_op(self, mesh: DeviceMesh, op_call, *args, **kwargs) -> None:
        out = op_call(*args, **kwargs)
        dtc = DTensorConverter(mesh, args, kwargs)
        for d_args, d_kwargs in dtc:
            # pyre can't find assertTrue anymore?
            self.assertEqual(dtc.successful(), True)
            d_out = op_call(*d_args, **d_kwargs)
            self.assertEqual(d_out.full_tensor(), out)

    def run_subtests(self, *args, **kwargs):
        return run_subtests(self, *args, **kwargs)


TestFunc = Callable[[...], object]


# wrapper to initialize comms (processgroup)
def with_comms(
    eager_init: Union[TestFunc, bool] = False, backend: Optional[str] = None
) -> TestFunc:
    def decorator(func, eager_init: bool = False, backend: Optional[str] = None):
        @wraps(func)  # pyre-ignore[6]
        def wrapper(
            self,
            *args: tuple[object],
            **kwargs: dict[str, Any],  # type: ignore[misc]
        ) -> None:
            self.init_pg(eager_init, backend)

            try:
                func(self, *args, **kwargs)  # type: ignore[misc]
            except Exception as e:
                dist.destroy_process_group()
                raise e

            self.destroy_pg()

        return wrapper

    return (
        decorator(func=eager_init)
        if callable(eager_init)
        else partial(decorator, eager_init=eager_init, backend=backend)
    )


class DTensorOpTestBase(MultiThreadedTestCase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    @property
    def device_type(self) -> str:
        return DEVICE_TYPE

    def build_device_mesh(self):
        return init_device_mesh(self.device_type, (self.world_size,))

    def setUp(self) -> None:
        super().setUp()
        self._spawn_threads()


# This is a class for converting args/kwargs of an op into distributed args/kwargs
class DTensorConverter:
    def __init__(
        self,
        mesh: DeviceMesh,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        self.hit = 0
        self.miss = 0
        self.mesh = mesh
        self.args = args
        self.kwargs = kwargs
        flatten_args, flatten_args_spec = tree_flatten(args)
        flatten_kwargs, flatten_kwargs_spec = tree_flatten(kwargs)

        self.flatten_args: list[object] = flatten_args
        self.flatten_args_spec: TreeSpec = flatten_args_spec
        self.flatten_kwargs: list[object] = flatten_kwargs
        self.flatten_kwargs_spec: TreeSpec = flatten_kwargs_spec

        choices_for_args = [
            self.gen_sharding_choices_for_arg(arg)
            for arg in self.flatten_args
            if isinstance(arg, torch.Tensor)
        ]

        choices_for_args.extend(
            self.gen_sharding_choices_for_arg(arg)
            for arg in self.flatten_kwargs
            if isinstance(arg, torch.Tensor)
        )

        self.sharding_combs: Iterator[Sequence[Placement]] = iter(
            itertools.product(*choices_for_args)
        )

    def successful(self) -> bool:
        return self.hit > 0 and self.miss == 0

    def is_supported_tensor(self, t: torch.Tensor) -> bool:
        # TODO: dist tensor need to support quantized and sparse
        # tensors, quantized tensor might be relatively easy, but
        # sparse tensor have special layouts that we need to possibly
        # deal with, until we are clear about them, we don't officially
        # support them.
        return not any(
            [
                t.is_sparse_csr,
                t.is_sparse,
                t.is_mkldnn,
                t.is_quantized,
                t.is_nested,
                torch._is_functional_tensor(t),
                t.is_neg(),
                t.is_conj(),
                t.device.type in ("lazy", "meta"),
                # We need a way to test if a tensor is batched but there
                # is no official APi to do it
                # torch._C._is_batched(t),
            ]
        )

    def gen_sharding_choices_for_arg(self, arg: torch.Tensor) -> Sequence[Placement]:
        mesh_size = self.mesh.size()
        sharding_choices: list[Placement] = [Replicate()]
        # c10d collective does not support bool tensor
        # for bool tensor we treat it as replicated
        if arg.dtype != torch.bool:
            # only generating choices with: replicate, or sharding
            # evenly on a dimension that could be sharded
            sharding_choices = sharding_choices + [
                Shard(i)
                for i, s in enumerate(arg.shape)
                if s > 1 and s % mesh_size == 0
            ]
        # TODO: add multi mesh choices
        # all_choices = itertools.product(
        #     *(self.mesh.ndim * [sharding_choices])
        # )
        return sharding_choices

    def __iter__(self) -> "DTensorConverter":
        return self

    def __next__(self) -> tuple[tuple[object, ...], dict[str, object]]:
        try:
            next_sharding_choices = next(self.sharding_combs)
            idx = 0

            new_args: list[object] = []
            for arg in self.flatten_args:
                if isinstance(arg, torch.Tensor):
                    new_args.append(
                        self.to_dist_tensor(
                            arg, self.mesh, [next_sharding_choices[idx]]
                        )
                    )
                    idx += 1
                else:
                    new_args.append(arg)

            new_kwargs: list[object] = []
            for arg in self.flatten_kwargs:
                if isinstance(arg, torch.Tensor):
                    new_kwargs.append(
                        self.to_dist_tensor(
                            arg, self.mesh, [next_sharding_choices[idx]]
                        )
                    )
                    idx += 1
                else:
                    new_kwargs.append(arg)

            return (
                tree_unflatten(new_args, self.flatten_args_spec),
                tree_unflatten(new_kwargs, self.flatten_kwargs_spec),
            )
        except StopIteration as e:
            raise StopIteration from e

    def to_dist_tensor(
        self, t: torch.Tensor, mesh: DeviceMesh, placements: list[Placement]
    ) -> torch.Tensor:
        if type(t) is torch.Tensor or type(t) is nn.Parameter or type(t) is LocalTensor:
            if self.is_supported_tensor(t):
                self.hit += 1
                if t.ndim == 0:
                    # scalar tensor by default will be replicated
                    r = distribute_tensor(t, mesh, [Replicate()] * mesh.ndim)
                else:
                    # distribute non-scalar tensors
                    r = distribute_tensor(t, mesh, placements)
                if isinstance(t, nn.Parameter):
                    r = nn.Parameter(  # type: ignore[assignment]
                        r, requires_grad=r.requires_grad
                    )
                return r
            else:
                self.miss += 1
                return t
        elif torch.overrides.is_tensor_like(t):
            # Blindly converting tensor subclasses to dist tensor can cause
            # unpredictable problems, we explicitly disable this conversion
            # for now (i.e. we don't support DTensor holding tensor subclass
            # until there's a strong reason later).
            self.miss += 1
            return t
        else:
            raise RuntimeError(f"Trying to convert to DTensor, but got {type(t)}")


class LocalDTensorTestBase(DTensorTestBase):
    @property
    def is_local_tensor_enabled(self) -> bool:
        return True

    def _handle_test_skip(self, msg: str) -> None:
        self.skipTest(msg)

    def _get_local_tensor_mode(self):
        return LocalTensorMode(frozenset(range(self.world_size)))

    def setUp(self) -> None:
        super().setUp()
        torch.autograd._enable_record_function(False)

    def tearDown(self) -> None:
        from torch.distributed.tensor import _random as random

        random._rng_tracker = None
        super().tearDown()
        torch.autograd._enable_record_function(True)

    @property
    def rank(self):
        return torch.SymInt(LocalIntNode({r: r for r in range(self.world_size)}))

    @rank.setter
    def rank(self, rank):
        pass

    def join_or_run(self, fn):
        @wraps(fn)
        def wrapper(self):
            fn()

        return types.MethodType(wrapper, self)

    def build_device_mesh(self) -> DeviceMesh:
        with maybe_disable_local_tensor_mode():
            return super().build_device_mesh()

    def init_pg(self, eager_init, backend: Optional[str] = None) -> None:
        dist.init_process_group("fake", rank=0, world_size=self.world_size)
        self._pg = dist.distributed_c10d._get_default_group()

    def destroy_pg(self, device_id: Optional[int] = None) -> None:
        dist.destroy_process_group(self._pg)
        self._pg = None

    def _spawn_processes(self) -> None:
        pass

    def run_test(self, test_name: str, parent_pipe) -> None:
        getattr(self, test_name)()

    def init_manual_seed_for_rank(self) -> None:
        torch.manual_seed(0)


def make_wrapped(fn, ctxs):
    @functools.wraps(fn)
    def wrapped(self):
        torch._dynamo.reset()
        stack = contextlib.ExitStack()
        for ctx in ctxs:
            if callable(ctx):
                stack.enter_context(ctx(self))
            else:
                stack.enter_context(ctx)
        try:
            out = fn(self)
        finally:
            stack.close()
        return out

    return wrapped


def create_local_tensor_test_class(orig_cls, skipped_tests=None):
    if skipped_tests is None:
        skipped_tests = []

    dct = orig_cls.__dict__.copy()
    for name in list(dct.keys()):
        fn = dct[name]
        if not callable(fn):
            continue
        elif name in skipped_tests:
            dct[name] = lambda self: self.skipTest("Skipped test")
        elif name.startswith("test_"):
            ctxs = [
                lambda test: test._get_local_tensor_mode(),
            ]
            dct[name] = make_wrapped(fn, ctxs)

    cls = type(
        orig_cls.__name__ + "WithLocalTensor",
        (LocalDTensorTestBase,) + orig_cls.__bases__,
        dct,
    )
    cls.__file__ = __file__
    return cls


@maybe_run_for_local_tensor
def map_local_tensor_for_rank(tensor, rank, func):
    return func(tensor, rank)


@maybe_run_for_local_tensor
def map_local_for_rank(rank, func):
    return func(rank)


def reduce_local_int(val, func):
    return func(val.node._local_ints)


def _convert_shard_order_dict_to_ShardOrder(shard_order):
    """Convert shard_order dict to ShardOrder"""
    return tuple(
        ShardOrderEntry(tensor_dim=tensor_dim, mesh_dims=tuple(mesh_dims))
        for tensor_dim, mesh_dims in shard_order.items()
    )


# TODO(zpcore): remove once the native redistribute supports shard_order arg
def redistribute(
    dtensor_input,
    device_mesh,
    placements,
    shard_order,
    use_graph_based_transform=True,
):
    """
    wrapper function to support shard_order for redistribution
    This is a simpler version of Redistribute, only considers the forward.
    """
    if placements is None:
        placements = shard_order_to_placement(shard_order, device_mesh)
    placements = tuple(placements)
    old_spec = dtensor_input._spec
    new_spec = copy.deepcopy(old_spec)
    new_spec.placements = placements
    if shard_order is not None:
        assert new_spec._maybe_update_placements_given_shard_order(shard_order)
    if old_spec == new_spec:
        return dtensor_input
    dtensor_input = DTensor.from_local(
        redistribute_local_tensor(
            dtensor_input.to_local(),
            old_spec,
            new_spec,
            use_graph_based_transform=use_graph_based_transform,
        ),
        device_mesh,
    )
    dtensor_input._spec = copy.deepcopy(new_spec)
    return dtensor_input  # returns DTensor


# TODO(zpcore): remove once the native distribute_tensor supports
# shard_order arg
def patched_distribute_tensor(
    input_tensor,
    device_mesh,
    placements,
    shard_order,
    use_graph_based_transform=True,
):
    """wrapper function to support shard_order for tensor distribution"""
    if placements is None:
        placements = shard_order_to_placement(shard_order, device_mesh)
    placements = tuple(placements)
    tensor_dt = distribute_tensor(input_tensor, device_mesh, placements)
    # fix the shard order
    return redistribute(
        tensor_dt, device_mesh, placements, shard_order, use_graph_based_transform
    )


# TODO(zpcore): remove once the native redistribute supports shard_order arg
def make_full_tensor(dtensor_input):
    """wrapper function to support DTensor.full_tensor"""
    return redistribute(
        dtensor_input, dtensor_input.device_mesh, placements=None, shard_order=()
    ).to_local()


def shard_order_to_placement(shard_order, mesh):
    """convert shard_order to placement with only Replicate() and Shard()"""
    placements: list[Any] = [Replicate() for _ in range(mesh.ndim)]
    if shard_order is not None:
        for entry in shard_order:
            tensor_dim = entry.tensor_dim
            mesh_dims = entry.mesh_dims
            for mesh_dim in mesh_dims:
                placements[mesh_dim] = Shard(tensor_dim)
    return tuple(placements)


def generate_shard_orders(mesh, tensor_rank):
    # Generate all possible sharding placement of tensor with rank
    # `tensor_rank` over mesh.
    def _split_list(lst: list, N: int):
        def compositions(n: int, k: int):
            # yields lists of length k, positive ints summing to n
            for cuts in itertools.combinations(range(1, n), k - 1):
                # add 0 and n as sentinels, then take consecutive differences
                yield [b - a for a, b in itertools.pairwise((0, *cuts, n))]

        length = len(lst)
        for comp in compositions(length, N):
            result = []
            start = 0
            for size in comp:
                result.append(lst[start : start + size])
                start += size
            yield result

    all_mesh = list(range(mesh.ndim))
    all_device_order = list(itertools.permutations(all_mesh))
    for device_order in all_device_order:
        # split on device orders, and assign each device order segment to a tensor dim
        for num_split in range(1, mesh.ndim + 1):
            for splitted_list in _split_list(list(range(mesh.ndim)), num_split):
                for tensor_dims in itertools.combinations(
                    range(tensor_rank), len(splitted_list)
                ):
                    shard_order = {}
                    assert len(tensor_dims) == len(splitted_list)
                    for tensor_dim, mesh_dims in zip(tensor_dims, splitted_list):
                        shard_order[tensor_dim] = device_order[
                            mesh_dims[0] : mesh_dims[-1] + 1
                        ]
                    yield _convert_shard_order_dict_to_ShardOrder(shard_order)
