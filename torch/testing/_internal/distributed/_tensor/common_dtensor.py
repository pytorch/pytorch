# mypy: allow-untyped-defs

# Copyright (c) Meta Platforms, Inc. and affiliates

import itertools
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, cast, Optional, TypeVar, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch._utils import _get_device_module
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    MultiThreadedTestCase,
    run_subtests,
    skip_if_lt_x_gpu,
    TEST_SKIPS,
)
from torch.testing._internal.common_utils import TEST_CUDA, TEST_HPU, TEST_XPU
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec


if TEST_CUDA:
    DEVICE_TYPE = "cuda"
    PG_BACKEND = "nccl"
    DEVICE_COUNT = _get_device_module("cuda").device_count()
elif TEST_HPU:
    DEVICE_TYPE = "hpu"
    PG_BACKEND = "hccl"
    DEVICE_COUNT = _get_device_module("hpu").device_count()
elif TEST_XPU:
    DEVICE_TYPE = "xpu"
    PG_BACKEND = "xccl"
    DEVICE_COUNT = _get_device_module("xpu").device_count()
else:
    DEVICE_TYPE = "cpu"
    PG_BACKEND = "gloo"

NUM_DEVICES = 4

# We use this as a proxy for "multiple GPUs exist"
if (TEST_CUDA or TEST_XPU or TEST_HPU) and DEVICE_COUNT > 1:
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


class DTensorTestBase(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    @property
    def device_type(self) -> str:
        # if enough GPU/XPU/HPU we can use those devices, otherwise we fallback to CPU
        if not (TEST_CUDA or TEST_XPU or TEST_HPU) or DEVICE_COUNT < self.world_size:
            return "cpu"
        else:
            return DEVICE_TYPE

    @property
    def backend(self) -> str:
        backend = dist.get_default_backend_for_device(DEVICE_TYPE)
        return backend

    def build_device_mesh(self) -> DeviceMesh:
        return DeviceMesh(self.device_type, list(range(self.world_size)))

    def init_pg(self, eager_init) -> None:
        if "nccl" in self.backend and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        if self.backend not in [
            "nccl",
            "gloo",
            "mpi",
            "cpu:gloo,cuda:nccl",
            "hccl",
            "xccl",
        ]:
            raise RuntimeError(f"Backend {self.backend} not supported!")

        device_id = None
        if "nccl" in self.backend or "xccl" in self.backend:
            # set device for nccl pg for collectives
            torch.accelerator.set_device_index(self.rank)
            # we only need to set device_id for nccl backend with eager init
            device_id = (
                torch.device(f"{self.device_type}:{self.rank}") if eager_init else None
            )
        # For nccl backend, bind the device to the process if device_id is not None
        # so the nccl communicator is immediately formed and we can use `ncclCommSplit`
        # for form subgroup to avoid unnecesssary overhead.
        dist.init_process_group(
            backend=self.backend,
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
        dist.barrier(device_ids=[device_id])
        dist.destroy_process_group()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

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
def with_comms(eager_init: Union[TestFunc, bool] = False) -> TestFunc:
    def decorator(func, eager_init: bool = False):
        @wraps(func)  # pyre-ignore[6]
        def wrapper(
            self, *args: tuple[object], **kwargs: dict[str, Any]  # type: ignore[misc]
        ) -> None:
            self.init_pg(eager_init)

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
        else partial(decorator, eager_init=eager_init)
    )


class DTensorOpTestBase(MultiThreadedTestCase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    @property
    def device_type(self) -> str:
        return DEVICE_TYPE

    def build_device_mesh(self):
        return DeviceMesh(self.device_type, list(range(self.world_size)))

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
        if type(t) is torch.Tensor or type(t) is nn.Parameter:
            if self.is_supported_tensor(t):
                self.hit += 1
                if t.ndim == 0:
                    # scalar tensor by default will be replicated
                    r = distribute_tensor(t, mesh, [Replicate()] * mesh.ndim)
                else:
                    # distribute non-scalar tensors
                    r = distribute_tensor(t, mesh, placements)
                if type(t) is nn.Parameter:
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
