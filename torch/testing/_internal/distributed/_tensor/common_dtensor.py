# Copyright (c) Meta Platforms, Inc. and affiliates

import itertools
import math
import sys
from functools import wraps
from typing import (
    Any,
    Callable,
    Iterator,
    Tuple,
    Dict,
    List,
    Sequence,
    TypeVar,
    cast,
)

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    MultiThreadedTestCase,
    TEST_SKIPS,
    skip_if_lt_x_gpu,
)

from torch.distributed._tensor import (
    DeviceMesh,
    Shard,
    Replicate,
    distribute_tensor,
)
from torch.distributed._tensor.placement_types import Placement

DEVICE_TYPE = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
PG_BACKEND = "nccl" if DEVICE_TYPE == "cuda" else "gloo"

NUM_DEVICES = 4

# We use this as a proxy for "multiple GPUs exist"
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    # when we actually have multiple GPUs, relax the requirement to smaller counts.
    NUM_DEVICES = min(NUM_DEVICES, torch.cuda.device_count())

T = TypeVar("T")


class MLPModule(nn.Module):
    def __init__(self, device):
        super().__init__()
        torch.manual_seed(5)
        self.net1 = nn.Linear(10, 16, device=device)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 10, device=device)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

    def reset_parameters(self):
        self.net1.reset_parameters()
        self.net2.reset_parameters()


class Attention(nn.Module):
    def __init__(self, dim, n_heads, world_size):
        super().__init__()
        self.head_dim = dim // n_heads
        self.n_heads = n_heads
        self.n_local_heads = n_heads // world_size

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask=None):
        bsz, seqlen, _ = x.size()
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
        queries = queries.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        keys = keys.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        values = values.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        queries = queries.transpose(1, 2)  # (bsz, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bsz, n_local_heads, seqlen, head_dim)
        values = values.transpose(1, 2)  # (bsz, n_local_heads, seqlen, head_dim)
        scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bsz, n_local_heads, seqlen, seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(queries)
        output = torch.matmul(scores, values)  # (bsz, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(self.relu(self.w1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id, dim, n_heads, world_size):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads, world_size)
        # self.attention_norm = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim, hidden_dim=4 * dim)
        # self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        # h = x + self.attention.forward(self.attention_norm(x), mask)
        h = x + self.attention.forward(x, mask)
        # out = h + self.feed_forward.forward(self.ffn_norm(h))
        out = h + self.feed_forward.forward(h)
        return out

# a toy transformer model, without positional encoding or dropout
class Transformer(nn.Module):
    def __init__(self, n_layers, vocab_size, dim, n_heads, world_size):
        super().__init__()
        torch.manual_seed(5)

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(n_layers):
            self.layers.append(TransformerBlock(layer_id, dim, n_heads, world_size))
        # self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        mask = None
        # mask is optional depending on encoder/decoder style
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            h = layer(h, mask)
        # h = self.norm(h)
        output = self.output(h).float()
        return output


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
    def backend(self) -> str:
        return PG_BACKEND

    def build_device_mesh(self) -> DeviceMesh:
        return DeviceMesh(DEVICE_TYPE, list(range(NUM_DEVICES)))

    def init_pg(self) -> None:
        if "nccl" in self.backend and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        if self.backend not in ["nccl", "gloo", "mpi", "cpu:gloo,cuda:nccl"]:
            raise RuntimeError(f"Backend {self.backend} not supported!")

        dist.init_process_group(
            backend=self.backend,
            world_size=self.world_size,
            rank=self.rank,  # pyre-ignore[16]
            init_method=f"file://{self.file_name}",  # pyre-ignore[16]
        )

        # set device for nccl pg for collectives
        if "nccl" in self.backend:
            torch.cuda.set_device(self.rank)

    def destroy_pg(self) -> None:
        # Wait for all ranks to reach here before starting shutdown.
        # FIXME dist.barrier deadlocks with multiple threads and NCCL: https://github.com/pytorch/pytorch/issues/95895
        # dist.all_reduce(torch.zeros((1,), device="cuda" if torch.cuda.is_available() else "cpu"))
        # FIXME can't use the above all_reduce as it causes hangs on bionic and focal. It hangs:
        #  test_dtensor.py  -- DTensorMeshTest.test_dtensor_device_mesh_device_conversion
        dist.barrier()
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


TestFunc = Callable[[object], object]

# wrapper to initialize comms (processgroup)
def with_comms(func: TestFunc) -> TestFunc:
    assert func is not None

    @wraps(func)  # pyre-ignore[6]
    def wrapper(
        self, *args: Tuple[object], **kwargs: Dict[str, Any]  # type: ignore[misc]
    ) -> None:
        # if backend not specified, and cuda available, then use nccl, else gloo
        if torch.cuda.is_available() and torch.cuda.device_count() >= self.world_size:
            self.device_type = "cuda"
        else:
            self.device_type = "cpu"

        self.init_pg()
        func(self, *args, **kwargs)  # type: ignore[misc]
        self.destroy_pg()

    return wrapper


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
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> None:
        self.hit = 0
        self.miss = 0
        self.mesh = mesh
        self.args = args
        self.kwargs = kwargs
        flatten_args, flatten_args_spec = tree_flatten(args)
        flatten_kwargs, flatten_kwargs_spec = tree_flatten(kwargs)

        self.flatten_args: List[object] = flatten_args
        self.flatten_args_spec: TreeSpec = flatten_args_spec
        self.flatten_kwargs: List[object] = flatten_kwargs
        self.flatten_kwargs_spec: TreeSpec = flatten_kwargs_spec

        choices_for_args = []
        for arg in self.flatten_args:
            if isinstance(arg, torch.Tensor):
                choices_for_args.append(self.gen_sharding_choices_for_arg(arg))

        for arg in self.flatten_kwargs:
            if isinstance(arg, torch.Tensor):
                choices_for_args.append(self.gen_sharding_choices_for_arg(arg))

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

    def gen_sharding_choices_for_arg(
        self, arg: torch.Tensor
    ) -> Sequence[Placement]:
        mesh_size = self.mesh.size()
        sharding_choices: List[Placement] = [Replicate()]
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

    def __next__(self) -> Tuple[Tuple[object, ...], Dict[str, object]]:
        try:
            next_sharding_choices = next(self.sharding_combs)
            idx = 0

            new_args: List[object] = []
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

            new_kwargs: List[object] = []
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
        self, t: torch.Tensor, mesh: DeviceMesh, placements: List[Placement]
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
            raise RuntimeError(
                f"Trying to convert to DTensor, but got {type(t)}"
            )
