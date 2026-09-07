# Owner(s): ["module: dynamo"]

import contextlib
import copy
import dataclasses
import functools
import importlib
import inspect
import io
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import threading
import unittest
from collections import namedtuple
from collections.abc import Callable
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.distributed as c10d
import torch.fx.traceback as fx_traceback
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx.operators
import torch.utils.cpp_extension
from torch._dynamo.aot_compile import (
    AOTCompiledFunction,
    AOTCompiledModel,
    ModelInput,
    SerializableCallable,
)
from torch._dynamo.aot_compile_types import BundledAOTAutogradSerializableCallable
from torch._dynamo.exc import PackageError, Unsupported
from torch._dynamo.graph_utils import _graph_device_types
from torch._dynamo.guards import CheckFunctionManager
from torch._dynamo.output_graph import get_builtins_dict
from torch._dynamo.package import (
    _current_cpu_codegen_target,
    DynamoCache,
    load_guards_state,
)
from torch._dynamo.precompile_context import PrecompileContext
from torch._functorch.aot_autograd import (
    aot_compile_joint_with_descriptors,
    aot_export_joint_with_descriptors,
)
from torch._guards import tracing, TracingContext
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx._graph_pickler import GraphPickler
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.passes.regional_inductor import regional_inductor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.utils.checkpoint import checkpoint


MY_LAMBDA = lambda x: x + 1  # noqa: E731

EPS = torch.tensor(1e-7)


def aot_eager_regional_inductor():
    from torch._dynamo.backends.common import aot_autograd
    from torch.fx.passes.regional_inductor import regional_inductor

    return aot_autograd(
        fw_compiler=regional_inductor,
        bw_compiler=regional_inductor,
    )


class SingleCondModel(torch.nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.fc1 = torch.nn.Linear(d, d)
        self.fc2 = torch.nn.Linear(d, d)

    def forward(self, x):
        x = self.fc1(x)

        def true_fn(x):
            return x * 2.0

        def false_fn(x):
            return x * 3.0

        x = torch.cond(x.shape[0] < 32, true_fn, false_fn, (x,))
        return self.fc2(x)


class MooType:
    def __init__(self, x):
        self.x = x


class CustomCompiledFunction(torch._dynamo.aot_compile.SerializableCallable):
    def __init__(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
        self.gm = gm
        self.example_inputs = example_inputs

    @classmethod
    def serialize_compile_artifacts(cls, fn) -> bytes:
        import sympy

        from torch._subclasses import FakeTensorMode
        from torch.fx._graph_pickler import Options

        state = fn.__dict__.copy()
        graph_reducer_override = GraphPickler.reducer_override

        def _graph_reducer_override(self, obj):
            if (
                inspect.isclass(obj)
                and issubclass(obj, sympy.Function)
                and hasattr(obj, "_torch_unpickler")
            ):
                return obj._torch_unpickler, (obj._torch_handler_name,)
            if isinstance(obj, FakeTensorMode):
                return type(None), ()
            return graph_reducer_override(self, obj)

        with patch.object(GraphPickler, "reducer_override", _graph_reducer_override):
            state["gm"] = GraphPickler.dumps(state["gm"], Options(ops_filter=None))
        return pickle.dumps(state)

    @classmethod
    def deserialize_compile_artifacts(cls, data: bytes):
        state = pickle.loads(data)
        fake_mode = torch._subclasses.FakeTensorMode()
        state["gm"] = GraphPickler.loads(state["gm"], fake_mode)
        state["gm"].recompile()
        return cls(**state)

    def __call__(self, *args, **kwargs):
        return self.gm(*args, **kwargs)


class MultiHeadSelfAttention(nn.Module):
    _flex_attention_cache: dict = {}
    _create_block_mask_fn = None

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = embed_dim // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.kv_dim)
        self.v_proj = nn.Linear(embed_dim, self.kv_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        self.enable_gqa = self.num_heads != self.num_kv_heads

        # Compile flex_attention with default compile_spec
        # This creates a nested torch.compile that triggers flex_attention_hop
        compile_spec = {
            "mode": "default",
            "fullgraph": True,
            "dynamic": False,
        }
        compile_key = tuple(sorted(compile_spec.items()))
        if compile_key not in MultiHeadSelfAttention._flex_attention_cache:
            MultiHeadSelfAttention._flex_attention_cache[compile_key] = torch.compile(  # noqa: UNSPECIFIED_BACKEND
                flex_attention, **compile_spec
            )
        self._flex_attention = MultiHeadSelfAttention._flex_attention_cache[compile_key]

        # Also compile create_block_mask
        if MultiHeadSelfAttention._create_block_mask_fn is None:
            MultiHeadSelfAttention._create_block_mask_fn = torch.compile(  # noqa: UNSPECIFIED_BACKEND
                create_block_mask, dynamic=False, fullgraph=True
            )

    def _shape_heads(self, x, B, S, num_heads):
        return x.view(B, S, num_heads, self.head_dim).transpose(1, 2)

    def _forward_local(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_mask,
    ) -> torch.Tensor:
        with fx_traceback.annotate({"compile_with_inductor": 1}):
            return self._flex_attention(
                query=query,
                key=key,
                value=value,
                block_mask=block_mask,
                enable_gqa=self.enable_gqa,
            )

    def _qkv_to_local(
        self,
        query,
        key,
        value,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from torch.distributed.tensor import Partial

        q_grad_placements = []
        kv_grad_placements = []

        for query_p, key_p, value_p in zip(
            query.placements, key.placements, value.placements
        ):
            if (
                (
                    query_p.is_shard(dim=0)
                    and key_p.is_shard(dim=0)
                    and value_p.is_shard(dim=0)
                )
                or (
                    query_p.is_shard(dim=1)
                    and key_p.is_shard(dim=1)
                    and value_p.is_shard(dim=1)
                )
                or (
                    query_p.is_replicate()
                    and key_p.is_replicate()
                    and value_p.is_replicate()
                )
            ):
                q_grad_placements.append(query_p)
                kv_grad_placements.append(key_p)
            elif (
                query_p.is_shard(dim=2)
                and key_p.is_replicate()
                and value_p.is_replicate()
            ):
                q_grad_placements.append(query_p)
                kv_grad_placements.append(Partial())
            else:
                raise NotImplementedError(
                    "Currently only supports Data Parallel, Tensor Parallel, "
                    "and all-gather based Context Parallel."
                )

            return (
                query.to_local(grad_placements=q_grad_placements),
                key.to_local(grad_placements=kv_grad_placements),
                value.to_local(grad_placements=kv_grad_placements),
            )

    def forward(self, x):
        from torch.distributed.tensor import DTensor

        B, S, _ = x.shape

        q = self._shape_heads(self.q_proj(x), B, S, self.num_heads)
        k = self._shape_heads(self.k_proj(x), B, S, self.num_kv_heads)
        v = self._shape_heads(self.v_proj(x), B, S, self.num_kv_heads)

        # Create block_mask inside forward to test cross-compilation
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        with fx_traceback.annotate({"compile_with_inductor": 1}):
            block_mask = MultiHeadSelfAttention._create_block_mask_fn(
                causal_mask, B, self.num_heads, S, S, device=x.device
            )

        if not any(isinstance(t, DTensor) for t in (q, k, v)):
            attn_out = self._forward_local(q, k, v, block_mask)
        else:
            q_local, k_local, v_local = self._qkv_to_local(q, k, v)
            attn_out_local = self._forward_local(q_local, k_local, v_local, block_mask)
            attn_out = DTensor.from_local(
                attn_out_local,
                device_mesh=q.device_mesh,
                placements=q.placements,
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, num_kv_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.dropout1(self.attn(self.ln1(x)))
        x = x + checkpoint(lambda inp: self.mlp(self.ln2(inp)), x, use_reentrant=False)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        num_kv_heads: int,
        device_mesh=None,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, num_kv_heads)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.device_mesh = device_mesh

    def forward(self, input_ids):
        from torch.distributed.tensor import Replicate

        input_ids = input_ids.redistribute(self.device_mesh, [Replicate()])
        x = self.embed(input_ids) + self.pos_embed[:, : input_ids.shape[1], :]

        for block in self.layers:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


class SimpleLinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x)


class ScaleModule(torch.nn.Module):
    def forward(self, x):
        return x * 2


AOT_HERMETIC_WEIGHT = torch.eye(3)


class HermeticModule(torch.nn.Module):
    def forward(self, x):
        return x @ AOT_HERMETIC_WEIGHT


GLOBAL_POOLING_CONFIG = {"pooling": "sum"}


class GlobalConfigModule(torch.nn.Module):
    def forward(self, x):
        if GLOBAL_POOLING_CONFIG["pooling"] == "sum":
            return x.sum(1)
        return x.mean(1) * 10.0


@contextmanager
def _set_pooling(mode):
    old = GLOBAL_POOLING_CONFIG["pooling"]
    GLOBAL_POOLING_CONFIG["pooling"] = mode
    try:
        yield
    finally:
        GLOBAL_POOLING_CONFIG["pooling"] = old


AOT_POOL_MODE = "sum"


class GlobalRebindModule(torch.nn.Module):
    def forward(self, x):
        if AOT_POOL_MODE == "sum":
            return x.sum(1)
        return x.mean(1) * 10.0


def global_rebind_fn(x):
    if AOT_POOL_MODE == "sum":
        return x.sum(1)
    return x.mean(1) * 10.0


@contextmanager
def _set_pool_mode(mode):
    global AOT_POOL_MODE
    old = AOT_POOL_MODE
    AOT_POOL_MODE = mode
    try:
        yield
    finally:
        AOT_POOL_MODE = old


AOT_ABSENT_WEIGHT = torch.eye(3) * 3.0


class AbsentGlobalModule(torch.nn.Module):
    def forward(self, x):
        return x @ AOT_ABSENT_WEIGHT


class ParentWithChildModule(torch.nn.Module):
    # Calling a CHILD module routes through nn.Module.__call__, whose hook-dict
    # guards are rooted at Dynamo's synthetic __import_torch_dot_nn_... alias.
    # That is the shape a reloaded artifact could not resolve.
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        # isinstance is a builtin: its BUILTIN_MATCH guard is rooted at
        # G['__builtins_dict___N'], the other name only the tracing process has.
        if not isinstance(x, torch.Tensor):
            raise TypeError(type(x))
        return self.lin(x)


def keep_global_guards(guard_entries):
    # Same policy the guard serializer enforces: drop only what cannot be
    # serialized, and in particular keep the global guards that the default
    # aot_compile filter drops wholesale.
    unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
    return [
        g.guard_type not in unsupported
        and not any(d in unsupported for d in g.derived_guard_types)
        for g in guard_entries
    ]


class RepeatInterleaveModule(torch.nn.Module):
    def forward(self, x):
        chunk = x.chunk(2, dim=-1)
        y = chunk[0]
        y_repeat = y.repeat_interleave(2, dim=-1)
        return y_repeat


class MultiModalMixin(torch.nn.Module):
    def forward(self, x):
        return super().forward(x)


class TextModel(torch.nn.Module):
    def forward(self, x):
        return x + 1


class TestVLLMModel(MultiModalMixin, TextModel):
    def forward(self, x):
        return super().forward(x)


def _subprocess_entry(fn, queue):
    try:
        fn()
    except BaseException as exc:
        import traceback

        queue.put((type(exc).__name__, str(exc), traceback.format_exc()))
        raise
    else:
        queue.put(None)


def _run_in_subprocess(fn):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_subprocess_entry, args=(fn, queue))
    proc.start()
    proc.join()
    result = queue.get()
    if result is not None:
        name, msg, tb = result
        raise AssertionError(f"Subprocess failure ({name}: {msg})\n{tb}")


def _subprocess_disable_guard_check():
    import torch
    from torch._dynamo import config

    with config.patch(enable_aot_compile=True):

        def fn(x, y):
            return x + y

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(*inputs)
        prev_grad = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(not prev_grad)
            try:
                compiled_fn(*inputs)
            except RuntimeError as exc:  # pragma: no cover
                if "GuardManager check failed" not in str(exc):
                    raise
            else:  # pragma: no cover
                raise AssertionError("Guard check should have failed")
            compiled_fn.disable_guard_check()
            actual = compiled_fn(*inputs)
            if not torch.allclose(actual, expected):
                raise AssertionError(
                    f"Expected tensors to be close, got {actual} vs {expected}"
                )
        finally:
            torch.set_grad_enabled(prev_grad)


def _subprocess_grad_mode_after_prior_compile():
    import torch
    from torch._dynamo import config

    with config.patch(enable_aot_compile=True):

        def warmup_fn(x, y):
            return x + y

        def target_fn(x, y):
            return x - y

        torch.compile(warmup_fn, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )
        torch._dynamo.reset()

        with torch.no_grad():
            compiled_fn = torch.compile(target_fn, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
                ((torch.randn(3, 4), torch.randn(3, 4)), {})
            )

        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        with torch.no_grad():
            actual = compiled_fn(*inputs)
            expected = target_fn(*inputs)
            if not torch.allclose(actual, expected):
                raise AssertionError(
                    f"Expected tensors to be close, got {actual} vs {expected}"
                )


def _subprocess_aot_compile_module():
    import torch
    from torch._dynamo import config

    with config.patch(enable_aot_compile=True):
        mod = SimpleLinearModule()
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="inductor",
            options={
                "guard_filter_fn": torch.compiler.skip_guard_on_globals_unsafe,
            },
        )

        @contextmanager
        def train_mode(mdl):
            mdl.train()
            yield

        @contextmanager
        def eval_mode(mdl):
            mdl.eval()
            yield

        inputs = [
            ModelInput(
                args=(torch.randn(3, 3),),
                kwargs={},
                contexts=[torch.no_grad(), eval_mode(model)],
            ),
            ModelInput(
                args=(torch.randn(3, 3),), kwargs={}, contexts=[train_mode(model)]
            ),
        ]
        if not isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
            raise AssertionError(
                f"Expected OptimizedModule, got {type(model).__name__}"
            )
        model._aot_compile(inputs)

        with torch.compiler.set_stance("fail_on_recompile"):
            model.eval()
            eager_inputs = (torch.randn(3, 3),)
            expected = mod(*eager_inputs)
            actual = model(*eager_inputs)
            if not torch.allclose(expected, actual):
                raise AssertionError(
                    f"Expected tensors to be close, got {actual} vs {expected}"
                )
            model.train()
            expected.sum().backward()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            model._save_aot_compiled_module(path)
            torch._dynamo.reset()
            model = torch.compile(
                mod,
                fullgraph=True,
                backend="inductor",
                options={
                    "guard_filter_fn": torch.compiler.skip_guard_on_globals_unsafe,
                },
            )
            if not isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
                raise AssertionError(
                    f"Expected OptimizedModule, got {type(model).__name__}"
                )
            with open(path, "rb") as f:
                data = f.read()
                model._load_aot_compiled_module(data)

            with torch.compiler.set_stance("fail_on_recompile"):
                model.eval()
                eager_inputs = (torch.randn(3, 3),)
                expected = mod(*eager_inputs)
                actual = model(*eager_inputs)
                if not torch.allclose(expected, actual):
                    raise AssertionError(
                        f"Expected tensors to be close, got {actual} vs {expected}"
                    )


def _subprocess_save_child_module_artifact(path):
    import torch
    from torch._dynamo import config

    with config.patch(enable_aot_compile=True):
        mod = ParentWithChildModule()
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile(
            [ModelInput(args=(torch.randn(4, 4),), kwargs={}, contexts=[])]
        )
        model._save_aot_compiled_module(path)
        torch.save(mod.state_dict(), path + ".state_dict")


class RedistributeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)

    def forward(self, x, d_x, mesh):
        x = self.linear(x)

        # need to do local import since tests don't always have c10d
        # and precompile needs this class to be available at the module
        # level.
        from torch.distributed.tensor import Replicate

        y = d_x.redistribute(mesh, placements=(Replicate(), Replicate()))

        return x, y


def wrap_forward_function(fn: Callable):
    @functools.wraps(fn, assigned=("__doc__", "__annotations__", "__type_params__"))
    def wrapped(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapped


@torch._dynamo.config.patch("enable_aot_compile", True)
@instantiate_parametrized_tests
class TestAOTCompile(torch._inductor.test_case.TestCase):
    def path(self):
        path = os.path.join(cache_dir(), f"package_{self.id()}")
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, "model.pt")

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        DynamoCache.clear()
        PrecompileContext.clear()

    def test_aot_compile_basic_fn(self):
        def fn(x, y):
            return x + y

        def backend(gm, example_inputs):
            return CustomCompiledFunction(gm, example_inputs)

        compiled_fn = torch.compile(fn, fullgraph=True, backend=backend).aot_compile(
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(*inputs)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(*inputs)
            self.assertEqual(expected, actual)

    def test_aot_compile_reloads_a_runtime_env_helper_faithfully(self):
        # A nested helper the compiled function closes over travels in the
        # runtime env and is rebuilt from its code object at load. Everything
        # it holds has to survive: an EMPTY cell failed the pickler, a None
        # cell came back empty, and __kwdefaults__ and __dict__ were dropped;
        # see FunctionPicklerBase. (The compiled function itself cannot have an
        # empty cell: capture reads all of its cells up front.)
        def outer():
            scale = None

            def helper(x, *, k=2):
                if x is None:
                    return unset
                if scale is None:
                    x = x + 1
                return x * k

            helper.tag = 2.0
            if helper is None:
                unset = 1
            return helper

        helper = outer()

        def fn(x):
            return helper(x) * 2

        def backend(gm, example_inputs):
            return CustomCompiledFunction(gm, example_inputs)

        inputs = (torch.randn(3),)
        expected = fn(*inputs)
        compiled_fn = torch.compile(fn, fullgraph=True, backend=backend).aot_compile(
            (inputs, {})
        )
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            self.assertEqual(expected, compiled_fn(*inputs))
        (cell,) = compiled_fn._artifacts.runtime_env.closure
        loaded = cell.cell_contents
        cells = dict(zip(loaded.__code__.co_freevars, loaded.__closure__))
        with self.assertRaisesRegex(ValueError, "empty"):
            cells["unset"].cell_contents
        self.assertIsNone(cells["scale"].cell_contents)
        self.assertEqual(loaded.__kwdefaults__, {"k": 2})
        self.assertEqual(loaded.tag, 2.0)

    def test_aot_compile_rejects_a_helper_with_an_unpicklable_attribute(self):
        # A helper's __dict__ travels with it now, so an attribute that cannot
        # pickle fails the save instead of being silently dropped.
        def outer():
            def helper(x):
                return x * 2

            helper.lock = threading.Lock()
            return helper

        helper = outer()

        def fn(x):
            return helper(x) + 1

        inputs = (torch.randn(3),)
        compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")
        compiled_fn = compiled_fn.aot_compile((inputs, {}))
        with self.assertRaisesRegex(TypeError, "cannot pickle '_thread.lock' object"):
            compiled_fn.save_compiled_function(self.path())

    def test_aot_compile_autocast_guard_reload(self):
        def fn(x):
            return x + 1 * x

        def backend(gm, example_inputs):
            return CustomCompiledFunction(gm, example_inputs)

        x = torch.randn(3, 4)
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            compiled_fn = torch.compile(
                fn, fullgraph=True, backend=backend
            ).aot_compile(((x,), {}))
            expected = fn(x)
            self.assertEqual(expected, compiled_fn(x))

        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(f)
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            actual = compiled_fn(x)
        self.assertEqual(expected, actual)

    def test_aot_compile_basic_forward(self):
        mod = SimpleLinearModule()

        def backend(gm, example_inputs):
            return CustomCompiledFunction(gm, example_inputs)

        compiled_fn = torch.compile(
            mod,
            fullgraph=True,
            backend=backend,
        ).forward.aot_compile(((torch.randn(3, 3),), {}))
        inputs = (torch.randn(3, 3),)
        expected = mod(*inputs)
        actual = compiled_fn(mod, *inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(mod, *inputs)
            self.assertEqual(expected, actual)

    def test_aot_compile_repeat_interleave(self):
        mod = RepeatInterleaveModule()

        def backend(gm, example_inputs):
            return CustomCompiledFunction(gm, example_inputs)

        inputs = (torch.randn(2, 4),)

        # The first dim should be dynamic to repro the issue of repeat_interleave
        # torch._dynamo.mark_dynamic(inputs[0], [0])

        compiled_fn = torch.compile(
            mod,
            fullgraph=True,
            backend=backend,
        ).forward.aot_compile((inputs, {}))

        expected = mod(*inputs)
        actual = compiled_fn(mod, *inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(mod, *inputs)
            self.assertEqual(expected, actual)

    def test_code_cache(self):
        from torch._dynamo.package import SerializedCode

        def foo():
            pass

        serialized_code = SerializedCode.from_code_object(foo.__code__)
        object.__setattr__(
            serialized_code, "co_consts", serialized_code.co_consts + ({1: 2},)
        )

        new_code = SerializedCode.to_code_object(serialized_code)
        new_serialized_code = SerializedCode.from_code_object(new_code)
        self.assertEqual(new_serialized_code, serialized_code)

    def test_decorated_function_aot(self):
        def check_inputs(fn):
            def _fn(*args, **kwargs):
                for arg in args:
                    assert arg.shape[0] > 1  # noqa: S101

                return fn(*args, **kwargs)

            return _fn

        @check_inputs
        def foo(x, y):
            a = x + x
            b = y + y
            c = a + b
            return c

        example_inputs = (torch.ones(3), torch.ones(3))
        expected = foo(*example_inputs)

        def backend(gm, example_inputs):
            return CustomCompiledFunction(gm, example_inputs)

        with torch.compiler.set_stance("fail_on_recompile"):
            compiled_fn = torch.compile(
                foo,
                fullgraph=True,
                backend=backend,
            ).aot_compile((example_inputs, {}))
            actual = compiled_fn(*example_inputs)
            self.assertEqual(expected, actual)

    def test_eager_backend(self):
        def fn(x, y):
            return x + y

        compiled_fn = torch.compile(fn, fullgraph=True, backend="eager").aot_compile(
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(*inputs)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(*inputs)
            self.assertEqual(expected, actual)

    def test_aot_eager_backend(self):
        def fn(x, y):
            return x + y

        compiled_fn = torch.compile(
            fn, fullgraph=True, backend="aot_eager"
        ).aot_compile(((torch.randn(3, 4), torch.randn(3, 4)), {}))
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(*inputs)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(*inputs)
            self.assertEqual(expected, actual)

    def test_decorated_function_with_functools_wrap_aot(self):
        def check_inputs(fn):
            @functools.wraps(fn)
            def _fn(*args, **kwargs):
                for arg in args:
                    assert arg.shape[0] > 1  # noqa: S101

                return fn(*args, **kwargs)

            return _fn

        @check_inputs
        def foo(x, y):
            a = x + x
            b = y + y
            c = a + b
            return c

        example_inputs = (torch.ones(3), torch.ones(3))
        expected = foo(*example_inputs)

        def backend(gm, example_inputs):
            return CustomCompiledFunction(gm, example_inputs)

        with torch.compiler.set_stance("fail_on_recompile"):
            compiled_fn = torch.compile(
                foo,
                fullgraph=True,
                backend=backend,
            ).aot_compile((example_inputs, {}))
            actual = compiled_fn(*example_inputs)
            self.assertEqual(expected, actual)

    def test_aot_compile_disable_guard_check(self):
        _run_in_subprocess(_subprocess_disable_guard_check)

    def test_aot_compile_grad_mode_after_prior_compile(self):
        _run_in_subprocess(_subprocess_grad_mode_after_prior_compile)

    def test_aot_compile_torch_func_vmap_grad(self):
        import torch.func as tf

        def value(pos):
            return torch.linalg.norm(pos[1] - pos[0])

        batched_grad = tf.vmap(tf.grad(value, argnums=0), in_dims=(0,))
        x = torch.randn(64, 2, 3, dtype=torch.float64)
        compiled_fn = torch.compile(
            batched_grad, fullgraph=True, dynamic=False
        ).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((x,), {})
        )
        expected = batched_grad(x)
        actual = compiled_fn(x)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                loaded_fn = torch.compiler.load_compiled_function(f)
            actual = loaded_fn(x)
            self.assertEqual(expected, actual)

    def test_aot_compile_source_info(self):
        from torch._dynamo.package import SourceInfo

        def fn(x, y):
            return MY_LAMBDA(x) + y

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )

        source_info = compiled_fn.source_info()
        self.assertIsInstance(source_info, SourceInfo)
        self.assertEqual(len(source_info.inlined_sources), 2)
        self.assertEqual(next(iter(source_info.inlined_sources)).module, __name__)
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(f)
        source_info = compiled_fn.source_info()
        self.assertIsInstance(source_info, SourceInfo)
        self.assertEqual(len(source_info.inlined_sources), 2)
        self.assertEqual(next(iter(source_info.inlined_sources)).module, __name__)

    def test_regional_inductor_backend(self):
        import torch.fx.traceback as fx_traceback

        def fn(x, y):
            sin = torch.sin(x)
            # Mark this region to be compiled with inductor
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                mul = sin * y
                add = mul + 1
            return torch.sin(add)

        def make_inputs():
            return (
                torch.randn(3, 4, requires_grad=True),
                torch.randn(3, 4, requires_grad=True),
            )

        compiled_fn = torch.compile(
            fn, fullgraph=True, backend=aot_eager_regional_inductor()
        ).aot_compile((make_inputs(), {}))
        test_inputs = make_inputs()
        self.assertEqual(compiled_fn(*test_inputs), fn(*test_inputs))
        compiled_fn(*test_inputs).sum().backward()
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(f)

        self.assertEqual(compiled_fn(*test_inputs), fn(*test_inputs))
        compiled_fn(*test_inputs).sum().backward()

    def test_aot_compile_graph_break_error_fmt(self):
        def foo(x, y):
            a = x + x
            torch._dynamo.graph_break()
            b = y + y
            c = a + b
            return c

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(foo, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
                ((torch.ones(3), torch.ones(3)), {})
            ),
            """\
Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_aot_compile.py", line N, in foo
    torch._dynamo.graph_break()""",
        )

    def test_guard_filter_override_aot(self):
        def check_inputs(fn):
            def _fn(*args, **kwargs):
                for arg in args:
                    assert arg.shape[0] > 1  # noqa: S101

                return fn(*args, **kwargs)

            return _fn

        @check_inputs
        def foo(x, y):
            a = x + x
            b = y + y
            c = a + b
            return c

        example_inputs = (torch.ones(3), torch.ones(3))
        expected = foo(*example_inputs)  # noqa: F841

        def backend(gm, example_inputs):
            return CustomCompiledFunction(gm, example_inputs)

        with torch.compiler.set_stance("fail_on_recompile"):
            with self.assertRaisesRegex(
                PackageError,
                "CLOSURE_MATCH guard cannot be serialized.",
            ):
                compiled_fn = torch.compile(  # noqa: F841
                    foo,
                    fullgraph=True,
                    backend=backend,
                    options={
                        "guard_filter_fn": lambda guard_entries: [
                            True for g in guard_entries
                        ]
                    },
                ).aot_compile((example_inputs, {}))

    def test_aot_compile_basic_fn_inductor(self):
        def fn(x, y):
            return x + y

        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor").aot_compile(
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(*inputs)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(*inputs)
            self.assertEqual(expected, actual)

    def test_aot_compile_module(self):
        _run_in_subprocess(_subprocess_aot_compile_module)

    def test_aot_compile_module_dispatches_on_global_guard(self):
        # The default guard_filter_fn drops all global guards, so a caller who
        # needs one honored has to opt in. Keeping it only works because
        # AOTCompiledModel.deserialize supplies the traced function's globals.
        mod = GlobalConfigModule()
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="inductor",
            options={"guard_filter_fn": keep_global_guards},
        )
        x = torch.randn(4, 8)

        expected = {}
        for mode in ("sum", "mean"):
            with _set_pooling(mode):
                expected[mode] = mod(x)
        self.assertNotEqual(expected["sum"].tolist(), expected["mean"].tolist())

        model._aot_compile(
            [
                ModelInput(args=(x,), kwargs={}, contexts=[_set_pooling("sum")]),
                ModelInput(args=(x,), kwargs={}, contexts=[_set_pooling("mean")]),
            ]
        )
        for mode in ("sum", "mean"):
            with _set_pooling(mode):
                self.assertEqual(model(x), expected[mode])

        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        reloaded = torch.compile(
            GlobalConfigModule(),
            fullgraph=True,
            backend="inductor",
            options={"guard_filter_fn": keep_global_guards},
        )
        reloaded._load_aot_compiled_module(data)
        for mode in ("sum", "mean"):
            with _set_pooling(mode):
                self.assertEqual(reloaded(x), expected[mode])

    def test_aot_compile_module_no_match_error(self):
        # Two inputs, so the message has to account for both rather than
        # reporting only the first one's guard failure. Vary dtype rather than
        # shape: aot_compile_module does not forward `dynamic`, so a second
        # shape goes automatic-dynamic and would subsume the unmatched input.
        model = torch.compile(ScaleModule(), fullgraph=True, backend="inductor")
        model._aot_compile(
            [
                ModelInput(
                    args=(torch.randn(3, 3, dtype=torch.float32),),
                    kwargs={},
                    contexts=[],
                ),
                ModelInput(
                    args=(torch.randn(3, 3, dtype=torch.float64),),
                    kwargs={},
                    contexts=[],
                ),
            ]
        )
        with self.assertRaises(RuntimeError) as ctx:
            model(torch.randn(3, 3, dtype=torch.float16))
        message = str(ctx.exception)
        self.assertIn("No AOT compiled graph matched this call", message)
        self.assertIn("Tried 2 compiled input(s)", message)
        self.assertIn("[0]", message)
        self.assertIn("[1]", message)
        # One line per input, not a multi-line GuardDebugInfo repr per input.
        self.assertEqual(len(message.splitlines()), 4)

    def test_no_match_message_survives_a_raising_guard(self):
        # __call__ has already established that nothing matched; re-evaluating
        # the guards to say WHY must not replace that answer with a secondary
        # failure from one entry, which hides both the entry that raised and
        # every other entry's reason.
        class RaisingGuardManager:
            def __init__(self, inner):
                self._inner = inner

            def check(self, f_locals):
                return self._inner.check(f_locals)

            def check_verbose(self, f_locals):
                raise KeyError("G['SOME_GLOBAL']")

        model = torch.compile(ScaleModule(), fullgraph=True, backend="inductor")
        model._aot_compile(
            [
                ModelInput(
                    args=(torch.randn(3, 3, dtype=torch.float32),),
                    kwargs={},
                    contexts=[],
                ),
                ModelInput(
                    args=(torch.randn(3, 3, dtype=torch.float64),),
                    kwargs={},
                    contexts=[],
                ),
            ]
        )
        results = model.forward.compiled_results
        results[0]._artifacts.guard_manager = RaisingGuardManager(
            results[0]._artifacts.guard_manager
        )
        with self.assertRaises(RuntimeError) as ctx:
            model(torch.randn(3, 3, dtype=torch.float16))
        message = str(ctx.exception)
        self.assertIn("No AOT compiled graph matched this call", message)
        self.assertIn("KeyError", message)
        self.assertIn("SOME_GLOBAL", message)
        # The entry that raised must not swallow the one that can explain itself.
        self.assertIn("dtype mismatch", message)
        self.assertEqual(len(message.splitlines()), 4)

    def test_aot_compile_module_disable_guard_check(self):
        # disable_guard_check() is the escape hatch for an artifact whose guards
        # fail on the serving machine; module dispatch has to honour it too, or
        # a module artifact has no opt-out while a function artifact does.
        model = torch.compile(ScaleModule(), fullgraph=True, backend="inductor")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )
        # requires_grad is guarded but does not change what the graph computes.
        x = torch.randn(3, 3, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "No AOT compiled graph matched"):
            model(x)
        model.forward.compiled_results[0].disable_guard_check()
        self.assertEqual(model(x), x * 2)

    def test_disable_guard_check_does_not_shadow_a_matching_result(self):
        # An opted-out result accepts anything, so it has to be the last resort
        # rather than a candidate in the ordinary scan: consulting it before
        # every real guard check has been tried serves the first artifact for a
        # call the second one was compiled for. Same shapes, same dtypes, no
        # error, different numbers.
        mod = GlobalConfigModule()
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="inductor",
            options={"guard_filter_fn": keep_global_guards},
        )
        x = torch.randn(4, 8)
        expected = {}
        for mode in ("sum", "mean"):
            with _set_pooling(mode):
                expected[mode] = mod(x)
        self.assertNotEqual(expected["sum"].tolist(), expected["mean"].tolist())

        model._aot_compile(
            [
                ModelInput(args=(x,), kwargs={}, contexts=[_set_pooling("sum")]),
                ModelInput(args=(x,), kwargs={}, contexts=[_set_pooling("mean")]),
            ]
        )
        model.forward.compiled_results[0].disable_guard_check()
        with _set_pooling("mean"):
            self.assertEqual(model(x), expected["mean"])

    def test_aot_compile_module_scope_resolves_through_forward_hook(self):
        # A registered forward hook makes get_traced_fn(model) return
        # Module._wrapped_call_impl, whose globals are torch/nn/modules/module.py.
        # The guard scope has to come from what was actually traced, model.forward.
        #
        # The hook is deliberately the identity: aot_compile_module traces
        # model.forward directly and never runs hooks, so a hook with an effect
        # would simply be dropped from the compiled result. This pins the guard
        # SCOPE resolution on a hooked module, not hook support.
        mod = GlobalConfigModule()
        mod.register_forward_hook(lambda m, i, o: o)
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="inductor",
            options={"guard_filter_fn": keep_global_guards},
        )
        x = torch.randn(4, 8)
        expected = {}
        for mode in ("sum", "mean"):
            with _set_pooling(mode):
                expected[mode] = mod(x)

        model._aot_compile(
            [
                ModelInput(args=(x,), kwargs={}, contexts=[_set_pooling(m)])
                for m in ("sum", "mean")
            ]
        )
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        reloaded_mod = GlobalConfigModule()
        reloaded_mod.register_forward_hook(lambda m, i, o: o)
        reloaded = torch.compile(
            reloaded_mod,
            fullgraph=True,
            backend="inductor",
            options={"guard_filter_fn": keep_global_guards},
        )
        reloaded._load_aot_compiled_module(data)
        for mode in ("sum", "mean"):
            with _set_pooling(mode):
                self.assertEqual(reloaded(x), expected[mode])

    def test_aot_compile_module_reload_is_hermetic(self):
        # Pins, deliberately, that a module artifact's bytecode reads the
        # globals serialized at capture: supplying a scope for guards must not
        # rewire what the compiled bytecode reads. The consequence is a known
        # limitation, not a goal: a same-shape rebind of a global tensor before
        # the load passes the guards (they read the live scope) and the graph
        # still serves the capture-time value. The rebind has to happen before
        # the load so the test can tell the two scopes apart.
        global AOT_HERMETIC_WEIGHT
        model = torch.compile(HermeticModule(), fullgraph=True, backend="inductor")
        x = torch.randn(3, 3)
        expected = model._orig_mod(x)
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()

        torch._dynamo.reset()
        saved = AOT_HERMETIC_WEIGHT
        try:
            AOT_HERMETIC_WEIGHT = AOT_HERMETIC_WEIGHT * 2
            reloaded = torch.compile(
                HermeticModule(), fullgraph=True, backend="inductor"
            )
            reloaded._load_aot_compiled_module(data)
            self.assertEqual(reloaded(x), expected)
        finally:
            AOT_HERMETIC_WEIGHT = saved

    def test_aot_compile_module_guards_track_rebound_global(self):
        # The other half of hermeticity. Guards resolve against the loading
        # process's scope dict itself, so a global rebound after the artifact is
        # loaded redirects dispatch. A copy taken at load time would go on
        # serving whichever graph matched then, with no error and a wrong answer.
        global AOT_POOL_MODE
        mod = GlobalRebindModule()
        x = torch.randn(4, 8)
        expected = {}
        for mode in ("sum", "mean"):
            with _set_pool_mode(mode):
                expected[mode] = mod(x)
        self.assertNotEqual(expected["sum"].tolist(), expected["mean"].tolist())

        model = torch.compile(
            mod,
            fullgraph=True,
            backend="inductor",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile(
            [
                ModelInput(args=(x,), kwargs={}, contexts=[_set_pool_mode(m)])
                for m in ("sum", "mean")
            ]
        )
        data = model._save_aot_compiled_module()

        torch._dynamo.reset()
        saved = AOT_POOL_MODE
        try:
            AOT_POOL_MODE = "sum"
            reloaded = torch.compile(
                GlobalRebindModule(),
                fullgraph=True,
                backend="inductor",
                options={"guard_filter_fn": keep_global_guards},
            )
            reloaded._load_aot_compiled_module(data)
            self.assertEqual(reloaded(x), expected["sum"])
            AOT_POOL_MODE = "mean"
            self.assertEqual(reloaded(x), expected["mean"])
        finally:
            AOT_POOL_MODE = saved

    def test_aot_compile_fn_guards_track_rebound_global(self):
        # Function artifacts get their guard scope from load_compiled_function's
        # f_globals. Same contract as the module test above: the live dict, not
        # a copy, so a rebind after load changes the guard's answer.
        global AOT_POOL_MODE
        x = torch.randn(4, 8)
        expected = {}
        for mode in ("sum", "mean"):
            with _set_pool_mode(mode):
                expected[mode] = global_rebind_fn(x)
        self.assertNotEqual(expected["sum"].tolist(), expected["mean"].tolist())

        with _set_pool_mode("sum"):
            compiled_fn = torch.compile(
                global_rebind_fn,
                fullgraph=True,
                backend="inductor",
                options={"guard_filter_fn": keep_global_guards},
            ).aot_compile(((x,), {}))
        compiled_fn.save_compiled_function(self.path())

        torch._dynamo.reset()
        saved = AOT_POOL_MODE
        try:
            AOT_POOL_MODE = "sum"
            with open(self.path(), "rb") as f:
                loaded = torch.compiler.load_compiled_function(f, f_globals=globals())
            self.assertEqual(loaded(x), expected["sum"])
            AOT_POOL_MODE = "mean"
            with self.assertRaisesRegex(RuntimeError, "AOT_POOL_MODE"):
                loaded(x)
        finally:
            AOT_POOL_MODE = saved

    def test_aot_compile_module_absent_global_fails_guard(self):
        # A guarded global the loading process does not have has to fail the
        # guard. Falling back to the serialized scope would make the guard a
        # no-op checked against capture-time state, which is the silent failure
        # mode -- the loud one is recoverable on the serving machine.
        mod = AbsentGlobalModule()
        x = torch.randn(3, 3)
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="inductor",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()

        torch._dynamo.reset()
        saved = globals().pop("AOT_ABSENT_WEIGHT")
        try:
            reloaded = torch.compile(
                AbsentGlobalModule(),
                fullgraph=True,
                backend="inductor",
                options={"guard_filter_fn": keep_global_guards},
            )
            reloaded._load_aot_compiled_module(data)
            with self.assertRaises(RuntimeError) as ctx:
                reloaded(x)
            self.assertIn("No AOT compiled graph matched", str(ctx.exception))
            self.assertIn("AOT_ABSENT_WEIGHT", str(ctx.exception))
        finally:
            globals()["AOT_ABSENT_WEIGHT"] = saved

    def test_aot_compile_module_import_alias_guard_survives_reload(self):
        # Dynamo mints __import_* aliases into the TRACING process's globals and
        # roots guards at them. A process that only loads never traced, so the
        # live module dict this artifact guards against has none of them -- and
        # every call died on KeyError on G['__import_torch_dot_nn_dot_modules_
        # dot_module'] before the aliases were seeded. The existing scope tests
        # cannot see it: the capture in the same process already leaked those
        # names into this module's globals, so they resolve. Pop them to get
        # what a fresh process sees.
        mod = ParentWithChildModule()
        x = torch.randn(4, 4)
        expected = mod(x)
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        g = globals()
        aliases = {k: g.pop(k) for k in [k for k in g if k.startswith("__import_")]}
        self.assertTrue(aliases, "capture leaked no aliases; the test cannot bite")
        try:
            fresh = ParentWithChildModule()
            fresh.load_state_dict(mod.state_dict())
            reloaded = torch.compile(
                fresh,
                fullgraph=True,
                backend="eager",
                options={"guard_filter_fn": keep_global_guards},
            )
            reloaded._load_aot_compiled_module(data)
            self.assertEqual(reloaded(x), expected)
        finally:
            g.update(aliases)

    def test_aot_compile_module_import_alias_guard_loads_across_processes(self):
        # The real deployment shape: the artifact is captured by a process that
        # never runs here, so the __import_* aliases its guards are rooted at
        # have to be seeded into this module's globals by the load itself.
        path = self.path()
        _run_in_subprocess(
            functools.partial(_subprocess_save_child_module_artifact, path)
        )
        mod = ParentWithChildModule()
        mod.load_state_dict(torch.load(path + ".state_dict"))
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        # The load seeds __import_*/__builtins_dict__ aliases into this module's
        # globals; strip whatever it adds so sibling tests do not inherit them.
        g = globals()
        preexisting = frozenset(g)

        def _restore_globals() -> None:
            for k in [k for k in g if k not in preexisting]:
                del g[k]

        self.addCleanup(_restore_globals)
        with open(path, "rb") as f:
            model._load_aot_compiled_module(f.read())
        (result,) = model.forward.compiled_results
        import_sources = result._artifacts.runtime_env.import_sources
        self.assertIn("__import_torch_dot_nn_dot_modules_dot_module", import_sources)
        for alias, module_name in import_sources.items():
            self.assertIs(globals()[alias], importlib.import_module(module_name))
        guards_state = load_guards_state(result._artifacts.guards_state)
        builtins_key = guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
        self.assertTrue(builtins_key)
        self.assertIs(globals()[builtins_key], get_builtins_dict(globals()))
        x = torch.randn(4, 4)
        self.assertEqual(model(x), mod(x))

    def test_load_seeds_exactly_the_recorded_import_aliases(self):
        # Loading may add only the aliases the artifact recorded, and must not
        # overwrite a name the loading process already has.
        mod = ParentWithChildModule()
        x = torch.randn(4, 4)
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()
        (result,) = model.forward.compiled_results
        import_sources = result._artifacts.runtime_env.import_sources
        self.assertTrue(import_sources)
        serialized_guards = result._artifacts.guards_state
        torch._dynamo.reset()

        scope = {
            k: v
            for k, v in globals().items()
            if not k.startswith(("__import_", "__builtins_dict__"))
        }
        kept_alias = next(iter(import_sources))
        sentinel = object()
        scope[kept_alias] = sentinel
        before = set(scope)
        (serialized,) = pickle.loads(data)
        AOTCompiledFunction.deserialize(serialized, guard_globals=scope)
        builtins_key = load_guards_state(
            serialized_guards
        ).output_graph.name_of_builtins_dict_key_in_fglobals
        self.assertEqual(
            set(scope) - before, (set(import_sources) - {kept_alias}) | {builtins_key}
        )
        self.assertIs(scope[kept_alias], sentinel)
        for alias, module_name in import_sources.items():
            if alias != kept_alias:
                self.assertIs(scope[alias], importlib.import_module(module_name))

    def test_aot_compile_module_partial_forward_falls_back_loudly(self):
        # A forward that is neither a function nor a bound method has no live
        # scope to resolve guards against. The artifact still loads, against the
        # reconstructed scope, but that has to be a warning: it is the one case
        # where a guarded global does not track the loading process.
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        x = torch.randn(3, 3)
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()

        mod = ScaleModule()
        mod.forward = functools.partial(ScaleModule.forward, mod)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            reloaded = AOTCompiledModel.deserialize(mod, data)
        (line,) = logs.output
        self.assertIn("ScaleModule.forward is functools.partial(", line)
        self.assertIn("no live guard scope", line)
        self.assertEqual(reloaded(x), x * 2)

    def test_aot_module_simplified_serializable_autograd(self):
        mod = SimpleLinearModule()
        compiled_fn: SerializableCallable = torch.compile(
            mod, fullgraph=True, backend="inductor"
        ).forward.aot_compile(((torch.randn(3, 3),), {}))
        backend_result = compiled_fn._artifacts.compiled_fn
        self.assertTrue(
            isinstance(
                backend_result,
                torch._dynamo.aot_compile.BundledAOTAutogradSerializableCallable,
            )
        )
        if not hasattr(backend_result.compiled_fn, "serialize"):
            raise AssertionError("Expected compiled_fn to have 'serialize' attribute")
        self.assertIsNotNone(backend_result.compiled_fn.serialize)

    def test_aot_compile_portable_guards_unsafe(self):
        def fn(xy):
            return xy[0] + xy[1]

        compiled_fn = torch.compile(  # noqa: UNSPECIFIED_BACKEND
            fn,
            fullgraph=True,
            options={"guard_filter_fn": torch.compiler.keep_portable_guards_unsafe},
        ).aot_compile((((torch.randn(3, 4), torch.randn(3, 4)),), {}))
        Tup = namedtuple("Tup", ["x", "y"])

        inputs = Tup(torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(inputs)
        actual = compiled_fn(inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(f)
        actual = compiled_fn(inputs)
        self.assertEqual(expected, actual)

    def test_aot_module_simplified_serializable_inference(self):
        def fn(x):
            return x.sin()

        compiled_fn: SerializableCallable = torch.compile(
            fn, fullgraph=True, backend="inductor"
        ).aot_compile(((torch.randn(3, 3),), {}))
        backend_result = compiled_fn._artifacts.compiled_fn
        self.assertTrue(
            isinstance(
                backend_result,
                torch._dynamo.aot_compile.BundledAOTAutogradSerializableCallable,
            )
        )
        if not hasattr(backend_result.compiled_fn, "serialize"):
            raise AssertionError("Expected compiled_fn to have 'serialize' attribute")
        self.assertIsNotNone(backend_result.compiled_fn.serialize)

    def test_aot_cache_predicate_not_pickleable(self):
        import torch._functorch.config as functorch_config
        import torch._inductor.config as inductor_config

        model = SingleCondModel().eval()

        old_cacheable = torch.ops.higher_order.cond._cacheable
        torch.ops.higher_order.cond._cacheable = True
        try:
            with (
                functorch_config.patch(
                    enable_autograd_cache=True,
                    force_non_lazy_backward_lowering=True,
                    strict_autograd_cache=True,
                ),
                inductor_config.patch(
                    fx_graph_cache=True,
                    fx_graph_remote_cache=False,
                ),
            ):
                compiled = torch.compile(model, backend="inductor", dynamic=True)

                # Test both branches of the cond predicate (x.shape[0] < 32).
                for batch_size in (16, 64):
                    inp = torch.randn(batch_size, 64)
                    expected = model(inp)
                    actual = compiled(inp)
                    self.assertEqual(expected, actual)

                # If the SymBool predicate is not pickleable, the FxGraphCache
                # silently bypasses instead of caching. Assert no bypass occurred.
                self.assertEqual(
                    torch._dynamo.utils.counters["inductor"]["fxgraph_cache_bypass"], 0
                )
        finally:
            torch.ops.higher_order.cond._cacheable = old_cacheable

    def test_fullgraph_capture_with_pytree_module(self):
        from torch._dynamo.functional_export import dynamo_graph_capture_for_export

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.linear1 = torch.nn.Linear(3, 3)
                self.linear2 = torch.nn.Linear(3, 3)
                self.linear3 = torch.nn.Linear(3, 3)

            def forward(self, x):
                return {
                    "y": self.linear2(x[2] + 1),
                    "z": self.linear3(x[1] - 1),
                    "w": self.linear(x[0]["b"] + 2),
                    "v": self.linear1(x[0]["a"] - 2),
                }

        mod = Module()
        compiled_mod = dynamo_graph_capture_for_export(mod)(
            (
                {"a": torch.randn(3, 3), "b": torch.randn(3, 3)},
                torch.randn(3, 3),
                torch.randn(3, 3),
            )
        )

        inputs = (
            {"a": torch.randn(3, 3), "b": torch.randn(3, 3)},
            torch.randn(3, 3),
            torch.randn(3, 3),
        )
        self.assertEqual(compiled_mod(inputs), mod(inputs))

    def test_dynamic_settings(self):
        def fn(x, y):
            return x + y

        def backend(gm, example_inputs):
            self.assertFalse(torch._dynamo.config.automatic_dynamic_shapes)
            return CustomCompiledFunction(gm, example_inputs)

        self.assertTrue(torch._dynamo.config.automatic_dynamic_shapes)
        compiled_fn = torch.compile(
            fn, fullgraph=True, backend=backend, dynamic=False
        ).aot_compile(((torch.randn(3, 4), torch.randn(3, 4)), {}))
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(*inputs)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)

    def test_fullgraph_capture_with_pytree_func(self):
        from torch._dynamo.functional_export import dynamo_graph_capture_for_export

        def foo(x):
            return {
                "y": x[2] + 1,
                "z": x[1] - 1,
                "w": x[0]["b"] + 2,
                "v": x[0]["a"] - 2,
            }

        compiled_foo = dynamo_graph_capture_for_export(foo)(
            (
                {"a": torch.randn(4, 3), "b": torch.randn(3, 2)},
                torch.randn(2, 3),
                torch.randn(3, 4),
            )
        )

        inputs = (
            {"a": torch.randn(4, 3), "b": torch.randn(3, 2)},
            torch.randn(2, 3),
            torch.randn(3, 4),
        )
        self.assertEqual(compiled_foo(inputs), foo(inputs))

    def test_fullgraph_capture_schema_self_arg_no_collision(self):
        """Regression: aten op schemas with `self` at non-first position
        (e.g. `aten.where.self(Tensor condition, Tensor self, Tensor other)`)
        must not produce `def forward(self, condition, self, other):` and
        SyntaxError at `graph_module.recompile()`."""
        from torch._dynamo.functional_export import dynamo_graph_capture_for_export

        cond = torch.tensor([True, False, True, False])
        x = torch.tensor(0.0)
        y = torch.tensor([1.0, 2.0, 3.0, 4.0])
        op = torch.ops.aten.where.self
        compiled = dynamo_graph_capture_for_export(op)(cond, x, y)
        self.assertEqual(compiled(cond, x, y), op(cond, x, y))

    def test_aot_compile_with_closure_save_and_load(self):
        tmp = 2

        def fn(x, y):
            return x + y + tmp

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(*inputs)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(f)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)

    def test_aot_compile_with_super_call(self):
        fn = TestVLLMModel()
        compiled_fn = torch.compile(fn.forward, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((torch.randn(3, 4),), {})
        )
        self.assertEqual(fn.forward.__code__.co_freevars, ("__class__",))
        inputs = (torch.randn(3, 4),)
        expected = fn(*inputs)
        actual = compiled_fn(fn, *inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(f)
        actual = compiled_fn(fn, *inputs)
        self.assertEqual(expected, actual)

    def test_aot_compile_with_global_tensor(self):
        def fn(x, y):
            return x + y + EPS

        def make_inputs():
            return (torch.randn(3, 4), torch.randn(3, 4))

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile((make_inputs(), {}))  # noqa: UNSPECIFIED_BACKEND

        test_inputs = make_inputs()
        self.assertEqual(compiled_fn(*test_inputs), fn(*test_inputs))

    def test_aot_compile_with_default_args(self):
        def fn(x, y=1):
            return x + x

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((torch.randn(3, 4),), {})
        )
        inputs = (torch.randn(3, 4),)
        expected = fn(*inputs)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(f)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_aot_compile_with_aoti(self):
        with torch.device(GPU_TYPE):
            from torch._dynamo.hooks import Hooks

            def fn(x, y):
                return x + y

            def make_inputs():
                return (torch.randn(3, 4), torch.randn(3, 4))

            compiled_fn = torch._dynamo.aot_compile.aot_compile_fullgraph(
                fn,
                (make_inputs(), {}),
                Hooks(),
                torch._TorchCompileAOTInductorWrapper(None, None, None),
            )

            test_inputs = make_inputs()
            expected = fn(*test_inputs)
            actual = compiled_fn(*test_inputs)
            self.assertEqual(expected, actual)
            compiled_fn.save_compiled_function(self.path())
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(*test_inputs)
            self.assertEqual(expected, actual)

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_aot_compile_with_aoti_module(self):
        with torch.device(GPU_TYPE):
            from torch._dynamo.hooks import Hooks

            mod = SimpleLinearModule()

            def make_inputs():
                return (torch.randn(4, 3),)

            compiled_mod = torch._dynamo.aot_compile.aot_compile_module(
                mod,
                [ModelInput(make_inputs(), {}, [])],
                Hooks(),
                torch._TorchCompileAOTInductorWrapper(None, None, None),
            )

            def get_grads(m: torch.nn.Module):
                return {name: p.grad for name, p in m.named_parameters()}

            original_mod = copy.deepcopy(mod)
            test_inputs = make_inputs()
            expected = mod(*test_inputs)
            expected.sum().backward()
            expected_grads = get_grads(mod)

            actual = compiled_mod(*test_inputs)
            self.assertEqual(expected, actual)
            serialized = compiled_mod.serialize()
            compiled_fn = AOTCompiledModel.deserialize(original_mod, serialized)
            actual = compiled_fn(*test_inputs)
            actual.sum().backward()
            self.assertEqual(get_grads(original_mod), expected_grads)

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_aot_compile_with_aoti_torch_compile(self):
        with torch.device(GPU_TYPE):

            def fn(x, y):
                return x + y

            def make_inputs():
                return (torch.randn(3, 4), torch.randn(3, 4))

            compiled_fn = torch.compile(  # noqa: UNSPECIFIED_BACKEND
                fn, fullgraph=True, options={"use_aoti": True}
            ).aot_compile((make_inputs(), {}))
            test_inputs = make_inputs()
            expected = fn(*test_inputs)
            actual = compiled_fn(*test_inputs)
            self.assertEqual(expected, actual)
            compiled_fn.save_compiled_function(self.path())
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(*test_inputs)
            self.assertEqual(compiled_fn._artifacts.backend_name, "aotinductor")
            self.assertEqual(expected, actual)

    @unittest.skipIf(not c10d.is_available(), "requires c10d")
    def test_aot_compile_with_redistribute(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Replicate
        from torch.testing._internal.distributed.fake_pg import FakeStore

        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake", store=fake_store, rank=0, world_size=4
        )
        try:
            mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
            input_tensor = torch.randn(32, 32, device="cpu")
            placements = (Replicate(), Replicate())
            d_input_tensor = DTensor.from_local(input_tensor, mesh, placements)
            mod = RedistributeModel()

            compiled_fn = torch.compile(  # noqa: UNSPECIFIED_BACKEND
                mod,
                fullgraph=True,
            ).forward.aot_compile(((input_tensor, d_input_tensor, mesh), {}))
            inputs = (input_tensor, d_input_tensor, mesh)
            expected = mod(*inputs)
            actual = compiled_fn(mod, *inputs)
            self.assertEqual(expected, actual)
            compiled_fn.save_compiled_function(self.path())
            torch._dynamo.reset()
            with torch.compiler.set_stance("fail_on_recompile"):
                with open(self.path(), "rb") as f:
                    compiled_fn = torch.compiler.load_compiled_function(f)
                actual = compiled_fn(mod, *inputs)
                self.assertEqual(expected, actual)
        finally:
            torch.distributed.destroy_process_group()

    def test_aot_compile_with_captured_module(self):
        mod = SimpleLinearModule()

        fn = mod.forward

        def with_processing(f, *args, **kwargs):
            return f(*args, **kwargs)

        fn = functools.partial(with_processing, fn)

        fn = wrap_forward_function(fn)
        mod.forward = fn

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((torch.randn(4, 3),), {})
        )
        mod.forward = compiled_fn
        with self.assertRaisesRegex(
            RuntimeError,
            r"Failed to serialize the following objects: \[SimpleLinearModule",
        ):
            compiled_fn.save_compiled_function(self.path())
        compiled_fn.save_compiled_function(
            self.path(),
            external_data={"mod": mod},
        )
        with open(self.path(), "rb") as f:
            with self.assertRaisesRegex(RuntimeError, "Missing required external ref"):
                torch.compiler.load_compiled_function(f)

        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(
                f,
                external_data={"mod": mod},
            )
            test_inputs = (torch.randn(4, 3),)
            expected = fn(*test_inputs)
            actual = compiled_fn(*test_inputs)
            self.assertEqual(expected, actual)

    def test_aot_compile_with_captured_module_2(self):
        mod = SimpleLinearModule()

        fn = mod.forward

        def with_processing(f, *args, **kwargs):
            return f(*args, **kwargs)

        fn = functools.partial(with_processing, fn)

        fn = wrap_forward_function(fn)

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((torch.randn(4, 3),), {})
        )
        mod.forward = compiled_fn
        with self.assertRaisesRegex(
            RuntimeError,
            r"Failed to serialize the following objects: \[SimpleLinearModule",
        ):
            compiled_fn.save_compiled_function(self.path())
        compiled_fn.save_compiled_function(
            self.path(),
            external_data={"mod": mod},
        )
        with open(self.path(), "rb") as f:
            with self.assertRaisesRegex(RuntimeError, "Missing required external ref"):
                torch.compiler.load_compiled_function(f)

        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(
                f,
                external_data={"mod": mod},
            )
            test_inputs = (torch.randn(4, 3),)
            expected = fn(*test_inputs)
            actual = compiled_fn(*test_inputs)
            self.assertEqual(expected, actual)

    def test_aot_compile_with_checkpoint(self):
        from torch.utils.checkpoint import checkpoint

        def fn(x, y):
            def compute(x, y):
                return x * 2 + y * 3

            return checkpoint(compute, x, y, use_reentrant=False)

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(*inputs)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(*inputs)
            self.assertEqual(expected, actual)

    def test_external_refs_validation(self):
        """Test that external refs tracking and f_globals parameter work correctly"""

        def fn(x, y):
            return MooType(x + y)

        def make_inputs():
            return (torch.randn(3, 4), torch.randn(3, 4))

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile((make_inputs(), {}))  # noqa: UNSPECIFIED_BACKEND
        test_inputs = make_inputs()
        expected = fn(*test_inputs)
        actual = compiled_fn(*test_inputs)
        self.assertEqual(expected.x, actual.x)
        compiled_fn.save_compiled_function(self.path())

        with self.assertRaisesRegex(RuntimeError, "Missing required external ref"):
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)

        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(
                f, f_globals=fn.__globals__
            )
        actual = compiled_fn(*test_inputs)
        self.assertEqual(expected.x, actual.x)

    def test_builtins_dict_survives_serialization(self):
        """Test that __builtins_dict__ is preserved through serialize/deserialize."""

        def fn(x):
            return x + 1, type

        x = torch.randn(4)
        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(((x,), {}))  # noqa: UNSPECIFIED_BACKEND

        # Save and reload without f_globals
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            loaded_fn = torch.compiler.load_compiled_function(
                f, f_globals=fn.__globals__
            )

        expected = fn(x)
        actual = loaded_fn(x)
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])

    def test_check_compatibility_compares_artifact_against_current_machine(self):
        # CompileArtifacts.check_compatibility must invoke the CACHED
        # SystemInfo's method with the current machine as `other`, the way
        # _DynamoCacheEntry.check_versions does. Reversed, the "artifact
        # predates cpu_codegen_target" skip is evaluated against the current
        # machine -- never None -- so every old artifact is rejected, and every
        # mismatch message reports the two sides the wrong way round.
        def fn(x):
            return x + 1

        compiled = torch.compile(fn, fullgraph=True, backend="eager").aot_compile(
            ((torch.randn(3, 3),), {})
        )
        artifacts = compiled._artifacts
        self.assertEqual(artifacts.device_type, "cpu")
        stale = ("mips", "DEFAULT", 128, (), None, "INVALID")

        # An eager artifact holds no generated code, so there is no baked vector
        # width to protect and the comparison must not run at all -- otherwise
        # capture-here/serve-there, which is the whole point of the feature,
        # rejects an artifact over a target it never used.
        self.assertEqual(artifacts.backend_name, "eager")
        artifacts.system_info = dataclasses.replace(
            artifacts.system_info, cpu_codegen_target=stale
        )
        artifacts.check_compatibility()

        # The rest is about the receiver order, which only a native backend
        # reaches.
        artifacts.backend_name = "inductor"
        current_target = _current_cpu_codegen_target()
        if current_target is None:
            # No usable C++ compiler, so there is no current target to compare
            # against and the skew arms below have nothing to assert. Skipping
            # rather than failing is the point of the lazy probe.
            self.skipTest("no CPU codegen target on this host")

        artifacts.system_info = dataclasses.replace(
            artifacts.system_info, cpu_codegen_target=None
        )
        artifacts.check_compatibility()

        artifacts.system_info = dataclasses.replace(
            artifacts.system_info, cpu_codegen_target=stale
        )
        with self.assertRaises(RuntimeError) as ctx:
            artifacts.check_compatibility()
        message = str(ctx.exception)
        self.assertIn(f"cached={stale}", message)
        self.assertIn(f"current={current_target}", message)

        artifacts.system_info = dataclasses.replace(
            artifacts.system_info, cpu_codegen_target=None, torch_version="0.0.0-fake"
        )
        with self.assertRaisesRegex(RuntimeError, "0.0.0-fake"):
            artifacts.check_compatibility()

    def test_inductor_cpu_capture_records_cpu_codegen_target(self):
        # Pins the recording side: a regression that records None silently
        # disarms check_compatibility via its predates-the-field skip.
        if _current_cpu_codegen_target() is None:
            self.skipTest("no CPU codegen target on this host")

        def fn(x):
            return x + 1

        compiled = torch.compile(fn, fullgraph=True, backend="inductor").aot_compile(
            ((torch.randn(3, 3),), {})
        )
        artifacts = compiled._artifacts
        self.assertEqual(artifacts.device_types, frozenset(("cpu",)))
        self.assertIsNotNone(artifacts.system_info.cpu_codegen_target)

        stale = ("mips", "DEFAULT", 128, (), None, "INVALID")
        artifacts.system_info = dataclasses.replace(
            artifacts.system_info, cpu_codegen_target=stale
        )
        with self.assertRaisesRegex(RuntimeError, "CPU codegen target"):
            artifacts.check_compatibility()

    def test_graph_device_types_ignores_placeholders_without_a_device(self):
        # Under dynamic shapes the leading placeholder is a SymInt, which has no
        # device. Reading only the first meta value reported "cpu" for this
        # all-accelerator graph, which armed the toolchain probe and a hard
        # load-time refusal over CPU code the artifact does not hold.
        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env):
            x = torch.empty(2, device="cuda")
            s0 = shape_env.create_unbacked_symint()
        graph = torch.fx.Graph()
        graph.placeholder("s0").meta["val"] = s0
        x_node = graph.placeholder("x")
        x_node.meta["val"] = x
        graph.call_function(torch.ops.aten.add.Tensor, (x_node, 1)).meta["val"] = x
        self.assertEqual(_graph_device_types(graph), frozenset(("cuda",)))

        graph = torch.fx.Graph()
        graph.placeholder("n").meta["val"] = 4
        self.assertEqual(_graph_device_types(graph), frozenset())
        self.assertEqual(_graph_device_types(None), frozenset())

    def test_graph_device_types_ignores_autocast_device_strings(self):
        # An autocast device type is a plain string positional arg of
        # _enter_autocast, not a device position, so it must not inject a
        # device no tensor lives on. torch.autocast("cuda", enabled=False) in
        # an otherwise CPU-only graph -- the recipe in autocast_mode's own
        # docstring -- would otherwise make the artifact refuse to load on the
        # very host that saved it. .to()/device= are still read.
        with FakeTensorMode():
            cpu = torch.empty(2)
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = cpu
        graph.call_function(torch.amp._enter_autocast, ("cuda", None, True, None))
        graph.call_function(torch.ops.aten.add.Tensor, (x, 1)).meta["val"] = cpu
        self.assertEqual(_graph_device_types(graph), frozenset(("cpu",)))

        # A checkpointed accelerator module enters torch.amp.autocast("cpu")
        # unconditionally; that "cpu" string must not arm the CPU codegen gate.
        with FakeTensorMode():
            cuda_meta = torch.empty(2, device="cuda")
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = cuda_meta
        graph.call_function(torch.amp._enter_autocast, ("cpu", None, True, None))
        self.assertEqual(_graph_device_types(graph), frozenset(("cuda",)))

        # Real device positions are still read: a .to() device arg and a
        # device= kwarg both count.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.call_method("to", (x, "mps"))
        graph.call_function(torch.ops.aten.ones.default, ([2],), {"device": "cuda"})
        self.assertEqual(_graph_device_types(graph), frozenset(("mps", "cuda")))

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_mixed_device_graph_arms_cpu_codegen_target(self):
        # A mixed cpu+accelerator graph collapses device_type to the
        # accelerator, but inductor still emits native CPU kernels for the cpu
        # half, so the codegen target must be recorded and compared anyway.
        if _current_cpu_codegen_target() is None:
            self.skipTest("no CPU codegen target on this host")

        def fn(x, y):
            return x + 1, y + 1

        compiled = torch.compile(fn, fullgraph=True, backend="inductor").aot_compile(
            ((torch.randn(4), torch.randn(4, device=GPU_TYPE)), {})
        )
        artifacts = compiled._artifacts
        self.assertEqual(artifacts.device_type, GPU_TYPE)
        self.assertIn("cpu", artifacts.device_types)
        self.assertIsNotNone(artifacts.system_info.cpu_codegen_target)

        stale = ("mips", "DEFAULT", 128, (), None, "INVALID")
        artifacts.system_info = dataclasses.replace(
            artifacts.system_info, cpu_codegen_target=stale
        )
        with self.assertRaisesRegex(RuntimeError, "CPU codegen target"):
            artifacts.check_compatibility()

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_cross_aot_compile(self):
        """Test cross-compilation using fake tensors and backward correctness"""
        from torch._subclasses.fake_tensor import FakeTensorMode

        def fn(x, y):
            return x + y

        with FakeTensorMode(allow_non_fake_inputs=True):
            fake_inputs = (
                torch.randn(3, 4, device=GPU_TYPE, requires_grad=True),
                torch.randn(3, 4, device=GPU_TYPE, requires_grad=True),
            )
        compiled_fn = torch.compile(  # noqa: UNSPECIFIED_BACKEND
            fn,
            fullgraph=True,
        ).aot_compile((fake_inputs, {}))

        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()

        with open(self.path(), "rb") as f:
            loaded_fn = torch.compiler.load_compiled_function(f)

        inputs = (
            torch.randn(3, 4, device=GPU_TYPE, requires_grad=True),
            torch.randn(3, 4, device=GPU_TYPE, requires_grad=True),
        )
        expected = fn(*inputs)
        actual = loaded_fn(*inputs)
        self.assertEqual(expected, actual)

        # Backward check: compare gradients between eager and loaded compiled function
        eager_loss = expected.sum()
        eager_loss.backward()
        eager_grads = tuple(inp.grad.clone() for inp in inputs)

        # Reset grads for compiled run
        for inp in inputs:
            inp.grad = None

        compiled_out = loaded_fn(*inputs)
        compiled_loss = compiled_out.sum()
        compiled_loss.backward()
        compiled_grads = tuple(inp.grad.clone() for inp in inputs)

        for eg, cg in zip(eager_grads, compiled_grads):
            self.assertEqual(eg, cg)

    @unittest.skipIf(not c10d.is_available(), "requires c10d")
    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_cross_compile_realistic_transformer_model(self):
        """
        Test cross-compilation with transformer model with DTensors,
        FlexAttention, and checkpointing using the compiler toolkit.
        Compares compiled execution against eager execution for bitwise
        equivalence of logits and gradients.
        """
        from torch.distributed._tensor import DTensor
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Placement, Replicate, Shard
        from torch.testing._internal.distributed.fake_pg import FakeStore

        def dtensorify_module(
            module: nn.Module,
            device_mesh,
            *,
            param_placements: list[Placement] | None = None,
            buffer_placements: list[Placement] | None = None,
        ) -> None:
            if param_placements is None:
                param_placements = [Replicate()]
            if buffer_placements is None:
                buffer_placements = [Replicate()]

            for name, p in list(module.named_parameters(recurse=False)):
                if p is None or isinstance(p, DTensor):
                    continue
                dt = DTensor.from_local(p.data, device_mesh, param_placements)
                new_p = nn.Parameter(dt, requires_grad=p.requires_grad)
                setattr(module, name, new_p)

            for name, b in list(module.named_buffers(recurse=False)):
                if b is None or isinstance(b, DTensor):
                    continue
                dt = DTensor.from_local(b, device_mesh, buffer_placements)
                module._buffers[name] = dt

            for child in module.children():
                dtensorify_module(
                    child,
                    device_mesh,
                    param_placements=param_placements,
                    buffer_placements=buffer_placements,
                )

        def init_weights_deterministic(module: nn.Module, seed: int = 42) -> None:
            """
            Initialize module weights deterministically using a fixed seed.
            This ensures reproducible results across eager and compiled runs.
            """
            torch.manual_seed(seed)
            getattr(torch, GPU_TYPE).manual_seed(seed)
            for name, param in module.named_parameters():
                if param.requires_grad:
                    local_param = (
                        param.to_local() if isinstance(param, DTensor) else param
                    )
                    local_param.data.normal_(mean=0.0, std=0.02)
            for name, buf in module.named_buffers():
                local_buf = buf.to_local() if isinstance(buf, DTensor) else buf
                local_buf.data.normal_(mean=0.0, std=0.02)

        fake_store = FakeStore()
        c10d.init_process_group(backend="fake", store=fake_store, rank=0, world_size=1)

        try:
            rank = c10d.get_rank()
            device = torch.device(f"{GPU_TYPE}:{rank}")
            vocab_size = 1000
            embed_dim = 256
            num_heads = 8
            num_kv_heads = 2
            num_layers = 2
            max_seq_len = 32
            batch_size = 2
            seq_len = 16

            device_mesh = init_device_mesh(
                GPU_TYPE,
                (1,),
                mesh_dim_names=("dp",),
            )

            with torch.device("meta"):
                model = Transformer(
                    vocab_size,
                    embed_dim,
                    num_heads,
                    num_layers,
                    max_seq_len,
                    num_kv_heads=num_kv_heads,
                    device_mesh=device_mesh,
                )

            dtensorify_module(
                model,
                device_mesh,
                param_placements=[Replicate()],
                buffer_placements=[Replicate()],
            )

            outer_fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
            with outer_fake_mode:
                # Convert meta tensors -> fake tensors on target device
                model.to_empty(device=device)

                local_input_ids = torch.randint(
                    0, vocab_size, (batch_size, seq_len), device=device
                )
                input_ids_dt = DTensor.from_local(
                    local_input_ids, device_mesh, [Shard(0)]
                )

            from torch._dynamo.functional_export import dynamo_graph_capture_for_export

            gm = dynamo_graph_capture_for_export(model)(input_ids_dt)

            fake_mode = gm.meta["fake_mode"]

            # Pre-create a temp file path and remove delete=False since we control cleanup
            with (
                tempfile.NamedTemporaryFile(suffix=".pt") as f,
                torch._functorch.config.patch(force_autograd_cache=True),
            ):
                serialization_path = f.name

                with contextlib.ExitStack() as stack:
                    if fake_mode is not None:
                        stack.enter_context(tracing(TracingContext(fake_mode)))
                        stack.enter_context(fake_mode)

                    jd = aot_export_joint_with_descriptors(
                        stack,
                        gm,
                        (input_ids_dt,),
                    )

                    compiled_wrapper = aot_compile_joint_with_descriptors(
                        jd,
                        fw_compiler=regional_inductor,
                        bw_compiler=regional_inductor,
                        serializable=True,
                    )

                    f.write(
                        BundledAOTAutogradSerializableCallable.serialize_compile_artifacts(
                            compiled_wrapper
                        )
                    )
                    f.flush()

                with open(serialization_path, "rb") as f_r:
                    loaded_fn = BundledAOTAutogradSerializableCallable.deserialize_compile_artifacts(
                        f_r.read()
                    )

                # Create compiled model with deterministic initialization
                local_input_ids = torch.randint(
                    0, vocab_size, (batch_size, seq_len), device=device
                )
                input_ids_dt = DTensor.from_local(
                    local_input_ids, device_mesh, [Shard(0)]
                )
                targets = torch.randint(
                    0, vocab_size, (batch_size, seq_len), device=device
                )

                compiled_model = Transformer(
                    vocab_size,
                    embed_dim,
                    num_heads,
                    num_layers,
                    max_seq_len,
                    num_kv_heads=num_kv_heads,
                    device_mesh=device_mesh,
                )
                dtensorify_module(
                    compiled_model,
                    device_mesh,
                    param_placements=[Replicate()],
                    buffer_placements=[Replicate()],
                )
                compiled_model.to_empty(device=device)
                init_weights_deterministic(compiled_model)

                eager_model = Transformer(
                    vocab_size,
                    embed_dim,
                    num_heads,
                    num_layers,
                    max_seq_len,
                    num_kv_heads=num_kv_heads,
                    device_mesh=device_mesh,
                )
                dtensorify_module(
                    eager_model,
                    device_mesh,
                    param_placements=[Replicate()],
                    buffer_placements=[Replicate()],
                )
                eager_model.to_empty(device=device)
                init_weights_deterministic(eager_model)

                # Run compiled forward pass
                (compiled_logits_dt,) = loaded_fn(
                    *compiled_model.parameters(),
                    *compiled_model.buffers(),
                    input_ids_dt,
                )
                compiled_logits = (
                    compiled_logits_dt.to_local()
                    if isinstance(compiled_logits_dt, DTensor)
                    else compiled_logits_dt
                )

                # Run eager forward pass with same input
                eager_logits_dt = eager_model(input_ids_dt)
                eager_logits = (
                    eager_logits_dt.to_local()
                    if isinstance(eager_logits_dt, DTensor)
                    else eager_logits_dt
                )

                # Compare logits for bitwise equivalence
                self.assertEqual(
                    compiled_logits,
                    eager_logits,
                    msg="Compiled and eager logits should be bitwise equivalent",
                )

                # Run backward pass on compiled model
                compiled_loss = F.cross_entropy(
                    compiled_logits.view(-1, vocab_size), targets.view(-1)
                )
                compiled_loss.backward()
                compiled_grads = {
                    name: p.grad.clone() if p.grad is not None else None
                    for name, p in compiled_model.named_parameters()
                }

                # Run backward pass on eager model
                eager_loss = F.cross_entropy(
                    eager_logits.view(-1, vocab_size), targets.view(-1)
                )
                eager_loss.backward()
                eager_grads = {
                    name: p.grad.clone() if p.grad is not None else None
                    for name, p in eager_model.named_parameters()
                }

                # Compare losses for bitwise equivalence
                self.assertEqual(
                    compiled_loss,
                    eager_loss,
                    msg="Compiled and eager losses should be bitwise equivalent",
                )

                # Compare gradients for bitwise equivalence
                for name in compiled_grads:
                    self.assertEqual(
                        compiled_grads[name],
                        eager_grads[name],
                        msg=lambda msg: f"{msg}\nGradients for {name} should be bitwise equivalent",
                    )
        finally:
            c10d.destroy_process_group()


class TestAOTCompilePickler(torch._inductor.test_case.TestCase):
    def test_pickler_rebuilds_a_nested_function_faithfully(self):
        # The pickler passed __qualname__ where FunctionType wants __name__, so
        # a reloaded function reported the dotted qualname as its __name__; it
        # read cell_contents unguarded, so an EMPTY cell raised ValueError out
        # of the pickler; and it dropped __kwdefaults__ and __dict__ outright.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        def outer():
            scale = None

            def inner(*, k=1):
                return unset, scale

            inner.__name__ = "renamed"
            inner.tag = 2.0
            if inner is None:
                unset = 1  # never runs, so the cell inner closes over stays empty
            return inner

        fn = outer()
        cells = dict(zip(fn.__code__.co_freevars, fn.__closure__))
        with self.assertRaisesRegex(ValueError, "empty"):
            cells["unset"].cell_contents
        buf = io.BytesIO()
        AOTCompilePickler({}, buf).dump(fn)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertEqual(out.__name__, "renamed")
        self.assertEqual(out.__qualname__, fn.__qualname__)
        self.assertEqual(out.__kwdefaults__, {"k": 1})
        self.assertEqual(out.tag, 2.0)
        cells = dict(zip(out.__code__.co_freevars, out.__closure__))
        with self.assertRaisesRegex(ValueError, "empty"):
            cells["unset"].cell_contents
        self.assertIsNone(cells["scale"].cell_contents)

    def test_pickler_resolves_and_keeps_a_serializable_annotation(self):
        # A <locals> function's annotations are resolved to real values and
        # kept verbatim when they serialize, so the reloaded function carries
        # the same annotations it was captured with.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        def outer():
            def inner(x: list[int]) -> int:
                return len(x)

            return inner

        fn = outer()
        buf = io.BytesIO()
        AOTCompilePickler({}, buf).dump(fn)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertEqual(out.__annotations__, {"x": list[int], "return": int})
        self.assertEqual(out([1, 2, 3]), 3)

    def test_pickler_drops_an_unpicklable_annotation_and_keeps_the_rest(self):
        # A <locals> class resolves fine on every version but pickle cannot
        # reference it, so annotating with one used to fail the whole dump. It
        # is now dropped per value, and a serializable sibling annotation on the
        # same function survives, so the function still reloads and runs.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        def outer():
            class Cfg:
                pass

            def inner(x: Cfg, y: int) -> int:
                return y

            return inner

        fn = outer()
        buf = io.BytesIO()
        AOTCompilePickler({}, buf).dump(fn)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertEqual(out.__annotations__, {"y": int, "return": int})
        self.assertEqual(out(object(), 5), 5)

    @unittest.skipIf(sys.version_info < (3, 12), "PEP 695 type params are 3.12+")
    def test_pickler_drops_unpicklable_type_params(self):
        # A PEP 695 function-scoped TypeVar pickles to its bare name and then
        # fails the module lookup, so a nested generic used to abort the dump
        # even after its annotations were pruned. The whole __type_params__
        # tuple is now dropped and the function still reloads and runs. Defined
        # via exec so this file still parses below 3.12.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        ns = {"__name__": __name__}
        exec(
            "def outer():\n"
            "    def inner[T](x: T) -> T:\n"
            "        return x\n"
            "    return inner\n",
            ns,
        )
        fn = ns["outer"]()
        buf = io.BytesIO()
        AOTCompilePickler({}, buf).dump(fn)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertEqual(out.__type_params__, ())
        self.assertEqual(out.__annotations__, {})
        self.assertEqual(out(5), 5)

    @unittest.skipIf(
        sys.version_info < (3, 14), "PEP 649 FORWARDREF annotations are 3.14+"
    )
    def test_pickler_drops_an_unresolvable_nested_annotation(self):
        # On 3.14 a TYPE_CHECKING-only name reads back as a ForwardRef even when
        # nested (list[Bar] -> list[ForwardRef('Bar')]), which pickle cannot
        # follow. Resolving raises, so the whole annotation set is dropped and
        # the function still reloads and runs.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        def outer():
            def inner(x: list[Bar]):  # noqa: F821
                return x

            return inner

        fn = outer()
        buf = io.BytesIO()
        AOTCompilePickler({}, buf).dump(fn)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertEqual(out.__annotations__, {})
        self.assertEqual(out([1, 2]), [1, 2])


class TestTritonKernelSerialization(torch._inductor.test_case.TestCase):
    """Tests for triton kernel side table serialization."""

    def test_kernel_side_table_serialization_roundtrip(self):
        """
        Test that the kernel_side_table is properly serialized and restored.

        This test verifies that when we serialize the triton kernel side table
        and then clear it (simulating a new process), deserialization properly
        restores the kernels so they can be looked up by index.

        Without this fix, deserialization in a new process would fail with:
            AssertionError: Kernel index X not found in id_to_kernel
        """
        from torch._dynamo.aot_compile_types import (
            _deserialize_triton_kernel,
            _serialize_triton_kernel,
        )
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        try:
            # Create a mock kernel-like object that mimics triton JITFunction structure.
            # Triton JITFunction has a `fn` attribute pointing to the wrapped function.
            class MockTritonKernel:
                def __init__(self, fn):
                    self.fn = fn

            # Use a real importable function (torch.sin) as the wrapped function
            mock_kernel = MockTritonKernel(torch.sin)

            # Add the kernel to the side table (this is what dynamo does during tracing)
            kernel_idx = kernel_side_table.add_kernel(mock_kernel)

            # Add some constant args too
            const_args = {"BLOCK_SIZE": 128, "num_warps": 4}
            const_args_idx = kernel_side_table.add_constant_args(const_args)

            # Simulate serialization: capture the kernel side table state
            triton_kernels = {
                idx: _serialize_triton_kernel(kernel)
                for idx, kernel in kernel_side_table.id_to_kernel.items()
            }
            triton_constant_args = dict(kernel_side_table.constant_args)

            # Simulate a new process by clearing the side table
            kernel_side_table.reset_table()

            # Verify the table is empty - looking up the kernel should fail
            with self.assertRaisesRegex(AssertionError, "not found in id_to_kernel"):
                kernel_side_table.get_kernel(kernel_idx)

            # Simulate deserialization: restore the kernel side table
            for idx, kernel_info in triton_kernels.items():
                restored_kernel = _deserialize_triton_kernel(kernel_info)
                kernel_side_table.id_to_kernel[idx] = restored_kernel
                kernel_side_table.kernel_to_id[restored_kernel] = idx

            for idx, args in triton_constant_args.items():
                kernel_side_table.constant_args[idx] = args

            # Now the kernel lookup should succeed
            restored = kernel_side_table.get_kernel(kernel_idx)
            # The restored kernel is torch.sin (the underlying function), not the mock wrapper
            self.assertIs(restored, torch.sin)

            # Constant args should also be restored
            self.assertEqual(
                kernel_side_table.constant_args[const_args_idx], const_args
            )
        finally:
            kernel_side_table.reset_table()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
