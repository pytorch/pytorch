# Owner(s): ["module: dynamo"]

import contextlib
import copy
import functools
import inspect
import multiprocessing as mp
import os
import pickle
import tempfile
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
from torch._dynamo.aot_compile import AOTCompiledModel, ModelInput, SerializableCallable
from torch._dynamo.aot_compile_types import BundledAOTAutogradSerializableCallable
from torch._dynamo.exc import PackageError, Unsupported
from torch._dynamo.package import DynamoCache
from torch._dynamo.precompile_context import PrecompileContext
from torch._functorch.aot_autograd import (
    aot_compile_joint_with_descriptors,
    aot_export_joint_with_descriptors,
)
from torch._guards import tracing, TracingContext
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx._graph_pickler import GraphPickler
from torch.fx.passes.regional_inductor import regional_inductor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    TEST_CUDA,
)
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
            MultiHeadSelfAttention._flex_attention_cache[compile_key] = torch.compile(
                flex_attention, **compile_spec
            )
        self._flex_attention = MultiHeadSelfAttention._flex_attention_cache[compile_key]

        # Also compile create_block_mask
        if MultiHeadSelfAttention._create_block_mask_fn is None:
            MultiHeadSelfAttention._create_block_mask_fn = torch.compile(
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
    except BaseException as exc:  # noqa: BLE001
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

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(
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
            assert torch.allclose(actual, expected)
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

        torch.compile(warmup_fn, fullgraph=True).aot_compile(
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )
        torch._dynamo.reset()

        with torch.no_grad():
            compiled_fn = torch.compile(target_fn, fullgraph=True).aot_compile(
                ((torch.randn(3, 4), torch.randn(3, 4)), {})
            )

        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        with torch.no_grad():
            actual = compiled_fn(*inputs)
            expected = target_fn(*inputs)
            assert torch.allclose(actual, expected)


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
        assert isinstance(model, torch._dynamo.eval_frame.OptimizedModule)
        model._aot_compile(inputs)

        with torch.compiler.set_stance("fail_on_recompile"):
            model.eval()
            eager_inputs = (torch.randn(3, 3),)
            expected = mod(*eager_inputs)
            actual = model(*eager_inputs)
            assert torch.allclose(expected, actual)
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
            assert isinstance(model, torch._dynamo.eval_frame.OptimizedModule)
            with open(path, "rb") as f:
                data = f.read()
                model._load_aot_compiled_module(data)

            with torch.compiler.set_stance("fail_on_recompile"):
                model.eval()
                eager_inputs = (torch.randn(3, 3),)
                expected = mod(*eager_inputs)
                actual = model(*eager_inputs)
                assert torch.allclose(expected, actual)


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
                    assert arg.shape[0] > 1

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

    def test_decorated_function_with_functools_wrap_aot(self):
        def check_inputs(fn):
            @functools.wraps(fn)
            def _fn(*args, **kwargs):
                for arg in args:
                    assert arg.shape[0] > 1

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

    def test_aot_compile_source_info(self):
        from torch._dynamo.package import SourceInfo

        def fn(x, y):
            return MY_LAMBDA(x) + y

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(
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
            lambda: torch.compile(foo, fullgraph=True).aot_compile(
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
                    assert arg.shape[0] > 1

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
        assert hasattr(backend_result.compiled_fn, "serialize")
        self.assertIsNotNone(backend_result.compiled_fn.serialize)

    def test_aot_compile_portable_guards_unsafe(self):
        def fn(xy):
            return xy[0] + xy[1]

        compiled_fn = torch.compile(
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
        assert hasattr(backend_result.compiled_fn, "serialize")
        self.assertIsNotNone(backend_result.compiled_fn.serialize)

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

    def test_aot_compile_with_closure_save_and_load(self):
        tmp = 2

        def fn(x, y):
            return x + y + tmp

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(
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
        compiled_fn = torch.compile(fn.forward, fullgraph=True).aot_compile(
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

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile((make_inputs(), {}))

        test_inputs = make_inputs()
        self.assertEqual(compiled_fn(*test_inputs), fn(*test_inputs))

    def test_aot_compile_with_default_args(self):
        def fn(x, y=1):
            return x + x

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(
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

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    def test_aot_compile_with_aoti(self):
        with torch.device("cuda"):
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

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    def test_aot_compile_with_aoti_module(self):
        with torch.device("cuda"):
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

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    def test_aot_compile_with_aoti_torch_compile(self):
        with torch.device("cuda"):

            def fn(x, y):
                return x + y

            def make_inputs():
                return (torch.randn(3, 4), torch.randn(3, 4))

            compiled_fn = torch.compile(
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

            compiled_fn = torch.compile(
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

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(
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

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(
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

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(
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

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile((make_inputs(), {}))
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

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    def test_cross_aot_compile(self):
        """Test cross-compilation using fake cuda tensors and backward correctness"""
        from torch._subclasses.fake_tensor import FakeTensorMode

        def fn(x, y):
            return x + y

        with FakeTensorMode(allow_non_fake_inputs=True):
            fake_inputs = (
                torch.randn(3, 4, device="cuda", requires_grad=True),
                torch.randn(3, 4, device="cuda", requires_grad=True),
            )
        compiled_fn = torch.compile(
            fn,
            fullgraph=True,
        ).aot_compile((fake_inputs, {}))

        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()

        with open(self.path(), "rb") as f:
            loaded_fn = torch.compiler.load_compiled_function(f)

        inputs = (
            torch.randn(3, 4, device="cuda", requires_grad=True),
            torch.randn(3, 4, device="cuda", requires_grad=True),
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
    @unittest.skipIf(not TEST_CUDA, "requires cuda")
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
            torch.cuda.manual_seed(seed)
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
            device = torch.device(f"cuda:{rank}")
            vocab_size = 1000
            embed_dim = 256
            num_heads = 8
            num_kv_heads = 2
            num_layers = 2
            max_seq_len = 32
            batch_size = 2
            seq_len = 16

            device_mesh = init_device_mesh(
                "cuda",
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

            gm = dynamo_graph_capture_for_export(model, restore_state_dict=True)(
                input_ids_dt
            )

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
                        msg=f"Gradients for {name} should be bitwise equivalent",
                    )
        finally:
            c10d.destroy_process_group()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
