# Owner(s): ["module: dynamo"]

import builtins
import collections
import contextlib
import copy
import dataclasses
import datetime
import functools
import gc
import importlib
import inspect
import io
import multiprocessing as mp
import os
import pickle
import re
import signal
import sys
import tempfile
import threading
import types
import typing
import unittest
import weakref
from collections import namedtuple
from collections.abc import Callable
from contextlib import contextmanager
from unittest.mock import Mock, patch

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
    _GuardScope,
    _names_a_missing_global,
    _resolve_guard_scope,
    _warn_dropped_module_dispatch,
    AOTCompiledFunction,
    AOTCompiledModel,
    ModelInput,
    SerializableCallable,
)
from torch._dynamo.aot_compile_types import BundledAOTAutogradSerializableCallable
from torch._dynamo.exc import PackageError, Unsupported
from torch._dynamo.graph_utils import _graph_device_types
from torch._dynamo.guards import CheckFunctionManager
from torch._dynamo.package import (
    _collapse_device_types,
    DynamoCache,
    load_guards_state,
    SystemInfo,
)
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.utils import CleanupHook
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
from torch.testing._internal.common_utils import (
    disable_gc,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils.checkpoint import checkpoint


MY_LAMBDA = lambda x: x + 1  # noqa: E731

EPS = torch.tensor(1e-7)
AOT_TEST_TYPEVAR = typing.TypeVar("AOT_TEST_TYPEVAR")
# The fixed-prefix families Dynamo mints through unique_id,
# unique_id_unbound_in and make_compiled_fn_name and binds into the module dict
# of the function being compiled -- this module's dict, for a function defined
# here -- plus the __import_* aliases a load seeds into its guard scope. Not
# every name a compile can bind: variables/builtin.py mints under the builtin's
# own __name__, which no literal tuple can enumerate. A leftover matters only
# when a later mint tries that exact name -- same prefix, at the index the
# counter is on -- which is what a test pre-binding the next minted name needs;
# an __import_* leftover reaches no counter, and is listed because it is
# indistinguishable from the alias a load has to bind; a ___unnamed_scope_* one,
# minted by install_global_by_id off id() and compile id rather than a counter,
# is listed for the same reason, bound to the live dict it stands in for the key
# a load has to seed.
_MINTED_PREFIXES = (
    "__import_",
    "__builtins_dict__",
    "___unnamed_scope",
    "__compiled_fn",
    "__resume_at",
    "__comprehension_",
    "__gen_rand_values",
    "__warnings_warn_wrapper",
)


def _aot_pep695_generic(body="return x"):
    # `def inner[T](x: T) -> T`, built via exec so this file parses below 3.12.
    ns = {"__name__": __name__}
    exec(
        "def outer():\n"
        "    def inner[T](x: T) -> T:\n"
        f"        {body}\n"
        "    return inner\n",
        ns,
    )
    return ns["outer"]()


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


# A namespace that belongs to no module, so Dynamo has no import source for it:
# an inlined frame reading a global from here is guarded through a minted
# ___unnamed_scope_<id(dict)>_c<n> key rather than through a module alias.
_UNNAMED_SCOPE_NS = {
    "__name__": "aot_compile_not_a_registered_module",
    # A tensor, so a graph reading it LIFTS it and records the minted key in
    # used_globals, where the str above is only specialized on.
    "AOT_NS_SCALE": torch.full((8,), 2.0),
    # A dynamic dim names the minted key in a shape guard's G[...] operands,
    # the one place the default filter, which drops the TENSOR_MATCH, leaves it.
    "AOT_NS_ROWS": torch.randn(8, 4),
}
torch._dynamo.mark_dynamic(_UNNAMED_SCOPE_NS["AOT_NS_ROWS"], 0)
exec(
    "AOT_NS_POOL_MODE = 'sum'\n"
    "def ns_pool_fn(x):\n"
    "    if AOT_NS_POOL_MODE == 'sum':\n"
    "        return x.sum(1)\n"
    "    return x.mean(1)\n"
    "def ns_scale_fn(x):\n"
    "    return x * AOT_NS_SCALE\n"
    "def ns_rows_fn(x):\n"
    "    return x + AOT_NS_ROWS.sum(0)\n",
    _UNNAMED_SCOPE_NS,
)


ns_pool_fn = _UNNAMED_SCOPE_NS["ns_pool_fn"]
ns_scale_fn = _UNNAMED_SCOPE_NS["ns_scale_fn"]
ns_rows_fn = _UNNAMED_SCOPE_NS["ns_rows_fn"]

# The same read from a REGISTERED module of its own: an inlined frame there roots
# its globals at Dynamo's __import_<module> alias, so the dynamic dim names that
# alias in the shape guard's operands instead.
_HELPER_MOD = types.ModuleType("aot_compile_helper_mod")
_HELPER_MOD.HELPER_ROWS = torch.randn(8, 4)
torch._dynamo.mark_dynamic(_HELPER_MOD.HELPER_ROWS, 0)
exec("def helper_rows_fn(x):\n    return x + HELPER_ROWS.sum(0)\n", vars(_HELPER_MOD))
sys.modules[_HELPER_MOD.__name__] = _HELPER_MOD
helper_rows_fn = _HELPER_MOD.helper_rows_fn


def tearDownModule():
    sys.modules.pop(_HELPER_MOD.__name__, None)


def calls_into_an_unnamed_scope(x):
    return ns_pool_fn(x)


class UnnamedScopeModule(torch.nn.Module):
    # One global reached through the unnamed scope's minted key and one, EPS,
    # a kept guard's own source IS, so a load has to seed the first and read
    # the second live.
    def forward(self, x):
        return ns_scale_fn(x) + EPS


class UnnamedScopeRowsModule(torch.nn.Module):
    def forward(self, x):
        return ns_rows_fn(x)


class ImportedRowsModule(torch.nn.Module):
    def forward(self, x):
        return helper_rows_fn(x)


class AttrDictModule(torch.nn.Module):
    # An attribute name ending in G, whose dynamic dim renders in a shape expr
    # as L['self'].myG['k'] -- a G['k'] substring that names no global.
    def __init__(self):
        super().__init__()
        self.myG = {"k": torch.randn(8, 4)}
        torch._dynamo.mark_dynamic(self.myG["k"], 0)

    def forward(self, x):
        return x + self.myG["k"].sum(0)


class SimpleLinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x)


class ScaleModule(torch.nn.Module):
    def forward(self, x):
        return x * 2


class PairModule(torch.nn.Module):
    def forward(self, x, y):
        return x + y


class CustomCallModule(torch.nn.Module):
    def __call__(self, x):
        return super().__call__(x) + 100

    def forward(self, x):
        return x * 2


# Module-level so its guards serialize: a local class fails the capture with
# PackageError before the artifact shape it is there to pin can be observed.
class InheritingLinear(torch.nn.Linear):
    pass


# One hook per per-instance dict nn.Module._call_impl dispatches on, keyed by
# the wording the dropped-dispatch warning uses for it. The _global_* dicts it
# also tests are deliberately outside the warning. A hook on the
# module aot_compile_module was handed is dropped; a FORWARD hook on a CHILD
# module is traced through nn.Module.__call__ and lands in the graph instead,
# while a CHILD's backward hook makes fullgraph capture refuse the module
# outright. All four carry an effect, so a capture that keeps them can be told
# from one that drops them: the forward ones through the RESULT the redirect's
# artifact produces, the backward ones through the GRADIENT it produces, which
# is the only thing they change.
_HOOK_REGISTRARS = {
    "forward pre-hooks": lambda m: m.register_forward_pre_hook(
        lambda mod, args: (args[0] + 1,)
    ),
    "forward hooks": lambda m: m.register_forward_hook(lambda mod, args, out: out * 3),
    "backward pre-hooks": lambda m: m.register_full_backward_pre_hook(
        lambda mod, grad_output: (grad_output[0] * 10,)
    ),
    "backward hooks": lambda m: m.register_full_backward_hook(
        lambda mod, grad_input, grad_output: (grad_input[0] * 10,)
    ),
}


class RaisesOnCompare:
    # Hashes into "foo"'s slot, so a guard's PyDict_Contains(d, "foo") has to
    # compare against this key; the raise leaves the exception set, which CPython
    # turns into SystemError at the pybind boundary of the guard tree. Every key
    # below reaches a raise through that leaf on purpose, so restoring its dead
    # `result == -1` branch turns their raises into ordinary mismatches and the
    # tests using them have to be rewritten.
    # `exc` picks the type: two raises at one index differ only by it, and it is
    # what the dedup key reads.
    def __init__(self, exc=ValueError):
        self.exc = exc

    def __hash__(self):
        return hash("foo")

    def __eq__(self, other):
        raise self.exc("boom from __eq__")


class InterruptsOnCompare:
    # RaisesOnCompare's slot, raising the BaseException it is given -- a
    # KeyboardInterrupt or SystemExit -- which the boundary wraps into
    # SystemError just the same.
    def __init__(self, exc):
        self.exc = exc

    def __hash__(self):
        return hash("foo")

    def __eq__(self, other):
        raise self.exc


class CyclesOnCompare:
    # RaisesOnCompare's slot, raising a SystemError chain that CLOSES on itself:
    # nothing stops user code from making one exception the cause of its own
    # cause, and the boundary wraps that like any other raise, leaving the unwrap
    # a chain with no end.
    def __hash__(self):
        return hash("foo")

    def __eq__(self, other):
        outer = SystemError("outer cycle")
        inner = SystemError("inner cycle")
        outer.__cause__ = inner
        inner.__cause__ = outer
        raise outer


class WrapsAnInterruptOnCompare:
    # RaisesOnCompare's slot again, raising the shape one unwrap hop misses: an
    # explicit `raise SystemError(...) from <interrupt>`, which the boundary wraps
    # in a second SystemError, leaving the interrupt two hops down.
    def __init__(self, exc):
        self.exc = exc

    def __hash__(self):
        return hash("foo")

    def __eq__(self, other):
        raise SystemError("inner") from self.exc


class HitsThenRaises:
    # Same collision, but the first `hits` compares answer: a dict lookup of
    # "foo" hits that many times and raises on every later probe, so a guard tree
    # that rejects the call cleanly in both dispatch passes still raises while the
    # report re-checks it. The message names the compare so a test can pin which
    # evaluation raised.
    def __init__(self, hits):
        self.hits = hits
        self.compares = 0

    def __hash__(self):
        return hash("foo")

    def __eq__(self, other):
        self.compares += 1
        if self.compares > self.hits:
            raise ValueError(f"boom on compare {self.compares}")
        return True


class RaisesThenHits:
    # The mirror of HitsThenRaises: the first `raises` compares raise and every
    # later one answers, so a tree that raises in the dispatch scan rejects the
    # call cleanly on the second pass and can explain itself in the report.
    def __init__(self, raises):
        self.raises = raises
        self.compares = 0

    def __hash__(self):
        return hash("foo")

    def __eq__(self, other):
        self.compares += 1
        if self.compares <= self.raises:
            raise ValueError(f"boom on compare {self.compares}")
        return True


class DictBranchModule(torch.nn.Module):
    def forward(self, x, d):
        if d is None:
            return x * 5
        if "foo" not in d:
            return x * 2
        return x * 3


class NeverReChecked:
    """Base for a stub guard manager whose LAST check() raises in a test that
    builds a no-match report.

    The report never re-checks an entry whose last evaluation raised, so this
    check_verbose is unreachable today; a regression that reaches it fails on
    the message below, which names the re-check, rather than on an
    AttributeError the report's handler would dress up as the tree's own raise
    -- and aliasing check_verbose to check would print the raise line the test
    expects and let that regression pass. The same hazard shapes the other
    stubs: one that ANSWERS in dispatch and reaches the report defines its own
    check_verbose, so a regression there quotes its rejection rather than a
    dressed-up AttributeError; one on a serving path meets no report and needs
    neither."""

    def check_verbose(self, f_locals):
        raise RuntimeError("the report re-checked a tree that raised in dispatch")


class RaisingTree(NeverReChecked):
    # A stub guard manager whose every check raises RuntimeError(text): the
    # dispatch semantics it pins do not depend on how a tree raised.
    def __init__(self, text):
        self.text = text
        self.checks = 0

    def check(self, f_locals):
        self.checks += 1
        raise RuntimeError(self.text)


class RaisesPerPass(NeverReChecked):
    # A stub guard manager whose every check raises a message naming the pass it
    # is on: dispatch evaluates an enabled tree once per pass, so the text says
    # which of two raises a report quotes.
    def __init__(self):
        self.checks = 0

    def check(self, f_locals):
        self.checks += 1
        raise RuntimeError(f"pass {self.checks} unhappy")


class Accepts:
    # A stub guard manager that matches every call.
    def check(self, f_locals):
        return True


class Rejects:
    # A stub guard manager that refuses every call, with the check_verbose a
    # report path would quote, as RaisesThenRejects has.
    def check(self, f_locals):
        return False

    def check_verbose(self, f_locals):
        return types.SimpleNamespace(
            result=False, verbose_code_parts=["Rejects refuses every call"]
        )


class RaisesThenAccepts:
    # A stub guard manager that raises in the scan and accepts on the second pass.
    def __init__(self):
        self.checks = 0

    def check(self, f_locals):
        self.checks += 1
        if self.checks == 1:
            raise RuntimeError("the scan is unhappy")
        return True


class UnprintableError(RuntimeError):
    # A tree's exception is user code down to its __str__.
    def __str__(self):
        raise TypeError("str() of the tree's exception raised")


class RaisesUnprintable:
    # A stub guard manager whose every check raises an UnprintableError.
    def check(self, f_locals):
        raise UnprintableError("never printed")


class RaisesOnceThenRejects:
    """Stub guard manager that raises `message` on its first check() and rejects
    on every later one: raised in the scan, rejected on the second pass, so the
    report quotes the rejection and the advice standing on it is qualified."""

    def __init__(self, message):
        self.checks = 0
        self.message = message

    def check(self, f_locals):
        self.checks += 1
        if self.checks == 1:
            raise RuntimeError(self.message)
        return False

    def check_verbose(self, f_locals):
        return types.SimpleNamespace(
            result=False, verbose_code_parts=["stub guard rejected"]
        )


class RejectsOnceThenRaises(NeverReChecked):
    """The mirror: rejects on its first check() and raises `message` on every
    later one, so its last evaluation raised and the report must not re-check
    it, while the rejection it did give came before any raise."""

    def __init__(self, message):
        self.checks = 0
        self.message = message

    def check(self, f_locals):
        self.checks += 1
        if self.checks == 1:
            return False
        raise RuntimeError(self.message)


# The sentence the no-match report appends to its ModelInput advice when every
# rejection that advice rests on followed a raise from its own tree.
CAVEAT_AFTER_A_RAISE = "which need not be the one that loaded them. Fix the raise first: every rejection this advice rests on followed a raise from its own tree, and a C++ throw out of a tree can leave that tree's relational guard state stale, so its next check can reject a call it fits or accept one it does not."


# Not the identity: an identity weight makes "read the serialized weight" and
# "dropped the matmul" produce the same tensor.
AOT_HERMETIC_WEIGHT = torch.eye(3) * 3


class HermeticModule(torch.nn.Module):
    def forward(self, x):
        return x @ AOT_HERMETIC_WEIGHT


class EpsOnlyModule(torch.nn.Module):
    def forward(self, x):
        return x * EPS


AOT_UNGUARDED_PARAM = torch.nn.Parameter(torch.ones(3))


class TwoGlobalsModule(torch.nn.Module):
    # Under keep_tensor_guards_unsafe the TENSOR_MATCH on EPS, a plain tensor,
    # is kept and the one on AOT_UNGUARDED_PARAM, a Parameter, is dropped.
    def forward(self, x):
        return x * EPS + AOT_UNGUARDED_PARAM


AOT_LIVE_SCALE = torch.tensor(4.0)


class TwoCertifiedModule(torch.nn.Module):
    # Both globals are plain tensors, so keep_tensor_guards_unsafe keeps a
    # TENSOR_MATCH on each and the certified set has two names in it, which
    # every other fixture here leaves at one.
    def forward(self, x):
        return x * EPS + AOT_LIVE_SCALE


class StoresEpsModule(torch.nn.Module):
    # Reads a certified global and rebinds it, which Dynamo replays as a
    # STORE_GLOBAL into the bytecode's globals rather than into the scope the
    # guards read.
    def forward(self, x):
        global EPS
        y = x * EPS
        EPS = EPS * 2
        return y


class ReturnsEpsModule(torch.nn.Module):
    # Returns a certified global instead of computing with it, so the graph input
    # it was lifted into is pruned as unused and used_globals -- filled from the
    # graph inputs' sources -- never records the name, while the generated
    # bytecode still reads it. The shape the load-time merge exists for.
    def forward(self, x):
        return x + 1, EPS


GLOBAL_POOLING_CONFIG = {"pooling": "sum"}


class GlobalConfigModule(torch.nn.Module):
    def forward(self, x):
        if GLOBAL_POOLING_CONFIG["pooling"] == "sum":
            return x.sum(1)
        return x.mean(1) * 10.0


class NumpyForwardModule(torch.nn.Module):
    @torch.compiler.wrap_numpy
    def forward(self, x):
        return x.sum(1)


def global_config_fn(x):
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


class CountedKey:
    # Hashes into `name`'s slot and answers False to the first `misses`
    # comparisons a lookup of `name` makes against it, so a guard on `name`
    # misses the global that many times and finds it after: one way a live guard
    # tree can reject a call in the dispatch scan and accept it on the re-check.
    # Which operand the dict puts on the left does not matter: str.__eq__
    # returns NotImplemented for a non-str, so this __eq__ runs either way. A
    # lookup that misses fails the tree wherever it sits, so each check() the
    # tree makes consumes exactly one miss however many guards read `name`.
    def __init__(self, name, misses):
        self.name = name
        self.compares, self.misses = 0, misses

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if other is self:
            return True
        self.compares += 1
        return self.compares > self.misses


AOT_BRANCH_SCALE = 3.0

_ACCEPTED_IN_THE_REPORT = "<guards did not accept this call in dispatch and accepted it here: a guard that does not answer consistently, or guarded state that changed between those evaluations>"


class ModeBranchGlobalModule(torch.nn.Module):
    # Only the mode == 1 branch reads a global, so one ModelInput's guards name
    # it and the other's do not.
    def forward(self, x, mode):
        if mode == 1:
            return x * AOT_BRANCH_SCALE
        return x * 2


class SetIterModule(torch.nn.Module):
    # Iterating a set guards each element through a set-index accessor, which
    # reports a failure with NO verbose code parts when the set it is asked about
    # is shorter than the index it recorded.
    def forward(self, x, tags):
        total = 0
        for _ in tags:
            total += 1
        return x * total


def drop_sequence_length_guards(guard_entries):
    # SEQUENCE_LENGTH is the guard that would otherwise name a set of the wrong
    # size before the per-element accessors are reached.
    return [g.guard_type != "SEQUENCE_LENGTH" for g in guard_entries]


class FlakyTwoTensor(TwoTensor):
    # TENSOR_SUBCLASS_METADATA_MATCH installs a LAMBDA_GUARD that calls this on
    # every check. A raise answers false, and check_verbose quotes str(exc) as
    # the entry's only verbose code part.
    metadata_error: Exception | None = None

    @classmethod
    def __metadata_guard__(cls, saved, current):
        if cls.metadata_error is not None:
            raise cls.metadata_error
        return saved == current


def make_masked_forward():
    # A tensor default makes inspect.Signature equality raise (Parameter.__eq__
    # takes bool() of `default == default`), whether or not the body reads it.
    def forward(self, x, mask=torch.ones(3)):
        return x * 2

    return forward


def make_scaling_forward(scale):
    def forward(self, x):
        return x * scale

    return forward


def make_mode_default_forward(default):
    # Signatures that differ only in the default they bind for `mode`.
    def forward(self, x, mode=default):
        return x * 3 if mode == 1 else x * 2

    return forward


def aot_compile_forward(mod, forward, *args):
    # Compiles `mod` for `forward` bound in place of its class's, so one module
    # can be compiled for several forwards; a class forward rebinds to itself.
    mod.forward = types.MethodType(forward, mod)
    model = torch.compile(mod, fullgraph=True, backend="eager")
    model._aot_compile([ModelInput(args=args, kwargs={}, contexts=[])])
    return model.forward.compiled_results[0]


def compile_mode_defaults(x, *doubles_call):
    # Two results of one module whose signatures differ only in the default
    # for `mode`: [0] defaults it to 1 and is compiled for `x`, [1] defaults it
    # to 0 and is compiled for `doubles_call`.
    mod = ScaleModule()
    triples = aot_compile_forward(mod, make_mode_default_forward(1), x)
    doubles = aot_compile_forward(mod, make_mode_default_forward(0), *doubles_call)
    return mod, triples, doubles


def compile_two_signatures(x, second_forward, *second_call):
    # One module, two results: [0] is ScaleModule's own forward compiled for `x`,
    # [1] is `second_forward` compiled for `second_call`. They bind alike iff
    # `second_forward`'s parameters (names, kinds, defaults) and closure cells
    # match forward's, so a caller wanting a non-shared binding varies one.
    mod = ScaleModule()
    doubled = aot_compile_forward(mod, ScaleModule.forward, x)
    second = aot_compile_forward(mod, second_forward, *second_call)
    return AOTCompiledModel(mod, [doubled, second])


class AlwaysRaises(NeverReChecked):
    # Stub guard manager whose every check() raises `message`, for a result on a
    # path that reaches no report: another result serves, or the call leaves
    # __call__ as an exception that records nothing.
    def __init__(self, message):
        self.message = message

    def check(self, f_locals):
        raise RuntimeError(self.message)


class RecordedThread(threading.Thread):
    # Keeps what its target raised, which threading would only print to stderr,
    # so the test that joins it can fail on it.
    failure: Exception | None = None

    def run(self):
        try:
            super().run()
        except Exception as e:
            self.failure = e


class RaisingReprModule(HermeticModule):
    # get_traced_fn formats an unsupported forward into its error before raising,
    # and formatting a functools.partial over this module reaches extra_repr.
    def extra_repr(self):
        raise ValueError("extra_repr")


AOT_POOL_MODE = "sum"

# A dynamic dim on a global is what makes a SHAPE_ENV guard name it -- as a
# literal G['AOT_DYN_ROWS'] inside a Python lambda, since the default filter
# keeps that guard and cpp symbolic shape guards are off by default.
AOT_DYN_ROWS = torch.randn(4, 3)
torch._dynamo.mark_dynamic(AOT_DYN_ROWS, 0)
AOT_SUMMED = torch.randn(4, 3)

# Two globals bound to the SAME tensor: make_dupe_guard refuses to relate a local
# source to a global one, so reading both is what gets a DUPLICATE_INPUT guard
# whose source_b is a GlobalSource.
AOT_DUPE_A = torch.randn(4, 4)
AOT_DUPE_B = AOT_DUPE_A

AOT_CPP_SHAPE_GLOBAL = torch.randn(8, 4)


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


AOT_GLOBAL_LOG: list[int] = []


class MutatesGlobalListModule(torch.nn.Module):
    def forward(self, x):
        AOT_GLOBAL_LOG.append(x.shape[0])
        return x + 1


class ReturnsGlobalListModule(torch.nn.Module):
    def forward(self, x):
        return x + 1, AOT_GLOBAL_LOG


def mutates_global_list(x):
    AOT_GLOBAL_LOG.append(x.shape[0])
    return x + 1


def returns_global_list(x):
    return x + 1, AOT_GLOBAL_LOG


# Two shapes whose generated bytecode LOAD_GLOBALs a global the graph never
# lifted as an input: the side-effect replay of a mutation, and the reconstruct
# of a returned global. As modules for the module load path and as plain
# functions for AOTCompiledFunction.deserialize.
_UNLIFTED_GLOBAL_MODULES = {
    "mutates": MutatesGlobalListModule,
    "returns": ReturnsGlobalListModule,
}
_UNLIFTED_GLOBAL_FUNCTIONS = {
    "mutates": mutates_global_list,
    "returns": returns_global_list,
}


class ParentWithChildModule(torch.nn.Module):
    # Calling a CHILD module routes through nn.Module.__call__, whose hook-dict
    # guards are rooted at Dynamo's synthetic __import_torch_dot_nn_... alias.
    # That is the shape a reloaded artifact could not resolve.
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        # Dynamo records a __builtins_dict___N key on every artifact; referring
        # to a builtin is what roots a BUILTIN_MATCH guard at it.
        # keep_global_guards drops that guard (it derives ID_MATCH), so no kept
        # guard is rooted at the key and a load must leave it unseeded, which is
        # the negative test_load_seeds_exactly_the_recorded_globals pins;
        # keep_builtin_guards keeps the guard instead, which is how the seeding
        # itself gets pinned.
        if not isinstance(x, torch.Tensor):
            raise TypeError(type(x))
        return self.lin(x)


def keep_global_guards(guard_entries):
    # Keep the global guards the default aot_compile filter drops wholesale, and
    # drop any guard whose type or derived type the serializer rejects. That is
    # stricter than the serializer, which lets TYPE_MATCH/BUILTIN_MATCH through
    # despite a derived ID_MATCH -- see keep_builtin_guards below -- and looser on
    # a TYPE_MATCH the serializer refuses for a local class.
    unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
    return [
        g.guard_type not in unsupported
        and not any(d in unsupported for d in g.derived_guard_types)
        for g in guard_entries
    ]


def keep_builtin_guards(guard_entries):
    # keep_global_guards drops BUILTIN_MATCH because it derives ID_MATCH; the
    # guard serializer accepts it anyway (its TYPE_MATCH/BUILTIN_MATCH branch
    # short-circuits before the derived-type check), so keeping it is the only
    # way to exercise a guard ROOTED at G['__builtins_dict___N'].
    unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
    return [
        g.guard_type in ("TYPE_MATCH", "BUILTIN_MATCH")
        or (
            g.guard_type not in unsupported
            and not any(d in unsupported for d in g.derived_guard_types)
        )
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


def _subprocess_save_child_module_artifact(path, guard_filter_fn):
    import torch
    from torch._dynamo import config

    with config.patch(enable_aot_compile=True):
        mod = ParentWithChildModule()
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": guard_filter_fn},
        )
        model._aot_compile(
            [ModelInput(args=(torch.randn(4, 4),), kwargs={}, contexts=[])]
        )
        model._save_aot_compiled_module(path)


def _subprocess_load_then_compile(path):
    import itertools

    import torch
    from torch._dynamo import bytecode_transformation, config

    with config.patch(enable_aot_compile=True):
        mod = ParentWithChildModule()
        model = torch.compile(mod, fullgraph=True, backend="eager")
        with open(path, "rb") as f:
            model._load_aot_compiled_module(f.read())
        # The only serve of the loaded artifact: a guard rooted at a name the
        # load failed to seed raises HERE, not at any assertion below.
        model(torch.randn(4, 4))

        # len() is what puts a BUILTIN_MATCH guard on the compile, so the served
        # call below evaluates a guard rooted at the builtins-dict key the mint
        # landed on rather than none at all.
        def later(x):
            return x + len(x)

        seeded = {k for k in globals() if k.startswith("__builtins_dict__")}
        if len(seeded) != 1:
            raise AssertionError(f"the load must seed one builtins key: {seeded}")
        (taken,) = seeded
        minted = []
        real_unique_id = bytecode_transformation.unique_id

        def recording_unique_id(*args, **kwargs):
            name = real_unique_id(*args, **kwargs)
            minted.append(name)
            return name

        # Land the next mint on the name the load seeded, read off that name and
        # not off both processes' counters happening to stop at the same index:
        # either side burning an id leaves the two names in no conflict and the
        # retry unrun -- and without the retry install_global raises in
        # CleanupHook.create. The mint goes through unique_id_unbound_in, which
        # reads unique_id out of its own module globals, so patching it there
        # sees every name install_global tried.
        rewound = itertools.count(int(taken.rpartition("_")[2]))
        x = torch.randn(3)
        expected = later(x)
        with (
            patch.object(bytecode_transformation, "unique_id", recording_unique_id),
            patch.object(bytecode_transformation, "_unique_id_counter", rewound),
        ):
            compiled = torch.compile(later, fullgraph=True, backend="eager")
            actual = compiled(x)
            if not torch.equal(actual, expected):
                raise AssertionError(f"the skip cost the result: {actual}")
            # A SERVED second call is what ties the answer to the name the mint
            # landed on: its builtin guard has to resolve through that key, in a
            # module dict that still carries the other process's binding of the
            # name it skipped.
            with config.patch(error_on_recompile=True):
                if not torch.equal(compiled(x), expected):
                    raise AssertionError("the served call answered differently")

        # The compile has to have MINTED the taken name and then installed
        # another; an install that overwrote it leaves this set unchanged.
        after = {k for k in globals() if k.startswith("__builtins_dict__")}
        if taken not in minted or after == seeded:
            raise AssertionError(f"no collision on {taken}: {minted} -> {after}")


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


def _aot_wraps_deco(f):
    @functools.wraps(f)
    def wrapper(x):
        return f(x) * 2

    return wrapper


def _aot_wraps_base(x):
    return x + 1


# functools.wraps gives the wrapper _aot_wraps_base's qualname: no "<locals>"
# marker, yet module + qualname resolve to the base, not to the wrapper.
_aot_wraps_helper = _aot_wraps_deco(_aot_wraps_base)


class BottomlessReduce:
    # Every save reduces to a fresh instance, so the pickler recurses without
    # bound on every Python version (3.14's C pickler no longer overflows on a
    # merely deep list).
    def __reduce__(self):
        return (BottomlessReduce, (BottomlessReduce(),))


class WritesBackOnReduce:
    # Reducing it writes onto the function it is stashed on, while that
    # function's __dict__ is being walked.
    def __init__(self, fn):
        self.fn = fn

    def __reduce__(self):
        self.fn.added = 1
        raise TypeError("cannot pickle WritesBackOnReduce")


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

    def test_aot_compile_rebuilds_a_wraps_wrapper_of_a_module_level_function(self):
        # Pickling the wrapper by reference finds the base function instead
        # ("not the same object"), so it is rebuilt from its code object, the
        # same fqn-mismatch rule the guard pickler applies.
        def outer():
            h = _aot_wraps_helper

            def fn(x):
                return h(x) + 1

            return fn

        fn = outer()
        x = torch.randn(3)
        compiled = torch.compile(fn, fullgraph=True, backend="aot_eager").aot_compile(
            ((x,), {})
        )
        compiled.save_compiled_function(self.path())
        torch._dynamo.reset()
        with open(self.path(), "rb") as f:
            loaded = torch.compiler.load_compiled_function(f)
        self.assertEqual(loaded(x), fn(x))

    def test_aot_compile_rebuilt_wrapper_keeps_its_own_module_scope(self):
        # functools.wraps copies __module__ from the wrappee, but the wrapper's
        # code was compiled against the decorator's module: an escaped wrapper
        # must reload with THAT scope, not the wrappee's (same-named globals
        # would otherwise read the wrong value silently).
        deco_mod = types.ModuleType("_aot_deco_mod_for_scope_test")
        deco_mod.SCALE = 100
        exec(
            "import functools\n"
            "def deco(f):\n"
            "    @functools.wraps(f)\n"
            "    def wrapper(x):\n"
            "        return f(x) * SCALE\n"
            "    return wrapper\n",
            deco_mod.__dict__,
        )
        sys.modules[deco_mod.__name__] = deco_mod
        self.addCleanup(sys.modules.pop, deco_mod.__name__, None)
        helper = deco_mod.deco(_aot_wraps_base)
        self.assertEqual(helper.__module__, __name__)

        def outer():
            def fn(x):
                return x + 1, helper

            return fn

        fn = outer()
        x = torch.randn(3)
        compiled = torch.compile(fn, fullgraph=True, backend="aot_eager").aot_compile(
            ((x,), {})
        )
        compiled.save_compiled_function(self.path())
        torch._dynamo.reset()
        with open(self.path(), "rb") as f:
            loaded = torch.compiler.load_compiled_function(f)
        self.assertEqual(loaded(x)[1](1), (1 + 1) * 100)

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
        # runtime env and is rebuilt from its code object at load. Its cells,
        # __defaults__, __kwdefaults__ and __dict__ have to survive: an EMPTY
        # cell failed the old pickler and __kwdefaults__ and __dict__ were
        # dropped; the None cell is asserted so the empty/None distinction stays
        # pinned; see FunctionPicklerBase. (The
        # compiled function itself cannot have an empty cell: capture reads all
        # of its cells up front.)
        def outer():
            scale = None

            def helper(x, y=3, *, k=2):
                if x is None:
                    return unset
                if scale is None:
                    x = x + 1
                return x * k + y

            helper.tag = 2.0
            if helper is None:
                unset = 1  # never runs, so the cell helper closes over stays empty
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
        self.assertEqual(loaded.__defaults__, (3,))
        self.assertEqual(loaded.tag, 2.0)

    def test_aot_compile_prunes_a_helpers_unpicklable_attribute(self):
        # A helper's __dict__ travels with it, but an entry that cannot pickle
        # is dropped per-entry rather than failing the whole save -- the runtime
        # never forces it, so the save succeeds, the reload runs, and the
        # picklable sibling entry survives.
        def outer():
            def helper(x):
                return x * 2

            helper.lock = threading.Lock()
            helper.tag = 2.0
            return helper

        helper = outer()

        def fn(x):
            return helper(x) + 1

        inputs = (torch.randn(3),)
        expected = fn(*inputs)
        compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")
        compiled_fn = compiled_fn.aot_compile((inputs, {}))
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            compiled_fn.save_compiled_function(self.path())
        (line,) = [l for l in logs.output if "dropping" in l]
        self.assertIn("helper.lock (lock) from the artifact", line)
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                loaded = torch.compiler.load_compiled_function(f)
            self.assertEqual(loaded(*inputs), expected)
        (cell,) = loaded._artifacts.runtime_env.closure
        self.assertEqual(cell.cell_contents.__dict__, {"tag": 2.0})

    def test_aot_compile_top_level_annotations_ride_unpruned(self):
        # Known limitation, pinned so a change to it is noticed: the compiled
        # function's OWN annotations travel on CompileArtifacts.signature, which
        # serialize() dumps whole, so a <locals> class there still fails the
        # save; the nested-helper prune does not reach it.
        def outer():
            class Cfg:
                pass

            def fn(x: Cfg):
                return x + 1

            return fn

        fn = outer()
        compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")
        compiled_fn = compiled_fn.aot_compile(((torch.randn(3),), {}))
        with self.assertRaises((AttributeError, pickle.PicklingError)) as cm:
            compiled_fn.save_compiled_function(self.path())
        self.assertIn("Cfg", str(cm.exception))

    def test_aot_compile_reloads_a_helpers_annotations(self):
        # The shipping path: an annotated helper reached through
        # runtime_env.closure keeps the annotations that pickle and drops the
        # <locals> class one.
        def outer():
            class Cfg:
                pass

            def helper(x: Cfg, scale: int = 2) -> int:
                return x * scale

            return helper

        helper = outer()

        def fn(x):
            return helper(x) + 1

        inputs = (torch.randn(3),)
        expected = fn(*inputs)
        compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")
        compiled_fn = compiled_fn.aot_compile((inputs, {}))
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                loaded = torch.compiler.load_compiled_function(f)
            self.assertEqual(loaded(*inputs), expected)
        (cell,) = loaded._artifacts.runtime_env.closure
        self.assertEqual(
            cell.cell_contents.__annotations__, {"scale": int, "return": int}
        )

    def test_save_guidance_when_a_closure_cell_cannot_pickle(self):
        # A closure cell of the compiled function itself, which the artifact
        # carries unpruned (the nested `unused` only exists to make `lock` a free
        # variable of fn), holds a threading.Lock. save preserves the
        # original error (callers/tests match on "cannot pickle") and appends
        # guidance pointing at external_data, rather than reconstructing the
        # exception (a TypeError subclass may take a non-message constructor).
        def outer():
            lock = threading.Lock()

            def fn(x):
                def unused():
                    return lock

                return x + 1

            return fn

        fn = outer()
        compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")
        compiled_fn = compiled_fn.aot_compile(((torch.randn(3),), {}))
        with self.assertRaises((TypeError, pickle.PicklingError)) as cm:
            compiled_fn.save_compiled_function(self.path())
        msg = str(cm.exception)
        # A real newline, not the escaped one a tuple repr would show: this
        # exception has a single argument, so str() renders the message itself.
        self.assertRegex(msg, r"cannot pickle[^\n]*\nSome value reached by the")
        self.assertIn("external_data", msg)

    def test_save_guidance_when_a_locals_class_default_cannot_pickle(self):
        # A <locals> class instance in __defaults__ rides unpruned and pickle
        # rejects it. The C pickler AOTCompilePickler subclasses raises
        # AttributeError "Can't get local object" (3.13 and earlier) or
        # PicklingError "Can't pickle local object" (3.14+); serialize() catches
        # both, and the offender's class name is asserted, which is independent
        # of either version's wording. It gets the external_data guidance
        # appended.
        def outer():
            class Cfg:
                def __init__(self):
                    self.v = 1

            def fn(x, cfg=Cfg()):
                return x + 1

            return fn

        fn = outer()
        compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")
        compiled_fn = compiled_fn.aot_compile(((torch.randn(3),), {}))
        with self.assertRaises((AttributeError, pickle.PicklingError)) as cm:
            compiled_fn.save_compiled_function(self.path())
        msg = str(cm.exception)
        self.assertIn("Cfg", msg)  # the offender, not pickle's per-version wording
        self.assertIn("not picklable", msg)
        self.assertIn("external_data", msg)

    def test_save_guidance_keeps_the_original_exception_object(self):
        # The guidance is appended to the SAME exception, not to a rebuilt one:
        # a TypeError subclass from a user __reduce__ may not take a message.
        class WeirdTypeError(TypeError):
            def __init__(self, code, detail):
                super().__init__(f"weird {code}", detail)

        class Unpicklable:
            def __reduce__(self):
                raise WeirdTypeError(7, "extra")

        def outer():
            value = Unpicklable()

            def fn(x):
                def unused():
                    return value

                return x + 1

            return fn

        fn = outer()
        compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")
        compiled_fn = compiled_fn.aot_compile(((torch.randn(3),), {}))
        with self.assertRaises(WeirdTypeError) as cm:
            compiled_fn.save_compiled_function(self.path())
        self.assertIs(type(cm.exception), WeirdTypeError)
        self.assertIn("weird 7", str(cm.exception))
        self.assertIn("external_data", str(cm.exception))
        self.assertEqual(cm.exception.args[1], "extra")
        # The head argument, not repr(args), is the prefix, so the tail is not
        # embedded twice (str() of a 2-arg exception is still a tuple repr).
        self.assertEqual(str(cm.exception).count("extra"), 1)
        self.assertIn("weird 7\\nSome value", str(cm.exception))

    def test_save_guidance_names_unmarked_modules_recorded_before_the_failure(self):
        # An unmarked nn.Module is recorded rather than raised; when the dump
        # then fails on a lock, the guidance names the module too, so the user
        # does not fix the lock only to hit the module error on the next save.
        def outer():
            mod = torch.nn.Linear(1, 1)
            zz_lock = threading.Lock()  # freevars are sorted: the module dumps first

            def fn(x):
                def unused():
                    return mod, zz_lock

                return x + 1

            return fn

        fn = outer()
        freevars = fn.__code__.co_freevars
        self.assertLess(freevars.index("mod"), freevars.index("zz_lock"))
        compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")
        compiled_fn = compiled_fn.aot_compile(((torch.randn(3),), {}))
        with self.assertRaises((TypeError, pickle.PicklingError)) as cm:
            compiled_fn.save_compiled_function(self.path())
        msg = str(cm.exception)
        self.assertIn("cannot pickle", msg)
        self.assertIn("unmarked nn.Modules", msg)
        self.assertIn("Linear", msg)
        self.assertIn("external_data", msg)

    def test_save_guidance_when_a_default_overflows_the_pickler(self):
        # A value in an unpruned slot that recurses without bound in the C
        # pickler raises RecursionError; it gets the same guidance and the
        # handler itself does not overflow.
        def outer():
            def fn(x, cfg=BottomlessReduce()):
                return x + 1

            return fn

        fn = outer()
        compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")
        compiled_fn = compiled_fn.aot_compile(((torch.randn(3),), {}))
        with self.assertRaises(RecursionError) as cm:
            compiled_fn.save_compiled_function(self.path())
        self.assertIn("external_data", str(cm.exception))

    def test_save_fails_loudly_on_a_helpers_unpicklable_kwdefault(self):
        # Overlaps two pins that predate this commit -- the pickler-level
        # test_pickler_does_not_prune_an_unpicklable_kwdefault, and the
        # "cannot pickle" plus external_data pair in the closure-cell test above
        # -- and adds the one axis neither covers: the unprunable slot belongs to
        # a nested HELPER reached through the runtime env rather than to fn
        # itself (the helper is a local so fn closes over it), and the save still
        # fails loudly with guidance instead of dropping it.
        def outer():
            def helper(x, *, lock=threading.Lock()):
                return x * 2

            return helper

        helper = outer()

        def fn(x):
            return helper(x) + 1

        compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")
        compiled_fn = compiled_fn.aot_compile(((torch.randn(3),), {}))
        with self.assertRaises((TypeError, pickle.PicklingError)) as cm:
            compiled_fn.save_compiled_function(self.path())
        msg = str(cm.exception)
        self.assertIn("cannot pickle", msg)
        self.assertIn("external_data", msg)

    def test_aot_compile_prunes_a_lock_behind_functools_wraps_wrapped(self):
        # functools.wraps writes __wrapped__ into the wrapper's __dict__ and
        # copies the wrappee's __dict__ too, so a helper that merely decorates
        # another function drags the wrapped one (and anything hanging off it)
        # into the artifact. The lock is pruned from both, __wrapped__ itself is
        # kept, and the save succeeds. With __dict__ carried verbatim the save
        # fails on the lock.
        def build():
            def base(x):
                return x * 3

            base.lock = threading.Lock()

            @functools.wraps(base)
            def helper(x):
                return x * 2

            return helper

        helper = build()

        def fn(x):
            return helper(x) + 1

        inputs = (torch.randn(3),)
        expected = fn(*inputs)
        compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")
        compiled_fn = compiled_fn.aot_compile((inputs, {}))
        # One warning per dropped entry, from the real dump only (the probe
        # picklers reach both functions too) and named by the code object, since
        # wraps gave helper base's __qualname__ (co_qualname is 3.11+; 3.10 gets
        # the bare co_name).
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            compiled_fn.save_compiled_function(self.path())
        self.assertEqual(len([l for l in logs.output if "dropping" in l]), 2)
        self.assertTrue(any("helper.lock (lock)" in l for l in logs.output))
        self.assertTrue(any("base.lock (lock)" in l for l in logs.output))
        if sys.version_info >= (3, 11):
            self.assertTrue(any("build.<locals>.helper.lock" in l for l in logs.output))
        with open(self.path(), "rb") as f:
            loaded = torch.compiler.load_compiled_function(f)
        self.assertEqual(loaded(*inputs), expected)
        (cell,) = loaded._artifacts.runtime_env.closure
        rebuilt = cell.cell_contents
        self.assertFalse(hasattr(rebuilt.__wrapped__, "lock"))
        self.assertEqual(set(rebuilt.__dict__), {"__wrapped__"})

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

    def _hide_leaked_dynamo_globals(self):
        # A capture in this process leaks Dynamo's synthetic globals into this
        # module dict. Pop every name at a _MINTED_PREFIXES prefix for the
        # duration of the test, and in cleanup strip whatever the test added
        # before putting the originals back. A test that LOADS an artifact needs
        # the __import_* half popped as well: its guards are rooted at those
        # aliases, so a leaked one resolves and hides the seeding the test
        # covers.
        g = globals()
        # Disown before popping: the hook that installed each name still owns it
        # and fires whenever its code object is collected, which after the
        # update below would take the restored name with it.
        leaked = {}
        for k in [k for k in g if k.startswith(_MINTED_PREFIXES)]:
            CleanupHook.disown(g, k)
            leaked[k] = g.pop(k)
        preexisting = frozenset(g)

        def restore():
            torch._dynamo.reset()
            for k in [k for k in g if k not in preexisting]:
                CleanupHook.disown(g, k)
                del g[k]
            g.update(leaked)

        self.addCleanup(restore)

    def _model_with_stub_trees(self, *stubs, opt_out=False):
        # One ScaleModule result per stub, each swapped in for the real guard
        # tree and opted out when asked: the artifact the stub-tree tests share.
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        inputs = [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        model._aot_compile(inputs * len(stubs))
        for result, stub in zip(model.forward.compiled_results, stubs):
            result._artifacts.guard_manager = stub
            if opt_out:
                result.disable_guard_check()
        return model

    def _check_module_global_guard_dispatch(self, make_mod, set_mode):
        # Shared body of the module global-guard tests: capture one ModelInput
        # per value of a guarded global, then check that dispatch follows the
        # live value -- in the capturing process and, after a
        # torch._dynamo.reset() plus save/load, on the load path, which has to
        # resolve a guard scope of its own. Both halves run in this process, and
        # the scope both resolve to is this module's dict, which the capture
        # leaks Dynamo's minted globals into, so hide them here rather than leave
        # them for a sibling test to inherit. The load seeds nothing into it: no
        # alias is in the serialized scope and no guard source names the
        # builtins-dict key.
        self._hide_leaked_dynamo_globals()
        mod = make_mod()
        x = torch.randn(4, 8)
        expected = {}
        for mode in ("sum", "mean"):
            with set_mode(mode):
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
                ModelInput(args=(x,), kwargs={}, contexts=[set_mode(m)])
                for m in ("sum", "mean")
            ]
        )
        for mode in ("sum", "mean"):
            with set_mode(mode):
                self.assertEqual(model(x), expected[mode])

        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        reloaded = torch.compile(
            make_mod(),
            fullgraph=True,
            backend="inductor",
            options={"guard_filter_fn": keep_global_guards},
        )
        reloaded._load_aot_compiled_module(data)
        for mode in ("sum", "mean"):
            with set_mode(mode):
                self.assertEqual(reloaded(x), expected[mode])

    def test_aot_compile_module_dispatches_on_global_guard(self):
        # The default guard_filter_fn drops all global guards, so a caller who
        # needs one honored has to opt in. Keeping it only works because
        # AOTCompiledModel.deserialize supplies the traced function's globals.
        self._check_module_global_guard_dispatch(GlobalConfigModule, _set_pooling)

    def test_aot_compile_module_load_rederives_the_key_the_bytecode_reads(self):
        # AOTCompiledModel.deserialize resolves model.forward's globals -- this
        # module's dict -- as the guard scope and passes no f_globals, so on every
        # module load fn.__globals__ is a dict of forward_callable's own: the same
        # split as a caller-supplied scope, on the path every module artifact
        # takes. Returning `len` is what makes the generated bytecode read the key.
        self._hide_leaked_dynamo_globals()

        class ReturnsBuiltinModule(torch.nn.Module):
            def forward(self, x):
                if not isinstance(x, torch.Tensor):
                    raise TypeError(type(x))
                return x + 1, len

        x = torch.randn(3)
        options = {"guard_filter_fn": keep_builtin_guards}
        model = torch.compile(
            ReturnsBuiltinModule(), fullgraph=True, backend="eager", options=options
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        result = model.forward.compiled_results[0]
        guards_state = load_guards_state(result._artifacts.guards_state)
        builtins_key = guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
        self.assertIn(builtins_key, result._artifacts.runtime_env.bytecode.co_names)
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()

        reloaded = torch.compile(
            ReturnsBuiltinModule(), fullgraph=True, backend="eager", options=options
        )
        reloaded._load_aot_compiled_module(data)
        loaded = reloaded.forward.compiled_results[0]
        g = globals()
        self.assertTrue(loaded.fn.__globals__ is not g)
        self.assertTrue(
            g[builtins_key] is builtins.__dict__,
            "the load left the module's guard scope without the live builtins",
        )
        self.assertTrue(
            loaded.fn.__globals__[builtins_key] is builtins.__dict__,
            "the bytecode kept the recording on the module load path",
        )
        self.assertEqual(reloaded(x), ReturnsBuiltinModule()(x))

    @parametrize("shape", sorted(_UNLIFTED_GLOBAL_MODULES))
    def test_aot_compile_module_load_binds_a_certified_global_the_graph_never_lifted(
        self, shape
    ):
        # get_runtime_env records in used_globals only the globals the graph
        # lifted as inputs, while external_refs holds every name the bytecode
        # LOAD_GLOBALs; a global the forward mutates or returns is in the second
        # set and not the first. The module load passes no f_globals, so the
        # only thing that can satisfy forward_callable's external-refs check
        # for that name is the load binding it out of the live scope -- the
        # per-call re-take in _serve comes too late for a check made at load.
        self._hide_leaked_dynamo_globals()
        # The kept guards pin the list's length, so capture and call both see it
        # empty, whatever the other shape left in it.
        del AOT_GLOBAL_LOG[:]
        self.addCleanup(AOT_GLOBAL_LOG.clear)
        cls = _UNLIFTED_GLOBAL_MODULES[shape]
        opts = {"guard_filter_fn": keep_global_guards}
        x = torch.randn(4, 8)
        model = torch.compile(cls(), fullgraph=True, backend="eager", options=opts)
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        (captured,) = model.forward.compiled_results
        runtime_env = captured._artifacts.runtime_env
        self.assertIn("AOT_GLOBAL_LOG", runtime_env.external_refs)
        self.assertNotIn("AOT_GLOBAL_LOG", runtime_env.used_globals)
        self.assertNotIn("AOT_GLOBAL_LOG", runtime_env.import_sources)
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()

        reloaded = torch.compile(cls(), fullgraph=True, backend="eager", options=opts)
        reloaded._load_aot_compiled_module(data)
        (loaded,) = reloaded.forward.compiled_results
        self.assertIn("AOT_GLOBAL_LOG", loaded._live_global_names)
        self.assertIs(loaded.fn.__globals__["AOT_GLOBAL_LOG"], AOT_GLOBAL_LOG)
        del AOT_GLOBAL_LOG[:]
        out = reloaded(x)
        if shape == "mutates":
            self.assertEqual(out, x + 1)
            self.assertEqual(AOT_GLOBAL_LOG, [4])
        else:
            self.assertEqual(out[0], x + 1)
            self.assertIs(out[1], AOT_GLOBAL_LOG)

    @parametrize("shape", sorted(_UNLIFTED_GLOBAL_FUNCTIONS))
    def test_aot_compile_function_load_binds_a_certified_global_the_graph_never_lifted(
        self, shape
    ):
        # The function-path sibling of the module test above, and the one that
        # separates this commit from its parent: a guard_globals=-only
        # AOTCompiledFunction.deserialize used to bind the scope per call only,
        # so forward_callable's check ran before anything bound AOT_GLOBAL_LOG
        # and the load raised Missing required external references.
        self._hide_leaked_dynamo_globals()
        del AOT_GLOBAL_LOG[:]
        self.addCleanup(AOT_GLOBAL_LOG.clear)
        fn = _UNLIFTED_GLOBAL_FUNCTIONS[shape]
        opts = {"guard_filter_fn": keep_global_guards}
        x = torch.randn(4, 8)
        compiled = torch.compile(fn, fullgraph=True, backend="eager", options=opts)
        captured = compiled.aot_compile(((x,), {}))
        runtime_env = captured._artifacts.runtime_env
        self.assertIn("AOT_GLOBAL_LOG", runtime_env.external_refs)
        self.assertNotIn("AOT_GLOBAL_LOG", runtime_env.used_globals)
        self.assertNotIn("AOT_GLOBAL_LOG", runtime_env.import_sources)
        captured.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            data = f.read()
        torch._dynamo.reset()

        loaded = AOTCompiledFunction.deserialize(data, guard_globals=globals())
        self.assertIn("AOT_GLOBAL_LOG", loaded._live_global_names)
        self.assertIs(loaded.fn.__globals__["AOT_GLOBAL_LOG"], AOT_GLOBAL_LOG)
        del AOT_GLOBAL_LOG[:]
        out = loaded(x)
        if shape == "mutates":
            self.assertEqual(out, x + 1)
            self.assertEqual(AOT_GLOBAL_LOG, [4])
        else:
            self.assertEqual(out[0], x + 1)
            self.assertIs(out[1], AOT_GLOBAL_LOG)

    def test_aot_compile_module_no_match_error(self):
        # Two inputs, so the message has to account for both rather than
        # reporting only the first one's guard failure. Vary dtype rather than
        # shape: aot_compile_module does not forward `dynamic`, so a second
        # shape goes automatic-dynamic and would subsume the unmatched input.
        # No graph runs on this path, so eager keeps the report identical.
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
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
        # One line per input, not a multi-line GuardDebugInfo repr per input: the
        # two entries are the two lines after the header, each says something
        # after its index, and the advice that follows is not indented, so it
        # is not a continuation of the second.
        lines = message.splitlines()
        self.assertEqual([line[:6] for line in lines[1:3]], ["  [0] ", "  [1] "])
        for line in lines[1:3]:
            self.assertTrue(line[6:].strip(), f"entry says nothing: {line!r}")
        self.assertFalse(lines[3].startswith(" "), lines[3])
        self.assertIn("Add a ModelInput", message)

    def test_no_match_report_joins_every_verbose_part_of_an_entry(self):
        # A symbolic-shape refusal is the ordinary multi-part entry: the SHAPE_ENV
        # guard is one lambda over every relation the tree holds, and a failure
        # quotes all of its exprs, satisfied ones included, so the line has to
        # carry each of them joined with "; " for the reader to find the one
        # this call broke.
        class TwoDynamicModule(torch.nn.Module):
            def forward(self, x, y):
                return x[: y.size(0)] + y

        x, y = torch.randn(4, 3), torch.randn(2, 3)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 0)
        model = torch.compile(TwoDynamicModule(), fullgraph=True, backend="eager")
        model._aot_compile([ModelInput(args=(x, y), kwargs={}, contexts=[])])
        with self.assertRaises(RuntimeError) as ctx:
            model(torch.randn(2, 3), torch.randn(4, 3))
        lines = str(ctx.exception).splitlines()
        self.assertEqual(len(lines), 3, lines)
        self.assertTrue(lines[1].startswith("  [0] "), lines[1])
        # The relation this call broke and the two bounds it satisfies, on the
        # one entry line.
        self.assertIn("L['y'].size()[0] <= L['x'].size()[0]", lines[1])
        self.assertIn("2 <= L['y'].size()[0]", lines[1])
        self.assertIn("2 <= L['x'].size()[0]", lines[1])
        # At least the two separators the join adds; a part's own free text (the
        # 0/1-specialization note) carries a "; " of its own, so not exactly two.
        self.assertGreaterEqual(lines[1].count("; "), 2, lines[1])

    def test_no_match_report_keeps_an_entry_on_one_line_past_odd_separators(self):
        # A verbose code part embeds the guard's own source line, which the read
        # producing it ends only at \n, so every other separator splitlines()
        # breaks on -- a comment can carry the three below -- survives into the
        # report, where splitlines() reads the entry-per-line format back.
        src = (
            "import torch\n"
            "\n"
            "\n"
            "class OddSepModule(torch.nn.Module):\n"
            "    def forward(self, x, mode):\n"
            "        if mode == 1:  # page\x0cbreak\x1erec\u2028sep\n"
            "            return x * 2\n"
            "        return x * 3\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "aot_odd_sep_mod.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(src)
            sys.path.insert(0, tmp)
            self.addCleanup(sys.path.remove, tmp)
            self.addCleanup(sys.modules.pop, "aot_odd_sep_mod", None)
            mod = importlib.import_module("aot_odd_sep_mod")
            x = torch.randn(3, 3)
            model = torch.compile(mod.OddSepModule(), fullgraph=True, backend="eager")
            model._aot_compile([ModelInput(args=(x, 1), kwargs={}, contexts=[])])
            with self.assertRaises(RuntimeError) as ctx:
                model(x, 2)
        message = str(ctx.exception)
        lines = message.splitlines()
        self.assertEqual(len(lines), 3, message)
        self.assertTrue(lines[1].startswith("  [0] L['mode'] == 1"), lines[1])
        self.assertIn("Add a ModelInput", lines[2])
        # The source has no spaces there, so this says all three separators
        # arrived in the part and were collapsed -- without which the test is
        # vacuous, and the last two a form-feed replace would not cover.
        self.assertIn("page break rec sep", message)

    def test_no_match_report_keeps_a_raise_on_one_line_past_odd_separators(self):
        # The raise line is the other report entry whose text is arbitrary user
        # data, so it needs the collapse the verbose part above gets: a message
        # carrying any separator splitlines() breaks on would otherwise split one
        # entry across several lines, and the whole-line assertions and total
        # line counts the other tests read the report with cannot survive that.
        # test_aot_compile_module_two_raisers_in_the_report_and_then_the_warning
        # asserts a raise line whole past \x0c and \x1e already; what this test
        # reads that it does not is \r and \u2028 at that site -- a collapse
        # written as two replaces passes there and fails here -- and the line
        # inventory of a report that carries a raise.
        self._hide_leaked_dynamo_globals()
        model = self._model_with_stub_trees(
            RaisingTree("page\x0cbreak\rrec\x1esep\u2028line")
        )
        with self.assertRaises(RuntimeError) as ctx:
            model(torch.randn(3, 3))
        message = str(ctx.exception)
        lines = message.splitlines()
        # Header, entry, fix-or-drop advice: one raiser and no answer, so neither
        # the ModelInput advice nor the every-tree-raised footer. A split entry
        # adds lines here; a count of entries starting with "  [" would still
        # read 1, the first fragment being the only one that does.
        self.assertEqual(len(lines), 3, message)
        # The raised text holds no spaces, so the whole line says all four
        # separators arrived and were collapsed -- without which the test is
        # vacuous; none of the four is \n, so a \n replace would cover none of
        # them.
        raised = "  [0] <guard check raised RuntimeError: page break rec sep line>"
        self.assertEqual(lines[1], raised)

    def test_no_match_message_survives_a_raising_guard(self):
        # Dispatch recorded [0]'s raise as no answer, and the report quotes that
        # record for [0] and still describes [1]; at the parent the scan's raise
        # left __call__ with no report at all. A DICT_NOT_CONTAINS guard is a
        # real way in: RaisesOnCompare above says how its leaf raises.
        model, x = self._aot_compile_dict_branches({}, None)
        with self.assertRaises(RuntimeError) as ctx:
            model(x, {RaisesOnCompare(): 1})
        message = str(ctx.exception)
        self.assertIn("No AOT compiled graph matched this call", message)
        # The SystemError names only a bound method, so the line quotes its cause.
        raised = "  [0] <guard check raised ValueError: boom from __eq__ (through the guard tree's pybind boundary)>"
        self.assertIn(raised, message.splitlines())
        # [1] was traced with d=None, so its guards never look inside the evil dict.
        self.assertIn("[1] L['d'] is None", message)
        # Two independent things to do: fix or drop [0], cover [1] with an input.
        self.assertIn("[0]'s guard check raised while checking this call", message)
        self.assertIn("Add a ModelInput", message)
        # [1]'s rejection followed no raise, so the advice rests on a trusted
        # answer and carries no post-throw caveat.
        self.assertNotIn("every rejection this advice rests on", message)
        # One line per input, not a multi-line GuardDebugInfo repr per input.
        # Counted rather than read off the report's total, which an advice line
        # moves without changing what an entry looks like.
        self.assertEqual(sum(ln.startswith("  [") for ln in message.splitlines()), 2)

    def test_no_match_message_names_the_first_raise_recorded(self):
        # `raised` is in recording order, so the raiser the report names and the
        # raise it chains are the first RECORDED, not the lowest index: [0] was
        # traced with d=None and rejects the dict inline, then raises on the
        # second pass, after [1] raised through the leaf in the scan. Both lines
        # are raise lines, and the footer and the chain are [1]'s.
        model, x = self._aot_compile_dict_branches(None, {})
        manager = model.forward.compiled_results[0]._live_guard_manager()
        answers = [False, RuntimeError("zero boom")]
        with patch.object(manager, "check", side_effect=answers):
            with self.assertRaises(RuntimeError) as ctx:
                model(x, {RaisesOnCompare(): 1})
        message = str(ctx.exception)
        lines = message.splitlines()
        self.assertIn("  [0] <guard check raised RuntimeError: zero boom>", lines)
        raised = "  [1] <guard check raised ValueError: boom from __eq__ (through the guard tree's pybind boundary)>"
        self.assertIn(raised, lines)
        self.assertIn("[1]'s guard check raised while checking this call", message)
        self.assertIsInstance(ctx.exception.__cause__, SystemError)
        self.assertEqual(str(ctx.exception.__cause__.__cause__), "boom from __eq__")

    def test_no_match_message_quotes_a_raise_whose_str_raises(self):
        # The exception is the tree's down to its __str__, so quoting it is the
        # last place a raise can still take the report with it; what __str__
        # raises is caught as the handlers catch, so an interrupt out of it
        # propagates as itself, as one out of the tree does. Not a RuntimeError,
        # so a Boom that escapes the call fails assertRaises rather than the pin.
        class Boom(Exception):
            def __init__(self, exc):
                self.exc = exc

            def __str__(self):
                raise self.exc

        model, x = self._aot_compile_dict_branches({})
        manager = model.forward.compiled_results[0]._live_guard_manager()
        boom = Boom(ValueError("unread"))
        with patch.object(manager, "check", side_effect=boom):
            with self.assertRaises(RuntimeError) as ctx:
                model(x, {})
        raised = "  [0] <guard check raised Boom: <str() raised ValueError>>"
        self.assertIn(raised, str(ctx.exception).splitlines())
        self.assertIs(ctx.exception.__cause__, boom)
        for exc in (KeyboardInterrupt, SystemExit):
            interrupt = exc("inside __str__")
            with self.subTest(exc=exc.__name__):
                with patch.object(manager, "check", side_effect=Boom(interrupt)):
                    with self.assertRaises(exc) as ctx:
                        model(x, {})
                self.assertIs(ctx.exception, interrupt)

    def test_aot_compile_module_report_replaces_the_systemerror_a_caller_caught(self):
        # At the parent the boundary's SystemError left __call__ as itself, so a
        # caller who saw it and wrote `except SystemError` catches nothing now:
        # the call raises the report, that SystemError one hop down its chain
        # and the ValueError from __eq__ one further.
        model, x = self._aot_compile_dict_branches({})
        caught: BaseException | None = None
        try:
            model(x, {RaisesOnCompare(): 1})
        except SystemError as e:
            caught = e
        except RuntimeError as e:
            caught = e
        self.assertIsInstance(caught, RuntimeError)
        self.assertIn("No AOT compiled graph matched this call", str(caught))
        self.assertIsInstance(caught.__cause__, SystemError)
        self.assertEqual(str(caught.__cause__.__cause__), "boom from __eq__")

    def test_no_match_message_drops_the_input_advice_when_the_one_tree_raised(self):
        # Nothing here rejected the call, so the usual "add a ModelInput" advice
        # is wrong: an input covering this call would guard the same dict and
        # raise the same way. The one raiser's fix-or-drop line is the report's
        # last word: an every-tree-raised footer under it would say "[0] raised"
        # a third time.
        model = torch.compile(DictBranchModule(), fullgraph=True, backend="eager")
        x = torch.randn(3, 3)
        model._aot_compile([ModelInput(args=(x, {}), kwargs={}, contexts=[])])
        with self.assertRaises(RuntimeError) as ctx:
            model(x, {RaisesOnCompare(): 1})
        lines = str(ctx.exception).splitlines()
        self.assertEqual(sum(ln.startswith("  [") for ln in lines), 1, lines)
        advice = "[0]'s guard check raised while checking this call; fix or drop that artifact."
        self.assertEqual(lines[-1], advice)
        self.assertNotIn("Add a ModelInput", str(ctx.exception))
        self.assertNotIn("Every guard tree raised", str(ctx.exception))

    def test_no_match_message_when_every_guard_tree_raised(self):
        # Two enabled trees, both raising in dispatch, and no opt-out: the shape
        # that earns the footer. The fix-or-drop line names only the FIRST
        # raiser, so without the footer [1]'s line would be the only word on [1],
        # and nothing would say that no guard rejected this call.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        x = torch.randn(3, 3)
        inputs = [ModelInput(args=(x,), kwargs={}, contexts=[]) for _ in range(2)]
        model._aot_compile(inputs)

        class Raises(NeverReChecked):
            def __init__(self, message):
                self.message = message

            def check(self, f_locals):
                raise RuntimeError(self.message)

        results = model.forward.compiled_results
        results[0]._artifacts.guard_manager = Raises("the first tree is unhappy")
        results[1]._artifacts.guard_manager = Raises("the second tree is unhappy")
        with self.assertRaises(RuntimeError) as ctx:
            model(x)
        self.assertEqual(
            str(ctx.exception).splitlines(),
            [
                "No AOT compiled graph matched this call. Tried 2 compiled input(s):",
                "  [0] <guard check raised RuntimeError: the first tree is unhappy>",
                "  [1] <guard check raised RuntimeError: the second tree is unhappy>",
                "[0]'s guard check raised while checking this call; fix or drop that artifact.",
                "Every guard tree raised while checking this call; the reasons above are those raises, not guards this call failed.",
            ],
        )
        self.assertEqual(str(ctx.exception.__cause__), "the first tree is unhappy")

    def test_no_match_message_advises_an_input_when_one_of_two_raisers_answered(self):
        # Two raisers again, but [1] rejected on the second pass: a ModelInput
        # could cover that rejection, so the advice stays and the footer, which
        # would call the reasons above nothing but raises, stays off.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        x = torch.randn(3, 3)
        inputs = [ModelInput(args=(x,), kwargs={}, contexts=[]) for _ in range(2)]
        model._aot_compile(inputs)

        class Raises(NeverReChecked):
            def check(self, f_locals):
                raise RuntimeError("the first tree is unhappy")

        class RaisesThenRejects:
            def __init__(self):
                self.checks = 0

            def check(self, f_locals):
                self.checks += 1
                if self.checks == 1:
                    raise RuntimeError("the second tree's first pass is unhappy")
                return False

            def check_verbose(self, f_locals):
                return types.SimpleNamespace(
                    result=False, verbose_code_parts=["the second tree's guard"]
                )

        results = model.forward.compiled_results
        results[0]._artifacts.guard_manager = Raises()
        stub = RaisesThenRejects()
        results[1]._artifacts.guard_manager = stub
        with self.assertRaises(RuntimeError) as ctx:
            model(x)
        message = str(ctx.exception)
        lines = message.splitlines()
        self.assertEqual(stub.checks, 2)
        raised = "  [0] <guard check raised RuntimeError: the first tree is unhappy>"
        self.assertIn(raised, lines)
        self.assertIn("  [1] the second tree's guard", lines)
        self.assertIn("[0]'s guard check raised while checking this call", message)
        self.assertIn("Add a ModelInput", message)
        # [1]'s rejection followed its own raise, so the advice is qualified.
        self.assertIn("followed a raise from its own tree", message)
        self.assertNotIn("Every guard tree raised", message)

    def test_no_match_report_claims_no_raise_for_an_artifact_with_no_inputs(self):
        # deserialize() accepts the empty list aot_compile_module refuses: no
        # entry raised and no entry answered, and the one advice that fits is to
        # add an input.
        model = ScaleModule()
        compiled = AOTCompiledModel.deserialize(model, pickle.dumps([]))
        self.assertEqual(compiled.compiled_results, [])
        with self.assertRaises(RuntimeError) as ctx:
            compiled(torch.randn(3, 3))
        message = str(ctx.exception)
        self.assertIn("Tried 0 compiled input(s):", message)
        self.assertNotIn("Every guard tree raised", message)
        self.assertIn("Add a ModelInput", message)

    def test_no_match_message_when_only_the_report_raises(self):
        # The other of _raised_line's two call sites: the report's own handler
        # around check_verbose, not the branch that quotes what dispatch's
        # handler recorded. Here both dispatch passes got a clean answer out of
        # the tree -- the DICT_NOT_CONTAINS guard found the key and rejected the
        # call -- and only the re-check that explains why raised. That is an
        # ordinary mismatch missing its reason, so the advice a new ModelInput
        # would satisfy still applies and "every guard tree raised" would be the
        # opposite of what happened. It is the only report-side raise here that
        # crosses the pybind boundary, so the clause on this line is the report
        # caller's own unwrap.
        model, x = self._aot_compile_dict_branches({})
        key = HitsThenRaises(hits=2)
        with self.assertRaises(RuntimeError) as ctx:
            model(x, {key: 1})
        message = str(ctx.exception)
        # Compare 3 is the report asking rather than reusing a dispatch answer,
        # and only a raise from dispatch is chained onto the report, so a report
        # raise has to carry what it was raised from in the line itself. hits=2
        # rests on codegen probing the key once per evaluation: codegen that
        # probed it twice per check() would move the raise into dispatch pass 2
        # with this entry line reading the same, and only the last two
        # assertions tell that apart (measured with a wrapper that calls check()
        # twice).
        raised = "[0] <guard check raised ValueError: boom on compare 3 (through the guard tree's pybind boundary)>"
        self.assertIn(f"  {raised}", message.splitlines())
        self.assertNotIn("Every guard tree raised", message)
        self.assertIn("Add a ModelInput", message)
        # Nothing raised in dispatch, so nothing is chained and no artifact is
        # named for fixing: the two readings a dispatch raise would change.
        self.assertIsNone(ctx.exception.__cause__)
        self.assertNotIn("fix or drop that artifact", message)

    def test_no_match_message_when_a_raise_did_not_cross_the_pybind_boundary(self):
        # Not every raise arrives as a SystemError with the real exception
        # chained behind it: a TORCH_CHECK inside the tree surfaces as a plain
        # RuntimeError with nothing to unwrap, pybind having translated it there,
        # and this stub's raise never reaches pybind at all. Either way the line
        # reports the exception itself and leaves the boundary clause off.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )

        class RaisingGuardManager(NeverReChecked):
            def check(self, f_locals):
                raise RuntimeError("guard tree is unhappy")

        artifacts = model.forward.compiled_results[0]._artifacts
        artifacts.guard_manager = RaisingGuardManager()
        with self.assertRaises(RuntimeError) as ctx:
            model(torch.randn(3, 3))
        message = str(ctx.exception)
        raised = "[0] <guard check raised RuntimeError: guard tree is unhappy>"
        self.assertIn(f"  {raised}", message.splitlines())
        self.assertNotIn("pybind boundary", message)
        # One raiser, whose fix-or-drop line closes the report on its own.
        self.assertIn("[0]'s guard check raised while checking this call", message)
        self.assertNotIn("Every guard tree raised", message)

    def test_no_match_message_reads_the_cause_not_the_handled_exception(self):
        # The SystemError CPython raises for a tree that returned with an
        # exception set carries that exception as its __cause__:
        # _PyErr_FormatFromCause sets __cause__ and __context__ alike. Only
        # __cause__ is evidence of that boundary, though -- PEP 3134 sets
        # __context__ for ANY exception raised while another is being handled --
        # so reading __context__ quotes what the CALLER was handling and
        # decorates it as the guard's reason.
        self._hide_leaked_dynamo_globals()

        class Boom(Exception):
            pass

        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )

        class RaisingGuardManager(NeverReChecked):
            def check(self, f_locals):
                raise SystemError("stub tree is unhappy")

        artifacts = model.forward.compiled_results[0]._artifacts
        artifacts.guard_manager = RaisingGuardManager()
        try:
            raise Boom("the caller was handling this")
        except Boom:
            with self.assertRaises(RuntimeError) as ctx:
                model(torch.randn(3, 3))
        message = str(ctx.exception)
        raised = "[0] <guard check raised SystemError: stub tree is unhappy>"
        self.assertIn(f"  {raised}", message.splitlines())
        self.assertNotIn("the caller was handling this", message)
        self.assertNotIn("pybind boundary", message)

    def test_aot_compile_module_no_match_does_not_suppress_a_handled_exception(self):
        # An ordinary no-match, with no raise to chain, has to raise bare rather
        # than `from None`: `from None` sets __suppress_context__, which erases
        # the exception a call made inside an `except` block was handling from the
        # traceback the user reads.
        class Boom(Exception):
            pass

        self._hide_leaked_dynamo_globals()
        model = self._model_with_stub_trees(Rejects())
        try:
            raise Boom("the caller was handling this")
        except Boom:
            with self.assertRaises(RuntimeError) as ctx:
                model(torch.randn(3, 3))
        self.assertIn("No AOT compiled graph matched this call", str(ctx.exception))
        self.assertIs(ctx.exception.__suppress_context__, False)
        self.assertIsInstance(ctx.exception.__context__, Boom)

    def test_no_match_message_hints_only_the_entry_that_named_a_missing_global(self):
        # One entry names a missing global and the other is a plain mismatch, and
        # neither advice covers the other's entry: defining AOT_BRANCH_SCALE
        # cannot make mode=2 satisfy [1]'s L['mode'] == 1, and a new ModelInput
        # for mode=2 would not resolve the global [1] failed on. Both entries
        # keep the name they failed on, and the one hint is addressed to [1].
        model, x = self._aot_compile_mode_branches()
        g = globals()
        self.addCleanup(g.__setitem__, "AOT_BRANCH_SCALE", g["AOT_BRANCH_SCALE"])
        del g["AOT_BRANCH_SCALE"]
        with self.assertRaises(RuntimeError) as ctx:
            model(x, 2)
        message = str(ctx.exception)
        self.assertIn("[0] L['mode'] == 0", message)
        self.assertIn("[1] KeyError on G['AOT_BRANCH_SCALE']", message)
        self.assertIn("For [1]: a guarded global is missing", message)
        self.assertEqual(message.count("a guarded global is missing"), 1)
        self.assertIn("the module the compiled function was traced in", message)
        self.assertIn("Add a ModelInput", message)
        # The hint is [1]'s: neither the plain mismatch nor the advice carries it.
        lines = message.splitlines()
        self.assertEqual(lines[1][:6], "  [0] ")
        self.assertNotIn("a guarded global is missing", lines[1])
        self.assertNotIn("a guarded global is missing", lines[-1])
        self.assertTrue(lines[-1].startswith("Add a ModelInput"), lines[-1])

    def test_no_match_message_hints_each_scope_a_mixed_model_failed_in(self):
        # The public constructor takes results of differing scopes: here a
        # CAPTURED one, whose guards still read this module's globals, next to
        # a SUPPLIED one a load re-rooted at vars(this module). The same deleted
        # name fails both, but the advice differs -- define it in the module the
        # function was traced in, or in the scope the artifact was loaded
        # against -- so each entry gets its own For line rather than [1] being
        # read [0]'s.
        global GLOBAL_POOLING_CONFIG

        x = torch.randn(4, 8)
        loaded = AOTCompiledModel.deserialize(
            GlobalConfigModule(), self._two_input_global_guard_artifact(x)
        )
        captured = torch.compile(
            GlobalConfigModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        captured._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        results = captured.forward.compiled_results + loaded.compiled_results[:1]
        self.assertEqual(
            [r._guard_scope for r in results],
            [_GuardScope.CAPTURED, _GuardScope.SUPPLIED],
        )
        mixed = AOTCompiledModel(GlobalConfigModule(), results)
        saved = GLOBAL_POOLING_CONFIG
        self.addCleanup(globals().__setitem__, "GLOBAL_POOLING_CONFIG", saved)
        del GLOBAL_POOLING_CONFIG
        with self.assertRaises(RuntimeError) as ctx:
            mixed(x)
        message = str(ctx.exception)
        self.assertIn("[0] KeyError on G['GLOBAL_POOLING_CONFIG']", message)
        self.assertIn("[1] KeyError on G['GLOBAL_POOLING_CONFIG']", message)
        hints = [line for line in message.splitlines() if line.startswith("For [")]
        self.assertEqual([line[:9] for line in hints], ["For [0]: ", "For [1]: "])
        self.assertIn("the module the compiled function was traced in", hints[0])
        self.assertIn(f"here vars({__name__})", hints[1])

    def test_no_match_message_keeps_the_advice_beside_a_raise(self):
        # The independence pinned two tests up by
        # test_no_match_message_hints_only_the_entry_that_named_a_missing_global,
        # now with a raise in play: [0] raised and has to be
        # fixed or dropped, and [1] names a global missing from the module this
        # captured artifact's guards resolve against. Those two lines stand on
        # their own entries; the ModelInput line, appended to every report,
        # stands on none -- it is about the uncovered mode=2 call itself. Read at
        # fixed indices, which pins that the three are adjacent and last as well,
        # and reads a dropped advice line as a message diff.
        model, x = self._aot_compile_mode_branches()
        tree = RaisingTree("the scanned tree is unhappy")
        model.forward.compiled_results[0]._artifacts.guard_manager = tree
        g = globals()
        self.addCleanup(g.__setitem__, "AOT_BRANCH_SCALE", g["AOT_BRANCH_SCALE"])
        del g["AOT_BRANCH_SCALE"]
        with self.assertRaises(RuntimeError) as ctx:
            model(x, 2)
        message = str(ctx.exception)
        lines = message.splitlines()
        raised = "  [0] <guard check raised RuntimeError: the scanned tree is unhappy>"
        self.assertIn(raised, lines)
        self.assertIn("[1] KeyError on G['AOT_BRANCH_SCALE']", message)
        # Header, the two entries, then the hint, the fix-or-drop line and the
        # ModelInput line.
        self.assertEqual(len(lines), 6, message)
        self.assertIn("For [1]: a guarded global is missing", lines[3])
        self.assertIn("[0]'s guard check raised while checking this call", lines[4])
        self.assertIn("Add a ModelInput covering this call", lines[5])

    def _install_global_probe(self, name, misses):
        # Re-keys this module's global `name` under a CountedKey. Both cleanups
        # are registered before the dict is touched, so an interrupt anywhere
        # leaves the name restored. addCleanup is LIFO, so the probe is popped
        # first and the name re-inserted second; reversed, the re-insert would
        # find the probe by __eq__ and store under it, and the pop would then
        # drop the name for good. Call _hide_leaked_dynamo_globals before this,
        # never after: its sweep calls str methods on every key of this dict.
        g = globals()
        probe, saved = CountedKey(name, misses), g[name]
        self.addCleanup(g.__setitem__, name, saved)
        self.addCleanup(g.pop, probe, None)
        del g[name]
        g[probe] = saved
        return probe, saved

    def test_module_dispatch_evaluates_a_matching_tree_once(self):
        # The scan calls the matching result's _serve rather than the result,
        # whose __call__ would evaluate the guards that just passed a second time.
        mod = ScaleModule()
        model = torch.compile(mod, fullgraph=True, backend="eager")
        x = torch.randn(4, 8)
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        manager = model.forward.compiled_results[0]._artifacts.guard_manager
        with patch.object(manager, "check", wraps=manager.check) as check:
            out = model(x)
        self.assertEqual(out, mod(x))
        self.assertEqual(check.call_count, 1)

    def test_module_dispatch_binds_a_call_once_for_results_sharing_a_signature(self):
        # Every result aot_compile_module produces carries an equal signature and
        # the same closure cells, so one bind serves them all: a call that scans
        # every result, matched or not, binds once rather than once per result.
        mod = ScaleModule()
        model = torch.compile(mod, fullgraph=True, backend="eager")
        dtypes = (torch.float32, torch.float64, torch.int64, torch.bfloat16)
        xs = [torch.ones(3, 3, dtype=dtype) for dtype in dtypes]
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[]) for x in xs])
        results = model.forward.compiled_results
        self.assertEqual(len(results), 4)
        self.assertTrue(model.forward._binds_alike(tuple(results)))
        binds = []
        bind = AOTCompiledFunction.prepare_f_locals

        def counted(result, *args, **kwargs):
            binds.append(result)
            return bind(result, *args, **kwargs)

        counting = patch.object(AOTCompiledFunction, "prepare_f_locals", counted)
        with counting, patch.object(results[3], "fn", wraps=results[3].fn) as served:
            self.assertEqual(model(xs[3]), mod(xs[3]))
            self.assertEqual(binds, [results[0]])
            binds.clear()
            with self.assertRaises(RuntimeError) as ctx:
                model(torch.ones(3, 3, dtype=torch.float16))
        message = str(ctx.exception)
        served.assert_called_once()
        # One bind for the call nothing matched as well, counted apart from the
        # matched call's, and one entry per result off that bind, whatever else
        # the report carries.
        self.assertEqual(binds, [results[0]])
        lines = message.splitlines()
        self.assertEqual(sum(line.startswith("  [") for line in lines), len(xs))
        self.assertIn("Add a ModelInput", message)

    def test_module_dispatch_decides_a_shared_binding_past_the_first_result(self):
        # Whether results bind alike is asked only once the first result has
        # refused a call, so a call it serves pays nothing for the shared
        # binding, and a single-result model never asks, served or not.
        mod, xs = ScaleModule(), (torch.randn(3, 3), torch.randn(3, 3).double())
        results = [aot_compile_forward(mod, ScaleModule.forward, x) for x in xs]
        combined = AOTCompiledModel(mod, results)
        patched = patch.object(AOTCompiledModel, "_binds_alike", return_value=True)
        with patched as verdict:
            self.assertEqual(combined(xs[0]), xs[0] * 2)
            verdict.assert_not_called()
            self.assertEqual(combined(xs[1]), xs[1] * 2)
            verdict.assert_called_once()
        single = AOTCompiledModel(mod, results[:1])
        with patched as verdict:
            self.assertEqual(single(xs[0]), xs[0] * 2)
            with self.assertRaisesRegex(RuntimeError, "No AOT compiled graph matched"):
                single(xs[1])
            verdict.assert_not_called()

    def test_module_dispatch_binds_per_result_when_signatures_differ(self):
        # Results assembled by hand need not agree on a signature, and the guards
        # of each read the names ITS signature bound (L['y'] here), so such a
        # model binds each result on its own -- and still only once: both
        # dispatch passes and the report ask about the same call, so a rebind
        # per pass would leave the rest of this file green and pay for itself
        # on every no-match.
        x = torch.randn(3, 3)

        def triple(self, y):
            return y * 3

        combined = compile_two_signatures(x, triple, x.double())
        results = combined.compiled_results
        self.assertFalse(combined._binds_alike(tuple(results)))
        self.assertEqual(combined(x.double()), x.double() * 3)
        self.assertEqual(combined(x), x * 2)
        binds = []
        bind = AOTCompiledFunction.prepare_f_locals

        def counted(result, *args, **kwargs):
            binds.append(result)
            return bind(result, *args, **kwargs)

        managers = [result._artifacts.guard_manager for result in results]
        checks = [patch.object(m, "check", wraps=m.check) for m in managers]
        no_match = self.assertRaisesRegex(RuntimeError, "No AOT compiled graph matched")
        with patch.object(AOTCompiledFunction, "prepare_f_locals", counted):
            with checks[0] as check0, checks[1] as check1, no_match:
                combined(x.half())
        # Both passes checked both results off one binding per result, in index
        # order, the same objects.
        self.assertEqual((check0.call_count, check1.call_count), (2, 2))
        self.assertEqual(len(binds), len(results))
        for i, (bind_of, result) in enumerate(zip(binds, results)):
            self.assertIs(bind_of, result, f"bind {i} is not of result [{i}]")

    def test_module_dispatch_wrong_arity_raises_type_error(self):
        # A call the one signature cannot bind is a caller error, not a guard
        # miss: prepare_f_locals is outside accepts()' try, so the TypeError a
        # plain module call raises leaves as itself; folded into the no-match
        # report it would arrive as that report's RuntimeError instead.
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )
        x = torch.randn(3, 3)
        with self.assertRaisesRegex(TypeError, "too many positional arguments"):
            model(x, x)
        with self.assertRaisesRegex(TypeError, "missing a required argument: 'x'"):
            model()

    def test_module_dispatch_later_result_that_cannot_bind_raises_type_error(self):
        # prepare_f_locals is outside accepts()' try for every result, not only
        # the first: [1]'s signature cannot take the call, so its bind's TypeError
        # leaves as a plain module call's would, [0]'s raise on record unlogged.
        # Both premises are pinned first: [1] binds on its own, and [0] raises.
        # assertNoLogs outermost: nested inside the raise it would check nothing.
        x = torch.randn(3, 3)

        def needs_z(self, y, z):
            return y * z

        combined = compile_two_signatures(x, needs_z, x, x)
        results = combined.compiled_results
        results[0]._artifacts.guard_manager = AlwaysRaises("zero is unhappy")
        self.assertFalse(combined._binds_alike(tuple(results)))
        with self.assertRaisesRegex(RuntimeError, "zero is unhappy"):
            results[0].guard_check(combined.model, x)
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            with self.assertRaisesRegex(TypeError, "missing a required argument: 'z'"):
                combined(x)

    def test_module_dispatch_rebinds_after_a_result_is_appended(self):
        # compiled_results is a public list, so the one-bind-per-call decision is
        # made again when its contents change: a result appended with its own
        # default for `mode` is bound from that default, not from [0]'s.
        x = torch.randn(3, 3)
        mod, triples, doubles = compile_mode_defaults(x, x.double())
        combined = AOTCompiledModel(mod, [triples])
        self.assertTrue(combined._binds_alike(tuple(combined.compiled_results)))
        combined.compiled_results.append(doubles)
        # Without the re-decision this call would be bound from [0], where mode
        # reads 1, and [1]'s `mode == 0` guard would reject it.
        self.assertEqual(combined(x.double()), x.double() * 2)
        self.assertFalse(combined._binds_alike(tuple(combined.compiled_results)))

    def test_module_dispatch_judges_a_replaced_result_on_its_own_binding(self):
        # [1] is compiled for mode=1 but defaults mode to 0, so a call leaving
        # mode out has no graph: bound from [0], where mode defaults to 1, its
        # guards would pass and serve x * 3 for a call eager answers x * 2.
        x = torch.randn(3, 3)
        mod, triples, doubles = compile_mode_defaults(x, x.double(), 1)
        alike = aot_compile_forward(mod, make_mode_default_forward(1), x.double())
        combined = AOTCompiledModel(mod, [triples, alike])
        self.assertTrue(combined._binds_alike(tuple(combined.compiled_results)))
        combined.compiled_results[1] = doubles
        self.assertEqual(combined(x.double(), 1), x.double() * 3)
        with self.assertRaisesRegex(RuntimeError, "No AOT compiled graph matched"):
            combined(x.double())
        self.assertFalse(combined._binds_alike(tuple(combined.compiled_results)))

    def test_module_dispatch_does_not_keep_a_dropped_result_alive(self):
        # The verdict remembers the results it was reached over by weak
        # reference, so a result the caller drops from compiled_results is not
        # kept alive by the model until a later call decides again.
        mod, xs = ScaleModule(), (torch.randn(3, 3), torch.randn(3, 3).double())
        results = [aot_compile_forward(mod, ScaleModule.forward, x) for x in xs]
        combined = AOTCompiledModel(mod, results)
        self.assertEqual(combined(xs[1]), xs[1] * 2)
        self.assertTrue(combined._binds_alike(tuple(combined.compiled_results)))
        dropped = weakref.ref(combined.compiled_results.pop())
        del results
        gc.collect()
        self.assertIsNone(dropped())
        self.assertEqual(combined(xs[0]), xs[0] * 2)

    def test_module_dispatch_judges_only_the_results_a_call_began_with(self):
        # [0]'s check() appends [1], which matches the call. Both passes iterate
        # the results the call began with, so the re-check pairs each result with
        # the binding the scan made for it and never reaches a result the scan
        # did not bind; the appended result is judged by the next call.
        mod, x = ScaleModule(), torch.randn(3, 3)
        first = aot_compile_forward(mod, ScaleModule.forward, x)
        later = aot_compile_forward(mod, ScaleModule.forward, x.double())
        combined = AOTCompiledModel(mod, [first])
        manager = first._artifacts.guard_manager
        check = manager.check

        def appending_check(f_locals):
            combined.compiled_results[1:] = [later]
            return check(f_locals)

        with patch.object(manager, "check", appending_check):
            with self.assertRaisesRegex(RuntimeError, "No AOT compiled graph matched"):
                combined(x.double())
        self.assertIs(combined.compiled_results[1], later)
        self.assertEqual(combined(x.double()), x.double() * 2)

    def test_module_dispatch_never_pairs_new_contents_with_a_stale_verdict(self):
        # A call entering while another thread is still deciding over the
        # appended list must not find the new contents already published beside
        # the old True: the decider is held inside its first _binding_key, before
        # it publishes anything, and the call made in that window decides for
        # itself and binds [1] from its own default.
        x = torch.randn(3, 3)
        mod, triples, doubles = compile_mode_defaults(x, x.double())
        combined = AOTCompiledModel(mod, [triples])
        self.assertTrue(combined._binds_alike(tuple(combined.compiled_results)))
        combined.compiled_results.append(doubles)
        entered, release = threading.Event(), threading.Event()
        binding_key = torch._dynamo.aot_compile._binding_key

        def held(artifacts):
            if threading.current_thread() is decider:
                entered.set()
                release.wait(timeout=10)
            return binding_key(artifacts)

        decider = RecordedThread(
            target=combined._binds_alike, args=(tuple(combined.compiled_results),)
        )
        with patch("torch._dynamo.aot_compile._binding_key", held):
            decider.start()
            # LIFO, so a failure below releases the decider before joining it.
            self.addCleanup(decider.join)
            self.addCleanup(release.set)
            self.assertTrue(entered.wait(timeout=10))
            # A stale True here would bind from [0], where mode reads 1, and [1]'s
            # `mode == 0` guard would reject the call.
            self.assertEqual(combined(x.double()), x.double() * 2)
            release.set()
            decider.join()
        self.assertIsNone(decider.failure)
        self.assertFalse(combined._binds_alike(tuple(combined.compiled_results)))

    def test_module_dispatch_never_pairs_old_contents_with_a_new_verdict(self):
        # The mirror: this call snapshots [0] and [1] and, before it decides, [1]
        # is popped and another thread decides True over the one-result list. Held
        # right after its first attribute store, whatever that thread has
        # published so far must not turn this call into a shared bind. With the
        # verdict and the contents in two fields the verdict landed first, so the
        # call identity-matched the contents still published beside it, bound [1]
        # from [0]'s `mode=1` default and served x * 3 for a call eager answers
        # x * 2; judged on its own binding, [1]'s `mode == 1` guard rejects it.
        x = torch.randn(3, 3)
        mod, first, second = compile_mode_defaults(x, x.double(), 1)
        entered, release = threading.Event(), threading.Event()
        decider = RecordedThread(target=lambda: combined._binds_alike((first,)))

        class Held(AOTCompiledModel):
            def __setattr__(self, name, value):
                super().__setattr__(name, value)
                if threading.current_thread() is decider and not entered.is_set():
                    entered.set()
                    release.wait(timeout=10)

        combined = Held(mod, [first, second])
        self.assertFalse(combined._binds_alike(tuple(combined.compiled_results)))
        binds_alike = AOTCompiledModel._binds_alike

        def racing(model, results):
            if threading.current_thread() is decider:
                return binds_alike(model, results)
            self.assertIs(combined.compiled_results.pop(), second)
            decider.start()
            # LIFO, so a failure below releases the decider before joining it.
            self.addCleanup(decider.join)
            self.addCleanup(release.set)
            self.assertTrue(entered.wait(timeout=10))
            return binds_alike(model, results)

        with patch.object(AOTCompiledModel, "_binds_alike", racing):
            with self.assertRaisesRegex(RuntimeError, "No AOT compiled graph matched"):
                combined(x.double())
            # Only racing starts the decider: a dispatch that stopped consulting
            # _binds_alike would still raise above and leave the race untested.
            self.assertIsNotNone(decider.ident)
            release.set()
            decider.join()
        self.assertIsNone(decider.failure)
        self.assertTrue(combined._binds_alike(tuple(combined.compiled_results)))

    def _aot_compile_dict_branches(self, *ds):
        # One DictBranchModule result per d: {} traces the `"foo" not in d`
        # branch (x * 2), whose DICT_NOT_CONTAINS guard RaisesOnCompare raises
        # through; None traces `d is None` (x * 5), whose guards never look inside d.
        x = torch.ones(3, 3)
        model = torch.compile(DictBranchModule(), fullgraph=True, backend="eager")
        inputs = [ModelInput(args=(x, d), kwargs={}, contexts=[]) for d in ds]
        model._aot_compile(inputs)
        return model, x

    def _aot_compile_mode_branches(self):
        self._hide_leaked_dynamo_globals()
        mod = ModeBranchGlobalModule()
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        x = torch.randn(3, 3)
        model._aot_compile(
            [
                ModelInput(args=(x, 0), kwargs={}, contexts=[]),
                ModelInput(args=(x, 1), kwargs={}, contexts=[]),
            ]
        )
        return model, x

    def _rescued_by_the_recheck(self, model, x):
        # [1]'s guard on AOT_BRANCH_SCALE misses the global once and finds it on
        # the next lookup, so the scan rejects [1] and the re-check accepts it:
        # the call is served [1]'s graph at the cost of one more check() of its
        # tree.
        _, saved = self._install_global_probe("AOT_BRANCH_SCALE", misses=1)
        manager = model.forward.compiled_results[1]._artifacts.guard_manager
        with patch.object(manager, "check", wraps=manager.check) as check:
            out = model(x, 1)
        self.assertEqual(out, x * saved)
        self.assertEqual(check.call_count, 2)

    def test_module_dispatch_serves_a_call_the_guard_tree_accepts(self):
        # A first check() can reject a call the same tree accepts on its next
        # evaluation, which is what it does for real when the dict-tag fast path
        # answers false without running the tree. That rejection is not an
        # answer about the call, so a second pass has to rescue it. The
        # rescuable result is [1]; without the re-check the call would reach the
        # no-match report, where [0]'s L['mode'] == 0 is one line and [1]'s real
        # match is refused beside it.
        model, x = self._aot_compile_mode_branches()
        self._rescued_by_the_recheck(model, x)

    def test_module_dispatch_rechecks_an_opted_out_result_whose_tree_accepts(self):
        # The same false rejection of [1], with both results opted out. A
        # re-check that skipped opted-out results would leave the call to the
        # opt-out stage, which serves the FIRST opted-out result: [0], whose
        # L['mode'] == 0 guard genuinely fails this call.
        model, x = self._aot_compile_mode_branches()
        for result in model.forward.compiled_results:
            result.disable_guard_check()
        self._rescued_by_the_recheck(model, x)

    def test_module_dispatch_serves_an_opted_out_result_from_any_position(self):
        # With [0] still checked and [1] opted out, a call neither guards is
        # served by [1] rather than raising the no-match report, the stage after
        # this one. x * 3 is deliberately not eager's answer for mode=2 (x * 2):
        # serving a graph compiled for a different call is what the opt-out
        # exists to do.
        mod = ModeBranchGlobalModule()
        model = torch.compile(mod, fullgraph=True, backend="eager")
        x = torch.randn(3, 3)
        model._aot_compile(
            [
                ModelInput(args=(x, 0), kwargs={}, contexts=[]),
                ModelInput(args=(x, 1), kwargs={}, contexts=[]),
            ]
        )
        model.forward.compiled_results[1].disable_guard_check()
        self.assertEqual(model(x, 2), x * AOT_BRANCH_SCALE)
        # [0]'s real match still outranks [1]'s opt-out, so this stage cannot
        # be hoisted into the scan.
        self.assertEqual(model(x, 0), x * 2)
        # With both opted out and nothing matching, index order decides.
        model.forward.compiled_results[0].disable_guard_check()
        self.assertEqual(model(x, 2), x * 2)

    def test_module_dispatch_rechecks_before_honouring_an_opt_out(self):
        # [0] opted out, [1] checked and falsely rejected once: the re-check
        # finds [1]'s real match before the opt-out stage can hand the call to [0].
        model, x = self._aot_compile_mode_branches()
        model.forward.compiled_results[0].disable_guard_check()
        self._rescued_by_the_recheck(model, x)

    @parametrize("leading_opt_outs", [0, 1, 2])
    def test_module_dispatch_no_match_raises_unless_a_result_opted_out(
        self, leading_opt_outs
    ):
        # Nothing mocked: a call neither result guards raises the no-match
        # report, and is handed to no result; with [0] opted out its graph runs
        # instead, and opting [1] out as well changes nothing. The graphs differ
        # (x * 2 and x * 3), so the number says which result answered.
        model, x = self._aot_compile_mode_branches()
        for result in model.forward.compiled_results[:leading_opt_outs]:
            result.disable_guard_check()
        if leading_opt_outs:
            self.assertEqual(model(x, 2), x * 2)
        else:
            with self.assertRaisesRegex(RuntimeError, "No AOT compiled graph matched"):
                model(x, 2)

    def test_aot_compile_module_disable_guard_check(self):
        # disable_guard_check() is the escape hatch for an artifact whose guards
        # fail on the serving machine; module dispatch has to honor it too, or
        # a module artifact has no opt-out while a function artifact does.
        model = torch.compile(ScaleModule(), fullgraph=True, backend="aot_eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )
        # requires_grad is part of TENSOR_MATCH, so the artifact recorded for
        # requires_grad=False cannot match this call; opting out serves it
        # anyway, which is exactly the escape hatch -- and what it serves is an
        # inference graph that aot_autograd's runtime wrapper runs with grad
        # disabled, so the result carries no grad history. That is what makes the
        # aot_eager backend above load-bearing: an eager backend installs no such
        # wrapper and would still record the grad history.
        x = torch.randn(3, 3, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "requires_grad mismatch"):
            model(x)
        model.forward.compiled_results[0].disable_guard_check()
        out = model(x)
        self.assertEqual(out, x.detach() * 2)
        self.assertFalse(out.requires_grad)

    def test_aot_compile_module_opted_out_result_keeps_its_place_in_the_scan(self):
        self._hide_leaked_dynamo_globals()
        # Opting out changes what a result accepts, not where it sits: the scan
        # serves the first result whose check() accepts, in index order, and an
        # opted-out result whose guards genuinely pass is served like any other.
        # The default guard filter drops the global guard, so both results
        # accept this call while computing different numbers, and only a mix of
        # one opted-out and one checked result tells an index-ordered scan from
        # one that consults the checked results first or holds the opted-out
        # ones back for the passes below. The call is made under the mode [1]
        # was traced for, so the graph such a scan serves is the one the process
        # global would pick, and only the number says which result answered.
        mod = GlobalConfigModule()
        model = torch.compile(mod, fullgraph=True, backend="eager")
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
        results = model.forward.compiled_results
        with _set_pooling("mean"):
            for result in results:
                self.assertTrue(result.guard_check(mod, x))
            results[0].disable_guard_check()
            self.assertEqual(model(x), expected["sum"])

    def test_aot_compile_module_opted_out_result_is_the_last_resort(self):
        self._hide_leaked_dynamo_globals()
        # An opted-out result accepts anything, so dispatch reaches it only after
        # every real guard check has failed -- the scan, and then the second
        # check() pass above. Two more wrong-answer bugs with no error raised
        # hide in the rest of that ordering. Consulting the opt-out in the scan
        # serves the first artifact for a call the second was compiled for --
        # same shapes, same dtypes, different numbers. And the last resort is
        # itself ordered: with every result opted out it serves the FIRST of
        # them, so a call no artifact guards gets the graph the earliest
        # opted-out ModelInput was traced for. A scan that skipped opted-out
        # results is not pinned here, for two different reasons: in the first
        # case the result that must serve is the CHECKED one, which the skip
        # never applies to, so [1] is served by the scan itself; in the second
        # both trees fail on "other" anyway and the answer this ordering demands
        # is the first opted-out result, which the last resort serves either
        # way. What tells a skipping scan apart is an opted-out result whose
        # tree the scan would accept:
        # test_aot_compile_module_opted_out_result_keeps_its_place_in_the_scan
        # above pins that with honest guards, and
        # test_module_dispatch_rechecks_an_opted_out_result_whose_tree_accepts
        # covers the false-rejection variant.
        mod = GlobalConfigModule()
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="eager",
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
        results = model.forward.compiled_results
        results[0].disable_guard_check()
        with _set_pooling("mean"):
            self.assertEqual(
                model(x),
                expected["mean"],
                msg="the scan must outrank the opted-out result",
            )
        # Two opted-out results, so first and last are different results below.
        results[1].disable_guard_check()
        with _set_pooling("other"):
            self.assertEqual(
                model(x),
                expected["sum"],
                msg="with both opted out, the last resort must serve the first",
            )

    def test_module_dispatch_shares_a_binding_past_a_tensor_default(self):
        # Every result's signature carries the same default object, so the
        # results share a binding; deciding that through Signature equality
        # raised `Boolean value of Tensor with more than one value is ambiguous`
        # on the first call, for _aot_compile and deserialize alike.
        mod = torch.nn.Module()
        mod.forward = types.MethodType(make_masked_forward(), mod)
        model = torch.compile(mod, fullgraph=True, backend="eager")
        xs = [torch.randn(3, 3), torch.randn(3, 3, dtype=torch.float64)]
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[]) for x in xs])
        compiled = model.forward
        self.assertTrue(compiled._binds_alike(tuple(compiled.compiled_results)))
        for x in xs:
            self.assertEqual(model(x), x * 2)
        # Each result unpickles on its own, so the loaded results hold distinct
        # default objects and bind per result: a false negative, not a false
        # share, pinned as the accepted limitation; a load that shared one memo
        # across the blobs would flip it to True and is free to.
        loaded = AOTCompiledModel.deserialize(mod, model.forward.serialize())
        self.assertFalse(loaded._binds_alike(tuple(loaded.compiled_results)))
        for x in xs:
            self.assertEqual(loaded(x), x * 2)

    def test_module_dispatch_shares_a_binding_across_closure_cells(self):
        # A forward closing over a cell shares a binding only while every
        # result holds the SAME cell, which results of one _aot_compile do and
        # results assembled from two modules -- same signature, same freevar
        # name, different cell -- do not.
        mod = torch.nn.Module()
        mod.forward = types.MethodType(make_scaling_forward(3.0), mod)
        model = torch.compile(mod, fullgraph=True, backend="eager")
        xs = [torch.randn(3, 3), torch.randn(3, 3, dtype=torch.float64)]
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[]) for x in xs])
        compiled = model.forward
        self.assertTrue(compiled._binds_alike(tuple(compiled.compiled_results)))
        for x in xs:
            self.assertEqual(model(x), x * 3)
        other = torch.nn.Module()
        other.forward = types.MethodType(make_scaling_forward(5.0), other)
        model2 = torch.compile(other, fullgraph=True, backend="eager")
        model2._aot_compile([ModelInput(args=(xs[1],), kwargs={}, contexts=[])])
        results = model.forward.compiled_results[:1] + model2.forward.compiled_results
        combined = AOTCompiledModel(mod, results)
        self.assertFalse(combined._binds_alike(tuple(combined.compiled_results)))
        self.assertEqual(combined(xs[0]), xs[0] * 3)
        self.assertEqual(combined(xs[1]), xs[1] * 5)
        # Loading gives each result a cell of its own, so a round trip binds per
        # result too: the same accepted limitation, equally free to flip.
        loaded = AOTCompiledModel.deserialize(mod, model.forward.serialize())
        self.assertFalse(loaded._binds_alike(tuple(loaded.compiled_results)))
        for x in xs:
            self.assertEqual(loaded(x), x * 3)

    def test_no_match_message_when_a_guard_answers_inconsistently(self):
        # Both dispatch passes ran [1]'s whole tree and both rejected the call,
        # so an accept while the report asks why contradicts them rather than
        # correcting them. The guards the tree just passed cannot be quoted as
        # the reason the call was refused, so GuardDebugInfo.result decides what
        # the entry says.
        model, x = self._aot_compile_mode_branches()
        probe, _ = self._install_global_probe("AOT_BRANCH_SCALE", misses=2)
        with self.assertRaises(RuntimeError) as ctx:
            model(x, 1)
        message = str(ctx.exception)
        # Two rejections in dispatch, then the report's accept: one lookup per
        # evaluation, since one DictGetItemGuardAccessor is rooted at the global.
        # That accessor skips the lookup on a check() whose manager's dict tag
        # matches (matches_dict_tag in guards.cpp, under any config); the probe's
        # pop/insert bumped this module dict's version past the tag the manager
        # holds, so no evaluation skips it.
        self.assertEqual(probe.compares, 3)
        self.assertIn(f"  [1] {_ACCEPTED_IN_THE_REPORT}", message)
        # One entry line per result: the explanation is all [1] contributes, so
        # no blank "  [1] " line follows it from an accept's empty verbose parts.
        entries = [line for line in message.splitlines() if line.startswith("  [")]
        self.assertEqual(len(entries), 2)
        # The footer is the report's, not [0]'s: a ModelInput captured for this
        # call gets a tree that matches it on the first evaluation, so it is a
        # mitigation for [1]'s entry as well as for [0]'s real mismatch.
        self.assertIn("[0] L['mode'] == 0", message)
        self.assertIn("Add a ModelInput", message)

    def test_no_match_message_qualifies_the_advice_when_the_re_check_raises(self):
        # The re-check raised where dispatch saw a rejection: the only entry line
        # is the re-check's raise, and the ModelInput advice is still qualified.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )

        class RaisesThenRejectsThenRaises(RaisesOnceThenRejects):
            def check_verbose(self, f_locals):
                raise RuntimeError("the re-check is unhappy")

        stub = RaisesThenRejectsThenRaises("the first pass is unhappy")
        model.forward.compiled_results[0]._artifacts.guard_manager = stub
        with self.assertRaises(RuntimeError) as ctx:
            served = model(torch.randn(3, 3))
            self.fail(f"dispatch served {served[0, 0].item()}, not a raise")
        message = str(ctx.exception)
        lines = message.splitlines()
        self.assertEqual(stub.checks, 2)
        self.assertEqual(sum(ln.startswith("  [") for ln in lines), 1, message)
        raised = "  [0] <guard check raised RuntimeError: the re-check is unhappy>"
        self.assertIn(raised, lines)
        self.assertIn("[0]'s guard check raised while checking this call", message)
        self.assertIn("Add a ModelInput", message)
        self.assertIn("every rejection this advice rests on", message)
        self.assertNotIn("Every guard tree raised", message)
        self.assertEqual(str(ctx.exception.__cause__), "the first pass is unhappy")

    def test_no_match_message_when_guarded_state_changes_before_the_report(self):
        # The entry's other cause, with every guard answering consistently:
        # guards read live state on each evaluation and __call__ holds no lock
        # across the dispatch, so a global that is wrong for both passes and
        # right again by the time the report evaluates the tree gets the same
        # accept. The rebind is placed in that window from the report's own
        # check_verbose, where a concurrent writer could land it.
        model, x = self._aot_compile_mode_branches()
        g = globals()
        saved = g["AOT_BRANCH_SCALE"]
        self.addCleanup(g.__setitem__, "AOT_BRANCH_SCALE", saved)
        g["AOT_BRANCH_SCALE"] = saved + 1
        manager = model.forward.compiled_results[1]._artifacts.guard_manager
        check_verbose = manager.check_verbose

        def rebind_then_check(f_locals):
            g["AOT_BRANCH_SCALE"] = saved
            return check_verbose(f_locals)

        with (
            patch.object(manager, "check", wraps=manager.check) as check,
            patch.object(manager, "check_verbose", side_effect=rebind_then_check),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                model(x, 1)
        message = str(ctx.exception)
        # Both passes ran the tree against the wrong value and refused; the
        # accept is the report's alone, and the call is refused all the same.
        self.assertEqual(check.call_count, 2)
        self.assertIn(f"  [1] {_ACCEPTED_IN_THE_REPORT}", message)
        entries = [line for line in message.splitlines() if line.startswith("  [")]
        self.assertEqual(len(entries), 2)
        self.assertIn("[0] L['mode'] == 0", message)

    def test_no_match_message_hint_covers_a_rebound_forward(self):
        # The load resolves the guard scope from model.forward, the INSTANCE
        # attribute, so the dict the guards read is the globals of the function
        # that attribute resolves to -- here a function over foreign globals,
        # which is not where resolving forward on the class lands. The hint
        # names that instance attribute, not the class's forward, which would
        # send this reader to a dict where defining the name does not restore
        # dispatch -- which is what the two assertions at the end measure.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)
        model = torch.compile(
            HermeticModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()

        torch._dynamo.reset()
        g = globals()
        saved = g.pop("AOT_HERMETIC_WEIGHT")
        try:
            # Same code object, foreign globals: get_traced_fn resolves this
            # function, so the scope the load re-roots the guards at is ns.
            ns: dict[str, object] = {"__builtins__": builtins}
            rebound = types.FunctionType(
                HermeticModule.forward.__code__, ns, "rebound_forward"
            )
            inst = HermeticModule()
            inst.forward = rebound.__get__(inst, HermeticModule)
            reloaded = torch.compile(inst, fullgraph=True, backend="eager")
            reloaded._load_aot_compiled_module(data)
            with self.assertRaises(RuntimeError) as ctx:
                reloaded(x)
            message = str(ctx.exception)
            self.assertIn("[0] KeyError on G['AOT_HERMETIC_WEIGHT']", message)
            self.assertIn(
                "the function this HermeticModule instance's forward resolves to",
                message,
            )
            # HermeticModule.forward, the class attribute, is defined in this
            # module; a hint naming it would point here.
            g["AOT_HERMETIC_WEIGHT"] = saved
            missing = r"KeyError on G\['AOT_HERMETIC_WEIGHT'\]"
            with self.assertRaisesRegex(RuntimeError, missing):
                reloaded(x)
            # The dict the load actually resolved: the guards hold it by
            # reference, so defining the name here is what lets them resolve and
            # the call be served. Bound to the serialized tensor itself, so the
            # product pins which dict restores dispatch, not which one the graph
            # read; test_aot_compile_module_reload_reads_the_live_global pins that.
            ns["AOT_HERMETIC_WEIGHT"] = saved
            self.assertEqual(reloaded(x), x @ saved)
        finally:
            g["AOT_HERMETIC_WEIGHT"] = saved

    @parametrize("wrapper", ("compile", "disable", "disable_nonrecursive"))
    def test_no_match_message_hint_sees_through_a_wrapped_rebound_forward(
        self, wrapper
    ):
        # torch.compile(mod.forward) and both torch._dynamo.disable forms rebind
        # forward to a wrapper in eval_frame's or external_utils' namespace; the
        # load resolves THROUGH it, so the hint must send the reader through too:
        # defining the name in the wrapper's namespace restores nothing.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)
        model = torch.compile(
            HermeticModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()

        torch._dynamo.reset()
        g = globals()
        saved = g.pop("AOT_HERMETIC_WEIGHT")
        try:
            inst = HermeticModule()
            if wrapper == "compile":
                inst.forward = torch.compile(inst.forward, backend="eager")
            else:
                recursive = wrapper == "disable"
                inst.forward = torch._dynamo.disable(inst.forward, recursive=recursive)
            wrapper_globals = inst.forward.__globals__
            self.assertIsNot(wrapper_globals, g)
            reloaded = torch.compile(inst, fullgraph=True, backend="eager")
            reloaded._load_aot_compiled_module(data)
            with self.assertRaises(RuntimeError) as ctx:
                reloaded(x)
            message = str(ctx.exception)
            self.assertIn("[0] KeyError on G['AOT_HERMETIC_WEIGHT']", message)
            self.assertIn(
                "this HermeticModule instance's forward resolves to, seen through the "
                "wrappers torch.compile, torch._dynamo.disable, run and optimize "
                "return and through any functools.wraps'd torch._dynamo.external_utils "
                "function to the function they wrap, which is the dict the guards "
                "hold",
                message,
            )
            wrapper_globals["AOT_HERMETIC_WEIGHT"] = saved
            try:
                with self.assertRaisesRegex(
                    RuntimeError, r"KeyError on G\['AOT_HERMETIC_WEIGHT'\]"
                ):
                    reloaded(x)
            finally:
                del wrapper_globals["AOT_HERMETIC_WEIGHT"]
            g["AOT_HERMETIC_WEIGHT"] = saved
            self.assertEqual(reloaded(x), x @ saved)
        finally:
            g["AOT_HERMETIC_WEIGHT"] = saved

    def test_no_match_message_hint_stays_neutral_for_a_supplied_scope(self):
        # deserialize skips _resolve_guard_scope when the caller passes
        # guard_globals=, so the guards hold THAT dict rather than the globals of
        # the function model.forward resolves to. Here that dict never had the
        # name (names_the_resolved_module reaches the wording through a stale
        # copy of this module's dict), and the assertions after the wording
        # measure the advice: the name defined in the supplied dict, where the
        # hint sends the reader, is what serves the call.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)
        model = torch.compile(
            HermeticModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()

        torch._dynamo.reset()
        scope: dict[str, object] = {"__builtins__": builtins}
        compiled = AOTCompiledModel.deserialize(
            HermeticModule(), data, guard_globals=scope
        )
        with self.assertRaises(RuntimeError) as ctx:
            compiled(x)
        message = str(ctx.exception)
        self.assertIn("[0] KeyError on G['AOT_HERMETIC_WEIGHT']", message)
        self.assertIn(
            "For [0]: a guarded global is missing from the live scope this artifact "
            "was loaded against; define it there",
            message,
        )
        self.assertNotIn("instance's forward", message)
        # The guards hold the supplied dict, not this module's, where the name
        # read below resolves and the class's forward is defined.
        self.assertIs(compiled.compiled_results[0]._guard_globals, scope)
        scope["AOT_HERMETIC_WEIGHT"] = AOT_HERMETIC_WEIGHT
        self.assertEqual(compiled(x), x @ AOT_HERMETIC_WEIGHT)

    def test_no_match_message_names_forward_only_for_the_dict_it_resolves_to(self):
        # Two SUPPLIED results whose guards hold different dicts: one a load
        # resolved from model.forward, so it is this module's, and one the
        # caller supplied, which no forward resolves to. The report resolves
        # forward once and compares that one dict against each entry's, so only
        # the entry holding it is told which forward -- whichever entry is first.
        global GLOBAL_POOLING_CONFIG

        x = torch.randn(4, 8)
        data = self._two_input_global_guard_artifact(x)
        by_forward = AOTCompiledModel.deserialize(GlobalConfigModule(), data)
        scope: dict[str, object] = {"__builtins__": builtins}
        by_caller = AOTCompiledModel.deserialize(
            GlobalConfigModule(), data, guard_globals=scope
        )
        self.assertIs(by_forward.compiled_results[0]._guard_globals, globals())
        self.assertIs(by_caller.compiled_results[0]._guard_globals, scope)
        saved = GLOBAL_POOLING_CONFIG
        self.addCleanup(globals().__setitem__, "GLOBAL_POOLING_CONFIG", saved)
        del GLOBAL_POOLING_CONFIG
        named = "this GlobalConfigModule instance's forward resolves to"
        neutral = "the live scope this artifact was loaded against; define it there"
        pair = by_forward.compiled_results[:1] + by_caller.compiled_results[:1]
        target = "torch._dynamo.aot_compile._resolve_guard_scope"
        cases = ((pair, [named, neutral]), (pair[::-1], [neutral, named]))
        for results, wording in cases:
            mixed = AOTCompiledModel(GlobalConfigModule(), results)
            resolve = patch(target, wraps=_resolve_guard_scope)
            with resolve as resolves, self.assertRaises(RuntimeError) as ctx:
                mixed(x)
            resolves.assert_called_once_with(mixed.model)
            message = str(ctx.exception)
            self.assertIn("[0] KeyError on G['GLOBAL_POOLING_CONFIG']", message)
            self.assertIn("[1] KeyError on G['GLOBAL_POOLING_CONFIG']", message)
            hints = [line for line in message.splitlines() if line.startswith("For [")]
            prefixes = [line[:9] for line in hints]
            self.assertEqual(prefixes, ["For [0]: ", "For [1]: "], message)
            for hint, want in zip(hints, wording):
                self.assertIn(want, hint)

    def test_no_match_message_shares_a_hint_line_across_entries_worded_alike(self):
        # Three SUPPLIED results, each holding a caller dict of its own: [0] and
        # [2] lack the global, [1] has it but not the key its guard reads. Hint
        # lines are keyed by sentence, not by dict, so [0] and [2] -- two dicts
        # with one wording, neither carrying a __name__ -- share one
        # non-contiguous "For [0, 2]:" line, and [1]'s "KeyError on
        # G['GLOBAL_POOLING_CONFIG']['pooling']" is the ordinary mismatch the
        # whole-part match keeps off it, as the function path pins for the
        # predicate itself. The model's forward is a partial no load resolves,
        # so the report's one resolve returns (None, reason) rather than
        # raising, and no entry is told which forward.
        x = torch.randn(4, 8)
        data = self._two_input_global_guard_artifact(x)
        absent: dict[str, object] = {"__builtins__": builtins}
        keyless: dict[str, object] = {**absent, "GLOBAL_POOLING_CONFIG": {}}
        loads = [
            AOTCompiledModel.deserialize(GlobalConfigModule(), data, guard_globals=s)
            for s in (absent, keyless, dict(absent))
        ]
        results = [loaded.compiled_results[0] for loaded in loads]
        self.assertEqual([r._guard_scope for r in results], [_GuardScope.SUPPLIED] * 3)
        self.assertEqual(len({id(r._guard_globals) for r in results}), 3)
        mixed = AOTCompiledModel(self._unresolvable_forward_module(), results)
        target = "torch._dynamo.aot_compile._resolve_guard_scope"
        with patch(target, wraps=_resolve_guard_scope) as resolves:
            with self.assertRaises(RuntimeError) as ctx:
                mixed(x)
        resolves.assert_called_once_with(mixed.model)
        lines = str(ctx.exception).splitlines()
        self.assertIn("Tried 3 compiled input(s)", lines[0])
        self.assertEqual(
            lines[1:4],
            [
                "  [0] KeyError on G['GLOBAL_POOLING_CONFIG']",
                "  [1] KeyError on G['GLOBAL_POOLING_CONFIG']['pooling']",
                "  [2] KeyError on G['GLOBAL_POOLING_CONFIG']",
            ],
        )
        hints = [line for line in lines if line.startswith("For [")]
        self.assertEqual(len(hints), 1, lines)
        self.assertTrue(
            hints[0].startswith(
                "For [0, 2]: a guarded global is missing from the live scope this "
                "artifact was loaded against; define it there"
            ),
            hints[0],
        )
        self.assertNotIn("instance's forward", hints[0])
        self.assertIn("Add a ModelInput", lines[-1])

    def test_no_match_report_resolves_forward_only_for_a_supplied_scope(self):
        # An in-process capture keeps the CAPTURED scope, whose hint never names
        # forward, so the report has no reason to resolve it -- and resolving it
        # runs user code (get_traced_fn formats a forward it refuses). Since the
        # CAPTURED wording reads the same whether or not the resolve ran, the
        # gate is pinned by counting resolves rather than by the wording.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(
            HermeticModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        x = torch.randn(3, 3)
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        resolve = patch(
            "torch._dynamo.aot_compile._resolve_guard_scope", wraps=_resolve_guard_scope
        )
        g = globals()
        saved = g.pop("AOT_HERMETIC_WEIGHT")
        try:
            with resolve as resolved, self.assertRaises(RuntimeError) as ctx:
                model(x)
            message = str(ctx.exception)
        finally:
            g["AOT_HERMETIC_WEIGHT"] = saved
        resolved.assert_not_called()
        self.assertIn("[0] KeyError on G['AOT_HERMETIC_WEIGHT']", message)
        self.assertIn("the module the compiled function was traced in", message)
        self.assertNotIn("instance's forward", message)
        self.assertIn("Add a ModelInput", message)

    def test_no_match_report_resolves_forward_once_past_a_resolve_that_raises(self):
        # Two SUPPLIED results loaded without guard_globals=, so both hold this
        # module's dict, on a RaisingReprModule whose forward is then rebound to
        # a partial: get_traced_fn formats the forward it refuses into its
        # error, and that repr raises the module's ValueError past what
        # _resolve_guard_scope catches. The report marks the attempt before it
        # tries, so [1] does not re-run that user code for a second raise. Both
        # entries read neutral whether it re-ran or not, so the count pins it.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)
        model = torch.compile(
            RaisingReprModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()

        torch._dynamo.reset()
        inst = RaisingReprModule()
        loads = [AOTCompiledModel.deserialize(inst, data) for _ in range(2)]
        results = [loaded.compiled_results[0] for loaded in loads]
        self.assertEqual([r._guard_scope for r in results], [_GuardScope.SUPPLIED] * 2)
        mixed = AOTCompiledModel(inst, results)
        inst.forward = functools.partial(HermeticModule.forward, inst)
        with self.assertRaises(ValueError):
            _resolve_guard_scope(inst)
        resolve = patch(
            "torch._dynamo.aot_compile._resolve_guard_scope", wraps=_resolve_guard_scope
        )
        g = globals()
        saved = g.pop("AOT_HERMETIC_WEIGHT")
        try:
            with resolve as resolves, self.assertRaises(RuntimeError) as ctx:
                mixed(x)
            message = str(ctx.exception)
        finally:
            g["AOT_HERMETIC_WEIGHT"] = saved
        # Not assert_called_once_with: a failure would repr inst, which raises.
        self.assertEqual(resolves.call_count, 1)
        self.assertIs(resolves.call_args.args[0], inst)
        self.assertIn("[0] KeyError on G['AOT_HERMETIC_WEIGHT']", message)
        self.assertIn("[1] KeyError on G['AOT_HERMETIC_WEIGHT']", message)
        self.assertIn(
            "For [0, 1]: a guarded global is missing from the live scope this "
            f"artifact was loaded against, here vars({__name__}); define it there",
            message,
        )
        self.assertNotIn("instance's forward", message)

    def test_no_match_report_survives_a_forward_resolve_that_raises(self):
        # deserialize without guard_globals= resolves the scope itself, so every
        # result is SUPPLIED and the report does re-resolve forward. A rebind
        # after that load is read only here, and this one raises the user's
        # ValueError past what _resolve_guard_scope catches: the report has to
        # arrive anyway, worded for a scope no forward resolves to.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)
        model = torch.compile(
            RaisingReprModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()

        torch._dynamo.reset()
        inst = RaisingReprModule()
        compiled = AOTCompiledModel.deserialize(inst, data)
        self.assertIs(compiled.compiled_results[0]._guard_scope, _GuardScope.SUPPLIED)
        inst.forward = functools.partial(HermeticModule.forward, inst)
        with self.assertRaises(ValueError):
            repr(inst.forward)
        g = globals()
        saved = g.pop("AOT_HERMETIC_WEIGHT")
        try:
            with (
                self.assertLogs("torch._dynamo.aot_compile", level="DEBUG") as logs,
                self.assertRaises(RuntimeError) as ctx,
            ):
                compiled(x)
            message = str(ctx.exception)
        finally:
            g["AOT_HERMETIC_WEIGHT"] = saved
        self.assertEqual(len(logs.output), 1)
        self.assertRegex(logs.output[0], r"RaisingReprModule\.forward: .* ValueError")
        self.assertIn("[0] KeyError on G['AOT_HERMETIC_WEIGHT']", message)
        self.assertIn(
            "For [0]: a guarded global is missing from the live scope this "
            "artifact was loaded against",
            message,
        )
        self.assertNotIn("instance's forward", message)
        self.assertIn("Add a ModelInput", message)

    def test_no_match_message_when_a_failure_names_no_guard(self):
        # A set index past the end of a shorter set answers
        # GuardDebugInfo(false, 0): the call did NOT match, yet there is no
        # verbose code part to quote, so an empty list cannot stand for "this tree
        # passed" -- reason.result is what says that. Read as a pass, this call
        # gets served a graph compiled for a set of another size.
        model = torch.compile(
            SetIterModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": drop_sequence_length_guards},
        )
        x = torch.randn(3, 3)
        one, two = object(), object()
        model._aot_compile([ModelInput(args=(x, {one, two}), kwargs={}, contexts=[])])
        with self.assertRaises(RuntimeError) as ctx:
            model(x, {one})
        message = str(ctx.exception)
        self.assertIn("[0] <guard check failed without naming a guard>", message)
        self.assertIn("Add a ModelInput", message)

    def test_no_match_message_when_a_failure_quotes_a_blank_message(self):
        # A guard that raises answers false with str(exc) as its ONE verbose code
        # part (GuardDebugInfo(false, get_exception_message(), 0) in guards.cpp),
        # so [""] is a non-empty part list whose entry still says nothing; a
        # whitespace message collapses to a blank tail the same way. The subclass
        # metadata guard is user code on that path and the default guard filter
        # keeps it.
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        x = torch.randn(3, 3)
        t = FlakyTwoTensor(x, x.clone())
        model._aot_compile([ModelInput(args=(t,), kwargs={}, contexts=[])])
        self.addCleanup(setattr, FlakyTwoTensor, "metadata_error", None)
        want = ["  [0] <guard check failed without naming a guard>"]
        for text in ("", " "):
            with self.subTest(text=text):
                FlakyTwoTensor.metadata_error = ValueError(text)
                with self.assertRaises(RuntimeError) as ctx:
                    model(t)
                message = str(ctx.exception)
                entries = [ln for ln in message.splitlines() if ln.startswith("  [")]
                self.assertEqual(entries, want, message)
                self.assertIn("Add a ModelInput", message)

    def test_no_match_report_names_the_results_the_dispatch_judged(self):
        # compiled_results is public, and the report indexes the binding the
        # dispatch built, one per result the call began with. [0]'s check()
        # splices a result in AHEAD of it mid-call: a report that re-read the
        # list by index would ask for a binding the dispatch never made, and an
        # IndexError would take its place; one that re-read it and zipped would
        # describe the spliced result against [0]'s binding under the same
        # header, so the spliced result's describer is watched as well. It is
        # the next call's to judge.
        mod, x = ScaleModule(), torch.randn(3, 3)
        first = aot_compile_forward(mod, make_scaling_forward(2), x)
        later = aot_compile_forward(mod, make_scaling_forward(3), x)
        combined = AOTCompiledModel(mod, [first])
        manager = first._live_guard_manager()
        check = manager.check

        def splicing_check(f_locals):
            # Idempotent: the re-check pass runs this check a second time.
            combined.compiled_results[:] = [later, first]
            return check(f_locals)

        later_manager = later._live_guard_manager()
        describers = Mock()
        describe, later_describe = manager.check_verbose, later_manager.check_verbose
        watch_first = patch.object(manager, "check_verbose", wraps=describe)
        watch_later = patch.object(later_manager, "check_verbose", wraps=later_describe)
        with watch_first as first_described, watch_later as later_described:
            describers.attach_mock(first_described, "first")
            describers.attach_mock(later_described, "later")
            with patch.object(manager, "check", splicing_check):
                with self.assertRaises(RuntimeError) as ctx:
                    combined(x.double())
                lines = str(ctx.exception).splitlines()
                self.assertIn("Tried 1 compiled input(s)", lines[0])
                self.assertEqual(sum(line.startswith("  [") for line in lines), 1)
                self.assertEqual([c[0] for c in describers.mock_calls], ["first"])
                # The next call judges the spliced list: two entries, in the
                # list's order, each described by its own tree.
                with self.assertRaises(RuntimeError) as ctx:
                    combined(x.double())
                lines = str(ctx.exception).splitlines()
                self.assertIn("Tried 2 compiled input(s)", lines[0])
                heads = [line[:6] for line in lines[1:3]]
                self.assertEqual(heads, ["  [0] ", "  [1] "])
                self.assertEqual(sum(line.startswith("  [") for line in lines), 2)
                order = [c[0] for c in describers.mock_calls]
                self.assertEqual(order, ["first", "later", "first"])

    def test_no_match_report_describes_the_other_entries_when_one_raises(self):
        # check_verbose runs paths check() does not -- the repr of a user object
        # inside a DIMENSION_DYNAMIC_MARKING_GUARD, for one -- so an entry's
        # describer can raise where dispatch's rejection was clean, and it must
        # not take the other entries' usable lines with it.
        mod, x = ScaleModule(), torch.randn(3, 3)
        first = aot_compile_forward(mod, make_scaling_forward(2), x)
        second = aot_compile_forward(mod, make_scaling_forward(3), x)
        combined = AOTCompiledModel(mod, [first, second])

        def raising_check_verbose(f_locals):
            # User code under the report: an opt-out flipped here must not turn
            # [1] into a withheld entry; the report judges the read it began with.
            second.disable_guard_check()
            raise ValueError("boom")

        manager = first._live_guard_manager()
        with patch.object(manager, "check_verbose", raising_check_verbose):
            with self.assertRaises(RuntimeError) as ctx:
                combined(x.double())
        lines = str(ctx.exception).splitlines()
        self.assertIn("Tried 2 compiled input(s)", lines[0])
        self.assertEqual(lines[1], "  [0] <guard check raised ValueError: boom>")
        self.assertTrue(lines[2].startswith("  [1] "), lines[2])
        self.assertNotIn("guard check raised", lines[2])
        self.assertIn("Add a ModelInput", lines[3])
        # Only a dispatch raise is chained, and only a dispatch raise names an
        # artifact to fix, so a raise this far in leaves both unread: the two
        # readings that tell a report raise from one in dispatch, which prints
        # the same line. The fix-or-drop line is emitted immediately above the
        # ModelInput one, so the assertion above reads its absence already.
        self.assertIsNone(ctx.exception.__cause__)

    def test_no_match_message_reads_a_systemerror_cause_not_the_handled_exception(self):
        # _PyErr_FormatFromCause sets __cause__ and __context__ alike, so only a
        # SystemError whose two differ tells the reads apart: raised inside an
        # `except ValueError:` block, its __context__ is what the CALLER was handling.
        self._hide_leaked_dynamo_globals()

        class Raises:
            def check(self, f_locals):
                raise SystemError("stub tree is unhappy")

        model = self._model_with_stub_trees(Raises())
        try:
            raise ValueError("the caller was handling this")
        except ValueError:
            with self.assertRaises(RuntimeError) as ctx:
                model(torch.randn(3, 3))
        message = str(ctx.exception)
        raised = "  [0] <guard check raised SystemError: stub tree is unhappy>"
        self.assertIn(raised, message.splitlines())
        self.assertNotIn("the caller was handling this", message)

    def test_no_match_message_survives_a_raise_whose_str_raises(self):
        # The exception is the tree's down to its __str__, so quoting it is the
        # last place a raise can still take the whole report with it.
        self._hide_leaked_dynamo_globals()
        model = self._model_with_stub_trees(RaisesUnprintable())
        with self.assertRaises(RuntimeError) as ctx:
            model(torch.randn(3, 3))
        lines = str(ctx.exception).splitlines()
        raised = "  [0] <guard check raised UnprintableError: <str() raised TypeError>>"
        self.assertIn(raised, lines)
        self.assertIsInstance(ctx.exception.__cause__, UnprintableError)

    def test_aot_compile_module_with_no_results_raises_the_no_match_report(self):
        # aot_compile_module refuses an empty list, but the constructor and
        # deserialize() take one, and the parent indexed results[0]: IndexError.
        # Both passes and the last resort are no-ops over nothing, so the call
        # ends in the report, with the one advice that fits it.
        compiled = AOTCompiledModel(ScaleModule(), [])
        with self.assertRaises(RuntimeError) as ctx:
            compiled(torch.randn(3, 3))
        lines = str(ctx.exception).splitlines()
        header = "No AOT compiled graph matched this call. Tried 0 compiled input(s):"
        self.assertEqual(lines[0], header)
        self.assertEqual(sum(ln.startswith("  [") for ln in lines), 0, lines)
        self.assertIn("Add a ModelInput", lines[-1])
        self.assertIsNone(ctx.exception.__cause__)

    def test_no_match_report_reads_the_dispatch_record_after_a_raise(self):
        # [0] raised in the scan and rejected on the second pass, so its line is
        # the accept line its check_verbose produces, not its raise line; [1]
        # raised in both passes with two texts and is quoted for the last, the
        # one that left it unanswered. The chain carries [0]'s raise: the FIRST
        # index that raised.
        self._hide_leaked_dynamo_globals()

        class RaisesThenRejects:
            def __init__(self):
                self.checks = 0

            def check(self, f_locals):
                self.checks += 1
                if self.checks == 1:
                    raise RuntimeError("the first pass is unhappy")
                return False

            def check_verbose(self, f_locals):
                return types.SimpleNamespace(result=True, verbose_code_parts=[])

        class RaisesTwice(RaisesThenRejects):
            def check(self, f_locals):
                self.checks += 1
                # Chained on purpose: only a SystemError is unwrapped to its cause.
                raise RuntimeError(f"pass {self.checks} unhappy") from ValueError("x")

        stubs = [RaisesThenRejects(), RaisesTwice()]
        model = self._model_with_stub_trees(*stubs)
        with self.assertRaises(RuntimeError) as ctx:
            model(torch.randn(3, 3))
        message = str(ctx.exception)
        lines = message.splitlines()
        self.assertEqual([stub.checks for stub in stubs], [2, 2])
        accepted = f"  [0] {_ACCEPTED_IN_THE_REPORT}"
        self.assertIn(accepted, lines)
        self.assertNotIn("ValueError", message)
        self.assertIn("  [1] <guard check raised RuntimeError: pass 2 unhappy>", lines)
        self.assertNotIn("pass 1 unhappy", message)
        self.assertEqual(str(ctx.exception.__cause__), "the first pass is unhappy")
        self.assertNotIn("the first pass is unhappy", message)
        self.assertIn("[0]'s guard check raised while checking this call", message)
        self.assertEqual(sum(ln.startswith("  [") for ln in lines), 2, lines)

    def test_no_match_message_quotes_the_last_raise_at_one_index(self):
        # Both dispatch passes evaluate an enabled tree, so one index can raise
        # twice with two different exceptions. The report quotes the later one:
        # `raised[i]` is overwritten in place, which the test above pins on its
        # entry line at [1]. What this one-input twin adds is the chain: it
        # carries the first index that raised, so with a single input that is
        # the overwritten exception and `__cause__` reads pass 2 too -- a reading
        # the two-input test cannot make, its chain being [0]'s only raise. The
        # pass the line names is the count: every check() here is inside a
        # handler that overwrites the record or re-raises, so a pass too few or
        # too many moves the text below.
        self._hide_leaked_dynamo_globals()
        model = self._model_with_stub_trees(RaisesPerPass())
        with self.assertRaises(RuntimeError) as ctx:
            model(torch.randn(3, 3))
        lines = str(ctx.exception).splitlines()
        raised = "  [0] <guard check raised RuntimeError: pass 2 unhappy>"
        self.assertEqual(lines[1], raised)
        self.assertEqual(str(ctx.exception.__cause__), "pass 2 unhappy")

    def test_aot_compile_module_two_raisers_in_the_report_and_then_the_warning(self):
        # [0] opted out and [1] enabled both raise, so the two records of a raise
        # part: the chain carries the FIRST index that raised, whose opt-out line
        # quotes no exception text, while the advice names the first ENABLED one.
        # [1]'s text holds two separators splitlines() breaks on, so its line
        # asserted whole pins the raise line's collapse.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)
        model = self._model_with_stub_trees(
            RaisingTree("the opted-out tree is unhappy"),
            RaisingTree("checked\x0ctree\x1eis unhappy"),
        )
        results = model.forward.compiled_results
        results[0].disable_guard_check()
        # The report chains the raise, so that path logs nothing about it.
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            with self.assertRaises(RuntimeError) as ctx:
                model(x)
        lines = str(ctx.exception).splitlines()
        self.assertEqual(sum(ln.startswith("  [") for ln in lines), 2, lines)
        withheld = "  [0] <opted out of guard checks; withheld because [1]'s guard check raised>"
        self.assertIn(withheld, lines)
        checked = "  [1] <guard check raised RuntimeError: checked tree is unhappy>"
        self.assertIn(checked, lines)
        self.assertIn("[1]'s raise, not a guard failure, is what withheld", lines[3])
        # Two raisers and no answer, the shape the every-tree-raised footer is
        # for, but the withheld line has already said what [1]'s raise cost:
        # with `not withheld` dropped from the footer's gate it lands here too.
        self.assertEqual(len(lines), 4, lines)
        self.assertNotIn("the opted-out tree is unhappy", str(ctx.exception))
        self.assertEqual(str(ctx.exception.__cause__), "the opted-out tree is unhappy")
        # Opted out as well, [1] withholds the last resort no longer, which serves
        # the first opted-out result: one warning per raise, each naming that
        # index as served and not the raiser it is about.
        results[1].disable_guard_check()
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
        warned = "\n".join(logs.output)
        self.assertEqual(len(logs.output), 2, warned)
        first = "[0]'s guard check raised RuntimeError: the opted-out tree is unhappy"
        self.assertIn(first, warned)
        # Raw, unlike the report line: only _raised_line collapses separators.
        raw = "[1]'s guard check raised RuntimeError: checked\x0ctree\x1eis unhappy"
        self.assertIn(raw, warned)
        self.assertEqual(warned.count("dispatch served [0]"), 2)

    def test_aot_compile_module_interrupt_out_of_a_guard_tree_propagates(self):
        # Neither dispatch handler, [0]'s inline check or accepts(), reads a
        # Ctrl-C inside a tree as an answer about this call: it leaves as itself,
        # where a handler that kept it would serve another result over it or end
        # the call in a report quoting it.
        self._hide_leaked_dynamo_globals()

        class Interrupts:
            def check(self, f_locals):
                raise KeyboardInterrupt("ctrl-c inside the tree")

        x = torch.randn(3, 3)
        # [1] accepts, so [0]'s handler keeping the interrupt would serve over it.
        with self.assertRaisesRegex(KeyboardInterrupt, "ctrl-c inside the tree"):
            self._model_with_stub_trees(Interrupts(), Accepts())(x)
        # Behind a raiser, the interrupt reaches accepts() with a raise on record.
        with self.assertRaisesRegex(KeyboardInterrupt, "ctrl-c inside the tree"):
            self._model_with_stub_trees(RaisingTree("unhappy"), Interrupts())(x)

    def test_aot_compile_module_raising_tree_does_not_reach_an_opted_out_result(self):
        # A tree that raised rejected nothing, so an opted-out result answering
        # on its strength serves a graph whose guards never passed. A plain call
        # raises here, so the honest answers are that raise or a report naming it.
        model, x = self._aot_compile_dict_branches({}, None)
        evil = {RaisesOnCompare(): 1}
        with self.assertRaisesRegex(ValueError, "boom from __eq__"):
            DictBranchModule()(x, evil)
        # [1] was traced with d=None, so opting it out arms x * 5 as the last
        # resort, which reading the raise as a plain non-match would serve. (The
        # parent never gets this far -- the raise leaves the scan.)
        model.forward.compiled_results[1].disable_guard_check()
        with self.assertRaises(RuntimeError) as ctx:
            model(x, evil)
        message = str(ctx.exception)
        lines = message.splitlines()
        raised = "  [0] <guard check raised ValueError: boom from __eq__ (through the guard tree's pybind boundary)>"
        self.assertIn(raised, lines)
        # The only path that reports an opted-out result: its guards say nothing
        # about why the call was refused, so the report names the artifact to fix.
        withheld = "  [1] <opted out of guard checks; withheld because [0]'s guard check raised>"
        self.assertIn(withheld, lines)
        self.assertIn("[0]'s raise, not a guard failure, is what withheld", message)
        self.assertNotIn("Add a ModelInput", message)
        # The withheld line above has already said what happened to [1], and it
        # is the report's last word: no every-tree-raised footer under it.
        self.assertNotIn("Every guard tree raised", message)
        chained = []
        # Start at the cause: the report itself quotes the raise in [0]'s line,
        # so a walk from ctx.exception would pass with no chaining at all.
        cause: BaseException | None = ctx.exception.__cause__
        while cause is not None:
            chained.append(str(cause))
            cause = cause.__cause__ or cause.__context__
        # The raise a plain call surfaces stays reachable from the report.
        self.assertTrue(any("boom from __eq__" in c for c in chained), chained)

    def test_aot_compile_module_raise_from_an_opted_out_result_withholds_nothing(self):
        # check() ignores the flag, so [0]'s tree still raises through the leaf in
        # both passes; nobody asked for its answer, so the raise vetoes nothing and
        # the last resort serves [0]'s graph once [1] rejects the dict. The serve
        # is warned about, and this raise arrives through the real pybind boundary,
        # so the text pins the unwrap the stub-tree warning tests cannot reach:
        # unwrapped it is the ValueError from __eq__, wrapped the boundary's
        # SystemError over a repr of the check method.
        model, x = self._aot_compile_dict_branches({}, None)
        model.forward.compiled_results[0].disable_guard_check()
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x, {RaisesOnCompare(): 1}), x * 2)
        self.assertEqual(len(logs.output), 1, "\n".join(logs.output))
        raised = "[0]'s guard check raised ValueError: boom from __eq__"
        self.assertIn(raised, logs.output[0])
        self.assertIn("dispatch served [0]", logs.output[0])

    def test_aot_compile_module_serves_a_later_match_over_a_raising_leaf(self):
        # The one place tolerating a raise changes an ANSWER: [0]'s tree raises
        # through the DICT_NOT_CONTAINS leaf and [1] accepts, so [1]'s graph is
        # served where the parent let the SystemError out (and eager raises the
        # ValueError, as the opted-out test above measures). [1] is patched to
        # accept because its real guards, traced on d=None, reject the dict.
        # The warning quotes the unwrapped raise here too, from the scan's site.
        model, x = self._aot_compile_dict_branches({}, None)
        manager = model.forward.compiled_results[1]._live_guard_manager()
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            with patch.object(manager, "check", return_value=True) as check:
                self.assertEqual(model(x, {RaisesOnCompare(): 1}), x * 5)
        # Served from the scan: [1] was asked once.
        self.assertEqual(check.call_count, 1)
        self.assertEqual(len(logs.output), 1, "\n".join(logs.output))
        raised = "[0]'s guard check raised ValueError: boom from __eq__"
        self.assertIn(raised, logs.output[0])
        self.assertIn("dispatch served [1]", logs.output[0])

    def test_aot_compile_module_wrapped_interrupt_out_of_a_guard_tree_propagates(self):
        # The pybind boundary wraps WHATEVER a leaf left set into SystemError, a
        # Ctrl-C inside a key's __eq__ included, so `except Exception` alone
        # reads the interrupt as no answer: served over when another result
        # accepts, otherwise the cause of a report when dispatch caught it, or
        # only a quoted line of one when check_verbose did. Each handler
        # re-raises when the unwrapped cause is not an Exception, so the
        # SystemError leaves the call as the boundary made it, the interrupt one
        # hop down.
        # Which handler sees the wrapped raise: [0]'s inline check, with [1]
        # patched to accept so a swallowed raise would be served over; accepts()
        # on [1], after [0] refused; the report's check_verbose, after [0]'s
        # check() refused in both passes.
        cases = (
            ("inline", ({}, None), 1, True),
            ("accepts", (None, {}), 0, False),
            ("report", ({},), 0, False),
        )
        boundary = "returned a result with an exception set"
        for handler, ds, patched, answer in cases:
            model, x = self._aot_compile_dict_branches(*ds)
            manager = model.forward.compiled_results[patched]._live_guard_manager()
            for exc in (KeyboardInterrupt, SystemExit):
                interrupt = exc("inside __eq__")
                evil = {InterruptsOnCompare(interrupt): 1}
                with self.subTest(exc=exc.__name__, handler=handler):
                    with (
                        patch.object(manager, "check", return_value=answer),
                        self.assertRaisesRegex(SystemError, boundary) as ctx,
                    ):
                        model(x, evil)
                    self.assertIs(ctx.exception.__cause__, interrupt)

    def test_aot_compile_module_doubly_wrapped_interrupt_propagates(self):
        # The boundary wraps whatever a leaf left set, an explicit `raise
        # SystemError(...) from KeyboardInterrupt(...)` included, so an interrupt
        # can arrive two SystemError hops down. _unwrapped_raise follows the whole
        # chain, so the handlers re-raise here as they do one hop down; unwrapping
        # one hop reads SystemError("inner") as an ordinary raise and the call ends
        # in the report, the interrupt quoted on [0]'s line.
        model, x = self._aot_compile_dict_branches({})
        interrupt = KeyboardInterrupt("ctrl-c")
        evil = {WrapsAnInterruptOnCompare(interrupt): 1}
        with self.assertRaisesRegex(
            SystemError, "returned a result with an exception set"
        ) as ctx:
            model(x, evil)
        inner = ctx.exception.__cause__
        self.assertIsInstance(inner, SystemError)
        self.assertIs(inner.__cause__, interrupt)

    @unittest.skipIf(not hasattr(signal, "SIGALRM"), "bounds a hang with SIGALRM")
    def test_aot_compile_module_raise_chained_in_a_cycle_reaches_the_report(self):
        # The chain the boundary hands back is user-built, so it can close on
        # itself: one exception the cause of its own cause is three lines of
        # __eq__. The walk remembers the links it passed and stops at the repeat,
        # quoting the link the cycle closed on; walking it unbounded returns
        # nothing, and it runs before any record exists, so there is nothing to
        # recover from. Under an alarm because the failure it pins is a hang:
        # unbounded, this test never finishes rather than failing.
        model, x = self._aot_compile_dict_branches({})
        evil = {CyclesOnCompare(): 1}

        def bail(signum, frame):
            raise TimeoutError("dispatch did not return")

        prior = signal.signal(signal.SIGALRM, bail)
        signal.setitimer(signal.ITIMER_REAL, 60)
        try:
            with self.assertRaises(RuntimeError) as ctx:
                model(x, evil)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prior)
        quoted = "  [0] <guard check raised SystemError: outer cycle (through the guard tree's pybind boundary)>"
        self.assertIn(quoted, str(ctx.exception).splitlines())

    def test_no_match_report_quotes_a_tree_that_raised_and_then_rejected(self):
        # A tree that raised in the scan and rejected cleanly on the second pass
        # has a guard to quote, so the discard takes it out of the raise branch and
        # the report evaluates it like any other rejection; without the discard the
        # line reads back the stale raise instead. The raise stays the report's
        # cause either way: only `unanswered` drops the index.
        model, x = self._aot_compile_dict_branches(None)
        manager = model.forward.compiled_results[0]._live_guard_manager()
        answers = [ValueError("boom in the scan"), False]
        with patch.object(manager, "check", side_effect=answers) as check:
            with self.assertRaises(RuntimeError) as ctx:
                model(x, {})
        # The scan raised, the second pass rejected; the report ran check_verbose.
        self.assertEqual(check.call_count, 2)
        message = str(ctx.exception)
        entry = next(ln for ln in message.splitlines() if ln.startswith("  ["))
        # The real guard traced on d=None, quoted as any other rejection is.
        self.assertIn("L['d'] is None", entry)
        self.assertNotIn("guard check raised", entry)
        self.assertIn("[0]'s guard check raised while checking this call", message)
        self.assertEqual(str(ctx.exception.__cause__), "boom in the scan")

    def test_no_match_report_reads_an_opted_out_tree_that_raised_as_opted_out(self):
        # check() ignores the flag, so an opted-out result whose tree raises is
        # recorded like any other: [1] is in `unanswered` here as well as opted
        # out. The opt-out is decided first, so [1] is reported once, as the
        # opt-out it is, and the raise line belongs to the enabled raiser its
        # withholding names -- swap the two branches and [1] reads as a raise.
        model, x = self._aot_compile_dict_branches({}, {})
        model.forward.compiled_results[1].disable_guard_check()
        with self.assertRaises(RuntimeError) as ctx:
            model(x, {RaisesOnCompare(): 1})
        lines = str(ctx.exception).splitlines()
        raised = "  [0] <guard check raised ValueError: boom from __eq__ (through the guard tree's pybind boundary)>"
        withheld = "  [1] <opted out of guard checks; withheld because [0]'s guard check raised>"
        entries = [ln for ln in lines if ln.startswith("  [")]
        self.assertEqual(entries, [raised, withheld])

    def test_aot_compile_module_serving_over_a_raise_frees_this_call_s_inputs(self):
        # A recorded raise keeps its traceback, and the traceback keeps this call's
        # frame and args, so all three serving exits clear the record: with the
        # cycle collector off, an input of a call that recorded a raise and then
        # served a graph dies at the return like any other call's. [0] raises
        # through the leaf in both passes throughout and is opted out, so its raise
        # vetoes no last resort; [1]'s patched answers pick which exit serves (its
        # real guards, traced on d=None, reject the dict, as the last-resort case
        # patches it to).
        for exit_name, answers, scale in (
            ("scan", [True], 5),
            ("second pass", [False, True], 5),
            ("last resort", [False, False], 2),
        ):
            with self.subTest(exit=exit_name):
                model, x = self._aot_compile_dict_branches({}, None)
                model.forward.compiled_results[0].disable_guard_check()
                manager = model.forward.compiled_results[1]._live_guard_manager()
                answered = iter(answers)
                key = RaisesOnCompare()
                ref = weakref.ref(key)
                # disable_gc, not gc.disable() and gc.enable(): the collector goes
                # back to whatever the caller had rather than always on.
                with disable_gc():
                    # A plain function, not a Mock: a Mock records the f_locals it
                    # was called with, and the assertion below runs while that
                    # record is still alive, so it would hold the very key.
                    # The serve warns, and the record holds formatted ints and
                    # strings only, so reading it here does not keep the input
                    # alive either.
                    with (
                        patch.object(manager, "check", lambda f_locals: next(answered)),
                        self.assertLogs("torch._dynamo.aot_compile", level="WARNING"),
                    ):
                        self.assertEqual(model(x, {key: 1}), x * scale)
                        del key
                        self.assertIsNone(ref())

    def test_no_match_message_quotes_a_tree_that_raised_then_answered(self):
        # A raise still withholds the opted-out last resort: dispatch cannot tell
        # this flavour, which returns with an exception merely set and leaves
        # nothing stale, from a C++ throw that skips the
        # _reset_relational_guard_state() call on check_nopybind_template's normal
        # exits -- RelationalGuard::reset_state has no Python binding, so nothing
        # here can clear that residue -- and lets a later rejection from the same
        # tree stand on the residue rather than on this call, so it holds the
        # opt-out back on any raise. But the raise is no longer all the report can
        # say about that entry -- the tree did reject this call on the second
        # pass, and that rejection is a real answer, so the report quotes the
        # guard it rejected on and the advice that rejection earns, and names the
        # raise as what withheld the opt-out and what to fix.
        x = torch.ones(3, 3)
        model = torch.compile(DictBranchModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [
                ModelInput(args=(x, {}), kwargs={}, contexts=[]),
                ModelInput(args=(x, None), kwargs={}, contexts=[]),
            ]
        )
        # As above: opting [1] out arms x * 5 for a call no artifact guards.
        model.forward.compiled_results[1].disable_guard_check()
        key = RaisesThenHits(raises=1)
        with self.assertRaises(RuntimeError) as ctx:
            served = model(x, {key: 1})
            self.fail(f"dispatch served {served[0, 0].item()}, not a raise")
        message = str(ctx.exception)
        # The scan raised, the second pass rejected the call, and the report
        # asked a third time; treating the raise as this entry's last word stops
        # at two. A lower bound, as above -- and raises=1 rests the same way on
        # one probe per evaluation: codegen that probed twice per check() would
        # leave the raise inside dispatch pass 1 and fail the assertions below on
        # a probe count rather than on report logic.
        self.assertGreaterEqual(key.compares, 3)
        self.assertIn("[0] not ___dict_contains('foo', L['d'])", message)
        self.assertNotIn("[0] <guard check raised", message)
        withheld = (
            "[1] <opted out of guard checks; withheld because [0]'s guard check raised>"
        )
        self.assertIn(withheld, message)
        self.assertIn("[0]'s raise, not a guard failure, is what withheld", message)
        self.assertIn("Add a ModelInput", message)
        # The withheld line says why [1] was withheld, not that [0]'s rejection
        # followed its raise; the advice standing on that rejection says so.
        self.assertIn("every rejection this advice rests on", message)
        # One line per input: the withheld opt-out is described once, and
        # reaching the re-check for it would add a second line about the guards
        # nobody asked about.
        self.assertEqual(
            sum(ln.startswith("  [") for ln in message.splitlines()), 2, message
        )
        chained = []
        # Start at the cause: a walk from ctx.exception would pass on a report
        # that quoted the raise itself and chained nothing.
        cause: BaseException | None = ctx.exception.__cause__
        while cause is not None:
            chained.append(str(cause))
            cause = cause.__cause__ or cause.__context__
        # No line of the report quotes the raise now, so the chain is the only
        # place left that names it.
        self.assertTrue(any("boom on compare 1" in c for c in chained), chained)

    def test_aot_compile_module_serves_a_later_match_after_an_earlier_raise(self):
        # The one path where tolerating a raise changes an ANSWER and not a
        # message: [0] raised, [1]'s guards passed on this call, and the parent
        # propagated [0]'s raise and served nothing. A raise is not a rejection
        # and says nothing about [1], so [1]'s graph is what dispatch owes the
        # caller; refusing would not undo the relational residue a C++ throw
        # leaves either, so the raise is served over and warned about.
        self._hide_leaked_dynamo_globals()
        mod = GlobalConfigModule()
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        x = torch.randn(4, 8)
        with _set_pooling("mean"):
            expected = mod(x)
        with _set_pooling("sum"):
            self.assertNotEqual(mod(x), expected)
        model._aot_compile(
            [
                ModelInput(args=(x,), kwargs={}, contexts=[_set_pooling("sum")]),
                ModelInput(args=(x,), kwargs={}, contexts=[_set_pooling("mean")]),
            ]
        )

        results = model.forward.compiled_results
        results[0]._artifacts.guard_manager = RaisingTree("guard tree is unhappy")
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            with _set_pooling("mean"):
                served = model(x)
        self.assertEqual(served, expected)
        # Checked once: [1] matched in the scan, so the serve was not pass 2's.
        self.assertEqual(results[0]._artifacts.guard_manager.checks, 1)
        # Nothing else records the raise on this path: the report is never built.
        # The whole line, once: the raiser, its raise, the index served and the
        # enabled tree's advice, which names both directions a stale tree bends.
        self.assertEqual(
            logs.output,
            [
                "WARNING:torch._dynamo.aot_compile:AOT compiled input [0]'s guard check raised RuntimeError: guard tree is unhappy; dispatch served [1] rather than propagating it. Fix or drop input [0]: its next check can reject a call it fits or accept one it does not."
            ],
        )

    @parametrize(
        "site", ("scan", "second_pass", "last_resort"), name_fn=lambda site: site
    )
    def test_aot_compile_module_warns_before_the_served_graph_runs(self, site):
        # The warning describes the dispatch decision, so it is logged before
        # _serve runs the graph: a served graph that raises leaves with the
        # warning logged, not swallowed along with the decision it was about.
        # One case per serving site, each the only detector of the ordering at
        # its own site: a match in the scan; the raiser's own second-pass accept
        # serving it with its scan raise on record; and an enabled [0] that
        # rejects in both passes beside an opted-out [1] that raises in both, so
        # the last resort serves [1].
        self._hide_leaked_dynamo_globals()
        stubs, raiser, served_index, text, opted_out = {
            "scan": ((RaisingTree("unhappy"), Accepts()), 0, 1, "unhappy", False),
            "second_pass": ((RaisesThenAccepts(),), 0, 0, "the scan is unhappy", False),
            "last_resort": ((Rejects(), RaisingTree("unhappy")), 1, 1, "unhappy", True),
        }[site]
        x = torch.randn(3, 3)
        model = self._model_with_stub_trees(*stubs)
        served = model.forward.compiled_results[served_index]
        if opted_out:
            served.disable_guard_check()
        graph_raise = RuntimeError("the graph raised")
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            with patch.object(served, "_serve", side_effect=graph_raise):
                with self.assertRaisesRegex(RuntimeError, "the graph raised"):
                    model(x)
        self.assertEqual(len(logs.output), 1, "\n".join(logs.output))
        raised = f"[{raiser}]'s guard check raised RuntimeError: {text}"
        self.assertIn(raised, logs.output[0])
        self.assertIn(f"dispatch served [{served_index}]", logs.output[0])
        if opted_out:
            self.assertIn("opted out of guard checks, but", logs.output[0])
            # The raiser is the last of the two results here, so the stale accept
            # has no later match to displace and the advice states the remedy alone.
            remedy = "accept a call it does not fit; fix or drop it."
            self.assertIn(remedy, logs.output[0])
            self.assertNotIn("ahead of a later match", logs.output[0])

    def test_aot_compile_module_warning_unwraps_a_pybind_boundary_raise(self):
        # This warning is the only record of a swallowed raise on the serving
        # paths, and a tree that returns to pybind with an exception merely set
        # arrives as a SystemError whose own str() is the bound method's repr:
        # naming that says nothing about what raised, and keying the per-index
        # dedup on it collapses every such raise at one index into one line.
        # This is raise_from_an_opted_out_result_withholds_nothing's model and
        # setup -- [0] raises through its DICT_NOT_CONTAINS leaf, [1] rejects the
        # dict without probing it, so opting [0] out arms the last resort and its
        # own raise vetoes nothing -- extended with the warning that call logs.
        model, x = self._aot_compile_dict_branches({}, None)
        model.forward.compiled_results[0].disable_guard_check()
        # Two defects at one index, told apart only once the SystemError is
        # unwrapped.
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x, {RaisesOnCompare(): 1}), x * 2)
            self.assertEqual(model(x, {RaisesOnCompare(TypeError): 1}), x * 2)
        warned = "\n".join(logs.output)
        self.assertIn("[0]'s guard check raised ValueError: boom from __eq__", warned)
        self.assertIn("[0]'s guard check raised TypeError: boom from __eq__", warned)
        self.assertNotIn("SystemError", warned)
        self.assertEqual(len(logs.output), 2, warned)

    def test_aot_compile_module_restores_torch_function_after_a_throw(self):
        # A tree that THROWS out of C++ returns through
        # RootGuardManager::check_nopybind_template's non-RAII restore and leaves
        # TorchFunction disabled on this thread. TENSOR_MATCH on a strided nested
        # tensor is one such tree: reading its strides fires a TORCH_CHECK.
        # GuardManagerWrapper.check puts the state back before dispatch reads the
        # throw as no answer, so the report is about the raise and not about what
        # the raise left behind -- with the wrapper's restore removed, [0]'s line
        # reads "GLOBAL_STATE changed: torch_function" and the advice is to add a
        # ModelInput, both of them artifacts of our own leak.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )
        nested = torch.nested.nested_tensor(
            [torch.randn(2, 3), torch.randn(3, 3)], layout=torch.strided
        )
        state = torch._C._get_torch_function_state()
        with self.assertRaises(RuntimeError) as ctx:
            model(nested)
        self.assertEqual(torch._C._get_torch_function_state(), state)
        message = str(ctx.exception)
        lines = message.splitlines()
        # One entry line for the one input, matched by shape and terminator: the
        # only report content in this file a test did not author, so where a
        # raise text that splits into two entries would show up. The TORCH_CHECK
        # sentence inside it is ATen's, so it is matched loosely: a reword there
        # must not fail a dispatch test, and TORCH_SHOW_CPP_STACKTRACES=1
        # (run_test.py sets it on every retry) appends the C++ backtrace to it.
        self.assertEqual(sum(ln.startswith("  [") for ln in lines), 1, message)
        line = next(ln for ln in lines if ln.startswith("  ["))
        prefix = "  [0] <guard check raised RuntimeError: "
        self.assertTrue(line.startswith(prefix), line)
        self.assertIn("doesn't support strides", line)
        self.assertTrue(line.endswith(">"), line)
        self.assertNotIn("GLOBAL_STATE changed", message)
        self.assertNotIn("Add a ModelInput", message)

    def test_aot_compile_module_restores_torch_function_after_the_report_throws(self):
        # The test above throws out of check() in both passes, so the report
        # quotes the dispatch record and never calls check_verbose, the module
        # path's only direct call of it, which has the same non-RAII exit. Here
        # check() refuses cleanly and the tree throws only when the report
        # describes it: TENSOR_MATCH's verbose failure branch calls is_parameter,
        # which runs the Parameter metaclass's __instancecheck__, patched to
        # raise. The report catches the throw and quotes it as [0]'s line; the
        # state it leaves is pinned.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )
        (result,) = model.forward.compiled_results
        manager, meta = result._live_guard_manager(), type(torch.nn.Parameter)

        def raising_instancecheck(*args):
            raise RuntimeError("__instancecheck__ raised")

        state = torch._C._get_torch_function_state()
        with (
            patch.object(manager, "check", return_value=False) as check,
            patch.object(meta, "__instancecheck__", raising_instancecheck),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                model(torch.randn(3, 3, dtype=torch.float64))
        # Both passes refused, so the throw is the report's check_verbose.
        self.assertEqual(check.call_count, 2)
        self.assertIn(
            "  [0] <guard check raised RuntimeError: __instancecheck__ raised>",
            str(ctx.exception).splitlines(),
        )
        self.assertEqual(torch._C._get_torch_function_state(), state)

    def test_aot_compile_function_restores_torch_function_after_a_throw(self):
        # load_compiled_function returns an AOTCompiledFunction, whose guard
        # check evaluates the same kind of tree through the same wrapper, so a
        # C++ throw would otherwise leave TorchFunction disabled on this thread
        # and silently stop a __torch_function__ subclass from dispatching
        # afterwards. The throw propagates; only the state it leaves is pinned.
        self._hide_leaked_dynamo_globals()

        def fn(x):
            return x * 2

        compiled_fn = torch.compile(fn, fullgraph=True, backend="eager").aot_compile(
            ((torch.randn(3, 3),), {})
        )
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with open(self.path(), "rb") as f:
            loaded = torch.compiler.load_compiled_function(f)
        # The same tree the module test throws out of: TENSOR_MATCH reading a
        # strided nested tensor's strides fires a TORCH_CHECK.
        nested = torch.nested.nested_tensor(
            [torch.randn(2, 3), torch.randn(3, 3)], layout=torch.strided
        )
        state = torch._C._get_torch_function_state()
        with self.assertRaisesRegex(RuntimeError, "doesn't support strides"):
            loaded(nested)
        self.assertEqual(torch._C._get_torch_function_state(), state)

    def test_no_match_message_advises_an_input_after_an_answer_then_a_raise(self):
        # The tree rejected on the scan and raised on the second pass: its line is
        # the raise, but a ModelInput could still cover the call, unqualified.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )

        stub = RejectsOnceThenRaises("the second pass is unhappy")
        model.forward.compiled_results[0]._artifacts.guard_manager = stub
        with self.assertRaises(RuntimeError) as ctx:
            model(torch.randn(3, 3))
        message = str(ctx.exception)
        lines = message.splitlines()
        self.assertEqual(stub.checks, 2)
        raised = "[0] <guard check raised RuntimeError: the second pass is unhappy>"
        self.assertIn(f"  {raised}", lines)
        # One entry line: this tree's last evaluation raised, so the report must
        # not re-check it, and NeverReChecked's line is what a re-check adds.
        self.assertEqual(sum(ln.startswith("  [") for ln in lines), 1, message)
        self.assertIn("Add a ModelInput", message)
        # The rejection came BEFORE the raise, so the advice carries no caveat.
        self.assertNotIn("every rejection this advice rests on", message)
        self.assertNotIn("Every guard tree raised", message)

    def test_no_match_message_qualifies_a_rejection_that_followed_a_raise(self):
        # The mirror image: raised on the scan, rejected on the second pass. The
        # ModelInput line stays and says the rejection it rests on followed a raise.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )

        stub = RaisesOnceThenRejects("the scan is unhappy")
        model.forward.compiled_results[0]._artifacts.guard_manager = stub
        with self.assertRaises(RuntimeError) as ctx:
            model(torch.randn(3, 3))
        message = str(ctx.exception)
        lines = message.splitlines()
        self.assertEqual(stub.checks, 2)
        self.assertIn("  [0] stub guard rejected", lines)
        self.assertIn("[0]'s guard check raised while checking this call", message)
        advice = next(ln for ln in lines if ln.startswith("Add a ModelInput"))
        # One paragraph: the caveat qualifies the advice, not a line of its own.
        self.assertTrue(advice.endswith(CAVEAT_AFTER_A_RAISE), advice)
        self.assertNotIn("Every guard tree raised", message)

    def test_no_match_message_qualifies_a_rejection_that_followed_a_cpp_throw(self):
        # The stub above raises from Python; this is the raise the caveat's clause
        # is about, out of a real tree. TENSOR_MATCH on a strided nested tensor
        # fires a TORCH_CHECK, and that throw skips the reset on
        # check_nopybind_template's exits, so NO_TENSOR_ALIASING still holds
        # L['x'] from the scan when the second pass visits x's manager -- before
        # y's, whose leaf threw -- and rejects the call there. That stale
        # rejection is the one the advice rests on; the report's re-check runs on
        # the state that rejection reset and quotes y's mismatch instead. Two
        # orderings besides the reset carry this: root accessors stay in
        # Guard.sort_key order (L['x'] before L['y']; the throw leaves before the
        # fail-count bump and sort in check_accessors_nopybind), and
        # TensorCheck::check_verbose compares dispatch key sets before it reads
        # strides, so the re-check names L['y'] rather than throwing again.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(PairModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [
                ModelInput(
                    args=(torch.randn(3, 3), torch.randn(3, 3)), kwargs={}, contexts=[]
                )
            ]
        )
        x = torch.randn(3, 3)
        nested = torch.nested.nested_tensor(
            [torch.randn(2, 3), torch.randn(3, 3)], layout=torch.strided
        )
        with self.assertRaises(RuntimeError) as ctx:
            model(x, nested)
        message = str(ctx.exception)
        lines = message.splitlines()
        cause = ctx.exception.__cause__
        self.assertIn("doesn't support strides", str(cause))
        # pybind translated a throw; no tree returned with an error set.
        self.assertIsNone(cause.__cause__)
        self.assertEqual(sum(ln.startswith("  [") for ln in lines), 1, message)
        entry = next(ln for ln in lines if ln.startswith("  ["))
        self.assertNotIn("<guard check raised", entry)
        self.assertIn("L['y']", entry)
        self.assertIn("[0]'s guard check raised while checking this call", message)
        advice = next(ln for ln in lines if ln.startswith("Add a ModelInput"))
        self.assertTrue(advice.endswith(CAVEAT_AFTER_A_RAISE), advice)
        self.assertNotIn("Every guard tree raised", message)
        # The residue itself, probed after the dispatch under test so that
        # dispatch ran on a tree nothing here had touched: the evaluation right
        # after the throw fails on x as a duplicate.
        result = model.forward.compiled_results[0]
        tree = result._live_guard_manager()
        f_locals = result.prepare_f_locals(model.forward.model, x, nested)
        with self.assertRaisesRegex(RuntimeError, "doesn't support strides"):
            tree.check(f_locals)
        stale = tree.check_verbose(f_locals).verbose_code_parts
        self.assertEqual(stale, ["Duplicate tensor found where not expected!"])

    def test_no_match_message_trusts_one_rejection_taken_before_a_raise(self):
        # One trusted rejection is enough to leave the advice unqualified: [0]
        # rejected and then raised, [1] raised and then rejected. Requiring every
        # rejection on record to be trusted would qualify this report on [1]'s.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [
                ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[]),
                ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[]),
            ]
        )

        results = model.forward.compiled_results
        first = RejectsOnceThenRaises("the second pass is unhappy")
        second = RaisesOnceThenRejects("the scan is unhappy")
        results[0]._artifacts.guard_manager = first
        results[1]._artifacts.guard_manager = second
        with self.assertRaises(RuntimeError) as ctx:
            served = model(torch.randn(3, 3))
            self.fail(f"dispatch served {served[0, 0].item()}, not a raise")
        message = str(ctx.exception)
        lines = message.splitlines()
        self.assertEqual((first.checks, second.checks), (2, 2))
        raised = "[0] <guard check raised RuntimeError: the second pass is unhappy>"
        self.assertIn(f"  {raised}", lines)
        self.assertIn("  [1] stub guard rejected", lines)
        self.assertEqual(sum(ln.startswith("  [") for ln in lines), 2, message)
        # `raised` is in recording order, and [1]'s scan raise came first.
        self.assertIn("[1]'s guard check raised while checking this call", message)
        self.assertEqual(str(ctx.exception.__cause__), "the scan is unhappy")
        self.assertIn("Add a ModelInput", message)
        self.assertNotIn("every rejection this advice rests on", message)
        self.assertNotIn("Every guard tree raised", message)

    def test_aot_compile_module_second_pass_warns_that_it_served_over_a_raise(self):
        # The second serving path that has to record a swallowed raise: nothing
        # matched in the scan, and the second pass found a match. Only the second
        # pass reaches [1]'s accept, so this warning is that call site's.
        # second_pass_warning_names_the_index_served below serves [1] over [0]'s
        # raise too, but with [1] raising in the scan as well; here it rejected
        # there, so the second pass logs one line, about [0] alone.
        self._hide_leaked_dynamo_globals()

        class RefusesThenAccepts:
            # RaisesThenAccepts with a rejection where its scan raise is.
            def __init__(self):
                self.checks = 0

            def check(self, f_locals):
                self.checks += 1
                return self.checks > 1

        x = torch.randn(3, 3)
        stub = RefusesThenAccepts()
        unhappy = RaisingTree("both passes are unhappy")
        model = self._model_with_stub_trees(unhappy, stub)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
        # Rejected in the scan and accepted on the second pass, so the serve --
        # and the warning with it -- is the second pass's.
        self.assertEqual(stub.checks, 2)
        warned = "\n".join(logs.output)
        self.assertEqual(len(logs.output), 1, warned)
        self.assertIn(
            "[0]'s guard check raised RuntimeError: both passes are unhappy", warned
        )
        self.assertIn("dispatch served [1]", warned)

    def test_aot_compile_module_last_resort_warns_about_another_index(self):
        # The last resort serves the FIRST opted-out result, which need not be
        # the one that raised, so the warning reads the raiser off `raised` and
        # the served index off the loop rather than assuming they coincide.
        # opt_out=True opts out BOTH results, so [1]'s raise vetoes nothing and
        # the first opted-out result, [0], is what the last resort serves; with
        # [1] enabled that raise would veto the last resort and this call would
        # raise the no-match report instead of reaching a serve.
        # The two-raisers test above pins that reading too, with both results
        # raising; here [0] never raised, so the closing assertNotIn catches a
        # warning made up for the served index, which that test cannot check.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)
        unhappy = RaisingTree("the second tree is unhappy")
        model = self._model_with_stub_trees(Rejects(), unhappy, opt_out=True)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
        # [1] raised in the scan and again on the second pass: one line about it.
        self.assertEqual(unhappy.checks, 2)
        warned = "\n".join(logs.output)
        self.assertEqual(len(logs.output), 1, warned)
        self.assertIn(
            "[1]'s guard check raised RuntimeError: the second tree is unhappy",
            warned,
        )
        self.assertIn("dispatch served [0]", warned)
        self.assertNotIn("[0]'s guard check raised", warned)

    def test_aot_compile_module_serves_a_tree_that_raised_and_then_accepted(self):
        # The one direction the veto is NOT applied to: [0] raised in the scan
        # and its own accept on the second pass is what serves. The veto holds
        # the last resort back because a raise rejected nothing and an unguarded
        # graph needs real rejections; an accept is that tree's own answer, and
        # vetoing it would not contain the stale relational state either (below),
        # so the warning names both directions a stale tree can bend an answer.
        self._hide_leaked_dynamo_globals()
        stub = RaisesThenAccepts()
        model = self._model_with_stub_trees(stub)
        x = torch.randn(3, 3)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            served = model(x)
        self.assertEqual(served, x * 2)
        # Two checks: the serve is the second pass's, the raiser the index served.
        self.assertEqual(stub.checks, 2)
        warned = "\n".join(logs.output)
        raised = "[0]'s guard check raised RuntimeError: the scan is unhappy"
        self.assertIn(raised, warned)
        self.assertIn("dispatch served [0]", warned)
        self.assertIn("reject a call it fits or accept one it does not", warned)
        # The same tree's next answer is acted on with no raise on record at all,
        # which is why a per-call veto is not what keeps a stale tree honest.
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            self.assertEqual(model(x), x * 2)

    def test_aot_compile_module_opted_out_advice_allows_the_raisers_own_accept(self):
        # Opted out, the same tree is served by its own second-pass accept, not
        # by the last resort, and the raise on record is its first pass's: the
        # advice reads the raiser's opt-out and names only the stale accept a
        # C++ throw can leave, since check() ignores the opt-out and would act
        # on it in index order, not the stale rejection the last resort absorbs.
        # The one result is the last one, so the advice names no later match to
        # displace: the whole line pins that too.
        self._hide_leaked_dynamo_globals()
        stub = RaisesThenAccepts()
        model = self._model_with_stub_trees(stub, opt_out=True)
        x = torch.randn(3, 3)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
        self.assertEqual(stub.checks, 2)
        self.assertEqual(
            logs.output,
            [
                "WARNING:torch._dynamo.aot_compile:AOT compiled input [0]'s guard check raised RuntimeError: the scan is unhappy; dispatch served [0] rather than propagating it. Input [0] opted out of guard checks, but its next check can accept a call it does not fit; fix or drop it."
            ],
        )

    def test_aot_compile_module_warning_advice_follows_the_raiser_not_the_served(self):
        # The advice is about the tree that raised, so it reads the RAISER's
        # opt-out: an enabled raiser served over by an opted-out match is told to
        # fix its tree, not given the opted-out tree's last-resort advice.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)
        unhappy = RaisingTree("guard tree is unhappy")
        model = self._model_with_stub_trees(unhappy, Accepts())
        # check() ignores the opt-out, so [1] matches in the scan and is served.
        model.forward.compiled_results[1].disable_guard_check()
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
        self.assertEqual(len(logs.output), 1, "\n".join(logs.output))
        self.assertIn("dispatch served [1]", logs.output[0])
        self.assertIn("Fix or drop input [0]", logs.output[0])
        self.assertNotIn("opted out of guard checks, but", logs.output[0])

    def test_aot_compile_module_second_pass_warning_names_the_index_served(self):
        # [0] raises in both passes and [1] raises in the scan, then accepts on
        # the second pass: both warnings name [1] as served, the index the second
        # pass's site logs, and not [0], the first raiser or the first result.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)
        stub = RaisesThenAccepts()
        model = self._model_with_stub_trees(RaisingTree("first tree unhappy"), stub)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
        self.assertEqual(stub.checks, 2)
        warned = "\n".join(logs.output)
        self.assertEqual(len(logs.output), 2, warned)
        first = "[0]'s guard check raised RuntimeError: first tree unhappy"
        second = "[1]'s guard check raised RuntimeError: the scan is unhappy"
        self.assertIn(first, warned)
        self.assertIn(second, warned)
        self.assertEqual(warned.count("dispatch served [1]"), 2, warned)

    def test_aot_compile_module_scan_warning_names_the_index_served(self):
        # [0] raises inline, [1] rejects and [2] matches in the scan: the scan's
        # site logs the index it served, [2], not the first result it scanned.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)
        unhappy = RaisingTree("unhappy")
        model = self._model_with_stub_trees(unhappy, Rejects(), Accepts())
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
        self.assertEqual(len(logs.output), 1, "\n".join(logs.output))
        self.assertIn("dispatch served [2]", logs.output[0])
        self.assertIn("Fix or drop input [0]", logs.output[0])

    def test_aot_compile_module_warns_once_per_model_not_per_process(self):
        # The dedup set is a field on the model, which is the whole reason it is
        # not torch._logging.warning_once: that cache is process-global, so the
        # first model to log would silence every later one with the same defect.
        # Nor is it the caller's to hand in, for the same reason: a set shared
        # across models would silence the second model the same way.
        self.assertNotIn("_warned", inspect.signature(AOTCompiledModel).parameters)
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)
        unhappy = RaisingTree("guard tree is unhappy")
        first = self._model_with_stub_trees(unhappy, opt_out=True)
        second = self._model_with_stub_trees(unhappy, opt_out=True)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(first(x), x * 2)
        self.assertEqual(len(logs.output), 1, "\n".join(logs.output))
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            first(x)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(second(x), x * 2)
        self.assertEqual(len(logs.output), 1, "\n".join(logs.output))
        self.assertIn("[0]'s guard check raised RuntimeError", logs.output[0])
        # Its raise vetoes nothing -- nobody asked about its guards -- so the
        # opted-out result itself is what the last resort serves.
        self.assertIn("dispatch served [0]", logs.output[0])
        # The helper's result opted out, so the advice is the opted-out tree's.
        self.assertIn("opted out of guard checks, but", logs.output[0])
        self.assertIn("does not fit; fix or drop it.", logs.output[0])
        self.assertNotIn("Fix or drop input", logs.output[0])

    def test_aot_compile_module_warns_once_per_exception_type_at_one_index(self):
        # The dedup key is (index, exception type name, opt-out state), not the
        # index: two defects at one index told apart by type alone warn twice; a
        # repeat of one is silent.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)

        class Raises:
            error = ValueError

            def check(self, f_locals):
                raise self.error("guard tree is unhappy")

        model = self._model_with_stub_trees(Raises(), opt_out=True)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
            Raises.error = TypeError
            self.assertEqual(model(x), x * 2)
            self.assertEqual(model(x), x * 2)
        self.assertEqual(len(logs.output), 2, "\n".join(logs.output))
        self.assertIn("[0]'s guard check raised ValueError: guard tree", logs.output[0])
        self.assertIn("[0]'s guard check raised TypeError: guard tree", logs.output[1])

    def test_aot_compile_module_warning_survives_a_raise_whose_str_raises(self):
        # logging formats a record lazily, so an exception object handed to
        # log.warning whose __str__ raises never reaches the log; the warning
        # quotes the text the report line quotes, produced up front.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)
        model = self._model_with_stub_trees(RaisesUnprintable(), opt_out=True)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
        self.assertEqual(len(logs.output), 1, "\n".join(logs.output))
        raised = "[0]'s guard check raised UnprintableError: <str() raised TypeError>"
        self.assertIn(raised, logs.output[0])

    def test_aot_compile_module_warns_again_about_a_replaced_index(self):
        # Two broken artifacts are two things to fix, so the warning walks every
        # raise on record rather than the first, and a raising tree raises on
        # every call, so the second call is silent: once per defect, not once
        # per call. The dedup key is (index, exception type name,
        # _guard_check_enabled) and compiled_results is public, so after a pop
        # the artifact at a warned-about index is a different one: the
        # (0, "RuntimeError", True) on record described zero's raise -- nothing
        # here opted out -- not the raise of the artifact now at [0]. The
        # warning compares the results to the ones it last logged about and
        # starts over when they differ. Two results are left after the pop,
        # so a reset hung off the binding verdict would fire here too: the
        # one-result twin below is what tells the two apart.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)
        zero = RaisingTree("zero is unhappy")
        one = RaisingTree("one is unhappy")
        model = self._model_with_stub_trees(zero, one, Accepts())
        results = model.forward.compiled_results
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
        warned = "\n".join(logs.output)
        self.assertEqual(len(logs.output), 2, warned)
        self.assertIn("[0]'s guard check raised RuntimeError: zero is unhappy", warned)
        self.assertIn("[1]'s guard check raised RuntimeError: one is unhappy", warned)
        self.assertEqual(warned.count("dispatch served [2]"), 2, warned)
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            model(x)
        results.pop(0)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
        self.assertEqual(len(logs.output), 1, "\n".join(logs.output))
        replaced = logs.output[0]
        self.assertIn("[0]'s guard check raised RuntimeError: one is unhappy", replaced)
        self.assertIn("dispatch served [1]", replaced)
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            model(x)

    def test_aot_compile_module_warns_again_about_a_replaced_single_result(self):
        # A one-result model binds nothing a second time, so a dedup reset tied
        # to the binding verdict never runs for it and the (0, "RuntimeError",
        # False) its first call logged would silence every artifact later put at
        # [0].
        # The displaced result is held across the swap, so only the identity
        # test in _same_results, not a dead weak reference, explains the reset.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)
        first = RaisingTree("the first artifact is unhappy")
        model = self._model_with_stub_trees(first, opt_out=True)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
        self.assertEqual(len(logs.output), 1, "\n".join(logs.output))
        self.assertIn("RuntimeError: the first artifact is unhappy", logs.output[0])
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            model(x)
        second = RaisingTree("the second artifact is unhappy")
        replacement = self._model_with_stub_trees(second, opt_out=True)
        displaced = model.forward.compiled_results[0]
        model.forward.compiled_results[0] = replacement.forward.compiled_results[0]
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
        self.assertEqual(len(logs.output), 1, "\n".join(logs.output))
        self.assertIn("RuntimeError: the second artifact is unhappy", logs.output[0])
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            model(x)
        self.assertIsNot(model.forward.compiled_results[0], displaced)

    def test_aot_compile_module_warns_again_after_the_raiser_opts_out(self):
        # disable_guard_check() flips a field on the same result, so the results
        # are still the ones last warned about and no reset runs; the opt-out
        # state is in the dedup key instead, since the advice it selects differs
        # and the opted-out one names a cost the enabled one does not.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)
        model = self._model_with_stub_trees(RaisingTree("unhappy"), Accepts())
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
        self.assertEqual(len(logs.output), 1, "\n".join(logs.output))
        self.assertIn("Fix or drop input [0]", logs.output[0])
        # check() ignores the flag, so [0] raises again and [1] is served again.
        model.forward.compiled_results[0].disable_guard_check()
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
        self.assertEqual(len(logs.output), 1, "\n".join(logs.output))
        self.assertIn("dispatch served [1]", logs.output[0])
        self.assertIn("Input [0] opted out of guard checks, but", logs.output[0])
        # A later result exists here, so the stale accept's cost names it.
        self.assertIn("does not fit ahead of a later match", logs.output[0])
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            self.assertEqual(model(x), x * 2)

    def test_aot_compile_module_does_not_warn_again_when_the_serve_moves(self):
        # The served index is in every warning but not in the key: the remedy is
        # the raiser whichever result answered, so a defect whose serve moves is
        # reported once. [0] raises in both passes, [1] matches in the scan; with
        # [1] refusing afterwards the same defect is served by [2] and is silent.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)
        stubs = (RaisingTree("unhappy"), Accepts(), Accepts())
        model = self._model_with_stub_trees(*stubs)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
        self.assertEqual(len(logs.output), 1, "\n".join(logs.output))
        self.assertIn("dispatch served [1]", logs.output[0])
        second = model.forward.compiled_results[1]._live_guard_manager()
        with patch.object(second, "check", return_value=False):
            with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
                self.assertEqual(model(x), x * 2)

    def test_aot_compile_module_warning_not_spent_by_an_interrupt_from_str(self):
        # _quoted catches Exception as the handlers do, so an interrupt out of the
        # recorded exception's __str__ leaves the warning call and discards the
        # graph whose guards passed. A key added before that call would spend the
        # one-shot on a warning the log never saw, so the defect is reported on
        # the next call instead: __str__ raises the interrupt once, and the call
        # after it warns. Boom, not RuntimeError, so an interrupt read as an
        # answer would fail assertRaises rather than the pin below.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)
        interrupt = KeyboardInterrupt("interrupt from __str__")

        class Boom(RuntimeError):
            printed = 0

            def __str__(self):
                Boom.printed += 1
                if Boom.printed == 1:
                    raise interrupt
                return "boom from __str__"

        class RaisesBoom:
            def check(self, f_locals):
                raise Boom

        model = self._model_with_stub_trees(RaisesBoom(), Accepts())
        with self.assertRaises(KeyboardInterrupt) as ctx:
            model(x)
        self.assertIs(ctx.exception, interrupt)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
        self.assertEqual(len(logs.output), 1, "\n".join(logs.output))
        raised = "[0]'s guard check raised Boom: boom from __str__"
        self.assertIn(raised, logs.output[0])
        self.assertIn("dispatch served [1]", logs.output[0])

    def test_aot_compile_module_warning_survives_a_serve_that_does_not_warn(self):
        # The three call sites gate _warn_swallowed on `if raised:`, and that
        # gate is what makes a list changed and changed back keep the old keys
        # in force: a serve with nothing on record never reaches the reset, so
        # it cannot install a fresh set over another results list. The
        # intermediate list rejects at [0] so its serve goes through the scan
        # site, the one the first call warned from; a lone Accepts() would be
        # served by the inline first-result path, which calls nothing here.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)
        model = self._model_with_stub_trees(RaisingTree("unhappy"), Accepts())
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
        self.assertEqual(len(logs.output), 1, "\n".join(logs.output))
        self.assertIn("dispatch served [1]", logs.output[0])
        warned_about = tuple(model.forward.compiled_results)
        healthy = self._model_with_stub_trees(Rejects(), Accepts())
        model.forward.compiled_results[:] = healthy.forward.compiled_results
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            self.assertEqual(model(x), x * 2)
        model.forward.compiled_results[:] = warned_about
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            self.assertEqual(model(x), x * 2)

    def test_aot_compile_module_warning_is_not_spent_while_the_logger_drops_it(self):
        # The one-shot is spent only by a warning the logger would emit: a serve
        # while the logger's level is above WARNING records nothing, so the
        # defect surfaces once the level comes down. assertNoLogs sets the
        # logger's level to the one given for its block.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3, 3)
        model = self._model_with_stub_trees(RaisingTree("unhappy"), Accepts())
        with self.assertNoLogs("torch._dynamo.aot_compile", level="ERROR"):
            self.assertEqual(model(x), x * 2)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
        self.assertEqual(len(logs.output), 1, "\n".join(logs.output))
        self.assertIn("[0]'s guard check raised RuntimeError: unhappy", logs.output[0])
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            self.assertEqual(model(x), x * 2)

    def test_aot_compile_module_interrupt_out_of_a_later_tree_propagates(self):
        # accepts() catches Exception, so a KeyboardInterrupt out of a later tree
        # leaves __call__ as itself, the earlier raise on record unlogged.
        # SystemExit travels the same path today; its iteration pins the handler
        # against widening to (Exception, SystemExit), not a second path. The
        # last call pins where the dedup is judged: the aborted calls consumed
        # nothing, so it still warns about [0]. Judging the pair at the raise
        # instead fails this and
        # test_aot_compile_module_two_raisers_in_the_report_and_then_the_warning.
        # assertNoLogs outermost: nested inside the raise it would check nothing.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        x = torch.randn(3, 3)
        inputs = [ModelInput(args=(x,), kwargs={}, contexts=[]) for _ in range(2)]
        model._aot_compile(inputs)

        class Interrupts:
            def __init__(self, interrupt):
                self.interrupt = interrupt

            def check(self, f_locals):
                raise self.interrupt("inside the tree")

        results = model.forward.compiled_results
        results[0]._artifacts.guard_manager = AlwaysRaises("zero is unhappy")
        real = results[1]._artifacts.guard_manager
        logger = "torch._dynamo.aot_compile"
        for interrupt in (KeyboardInterrupt, SystemExit):
            results[1]._artifacts.guard_manager = Interrupts(interrupt)
            with self.assertNoLogs(logger, level="WARNING"):
                with self.assertRaisesRegex(interrupt, "inside the tree"):
                    model(x)
        results[1]._artifacts.guard_manager = real
        with self.assertLogs(logger, level="WARNING") as logs:
            self.assertEqual(model(x), x * 2)
        self.assertEqual(len(logs.output), 1)
        warned = logs.output[0]
        self.assertIn("[0]'s guard check raised RuntimeError: zero is unhappy", warned)
        self.assertIn("dispatch served [1]", warned)

    def test_aot_compile_module_ordinary_dispatch_warns_about_nothing(self):
        # The swallowed-raise warning is about a raise, so an ordinary dispatch
        # that walks past a non-matching input to a matching one is silent.
        # Both graphs compute x * 2, so the check counts are what say the walk
        # happened: [0]'s real tree rejected a float32 call its float64 capture
        # does not fit, and [1]'s accepted it.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [
                ModelInput(
                    args=(torch.randn(3, 3, dtype=torch.float64),),
                    kwargs={},
                    contexts=[],
                ),
                ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[]),
            ]
        )
        results = model.forward.compiled_results
        first, later = (result._artifacts.guard_manager for result in results)
        x = torch.randn(3, 3)
        with (
            patch.object(first, "check", wraps=first.check) as first_check,
            patch.object(later, "check", wraps=later.check) as later_check,
            self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"),
        ):
            served = model(x)
        self.assertEqual(served, x * 2)
        self.assertEqual(first_check.call_count, 1)
        self.assertEqual(later_check.call_count, 1)

    def test_aot_compile_module_scope_resolves_through_forward_hook(self):
        # A registered forward hook makes get_traced_fn(model) return
        # Module._wrapped_call_impl, whose globals are torch/nn/modules/module.py.
        # The guard scope has to come from what was actually traced, model.forward.
        #
        # The hook is deliberately the identity: aot_compile_module traces
        # model.forward directly and never runs hooks, so a hook with an effect
        # would simply be dropped from the compiled result. This pins the guard
        # SCOPE resolution on a hooked module, not hook support.
        def make_mod():
            mod = GlobalConfigModule()
            mod.register_forward_hook(lambda m, i, o: o)
            return mod

        self._check_module_global_guard_dispatch(make_mod, _set_pooling)

    def test_aot_compile_module_reload_reads_the_live_global(self):
        # A module artifact's bytecode reads a guarded global's LIVE value, not
        # the value serialized at capture -- narrower than a function artifact
        # loaded with an f_globals, which merges the whole dict. The load feeds
        # the scope resolved from model.forward to the guards and merges the
        # certified names into the bytecode's globals, and _serve re-reads them
        # out of that scope before every call, skipping the recorded
        # __builtins_dict___N key whether or not a guard reads it.
        # keep_global_guards is what makes that guard exist at all -- the default
        # aot_compile filter drops every global guard, which would leave nothing
        # guarding AOT_HERMETIC_WEIGHT. That scope is this module's dict, which
        # the capture leaks Dynamo's generated names into, and the load seeds
        # nothing into it: the only kept global guard is rooted at
        # AOT_HERMETIC_WEIGHT, not at an import alias or the builtins-dict key.
        # What each call reads is pinned by
        # test_aot_compile_module_rebind_after_load_is_served.
        global AOT_HERMETIC_WEIGHT

        self._hide_leaked_dynamo_globals()
        mod = HermeticModule()
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        x = torch.randn(3, 3)
        captured = mod(x)
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()

        torch._dynamo.reset()
        saved = AOT_HERMETIC_WEIGHT
        try:
            # Shape and dtype survive the rebind, so the kept TENSOR_MATCH
            # passes either way: what makes the call follow the rebind is that
            # the bytecode reads the scope rather than the value serialized with
            # the artifact, so it serves this tensor and not the capture-time
            # product.
            AOT_HERMETIC_WEIGHT = saved * 2
            live = HermeticModule()(x)
            self.assertNotEqual(captured.tolist(), live.tolist())
            reloaded = torch.compile(
                HermeticModule(),
                fullgraph=True,
                backend="eager",
                options={"guard_filter_fn": keep_global_guards},
            )
            reloaded._load_aot_compiled_module(data)
            self.assertEqual(reloaded(x), live)
            # The scope is re-read before every call, not once at load, so a
            # rebind a kept TENSOR_MATCH accepts -- it checks metadata, not
            # values -- is what the graph computes with.
            AOT_HERMETIC_WEIGHT = saved * 3
            self.assertEqual(reloaded(x), x @ (saved * 3))
            # A rebind the graph cannot serve is refused rather than served.
            AOT_HERMETIC_WEIGHT = saved.to(torch.float64)
            with self.assertRaisesRegex(
                RuntimeError, r"G\['AOT_HERMETIC_WEIGHT'\].*dtype mismatch"
            ):
                reloaded(x)
        finally:
            AOT_HERMETIC_WEIGHT = saved

    def test_aot_compile_module_guards_track_rebound_global(self):
        # The other half of the contract. Guards resolve against the loading
        # process's scope dict itself, so a global rebound after the artifact is
        # loaded redirects dispatch. A copy taken at load time would go on
        # serving whichever graph matched then, with no error and a wrong answer.
        # The graph specialized on this global rather than lifting it, so the
        # bytecode never reads it: its globals carry the name, and every call
        # refreshes it, but only the guards consult its value.
        #
        # _set_pool_mode REBINDS the global, which is what makes the helper's
        # post-load checks pin the by-reference read. The helper's other callers
        # pass _set_pooling, which mutates a container in place, and a copy of
        # the scope shows that just as well.
        from torch._dynamo.source import get_global_source_name

        self._hide_leaked_dynamo_globals()
        model = torch.compile(
            GlobalRebindModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile(
            [ModelInput(args=(torch.randn(4, 8),), kwargs={}, contexts=[])]
        )
        (captured,) = model.forward.compiled_results
        output_graph = load_guards_state(captured._artifacts.guards_state).output_graph
        sources = [guard.originating_source for guard in output_graph.guards]
        roots = {get_global_source_name(source) for source in sources}
        self.assertIn("AOT_POOL_MODE", roots)
        self.assertIn("AOT_POOL_MODE", output_graph.global_scope)
        # Two halves: used_globals says the graph did not lift it, external_refs
        # that no LOAD_GLOBAL in the bytecode reads it. Not co_names, which keeps
        # the traced function's whole name table whether an instruction uses it.
        runtime_env = captured._artifacts.runtime_env
        self.assertNotIn("AOT_POOL_MODE", runtime_env.used_globals)
        self.assertNotIn("AOT_POOL_MODE", runtime_env.external_refs)

        torch._dynamo.reset()
        self._check_module_global_guard_dispatch(GlobalRebindModule, _set_pool_mode)

    @parametrize(
        "hooks",
        sorted(_HOOK_REGISTRARS),
        name_fn=lambda hooks: hooks.replace(" ", "_").replace("-", "_"),
    )
    def test_aot_compile_module_warns_on_dropped_hooks(self, hooks):
        # Every per-instance dict _call_impl dispatches on, warned on both
        # paths: the capture traces model.forward and the load calls the
        # artifact, so neither runs a hook the loading process registered.
        def make_mod():
            mod = ScaleModule()
            _HOOK_REGISTRARS[hooks](mod)
            return mod

        def naming_the_hooks(logs):
            return [
                line
                for line in logs.output
                if f"{hooks} registered" in line and "do NOT run" in line
            ]

        model = torch.compile(make_mod(), fullgraph=True, backend="eager")
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            model._aot_compile(
                [ModelInput(args=(torch.randn(3),), kwargs={}, contexts=[])]
            )
        self.assertEqual(len(naming_the_hooks(logs)), 1, "\n".join(logs.output))
        # Both kinds have a capture to redirect to, but the backward one is
        # conditional: it needs compiled autograd enabled around that trace, and
        # its artifact cannot be reloaded, so its warning has to say both.
        redirect = "AOT compile torch.compile(model).forward"
        self.assertIn(redirect, naming_the_hooks(logs)[0])
        conditional = "compiled autograd enabled around the capture"
        if hooks.startswith("backward"):
            self.assertIn(conditional, naming_the_hooks(logs)[0])
        else:
            self.assertNotIn(conditional, naming_the_hooks(logs)[0])
        # Following the redirect means calling a function, not a module, so both
        # wordings have to say what its first argument is: a module carrying only
        # a backward hook gets this warning and no other, and without the clause
        # a caller who did as it says fails the recorded len(L['args']) == 1
        # guard on the first call.
        self.assertIn(
            "with the module as its first argument", naming_the_hooks(logs)[0]
        )

        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        reloaded = torch.compile(make_mod(), fullgraph=True, backend="eager")
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            reloaded._load_aot_compiled_module(data)
        self.assertEqual(len(naming_the_hooks(logs)), 1, "\n".join(logs.output))

    def test_aot_compile_module_warns_one_record_per_group(self):
        # One record per group rather than per dict, and all three wordings at
        # once: a module carrying both forward kinds, both backward kinds and a
        # real __call__ override gets three records, each naming every kind in
        # its group. Nothing else pins either join, and the parametrization above
        # registers one kind at a time by construction.
        mod = CustomCallModule()
        for register in _HOOK_REGISTRARS.values():
            register(mod)
        x = torch.ones(3)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            torch.compile(mod, fullgraph=True, backend="eager")._aot_compile(
                [ModelInput(args=(x,), kwargs={}, contexts=[])]
            )
        self.assertEqual(len(logs.output), 3, "\n".join(logs.output))
        joined = "\n".join(logs.output)
        self.assertIn("has forward pre-hooks, forward hooks registered", joined)
        self.assertIn("has backward pre-hooks, backward hooks registered", joined)
        self.assertIn("overrides __call__", joined)

    def test_aot_compile_module_after_load_hook_is_silently_dropped(self):
        # The residue of the warning above: a hook registered after the load is
        # dropped with no warning at all, and this pins what that silence costs
        # -- a wrong answer rather than an error.
        self._hide_leaked_dynamo_globals()
        x = torch.randn(3)
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()

        torch._dynamo.reset()
        mod = ScaleModule()
        reloaded = torch.compile(mod, fullgraph=True, backend="eager")
        reloaded._load_aot_compiled_module(data)
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            mod.register_forward_hook(lambda m, i, o: o * 100)
            served = reloaded(x)
        self.assertEqual(served, ScaleModule()(x))
        # The very module the artifact holds now answers differently in eager.
        self.assertNotEqual(served.tolist(), mod(x).tolist())

    def test_aot_compile_module_hook_warning_fires_once(self):
        # Dropped hooks are a property of the model, not of a ModelInput, so the
        # paragraph must not repeat once per compiled result:
        # _warn_dropped_module_dispatch runs ahead of the per-ModelInput loop on
        # the capture path and once per deserialize on the load path. The
        # parametrized test_aot_compile_module_warns_on_dropped_hooks above
        # asserts one record per path with a single ModelInput, which cannot tell
        # once per model from once per ModelInput, so this one captures two and
        # asserts the compiled-result count beside the count of warnings.
        self._hide_leaked_dynamo_globals()

        def make_mod():
            mod = ScaleModule()
            mod.register_forward_hook(lambda m, i, o: o * 100)
            return mod

        def hook_warnings(logs):
            hooked = "forward hooks registered"
            return [r for r in logs.records if hooked in r.getMessage()]

        inputs = [
            ModelInput(args=(torch.randn(n),), kwargs={}, contexts=[]) for n in (3, 4)
        ]
        model = torch.compile(make_mod(), fullgraph=True, backend="eager")
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            model._aot_compile(inputs)
        self.assertEqual(len(model.forward.compiled_results), 2)
        self.assertEqual(len(hook_warnings(logs)), 1, logs.output)

        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        reloaded = torch.compile(make_mod(), fullgraph=True, backend="eager")
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            reloaded._load_aot_compiled_module(data)
        self.assertEqual(len(reloaded.forward.compiled_results), 2)
        self.assertEqual(len(hook_warnings(logs)), 1, logs.output)

    def test_aot_compile_module_warns_on_custom_call(self):
        # The other thing tracing model.forward skips. No hook dict records it,
        # so it needs a warning of its own: eager dispatches through
        # type(model).__call__, the compiled forward does not. Warned on both
        # paths, like a dropped hook, since the load has its own module to check.
        def overriding(logs):
            return [line for line in logs.output if "overrides __call__" in line]

        mod = CustomCallModule()
        x = torch.ones(3)
        eager = mod(x)
        model = torch.compile(mod, fullgraph=True, backend="eager")
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        self.assertEqual(len(overriding(logs)), 1, "\n".join(logs.output))
        # The redirect's clause, pinned for this third wording too; the two hook
        # wordings have it asserted in the dropped-hook warning test above.
        self.assertIn("with the module as its first argument", overriding(logs)[0])
        self.assertNotEqual(eager.tolist(), model(x).tolist())

        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        reloaded = torch.compile(CustomCallModule(), fullgraph=True, backend="eager")
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            reloaded._load_aot_compiled_module(data)
        self.assertEqual(len(overriding(logs)), 1, "\n".join(logs.output))

    def test_aot_compile_module_instance_call_is_not_warned_about(self):
        # An instance attribute named __call__ is not an override: CPython
        # resolves a special method on the type, so eager ignores it and the
        # artifact matches. An instance-first probe would warn that the result
        # may differ from eager -- on every load as well as the capture -- and
        # send the caller to a redirect that runs the attribute:
        # OptimizedModule._initialize reads self._orig_mod.__call__ off the
        # INSTANCE, so torch.compile(model).forward wraps what the instance dict
        # holds rather than _wrapped_call_impl.
        mod = ScaleModule()
        mod.__call__ = lambda *args, **kwargs: torch.zeros(3)
        x = torch.ones(3)
        self.assertEqual(mod(x), x * 2)
        model = torch.compile(mod, fullgraph=True, backend="eager")
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        self.assertEqual(model(x), x * 2)

        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        loading = ScaleModule()
        loading.__call__ = lambda *args, **kwargs: torch.zeros(3)
        reloaded = torch.compile(loading, fullgraph=True, backend="eager")
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            reloaded._load_aot_compiled_module(data)
        self.assertEqual(reloaded(x), x * 2)

    def test_aot_compile_module_fx_call_wrapper_is_not_warned_about(self):
        # fx.GraphModule reinstalls a wrapper as its per-instance class's
        # __call__ on every recompile, and recompile always runs from __init__,
        # so a bare type lookup calls every GraphModule an override -- including
        # ExportedProgram.module(), a plausible input here. That wrapper only
        # prettifies tracebacks and delegates to nn.Module.__call__, so eager
        # runs the hooks and the artifact matches; warning would send the caller
        # after a non-problem. A subclass that really does define __call__ is
        # still an override, and the delegation is what finds it there.
        x = torch.ones(3)
        gm = torch.fx.symbolic_trace(ScaleModule())
        self.assertIsNot(type(gm).__call__, torch.nn.Module.__call__)
        model = torch.compile(gm, fullgraph=True, backend="eager")
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        self.assertEqual(model(x), x * 2)
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        reloaded = torch.compile(
            torch.fx.symbolic_trace(ScaleModule()), fullgraph=True, backend="eager"
        )
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            reloaded._load_aot_compiled_module(data)
        self.assertEqual(reloaded(x), x * 2)

        class OverridingGraphModule(torch.fx.GraphModule):
            def __call__(self, arg):
                return arg * 3

        overriding = OverridingGraphModule(ScaleModule(), gm.graph)
        self.assertEqual(overriding(x), x * 3)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            torch.compile(overriding, fullgraph=True, backend="eager")._aot_compile(
                [ModelInput(args=(x,), kwargs={}, contexts=[])]
            )
        overriding_records = [ln for ln in logs.output if "overrides __call__" in ln]
        self.assertEqual(len(overriding_records), 1, "\n".join(logs.output))

        # A traced module whose __class__ was swapped afterwards, the pattern
        # parametrize, fully_shard and replicate all use: type(model) is then a
        # fresh subclass of the class FX wrapped, whose vars still hold that
        # wrapper, so a probe keyed on type(model) alone calls it an override --
        # for a module whose eager call is nn.Module's and whose artifact
        # matches, asserted below.
        swapped = torch.fx.symbolic_trace(torch.nn.Linear(3, 3))
        torch.nn.utils.parametrize.register_parametrization(
            swapped, "weight", torch.nn.Identity()
        )
        self.assertIsNot(type(swapped), type(swapped)._wrapped_call.cls)
        self.assertTrue(issubclass(type(swapped), type(swapped)._wrapped_call.cls))
        self.assertNotIn("_wrapped_call", vars(type(swapped)))
        wide = torch.ones(2, 3)
        eager = swapped(wide)
        compiled = torch.compile(swapped, fullgraph=True, backend="eager")
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            compiled._aot_compile([ModelInput(args=(wide,), kwargs={}, contexts=[])])
        self.assertEqual(compiled(wide), eager)
        # That capture itself installed a second wrapper, on the fresh subclass
        # this time: serializing the guards pickles the GraphModule, whose
        # __reduce__ recompiles it onto type(self), and recompile reads
        # vars(cls), which does not see the inherited _wrapped_call, so the
        # subclass gets a _WrappedCall of its own, keyed on itself with no
        # cls_call. Two classes on the MRO now carry one, so skipping a single
        # class identity leaves the other to be called an override. Neither is
        # one, and the artifact still answers Linear's forward, so every later
        # probe in this process -- a load, or this second capture -- has to stay
        # silent too. (FX's own lookup has the same shape and recurses in eager
        # on this module, so its eager answer is not something to compare the
        # artifact against.)
        for c in type(swapped).__mro__[:2]:
            self.assertIn("__call__", vars(c))
            self.assertIs(vars(c)["_wrapped_call"].cls, c)
            self.assertIsNone(vars(c)["_wrapped_call"].cls_call)
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            compiled._aot_compile([ModelInput(args=(wide,), kwargs={}, contexts=[])])
        self.assertEqual(compiled(wide), eager)

        # The mirror case: a __call__ assigned onto the per-instance class
        # afterwards sits exactly where the wrapper did, so a probe that skipped
        # that class would drop a real override. Eager runs it, the artifact does
        # not, and it warns.
        assigned = torch.fx.symbolic_trace(ScaleModule())
        type(assigned).__call__ = lambda self, arg: arg * 3
        self.assertEqual(assigned(x), x * 3)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            torch.compile(assigned, fullgraph=True, backend="eager")._aot_compile(
                [ModelInput(args=(x,), kwargs={}, contexts=[])]
            )
        overriding_records = [ln for ln in logs.output if "overrides __call__" in ln]
        self.assertEqual(len(overriding_records), 1, "\n".join(logs.output))

        # And a wrapper whose cls_call was set no longer delegates to
        # nn.Module.__call__, so it is an override however it is identified:
        # dynamo_graph_capture_for_export hooks a hooked root's wrapper that way.
        # Probed directly, since its forward graph-breaks on the bytecode
        # flattener before a capture could reach the warning.
        from torch._dynamo.functional_export import dynamo_graph_capture_for_export

        hooked = ScaleModule()
        hooked.register_forward_hook(lambda m, i, o: o + 1)
        exported = dynamo_graph_capture_for_export(hooked)(x)
        self.assertIsNotNone(type(exported)._wrapped_call.cls_call)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            _warn_dropped_module_dispatch(exported)
        overriding_records = [ln for ln in logs.output if "overrides __call__" in ln]
        self.assertEqual(len(overriding_records), 1, "\n".join(logs.output))
        # And still one when that wrapper sits behind a class owning neither
        # attribute, the __class__ swap again: the walk passes the fresh
        # subclass and reaches the wrapper, and cls_call keeps it an override.
        exported.__class__ = type("Swapped", (type(exported),), {})
        self.assertNotIn("__call__", vars(type(exported)))
        self.assertNotIn("_wrapped_call", vars(type(exported)))
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            _warn_dropped_module_dispatch(exported)
        overriding_records = [ln for ln in logs.output if "overrides __call__" in ln]
        self.assertEqual(len(overriding_records), 1, "\n".join(logs.output))

        # _LazyGraphModule defers the recompile that installs the wrapper to the
        # first call or code access, so until then no class on its MRO owns a
        # __call__ and the probe is silent for want of a wrapper; forced, it is
        # the ordinary skipped shape.
        from torch.fx._lazy_graph_module import _LazyGraphModule

        lazy = _LazyGraphModule(ScaleModule(), gm.graph)
        self.assertTrue(lazy._needs_recompile())
        owning = [c for c in type(lazy).__mro__ if "__call__" in vars(c)]
        self.assertEqual(owning, [torch.nn.Module])
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            _warn_dropped_module_dispatch(lazy)
        self.assertEqual(lazy(x), x * 2)
        self.assertFalse(lazy._needs_recompile())
        self.assertIn("__call__", vars(type(lazy)))
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            _warn_dropped_module_dispatch(lazy)

    @parametrize(
        "hooks",
        sorted(_HOOK_REGISTRARS),
        name_fn=lambda hooks: hooks.replace(" ", "_").replace("-", "_"),
    )
    def test_aot_compile_module_dropped_hook_advice(self, hooks):
        # The escape hatch the warning prints, exercised rather than only spelled:
        # AOT compiling torch.compile(model).forward traces _wrapped_call_impl,
        # so a forward hook survives there and the artifact reproduces eager. The
        # same capture keeps a module-level backward hook only with compiled
        # autograd enabled around it, and that artifact cannot be reloaded, which
        # is the pair of conditions its warning states.
        mod = ScaleModule()
        _HOOK_REGISTRARS[hooks](mod)
        x = torch.ones(3, requires_grad=True)
        redirected = torch.compile(mod, fullgraph=True, backend="eager").forward
        if hooks.startswith("backward"):

            def grad_of(fn):
                t = torch.ones(3, requires_grad=True)
                fn(t).sum().backward()
                return t.grad

            with self.assertRaisesRegex(Unsupported, "Module-level backwards hooks"):
                redirected.aot_compile(((x,), {}))
            # The config flag the graph break's own hint names does not reach
            # here: aot_compile never enters the context that reads it, so only
            # the context manager lifts the refusal.
            with torch._dynamo.config.patch(compiled_autograd=True):
                with self.assertRaisesRegex(
                    Unsupported, "Module-level backwards hooks"
                ):
                    redirected.aot_compile(((x,), {}))
            enable = torch._dynamo.compiled_autograd._enable
            with enable(torch.compile(backend="eager")):
                artifact = redirected.aot_compile(((x,), {}))
            self.assertNotEqual(grad_of(mod).tolist(), grad_of(ScaleModule()).tolist())
            self.assertEqual(grad_of(lambda t: artifact(mod, t)), grad_of(mod))
            # What the warning's second half is about: the artifact saves, and
            # the reload raises rather than dropping the hook silently.
            artifact.save_compiled_function(self.path())
            with open(self.path(), "rb") as f:
                with self.assertRaisesRegex(AttributeError, "_in_graph_bw_hooks"):
                    torch.compiler.load_compiled_function(f)
            return
        eager = mod(x)
        self.assertNotEqual(eager.tolist(), ScaleModule()(x).tolist())
        self.assertEqual(redirected.aot_compile(((x,), {}))(mod, x), eager)

    def test_aot_compile_module_lazy_hook_advice_says_to_wrap_again(self):
        # LazyModuleMixin registers its initializer as a forward pre-hook, so
        # every uninitialized lazy module trips the dropped-hook warning -- which
        # is right, a load onto one really would not materialize its parameters.
        # The redirect it prints is what does not apply: _initialize pins the
        # wrapper's forward to _call_lazy_check, which carries no aot_compile,
        # and the pin outlives the initializer, so materializing in place does
        # not open it and the wording has to send the caller back to torch.compile.
        from torch._dynamo.eval_frame import OptimizedModule

        lazy = torch.nn.LazyLinear(8)
        self.assertTrue(lazy._forward_pre_hooks)
        wrapper = torch.compile(lazy, fullgraph=True, backend="eager")
        self.assertIs(wrapper.forward.__func__, OptimizedModule._call_lazy_check)
        self.assertFalse(hasattr(wrapper.forward, "aot_compile"))
        lazy(torch.ones(4, 4))
        self.assertFalse(lazy._forward_pre_hooks)
        self.assertIs(wrapper.forward.__func__, OptimizedModule._call_lazy_check)
        self.assertFalse(hasattr(wrapper.forward, "aot_compile"))
        # Wrapping the materialized module again is what opens the redirect.
        again = torch.compile(lazy, fullgraph=True, backend="eager")
        self.assertTrue(hasattr(again.forward, "aot_compile"))

        # Reachable with no lazy capture anywhere: the load inspects the LOADING
        # process's module, so a donor artifact from any module gets the advice.
        x = torch.ones(3)
        donor = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        donor._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = donor._save_aot_compiled_module()
        torch._dynamo.reset()
        loading = torch.compile(torch.nn.LazyLinear(8), fullgraph=True, backend="eager")
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            loading._load_aot_compiled_module(data)
        hooked = [ln for ln in logs.output if "forward pre-hooks registered" in ln]
        self.assertEqual(len(hooked), 1, "\n".join(logs.output))
        self.assertIn("carries a lazy initializer", hooked[0])
        self.assertIn("then wrap it again", hooked[0])

    def test_aot_compile_module_redirect_arg_clause_is_conditional(self):
        # The redirect's artifact takes the module as its first argument only on
        # the branch _initialize takes for a user module, which wraps __call__ so
        # get_traced_fn unwraps to _wrapped_call_impl(self, *args, **kwargs).
        # Under config.wrap_top_frame -- and for a model.forward dynamo skips --
        # it wraps the module instead, and wrap_inline's inner closes over it, so
        # that artifact takes only the forward arguments and the flat clause
        # would be one argument too many.
        from torch._dynamo.eval_frame import OptimizedModule

        x = torch.ones(3)
        mod = ScaleModule()
        with torch._dynamo.config.patch(wrap_top_frame=True):
            artifact = torch.compile(
                mod, fullgraph=True, backend="eager"
            ).forward.aot_compile(((x,), {}))
        self.assertEqual(artifact(x), x * 2)
        with self.assertRaisesRegex(RuntimeError, "GuardManager check failed"):
            artifact(mod, x)
        # Which of the two a caller gets is decided by the FILE forward is
        # DEFINED in, so the clause names that rather than stock torch.nn
        # classes: a subclass that inherits forward is skipped just as a Linear
        # is, and one that overrides it is not.
        self.assertTrue(OptimizedModule._forward_has_skip_rule(torch.nn.Linear(3, 3)))

        class OverridingLinear(torch.nn.Linear):
            def forward(self, arg):
                return super().forward(arg) * 3

        self.assertTrue(OptimizedModule._forward_has_skip_rule(InheritingLinear(3, 3)))
        self.assertFalse(OptimizedModule._forward_has_skip_rule(OverridingLinear(3, 3)))
        # Nor does the clause name a list, since none spells the rule: check_file
        # consults LEGACY_MOD_INLINELIST before MOD_SKIPLIST, so QuantStub's
        # forward is inlined although it lives under the skipped torch/ao/.
        from torch.ao.quantization import QuantStub

        self.assertFalse(OptimizedModule._forward_has_skip_rule(QuantStub()))
        # And the artifact follows the predicate: the inheriting subclass's takes
        # only the forward arguments and refuses the module-first call.
        inheriting = InheritingLinear(3, 3)
        artifact = torch.compile(
            inheriting, fullgraph=True, backend="eager"
        ).forward.aot_compile(((x,), {}))
        self.assertEqual(artifact(x), inheriting(x))
        with self.assertRaisesRegex(RuntimeError, "GuardManager check failed"):
            artifact(inheriting, x)
        # So the clause all three warnings print is qualified rather than flat.
        torch._dynamo.reset()
        hooked = ScaleModule()
        hooked.register_forward_hook(lambda m, i, o: o)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            torch.compile(hooked, fullgraph=True, backend="eager")._aot_compile(
                [ModelInput(args=(x,), kwargs={}, contexts=[])]
            )
        naming = [ln for ln in logs.output if "forward hooks registered" in ln]
        self.assertEqual(len(naming), 1, "\n".join(logs.output))
        self.assertIn("takes only the forward arguments", naming[0])
        self.assertIn("OptimizedModule._forward_has_skip_rule(model)", naming[0])
        self.assertNotIn("MOD_SKIPLIST", naming[0])

    def test_aot_compile_module_global_hooks_are_not_warned_about(self):
        # The four _global_* dicts are deliberately outside the warning's list:
        # an artifact served through OptimizedModule runs them on the wrapper's
        # own _call_impl, so naming them would be a false positive on the only
        # module load path in tree.
        fired = []
        handle = torch.nn.modules.module.register_module_forward_hook(
            lambda mod, args, out: fired.append(type(mod).__name__)
        )
        self.addCleanup(handle.remove)
        x = torch.ones(3)
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        reloaded = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            reloaded._load_aot_compiled_module(data)
        fired.clear()
        self.assertEqual(reloaded(x), x * 2)
        self.assertEqual(fired, ["OptimizedModule"])

    def test_aot_compile_module_capture_warns_about_the_traced_module(self):
        # aot_compile_module's caller in eval_frame passes _orig_mod, but a direct
        # caller may pass the wrapper, so the unwrap happens here rather than only
        # for the warning: tracing an OptimizedModule.forward reaches eval_frame's
        # compile_wrapper and dies on set_eval_frame, and the wrapper would also be
        # stored as the self the recorded type-id guard is checked against, so a
        # call would match nothing. Asserting the artifact answers covers both.
        # Warning about the wrapper rather than the module it wraps would report a
        # __call__ override nobody wrote and stay silent about the dropped hooks.
        from torch._dynamo.eval_frame import innermost_backend
        from torch._dynamo.hooks import Hooks

        self._hide_leaked_dynamo_globals()
        mod = ScaleModule()
        mod.register_forward_hook(lambda m, i, o: o * 100)
        wrapper = torch.compile(mod, fullgraph=True, backend="eager")
        x = torch.randn(3)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            compiled = torch._dynamo.aot_compile.aot_compile_module(
                wrapper,
                [ModelInput(args=(x,), kwargs={}, contexts=[])],
                Hooks(),
                innermost_backend(wrapper.dynamo_ctx.callback),
            )
        # Every record, not just a filtered one: warning about the wrapper emits
        # exactly one too, and this is the only assertion that the capture is
        # otherwise quiet. assertEqual drops a non-str msg, so join them.
        self.assertEqual(len(logs.output), 1, "\n".join(logs.output))
        self.assertIn("ScaleModule has forward hooks registered", logs.output[0])
        self.assertIs(compiled.model, mod)
        # What got traced is ScaleModule.forward, so the artifact answers -- and it
        # answers without the hook, which eager still runs.
        self.assertEqual(compiled(x), x * 2)
        self.assertEqual(mod(x), x * 200)

    def _two_input_global_guard_artifact(self, x):
        # A module artifact with a kept global guard and one compiled graph per
        # pooling mode, saved and reset so a load has to resolve a scope.
        model = torch.compile(
            GlobalConfigModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile(
            [
                ModelInput(args=(x,), kwargs={}, contexts=[_set_pooling(mode)])
                for mode in ("sum", "mean")
            ]
        )
        data = model._save_aot_compiled_module()
        self.assertEqual(len(pickle.loads(data)), 2)
        torch._dynamo.reset()
        return data

    @staticmethod
    def _unresolvable_forward_module():
        # A partial is neither a plain function nor a bound method, so
        # get_traced_fn cannot resolve it to a Python function whose globals
        # could serve as the guard scope.
        mod = GlobalConfigModule()
        mod.forward = functools.partial(GlobalConfigModule.forward, mod)
        return mod

    def test_aot_compile_module_fallback_scope_warns_once(self):
        # Falling back is a property of the model, not of a ModelInput, so the
        # paragraph must not repeat once per compiled result.
        x = torch.randn(4, 8)
        data = self._two_input_global_guard_artifact(x)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            compiled = AOTCompiledModel.deserialize(
                self._unresolvable_forward_module(), data
            )
        fallback = [r for r in logs.records if "no live guard scope" in r.getMessage()]
        self.assertEqual(len(fallback), 1)
        self.assertIn("GlobalConfigModule.forward (partial)", fallback[0].getMessage())
        self.assertEqual(len(compiled.compiled_results), 2)
        for result in compiled.compiled_results:
            self.assertIs(result._guard_scope, _GuardScope.RECONSTRUCTED)
            self.assertTrue(result._has_global_guards)

    def test_aot_compile_module_fallback_scope_names_a_qualnamed_forward(self):
        # The other half of the report. A bare partial has no __qualname__, and
        # neither does an nn.Module instance, so functools.wraps is how the
        # forward this path cannot resolve acquires one.
        x = torch.randn(4, 8)
        data = self._two_input_global_guard_artifact(x)
        mod = GlobalConfigModule()
        mod.forward = functools.wraps(GlobalConfigModule.forward)(
            functools.partial(GlobalConfigModule.forward, mod)
        )
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            AOTCompiledModel.deserialize(mod, data)
        self.assertIn(
            "GlobalConfigModule.forward (partial named GlobalConfigModule.forward)",
            "\n".join(logs.output),
        )

    def test_aot_compile_module_deserialize_refuses_a_targetless_dynamo_wrapper(
        self,
    ):
        # error_on_graph_break, patch_dynamo_config, disable_nested_graph_breaks
        # and override_cudagraphs bind wrap_dunder_call_ctx_manager's inner, which
        # skips functools.wraps: no __wrapped__ to follow, so the hop stops and
        # the reason names those decorators instead of "inner resolves to inner".
        mod = GlobalConfigModule()
        mod.forward = torch._dynamo.error_on_graph_break(True)(mod.forward)
        self.assertIs(mod.forward.__globals__, vars(torch._dynamo.external_utils))
        self.assertFalse(hasattr(mod.forward, "__wrapped__"))
        self._assert_forward_refused(
            mod,
            torch._dynamo.external_utils,
            "GlobalConfigModule.forward (function named "
            "wrap_dunder_call_ctx_manager.<locals>.inner) is a "
            "torch._dynamo.external_utils function with no __wrapped__ to see "
            "through to the forward it wraps -- the wrapper "
            "torch._dynamo.error_on_graph_break, patch_dynamo_config, "
            "disable_nested_graph_breaks and override_cudagraphs return skips "
            "functools.wraps; bind the forward that decorator wrapped as "
            "model.forward instead",
        )

    def test_aot_compile_module_deserialize_refuses_a_class_body_wrap_numpy_forward(
        self,
    ):
        # A class-body @torch.compiler.wrap_numpy forward is a bound method of
        # external_utils' wrap, which the hop leaves alone, so get_traced_fn
        # resolves to wrap in a torch namespace. functools.wraps copied the
        # forward's __qualname__ onto wrap, so the reason names wrap by its code.
        mod = NumpyForwardModule()
        wrap = mod.forward.__func__
        self.assertEqual(wrap.__code__.co_name, "wrap")
        self.assertIs(wrap.__globals__, vars(torch._dynamo.external_utils))
        code_name = getattr(wrap.__code__, "co_qualname", "wrap")
        self._assert_forward_refused(
            mod,
            torch._dynamo.external_utils,
            "NumpyForwardModule.forward (method named NumpyForwardModule.forward) "
            f"resolves to {code_name}, a functools.wraps'd wrapper over "
            "NumpyForwardModule.forward, whose globals are "
            "torch._dynamo.external_utils's namespace, a torch module a load "
            "neither roots guards in nor seeds; bind the module's own forward, "
            "defined outside torch, as model.forward instead",
        )

    def test_aot_compile_module_fallback_names_a_compiled_partial_forward(self):
        # torch.compile over a partial wraps it in wrap_inline, so the hop lands
        # on the partial and get_traced_fn fails THERE. Falling back is right; the
        # reason has to say what the wrapper reached rather than tell the user to
        # make model.forward a plain function, which the compile_wrapper already is.
        mod = GlobalConfigModule()
        mod.forward = torch.compile(
            functools.partial(GlobalConfigModule.forward, mod), backend="eager"
        )
        self._assert_forward_refused(
            mod,
            torch._dynamo.eval_frame,
            "GlobalConfigModule.forward (function named wrap_inline.<locals>.inner) "
            "resolves through a Dynamo wrapper to an instance of partial, which "
            "get_traced_fn cannot resolve to a Python function; bind a plain "
            "function or bound method as model.forward instead",
        )

    def test_aot_compile_module_fallback_names_a_compiled_builtin_forward(self):
        # torch.compile over a C-implemented bound method (a tensor's sum) fails
        # in get_traced_fn the other way: it has a __self__, so the __func__ read
        # raises AttributeError, not RuntimeError. Only the un-hopped builtin,
        # bound as forward directly, gets the as-given cannot-resolve reason.
        x = torch.randn(4, 8)
        mod = GlobalConfigModule()
        mod.forward = torch.compile(x.sum, backend="eager")
        resolved = torch._dynamo.eval_frame.innermost_fn(mod.forward).__wrapped__
        with self.assertRaises(AttributeError):
            torch._dynamo.convert_frame.get_traced_fn(resolved)
        self._assert_forward_refused(
            mod,
            torch._dynamo.eval_frame,
            "GlobalConfigModule.forward (function named Tensor.sum) resolves "
            "through a Dynamo wrapper to an instance of builtin_function_or_method, "
            "which get_traced_fn cannot resolve to a Python function; bind a plain "
            "function or bound method as model.forward instead",
        )
        mod.forward = x.sum
        scope, reason = _resolve_guard_scope(mod)
        self.assertIsNone(scope)
        self.assertIn(
            "cannot resolve GlobalConfigModule.forward (builtin_function_or_method "
            "named Tensor.sum) to a Python function",
            reason,
        )

    def test_aot_compile_module_alias_only_globals_load_silently(self):
        # Dynamo's own __import_* aliases are the case neither half of the
        # fallback warning covers: the rebuilt scope carries every recorded one
        # freshly imported, so a guard rooted at one is satisfied there and the
        # call answers. ParentWithChildModule's child call roots a kept guard at
        # an alias and at nothing else, so warning here would predict a no-match
        # report for an artifact that loads and answers.
        x = torch.randn(4, 4)
        model = torch.compile(
            ParentWithChildModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        self._hide_leaked_dynamo_globals()

        mod = ParentWithChildModule()
        expected = mod(x)
        mod.forward = functools.partial(ParentWithChildModule.forward, mod)
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            compiled = AOTCompiledModel.deserialize(mod, data)
        for result in compiled.compiled_results:
            self.assertIs(result._guard_scope, _GuardScope.RECONSTRUCTED)
            self.assertFalse(result._has_global_guards)
            # Armed: past the builtins key the serializer always writes, the
            # recorded scope is non-empty and all aliases, the shape the
            # subtraction exists for.
            state = load_guards_state(result._artifacts.guards_state)
            recorded = set(state.output_graph.global_scope)
            recorded -= {state.output_graph.name_of_builtins_dict_key_in_fglobals}
            aliases = set(result._artifacts.runtime_env.import_sources)
            self.assertTrue(recorded & aliases)
            self.assertFalse(recorded - aliases)
        self.assertEqual(compiled(x), expected)

    def test_aot_compile_module_fallback_scope_hint_names_the_forward(self):
        # The rebuilt scope has two producers: a function artifact loaded without
        # an f_globals, and a module whose forward could not be resolved. Only
        # the second may be blamed on get_traced_fn.
        x = torch.randn(4, 8)
        data = self._two_input_global_guard_artifact(x)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING"):
            compiled = AOTCompiledModel.deserialize(
                self._unresolvable_forward_module(), data
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "rebuilt because get_traced_fn cannot resolve GlobalConfigModule"
            r"\.forward \(partial\) to a Python function",
        ) as ctx:
            compiled(x)
        # The advice for this producer has to stay actionable, and only for this
        # one: no module load path takes an f_globals, so it names the forward and
        # the scope deserialize does take, while the producer below keeps the
        # f_globals advice.
        message = str(ctx.exception)
        self.assertIn("KeyError on G['GLOBAL_POOLING_CONFIG']", message)
        self.assertIn("missing from the scope rebuilt from the artifact", message)
        self.assertIn("make model.forward a plain function or bound method", message)
        self.assertIn("guard_globals= scope that carries the name", message)
        self.assertNotIn("a complete live scope", message)
        self.assertNotIn("define it there", message)
        # A reintroduction guard, like the copy in the function test below: the
        # clause the parent had here, "which holds only the globals the graph
        # lifted", was false for this shape, which lifts none at all.
        self.assertNotIn("holds only the globals", message)

        fn = torch.compile(
            global_config_fn,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        ).aot_compile(((x,), {}))
        fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with open(self.path(), "rb") as f:
            loaded = torch.compiler.load_compiled_function(f)
        self.assertIs(loaded._guard_scope, _GuardScope.RECONSTRUCTED)
        with self.assertRaises(RuntimeError) as ctx:
            loaded(x)
        self.assertIn("missing from the scope rebuilt", str(ctx.exception))
        self.assertIn("a complete live scope", str(ctx.exception))
        self.assertNotIn("get_traced_fn", str(ctx.exception))

    def test_aot_compile_module_fallback_scope_runs_without_global_guards(self):
        # The fallback warns rather than refuses, and with no global guard to
        # satisfy it serves the artifact silently: the warning above fires only
        # because that artifact kept one, and the rebuilt scope satisfies all zero
        # of these.
        x = torch.randn(4, 8)
        model = torch.compile(GlobalConfigModule(), fullgraph=True, backend="eager")
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            compiled = AOTCompiledModel.deserialize(
                self._unresolvable_forward_module(), data
            )
        result = compiled.compiled_results[0]
        self.assertIs(result._guard_scope, _GuardScope.RECONSTRUCTED)
        self.assertFalse(result._has_global_guards)
        self.assertEqual(compiled(x), x.sum(1))

    def test_aot_compile_module_deserialize_unwraps_optimized_module(self):
        # An OptimizedModule's forward resolves to a function defined in
        # eval_frame, so a caller who hands the wrapper to this classmethod must
        # get neither that module's namespace as the guard scope nor the aliases
        # a load seeds there. ParentWithChildModule is what makes the seeding
        # observable at all: its child call roots a kept guard at an __import_*
        # alias, so the load really does write a name into whichever scope it
        # resolved.
        x = torch.randn(4, 4)
        model = torch.compile(
            ParentWithChildModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()

        # After the capture, whose own leaked aliases would otherwise stand in
        # for the ones the load has to seed: seeding leaves an already-bound name
        # alone.
        self._hide_leaked_dynamo_globals()
        g = globals()

        eval_frame_globals = vars(torch._dynamo.eval_frame)
        preexisting = frozenset(eval_frame_globals)

        def strip_seeded():
            # Reached only when the assertion below fails, i.e. when a scope
            # resolved off the wrapper really was eval_frame's own namespace:
            # what a mis-rooted load seeds there must not outlive this test.
            for key in [
                k
                for k in eval_frame_globals
                if k.startswith(_MINTED_PREFIXES) and k not in preexisting
            ]:
                del eval_frame_globals[key]

        self.addCleanup(strip_seeded)
        mod = ParentWithChildModule()
        expected = mod(x)
        wrapper = torch.compile(mod, fullgraph=True, backend="eager")
        compiled = AOTCompiledModel.deserialize(wrapper, data)
        # The recorded alias lands in the namespace ParentWithChildModule.forward
        # is defined in, which is this test module, and nowhere else.
        # keep_global_guards drops BUILTIN_MATCH, so no builtins key comes with it.
        self.assertEqual(
            {k for k in g if k.startswith(_MINTED_PREFIXES)},
            {"__import_torch_dot_nn_dot_modules_dot_module"},
        )
        self.assertEqual(
            {k for k in eval_frame_globals if k.startswith(_MINTED_PREFIXES)},
            {k for k in preexisting if k.startswith(_MINTED_PREFIXES)},
        )
        self.assertEqual(compiled(x), expected)
        self.assertIs(compiled.model, wrapper._orig_mod)

        # Wrappers nest: OptimizedModule.__reduce__ rebuilds a deepcopied wrapper
        # without the metadata innermost_fn follows, so torch.compile wraps it
        # again instead of collapsing onto the module. One unwrap would stop at
        # the inner wrapper, whose forward is eval_frame's again.
        nested = torch.compile(copy.deepcopy(wrapper), fullgraph=True, backend="eager")
        self.assertIsInstance(nested._orig_mod, type(nested))
        compiled = AOTCompiledModel.deserialize(nested, data)
        self.assertEqual(
            {k for k in eval_frame_globals if k.startswith(_MINTED_PREFIXES)},
            {k for k in preexisting if k.startswith(_MINTED_PREFIXES)},
        )
        self.assertEqual(compiled(x), expected)
        self.assertIs(compiled.model, nested._orig_mod._orig_mod)

    def test_aot_compile_module_default_filter_keeps_the_serialized_global(self):
        # The resolved scope is the guard scope unconditionally, but the compiled
        # bytecode only reads it when a guard is rooted at a global. Under the
        # default guard_filter_fn every global guard is dropped, so no guard would
        # check a live value substituted for the one the graph was traced with,
        # and the artifact keeps reading the globals serialized with it.
        global EPS

        class EpsModule(torch.nn.Module):
            def forward(self, x):
                return x + EPS

        x = torch.randn(3, 4)
        model = torch.compile(EpsModule(), fullgraph=True, backend="eager")
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()

        torch._dynamo.reset()
        saved = EPS
        try:
            EPS = saved + 2.0
            reloaded = torch.compile(EpsModule(), fullgraph=True, backend="eager")
            reloaded._load_aot_compiled_module(data)
            (result,) = reloaded.forward.compiled_results
            self.assertIs(result._guard_scope, _GuardScope.SUPPLIED)
            self.assertFalse(result._has_global_guards)
            self.assertEqual(reloaded(x), x + saved)
            self.assertNotEqual(reloaded(x).tolist(), (x + EPS).tolist())
        finally:
            EPS = saved

    def test_aot_compile_module_only_the_guarded_global_is_read_live(self):
        # keep_tensor_guards_unsafe keeps TENSOR_MATCH on a plain tensor and drops
        # it on a Parameter, so this artifact has one global a guard checks and one
        # nothing checks. Only the checked one may be read out of the live scope: a
        # passing guard is what certifies that a live value is the one the graph
        # was compiled for.
        global EPS, AOT_UNGUARDED_PARAM

        self._hide_leaked_dynamo_globals()
        saved_eps, saved_param = EPS, AOT_UNGUARDED_PARAM
        self.addCleanup(globals().__setitem__, "EPS", saved_eps)
        self.addCleanup(globals().__setitem__, "AOT_UNGUARDED_PARAM", saved_param)

        x = torch.randn(3)
        keep_tensors = torch.compiler.keep_tensor_guards_unsafe
        model = torch.compile(
            TwoGlobalsModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_tensors},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()

        # EPS keeps its metadata, so its kept guard passes and the graph has to use
        # the new value. The Parameter changes dtype, which nothing checks, so
        # substituting the whole live namespace would silently change the output.
        EPS = torch.tensor(2.0)
        AOT_UNGUARDED_PARAM = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
        reloaded = torch.compile(
            TwoGlobalsModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_tensors},
        )
        reloaded._load_aot_compiled_module(data)
        actual = reloaded(x)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual, x * EPS + saved_param)
        self.assertNotEqual(actual.tolist(), (x * saved_eps + saved_param).tolist())

    def test_aot_compile_module_unnamed_scope_key_is_seeded_from_the_artifact(self):
        # The ___unnamed_scope_<id>_c<n> key embeds id() of a dict in the tracing
        # process, so the live scope a module load resolves never carries it,
        # and the guard rooted there failed every call where the parent's
        # rebuilt scope, built from used_globals, answered. The load now seeds
        # that recording -- the same object the bytecode reads -- and the other
        # guarded global keeps its live read.
        global EPS

        self.addCleanup(globals().__setitem__, "EPS", EPS)
        x = torch.randn(4, 8)
        model = torch.compile(
            UnnamedScopeModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        (captured,) = model.forward.compiled_results
        output_graph = load_guards_state(captured._artifacts.guards_state).output_graph
        (key,) = [n for n in output_graph.global_scope if "___unnamed_scope" in n]
        # Armed: the graph lifted AOT_NS_SCALE, so the artifact carries the dict.
        self.assertIn(key, captured._artifacts.runtime_env.used_globals)
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        # The capture bound the key to the live dict here; a loading process
        # never has it, and the seeding must not replace a binding it finds.
        self.assertIs(globals()[key], _UNNAMED_SCOPE_NS)
        reloaded = AOTCompiledModel.deserialize(UnnamedScopeModule(), data)
        self.assertEqual(reloaded(x), x * 2.0 + EPS)
        self.assertIs(globals()[key], _UNNAMED_SCOPE_NS)
        self._hide_leaked_dynamo_globals()
        self.assertNotIn(key, globals())

        EPS = torch.tensor(3.0)
        reloaded = AOTCompiledModel.deserialize(UnnamedScopeModule(), data)
        self.assertEqual(reloaded(x), x * 2.0 + EPS)
        (result,) = reloaded.compiled_results
        self.assertIs(result._guard_scope, _GuardScope.SUPPLIED)
        self.assertIs(globals()[key], result._artifacts.runtime_env.used_globals[key])

    def test_aot_compile_module_unnamed_scope_builtins_dict_travels_by_reference(self):
        # exec inserted the live builtins dict under the namespace's __builtins__,
        # and the artifact carried that dict by value: every builtin plus whatever
        # an extension module stashes there. pybind11 < 2.13 on CPython < 3.12
        # keeps its internals in a PyCapsule under a __pybind11_internals_v4_*
        # key, so the dump failed on the 3.10/3.11 CI shards, whose scipy is one
        # such module, and passed on 3.12. Stand in for it with a real capsule.
        builtins.__dict__["__pybind11_internals_v4_test__"] = datetime.datetime_CAPI
        self.addCleanup(builtins.__dict__.pop, "__pybind11_internals_v4_test__")
        x = torch.randn(4)
        model = torch.compile(UnnamedScopeRowsModule(), fullgraph=True, backend="eager")
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        self._hide_leaked_dynamo_globals()
        reloaded = AOTCompiledModel.deserialize(UnnamedScopeRowsModule(), data)
        self.assertEqual(reloaded(x), x + _UNNAMED_SCOPE_NS["AOT_NS_ROWS"].sum(0))
        (result,) = reloaded.compiled_results
        (scope,) = result._artifacts.runtime_env.used_globals.values()
        self.assertIs(scope["__builtins__"], builtins.__dict__)
        self.assertIsNot(scope, _UNNAMED_SCOPE_NS)

    def test_aot_compile_module_unnamed_scope_key_is_disowned_when_left_alone(self):
        # A compile in this process that bound the key still owns it through a
        # CleanupHook that deletes the binding once its code object is collected.
        # The guards this load installs read that binding by reference for as
        # long as the artifact lives, so the load takes the ownership even though
        # it leaves the value alone, as the builtins branch does; otherwise the
        # artifact fails with KeyError on G['___unnamed_scope_...'] at its first
        # call after that collection.
        x = torch.randn(4, 8)
        model = torch.compile(
            UnnamedScopeModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        (captured,) = model.forward.compiled_results
        output_graph = load_guards_state(captured._artifacts.guards_state).output_graph
        (key,) = [n for n in output_graph.global_scope if "___unnamed_scope" in n]
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        self._hide_leaked_dynamo_globals()
        # The capture's own hook never reaches CleanupManager on this path, so
        # stand in for a live compile that installed the key.
        hook = CleanupHook.create(globals(), key, _UNNAMED_SCOPE_NS)
        reloaded = AOTCompiledModel.deserialize(UnnamedScopeModule(), data)
        self.assertIs(globals()[key], _UNNAMED_SCOPE_NS)
        hook()
        self.assertIs(globals()[key], _UNNAMED_SCOPE_NS)
        self.assertEqual(reloaded(x), x * 2.0 + EPS)

    def test_aot_compile_module_sub_path_global_is_not_read_live(self):
        # A guard rooted at a SUB-PATH of a global certifies that path, not the
        # object LOAD_GLOBAL produces: keep_tensor_guards_unsafe keeps the
        # TENSOR_MATCH on AOT_NESTED_GLOBAL["a"] and drops the Parameter's, so
        # substituting the container would serve the graph an uncertified ["b"]
        # -- the hole the test above closes, one level of nesting down.
        from torch._dynamo.source import get_global_source_name, GlobalSource

        global AOT_NESTED_GLOBAL

        self._hide_leaked_dynamo_globals()
        AOT_NESTED_GLOBAL = {
            "a": torch.randn(3),
            "b": torch.nn.Parameter(torch.ones(3)),
        }
        saved = dict(AOT_NESTED_GLOBAL)

        class NestedGlobalModule(torch.nn.Module):
            def forward(self, x):
                return x * AOT_NESTED_GLOBAL["a"] + AOT_NESTED_GLOBAL["b"]

        x = torch.randn(3)
        keep_tensors = torch.compiler.keep_tensor_guards_unsafe
        model = torch.compile(
            NestedGlobalModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_tensors},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        (captured,) = model.forward.compiled_results
        output_graph = load_guards_state(captured._artifacts.guards_state).output_graph
        sources = [guard.originating_source for guard in output_graph.guards]
        # Armed: a kept guard does reach this name, but only through a sub-path,
        # so it is a name the root-keyed pick would have substituted.
        roots = {get_global_source_name(source) for source in sources}
        self.assertIn("AOT_NESTED_GLOBAL", roots)
        self.assertNotIn(GlobalSource("AOT_NESTED_GLOBAL"), sources)
        used_globals = captured._artifacts.runtime_env.used_globals
        self.assertIn("AOT_NESTED_GLOBAL", used_globals)

        data = model._save_aot_compiled_module()
        expected = x * saved["a"] + saved["b"]
        torch._dynamo.reset()

        # ["a"] keeps its metadata, so its kept guard passes on the live dict;
        # ["b"] changes dtype, which nothing checks.
        AOT_NESTED_GLOBAL = {
            "a": torch.randn(3),
            "b": torch.nn.Parameter(torch.ones(3, dtype=torch.float64)),
        }
        reloaded = torch.compile(
            NestedGlobalModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_tensors},
        )
        reloaded._load_aot_compiled_module(data)
        (result,) = reloaded.forward.compiled_results
        self.assertIs(result._guard_scope, _GuardScope.SUPPLIED)
        self.assertTrue(result._has_global_guards)
        actual = reloaded(x)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual, expected)
        live = x * AOT_NESTED_GLOBAL["a"] + AOT_NESTED_GLOBAL["b"]
        self.assertNotEqual(actual.tolist(), live.tolist())

    def test_aot_compile_module_rebind_after_load_is_served(self):
        # The pick is re-taken before every call, not once at load. A kept
        # TENSOR_MATCH certifies metadata, so a rebind to a DIFFERENT tensor of
        # identical metadata passes the check, and the graph has to compute with
        # the new one: a load-time snapshot would pass that same check and
        # silently serve the old data. The Parameter nothing checks is rebound
        # the same way and must NOT be served, so the re-take covers exactly the
        # certified set and no more.
        global EPS, AOT_UNGUARDED_PARAM

        load_time_param = AOT_UNGUARDED_PARAM
        self.addCleanup(globals().__setitem__, "AOT_UNGUARDED_PARAM", load_time_param)
        (x,), reloaded, _ = self._load_armed_module(TwoGlobalsModule, 3)
        load_time_eps = EPS
        self.assertEqual(reloaded(x), x * load_time_eps + load_time_param)

        rebound = torch.tensor(2.0)
        self.assertEqual(rebound.dtype, load_time_eps.dtype)
        self.assertEqual(rebound.shape, load_time_eps.shape)
        self.assertEqual(rebound.device, load_time_eps.device)
        self.assertEqual(rebound.requires_grad, load_time_eps.requires_grad)
        self.assertNotEqual(rebound.item(), load_time_eps.item())
        EPS = rebound
        AOT_UNGUARDED_PARAM = torch.nn.Parameter(torch.full((3,), 9.0))
        self.assertEqual(reloaded(x), x * rebound + load_time_param)

    def _load_armed_module(self, module_cls, *sizes, names=("EPS",)):
        # module_cls compiled for one ModelInput per size, saved, and loaded
        # under keep_tensor_guards_unsafe, which keeps the TENSOR_MATCH on each
        # plain-tensor global it reads and so arms the re-read of `names` on
        # every result: a served load-time value is then the caller's doing and
        # not an unarmed artifact. EVERY name in `names` is bound away from its
        # module-level value for the capture -- EPS's own 1e-7 leaves `x * EPS`
        # inside assertEqual's tolerance of zero, where a baseline assertion
        # about the served value would hold against a served zero too -- and
        # restored at cleanup, so the caller rebinds them freely. Each is bound a
        # second time between the save and the load, to a tensor of the same
        # metadata and a different value, so the LOAD-time value the caller
        # baselines against differs from the pickled one in value and not only in
        # identity: a merge that read the artifact's own dict, that binds only
        # the first name, or that does not run at all, then fails the assertion
        # below instead of agreeing numerically.
        scope = globals()

        self._hide_leaked_dynamo_globals()
        for i, name in enumerate(names):
            self.addCleanup(scope.__setitem__, name, scope[name])
            scope[name] = torch.tensor(5.0 + i)
        xs = [torch.randn(n) for n in sizes]
        options = {"guard_filter_fn": torch.compiler.keep_tensor_guards_unsafe}
        model = torch.compile(
            module_cls(), fullgraph=True, backend="eager", options=options
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[]) for x in xs])
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        load_time = {name: torch.tensor(6.0 + i) for i, name in enumerate(names)}
        scope.update(load_time)
        reloaded = torch.compile(
            module_cls(), fullgraph=True, backend="eager", options=options
        )
        reloaded._load_aot_compiled_module(data)
        results = reloaded.forward.compiled_results
        self.assertEqual(len(results), len(sizes))
        for result in results:
            self.assertEqual(result._live_global_names, names)
            # The load-time merge, which no call has had a chance to redo: the
            # bytecode's globals hold the scope's tensor and not the pickled one,
            # for every certified name and not just the first.
            for name, value in load_time.items():
                self.assertIs(result.fn.__globals__[name], value)
        return xs, reloaded, results

    def test_aot_compile_module_every_certified_global_is_re_read(self):
        # Two certified names, which only the sort case below also loads: with
        # one, a re-read that serves the first name and leaves the rest stale, or
        # that abandons them at the first name the scope no longer binds, passes
        # everything.
        # The recorded set is sorted, so AOT_LIVE_SCALE comes first however the
        # guards were collected -- the case below pins that -- and is the name
        # deleted here. The helper binds both names twice and restores both, so
        # the baselines are read after it, not before its pre-capture binding.
        global EPS, AOT_LIVE_SCALE

        (x,), reloaded, (result,) = self._load_armed_module(
            TwoCertifiedModule, 3, names=("AOT_LIVE_SCALE", "EPS")
        )
        load_time_eps, load_time_scale = EPS, AOT_LIVE_SCALE
        self.assertEqual(reloaded(x), x * load_time_eps + load_time_scale)

        EPS, AOT_LIVE_SCALE = torch.tensor(2.0), torch.tensor(9.0)
        self.assertEqual(reloaded(x), x * 2.0 + 9.0)
        self.assertIs(result.fn.__globals__["EPS"], EPS)
        self.assertIs(result.fn.__globals__["AOT_LIVE_SCALE"], AOT_LIVE_SCALE)

        # A name the scope no longer binds is skipped rather than ending the
        # re-read: the guard rooted at it refuses the call while the check is on,
        # so the opt-out is what lets the names after it be observed. EPS sorts
        # after the deleted name and has to be re-read all the same, while the
        # deleted one keeps the last value read.
        result.disable_guard_check()
        del AOT_LIVE_SCALE
        EPS = torch.tensor(3.0)
        self.assertEqual(reloaded(x), x * 3.0 + 9.0)

    def test_aot_compile_module_certified_names_are_sorted_at_load(self):
        # _live_global_names is the SORTED certified set, which is what lets the
        # case above name the global it deletes. _guard_source_globals hands back
        # a set, whose two-element order follows string-hash randomization, so an
        # assertion on the recorded tuple pins the sort only by luck of the seed;
        # a stub that hands back the reverse-sorted order pins it under every
        # seed instead, and the helper's own assertion is what fails when the
        # sort goes. What this case owns is the stub's side: that the load was
        # handed the reverse order, without which that assertion pins nothing.
        real = torch._dynamo.aot_compile._guard_source_globals
        handed = []

        def reverse_sorted(output_graph):
            handed.append(sorted(real(output_graph), reverse=True))
            return handed[-1]

        with patch("torch._dynamo.aot_compile._guard_source_globals", reverse_sorted):
            _, _, _ = self._load_armed_module(
                TwoCertifiedModule, 3, names=("AOT_LIVE_SCALE", "EPS")
            )
        self.assertEqual(handed, [["EPS", "AOT_LIVE_SCALE"]])

    def test_aot_compile_module_store_global_does_not_accumulate(self):
        # A forward that rebinds a certified global itself: the replayed
        # STORE_GLOBAL lands in the bytecode's globals, so before this commit its
        # store accumulated there call over call, the stored value walking
        # EPS * 2, EPS * 4, EPS * 8 while the three calls answered x * EPS,
        # x * 2 EPS, x * 4 EPS, and the guards went on passing on the scope,
        # which the store never reaches.
        # The re-read takes the scope's value back before each call, so three
        # calls answer alike -- the one BC change here that needs no rebind by
        # the caller to observe. Not writing the scope, which eager would, is
        # pre-existing and unchanged.
        (x,), reloaded, (result,) = self._load_armed_module(StoresEpsModule, 3)
        load_time_eps = EPS
        for _ in range(3):
            self.assertEqual(reloaded(x), x * load_time_eps)
        self.assertIs(globals()["EPS"], load_time_eps)
        self.assertEqual(result.fn.__globals__["EPS"], load_time_eps * 2)

    def test_aot_compile_module_load_binds_a_global_off_the_graph(self):
        # The shape the load-time merge exists for, and the one the per-call
        # re-read cannot cover on its own: EPS is returned rather than computed
        # with, so the placeholder it was lifted into is pruned and
        # used_globals, filled from the graph inputs' sources, never records the
        # name, while the generated bytecode reads it. forward_callable builds
        # fn.__globals__ out of import_sources and used_globals alone and runs
        # _check_external_refs at load, before any call, so with the merge gone
        # the load itself refuses this artifact.
        global EPS

        (x,), reloaded, (result,) = self._load_armed_module(ReturnsEpsModule, 3)
        env = result._artifacts.runtime_env
        self.assertIn("EPS", env.external_refs)
        self.assertNotIn("EPS", env.used_globals)
        self.assertNotIn("EPS", env.import_sources)
        # Bound before any call, so the merge is the only thing that can have.
        self.assertIs(result.fn.__globals__["EPS"], EPS)
        self.assertEqual(reloaded(x), (x + 1, EPS))

        rebound = torch.tensor(2.0)
        EPS = rebound
        self.assertEqual(reloaded(x), (x + 1, rebound))
        self.assertIs(result.fn.__globals__["EPS"], rebound)

    def test_aot_compile_module_global_bound_to_none_is_re_read(self):
        # A scope that binds a certified name to None BINDS it, so the graph has
        # to answer None: the distinction _UNBOUND exists for, and the one thing
        # separating it from an absent name. With None as the absent-name default
        # the read cannot tell the two apart, skips the binding and leaves the
        # value the last read wrote, so the graph computes with a value the scope
        # no longer holds while every guard passes -- the failure class this
        # commit exists to remove, on the one lookup the sentinel was added for.
        # Observed on the forward that RETURNS the global, so None reaches the
        # answer instead of an operator, and with the check off, since None is a
        # binding no kept TENSOR_MATCH accepts: dispatch refuses it on both
        # checked passes and the opted-out last resort is what serves the call.
        global EPS

        (x,), reloaded, (result,) = self._load_armed_module(ReturnsEpsModule, 3)
        rebound = torch.tensor(2.0)
        EPS = rebound
        self.assertEqual(reloaded(x), (x + 1, rebound))
        self.assertIs(result.fn.__globals__["EPS"], rebound)

        result.disable_guard_check()
        EPS = None
        answer = reloaded(x)
        self.assertIsNone(answer[1])
        self.assertIsNone(result.fn.__globals__["EPS"])

    def test_aot_compile_module_sweep_rebind_is_served(self):
        # Dispatch's sweep over results[1:] re-reads the guarded global as well.
        # Two results compiled for different input shapes: the first pass
        # refuses the (4,) call on [0], and the sweep serves it from [1], which
        # has to compute with the rebind its own guards just accepted, not the
        # value the load merged into its globals.
        global EPS

        (x3, x4), reloaded, results = self._load_armed_module(EpsOnlyModule, 3, 4)
        load_time_eps = EPS
        rebound = torch.tensor(2.0)
        self.assertNotEqual(rebound.item(), load_time_eps.item())
        EPS = rebound
        served = reloaded(x4)
        self.assertEqual(served, x4 * rebound)
        self.assertNotEqual(served.tolist(), (x4 * load_time_eps).tolist())
        # [1] served the call and re-read; [0] refused it and read nothing.
        self.assertIs(results[1].fn.__globals__["EPS"], rebound)
        self.assertIs(results[0].fn.__globals__["EPS"], load_time_eps)
        # Called directly, AOTCompiledFunction.__call__ re-reads it as well.
        self.assertEqual(results[0](reloaded.forward.model, x3), x3 * rebound)
        self.assertIs(results[0].fn.__globals__["EPS"], rebound)

    def test_aot_compile_module_second_pass_rebind_is_served(self):
        # Dispatch's re-check pass re-reads the guarded global as well. A check()
        # can reject from the recursive dict-tag fast path without running the
        # tree, which answers nothing about the call, so the pass that rescues it
        # serves a call whose guards did pass and owes the graph the same live
        # value the first pass would have handed it -- not the value an EARLIER
        # accepted call re-read into the bytecode's globals, which persists there
        # between calls. Stubbing the scan's answer forces that pass on a LOADED
        # artifact, the only kind the re-read is armed for.
        global EPS

        (x,), reloaded, (result,) = self._load_armed_module(EpsOnlyModule, 3)
        load_time_eps = EPS
        # A call the first pass accepts leaves its re-read in fn.__globals__.
        first_rebind = torch.tensor(3.0)
        EPS = first_rebind
        self.assertEqual(reloaded(x), x * first_rebind)
        self.assertIs(result.fn.__globals__["EPS"], first_rebind)

        rebound = torch.tensor(2.0)
        self.assertNotEqual(rebound.item(), load_time_eps.item())
        EPS = rebound
        manager = result._artifacts.guard_manager
        real_check, answers = manager.check, []

        def rejects_the_scan(*args, **kwargs):
            # False without running the tree, which is what the fast path this
            # covers answers; every check() after the scan's runs the real tree,
            # which the rebind above passes. Stubbed rather than forced with
            # _install_global_probe, whose one miss is a budget ANY reader of the
            # name spends: in whole-file order a reader that got there first left
            # the scan accepting and the re-check pass, the thing under test,
            # unexercised.
            answers.append(None)
            return len(answers) > 1 and real_check(*args, **kwargs)

        with patch.object(manager, "check", side_effect=rejects_the_scan) as check:
            served = reloaded(x)
        # The scan's check() rejected the call and the re-check's accepted it.
        self.assertEqual(check.call_count, 2)
        self.assertEqual(served, x * rebound)
        self.assertNotEqual(served.tolist(), (x * load_time_eps).tolist())
        self.assertNotEqual(served.tolist(), (x * first_rebind).tolist())

    def test_aot_compile_module_opted_out_load_rebind_is_served(self):
        # The re-read reaches the last resort too, the site that serves a result
        # which opted out: the name set is recorded by the load, before
        # disable_guard_check() flips anything, so the graph goes on following the
        # guard scope -- here past a rebind the kept TENSOR_MATCH refuses, which
        # is what pushes dispatch off both earlier passes onto that site.
        global EPS

        (x,), reloaded, (result,) = self._load_armed_module(EpsOnlyModule, 3)
        load_time_eps = EPS
        self.assertEqual(reloaded(x), x * load_time_eps)

        rebound = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.assertNotEqual(rebound.item(), load_time_eps.item())
        EPS = rebound
        with self.assertRaises(RuntimeError) as ctx:
            reloaded(x)
        message = str(ctx.exception)
        self.assertIn("No AOT compiled graph matched this call", message)
        # EPS's own guard is the refusal, not a missing name or another guard.
        self.assertIn(
            "expected type of 'G['EPS']' to be <class 'torch.Tensor'>, "
            "but found <class 'torch.nn.parameter.Parameter'>",
            message,
        )
        result.disable_guard_check()
        served = reloaded(x)
        self.assertEqual(served, x * rebound)
        self.assertNotEqual(served.tolist(), (x * load_time_eps).tolist())
        # A name the scope no longer binds is skipped, not deleted, so the
        # opted-out call serves the last value read.
        del EPS
        self.assertEqual(reloaded(x), x * rebound)

    def test_aot_compile_module_global_deleted_under_the_re_read_is_skipped(self):
        # The scope is live, so a del of the guarded name can land between the
        # guard check and _serve's read of it. The read is one get() with a
        # default, so the name is skipped and the bytecode's globals keep the
        # last value read, where a membership test and then a read would let a
        # bare KeyError out of a call whose guards passed. A dict that answers
        # the membership test and raises from __getitem__ stands in for that
        # window, and dict.get consults neither of the two; the guards hold the
        # live module dict by reference and go on passing.
        global EPS

        (x,), reloaded, (result,) = self._load_armed_module(EpsOnlyModule, 3)
        rebound = torch.tensor(2.0)
        EPS = rebound
        self.assertEqual(reloaded(x), x * rebound)
        self.assertIs(result.fn.__globals__["EPS"], rebound)

        class DeletedBetweenTestAndRead(dict):
            def __contains__(self, name):
                return True

            def __getitem__(self, name):
                raise KeyError(name)

        with patch.object(result, "_guard_globals", DeletedBetweenTestAndRead()):
            self.assertEqual(reloaded(x), x * rebound)

    def test_aot_compile_module_missing_hook_scope_is_not_asked_to_fabricate(self):
        # guard_globals is the caller's own mapping, so neither read of a
        # certified name out of it may go through __getitem__: a defaultdict
        # would be MUTATED by the read and the graph would then compute with the
        # value it fabricated, every guard passing on it -- the exact failure
        # this commit exists to remove. Both reads take the dict.get default, so
        # the load leaves the dict as the caller passed it and the name is
        # skipped, the guard rooted at it refusing the call until the caller
        # opts out, after which the bytecode's globals still hold the value the
        # artifact was serialized with.
        global EPS

        self._hide_leaked_dynamo_globals()
        self.addCleanup(globals().__setitem__, "EPS", EPS)
        EPS = torch.tensor(5.0)
        x = torch.randn(3)
        options = {"guard_filter_fn": torch.compiler.keep_tensor_guards_unsafe}
        model = torch.compile(
            EpsOnlyModule(), fullgraph=True, backend="eager", options=options
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()

        fabricated = torch.tensor(99.0)
        scope = collections.defaultdict(lambda: fabricated)
        compiled = AOTCompiledModel.deserialize(
            EpsOnlyModule(), data, guard_globals=scope
        )
        (result,) = compiled.compiled_results
        self.assertEqual(result._live_global_names, ("EPS",))
        # An unmutated dict is the pin: the only path that can bind `fabricated`
        # in the bytecode's globals is a subscript of this one, which would leave
        # the fabricated entry in it.
        self.assertEqual(dict(scope), {})
        with self.assertRaisesRegex(RuntimeError, r"KeyError on G\['EPS'\]"):
            compiled(x)
        result.disable_guard_check()
        self.assertEqual(compiled(x), x * 5.0)
        self.assertEqual(dict(scope), {})

    @torch._dynamo.config.patch(enable_cpp_symbolic_shape_guards=True)
    def test_aot_compile_module_shape_only_global_arms_the_guard_scope(self):
        # A global reached only through a shape guard is named by no guard's
        # originating_source, so a source-derived root set misses it while the
        # serialized global_scope -- the serializer's own record of what the kept
        # guards read -- carries it. The rebuilt scope does carry that global,
        # lifted into the graph, so its guard is compared against the value
        # serialized with it and cannot fail -- the vacuous half of what the
        # fallback costs, which is why the warning has to fire for it too.
        from torch._dynamo.source import get_global_source_name

        global AOT_DYNAMIC_GLOBAL

        self._hide_leaked_dynamo_globals()
        AOT_DYNAMIC_GLOBAL = torch.randn(8, 4)
        torch._dynamo.mark_dynamic(AOT_DYNAMIC_GLOBAL, 0)

        class DynamicGlobalModule(torch.nn.Module):
            def forward(self, x):
                return x + AOT_DYNAMIC_GLOBAL.sum(0)

        x = torch.randn(4)
        model = torch.compile(DynamicGlobalModule(), fullgraph=True, backend="eager")
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        (captured,) = model.forward.compiled_results
        output_graph = load_guards_state(captured._artifacts.guards_state).output_graph
        sources = [guard.originating_source for guard in output_graph.guards]
        sources += output_graph.guard_on_key_order
        roots = {get_global_source_name(source) for source in sources}
        self.assertNotIn("AOT_DYNAMIC_GLOBAL", roots)
        self.assertIn("AOT_DYNAMIC_GLOBAL", output_graph.global_scope)

        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        mod = DynamicGlobalModule()
        mod.forward = functools.partial(DynamicGlobalModule.forward, mod)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            compiled = AOTCompiledModel.deserialize(mod, data)
        fallback = [r for r in logs.records if "no live guard scope" in r.getMessage()]
        self.assertEqual(len(fallback), 1)
        for result in compiled.compiled_results:
            self.assertTrue(result._has_global_guards)

    @torch._dynamo.config.patch(enable_cpp_symbolic_shape_guards=True)
    def test_aot_compile_module_shape_only_global_is_not_read_live(self):
        # Same channel as the test above, with a resolvable forward so the load
        # reaches the live-value pick. The serialized global_scope carries this
        # global, but no kept guard is rooted at it -- the surviving SHAPE_ENV
        # check reads a size, not a value -- so the pick must leave it alone and
        # the graph must keep the tensor serialized with it. Substituting would
        # hand the graph a float64 tensor with nothing having read the dtype.
        global AOT_SHAPE_ONLY_GLOBAL

        self._hide_leaked_dynamo_globals()
        AOT_SHAPE_ONLY_GLOBAL = torch.randn(8, 4)
        saved = AOT_SHAPE_ONLY_GLOBAL
        torch._dynamo.mark_dynamic(AOT_SHAPE_ONLY_GLOBAL, 0)

        class ShapeOnlyGlobalModule(torch.nn.Module):
            def forward(self, x):
                return x + AOT_SHAPE_ONLY_GLOBAL.sum(0)

        x = torch.randn(4)
        model = torch.compile(ShapeOnlyGlobalModule(), fullgraph=True, backend="eager")
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()

        AOT_SHAPE_ONLY_GLOBAL = torch.randn(8, 4, dtype=torch.float64)
        compiled = AOTCompiledModel.deserialize(ShapeOnlyGlobalModule(), data)
        (result,) = compiled.compiled_results
        self.assertIs(result._guard_scope, _GuardScope.SUPPLIED)
        self.assertTrue(result._has_global_guards)
        actual = compiled(x)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual, x + saved.sum(0))

    def test_aot_compile_module_python_shape_guard_global_arms_the_guard_scope(self):
        # The default-config shape of ..._shape_only_global_arms_the_guard_scope.
        # A SHAPE_ENV guard installed as a Python lambda reads its G['NAME']
        # operands from the resolved scope too, but the serializer records none
        # of them in global_scope (shape_env_sources is filled from the cpp code
        # parts alone), so arming off that scope read this artifact as holding no
        # global guard and the fallback warning stayed silent about a guard that
        # cannot fail: the rebuilt scope hands the lambda the tensor serialized
        # with the graph.
        global AOT_DYNAMIC_GLOBAL

        self._hide_leaked_dynamo_globals()
        AOT_DYNAMIC_GLOBAL = torch.randn(8, 4)
        torch._dynamo.mark_dynamic(AOT_DYNAMIC_GLOBAL, 0)

        class DynamicGlobalModule(torch.nn.Module):
            def forward(self, x):
                return x + AOT_DYNAMIC_GLOBAL.sum(0)

        x = torch.randn(4)
        model = torch.compile(DynamicGlobalModule(), fullgraph=True, backend="eager")
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        (captured,) = model.forward.compiled_results
        guards_state = load_guards_state(captured._artifacts.guards_state)
        self.assertTrue(guards_state.shape_code_parts.python_fallback)
        exprs = guards_state.shape_code_parts.python_code_parts.exprs
        self.assertTrue(any("G['AOT_DYNAMIC_GLOBAL']" in e for e in exprs), exprs)
        self.assertNotIn("AOT_DYNAMIC_GLOBAL", guards_state.output_graph.global_scope)

        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        mod = DynamicGlobalModule()
        mod.forward = functools.partial(DynamicGlobalModule.forward, mod)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            compiled = AOTCompiledModel.deserialize(mod, data)
        fallback = [r for r in logs.records if "no live guard scope" in r.getMessage()]
        self.assertEqual(len(fallback), 1)
        for result in compiled.compiled_results:
            self.assertTrue(result._has_global_guards)
        # The vacuous half itself: a rebind to a shape the range guard refuses
        # is invisible to a guard reading the rebuilt scope.
        AOT_DYNAMIC_GLOBAL = torch.randn(1, 4)
        self.assertEqual(compiled(x).shape, x.shape)

    def test_aot_compile_module_shape_guard_alias_is_seeded_from_the_artifact(self):
        # The default filter drops every guard rooted at an __import_* alias, so
        # an alias only a Python-form shape guard reads is absent from the
        # serialized global_scope, and a seeding gated on that scope alone left
        # it unbound: the load succeeded and every call failed with KeyError on
        # G['__import_...'], where the rebuilt scope, which imports every
        # recorded alias, answered. The seeding gates on the same widened set
        # the arming reads.
        alias = "__import_" + _HELPER_MOD.__name__
        x = torch.randn(4)
        model = torch.compile(ImportedRowsModule(), fullgraph=True, backend="eager")
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        (captured,) = model.forward.compiled_results
        guards_state = load_guards_state(captured._artifacts.guards_state)
        self.assertTrue(guards_state.shape_code_parts.python_fallback)
        exprs = guards_state.shape_code_parts.python_code_parts.exprs
        self.assertTrue(any(f"G['{alias}']" in e for e in exprs), exprs)
        self.assertNotIn(alias, guards_state.output_graph.global_scope)
        self.assertIn(alias, captured._artifacts.runtime_env.import_sources)
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        self._hide_leaked_dynamo_globals()
        self.assertNotIn(alias, globals())
        reloaded = AOTCompiledModel.deserialize(ImportedRowsModule(), data)
        (result,) = reloaded.compiled_results
        self.assertIs(result._guard_scope, _GuardScope.SUPPLIED)
        # An alias is not a user global, so the artifact holds no global guard.
        self.assertFalse(result._has_global_guards)
        self.assertEqual(reloaded(x), x + _HELPER_MOD.HELPER_ROWS.sum(0))
        self.assertIs(globals()[alias], _HELPER_MOD)

    def test_aot_compile_module_dropped_shape_guard_seeds_no_alias(self):
        # A filter that drops SHAPE_ENV leaves the widened set nothing to read:
        # the builder records shape_code_parts on the save pass only, which runs
        # over the kept guards, so the artifact carries no lambda text and the
        # load binds no alias for a guard that does not exist.
        from torch._dynamo.source import ShapeEnvSource

        alias = "__import_" + _HELPER_MOD.__name__
        x = torch.randn(4)
        model = torch.compile(
            ImportedRowsModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        (captured,) = model.forward.compiled_results
        guards_state = load_guards_state(captured._artifacts.guards_state)
        sources = [g.originating_source for g in guards_state.output_graph.guards]
        self.assertFalse(any(isinstance(s, ShapeEnvSource) for s in sources))
        self.assertIsNone(guards_state.shape_code_parts)
        self.assertIn(alias, captured._artifacts.runtime_env.import_sources)
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        self._hide_leaked_dynamo_globals()
        self.assertNotIn(alias, globals())
        reloaded = AOTCompiledModel.deserialize(ImportedRowsModule(), data)
        (result,) = reloaded.compiled_results
        self.assertFalse(result._has_global_guards)
        self.assertEqual(reloaded(x), x + _HELPER_MOD.HELPER_ROWS.sum(0))
        self.assertNotIn(alias, globals())

    def test_aot_compile_module_shape_guard_unnamed_scope_key_is_seeded(self):
        # ..._unnamed_scope_key_is_seeded_from_the_artifact reached through the
        # shape channel on the default filter: the tensor's own TENSOR_MATCH is
        # global and dropped, so the key is in no kept guard's source and not in
        # global_scope, while used_globals carries the dict the lambda reads.
        x = torch.randn(4)
        model = torch.compile(UnnamedScopeRowsModule(), fullgraph=True, backend="eager")
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        (captured,) = model.forward.compiled_results
        guards_state = load_guards_state(captured._artifacts.guards_state)
        exprs = guards_state.shape_code_parts.python_code_parts.exprs
        (key,) = captured._artifacts.runtime_env.used_globals
        self.assertTrue(key.startswith("___unnamed_scope"), key)
        self.assertTrue(any(f"G['{key}']" in e for e in exprs), exprs)
        self.assertNotIn(key, guards_state.output_graph.global_scope)
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        self._hide_leaked_dynamo_globals()
        self.assertNotIn(key, globals())
        reloaded = AOTCompiledModel.deserialize(UnnamedScopeRowsModule(), data)
        self.assertEqual(reloaded(x), x + _UNNAMED_SCOPE_NS["AOT_NS_ROWS"].sum(0))
        (result,) = reloaded.compiled_results
        self.assertIs(globals()[key], result._artifacts.runtime_env.used_globals[key])

    def test_aot_compile_module_attribute_dict_shape_guard_holds_no_global(self):
        # Shape exprs are source names, so a dict read off an attribute renders
        # as L['self'].myG['k']; an unanchored operand scan took k for a global
        # and armed the fallback warning on an artifact holding no global guard.
        x = torch.randn(4)
        model = torch.compile(AttrDictModule(), fullgraph=True, backend="eager")
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        (captured,) = model.forward.compiled_results
        guards_state = load_guards_state(captured._artifacts.guards_state)
        exprs = guards_state.shape_code_parts.python_code_parts.exprs
        self.assertTrue(any("L['self'].myG['k']" in e for e in exprs), exprs)
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        compiled = AOTCompiledModel.deserialize(AttrDictModule(), data)
        (result,) = compiled.compiled_results
        self.assertFalse(result._has_global_guards)

    def test_aot_compile_module_key_order_only_global_is_not_read_live(self):
        # A global certified by nothing but a guard_on_key_order entry. Iterating
        # a module-level dict roots its TYPE_MATCH and its key-order check at the
        # same GlobalSource, the default filter drops the guard for being global,
        # and guard_filter_fn never prunes the key-order set -- so no kept guard
        # checks these tensors and the pick must leave the name alone. Reading it
        # live would serve values nothing ever compared against.
        from torch._dynamo.source import get_global_source_name

        global AOT_KEY_ORDER_GLOBAL

        self._hide_leaked_dynamo_globals()
        AOT_KEY_ORDER_GLOBAL = {"a": torch.ones(4), "b": torch.ones(4) * 2}
        saved = dict(AOT_KEY_ORDER_GLOBAL)

        class KeyOrderGlobalModule(torch.nn.Module):
            def forward(self, x):
                out = x
                for k in AOT_KEY_ORDER_GLOBAL:
                    out = out + AOT_KEY_ORDER_GLOBAL[k]
                return out

        x = torch.randn(4)
        model = torch.compile(KeyOrderGlobalModule(), fullgraph=True, backend="eager")
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        (captured,) = model.forward.compiled_results
        output_graph = load_guards_state(captured._artifacts.guards_state).output_graph
        guards = output_graph.guards
        own = {get_global_source_name(g.originating_source) for g in guards}
        ko_sources = output_graph.guard_on_key_order
        key_order = {get_global_source_name(source) for source in ko_sources}
        self.assertNotIn("AOT_KEY_ORDER_GLOBAL", own)
        self.assertIn("AOT_KEY_ORDER_GLOBAL", key_order)
        # Armed and not vacuous: the name is in the serialized global_scope that
        # arms the pick, and the graph carries the value the narrowing keeps
        # serving instead.
        self.assertIn("AOT_KEY_ORDER_GLOBAL", output_graph.global_scope)
        used_globals = captured._artifacts.runtime_env.used_globals
        self.assertIn("AOT_KEY_ORDER_GLOBAL", used_globals)

        data = model._save_aot_compiled_module()
        expected = x + saved["a"] + saved["b"]
        torch._dynamo.reset()

        # Same type and keys, different values: only a value check could tell.
        AOT_KEY_ORDER_GLOBAL = {"a": torch.ones(4) * 10, "b": torch.ones(4) * 20}
        mod = KeyOrderGlobalModule()
        reloaded = torch.compile(mod, fullgraph=True, backend="eager")
        reloaded._load_aot_compiled_module(data)
        (result,) = reloaded.forward.compiled_results
        self.assertIs(result._guard_scope, _GuardScope.SUPPLIED)
        self.assertTrue(result._has_global_guards)
        actual = reloaded(x)
        self.assertEqual(actual, expected)
        live = x + AOT_KEY_ORDER_GLOBAL["a"] + AOT_KEY_ORDER_GLOBAL["b"]
        self.assertNotEqual(actual.tolist(), live.tolist())

    def _assert_forward_refused(self, mod, namespace, reason):
        # Both halves of a refusal, each on the artifact that can show it: the
        # warning and the hint need a kept guard on a user global, which
        # GlobalConfigModule's artifact has and ParentWithChildModule's does not;
        # the seeding needs a kept guard rooted at an __import_* alias, which
        # only ParentWithChildModule's child call records, so that artifact is
        # what would have written into the namespace a resolution reached.
        x = torch.randn(4, 8)
        data = self._two_input_global_guard_artifact(x)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            compiled = AOTCompiledModel.deserialize(mod, data)
        fallback = [r for r in logs.records if "no live guard scope" in r.getMessage()]
        self.assertEqual(len(fallback), 1)
        self.assertIn(f"report no match. {reason}.", fallback[0].getMessage())
        for result in compiled.compiled_results:
            self.assertIs(result._guard_scope, _GuardScope.RECONSTRUCTED)
            self.assertEqual(result._forward_not_resolved_reason, reason)
        with self.assertRaises(RuntimeError) as ctx:
            compiled(x)
        message = str(ctx.exception)
        self.assertIn("KeyError on G['GLOBAL_POOLING_CONFIG']", message)
        self.assertIn(f"rebuilt because {reason}, or pass", message)

        x = torch.randn(4, 4)
        model = torch.compile(
            ParentWithChildModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        # The capture leaked its own aliases into this module's dict, which is
        # the namespace under test when the forward is a module defined here.
        self._hide_leaked_dynamo_globals()
        scope = vars(namespace)
        preexisting = frozenset(scope)

        def strip_seeded():
            # Reached only when the assertion below fails: what a mis-rooted load
            # seeds into a namespace it resolved must not outlive this test.
            for key in [k for k in scope if k not in preexisting]:
                del scope[key]

        self.addCleanup(strip_seeded)
        compiled = AOTCompiledModel.deserialize(mod, data)
        self.assertEqual({k for k in scope if k not in preexisting}, set())
        for result in compiled.compiled_results:
            self.assertIs(result._guard_scope, _GuardScope.RECONSTRUCTED)

    def test_aot_compile_module_deserialize_refuses_an_nn_module_forward(self):
        # get_traced_fn rewrites an nn.Module argument to THAT module's forward,
        # so resolving from an nn.Module forward would root the guards in its
        # defining namespace and seed it -- this module's, for a module defined
        # here, which is what tells this refusal apart from the torch-namespace
        # one below: a torch.nn.ReLU() forward is refused by both. Plain
        # assignment cannot produce one, since nn.Module.__setattr__ files a
        # Module under _modules and the class attribute keeps winning the lookup;
        # object.__setattr__ can. Fall back to the rebuilt scope instead.
        mod = GlobalConfigModule()
        object.__setattr__(mod, "forward", ParentWithChildModule())
        self._assert_forward_refused(
            mod,
            sys.modules[__name__],
            "GlobalConfigModule.forward (ParentWithChildModule) is an nn.Module, "
            "which get_traced_fn would rewrite to that module's forward, rooting "
            "the guards in its defining namespace; bind a plain function or bound "
            "method as model.forward instead",
        )

    def test_aot_compile_module_deserialize_refuses_a_forwardless_module(self):
        # nn.Module.forward is a class attribute bound to the module-level
        # _forward_unimplemented, so on a module that never overrode it
        # model.forward is a bound method get_traced_fn resolves to a function
        # whose globals are torch.nn.modules.module's namespace -- the dict a
        # hooked module's dispatch would have handed over, reached by the front
        # door.
        class Forwardless(torch.nn.Module):
            pass

        mod = Forwardless()
        unimplemented = torch.nn.modules.module._forward_unimplemented
        self.assertIs(mod.forward.__func__, unimplemented)
        self._assert_forward_refused(
            mod,
            torch.nn.modules.module,
            "Forwardless.forward (method named _forward_unimplemented) resolves to "
            "_forward_unimplemented, whose globals are torch.nn.modules.module's "
            "namespace, a torch module a load neither roots guards in nor seeds; "
            "bind the module's own forward, defined outside torch, as "
            "model.forward instead",
        )

    def test_aot_compile_module_deserialize_refuses_a_lazy_graph_module_forward(self):
        # _LazyGraphModule defers codegen: until real_recompile its forward is
        # the class attribute _lazy_forward, a function defined in
        # torch.fx._lazy_graph_module. After it, the forward is the one every
        # GraphModule has, exec'd into a private copy of its codegen globals that
        # is no module's namespace, so it resolves.
        from torch.fx._lazy_graph_module import _LazyGraphModule

        class Tiny(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x) + 1

        mod = _LazyGraphModule.from_graphmodule(torch.fx.symbolic_trace(Tiny()))
        self.assertIs(mod.forward.__func__, _LazyGraphModule._lazy_forward)
        self._assert_forward_refused(
            mod,
            torch.fx._lazy_graph_module,
            f"{type(mod).__name__}.forward (method named "
            "_LazyGraphModule._lazy_forward) resolves to "
            "_LazyGraphModule._lazy_forward, whose globals are "
            "torch.fx._lazy_graph_module's namespace, a torch module a load "
            "neither roots guards in nor seeds; bind the module's own forward, "
            "defined outside torch, as model.forward instead",
        )
        mod.real_recompile()
        self.assertIsNot(mod.forward.__func__, _LazyGraphModule._lazy_forward)
        scope, reason = _resolve_guard_scope(mod)
        self.assertIsNone(reason)
        self.assertIs(scope, mod.forward.__globals__)
        self.assertNotIn("__name__", scope)

    def test_aot_compile_module_deserialize_resolves_a_graph_module_forward(self):
        # The boundary of the torch refusal: a GraphModule's forward is exec'd
        # into a private per-instance copy of its codegen globals, so its scope
        # is no module's namespace and its function's __module__ is None. Capture
        # and load agree on that dict, and the names fx emits into it are module
        # aliases, never user globals, so nothing there is read live either way.
        class Tiny(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x) + 1

        x = torch.randn(4)
        model = torch.compile(
            torch.fx.symbolic_trace(Tiny()),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()

        gm = torch.fx.symbolic_trace(Tiny())
        self.assertIsNone(gm.forward.__func__.__module__)
        scope, reason = _resolve_guard_scope(gm)
        self.assertIsNone(reason)
        self.assertIs(scope, gm.forward.__globals__)
        self.assertNotIn("__name__", scope)
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            compiled = AOTCompiledModel.deserialize(gm, data)
        for result in compiled.compiled_results:
            self.assertIs(result._guard_scope, _GuardScope.SUPPLIED)
            self.assertFalse(result._has_global_guards)
        self.assertEqual(compiled(x), gm(x))

    def test_aot_compile_module_missing_global_hint_names_the_resolved_module(self):
        # On a module load the caller passed no scope, so "define it there" alone
        # names nothing; the dict torch resolved is a module's namespace, and the
        # hint says which one.
        global GLOBAL_POOLING_CONFIG

        x = torch.randn(4, 8)
        data = self._two_input_global_guard_artifact(x)
        compiled = AOTCompiledModel.deserialize(GlobalConfigModule(), data)
        for result in compiled.compiled_results:
            self.assertIs(result._guard_scope, _GuardScope.SUPPLIED)
        saved = GLOBAL_POOLING_CONFIG
        self.addCleanup(globals().__setitem__, "GLOBAL_POOLING_CONFIG", saved)
        del GLOBAL_POOLING_CONFIG
        with self.assertRaises(RuntimeError) as ctx:
            compiled(x)
        message = str(ctx.exception)
        self.assertIn("KeyError on G['GLOBAL_POOLING_CONFIG']", message)
        self.assertIn(f", here vars({__name__}); define it there", message)
        # Both entries failed on the same name in the one scope the load
        # resolved, so they share one hint line rather than repeating it.
        self.assertEqual(message.count("a guarded global is missing"), 1)
        self.assertIn("For [0, 1]: a guarded global is missing", message)

        # A copy of that namespace carries its __name__ but is not the module, and
        # sending the reader to the module would be wrong: the name has to land
        # in the copy, which lacks it like its source still does.
        scope = dict(globals())
        compiled = AOTCompiledModel.deserialize(
            GlobalConfigModule(), data, guard_globals=scope
        )
        with self.assertRaises(RuntimeError) as ctx:
            compiled(x)
        message = str(ctx.exception)
        self.assertIn("KeyError on G['GLOBAL_POOLING_CONFIG']", message)
        self.assertIn("loaded against; define it there", message)
        self.assertEqual(message.count("a guarded global is missing"), 1)
        self.assertIn("For [0, 1]: a guarded global is missing", message)
        self.assertNotIn("vars(", message)

    @parametrize("wrap_top_frame", (False, True))
    def test_aot_compile_module_deserialize_refuses_a_compiled_module_forward(
        self, wrap_top_frame
    ):
        # inst.forward = torch.compile(inst).forward wraps the module's DISPATCH,
        # not its forward: the bound nn.Module.__call__ (_wrapped_call_impl in
        # torch.nn.modules.module's namespace), or under config.wrap_top_frame the
        # module itself, which the hop reaches as an nn.Module. ParentWithChildModule
        # roots a kept guard at an __import_* alias, which a resolution would seed.
        x = torch.randn(4, 4)
        model = torch.compile(
            ParentWithChildModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        module_globals = vars(torch.nn.modules.module)
        preexisting = frozenset(module_globals)
        inst = ParentWithChildModule()
        expected = inst(x)
        with torch._dynamo.config.patch(wrap_top_frame=wrap_top_frame):
            inst.forward = torch.compile(inst, backend="eager").forward
        resolved = torch._dynamo.eval_frame.innermost_fn(inst.forward)
        if wrap_top_frame:
            # _initialize wraps the module, then __call__ wraps that inner again.
            inner = resolved.__wrapped__
            self.assertIs(inner.__globals__, vars(torch._dynamo.external_utils))
            self.assertIs(inner.__wrapped__, inst)
            reason = (
                "ParentWithChildModule.forward (function named "
                "wrap_inline.<locals>.inner) resolves through a Dynamo wrapper to "
                "an nn.Module, the module's dispatch rather than its forward; bind "
                "that module's forward, or a wrapper over the forward rather than "
                "over the module, as model.forward instead"
            )
        else:
            self.assertEqual(resolved, inst.__call__)
            reason = (
                "ParentWithChildModule.forward (function named "
                "Module._wrapped_call_impl) resolves through a Dynamo wrapper to "
                "Module._wrapped_call_impl, whose globals are "
                "torch.nn.modules.module's namespace, a torch module a load "
                "neither roots guards in nor seeds; bind the module's own forward, "
                "defined outside torch, as model.forward instead"
            )
        # The kept guard is alias-rooted, which the fallback warning stays silent
        # on, so the reason is read where the hint reads it.
        compiled = AOTCompiledModel.deserialize(inst, data)
        for result in compiled.compiled_results:
            self.assertIs(result._guard_scope, _GuardScope.RECONSTRUCTED)
            self.assertEqual(result._forward_not_resolved_reason, reason)
        self.assertEqual({k for k in module_globals if k not in preexisting}, set())
        self.assertEqual(compiled(x), expected)

    @parametrize("wrap_top_frame", (False, True))
    def test_aot_compile_module_deserialize_refuses_a_lazy_module_forward(
        self, wrap_top_frame
    ):
        # For a module carrying _initialize_hook OptimizedModule._initialize
        # rebinds forward to the bound _call_lazy_check AFTER its wrap_top_frame
        # branch, so on either setting the forward is a bound method nothing hops
        # and get_traced_fn resolves to eval_frame's _call_lazy_check.
        mod = GlobalConfigModule()
        with torch._dynamo.config.patch(wrap_top_frame=wrap_top_frame):
            mod.forward = torch.compile(torch.nn.LazyLinear(8), backend="eager").forward
        lazy_check = torch._dynamo.eval_frame.OptimizedModule._call_lazy_check
        self.assertIs(mod.forward.__func__, lazy_check)
        self.assertIs(torch._dynamo.eval_frame.innermost_fn(mod.forward), mod.forward)
        self._assert_forward_refused(
            mod,
            torch._dynamo.eval_frame,
            "GlobalConfigModule.forward (method named "
            "OptimizedModule._call_lazy_check) resolves to "
            "OptimizedModule._call_lazy_check, whose globals are "
            "torch._dynamo.eval_frame's namespace, a torch module a load neither "
            "roots guards in nor seeds; bind the module's own forward, defined "
            "outside torch, as model.forward instead",
        )

    def test_aot_compile_module_deserialize_refuses_a_bound_module_call_forward(self):
        # mod.forward = other.__call__ binds the dispatch with no Dynamo wrapper
        # in front (a bound method survives nn.Module.__setattr__), and nothing
        # hops it. A capture of this shape records torch.nn.modules.module's
        # namespace, so resolving would agree with it; the namespace test trades
        # that for not seeding a process-wide torch namespace.
        mod = GlobalConfigModule()
        mod.forward = GlobalConfigModule().__call__
        self.assertIn("forward", vars(mod))
        self.assertIs(torch._dynamo.eval_frame.innermost_fn(mod.forward), mod.forward)
        self._assert_forward_refused(
            mod,
            torch.nn.modules.module,
            "GlobalConfigModule.forward (method named Module._wrapped_call_impl) "
            "resolves to Module._wrapped_call_impl, whose globals are "
            "torch.nn.modules.module's namespace, a torch module a load neither "
            "roots guards in nor seeds; bind the module's own forward, defined "
            "outside torch, as model.forward instead",
        )

    def test_aot_compile_module_deserialize_refuses_a_class_compiled_call_forward(
        self,
    ):
        # torch.compile(Cls) rebinds Cls.__call__ to a compile_wrapper, so an
        # instance's __call__ is a bound method whose __func__ owns eval_frame's
        # namespace: nothing hops a bound method, and the namespace test refuses
        # the wraps'd compile_wrapper get_traced_fn resolves, named by its code
        # since it copied _wrapped_call_impl's __qualname__.
        class Compiled(torch.nn.Module):
            def forward(self, x):
                return x

        torch.compile(Compiled, backend="eager")
        mod = GlobalConfigModule()
        mod.forward = Compiled().__call__
        wrapper = mod.forward.__func__
        self.assertIs(wrapper.__globals__, vars(torch._dynamo.eval_frame))
        self.assertIs(torch._dynamo.eval_frame.innermost_fn(mod.forward), mod.forward)
        code_name = getattr(wrapper.__code__, "co_qualname", wrapper.__code__.co_name)
        self._assert_forward_refused(
            mod,
            torch._dynamo.eval_frame,
            "GlobalConfigModule.forward (method named Module._wrapped_call_impl) "
            f"resolves to {code_name}, a functools.wraps'd wrapper over "
            "Module._wrapped_call_impl, whose globals are torch._dynamo.eval_frame's "
            "namespace, a torch module a load neither roots guards in nor seeds; "
            "bind the module's own forward, defined outside torch, as model.forward "
            "instead",
        )

    def test_aot_compile_module_deserialize_unwraps_a_compiled_forward(self):
        # torch.compile(mod.forward) bound back on the instance is a plain
        # function, eval_frame's wraps'd compile_wrapper: neither the
        # OptimizedModule unwrap nor the nn.Module refusal sees it, and the
        # torch-namespace rule refused its __globals__. The scope is the forward's.
        x = torch.randn(4, 8)
        data = self._two_input_global_guard_artifact(x)
        mod = GlobalConfigModule()
        expected = {}
        for mode in ("sum", "mean"):
            with _set_pooling(mode):
                expected[mode] = mod(x)
        mod.forward = torch.compile(mod.forward, backend="eager")
        eval_frame_globals = vars(torch._dynamo.eval_frame)
        preexisting = frozenset(eval_frame_globals)
        wrapper = torch.compile(mod, fullgraph=True, backend="eager")
        wrapper._load_aot_compiled_module(data)
        for mode in ("sum", "mean"):
            with _set_pooling(mode):
                self.assertEqual(wrapper(x), expected[mode])
        for result in wrapper.forward.compiled_results:
            self.assertIs(result._guard_scope, _GuardScope.SUPPLIED)
            self.assertIs(result._guard_globals, globals())
        self.assertEqual({k for k in eval_frame_globals if k not in preexisting}, set())

    @parametrize("depth", (1, 2))
    def test_aot_compile_module_deserialize_unwraps_a_wrap_inline_forward(self, depth):
        # Under config.wrap_top_frame torch.compile(mod.forward) wraps the bound
        # method in external_utils.wrap_inline's inner before minting its own
        # wrapper, so innermost_fn ends on inner, a wraps copy owning
        # external_utils' namespace; one hop per stacked compile reaches the
        # forward. The chain pin is what makes depth 2 measure the second hop.
        x = torch.randn(4, 8)
        data = self._two_input_global_guard_artifact(x)
        mod = GlobalConfigModule()
        expected = {}
        for mode in ("sum", "mean"):
            with _set_pooling(mode):
                expected[mode] = mod(x)
        external_utils_globals = vars(torch._dynamo.external_utils)
        preexisting = frozenset(external_utils_globals)
        bound_forward = mod.forward
        with torch._dynamo.config.patch(wrap_top_frame=True):
            for _ in range(depth):
                mod.forward = torch.compile(mod.forward, backend="eager")
        chain = []
        resolved = torch._dynamo.eval_frame.innermost_fn(mod.forward)
        while hasattr(resolved, "__wrapped__"):
            chain.append((resolved.__code__.co_name, resolved.__globals__))
            resolved = resolved.__wrapped__
        self.assertEqual(chain, [("inner", external_utils_globals)] * depth)
        self.assertEqual(resolved, bound_forward)
        wrapper = torch.compile(mod, fullgraph=True, backend="eager")
        wrapper._load_aot_compiled_module(data)
        for mode in ("sum", "mean"):
            with _set_pooling(mode):
                self.assertEqual(wrapper(x), expected[mode])
        for result in wrapper.forward.compiled_results:
            self.assertIs(result._guard_scope, _GuardScope.SUPPLIED)
            self.assertIs(result._guard_globals, globals())
        self.assertEqual(
            {k for k in external_utils_globals if k not in preexisting}, set()
        )

    def test_aot_compile_module_deserialize_hops_to_a_compiled_torch_forward(self):
        # The other way into wrap_inline needs no config: Dynamo wraps a forward
        # DEFINED under torch/ (trace_rules.check), nn.Linear's own included. The
        # hop lands on that forward, not on external_utils' inner, which the
        # reason shows by naming the namespace it owns; the artifact roots no
        # guard at a user global, so the fallback is silent and the call answers.
        x = torch.randn(4, 8)
        model = torch.compile(torch.nn.Linear(8, 3), fullgraph=True, backend="eager")
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        mod = torch.nn.Linear(8, 3)
        expected = mod(x)
        mod.forward = torch.compile(mod.forward, backend="eager")
        external_utils_globals = vars(torch._dynamo.external_utils)
        preexisting = frozenset(external_utils_globals)
        wrapper = torch.compile(mod, fullgraph=True, backend="eager")
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            wrapper._load_aot_compiled_module(data)
        self.assertEqual(wrapper(x), expected)
        (result,) = wrapper.forward.compiled_results
        self.assertIs(result._guard_scope, _GuardScope.RECONSTRUCTED)
        self.assertFalse(result._has_global_guards)
        self.assertIn(
            "Linear.forward (function named Linear.forward) resolves through a "
            "Dynamo wrapper to Linear.forward, whose globals are "
            "torch.nn.modules.linear's namespace",
            result._forward_not_resolved_reason,
        )
        self.assertEqual(
            {k for k in external_utils_globals if k not in preexisting}, set()
        )

    @parametrize("recursive", (True, False))
    def test_aot_compile_module_deserialize_unwraps_a_disabled_forward(self, recursive):
        # Both torch._dynamo.disable wrappers carry the pair innermost_fn follows;
        # the recursive one is eval_frame's function, the non-recursive one
        # external_utils' get_nonrecursive_disable_wrapper, which also satisfies
        # the __wrapped__ hop's predicate: recursive_False is served by either
        # mechanism and only recursive_True catches a load that drops innermost_fn.
        # The innermost_fn assertion records which mechanism gets there first.
        x = torch.randn(4, 8)
        data = self._two_input_global_guard_artifact(x)
        mod = GlobalConfigModule()
        expected = {}
        for mode in ("sum", "mean"):
            with _set_pooling(mode):
                expected[mode] = mod(x)
        bound_forward = mod.forward
        mod.forward = torch._dynamo.disable(mod.forward, recursive=recursive)
        wrapper_globals = mod.forward.__globals__
        owner = torch._dynamo.eval_frame if recursive else torch._dynamo.external_utils
        self.assertIs(wrapper_globals, vars(owner))
        preexisting = frozenset(wrapper_globals)
        self.assertEqual(mod.forward.__wrapped__, bound_forward)
        self.assertEqual(
            torch._dynamo.eval_frame.innermost_fn(mod.forward), bound_forward
        )
        wrapper = torch.compile(mod, fullgraph=True, backend="eager")
        wrapper._load_aot_compiled_module(data)
        for mode in ("sum", "mean"):
            with _set_pooling(mode):
                self.assertEqual(wrapper(x), expected[mode])
        for result in wrapper.forward.compiled_results:
            self.assertIs(result._guard_scope, _GuardScope.SUPPLIED)
            self.assertIs(result._guard_globals, globals())
        self.assertEqual({k for k in wrapper_globals if k not in preexisting}, set())

    def test_aot_compile_module_deserialize_takes_a_guard_globals_scope(self):
        # A caller who does not want the defining module's namespace read or
        # written to can pass its own dict. ParentWithChildModule roots a kept
        # guard at an __import_* alias, which makes the seeding observable: the
        # alias has to land in the supplied dict and nowhere else.
        x = torch.randn(4, 4)
        model = torch.compile(
            ParentWithChildModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        self._hide_leaked_dynamo_globals()

        g = globals()
        scope = dict(g)
        mod = ParentWithChildModule()
        expected = mod(x)
        compiled = AOTCompiledModel.deserialize(mod, data, guard_globals=scope)
        for result in compiled.compiled_results:
            self.assertIs(result._guard_scope, _GuardScope.SUPPLIED)
        self.assertEqual(
            {k for k in scope if k.startswith(_MINTED_PREFIXES)},
            {"__import_torch_dot_nn_dot_modules_dot_module"},
        )
        self.assertEqual({k for k in g if k.startswith(_MINTED_PREFIXES)}, set())
        self.assertEqual(compiled(x), expected)

    def test_aot_compile_module_deserialize_keeps_an_orig_mod_submodule(self):
        # _orig_mod is a registrable submodule name, so a duck-typed unwrap would
        # hand deserialize the child and run the parent's graph against the child's
        # parameters. Only an OptimizedModule is a wrapper.
        class Child(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.full((), 3.0))

            def forward(self, x):
                return x * self.scale

        class ParentWithOrigMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._orig_mod = Child()
                self.scale = torch.nn.Parameter(torch.full((), 2.0))

            def forward(self, x):
                return x * self.scale

        x = torch.ones(3)
        skip_guards = torch.compiler.skip_all_guards_unsafe
        model = torch.compile(
            ParentWithOrigMod(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": skip_guards},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()
        torch._dynamo.reset()
        fresh = ParentWithOrigMod()
        compiled = AOTCompiledModel.deserialize(fresh, data)
        self.assertIs(compiled.model, fresh)
        self.assertEqual(compiled(x), x * 2.0)

    def test_aot_compile_fn_guards_track_rebound_global(self):
        # Function artifacts get their guard scope from load_compiled_function's
        # f_globals, and the contract is that dict itself rather than a copy of
        # it: a global rebound after the load changes the guard's answer on the
        # next call, so a stale copy would keep serving the graph that matched at
        # load time.
        x = torch.randn(4, 8)
        expected = {}
        for mode in ("sum", "mean"):
            with _set_pool_mode(mode):
                expected[mode] = global_rebind_fn(x)
        self.assertNotEqual(expected["sum"].tolist(), expected["mean"].tolist())

        # f_globals below is this module's dict, which the capture leaks Dynamo's
        # generated globals into; hide them for the duration of the test, and
        # strip what it adds in cleanup. The load seeds nothing here -- the only
        # kept global guard is rooted at AOT_POOL_MODE, not at an alias or the
        # builtins-dict key.
        self._hide_leaked_dynamo_globals()
        with _set_pool_mode("sum"):
            compiled_fn = torch.compile(
                global_rebind_fn,
                fullgraph=True,
                backend="eager",
                options={"guard_filter_fn": keep_global_guards},
            ).aot_compile(((x,), {}))
        compiled_fn.save_compiled_function(self.path())

        torch._dynamo.reset()
        with _set_pool_mode("sum"):
            with open(self.path(), "rb") as f:
                loaded = torch.compiler.load_compiled_function(f, f_globals=globals())
            self.assertEqual(loaded(x), expected["sum"])
        with _set_pool_mode("mean"):
            with self.assertRaisesRegex(RuntimeError, "AOT_POOL_MODE"):
                loaded(x)

    def test_load_compiled_function_f_globals_guard_checks_the_tensor_type(self):
        # A kept TENSOR_MATCH compares the tensor's exact Python type before any
        # of its metadata, so matching dtype, shape, strides, device and
        # requires_grad is not enough for a rebind to pass: an nn.Parameter
        # substituted for the plain tensor the graph was traced with fails the
        # guard, where the plain tensor it replaces passes. Rebinding it in the
        # dict after the load is what needs this commit -- the parent built the
        # guard manager against a snapshot, so only a value already bound when
        # load_compiled_function ran was ever compared.
        def fn(x):
            return x * EPS

        x = torch.randn(3, 4)
        self._hide_leaked_dynamo_globals()
        compiled_fn = torch.compile(
            fn,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        ).aot_compile(((x,), {}))
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()

        scope = {"EPS": EPS.clone()}
        with open(self.path(), "rb") as f:
            loaded = torch.compiler.load_compiled_function(f, f_globals=scope)
        self.assertEqual(loaded(x), x * EPS)

        param = torch.nn.Parameter(EPS.clone(), requires_grad=False)
        self.assertEqual(param.dtype, scope["EPS"].dtype)
        self.assertEqual(param.stride(), scope["EPS"].stride())
        scope["EPS"] = param
        with self.assertRaisesRegex(
            RuntimeError,
            r"expected type of 'G\['EPS'\]' to be <class 'torch\.Tensor'>, "
            r"but found <class 'torch\.nn\.parameter\.Parameter'>",
        ):
            loaded(x)

    def test_load_compiled_function_f_globals_merges_for_the_bytecode(self):
        # The load-time half of the f_globals contract: the dict is merged OVER
        # the globals serialized with the artifact, so a name it omits still
        # resolves and a name it binds is what the graph computes with. That is
        # all a global no kept guard is rooted at ever gets -- the per-call
        # re-read covers the certified names only, and under the default filter
        # here there are none, so a rebind of the scope after the load is not
        # followed. It is here because the other half REPLACES the guard scope,
        # and the two are easy to conflate.
        def fn(x):
            return x * EPS

        x = torch.randn(3, 4)
        compiled_fn = torch.compile(fn, fullgraph=True, backend="eager").aot_compile(
            ((x,), {})
        )
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()

        with open(self.path(), "rb") as f:
            omitted = torch.compiler.load_compiled_function(f, f_globals={})
        self.assertEqual(omitted(x), x * EPS)

        live = torch.tensor(2.0)
        scope = {"EPS": live}
        with open(self.path(), "rb") as f:
            bound = torch.compiler.load_compiled_function(f, f_globals=scope)
        self.assertEqual(bound._live_global_names, ())
        self.assertEqual(bound(x), x * live)
        scope["EPS"] = torch.tensor(3.0)
        self.assertEqual(bound(x), x * live)

    def test_load_compiled_function_empty_f_globals_is_an_empty_guard_scope(self):
        # f_globals={} and omitting f_globals are different modes now: the empty
        # dict is a live scope with nothing in it, so every kept global guard
        # fails until the caller binds the name in the dict they still hold,
        # while omitting the argument resolves the guards against the scope
        # rebuilt from the artifact. Normalizing the empty dict to "no scope"
        # would make a dict live by reference only while it is non-empty, so a
        # caller who loads first and populates after would silently get the
        # rebuilt scope forever, with nothing raising to say so.
        def fn(x):
            return x * EPS

        x = torch.randn(3, 4)
        self._hide_leaked_dynamo_globals()
        compiled_fn = torch.compile(
            fn,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        ).aot_compile(((x,), {}))
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()

        with open(self.path(), "rb") as f:
            omitted = torch.compiler.load_compiled_function(f)
        self.assertEqual(omitted(x), x * EPS)

        scope: dict[str, object] = {}
        with open(self.path(), "rb") as f:
            loaded = torch.compiler.load_compiled_function(f, f_globals=scope)
        with self.assertRaisesRegex(RuntimeError, r"KeyError on G\['EPS'\]"):
            loaded(x)
        scope["EPS"] = EPS
        self.assertEqual(loaded(x), x * EPS)

    def test_load_compiled_function_f_globals_governs_a_cpp_shape_guard(self):
        # The carve-out both public texts state is for the LAMBDA form of a
        # symbolic-shape guard; captured with enable_cpp_symbolic_shape_guards
        # the guard goes out as C++ with the global's size as an operand, whose
        # manager hangs off the live globals dict like any other global guard's.
        # The C++ compile is not what this observes: the operand managers are
        # built before it is attempted, so this fails with no compiler too. The
        # config wraps the capture alone: the load reads what that capture
        # recorded and never this config.
        self._hide_leaked_dynamo_globals()
        # mark_dynamic writes its marking onto the tensor, and this one is a
        # module global that outlives the test.
        for name in (
            "_dynamo_dynamic_indices",
            "_dynamo_hint_overrides",
            "_specialize_on",
            "_has_dynamo_dim_marking",
        ):
            self.addCleanup(delattr, AOT_CPP_SHAPE_GLOBAL, name)
        torch._dynamo.mark_dynamic(AOT_CPP_SHAPE_GLOBAL, 0)

        def fn(x):
            return x + AOT_CPP_SHAPE_GLOBAL.sum(0)

        x = torch.randn(4)
        with torch._dynamo.config.patch(enable_cpp_symbolic_shape_guards=True):
            compiled_fn = torch.compile(
                fn, fullgraph=True, backend="eager"
            ).aot_compile(((x,), {}))
        shape_code_parts = load_guards_state(
            compiled_fn._artifacts.guards_state
        ).shape_code_parts
        self.assertFalse(shape_code_parts.python_fallback)
        self.assertEqual(
            [source.name for source in shape_code_parts.shape_env_sources],
            ["G['AOT_CPP_SHAPE_GLOBAL'].size()[0]"],
        )
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()

        scope: dict[str, object] = {}
        with open(self.path(), "rb") as f:
            loaded = torch.compiler.load_compiled_function(f, f_globals=scope)
        with self.assertRaisesRegex(
            RuntimeError, r"KeyError on G\['AOT_CPP_SHAPE_GLOBAL'\]"
        ):
            loaded(x)
        scope["AOT_CPP_SHAPE_GLOBAL"] = AOT_CPP_SHAPE_GLOBAL
        self.assertEqual(loaded(x), x + AOT_CPP_SHAPE_GLOBAL.sum(0))

    def test_load_compiled_function_f_globals_is_seeded_in_place(self):
        # f_globals is the guard scope now, so what _seed_guard_scope writes for
        # a kept guard rooted at a Dynamo-minted name lands in the caller's own
        # dict, with no CleanupHook to take it back out. keep_builtin_guards
        # keeps the BUILTIN_MATCH rooted at the builtins-dict key, which is what
        # makes the seeding run at all -- under the default filter no global
        # guard survives, so nothing is written.
        def fn(x):
            if isinstance(x, torch.Tensor):
                return x + 1
            return x

        x = torch.randn(3, 3)
        compiled_fn = torch.compile(
            fn,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_builtin_guards},
        ).aot_compile(((x,), {}))
        guards_state = load_guards_state(compiled_fn._artifacts.guards_state)
        builtins_key = guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
        self.assertTrue(builtins_key)
        compiled_fn.save_compiled_function(self.path())

        torch._dynamo.reset()
        scope: dict[str, object] = {"__name__": "caller_module"}
        before = set(scope)
        with open(self.path(), "rb") as f:
            loaded = torch.compiler.load_compiled_function(f, f_globals=scope)
        self.assertEqual(set(scope) - before, {builtins_key, "__builtins__"})
        self.assertEqual(loaded(x), fn(x))

    def test_load_compiled_function_f_globals_accepted_rebind_is_served(self):
        # A global a kept guard's own source IS is re-read from f_globals before
        # every call, so a rebind a kept TENSOR_MATCH accepts -- it compares
        # metadata, not values -- is what the graph computes with, not the value
        # the load snapshotted. Serving the snapshot instead would be a wrong
        # answer with nothing raising: the guard the swap satisfies certifies
        # exactly the metadata the graph was compiled for, so the new tensor is
        # the one it should read. test_..._guard_checks_the_tensor_type pins the
        # rejected end of the same swap. The Parameter no kept guard is rooted
        # at (keep_tensor_guards_unsafe drops TENSOR_MATCH on a Parameter) is
        # rebound the same way and must NOT be served, so the re-read covers
        # exactly the certified set and no more, as
        # test_aot_compile_module_rebind_after_load_is_served pins for the
        # module path.
        from torch._dynamo.source import get_global_source_name, GlobalSource

        global AOT_UNGUARDED_PARAM

        self._hide_leaked_dynamo_globals()
        AOT_UNGUARDED_PARAM = torch.nn.Parameter(torch.ones(3))
        load_time_param = AOT_UNGUARDED_PARAM

        def fn(x):
            return x * EPS + AOT_UNGUARDED_PARAM

        x = torch.randn(3)
        options = {"guard_filter_fn": torch.compiler.keep_tensor_guards_unsafe}
        compiled = torch.compile(fn, fullgraph=True, backend="eager", options=options)
        captured = compiled.aot_compile(((x,), {}))
        # The rebind being ACCEPTED has to be a guard's answer rather than an
        # absent one: both the served value and guard_check returning True also
        # hold for an artifact that kept no guard on G['EPS'] at all. And the
        # Parameter has to be one no kept guard reaches, or its leg would hold
        # for a guarded global that failed to be re-read.
        output_graph = load_guards_state(captured._artifacts.guards_state).output_graph
        sources = [guard.originating_source for guard in output_graph.guards]
        self.assertIn(GlobalSource("EPS"), sources)
        roots = {get_global_source_name(source) for source in sources}
        self.assertNotIn("AOT_UNGUARDED_PARAM", roots)
        captured.save_compiled_function(self.path())
        torch._dynamo.reset()

        # Not EPS.clone(): x * 1e-7 sits inside assertEqual's tolerance of zero,
        # so the first call could not tell the scope's value from the serialized.
        load_time = torch.tensor(3.0)
        scope = {"EPS": load_time, "AOT_UNGUARDED_PARAM": load_time_param}
        with open(self.path(), "rb") as f:
            loaded = torch.compiler.load_compiled_function(f, f_globals=scope)
        self.assertEqual(loaded(x), x * load_time + load_time_param)

        rebound = torch.tensor(5.0)
        self.assertEqual(rebound.dtype, load_time.dtype)
        self.assertEqual(rebound.shape, load_time.shape)
        self.assertEqual(rebound.device, load_time.device)
        self.assertEqual(rebound.requires_grad, load_time.requires_grad)
        self.assertNotEqual(rebound.item(), load_time.item())
        scope["EPS"] = rebound
        scope["AOT_UNGUARDED_PARAM"] = torch.nn.Parameter(torch.full((3,), 9.0))
        self.assertEqual(loaded(x), x * rebound + load_time_param)

    def test_load_compiled_function_graph_rebind_of_a_guarded_global_is_re_read(self):
        # A forward that rebinds a guarded global has Dynamo replay the store as
        # a STORE_GLOBAL into the bytecode's own globals, never into f_globals,
        # which is the dict the guards read and just certified. The re-read
        # takes the scope's value back before the next call, so the store does
        # not accumulate across calls: a value no guard checked is not served
        # in place of the one they passed. Eager stores into the dict its guards
        # read; here that dict is the caller's, and the graph leaves it alone.
        # Deliberately so: a forward that accumulates this way serves the scope's
        # value on every call where eager counts up, because the alternative --
        # leaving a stored name out of the re-read -- serves the stored value
        # after a rebind of the scope the guards accepted, the stale read the
        # re-read exists to remove. The tail pins that rebind being followed.
        self.addCleanup(globals().__setitem__, "EPS", EPS)

        def fn(x):
            global EPS
            y = x * EPS
            EPS = EPS * 2
            return y

        x = torch.randn(3, 4)
        self._hide_leaked_dynamo_globals()
        compiled_fn = torch.compile(
            fn,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        ).aot_compile(((x,), {}))
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()

        # Not EPS.clone(): doubling 1e-7 stays inside assertEqual's tolerance.
        load_time = torch.tensor(3.0)
        scope = {"EPS": load_time}
        with open(self.path(), "rb") as f:
            loaded = torch.compiler.load_compiled_function(f, f_globals=scope)
        self.assertEqual(loaded._live_global_names, ("EPS",))
        self.assertEqual(loaded(x), x * load_time)
        # The store landed in the bytecode's globals and left the scope alone.
        self.assertIs(scope["EPS"], load_time)
        self.assertEqual(loaded.fn.__globals__["EPS"], load_time * 2)
        served = loaded(x)
        self.assertEqual(served, x * load_time)
        self.assertNotEqual(served.tolist(), (x * load_time * 2).tolist())
        self.assertIs(scope["EPS"], load_time)
        # A rebind the guards accept after the store is what the next call
        # computes with, not the stored value; both alternatives serve 2x or 4x.
        rebound = torch.tensor(5.0)
        scope["EPS"] = rebound
        served = loaded(x)
        self.assertEqual(served, x * rebound)
        self.assertNotEqual(served.tolist(), (x * load_time * 2).tolist())
        self.assertIs(scope["EPS"], rebound)
        self.assertEqual(loaded.fn.__globals__["EPS"], rebound * 2)

    def test_load_compiled_function_sub_path_global_is_not_re_read(self):
        # The function-path pin of the carve-out
        # test_aot_compile_module_sub_path_global_is_not_read_live pins for the
        # module path: the re-read covers a global a kept guard's own source IS,
        # not a container a guard reaches only through a sub-path, whose other
        # members nothing certifies. EPS is the former and is followed;
        # AOT_NESTED_GLOBAL is the latter and keeps its load-time value even
        # though the kept guard on ["a"] passes on the rebound container.
        from torch._dynamo.source import get_global_source_name, GlobalSource

        global AOT_NESTED_GLOBAL

        self._hide_leaked_dynamo_globals()
        AOT_NESTED_GLOBAL = {
            "a": torch.randn(3),
            "b": torch.nn.Parameter(torch.ones(3)),
        }

        def fn(x):
            return x * EPS + AOT_NESTED_GLOBAL["a"] + AOT_NESTED_GLOBAL["b"]

        x = torch.randn(3)
        options = {"guard_filter_fn": torch.compiler.keep_tensor_guards_unsafe}
        compiled = torch.compile(fn, fullgraph=True, backend="eager", options=options)
        captured = compiled.aot_compile(((x,), {}))
        output_graph = load_guards_state(captured._artifacts.guards_state).output_graph
        sources = [guard.originating_source for guard in output_graph.guards]
        # Armed: a kept guard does reach AOT_NESTED_GLOBAL, but only through a
        # sub-path; without this the tail would hold for a container no guard
        # touches at all.
        roots = {get_global_source_name(source) for source in sources}
        self.assertIn("AOT_NESTED_GLOBAL", roots)
        self.assertNotIn(GlobalSource("AOT_NESTED_GLOBAL"), sources)
        captured.save_compiled_function(self.path())
        torch._dynamo.reset()

        load_time_eps = torch.tensor(3.0)
        load_time = {"a": torch.randn(3), "b": torch.nn.Parameter(torch.ones(3))}
        scope = {"EPS": load_time_eps, "AOT_NESTED_GLOBAL": load_time}
        with open(self.path(), "rb") as f:
            loaded = torch.compiler.load_compiled_function(f, f_globals=scope)
        self.assertEqual(loaded._live_global_names, ("EPS",))
        expected = x * load_time_eps + load_time["a"] + load_time["b"]
        self.assertEqual(loaded(x), expected)

        # ["a"] keeps its metadata, so its kept guard passes on the rebound
        # container; ["b"] changes dtype, which nothing checks.
        rebound_eps = torch.tensor(5.0)
        rebound = {
            "a": torch.randn(3),
            "b": torch.nn.Parameter(torch.ones(3, dtype=torch.float64)),
        }
        scope["EPS"] = rebound_eps
        scope["AOT_NESTED_GLOBAL"] = rebound
        actual = loaded(x)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual, x * rebound_eps + load_time["a"] + load_time["b"])
        live = x * rebound_eps + rebound["a"] + rebound["b"]
        self.assertNotEqual(actual.tolist(), live.tolist())

    def test_load_compiled_function_opted_out_rebind_is_served_unchecked(self):
        # disable_guard_check() does not stop the per-call re-read: the names a
        # kept guard's own source IS are recorded by the load, before the opt-out
        # flips anything, so an opted-out artifact goes on serving whatever
        # f_globals binds -- including the nn.Parameter the kept TENSOR_MATCH
        # rejects while the check is on, which is the same swap
        # test_..._guard_checks_the_tensor_type pins at the other end. That is
        # the opt-out being unsafe as documented rather than an accident:
        # holding the load-time value here would be no more checked, only stale.
        # The tail pins the other half of that entry: a name the scope no longer
        # binds is skipped, not reverted, so the graph keeps whatever the
        # bytecode's globals last held.
        def fn(x):
            return x * EPS

        x = torch.randn(3, 4)
        self._hide_leaked_dynamo_globals()
        compiled_fn = torch.compile(
            fn,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        ).aot_compile(((x,), {}))
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()

        # Not EPS.clone(): see test_..._f_globals_accepted_rebind_is_served.
        load_time = torch.tensor(3.0)
        scope = {"EPS": load_time}
        with open(self.path(), "rb") as f:
            loaded = torch.compiler.load_compiled_function(f, f_globals=scope)
        self.assertEqual(loaded(x), x * load_time)

        rebound = torch.nn.Parameter(torch.tensor(5.0), requires_grad=False)
        self.assertNotEqual(rebound.item(), load_time.item())
        scope["EPS"] = rebound
        with self.assertRaisesRegex(RuntimeError, "GuardManager check failed"):
            loaded(x)
        loaded.disable_guard_check()
        served = loaded(x)
        self.assertEqual(served, x * rebound)
        self.assertNotEqual(served.tolist(), (x * load_time).tolist())

        del scope["EPS"]
        served = loaded(x)
        self.assertEqual(served, x * rebound)
        self.assertNotEqual(served.tolist(), (x * load_time).tolist())
        self.assertNotEqual(served.tolist(), (x * EPS).tolist())

    def test_aot_compile_fn_missing_global_hint_names_f_globals(self):
        # Loaded without f_globals, a function artifact's guards resolve against
        # the scope rebuilt from the serialized bytecode, which carries the
        # globals the graph lifted and the module aliases it imports -- and a
        # global the tracing branch specialized on is neither. Nothing this
        # module defines can be reached from there, so the failure has to point
        # at f_globals rather than at a name to define. (No need to unbind
        # AOT_POOL_MODE here: the guard cannot see this module's globals either
        # way.)
        x = torch.randn(4, 8)
        self._hide_leaked_dynamo_globals()
        with _set_pool_mode("sum"):
            compiled_fn = torch.compile(
                global_rebind_fn,
                fullgraph=True,
                backend="eager",
                options={"guard_filter_fn": keep_global_guards},
            ).aot_compile(((x,), {}))
        compiled_fn.save_compiled_function(self.path())

        torch._dynamo.reset()
        with open(self.path(), "rb") as f:
            loaded = torch.compiler.load_compiled_function(f)
        self.assertIs(loaded._guard_scope, _GuardScope.RECONSTRUCTED)
        with self.assertRaises(RuntimeError) as ctx:
            loaded(x)
        message = str(ctx.exception)
        self.assertIn("a complete live scope", message)
        # The scope that binds the name is the one that DEFINED the function,
        # which in the cross-process case AOT compile exists for is not the
        # module doing the loading, so the advice has to name that module's own
        # vars(mod) rather than the loader's globals() -- and spell the argument,
        # since a bare vars() is the caller's locals().
        self.assertIn("vars(mod) for the module mod that defined the function", message)
        # Not the advice for a live scope the caller could add the name to.
        self.assertNotIn("define it there", message)

    def test_aot_compile_fn_extra_globals_only_load_is_still_reconstructed(self):
        # deserialize's positional f_globals is _extra_globals, which
        # forward_callable MERGES into the scope it rebuilds -- so this load's
        # guards do read the names the caller passed, and the state is still
        # RECONSTRUCTED. That is the accurate state, not a gap: the merge copies
        # the caller's dict, so the supplied-scope advice to define the name in
        # it would be false here, while the advice this state gives -- load with
        # an f_globals= that carries the name -- is what the second half proves
        # works.
        x = torch.randn(4, 8)
        self._hide_leaked_dynamo_globals()
        with _set_pool_mode("sum"):
            compiled_fn = torch.compile(
                global_rebind_fn,
                fullgraph=True,
                backend="eager",
                options={"guard_filter_fn": keep_global_guards},
            ).aot_compile(((x,), {}))
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            data = f.read()

        torch._dynamo.reset()
        partial: dict[str, object] = {"AOT_UNRELATED_NAME": 1}
        loaded = AOTCompiledFunction.deserialize(data, f_globals=partial)
        self.assertIs(loaded._guard_scope, _GuardScope.RECONSTRUCTED)
        # The caller's names really are in the dict the guards read, which is
        # why the hint cannot say the scope holds only what the graph lifted.
        self.assertIn("AOT_UNRELATED_NAME", loaded.fn.__globals__)
        self.assertNotIn("AOT_POOL_MODE", loaded.fn.__globals__)
        with self.assertRaises(RuntimeError) as ctx:
            loaded(x)
        message = str(ctx.exception)
        self.assertIn("KeyError on G['AOT_POOL_MODE']", message)
        self.assertIn("missing from the scope rebuilt from the artifact", message)
        # The clause the parent had here, "which holds only the globals the graph
        # lifted", was false for this shape.
        self.assertNotIn("holds only the globals", message)
        # Binding the name in the dict the caller passed does not reach these
        # guards, so the supplied-scope wording would misdirect this reader.
        partial["AOT_POOL_MODE"] = "sum"
        with self.assertRaises(RuntimeError):
            loaded(x)

        # What the hint does tell them to do resolves the guard.
        torch._dynamo.reset()
        complete = AOTCompiledFunction.deserialize(
            data, f_globals={"AOT_POOL_MODE": "sum"}
        )
        self.assertEqual(complete(x), x.sum(1))

    def test_aot_compile_fn_missing_global_hint_names_the_supplied_scope(self):
        # With a live scope supplied, the same failure means the opposite of the
        # reconstructed-scope one above: the caller already passed a scope, and
        # what is missing is the name in it.
        x = torch.randn(4, 8)
        self._hide_leaked_dynamo_globals()
        with _set_pool_mode("sum"):
            compiled_fn = torch.compile(
                global_rebind_fn,
                fullgraph=True,
                backend="eager",
                options={"guard_filter_fn": keep_global_guards},
            ).aot_compile(((x,), {}))
        compiled_fn.save_compiled_function(self.path())

        torch._dynamo.reset()
        g = globals()
        saved = g.pop("AOT_POOL_MODE")
        try:
            with open(self.path(), "rb") as f:
                loaded = torch.compiler.load_compiled_function(f, f_globals=g)
            with self.assertRaises(RuntimeError) as ctx:
                loaded(x)
            message = str(ctx.exception)
            # Assert on a phrase unique to the supplied-scope wording: "define
            # it there" is in the captured-scope advice too, so asserting on
            # that alone would pass either way.
            self.assertIn("the live scope this artifact was loaded against", message)
            self.assertNotIn("traced in", message)
        finally:
            g["AOT_POOL_MODE"] = saved

        # An empty dict is still a scope the caller supplied. The state is
        # settled on whether f_globals was passed, not on whether it has names
        # in it, so this caller gets the advice for a scope they can add the
        # name to rather than being sent to supply the scope they just passed.
        torch._dynamo.reset()
        with open(self.path(), "rb") as f:
            empty_loaded = torch.compiler.load_compiled_function(f, f_globals={})
        self.assertIs(empty_loaded._guard_scope, _GuardScope.SUPPLIED)
        with self.assertRaises(RuntimeError) as ctx:
            empty_loaded(x)
        self.assertIn(
            "the live scope this artifact was loaded against", str(ctx.exception)
        )

    def test_aot_compile_fn_missing_nested_key_gets_no_missing_global_hint(self):
        # The global itself resolved and only a key inside it is absent, so
        # there is no missing global to advise about. Matching the verbose code
        # part whole is what tells the two apart: any substring match sees the
        # G['GLOBAL_POOLING_CONFIG'] prefix and fires.
        x = torch.randn(4, 8)
        # f_globals below is this module's dict, which the capture leaks Dynamo's
        # generated globals into; hide them for the duration of the test, and
        # strip what it adds in cleanup. The load seeds nothing here -- every
        # kept global guard is rooted at GLOBAL_POOLING_CONFIG, not at an alias
        # or the builtins-dict key.
        self._hide_leaked_dynamo_globals()
        compiled_fn = torch.compile(
            global_config_fn,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        ).aot_compile(((x,), {}))
        compiled_fn.save_compiled_function(self.path())

        torch._dynamo.reset()
        g = globals()
        saved = GLOBAL_POOLING_CONFIG.pop("pooling")
        try:
            with open(self.path(), "rb") as f:
                loaded = torch.compiler.load_compiled_function(f, f_globals=g)
            self.assertIs(loaded._guard_scope, _GuardScope.SUPPLIED)
            with self.assertRaises(RuntimeError) as ctx:
                loaded(x)
        finally:
            GLOBAL_POOLING_CONFIG["pooling"] = saved
        message = str(ctx.exception)
        self.assertIn("KeyError on G['GLOBAL_POOLING_CONFIG']['pooling']", message)
        self.assertNotIn("a guarded global is missing", message)

    def test_aot_compile_fn_missing_global_hint_names_the_tracing_scope(self):
        # A never-serialized artifact's guards hold the globals they were traced
        # against BY REFERENCE, and _guard_globals is None there too -- so a state
        # derived from _guard_globals at the point of failure could not tell this
        # apart from a load that supplied no scope, and would send the user off to
        # find a scope carrying a name that is missing from a dict they already
        # own. The advice is the one a live scope gets, and the second half proves
        # it is true rather than merely present: the name deleted after capture,
        # defined again in that same dict, makes the guard resolve.
        x = torch.randn(4, 8)
        self._hide_leaked_dynamo_globals()
        compiled_fn = torch.compile(
            global_rebind_fn,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        ).aot_compile(((x,), {}))
        self.assertIs(compiled_fn._guard_scope, _GuardScope.CAPTURED)
        self.assertIsNone(compiled_fn._guard_globals)

        g = globals()
        saved = g.pop("AOT_POOL_MODE")
        try:
            with self.assertRaises(RuntimeError) as ctx:
                compiled_fn(x)
            message = str(ctx.exception)
            self.assertIn("KeyError on G['AOT_POOL_MODE']", message)
            self.assertIn("the module the compiled function was traced in", message)
            self.assertNotIn("a complete live scope", message)
            # The advice continues the last line of str(GuardDebugInfo), which
            # ends in a newline: appending straight to it starts a line with a
            # space.
            self.assertIn(") -- a guarded global is missing", message)
            g["AOT_POOL_MODE"] = "sum"
            self.assertEqual(compiled_fn(x), x.sum(1))
        finally:
            g["AOT_POOL_MODE"] = saved

    def test_missing_global_hint_skips_a_name_dynamo_minted(self):
        # The strings are the verbose code part a guard tree reports for a
        # missing global, as the tests above read it off a real failure.
        self.assertTrue(_names_a_missing_global("KeyError on G['AOT_POOL_MODE']"))
        # A name Dynamo minted is not one the caller wrote, so "define it" is
        # advice they cannot act on: a load seeds each minted name a kept guard
        # is rooted at, and a KeyError on one reports a gap in that seeding.
        self.assertFalse(
            _names_a_missing_global("KeyError on G['__builtins_dict___0']")
        )
        self.assertFalse(
            _names_a_missing_global("KeyError on G['__import_torch_dot_nn']")
        )
        # The key an inlined frame whose globals belong to no module is guarded
        # through: it embeds id() of a dict in the tracing process, so no
        # module's vars() in a loading process holds it.
        self.assertFalse(
            _names_a_missing_global("KeyError on G['___unnamed_scope_140234512_c0']")
        )
        # A user name that merely starts with underscores is still theirs.
        self.assertTrue(_names_a_missing_global("KeyError on G['__my_config']"))

    def test_aot_compile_fn_missing_unnamed_scope_gets_no_missing_global_hint(self):
        # A load rebuilds no ___unnamed_scope_<id>_c<n> key, so an inlined frame
        # guarded through one reaches __call__ as exactly the KeyError shape the
        # hint fires on -- and the name embeds id() of a dict in the tracing
        # process, so the reconstructed-scope advice, which asks for the defining
        # module's vars(), names a scope that cannot hold it.
        x = torch.randn(4, 8)
        self._hide_leaked_dynamo_globals()
        compiled_fn = torch.compile(
            calls_into_an_unnamed_scope,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        ).aot_compile(((x,), {}))
        compiled_fn.save_compiled_function(self.path())

        torch._dynamo.reset()
        with open(self.path(), "rb") as f:
            loaded = torch.compiler.load_compiled_function(f)
        self.assertIs(loaded._guard_scope, _GuardScope.RECONSTRUCTED)
        with self.assertRaises(RuntimeError) as ctx:
            loaded(x)
        message = str(ctx.exception)
        self.assertIn("KeyError on G['___unnamed_scope", message)
        self.assertNotIn("a guarded global is missing", message)

    def test_repr_of_an_artifact_whose_post_init_raised(self):
        # fn is a field with no default, so leaving it in the generated __repr__
        # and __eq__ makes both raise AttributeError on an instance
        # __post_init__ abandoned -- and check_compatibility abandons one on a
        # version mismatch. Anything rendering frame locals while formatting
        # that traceback (pytest --showlocals) would then report the
        # AttributeError instead of the mismatch.
        def fn(x):
            return x + 1

        x = torch.randn(4, 8)
        compiled_fn = torch.compile(fn, fullgraph=True, backend="eager").aot_compile(
            ((x,), {})
        )
        artifacts = compiled_fn._artifacts
        stale = dataclasses.replace(
            artifacts,
            system_info=dataclasses.replace(
                artifacts.system_info, torch_version="0.0.0-not-this-one"
            ),
        )
        # Caught by hand rather than with assertRaises: that context manager
        # hands back an exception whose traceback it has already dropped, and the
        # instance under construction is only reachable through it.
        abandoned = None
        try:
            AOTCompiledFunction(_artifacts=stale)
        except RuntimeError as e:
            self.assertIn("different PyTorch version", str(e))
            tb = e.__traceback__
            while tb is not None:
                if tb.tb_frame.f_code.co_name == "__post_init__":
                    abandoned = tb.tb_frame.f_locals["self"]
                tb = tb.tb_next
        self.assertIsInstance(abandoned, AOTCompiledFunction)
        self.assertIn("AOTCompiledFunction", repr(abandoned))
        self.assertEqual(abandoned, abandoned)

    def test_aot_compile_module_absent_global_fails_guard(self):
        # A guarded global the loading process does not have has to fail the
        # guard. Falling back to the serialized scope would make the guard a
        # no-op checked against capture-time state, which is the silent failure
        # mode -- the loud one is recoverable on the serving machine. A module
        # load takes no f_globals=: eval_frame's _load_aot_compiled_module takes
        # only the bytes, though deserialize still takes guard_globals=. The
        # hint names the instance's forward whenever the scope model.forward
        # resolves to is the dict the guards hold, as it is for a load that
        # passed no guard_globals=, resolved forward, and is not rebound after
        # the load -- the rebound sibling's included, since it rebinds BEFORE
        # loading -- so the substring asserted below fits that sibling too;
        # what pins this non-rebound load is the recovery arm at the end --
        # defining the name in globals() serves the call here, where the same
        # binding in test_no_match_message_hint_covers_a_rebound_forward still
        # raises.
        x = torch.randn(3, 3)
        self._hide_leaked_dynamo_globals()
        model = torch.compile(
            HermeticModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        data = model._save_aot_compiled_module()

        torch._dynamo.reset()
        g = globals()
        saved = g.pop("AOT_HERMETIC_WEIGHT")
        try:
            # No filter on the reload: the load installs the artifact's own
            # guards and nothing is captured here, so the loading process's
            # filter has no say in what a loaded artifact checks.
            reloaded = torch.compile(HermeticModule(), fullgraph=True, backend="eager")
            reloaded._load_aot_compiled_module(data)
            with self.assertRaises(RuntimeError) as ctx:
                reloaded(x)
            message = str(ctx.exception)
            self.assertIn("No AOT compiled graph matched", message)
            self.assertIn("[0] KeyError on G['AOT_HERMETIC_WEIGHT']", message)
            self.assertIn(
                "the globals of the function this HermeticModule instance's forward "
                "resolves to",
                message,
            )
            # The advice is not something the report decides for a missing
            # global: it closes the report for any guard failure, this one
            # included, and asserting it pins that the hint stays an addition to
            # it rather than a replacement. A missing global is still the
            # mismatch the advice is for -- the report cannot know whether a new
            # ModelInput would read the absent name; one captured for a branch
            # that does not is served with the name still absent.
            self.assertIn("Add a ModelInput", message)
            # Taking the advice restores dispatch AND the value: the guards
            # read the live scope, so the new binding passes the kept
            # TENSOR_MATCH -- which checks metadata, not values -- and every
            # call re-reads the globals those guards' own sources are, whether or
            # not the scope bound them at load. A name armed only by what the scope
            # held then would leave this call computing with the value
            # serialized with the artifact, which no guard ever compared.
            g["AOT_HERMETIC_WEIGHT"] = saved * 2
            self.assertEqual(reloaded(x), x @ (saved * 2))
        finally:
            g["AOT_HERMETIC_WEIGHT"] = saved

    def test_aot_compile_module_import_alias_guard_loads_across_processes(self):
        # The real deployment shape: the artifact is captured by a process that
        # never runs here, so the __import_* aliases its guards are rooted at
        # have to be seeded into this module's globals by the load itself.
        path = self.path()
        _run_in_subprocess(
            functools.partial(
                _subprocess_save_child_module_artifact, path, keep_global_guards
            )
        )
        # This process's module is freshly initialized and no state dict crosses
        # from the capturing process, so the closing assertion also pins that
        # dispatch reads its LIVE parameters rather than baked-in tensors.
        mod = ParentWithChildModule()
        model = torch.compile(mod, fullgraph=True, backend="eager")
        self._hide_leaked_dynamo_globals()
        g = globals()
        self.assertEqual({k for k in g if k.startswith("__import_")}, set())
        with open(path, "rb") as f:
            model._load_aot_compiled_module(f.read())
        (result,) = model.forward.compiled_results
        import_sources = result._artifacts.runtime_env.import_sources
        guards_state = load_guards_state(result._artifacts.guards_state)
        # Assert against what the load actually wrote into this live namespace,
        # not against the artifact re-filtered by the gate under test. Read the
        # guarded alias off the serialized global_scope, pruned to exactly the
        # guarded names, rather than off a literal or off import_sources' order.
        guarded = set(import_sources) & set(guards_state.output_graph.global_scope)
        self.assertLess(len(guarded), len(import_sources))
        # Named before the unpacking below: an nn.Module internal that adds or
        # drops a guarded alias would otherwise report a bare ValueError.
        self.assertEqual(len(guarded), 1, f"expected one guarded alias: {guarded}")
        (alias,) = guarded
        self.assertEqual({k for k in g if k.startswith("__import_")}, guarded)
        self.assertIs(g[alias], importlib.import_module(import_sources[alias]))
        # keep_global_guards drops BUILTIN_MATCH, so no kept guard is rooted at
        # the recorded builtins key and the load must leave it unbound.
        builtins_key = guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
        self.assertTrue(builtins_key)
        self.assertNotIn(builtins_key, g)
        x = torch.randn(4, 4)
        self.assertEqual(model(x), mod(x))

    def test_load_then_compile_survives_baked_in_global_collision(self):
        # output_graph.install_global's retry loop reached across processes: the
        # name the mint collides with is one a real load seeded out of another
        # process's artifact, where the sibling in-process test can only pre-bind
        # a sentinel string to it.
        path = self.path()
        _run_in_subprocess(
            functools.partial(
                _subprocess_save_child_module_artifact, path, keep_builtin_guards
            )
        )
        _run_in_subprocess(functools.partial(_subprocess_load_then_compile, path))

    @parametrize("mint_site", ("install_global", "resume_function", "comprehension"))
    def test_mint_skips_a_name_baked_in_by_another_process(self, mint_site):
        # A load in a fresh process binds names its own counter is still behind:
        # a captured __builtins_dict___N key, and the __resume_at_* globals
        # CompilePackage.install() re-installs. In process, unique_id is already
        # ahead of any baked-in index, so rewinding the counter and pre-binding
        # the name the next mint produces puts each site in that same position.
        # The name has to come from the compile: a hardcoded index goes green
        # covering nothing as soon as anything else burns an id first, because
        # the skip loop then never runs.
        import itertools

        from torch._dynamo import bytecode_transformation

        def fullgraph_fn(x):
            return x + len(x)

        def graph_breaking_fn(x):
            y = x + 1
            torch._dynamo.graph_break()
            return y * 2

        def comprehension_fn(x):
            y = x + 1
            return y, [torch._dynamo.graph_break() or i for i in range(2)]

        if mint_site == "comprehension" and sys.version_info < (3, 12):
            # Comprehensions are inlined, and so can break, only from 3.12 on.
            self.skipTest("inlined comprehensions are 3.12+")

        fn, fullgraph, minted_prefix = {
            "install_global": (fullgraph_fn, True, "__builtins_dict__"),
            "resume_function": (graph_breaking_fn, False, "__resume_at"),
            "comprehension": (comprehension_fn, False, "__comprehension_"),
        }[mint_site]
        self._hide_leaked_dynamo_globals()
        g = globals()
        taken = "taken by another process"
        x = torch.randn(3)
        expected = fn(x)

        def compile_with_taken_names(*names):
            # A fresh counter mints the same sequence of names on every run, so
            # each phase learns the name the next one pre-binds.
            torch._dynamo.reset()
            for k in [k for k in list(g) if k.startswith(_MINTED_PREFIXES)]:
                # reset() leaves the hook that installed this name still
                # owning it in _cleanup_owners, and a hook fires whenever its
                # code object is collected -- after the bind below, that would
                # pop this phase's sentinel.
                CleanupHook.disown(g, k)
                del g[k]
            for name in names:
                g[name] = taken
            with patch.object(
                bytecode_transformation, "_unique_id_counter", itertools.count()
            ):
                compiled = torch.compile(fn, fullgraph=fullgraph, backend="eager")
                # A later call must be SERVED rather than recompiled: for
                # install_global the builtin guards have to hit through the key
                # the skip landed on, in a module dict that still carries the
                # other process's binding. Comparing results alone would not see
                # a guard that missed and recompiled to the same answer.
                self.assertEqual(compiled(x), expected)
                with torch._dynamo.config.patch(error_on_recompile=True):
                    self.assertEqual(compiled(x), expected)
            return [
                k
                for k in g
                if k.startswith(minted_prefix) and not isinstance(g[k], str)
            ]

        # Two taken names, not one: the skip has to advance past both, so a retry
        # that fires only once still hands the install a bound name and raises.
        (minted,) = compile_with_taken_names()
        (skipped_to,) = compile_with_taken_names(minted)
        (installed,) = compile_with_taken_names(minted, skipped_to)
        self.assertNotIn(installed, (minted, skipped_to))
        self.assertEqual(g[minted], taken)
        self.assertEqual(g[skipped_to], taken)
        # Every name here ends in its counter index, and each retry must step
        # that counter rather than decorate the name: the other process mints
        # from a counter too, so a name reached any other way is not one it will
        # skip past in turn. The assertions above hold either way.
        indexes = [int(n.rpartition("_")[2]) for n in (minted, skipped_to, installed)]
        start = indexes[0]
        self.assertEqual(indexes, [start, start + 1, start + 2])

    def test_kept_builtin_match_guard_reads_the_seeded_builtins_dict(self):
        # keep_global_guards drops BUILTIN_MATCH (it derives ID_MATCH), and the
        # seeding only writes a name some kept guard is rooted at, so under that
        # filter the builtins half never runs. Keep the guard instead -- the
        # serializer accepts it -- and the loaded artifact really evaluates a guard
        # rooted at G['__builtins_dict___N'] against a scope that has neither that
        # key, the __import_* aliases, nor even __builtins__.
        def fn(x):
            if isinstance(x, torch.Tensor):
                return x + 1
            return x

        x = torch.randn(3, 3)
        compiled_fn = torch.compile(
            fn,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_builtin_guards},
        ).aot_compile(((x,), {}))
        guards_state = load_guards_state(compiled_fn._artifacts.guards_state)
        builtins_key = guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
        self.assertTrue(builtins_key)
        kept = [str(g) for g in guards_state.output_graph.guards]
        self.assertTrue(
            any("BUILTIN_MATCH" in g for g in kept), f"no BUILTIN_MATCH kept: {kept}"
        )
        compiled_fn.save_compiled_function(self.path())

        torch._dynamo.reset()
        scope: dict[str, object] = {"__name__": "fresh_module"}
        with open(self.path(), "rb") as f:
            loaded = AOTCompiledFunction.deserialize(f.read(), guard_globals=scope)
        self.assertEqual(loaded(x), fn(x))
        self.assertIn(builtins_key, scope)
        self.assertIn("__builtins__", scope)

        # The seeded __builtins__ is the LIVE dict, not a snapshot: the kept
        # BUILTIN_MATCH is an ID_MATCH on G[builtins_key]['isinstance'], so a copy
        # would go on passing after the builtin it read is rebound. The stand-in
        # below behaves exactly like the real isinstance and differs only in
        # identity, so a guard that still passes is reading a stale dict.
        live = scope["__builtins__"] is builtins.__dict__
        self.assertTrue(live, "seeded a snapshot instead of the live builtins dict")
        real_isinstance = builtins.isinstance
        try:
            builtins.isinstance = lambda obj, cls: real_isinstance(obj, cls)
            self.assertFalse(loaded.guard_check(x))
        finally:
            builtins.isinstance = real_isinstance
        self.assertTrue(loaded.guard_check(x))

        # And with no scope supplied: the guards resolve against the scope
        # rebuilt from the artifact, which carries only what the graph lifted, so
        # this path needs the same seeding -- it is not a caller courtesy.
        torch._dynamo.reset()
        with open(self.path(), "rb") as f:
            default_loaded = AOTCompiledFunction.deserialize(f.read())
        self.assertEqual(default_loaded(x), fn(x))

    @parametrize("keeps_global_guard", (False, True))
    def test_load_seeds_no_name_no_guard_is_rooted_at(self, keeps_global_guard):
        # The seeding writes into a scope that may be a user module's live
        # namespace and installs no CleanupHook, so it is gated per name on a kept
        # guard being rooted at that name. Neither artifact here roots one at the
        # builtins-dict key or at an alias, so both must leave the scope alone --
        # including the keeps_global_guard artifact, whose kept guard is rooted at
        # AOT_POOL_MODE. A gate that only asks whether SOME guard is rooted at SOME
        # global admits that one and writes the builtins key for nothing. The
        # artifact records the key either way, so nothing else stops it.
        x = torch.randn(4, 8)
        options = {"guard_filter_fn": keep_global_guards} if keeps_global_guard else {}
        compiled_fn = torch.compile(
            global_rebind_fn, fullgraph=True, backend="eager", options=options
        ).aot_compile(((x,), {}))
        guards_state = load_guards_state(compiled_fn._artifacts.guards_state)
        builtins_key = guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
        import_sources = compiled_fn._artifacts.runtime_env.import_sources
        self.assertTrue(builtins_key)
        self.assertTrue(import_sources)
        kept = [str(g) for g in guards_state.output_graph.guards]
        self.assertEqual(any("G[" in g for g in kept), keeps_global_guard)
        self.assertFalse(any(builtins_key in g for g in kept))
        # The kept global guard here is rooted at AOT_POOL_MODE, not at an alias.
        global_scope = guards_state.output_graph.global_scope
        self.assertFalse(set(import_sources) & set(global_scope))
        compiled_fn.save_compiled_function(self.path())

        torch._dynamo.reset()
        # AOT_POOL_MODE so the kept global guard can resolve.
        scope: dict[str, object] = {"AOT_POOL_MODE": "sum"}
        before = set(scope)
        with open(self.path(), "rb") as f:
            loaded = AOTCompiledFunction.deserialize(f.read(), guard_globals=scope)
        self.assertEqual(loaded(x), global_rebind_fn(x))
        self.assertEqual(set(scope), before)

        # An EMPTY scope is a scope, not the absence of one, and there is nothing
        # for the seeding to add to it here either: the kept global guard has no
        # AOT_POOL_MODE to read, so it must fail rather than fall back to the
        # value serialized with the artifact.
        torch._dynamo.reset()
        empty: dict[str, object] = {}
        with open(self.path(), "rb") as f:
            loaded = AOTCompiledFunction.deserialize(f.read(), guard_globals=empty)
        if keeps_global_guard:
            missing = r"KeyError on G\['AOT_POOL_MODE'\]"
            with self.assertRaisesRegex(RuntimeError, missing):
                loaded(x)
        else:
            self.assertEqual(loaded(x), global_rebind_fn(x))
        self.assertEqual(empty, {})

    def test_load_into_an_empty_scope_refuses_a_lifted_global(self):
        # An empty guard_globals is an empty scope, not the absence of one: the
        # load must resolve global guards against it rather than fall back to the
        # scope rebuilt from the artifact. A LIFTED global is what makes the two
        # distinguishable -- it is serialized with the artifact, so the rebuilt
        # scope binds it and the failure changes shape: the fallback's TENSOR_MATCH
        # trips on the serialized FakeTensor's type, where this gate reports the
        # name as missing. A specialized global, as in the arm above, is absent
        # from both scopes and fails the guard either way.
        def fn(x):
            return x * EPS

        x = torch.randn(3, 4)
        expected = fn(x)
        self._hide_leaked_dynamo_globals()
        compiled_fn = torch.compile(
            fn,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        ).aot_compile(((x,), {}))
        guards_state = load_guards_state(compiled_fn._artifacts.guards_state)
        kept = [str(guard) for guard in guards_state.output_graph.guards]
        self.assertTrue(any("G['EPS']" in guard for guard in kept), kept)
        self.assertIn("EPS", compiled_fn._artifacts.runtime_env.used_globals)
        compiled_fn.save_compiled_function(self.path())

        torch._dynamo.reset()
        with open(self.path(), "rb") as f:
            loaded = AOTCompiledFunction.deserialize(f.read(), guard_globals={})
        # The rebuilt scope does bind EPS (as a FakeTensor), so the KeyError below
        # is this gate's answer and not the fallback's type-mismatch failure.
        self.assertIn("EPS", loaded.fn.__globals__)
        with self.assertRaisesRegex(RuntimeError, r"KeyError on G\['EPS'\]"):
            loaded(x)

        # Passing no scope at all is the case that DOES fall back, and it serves
        # the value serialized with the artifact.
        torch._dynamo.reset()
        with open(self.path(), "rb") as f:
            default_loaded = AOTCompiledFunction.deserialize(f.read())
        self.assertEqual(default_loaded(x), expected)

    def test_load_resolves_a_python_shape_guard_against_the_guard_scope(self):
        # A SHAPE_ENV guard roots at a ShapeEnvSource, which is not global, so the
        # default filter keeps it, and with enable_cpp_symbolic_shape_guards off it
        # is a Python lambda rather than a node in the C++ globals tree. Its
        # G['NAME'] has to resolve where that tree does. The serialized
        # global_scope never binds a name only the lambda reads (shape_env_sources
        # is filled from the cpp code parts alone), so a lambda closing over it
        # raised KeyError on every load, the default one included.
        def fn(x):
            return x * 2 + AOT_DYN_ROWS.sum(0)

        x = torch.randn(3)
        expected = fn(x)
        compiled_fn = torch.compile(fn, fullgraph=True, backend="eager").aot_compile(
            ((x,), {})
        )
        guards_state = load_guards_state(compiled_fn._artifacts.guards_state)
        self.assertTrue(guards_state.shape_code_parts.python_fallback)
        exprs = guards_state.shape_code_parts.python_code_parts.exprs
        self.assertTrue(any("G['AOT_DYN_ROWS']" in e for e in exprs), exprs)
        self.assertNotIn("AOT_DYN_ROWS", guards_state.output_graph.global_scope)
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            data = f.read()

        torch._dynamo.reset()
        self.assertEqual(AOTCompiledFunction.deserialize(data)(x), expected)

        # Specification, not regression: the name is absent from the serialized
        # scope too (asserted above), so a lambda over either dict raises
        # KeyError here. The default load above and the one-row arm below are
        # what separate the fix from the parent.
        torch._dynamo.reset()
        loaded = AOTCompiledFunction.deserialize(data, guard_globals={})
        with self.assertRaisesRegex(RuntimeError, r"'AOT_DYN_ROWS'"):
            loaded(x)

        # One row trips the range guard the dynamic dim minted. A failing lambda
        # reports its whole code-part list, so the regex does not single out the
        # tripped expression; it tells a lambda that resolved the name and failed
        # on the caller's tensor from one whose only part is the KeyError name,
        # which is what a lambda over the serialized scope produced.
        torch._dynamo.reset()
        short = {"AOT_DYN_ROWS": torch.randn(1, 3)}
        loaded = AOTCompiledFunction.deserialize(data, guard_globals=short)
        range_guard = r"2 <= G\['AOT_DYN_ROWS'\]\.size\(\)\[0\]"
        with self.assertRaisesRegex(RuntimeError, range_guard):
            loaded(x)

    def test_load_python_shape_guard_cannot_pass_on_the_serialized_scope(self):
        # With a kept global guard the tensor IS in the serialized global_scope, as
        # a FakeTensor carrying the traced sizes, and a lambda reading that scope
        # agrees with the artifact whatever the caller binds: a live tensor whose
        # shape violates the guard passed guard_check. assume_static_by_default
        # rather than mark_dynamic, because a kept TENSOR_MATCH rejects a marked
        # tensor on load.
        def fn(x):
            return x * 2 + AOT_SUMMED.sum(0)

        x = torch.randn(3)
        with torch._dynamo.config.patch(assume_static_by_default=False):
            compiled_fn = torch.compile(
                fn,
                fullgraph=True,
                backend="eager",
                options={"guard_filter_fn": keep_global_guards},
            ).aot_compile(((x,), {}))
        guards_state = load_guards_state(compiled_fn._artifacts.guards_state)
        self.assertTrue(guards_state.shape_code_parts.python_fallback)
        exprs = guards_state.shape_code_parts.python_code_parts.exprs
        equality = "G['AOT_SUMMED'].size()[1] == L['x'].size()[0]"
        self.assertIn(equality, exprs)
        self.assertIn("AOT_SUMMED", guards_state.output_graph.global_scope)
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            data = f.read()

        torch._dynamo.reset()
        wide = {"AOT_SUMMED": torch.randn(4, 5)}
        loaded = AOTCompiledFunction.deserialize(data, guard_globals=wide)
        with self.assertRaisesRegex(RuntimeError, re.escape(equality)):
            loaded(x)

        torch._dynamo.reset()
        tall = {"AOT_SUMMED": torch.randn(7, 3)}
        loaded = AOTCompiledFunction.deserialize(data, guard_globals=tall)
        self.assertTrue(loaded.guard_check(x))

    @parametrize("guard_reads_builtins", (True, False))
    def test_load_disowns_only_a_builtins_key_a_guard_reads(self, guard_reads_builtins):
        # The builtins-dict key a load would seed can already be bound in the
        # target scope by a LIVE compile in that module, behind a CleanupHook that
        # deletes the binding when its code object is collected. Both halves of the
        # builtins branch are load-bearing there and nothing else covers them: with
        # a guard rooted at the key, CleanupHook.disown has to strip that ownership
        # or the artifact stops working once that compile is collected; with no
        # guard rooted at it, the branch must not run at all, or it strips the live
        # compile's ownership and the binding outlives the compile that made it.
        # Rewinding the unique-id counter is what puts the two in the position a
        # load in a fresh process is in: in process the counter is already past any
        # index an artifact baked in, so capture under a fresh counter, then let the
        # live compile mint the same name under another.
        import itertools

        from torch._dynamo import bytecode_transformation
        from torch._dynamo.utils import _cleanup_owners, CleanupManager

        def fn(x):
            if isinstance(x, torch.Tensor):
                return x + 1
            return x

        if guard_reads_builtins:
            target, x = fn, torch.randn(3, 3)
            options = {"guard_filter_fn": keep_builtin_guards}
        else:
            target, x = global_rebind_fn, torch.randn(4, 8)
            options = {"guard_filter_fn": keep_global_guards}
        expected = target(x)
        self._hide_leaked_dynamo_globals()
        g = globals()

        # Two of the private internals here are load-bearing and two are not: the
        # _unique_id_counter rewind is the only way in process to put a live
        # compile on the artifact's exact key name, and _cleanup_owners IS the
        # ownership under test. Recovering this capture's hooks by diffing
        # CleanupManager.instance.values, and firing them by hand instead of
        # waiting for collection, are only means -- any way of running the hooks
        # this capture registered would serve.
        def with_a_fresh_counter(action):
            # Frees the minted name so the next capture's counter really lands
            # on it, and hands back the CleanupManager entries that capture
            # registered: a hook fires when its code object is collected, and in
            # process that is at the mercy of whatever else still references the
            # code, so the test runs the entry itself. _remove_id both fires and
            # deregisters, so a later real collection cannot fire them a second
            # time and strip the name from whoever owns it by then.
            torch._dynamo.reset()
            for name in [k for k in list(g) if k.startswith("__builtins_dict__")]:
                # Disown before deleting, as the other deleting sites here do:
                # a capture takes ownership of the name even when, as with
                # aot_compile, it registers no hook that could fire.
                CleanupHook.disown(g, name)
                del g[name]
            before = set(CleanupManager.instance.values)
            with patch.object(
                bytecode_transformation, "_unique_id_counter", itertools.count()
            ):
                result = action()
            return result, set(CleanupManager.instance.values) - before

        # aot_compile registers no hook with CleanupManager, so this capture's
        # ownership of the name can only ever be superseded, never fired.
        compiled_fn, capture_entries = with_a_fresh_counter(
            lambda: torch.compile(
                target, fullgraph=True, backend="eager", options=options
            ).aot_compile(((x,), {}))
        )
        self.assertEqual(capture_entries, set())
        guards_state = load_guards_state(compiled_fn._artifacts.guards_state)
        builtins_key = guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
        kept = [str(guard) for guard in guards_state.output_graph.guards]
        self.assertEqual(any(builtins_key in k for k in kept), guard_reads_builtins)
        compiled_fn.save_compiled_function(self.path())
        del compiled_fn

        def compile_live():
            live = torch.compile(target, fullgraph=True, backend="eager")
            live(x)
            return live

        _, live_entries = with_a_fresh_counter(compile_live)
        # The two captures burn unique ids independently; if they ever stopped
        # agreeing on the name there would be nothing here to disown, so pin that
        # the live compile really owns the artifact's key rather than assume it.
        self.assertIn((id(g), builtins_key), _cleanup_owners)

        with open(self.path(), "rb") as f:
            loaded = AOTCompiledFunction.deserialize(f.read(), guard_globals=g)
        self.assertEqual(
            (id(g), builtins_key) in _cleanup_owners, not guard_reads_builtins
        )

        for idx in live_entries:
            CleanupManager.instance._remove_id(idx)
        # The live compile is gone: it keeps its hands off a name the load took
        # over, and takes back one the load had no business touching.
        self.assertEqual(builtins_key in g, guard_reads_builtins)
        self.assertEqual(loaded(x), expected)

        # Load-then-compile in one module dict: the key the load left behind is
        # in the way of a compile whose own counter is still behind it, which is
        # the collision the mint skip below this commit exists for.
        if guard_reads_builtins:
            torch._dynamo.reset()
            with patch.object(
                bytecode_transformation, "_unique_id_counter", itertools.count()
            ):
                self.assertEqual(compile_live()(x), expected)

    @parametrize("prebound", ("builtins_key", "__builtins__"))
    def test_load_keeps_a_prebound_builtins_scope(self, prebound):
        # Neither builtins branch may replace what the loading scope already
        # binds: a pre-bound builtins-dict key suppresses the __builtins__
        # insertion too (it is nested under it), and a pre-bound __builtins__ is
        # what the inserted key is derived FROM rather than a name to overwrite.
        # keep_builtin_guards makes the artifact really read the dict that
        # survives; a copy of the real builtins satisfies the guard.
        def fn(x):
            if isinstance(x, torch.Tensor):
                return x + 1
            return x

        x = torch.randn(3, 3)
        compiled_fn = torch.compile(
            fn,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_builtin_guards},
        ).aot_compile(((x,), {}))
        guards_state = load_guards_state(compiled_fn._artifacts.guards_state)
        builtins_key = guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
        self.assertTrue(builtins_key)
        compiled_fn.save_compiled_function(self.path())

        torch._dynamo.reset()
        copied_builtins = dict(builtins.__dict__)
        name = builtins_key if prebound == "builtins_key" else "__builtins__"
        scope: dict[str, object] = {}
        hook = CleanupHook.create(scope, name, copied_builtins)
        self.addCleanup(CleanupHook.disown, scope, name)
        with open(self.path(), "rb") as f:
            loaded = AOTCompiledFunction.deserialize(f.read(), guard_globals=scope)
        if prebound == "builtins_key":
            # With the key pre-bound the load has nothing to add, so a seeding
            # that never ran passes every other assertion here. The disown it
            # runs before the snapshot check is its one trace: the hook firing
            # afterwards must not take the binding.
            hook()
            self.assertTrue(scope.get(name) is copied_builtins, "never disowned")
        self.assertEqual(loaded(x), fn(x))
        self.assertIs(scope[name], copied_builtins)
        self.assertIs(scope[builtins_key], copied_builtins)
        # keep_builtin_guards leaves nothing rooted at an alias, so the builtins
        # key is the only name this load may add.
        if prebound == "__builtins__":
            added = {builtins_key}
        else:
            added = set()
            self.assertNotIn("__builtins__", scope)
        self.assertEqual(set(scope) - {name}, added)

    def test_load_rederives_a_builtins_key_bound_to_its_own_snapshot(self):
        # When the GENERATED bytecode reads the builtins-dict key -- `len` in the
        # output is reconstructed through DictGetItemSource(GlobalSource(key)),
        # while the constant-folded isinstance is not -- get_runtime_env records a
        # pickle-filtered COPY of the tracing builtins under that key, and
        # forward_callable spreads it into fn.__globals__, which IS the default
        # guard scope. Left in place, that snapshot leaves the ID_MATCH guard
        # rooted there passing after the builtin it read is rebound, so the load
        # re-derives over it -- and over nothing else: the second arm loads the
        # same artifact into a scope whose binding the loading process chose.
        def fn(x):
            if isinstance(x, torch.Tensor):
                return x + 1, len
            return x, len

        x = torch.randn(3, 3)
        compiled_fn = torch.compile(
            fn,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_builtin_guards},
        ).aot_compile(((x,), {}))
        runtime_env = compiled_fn._artifacts.runtime_env
        guards_state = load_guards_state(compiled_fn._artifacts.guards_state)
        builtins_key = guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
        self.assertIn(builtins_key, runtime_env.bytecode.co_names)
        self.assertIn(builtins_key, runtime_env.used_globals)
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            data = f.read()

        torch._dynamo.reset()
        loaded = AOTCompiledFunction.deserialize(data)
        recorded = loaded._artifacts.runtime_env.used_globals[builtins_key]
        # Identity, asserted through assertTrue: a whole builtins dict in an
        # assertIs report buries the reason it failed.
        self.assertTrue(recorded is not builtins.__dict__, "recorded the live dict")
        scope = loaded.fn.__globals__
        self.assertTrue(
            scope[builtins_key] is builtins.__dict__,
            "the load left its own pickled builtins snapshot in the guard scope",
        )
        real_isinstance = builtins.isinstance
        try:
            # Behaves exactly like the real isinstance and differs only in
            # identity, so a guard that still passes is reading the snapshot.
            builtins.isinstance = lambda obj, cls: real_isinstance(obj, cls)
            self.assertFalse(loaded.guard_check(x))
        finally:
            builtins.isinstance = real_isinstance
        self.assertTrue(loaded.guard_check(x))
        self.assertEqual(loaded(x), fn(x))

        torch._dynamo.reset()
        chosen = dict(builtins.__dict__)
        caller_scope: dict[str, object] = {}
        # Bound through a CleanupHook and fired after the load: the disown that
        # runs ahead of the snapshot check tells "admitted and left alone" from a
        # seeding that never ran, which the other assertions here cannot.
        hook = CleanupHook.create(caller_scope, builtins_key, chosen)
        self.addCleanup(CleanupHook.disown, caller_scope, builtins_key)
        loaded = AOTCompiledFunction.deserialize(data, guard_globals=caller_scope)
        hook()
        self.assertTrue(
            caller_scope.get(builtins_key) is chosen,
            "the load replaced, or never disowned, the builtins dict it was handed",
        )
        self.assertNotIn("__builtins__", caller_scope)
        # A caller-supplied guard scope is a DIFFERENT dict from the one the
        # bytecode reads, so leaving the guard scope's binding alone must not leave
        # the recording behind in fn.__globals__.
        self.assertTrue(
            loaded.fn.__globals__[builtins_key] is builtins.__dict__,
            "the bytecode kept the snapshot under a caller-chosen guard scope",
        )
        self.assertEqual(loaded(x), fn(x))
        # Re-checked after the call: the guard scope's binding is never copied
        # over the bytecode's. __post_init__ subtracts this key from the arming
        # set on every load path, and the certified set it picks from takes bare
        # GlobalSource roots only, while every source under this key is a
        # DictGetItemSource, so it is never in _live_global_names and _serve
        # never copies it.
        self.assertTrue(
            loaded.fn.__globals__[builtins_key] is builtins.__dict__,
            "a call rewired the bytecode to the guard scope's binding",
        )

        # Loading the same bytes into the same dict twice: forward_callable lets
        # f_globals override the recording, so the second load finds the key
        # already bound and takes the other branch. Both have to agree on what the
        # bytecode reads.
        shared: dict[str, object] = {}
        for _ in range(2):
            torch._dynamo.reset()
            loaded = AOTCompiledFunction.deserialize(
                data, f_globals=shared, guard_globals=shared
            )
            self.assertTrue(
                loaded.fn.__globals__[builtins_key] is builtins.__dict__,
                "the bytecode is reading a dict this load chose the order of",
            )
            self.assertTrue(shared[builtins_key] is builtins.__dict__)
            self.assertEqual(loaded(x), fn(x))

    def test_load_rederives_over_a_builtin_the_loading_process_lacks(self):
        # The re-derive only ever fires when the GENERATED bytecode reads the key,
        # since that is what makes get_runtime_env record it, so it also decides
        # what the bytecode subscripts, in a caller's guard scope as much as the
        # default one -- and the recording it replaces is filtered for picklability
        # alone, never narrowed to names the loading process has. A builtin missing
        # here therefore stops being readable: a kept guard on that name reports it,
        # and with that guard filtered out the read fails inside the generated
        # bytecode instead.
        def fn(x):
            if isinstance(x, torch.Tensor):
                return x + 1, aot_probe  # noqa: F821
            return x, aot_probe  # noqa: F821

        def drop_the_probe_guard(guard_entries):
            kept = keep_builtin_guards(guard_entries)
            return [
                keep and "aot_probe" not in str(entry.orig_guard.originating_source)
                for keep, entry in zip(kept, guard_entries)
            ]

        def save(guard_filter_fn):
            torch._dynamo.reset()
            compiled_fn = torch.compile(
                fn,
                fullgraph=True,
                backend="eager",
                options={"guard_filter_fn": guard_filter_fn},
            ).aot_compile(((x,), {}))
            runtime_env = compiled_fn._artifacts.runtime_env
            guards_state = load_guards_state(compiled_fn._artifacts.guards_state)
            key = guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
            self.assertIn(key, runtime_env.bytecode.co_names)
            self.assertIn("aot_probe", runtime_env.used_globals[key])
            compiled_fn.save_compiled_function(self.path())
            with open(self.path(), "rb") as f:
                return f.read(), key

        x = torch.randn(3, 3)
        # Picklable by reference, so the recording keeps it and only the LOADING
        # process is the one without it.
        builtins.aot_probe = os.getcwd
        try:
            guarded, builtins_key = save(keep_builtin_guards)
            unguarded, unguarded_key = save(drop_the_probe_guard)
        finally:
            del builtins.aot_probe

        torch._dynamo.reset()
        loaded = AOTCompiledFunction.deserialize(guarded)
        self.assertTrue(
            loaded.fn.__globals__[builtins_key] is builtins.__dict__,
            "the load left its own recording under the builtins key",
        )
        self.assertFalse(loaded.guard_check(x))
        with self.assertRaises(RuntimeError) as guard_failure:
            loaded(x)
        self.assertIn(
            f"KeyError on G['{builtins_key}']['aot_probe']",
            str(guard_failure.exception),
        )

        torch._dynamo.reset()
        loaded = AOTCompiledFunction.deserialize(unguarded)
        # A guard rooted at the key is what admits the re-derive, but it is the
        # guard on isinstance here, and nothing checks the other names the
        # bytecode subscripts out of the dict that just got swapped.
        self.assertTrue(loaded.guard_check(x))
        with self.assertRaises(KeyError) as read_failure:
            loaded(x)
        self.assertEqual(read_failure.exception.args, ("aot_probe",))

        # Through the public wrapper the guards get the caller's dict, which is
        # not the one the bytecode subscripts, so the re-derive has to reach both.
        # Reaching only the guard scope would let this load serve off a builtin the
        # process no longer has while its guard reads the live builtins the
        # caller's dict now holds; it reads the live builtins on both sides
        # instead, so it fails exactly like the arm above.
        torch._dynamo.reset()
        scope: dict[str, object] = {}
        loaded = torch.compiler.load_compiled_function(
            io.BytesIO(unguarded), f_globals=scope
        )
        self.assertTrue(
            loaded.fn.__globals__[unguarded_key] is builtins.__dict__,
            "the bytecode kept the recording under a caller-supplied f_globals",
        )
        self.assertTrue(loaded.guard_check(x))
        with self.assertRaises(KeyError) as caller_scope_failure:
            loaded(x)
        self.assertEqual(caller_scope_failure.exception.args, ("aot_probe",))
        self.assertTrue(
            scope[unguarded_key] is builtins.__dict__,
            "the re-derive did not land in the caller's dict",
        )

    def test_load_refuses_a_non_dict_builtins_binding(self):
        # guard_globals is the one dict on this path that comes from outside torch,
        # so a bad __builtins__ in it has to be named: get_builtins_dict would
        # otherwise raise a bare AttributeError out of Dynamo internals. A module
        # is the other legal binding and is derived from, not replaced. Calling a
        # child module roots kept guards at an __import_ alias as well, which is
        # what lets the last arm see that a refused load wrote nothing, and
        # returning `len` makes the bytecode read the key, so fn.__globals__ holds
        # a recording for the f_globals arm to reach.
        def fn(mod, x):
            if isinstance(x, torch.Tensor):
                return mod(x), len
            return x, len

        lin, x = torch.nn.Linear(3, 3), torch.randn(3, 3)
        compiled_fn = torch.compile(
            fn,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_builtin_guards},
        ).aot_compile(((lin, x), {}))
        runtime_env = compiled_fn._artifacts.runtime_env
        guards_state = load_guards_state(compiled_fn._artifacts.guards_state)
        builtins_key = guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
        self.assertIn(builtins_key, runtime_env.bytecode.co_names)
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            data = f.read()

        torch._dynamo.reset()
        with self.assertRaisesRegex(
            TypeError,
            r"guard_globals\['__builtins__'\] must be a dict or a module, "
            r"got NoneType",
        ):
            AOTCompiledFunction.deserialize(data, guard_globals={"__builtins__": None})

        # f_globals lands in fn.__globals__, which IS the default guard scope, so
        # the message has to name the parameter the bad binding arrived by.
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            TypeError,
            r"f_globals\['__builtins__'\] must be a dict or a module, got NoneType",
        ):
            AOTCompiledFunction.deserialize(data, f_globals={"__builtins__": None})

        # load_compiled_function forwards the caller's f_globals as guard_globals,
        # so one dict arrives by both routes; the message has to name the parameter
        # the public signature actually has.
        torch._dynamo.reset()
        bad_scope: dict[str, object] = {"__builtins__": None}
        with open(self.path(), "rb") as f:
            with self.assertRaisesRegex(
                TypeError,
                r"f_globals\['__builtins__'\] must be a dict or a module, got NoneType",
            ):
                torch.compiler.load_compiled_function(f, f_globals=bad_scope)

        # With a distinct guard_globals, f_globals is the bytecode's dict alone;
        # before the re-derive reached it this load succeeded off the recording.
        # The refusal has to come before the aliases and the key are seeded into
        # the guard scope, or a refused load would leave four names behind in a
        # dict that may be a user module's namespace.
        torch._dynamo.reset()
        guard_only: dict[str, object] = {}
        with self.assertRaisesRegex(
            TypeError,
            r"f_globals\['__builtins__'\] must be a dict or a module, got NoneType",
        ):
            AOTCompiledFunction.deserialize(
                data, f_globals={"__builtins__": None}, guard_globals=guard_only
            )
        self.assertEqual(guard_only, {})

        torch._dynamo.reset()
        scope: dict[str, object] = {"__builtins__": builtins}
        loaded = AOTCompiledFunction.deserialize(data, guard_globals=scope)
        self.assertEqual(loaded(lin, x), fn(lin, x))
        self.assertIs(scope["__builtins__"], builtins)

        # The refusal comes before the load writes anything: a key a live
        # compile's CleanupHook owns keeps its owner, so the hook still takes the
        # binding back, and no alias is seeded. A pre-bound key is what lets the
        # load skip re-deriving, which does not excuse the binding.
        torch._dynamo.reset()
        guards_state = load_guards_state(compiled_fn._artifacts.guards_state)
        builtins_key = guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
        pruned = guards_state.output_graph.global_scope
        self.assertTrue(any(k.startswith("__import_") for k in pruned))
        scope = {"__builtins__": None}
        hook = CleanupHook.create(scope, builtins_key, dict(builtins.__dict__))
        self.addCleanup(CleanupHook.disown, scope, builtins_key)
        with self.assertRaisesRegex(TypeError, r"guard_globals\['__builtins__'\]"):
            AOTCompiledFunction.deserialize(data, guard_globals=scope)
        self.assertEqual(set(scope), {"__builtins__", builtins_key})
        hook()
        self.assertNotIn(builtins_key, scope)

    def test_load_seeds_exactly_the_recorded_globals(self):
        # Loading may add only the aliases a kept guard is actually rooted at --
        # the artifact records more than that, and it records a builtins-dict key
        # no guard here reads -- and must not overwrite a name the loading process
        # already has: the second load leaves a sentinel on the guarded alias, and
        # the artifact then fails that guard instead of silently reading the
        # sentinel.
        self._hide_leaked_dynamo_globals()
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
        guards_state = load_guards_state(result._artifacts.guards_state)
        builtins_key = guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
        # Read the guarded alias off the serialized global_scope, which is pruned
        # to exactly the guarded names, rather than off import_sources' order.
        guarded = set(import_sources) & set(guards_state.output_graph.global_scope)
        self.assertLess(len(guarded), len(import_sources))
        # Named before the unpacking below: an nn.Module internal that adds or
        # drops a guarded alias would otherwise report a bare ValueError.
        self.assertEqual(len(guarded), 1, f"expected one guarded alias: {guarded}")
        (kept_alias,) = guarded
        # Through the function loader, which is the seeding's own layer: a
        # module load seeds the scope it resolved from model.forward or was
        # handed as guard_globals=, which
        # test_aot_compile_module_deserialize_takes_a_guard_globals_scope pins.
        (serialized,) = pickle.loads(data)
        expected = mod(x)
        torch._dynamo.reset()

        # Filtered here rather than by the helper above: the capture just minted
        # both families into this module AFTER the helper hid the older ones. The
        # helper's cleanup still strips what this capture added, which
        # aot_compile never does.
        base = {
            k: v
            for k, v in globals().items()
            if not k.startswith(("__import_", "__builtins_dict__"))
        }
        scope = dict(base)
        before = set(scope)
        loaded = AOTCompiledFunction.deserialize(serialized, guard_globals=scope)
        self.assertEqual(set(scope) - before, {kept_alias})
        self.assertNotIn(builtins_key, scope)
        self.assertIs(
            scope[kept_alias], importlib.import_module(import_sources[kept_alias])
        )
        self.assertEqual(loaded(mod, x), expected)

        torch._dynamo.reset()
        scope = dict(base)
        sentinel = object()
        scope[kept_alias] = sentinel
        before = set(scope)
        loaded = AOTCompiledFunction.deserialize(serialized, guard_globals=scope)
        self.assertEqual(set(scope), before)
        self.assertIs(scope[kept_alias], sentinel)
        with self.assertRaises(RuntimeError) as ctx:
            loaded(mod, x)
        self.assertIn("GuardManager check failed", str(ctx.exception))
        self.assertIn(kept_alias, str(ctx.exception))

    def test_builtins_key_gate_covers_the_other_serializer_channels(self):
        # The builtins-key gate matches the deserialized guards' own roots, while
        # the serializer's pruning scan reads two more channels: the shape-env
        # sources it substitutes for a ShapeEnvSource guard, and DUPLICATE_INPUT's
        # source_b, collected in additional_used_global_vars. Both can root at a
        # global -- this artifact roots each at AOT_DUPE_A -- but never at the
        # builtins key, so the gate can ignore them; nothing fails if a future
        # guard type starts rooting one there, so pin it on an artifact that
        # really carries a global through both.
        from torch._dynamo.source import get_global_source_name

        def fn(x):
            if isinstance(x, torch.Tensor):
                return x * 2, AOT_DUPE_A + AOT_DUPE_B
            return x

        x = torch.randn(5)
        additional = []
        serialize_guards = CheckFunctionManager.serialize_guards

        def record(self, *args, **kwargs):
            # additional_used_global_vars lives on the manager and is never
            # serialized, so read it off the instance the capture builds.
            additional.append(set(self.additional_used_global_vars))
            return serialize_guards(self, *args, **kwargs)

        # enable_cpp_symbolic_shape_guards is what makes shape_env_sources
        # non-empty at all -- it is filled from the cpp code parts'
        # source_to_symbol -- and assume_static_by_default=False is what puts a
        # GLOBAL-rooted source in it, by making the dims of AOT_DUPE_A dynamic
        # too. With only the input dynamic every entry names no global, and the
        # assertNotIn on that half is satisfied by {None}.
        with (
            patch.object(CheckFunctionManager, "serialize_guards", record),
            torch._dynamo.config.patch(
                enable_cpp_symbolic_shape_guards=True, assume_static_by_default=False
            ),
        ):
            compiled_fn = torch.compile(
                fn,
                fullgraph=True,
                backend="eager",
                options={"guard_filter_fn": keep_builtin_guards},
            ).aot_compile(((x,), {}))

        guards_state = load_guards_state(compiled_fn._artifacts.guards_state)
        builtins_key = guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
        self.assertTrue(builtins_key)
        shape_env_sources = guards_state.shape_code_parts.shape_env_sources
        self.assertEqual(len(additional), 1, f"expected one capture: {additional}")
        (dupe_globals,) = additional
        shape_globals = {get_global_source_name(s) for s in shape_env_sources}
        # Both channels really carry a global, or the two assertions below say
        # nothing.
        self.assertIn("AOT_DUPE_A", shape_globals)
        self.assertIn("AOT_DUPE_A", dupe_globals)
        self.assertNotIn(builtins_key, shape_globals)
        self.assertNotIn(builtins_key, dupe_globals)

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
                    graph_partition=True,
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

    def test_graph_device_types_scans_the_whole_graph(self):
        # example_value is the key a Dynamo capture populates; the graphs below
        # use "val".
        with FakeTensorMode():
            cpu = torch.empty(2)
            cuda = torch.empty(2, device="cuda")
        graph = torch.fx.Graph()
        c = graph.placeholder("c")
        c.meta["example_value"] = cpu
        g = graph.placeholder("g")
        g.meta["example_value"] = cuda
        c_sum = graph.call_function(torch.ops.aten.sum.default, (c,))
        c_sum.meta["example_value"] = cpu
        g_sum = graph.call_function(torch.ops.aten.sum.default, (g,))
        g_sum.meta["example_value"] = cuda
        graph.output((c_sum, g_sum))
        devices = _graph_device_types(graph)
        self.assertEqual(devices, frozenset(("cpu", "cuda")))
        self.assertEqual(_collapse_device_types(devices), "cuda")

    def test_graph_device_types_reads_a_get_attr_submodule_body(self):
        # A cond branch is a submodule the parent graph only references, and the
        # cond node's meta shows what the branch returned (cpu) rather than the
        # cuda it used -- the shape torch.cond emits.
        with FakeTensorMode():
            cpu = torch.empty(2)
        body = torch.fx.Graph()
        body_x = body.placeholder("x")
        on_cuda = body.call_method("cuda", (body_x,))
        body.output((body.call_method("cpu", (on_cuda,)),))
        parent = torch.fx.Graph()
        x = parent.placeholder("x")
        x.meta["val"] = cpu
        branch = parent.get_attr("cond_true_0")
        cond = parent.call_function(
            torch.ops.higher_order.cond, (x, branch, branch, (x,))
        )
        cond.meta["val"] = (cpu,)
        parent.output((cond,))
        root = {"cond_true_0": torch.fx.GraphModule({}, body)}
        devices = _graph_device_types(torch.fx.GraphModule(root, parent).graph)
        self.assertEqual(devices, frozenset(("cpu", "cuda")))
        self.assertEqual(_collapse_device_types(devices), "cuda")

    def test_graph_device_types_scans_a_reused_body_once(self):
        # A reused region installs one body and emits a get_attr per call site
        # (invoke_subgraph does), so a scan per node is exponential in nesting
        # depth. The dotted target is the other half: a get_attr target is a
        # qualified name, not a single attribute.
        leaf = torch.fx.Graph()
        leaf.output((leaf.call_method("cuda", (leaf.placeholder("x"),)),))
        mid = torch.fx.Graph()
        for _ in range(4):
            mid.get_attr("leaf_0")
        mid.output(())
        mid_gm = torch.fx.GraphModule({"leaf_0": torch.fx.GraphModule({}, leaf)}, mid)
        parent = torch.fx.Graph()
        for _ in range(4):
            parent.get_attr("wrap.mid_0")
        parent.output(())
        graph = torch.fx.GraphModule({"wrap.mid_0": mid_gm}, parent).graph
        calls = []
        real = _graph_device_types

        def counting(*args, **kwargs):
            calls.append(args[0])
            return real(*args, **kwargs)

        with patch("torch._dynamo.graph_utils._graph_device_types", counting):
            devices = counting(graph)
        self.assertEqual(devices, frozenset(("cuda",)))
        # 1 + 4 + 4: each body is entered once, the other calls return at once.
        self.assertEqual(len(calls), 9)

    def test_graph_device_types_stops_on_a_body_that_reaches_itself(self):
        # Nothing in FX forbids it, and without a guard the scan never returns.
        graph = torch.fx.Graph()
        graph.output((graph.get_attr("loop"),))
        gm = torch.fx.GraphModule({"loop": torch.nn.Module()}, graph)
        gm.loop = gm
        self.assertEqual(_graph_device_types(gm.graph), frozenset())

    def test_graph_device_types_ignores_placeholders_without_a_device(self):
        # A dynamic-shape capture can lead with a SymInt placeholder -- it does
        # under the default config; the test harness canonicalizes tensors first --
        # which has no device of its own, the shape the first graph below imitates.
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
        # An autocast device type is a plain positional arg of _enter_autocast;
        # read as a device, a CPU-only graph would refuse to load where it was
        # saved.
        with FakeTensorMode():
            cpu = torch.empty(2)
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = cpu
        graph.call_function(torch.amp._enter_autocast, ("cuda", None, True, None))
        graph.call_function(torch.ops.aten.add.Tensor, (x, 1)).meta["val"] = cpu
        self.assertEqual(_graph_device_types(graph), frozenset(("cpu",)))

        # The other direction, where the collapse would hide the mistake, so
        # what this pins is the reported set itself.
        with FakeTensorMode():
            cuda = torch.empty(2, device="cuda")
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = cuda
        graph.call_function(torch.amp._enter_autocast, ("cpu", None, True, None))
        self.assertEqual(_graph_device_types(graph), frozenset(("cuda",)))

        # Real device positions are still read.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.call_method("to", (x, "mps"))
        graph.call_function(torch.ops.aten.ones.default, ([2],), {"device": "cuda"})
        self.assertEqual(_graph_device_types(graph), frozenset(("mps", "cuda")))

    @parametrize("method", ("cpu", "cuda", "xpu", "ipu", "mtia"))
    def test_graph_device_types_reads_a_device_naming_method(self, method):
        # x.cuda() names the device in the method, so a graph without meta has
        # nothing else to read and would collapse to "cpu".
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.call_method(method, (x,))
        devices = _graph_device_types(graph)
        self.assertEqual(devices, frozenset((method,)))
        self.assertEqual(_collapse_device_types(devices), method)

    def test_graph_device_types_reads_the_privateuse1_method(self):
        # generate_methods_for_privateuse1_backend() gives a renamed backend a
        # method named after it (.npu()), so the scan reads the name rather than
        # baking in the default. The getter is patched rather than the backend
        # really renamed: a real rename is once-per-process and would leak into
        # every later test, and asserting on the unrenamed default name would
        # pass against a scan that hardcoded it.
        with patch.object(
            torch._C, "_get_privateuse1_backend_name", return_value="npu"
        ):
            graph = torch.fx.Graph()
            x = graph.placeholder("x")
            graph.call_method("npu", (x,))
            self.assertEqual(_graph_device_types(graph), frozenset(("npu",)))

    def test_graph_device_types_reads_a_bare_device_index(self):
        # Dynamo emits a bare index in a device position (device=0, x.to(0)),
        # which torch.device resolves against the accelerator the build
        # provides, not against the machine running the test.
        try:
            expected = frozenset((torch.device(0).type,))
        except RuntimeError:
            # A build with no accelerator has no device for index 0 to name, so
            # this arm cannot be exercised here at all -- skip rather than
            # assert the empty answer a helper ignoring integers also gives.
            self.skipTest("no accelerator in this build for index 0 to name")

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.call_method("to", (x, 0))
        graph.call_function(torch.ops.aten.ones.default, ([2],), {"device": 0})
        self.assertEqual(_graph_device_types(graph), expected)

    @parametrize("spec", ("not_a_device", 2**63, True))
    def test_graph_device_types_ignores_an_unparsable_device_position(self, spec):
        # A value torch.device rejects names no device rather than aborting an
        # otherwise fine compile: an unknown name raises RuntimeError, an
        # oversized index ValueError. True is deliberately excluded from the
        # index arm, so it never reaches torch.device at all.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.call_method("to", (x, spec))
        self.assertEqual(_graph_device_types(graph), frozenset())

    def test_graph_device_types_drops_the_meta_device(self):
        # meta is an abstract device: kept in, it wins the collapse over cpu
        # and records device_type="meta", a string no host check can be run for.
        with FakeTensorMode():
            meta = torch.empty(2, device="meta")
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = meta
        node = graph.call_function(
            torch.ops.aten.ones.default, ([2],), {"device": "meta"}
        )
        node.meta["val"] = meta
        self.assertEqual(_graph_device_types(graph), frozenset())
        self.assertEqual(_collapse_device_types(_graph_device_types(graph)), "cpu")

    def test_a_recorded_gpu_device_type_arms_the_load_check(self):
        # What the flip from "cpu" to an accelerator buys, and what it costs: a
        # recorded "cuda" is what makes check_compatibility compare the GPU
        # name, so an artifact saved on one GPU is now refused on another, and
        # the AOT load path calls this unguarded. Both names are fabricated --
        # SystemInfo.current().gpu_name is None on a host without a GPU, which
        # is a mismatch on one side of the check only -- and is_available is
        # patched so the refusal under test is the GPU-name one there too.
        here = dataclasses.replace(SystemInfo.current(), gpu_name="This GPU")
        saved = dataclasses.replace(here, gpu_name="Some Other GPU")
        with patch.object(torch.cuda, "is_available", return_value=True):
            here.check_compatibility(saved, "cpu")
            with self.assertRaisesRegex(RuntimeError, "created with different GPU"):
                here.check_compatibility(saved, "cuda")
            # The two load paths pass the saved info in opposite positions (AOT
            # as other, caching precompile as self). With both names known they
            # agree; only the side passed as other is required to be known, so
            # an unknown name splits them.
            with self.assertRaisesRegex(RuntimeError, "created with different GPU"):
                saved.check_compatibility(here, "cuda")
            unknown = dataclasses.replace(here, gpu_name=None)
            with self.assertRaisesRegex(RuntimeError, "created with different GPU"):
                unknown.check_compatibility(saved, "cuda")
            saved.check_compatibility(unknown, "cuda")

    def test_a_recorded_device_the_host_lacks_refuses_the_compile(self):
        # __post_init__ runs at the end of a compile as well as on load, so the
        # flip reaches the compile: a graph read as cuda no longer compiles
        # where cuda is missing, which recording "cpu" let through. No loadable
        # artifact is lost -- such a host records toolkit_version=None, which
        # every load of a cuda artifact refuses.
        def fn(x):
            return x + 1

        compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")
        artifacts = compiled_fn.aot_compile(((torch.randn(3),), {}))._artifacts
        artifacts = dataclasses.replace(artifacts, device_type="cuda")
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "cuda is not available"):
                AOTCompiledFunction(artifacts)

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
    def test_pickler_carries_a_docstring(self):
        # A native docstring lives in the code object, one assigned after
        # definition does not; both travel in the pickle state. The old rebuild
        # (types.FunctionType over the code object) kept the native one and
        # dropped an assigned one; on this base a doc=None would drop both,
        # since _apply_function_state assigns __doc__ unconditionally.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        def outer():
            def inner(x):
                """native"""
                return x

            def assigned(x):
                return x

            assigned.__doc__ = "assigned by a decorator"
            return inner, assigned

        fns = outer()
        buf = io.BytesIO()
        AOTCompilePickler({}, buf).dump(fns)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertEqual(
            [f.__doc__ for f in out], ["native", "assigned by a decorator"]
        )

    def test_pickler_prunes_an_unpicklable_docstring(self):
        # Nothing on the load path forces __doc__, so an unpicklable one is
        # dropped to None with a warning rather than failing the dump; carried
        # verbatim the dump fails on the lock.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        def outer():
            def inner(x):
                """native docstring, so the pruned None has to override it"""
                return inner if x is None else x  # closes over itself: reduced twice

            inner.__doc__ = threading.Lock()
            return inner

        fn = outer()
        buf = io.BytesIO()
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            AOTCompilePickler({}, buf).dump(fn)
        (line,) = [l for l in logs.output if "dropping" in l]
        self.assertIn("inner.__doc__ (lock) from the artifact", line)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertIsNone(out.__doc__)
        self.assertEqual(out(5), 5)

    def test_pickler_keeps_an_external_modules_method_by_reference(self):
        # The receiver is external data, so it is the live object at load and
        # pickle's default getattr(receiver, name) resolves the method on it.
        # The shared bound-method reducer would instead carry __func__ (an
        # nn.Module defines __getattr__) and rebuild it by value, which for a
        # local subclass fails on the __class__ cell of its super() call.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        def make():
            class LocalMod(torch.nn.Linear):
                def forward(self, x):
                    return super().forward(x) + 1

            return LocalMod(2, 2)

        mod = make()
        buf = io.BytesIO()
        AOTCompilePickler({"mod": mod}, buf).dump(mod.forward)
        out = AOTCompileUnpickler({"mod": mod}, io.BytesIO(buf.getvalue())).load()
        self.assertIs(out.__func__, type(mod).forward)
        self.assertIs(out.__self__, mod)

    def test_pickler_does_not_prune_an_unpicklable_kwdefault(self):
        # Unlike __doc__/annotations, __kwdefaults__ is never pruned: a function
        # cannot be called without it, so an unpicklable kwdefault fails loudly.
        from torch._dynamo.aot_compile import AOTCompilePickler

        def outer():
            def inner(*, k=threading.Lock()):
                return k

            return inner

        fn = outer()
        buf = io.BytesIO()
        with self.assertRaisesRegex((TypeError, pickle.PicklingError), "cannot pickle"):
            AOTCompilePickler({}, buf).dump(fn)

    def test_pickler_breaks_a_dict_cycle_between_nested_functions(self):
        # Nested functions whose __dict__ entries point at themselves and each
        # other re-enter _dumps_cleanly mid-probe. Without the in-flight
        # short-circuit the probe recurses until RecursionError, which it reads
        # as "unpicklable", and the pair loses picklable entries (here `g` and
        # `f.g` vanish); with it the cycle terminates, only the lock is pruned,
        # and the rebuilt pair still refers to itself.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        def outer():
            def f(x):
                return x + 1

            def g(x):
                return x + 2

            def top(x):
                return x

            f.f, f.g, g.g, g.f, g.lock = f, g, g, f, threading.Lock()
            top.f, top.g = f, g
            return top

        top = outer()
        buf = io.BytesIO()
        AOTCompilePickler({}, buf).dump(top)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertIs(out.g.f, out.f)
        self.assertIs(out.f.g, out.g)
        self.assertIs(out.f.f, out.f)
        self.assertFalse(hasattr(out.g, "lock"))
        self.assertEqual((out.f(1), out.g(1)), (2, 3))

    def test_pickler_prunes_an_unmarked_module_from_a_nested_functions_dict(self):
        # persistent_id records an nn.Module rather than raising, so a Module
        # reached through a nested function's __dict__ would dump here and then
        # poison serialize(); _dumps_cleanly prunes it instead -- unless the user
        # marked it as external data, in which case it is kept by reference.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        mod = torch.nn.Linear(1, 1)

        def outer():
            def helper(x):
                return x

            helper.mod = mod
            return helper

        fn = outer()
        buf = io.BytesIO()
        pickler = AOTCompilePickler({}, buf)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            pickler.dump(fn)
        self.assertEqual(pickler.errors, {})
        (line,) = [l for l in logs.output if "dropping" in l]
        self.assertIn("not marked as external data (Linear)", line)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertFalse(hasattr(out, "mod"))
        buf = io.BytesIO()
        AOTCompilePickler({"mod": mod}, buf).dump(fn)
        out = AOTCompileUnpickler({"mod": mod}, io.BytesIO(buf.getvalue())).load()
        self.assertIs(out.mod, mod)

    def test_pickler_rebuilds_a_nested_function_faithfully(self):
        # The old rebuild passed __qualname__ where FunctionType wants __name__,
        # raised on an EMPTY cell, and dropped __kwdefaults__ (a reloaded
        # `def f(x, *, k=2)` failed with TypeError when called without k).
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        def outer():
            scale = None

            def inner(*, k=1):
                return unset, scale

            def scaled(x, *, k=2):
                return x * k

            inner.__name__ = "renamed"
            inner.__qualname__ = "reassigned.qualname"  # differs from co_qualname
            if inner is None:
                unset = 1  # never runs, so the cell inner closes over stays empty
            return inner, scaled

        fn, scaled = outer()
        cells = dict(zip(fn.__code__.co_freevars, fn.__closure__))
        with self.assertRaisesRegex(ValueError, "empty"):
            cells["unset"].cell_contents
        buf = io.BytesIO()
        AOTCompilePickler({}, buf).dump((fn, scaled))
        out, out_scaled = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertEqual(out.__name__, "renamed")
        self.assertEqual(out.__qualname__, "reassigned.qualname")
        self.assertEqual(out.__kwdefaults__, {"k": 1})
        self.assertEqual(out_scaled(3), 6)
        cells = dict(zip(out.__code__.co_freevars, out.__closure__))
        with self.assertRaisesRegex(ValueError, "empty"):
            cells["unset"].cell_contents
        self.assertIsNone(cells["scale"].cell_contents)

    def test_pickler_carries_a_dict_entry(self):
        # A helper's __dict__ travels with it; a rebuild from the code object
        # alone starts with an empty one.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        def outer():
            def inner(x):
                return x

            inner.tag = 2.0
            return inner

        fn = outer()
        buf = io.BytesIO()
        dumps = [0]
        real_dump = AOTCompilePickler.dump

        def counting_dump(self, obj):
            dumps[0] += 1
            return real_dump(self, obj)

        with patch.object(AOTCompilePickler, "dump", counting_dump):
            AOTCompilePickler({}, buf).dump(fn)
        self.assertEqual(dumps[0], 1)  # a literal entry is not probed
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertEqual(out.tag, 2.0)

    def test_pickler_warns_once_for_a_function_reduced_twice(self):
        # A function that closes over itself is reduced twice (the closure is a
        # reduce ARGUMENT, so the second pass finds it not yet memoized); the
        # real dump's per-function memo keeps the drop to one probe and one
        # warning.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        def outer():
            def inner(x):
                return inner if x is None else x

            inner.lock = threading.Lock()
            return inner

        fn = outer()
        buf = io.BytesIO()
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            AOTCompilePickler({}, buf).dump(fn)
        (line,) = [l for l in logs.output if "dropping" in l]
        self.assertIn(
            "inner.lock (lock) from the artifact: it does not pickle (TypeError); pass it in external_data to keep it (function defined at",
            line,
        )
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertFalse(hasattr(out, "lock"))
        self.assertEqual(out(5), 5)

    def test_pickler_survives_a_reduce_that_writes_back_an_attribute(self):
        # A probe runs user __reduce__ code; one that writes onto the function's
        # __dict__ while it is being walked must not raise "dictionary changed
        # size during iteration" out of the dump. The walk is over a snapshot,
        # so the written entry is not carried either.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        def outer():
            def inner(x):
                return x

            inner.hostile = WritesBackOnReduce(inner)
            return inner

        fn = outer()
        buf = io.BytesIO()
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING"):
            AOTCompilePickler({}, buf).dump(fn)
        self.assertEqual(fn.added, 1)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertEqual(out.__dict__, {})
        self.assertEqual(out(5), 5)

    def test_pickler_never_reuses_a_probes_attribute_set(self):
        # An attribute set computed inside a probe may be over-pruned, so only
        # the real dump's sets are memoized. a is unpicklable through a
        # kwdefault; probing a walks b, whose probe of c fails because c reaches
        # the in-flight a, so b's PROBE set is empty. The real dump computes b's
        # set afresh, after a is final: c re-probes clean (it drops only a), so
        # b keeps c. Reusing the probe's set would lose b.c silently.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        def outer():
            def a(*, k=threading.Lock()):
                return k

            def b():
                return "b!"

            def c():
                return "c!"

            a.b, b.c, c.a = b, c, a

            def top(x):
                return x

            top.a, top.b = a, b  # a first: its probe is what over-prunes b
            return top

        fn = outer()
        buf = io.BytesIO()
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING"):
            AOTCompilePickler({}, buf).dump(fn)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertFalse(hasattr(out, "a"))
        self.assertEqual(out.b.c(), "c!")
        self.assertFalse(hasattr(out.b.c, "a"))

    def test_pickler_prunes_an_entry_that_overflows_the_probe(self):
        # A recursion overflow inside the probe counts as unpicklable: the entry
        # is dropped with a warning and the save succeeds. The guard pickler
        # turns the same condition into a PackageError, since it has a bypass to
        # fall back to; this pickler does not, so the divergence is deliberate.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        def outer():
            def helper(x):
                return x

            helper.deep = BottomlessReduce()
            return helper

        fn = outer()
        buf = io.BytesIO()
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            AOTCompilePickler({}, buf).dump(fn)
        (line,) = [l for l in logs.output if "dropping" in l]
        self.assertIn("helper.deep (BottomlessReduce)", line)
        self.assertIn("it does not pickle (RecursionError)", line)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertFalse(hasattr(out, "deep"))
        self.assertEqual(out(5), 5)

    def test_pickler_names_the_modules_inside_a_dropped_container(self):
        # A picklable container holding unmarked Modules is dropped whole; the
        # warning names the container's type and every Module inside it.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        def outer():
            def helper(x):
                return x

            helper.mods = [torch.nn.Linear(1, 1), torch.nn.ReLU()]
            return helper

        fn = outer()
        buf = io.BytesIO()
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            AOTCompilePickler({}, buf).dump(fn)
        (line,) = [l for l in logs.output if "dropping" in l]
        self.assertIn("helper.mods (list)", line)
        self.assertIn("not marked as external data (Linear, ReLU)", line)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertFalse(hasattr(out, "mods"))

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
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            AOTCompilePickler({}, buf).dump(fn)
        (line,) = [l for l in logs.output if "dropping" in l]
        self.assertIn("inner.__annotations__['x'] (type)", line)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertEqual(out.__annotations__, {"y": int, "return": int})
        self.assertEqual(out(object(), 5), 5)

    def test_pickler_survives_a_reduce_that_writes_back_an_annotation(self):
        # A probe runs user __reduce__ code; one that writes onto the function's
        # __annotations__ while they are being walked must not raise
        # "dictionary changed size during iteration" out of the dump. The hazard
        # is pre-3.14 only (the 3.14 read returns a copy); there this checks no
        # more than that the write-back is harmless.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        def outer():
            class WritesBack:
                def __reduce__(self):
                    inner.__annotations__["added"] = int
                    raise TypeError("cannot pickle WritesBack")

            def inner(x: "WritesBack", y: int) -> int:
                return y

            inner.__annotations__["x"] = WritesBack()
            return inner

        fn = outer()
        buf = io.BytesIO()
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING"):
            AOTCompilePickler({}, buf).dump(fn)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertEqual(out.__annotations__, {"y": int, "return": int})

    def test_pickler_keeps_picklable_type_params(self):
        # __type_params__ is carried when its elements pickle: a module-level
        # TypeVar pickles by reference, so a helper handed one reloads with the
        # same object (a rebuild from the code object alone gives ()). Ungated:
        # below 3.12 the slot lives in __dict__ and the reducer's write has to
        # win over the carried dict; on 3.12+ it must not leak into __dict__.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        def outer():
            def inner(x):
                return x

            inner.__type_params__ = (AOT_TEST_TYPEVAR,)
            return inner

        fn = outer()
        buf = io.BytesIO()
        AOTCompilePickler({}, buf).dump(fn)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertIs(out.__type_params__[0], AOT_TEST_TYPEVAR)
        if sys.version_info >= (3, 12):
            self.assertNotIn("__type_params__", out.__dict__)
        self.assertEqual(out(5), 5)

    @unittest.skipIf(sys.version_info < (3, 12), "PEP 695 type params are 3.12+")
    def test_pickler_keeps_pep695_type_params_given_as_external_data(self):
        # The realistic kept case: a PEP 695 TypeVar never pickles on its own,
        # but handed over as external data it is written by reference (the
        # probe pickler shares external_data), so the generic reloads with its
        # type params and annotations intact.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        fn = _aot_pep695_generic()
        (tv,) = fn.__type_params__
        buf = io.BytesIO()
        AOTCompilePickler({"T": tv}, buf).dump(fn)
        out = AOTCompileUnpickler({"T": tv}, io.BytesIO(buf.getvalue())).load()
        self.assertIs(out.__type_params__[0], tv)
        self.assertEqual(out.__annotations__, {"x": tv, "return": tv})
        self.assertEqual(out(5), 5)

    @unittest.skipIf(sys.version_info < (3, 12), "PEP 695 type params are 3.12+")
    def test_pickler_drops_unpicklable_type_params(self):
        # A PEP 695 function-scoped TypeVar pickles by name as typing.T and
        # fails pickle's identity check against it, so carrying the tuple
        # verbatim would abort the dump. The whole __type_params__ tuple is
        # dropped instead and the function still reloads and runs. Defined via
        # exec so this file still parses below 3.12.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        fn = _aot_pep695_generic()
        buf = io.BytesIO()
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            AOTCompilePickler({}, buf).dump(fn)
        (line,) = [l for l in logs.output if "__type_params__" in l]
        self.assertIn("inner.__type_params__ (TypeVar)", line)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertEqual(out.__type_params__, ())
        self.assertEqual(out(5), 5)

    @unittest.skipIf(sys.version_info < (3, 12), "PEP 695 type params are 3.12+")
    def test_pickler_warns_once_for_a_generic_reduced_twice(self):
        # The type_params memo, like the one for the attributes: a generic that
        # closes over itself is reduced twice, and the drop has to be probed and
        # warned once. The self-reference is the only closure cell here -- PEP
        # 695 compiles `T` into the hidden type-params scope, so it is not a
        # freevar of the body -- which is what a `T` the body itself reads would
        # change (a cell is never pruned, so that shape still fails the dump).
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        fn = _aot_pep695_generic(body="return inner if x is None else x")
        (cell,) = fn.__closure__
        self.assertIs(cell.cell_contents, fn)
        buf = io.BytesIO()
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            AOTCompilePickler({}, buf).dump(fn)
        (line,) = [l for l in logs.output if "__type_params__" in l]
        self.assertIn("inner.__type_params__ (TypeVar)", line)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertEqual(out.__type_params__, ())
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
        with self.assertLogs("torch._dynamo.aot_compile", level="DEBUG") as logs:
            AOTCompilePickler({}, buf).dump(fn)
        self.assertTrue(any("dropping the annotations of" in l for l in logs.output))
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertEqual(out.__annotations__, {})
        self.assertEqual(out([1, 2]), [1, 2])

    def test_pickler_does_not_persist_a_wrong_false_across_an_inflight_seed(self):
        # An in-flight probe must not leave a wrong False in the shared cache.
        # f is unpicklable via an UNPRUNED slot (a Lock kwdefault); f and g
        # reference each other, and h carries g. Probing f seeds an optimistic
        # in-flight state, g re-enters f mid-probe, keeps g.f, dumps f, hits the
        # lock and raises -- which an earlier version cached as g being
        # unpicklable, so an UNRELATED h silently lost its .g and died at call
        # time. The in-flight result is no longer cached, so h keeps g.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        def outer():
            lock = threading.Lock()

            def f(*, k=lock):
                return k

            def g():
                return "g!"

            def h():
                return "h!"

            f.g = g
            g.f = f
            h.g = g

            def top(x):
                return x

            top.f = f  # inserted before h, so f probes (and taints) g first
            top.h = h
            return top

        fn = outer()
        buf = io.BytesIO()
        AOTCompilePickler({}, buf).dump(fn)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertTrue(hasattr(out.h, "g"))
        self.assertEqual(out.h.g(), "g!")

    def test_pickler_probes_a_cyclic_cluster_in_bounded_time(self):
        # Eight nested functions fully connected through __dict__, one of them
        # unpicklable through a kwdefault. Parking the leaned verdicts and
        # caching the outermost probe's keeps this to a few dozen probe dumps;
        # re-deriving a leaned False on every re-entry is exponential (hundreds
        # of thousands of dumps for this graph).
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        dumps = [0]
        real_dump = AOTCompilePickler.dump

        def counting_dump(self, obj):
            dumps[0] += 1
            return real_dump(self, obj)

        def outer():
            fns = [(lambda x, i=i: x + i) for i in range(8)]
            for f in fns:
                for i, g in enumerate(fns):
                    setattr(f, f"f{i}", g)
            fns[0].__kwdefaults__ = {"k": threading.Lock()}

            def top(x):
                return x

            for i, f in enumerate(fns):
                setattr(top, f"f{i}", f)
            return top

        top = outer()
        buf = io.BytesIO()
        with patch.object(AOTCompilePickler, "dump", counting_dump):
            AOTCompilePickler({}, buf).dump(top)
        self.assertLess(dumps[0], 32)  # 16 today; quadratic on 8 nodes is 64
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertFalse(hasattr(out, "f0"))
        self.assertFalse(hasattr(out.f1, "f0"))
        self.assertIs(out.f1.f2, out.f2)
        self.assertIs(out.f7.f1, out.f1)

    def test_pickler_handles_mutually_referencing_annotations(self):
        # Two <locals> functions annotated with each other form an annotation
        # cycle: probing whether one dumps cleanly re-probes the other, which
        # re-probes the first. A value whose probe is still in flight answers
        # True (recorded as a lean), so the re-entrant probe short-circuits
        # instead of recursing forever; pickle's own memo then serializes the
        # actual cycle.
        from torch._dynamo.aot_compile import AOTCompilePickler, AOTCompileUnpickler

        def outer():
            def a(x):
                return x

            def b(x):
                return x

            a.__annotations__ = {"x": b}
            b.__annotations__ = {"x": a}
            return a

        fn = outer()
        buf = io.BytesIO()
        AOTCompilePickler({}, buf).dump(fn)
        out = AOTCompileUnpickler({}, io.BytesIO(buf.getvalue())).load()
        self.assertEqual(out(5), 5)
        self.assertEqual(out.__annotations__["x"].__annotations__["x"], out)


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
