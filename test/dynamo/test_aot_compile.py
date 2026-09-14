# Owner(s): ["module: dynamo"]

import builtins
import contextlib
import copy
import dataclasses
import datetime
import functools
import importlib
import inspect
import io
import multiprocessing as mp
import os
import pickle
import re
import sys
import tempfile
import threading
import types
import typing
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
from torch._dynamo.guards import CheckFunctionManager
from torch._dynamo.package import DynamoCache, load_guards_state
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
from torch.fx.passes.regional_inductor import regional_inductor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
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
    # Hashes into "foo"'s slot, so a guard's PyDict_Contains("foo", d) has to
    # compare against this key; the raise leaves the exception set, which CPython
    # turns into SystemError at the pybind boundary of the guard tree. The three
    # keys here reach a raise through that leaf on purpose, so restoring its dead
    # `result == -1` branch turns their raises into ordinary mismatches and the
    # tests using them have to be rewritten. The stub-guard-manager tests pin the
    # same dispatch semantics without any leaf at all.
    def __hash__(self):
        return hash("foo")

    def __eq__(self, other):
        raise ValueError("boom from __eq__")


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
    # Base for a stub guard manager whose check() raises on every evaluation: the
    # report never re-checks an entry whose last evaluation raised, so a
    # regression that does fails on this line, which names the re-check, rather
    # than on an AttributeError the report's handler would dress up as the tree's
    # own raise -- and the same exception would let that regression pass.
    def check_verbose(self, f_locals):
        raise RuntimeError("the report re-checked a tree that raised in dispatch")


# Not the identity: an identity weight makes "read the serialized weight" and
# "dropped the matmul" produce the same tensor.
AOT_HERMETIC_WEIGHT = torch.eye(3) * 3


class HermeticModule(torch.nn.Module):
    def forward(self, x):
        return x @ AOT_HERMETIC_WEIGHT


GLOBAL_POOLING_CONFIG = {"pooling": "sum"}


class GlobalConfigModule(torch.nn.Module):
    def forward(self, x):
        if GLOBAL_POOLING_CONFIG["pooling"] == "sum":
            return x.sum(1)
        return x.mean(1) * 10.0


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
    # Hashes into `name`'s slot and counts every comparison a lookup of `name`
    # makes against it, answering False for the first `misses` of them. With
    # misses=0 it just counts how many times a guard on `name` was evaluated;
    # with misses=1 the first evaluation misses the global and the next one finds
    # it, which is one way a live guard tree can reject a call in the dispatch
    # scan and then accept it on the full re-check.
    def __init__(self, name, misses=0):
        self.name = name
        self.misses = misses
        self.compares = 0

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        self.compares += 1
        return self.compares > self.misses


AOT_BRANCH_SCALE = 3.0


class ModeBranchGlobalModule(torch.nn.Module):
    # Only the mode == 1 branch reads a global, so one ModelInput's guards name
    # it and the other's do not.
    def forward(self, x, mode):
        if mode == 1:
            return x * AOT_BRANCH_SCALE
        return x * 2


class SelfModeBranchGlobalModule(torch.nn.Module):
    # The same branch on an attribute rather than an argument: the guard on
    # self.mode is a LOCAL_UNSPECIALIZED_NN_MODULE source, which sorts after
    # GLOBAL, so the root's G accessor is installed -- and fails -- before the
    # self.mode guard (the L['x'] TENSOR_MATCH, a LOCAL, comes before both).
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == 1:
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


class TensorDefaultModule(torch.nn.Module):
    # A tensor default makes inspect.Signature equality raise (Parameter.__eq__
    # takes bool() of `default == default`), whether or not the body reads it.
    def forward(self, x, mask=torch.ones(3)):
        return x * 2


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
    mod.forward = types.MethodType(forward, mod)
    model = torch.compile(mod, fullgraph=True, backend="eager")
    model._aot_compile([ModelInput(args=args, kwargs={}, contexts=[])])
    return model.forward.compiled_results[0]


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
        self.assertIn("[0]", message)
        self.assertIn("[1]", message)
        # One line per input, not a multi-line GuardDebugInfo repr per input: the
        # two entries are the two lines after the header, and the advice that
        # follows is not indented, so it is not a continuation of the second.
        lines = message.splitlines()
        self.assertEqual([line[:5] for line in lines[1:3]], ["  [0]", "  [1]"])
        self.assertFalse(lines[3].startswith(" "), lines[3])
        self.assertIn("Add a ModelInput", message)

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
        # The raise line is the one report entry whose text is arbitrary user
        # data, so it needs the collapse the verbose part above gets: a message
        # carrying any separator splitlines() breaks on would otherwise split one
        # entry across several lines, and the whole-line assertions and
        # one-entry-per-input counts the other tests read the report with cannot
        # survive that.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )

        class RaisingGuardManager(NeverReChecked):
            def check(self, f_locals):
                raise RuntimeError("page\x0cbreak\rrec\x1esep\u2028line")

        artifacts = model.forward.compiled_results[0]._artifacts
        artifacts.guard_manager = RaisingGuardManager()
        with self.assertRaises(RuntimeError) as ctx:
            model(torch.randn(3, 3))
        message = str(ctx.exception)
        lines = message.splitlines()
        self.assertEqual(sum(ln.startswith("  [") for ln in lines), 1, message)
        # The message holds no spaces, so this says all four separators arrived
        # and were collapsed -- without which the test is vacuous, and the last
        # three a \n replace would not cover.
        raised = "  [0] <guard check raised RuntimeError: page break rec sep line>"
        self.assertIn(raised, lines)

    def test_aot_compile_module_wrong_arity_raises_type_error(self):
        # A call the signature cannot bind is a caller error, not a guard miss:
        # it surfaces as the TypeError the plain module would raise, not as a
        # no-match report telling the user to add a ModelInput. Nothing catches
        # the bind's TypeError on the way out, and it has to arrive alone: a fold
        # into the no-match report would quote inspect's text as one entry's
        # reason, so neither the type nor the message tells the two apart -- only
        # the absence of the report does.
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )
        x = torch.randn(3, 3)
        with self.assertRaises(TypeError) as ctx:
            model(x, x)
        self.assertIn("too many positional arguments", str(ctx.exception))
        self.assertNotIn("No AOT compiled graph matched", str(ctx.exception))
        with self.assertRaises(TypeError) as ctx:
            model()
        self.assertIn("missing a required argument: 'x'", str(ctx.exception))
        self.assertNotIn("No AOT compiled graph matched", str(ctx.exception))

    def test_no_match_message_survives_a_raising_guard(self):
        # __call__ has already established that nothing matched; re-evaluating
        # the guards to say WHY must not replace that answer with a secondary
        # failure from one entry, which hides both the entry that raised and
        # every other entry's reason. A DICT_NOT_CONTAINS guard is a real way to
        # get there: it runs PyDict_Contains, whose comparison against a
        # colliding key raises, and the C++ tree returns to pybind with the
        # exception still set, which CPython reports as SystemError.
        model = torch.compile(DictBranchModule(), fullgraph=True, backend="eager")
        x = torch.randn(3, 3)
        model._aot_compile(
            [
                ModelInput(args=(x, {}), kwargs={}, contexts=[]),
                ModelInput(args=(x, None), kwargs={}, contexts=[]),
            ]
        )
        with self.assertRaises(RuntimeError) as ctx:
            model(x, {RaisesOnCompare(): 1})
        message = str(ctx.exception)
        self.assertIn("No AOT compiled graph matched this call", message)
        # The SystemError itself says only which bound method returned with an
        # error set, so the line reports the exception behind it.
        raised = "[0] <guard check raised ValueError: boom from __eq__ (through the guard tree's pybind boundary)>"
        self.assertIn(f"  {raised}", message.splitlines())
        # The entry that raised must not swallow the one that can explain itself:
        # [1] was traced with d=None, so its guards never touch the evil dict.
        self.assertIn("[1] L['d'] is None", message)
        # Two independent things to do: [0] raised and has to be fixed or dropped,
        # and [1] rejected the call, which an input covering it would fix.
        self.assertIn("[0]'s guard check raised while checking this call", message)
        self.assertIn("Add a ModelInput", message)
        # [1]'s rejection followed no raise, so the advice rests on a trusted
        # answer and carries no post-throw caveat.
        self.assertNotIn("advice above rests only on rejections", message)
        # One line per input, not a multi-line GuardDebugInfo repr per input.
        # Counted rather than read off the report's total, which an advice line
        # moves without changing what an entry looks like.
        self.assertEqual(sum(ln.startswith("  [") for ln in message.splitlines()), 2)

    def test_no_match_message_when_every_guard_tree_raised(self):
        # Nothing here rejected the call, so the usual "add a ModelInput" advice
        # is wrong: an input covering this call would guard the same dict and
        # raise the same way.
        model = torch.compile(DictBranchModule(), fullgraph=True, backend="eager")
        x = torch.randn(3, 3)
        model._aot_compile([ModelInput(args=(x, {}), kwargs={}, contexts=[])])
        with self.assertRaises(RuntimeError) as ctx:
            model(x, {RaisesOnCompare(): 1})
        message = str(ctx.exception)
        self.assertIn("Every guard tree raised", message)
        self.assertNotIn("Add a ModelInput", message)

    def test_no_match_report_claims_no_raise_for_an_artifact_with_no_inputs(self):
        # deserialize() takes a list of any length, the empty one included, where
        # aot_compile_module refuses it -- so the line above has to stand on the
        # raises on record and not on the absence of an entry that answered,
        # which is vacuously true of a report with no entries at all. The one
        # advice that fits an artifact holding no inputs is to add one, which a
        # gate on an entry that answered drops for the same vacuous reason.
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
        # The other half of the same handler: here both dispatch passes got a
        # clean answer out of the tree -- the DICT_NOT_CONTAINS guard found the
        # key and rejected the call -- and only the re-check that explains why
        # raised. That is an ordinary mismatch missing its reason, so the advice a
        # new ModelInput would satisfy still applies and "every guard tree raised"
        # would be the opposite of what happened.
        model = torch.compile(DictBranchModule(), fullgraph=True, backend="eager")
        x = torch.randn(3, 3)
        model._aot_compile([ModelInput(args=(x, {}), kwargs={}, contexts=[])])
        key = HitsThenRaises(hits=2)
        with self.assertRaises(RuntimeError) as ctx:
            model(x, {key: 1})
        message = str(ctx.exception)
        # The report asked a third time rather than reusing a dispatch answer.
        # A lower bound: how often one evaluation probes the key is guard
        # codegen's business, and the evaluation count itself is pinned
        # codegen-independently by the stub tests' `checks`. hits=2 does rest on
        # one probe per evaluation, though: codegen that probed the key twice per
        # check() would move the raise into dispatch pass 2 and fail the two
        # assertions below on a probe count rather than on report logic.
        self.assertGreaterEqual(key.compares, 3)
        # Only a raise from dispatch is chained onto the report, so a report raise
        # has to carry what it was raised from in the line itself.
        raised = "[0] <guard check raised ValueError: boom on compare 3 (through the guard tree's pybind boundary)>"
        self.assertIn(raised, message)
        self.assertNotIn("Every guard tree raised", message)
        self.assertIn("Add a ModelInput", message)

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
        self.assertIn(raised, message)
        self.assertNotIn("pybind boundary", message)
        self.assertIn("Every guard tree raised", message)

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
        self.assertIn(raised, message)
        self.assertNotIn("the caller was handling this", message)
        self.assertNotIn("pybind boundary", message)

    def test_aot_compile_module_no_match_does_not_suppress_a_handled_exception(self):
        # An ordinary no-match, with no raise to chain, has to raise bare rather
        # than `from None`: `from None` sets __suppress_context__, which erases
        # the exception a call made inside an `except` block was handling from the
        # traceback the user reads.
        class Boom(Exception):
            pass

        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [
                ModelInput(
                    args=(torch.randn(3, 3, dtype=torch.float32),),
                    kwargs={},
                    contexts=[],
                )
            ]
        )
        try:
            raise Boom("the caller was handling this")
        except Boom:
            with self.assertRaises(RuntimeError) as ctx:
                model(torch.randn(3, 3, dtype=torch.float16))
        self.assertIn("No AOT compiled graph matched this call", str(ctx.exception))
        self.assertIs(ctx.exception.__suppress_context__, False)
        self.assertIsInstance(ctx.exception.__context__, Boom)

    def test_no_match_message_keeps_the_advice_for_every_entry(self):
        # One entry names a missing global and the other is a plain mismatch, and
        # neither advice covers the other's entry: defining AOT_BRANCH_SCALE
        # cannot make mode=2 satisfy [1]'s L['mode'] == 1, and a new ModelInput
        # for mode=2 would not resolve the global [1] failed on. Reporting only
        # one of them asserts something untrue about the whole call.
        model = torch.compile(
            ModeBranchGlobalModule(),
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
        g = globals()
        saved = g.pop("AOT_BRANCH_SCALE")
        try:
            with self.assertRaises(RuntimeError) as ctx:
                model(x, 2)
            message = str(ctx.exception)
        finally:
            g["AOT_BRANCH_SCALE"] = saved
        self.assertIn("[0] L['mode'] == 0", message)
        self.assertIn("[1] KeyError on G['AOT_BRANCH_SCALE']", message)
        self.assertIn("For [1]: a guarded global is missing", message)
        self.assertIn("the module the compiled function was traced in", message)
        self.assertIn("Add a ModelInput", message)

    def test_no_match_message_keeps_the_advice_beside_a_raise(self):
        # The same independence with a raise in play: [0] raised and has to be
        # fixed or dropped, [1] names a global the load can define, and [1]'s
        # rejection is what a new ModelInput would answer. All three lines stand
        # on their own entries, in the order the report emits them.
        model = torch.compile(
            ModeBranchGlobalModule(),
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

        class Raises(NeverReChecked):
            def check(self, f_locals):
                raise RuntimeError("the scanned tree is unhappy")

        model.forward.compiled_results[0]._artifacts.guard_manager = Raises()
        g = globals()
        saved = g.pop("AOT_BRANCH_SCALE")
        self.addCleanup(g.__setitem__, "AOT_BRANCH_SCALE", saved)
        with self.assertRaises(RuntimeError) as ctx:
            model(x, 2)
        message = str(ctx.exception)
        lines = message.splitlines()
        raised = "  [0] <guard check raised RuntimeError: the scanned tree is unhappy>"
        self.assertIn(raised, lines)
        self.assertIn("[1] KeyError on G['AOT_BRANCH_SCALE']", message)

        def line_of(text):
            # Located rather than searched for, so a dropped advice line reads as
            # a message diff and not as a StopIteration from inside the test.
            found = [i for i, ln in enumerate(lines) if text in ln]
            self.assertEqual(len(found), 1, message)
            return found[0]

        hint = line_of("a guarded global is missing")
        fix = line_of("fix or drop that artifact")
        add = line_of("Add a ModelInput covering this call")
        self.assertLess(hint, fix)
        self.assertLess(fix, add)

    def test_no_match_message_keeps_the_advice_beside_a_withheld_opt_out(self):
        # The other pairing: with an opted-out input in the list the fix-or-drop
        # advice comes from the withheld branch instead, and it still has to
        # coexist with a missing global named by a third entry. [0] raised, [1]
        # names the global, [2] opted out and was withheld by [0]'s raise.
        model = torch.compile(
            ModeBranchGlobalModule(),
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        x = torch.randn(3, 3)
        model._aot_compile(
            [
                ModelInput(args=(x, 0), kwargs={}, contexts=[]),
                ModelInput(args=(x, 1), kwargs={}, contexts=[]),
                # A dtype no call below can satisfy: an opted-out result is still
                # evaluated by the scan, and one whose tree ACCEPTS is served
                # there rather than withheld.
                ModelInput(
                    args=(torch.randn(3, 3, dtype=torch.float64), 0),
                    kwargs={},
                    contexts=[],
                ),
            ]
        )

        class Raises(NeverReChecked):
            def check(self, f_locals):
                raise RuntimeError("the scanned tree is unhappy")

        results = model.forward.compiled_results
        results[0]._artifacts.guard_manager = Raises()
        results[2].disable_guard_check()
        g = globals()
        saved = g.pop("AOT_BRANCH_SCALE")
        self.addCleanup(g.__setitem__, "AOT_BRANCH_SCALE", saved)
        with self.assertRaises(RuntimeError) as ctx:
            model(x, 2)
        message = str(ctx.exception)
        lines = message.splitlines()
        raised = "  [0] <guard check raised RuntimeError: the scanned tree is unhappy>"
        self.assertIn(raised, lines)
        self.assertIn("[1] KeyError on G['AOT_BRANCH_SCALE']", message)
        self.assertIn(
            "  [2] <opted out of guard checks; withheld because [0]'s guard "
            "check raised>",
            lines,
        )

        def line_of(text):
            found = [i for i, ln in enumerate(lines) if text in ln]
            self.assertEqual(len(found), 1, message)
            return found[0]

        hint = line_of("a guarded global is missing")
        fix = line_of("fix or drop that artifact")
        add = line_of("Add a ModelInput covering this call")
        self.assertLess(hint, fix)
        self.assertLess(fix, add)
        self.assertIn("not a guard failure", lines[fix])

    def _install_global_probe(self, name, misses=0):
        # Re-keys this module's global `name` under a CountedKey. Restored by
        # cleanups, not a finally: nothing between the pop and the insert may
        # leave this dict without the name. addCleanup is LIFO, so the probe is
        # registered second to be removed first; reversed, the re-insert would
        # find the probe by __eq__ and store under it, and the pop would then
        # drop the name for good.
        g = globals()
        probe, saved = CountedKey(name, misses), g.pop(name)
        self.addCleanup(g.__setitem__, name, saved)
        self.addCleanup(g.pop, probe, None)
        g[probe] = saved
        return probe, saved

    def test_module_dispatch_evaluates_a_matching_tree_once(self):
        # The scan calls the matching result's declared `fn` field rather than the
        # result, whose __call__ would evaluate the guards that just passed a
        # second time: ONE evaluation per matching call.
        mod = GlobalConfigModule()
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        x = torch.randn(4, 8)
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        probe, _ = self._install_global_probe("GLOBAL_POOLING_CONFIG")
        out = model(x)
        # Read before the cleanup: deleting probe from g looks it up, and only
        # CPython's identity-first key compare keeps that off the count.
        compares = probe.compares
        self.assertEqual(out, mod(x))
        # The guarded lookup happens once per evaluation, and running the graph
        # does not read the global at all, so this counts the evaluations.
        self.assertEqual(compares, 1)

    def test_module_dispatch_binds_a_call_once_for_results_sharing_a_signature(self):
        # Every result aot_compile_module produces carries an equal signature and
        # the same closure cells, so one bind serves them all: a call that scans
        # every result, matched or not, binds once rather than once per result.
        mod = ScaleModule()
        model = torch.compile(mod, fullgraph=True, backend="eager")
        dtypes = (torch.float32, torch.float64, torch.int64, torch.bfloat16)
        xs = [torch.ones(3, 3, dtype=dtype) for dtype in dtypes]
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[]) for x in xs])
        compiled = model.forward
        self.assertTrue(compiled._binds_alike(tuple(compiled.compiled_results)))
        binds = []
        bind = AOTCompiledFunction.prepare_f_locals

        def counted(result, *args, **kwargs):
            binds.append(result)
            return bind(result, *args, **kwargs)

        with patch.object(AOTCompiledFunction, "prepare_f_locals", counted):
            self.assertEqual(model(xs[3]), mod(xs[3]))
            self.assertEqual(len(binds), 1)
            binds.clear()
            with self.assertRaises(RuntimeError) as ctx:
                model(torch.ones(3, 3, dtype=torch.float16))
            self.assertEqual(len(binds), 1)
        # One entry per result off that single bind, whatever else the report
        # carries.
        lines = str(ctx.exception).splitlines()
        self.assertEqual(sum(line.startswith("  [") for line in lines), len(xs))
        self.assertIn("Add a ModelInput", str(ctx.exception))

    def test_module_dispatch_binds_per_result_when_signatures_differ(self):
        # Results assembled by hand need not agree on a signature, and the guards
        # of each read the names ITS signature bound (L['y'] here), so such a
        # model binds each result on its own.
        mod = ScaleModule()
        x = torch.randn(3, 3)
        model = torch.compile(mod, fullgraph=True, backend="eager")
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        doubled = model.forward.compiled_results

        def triple(self, y):
            return y * 3

        mod.forward = types.MethodType(triple, mod)
        model = torch.compile(mod, fullgraph=True, backend="eager")
        model._aot_compile([ModelInput(args=(x.double(),), kwargs={}, contexts=[])])
        combined = AOTCompiledModel(mod, doubled + model.forward.compiled_results)
        self.assertFalse(combined._binds_alike(tuple(combined.compiled_results)))
        self.assertEqual(combined(x.double()), x.double() * 3)
        self.assertEqual(combined(x), x * 2)

    def test_module_dispatch_rebinds_after_a_result_is_appended(self):
        # compiled_results is a public list, so the one-bind-per-call decision is
        # made again when its contents change: a result appended with its own
        # default for `mode` is bound from that default, not from [0]'s.
        mod, x = ScaleModule(), torch.randn(3, 3)
        triples, doubles = make_mode_default_forward(1), make_mode_default_forward(0)
        combined = AOTCompiledModel(mod, [aot_compile_forward(mod, triples, x)])
        self.assertTrue(combined._binds_alike(tuple(combined.compiled_results)))
        combined.compiled_results.append(aot_compile_forward(mod, doubles, x.double()))
        # Without the re-decision this call would be bound from [0], where mode
        # reads 1, and [1]'s `mode == 0` guard would reject it.
        self.assertEqual(combined(x.double()), x.double() * 2)
        self.assertFalse(combined._binds_alike(tuple(combined.compiled_results)))

    def test_module_dispatch_judges_a_replaced_result_on_its_own_binding(self):
        # [1] is compiled for mode=1 but defaults mode to 0, so a call leaving
        # mode out has no graph: bound from [0], where mode defaults to 1, its
        # guards would pass and serve x * 3 for a call eager answers x * 2.
        mod, x = ScaleModule(), torch.randn(3, 3)
        triples, doubles = make_mode_default_forward(1), make_mode_default_forward(0)
        results = [aot_compile_forward(mod, triples, t) for t in (x, x.double())]
        combined = AOTCompiledModel(mod, results)
        self.assertTrue(combined._binds_alike(tuple(combined.compiled_results)))
        combined.compiled_results[1] = aot_compile_forward(mod, doubles, x.double(), 1)
        self.assertEqual(combined(x.double(), 1), x.double() * 3)
        with self.assertRaisesRegex(RuntimeError, "No AOT compiled graph matched"):
            combined(x.double())
        self.assertFalse(combined._binds_alike(tuple(combined.compiled_results)))

    def test_module_dispatch_judges_only_the_results_a_call_began_with(self):
        # The same replacement, made while the call is in flight: [0]'s check()
        # appends [1], compiled for mode=1 but defaulting mode to 0. A scan that
        # read the live list would reach [1] on this call and judge it on the
        # binding decided over [0] alone, where mode reads 1, and serve x * 3 for
        # a call eager answers x * 2. The call judges the results it began with
        # and the next call decides over both.
        mod, x = ScaleModule(), torch.randn(3, 3)
        triples, doubles = make_mode_default_forward(1), make_mode_default_forward(0)
        first = aot_compile_forward(mod, triples, x)
        later = aot_compile_forward(mod, doubles, x.double(), 1)
        combined = AOTCompiledModel(mod, [first])
        manager = first._live_guard_manager()
        check = manager.check

        def appending_check(f_locals):
            combined.compiled_results[1:] = [later]
            return check(f_locals)

        with patch.object(manager, "check", appending_check):
            with self.assertRaisesRegex(RuntimeError, "No AOT compiled graph matched"):
                combined(x.double())
        self.assertIs(combined.compiled_results[1], later)
        self.assertEqual(combined(x.double(), 1), x.double() * 3)
        self.assertFalse(combined._binds_alike(tuple(combined.compiled_results)))

    def test_module_dispatch_never_pairs_new_contents_with_a_stale_verdict(self):
        # A call entering while another thread is still deciding over the
        # appended list must not find the new contents already published beside
        # the old True: the decider is held inside its first _binding_key, before
        # it publishes anything, and the call made in that window decides for
        # itself and binds [1] from its own default.
        mod, x = ScaleModule(), torch.randn(3, 3)
        triples, doubles = make_mode_default_forward(1), make_mode_default_forward(0)
        combined = AOTCompiledModel(mod, [aot_compile_forward(mod, triples, x)])
        self.assertTrue(combined._binds_alike(tuple(combined.compiled_results)))
        combined.compiled_results.append(aot_compile_forward(mod, doubles, x.double()))
        entered, release = threading.Event(), threading.Event()
        binding_key = torch._dynamo.aot_compile._binding_key

        def held(artifacts):
            if threading.current_thread() is decider:
                entered.set()
                release.wait()
            return binding_key(artifacts)

        decider = threading.Thread(
            target=combined._binds_alike, args=(tuple(combined.compiled_results),)
        )
        self.addCleanup(decider.join)
        self.addCleanup(release.set)
        with patch("torch._dynamo.aot_compile._binding_key", held):
            decider.start()
            self.assertTrue(entered.wait(timeout=60))
            # A stale True here would bind from [0], where mode reads 1, and [1]'s
            # `mode == 0` guard would reject the call.
            self.assertEqual(combined(x.double()), x.double() * 2)
            release.set()
            decider.join()
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
        mod, x = ScaleModule(), torch.randn(3, 3)
        triples, doubles = make_mode_default_forward(1), make_mode_default_forward(0)
        first = aot_compile_forward(mod, triples, x)
        second = aot_compile_forward(mod, doubles, x.double(), 1)
        entered, release = threading.Event(), threading.Event()
        decider = threading.Thread(target=lambda: combined._binds_alike((first,)))
        self.addCleanup(release.set)

        class Held(AOTCompiledModel):
            def __setattr__(self, name, value):
                super().__setattr__(name, value)
                if threading.current_thread() is decider and not entered.is_set():
                    entered.set()
                    release.wait()

        combined = Held(mod, [first, second])
        self.assertFalse(combined._binds_alike(tuple(combined.compiled_results)))
        binds_alike = AOTCompiledModel._binds_alike

        def racing(model, results):
            if threading.current_thread() is decider:
                return binds_alike(model, results)
            self.assertIs(combined.compiled_results.pop(), second)
            decider.start()
            self.assertTrue(entered.wait(timeout=60))
            return binds_alike(model, results)

        with patch.object(AOTCompiledModel, "_binds_alike", racing):
            with self.assertRaisesRegex(RuntimeError, "No AOT compiled graph matched"):
                combined(x.double())
            release.set()
            decider.join()
        self.assertTrue(combined._binds_alike(tuple(combined.compiled_results)))

    def test_module_dispatch_serves_a_call_the_guard_tree_accepts(self):
        # A first check() can reject a call the same tree accepts on its next
        # evaluation, which is what it does for real when the dict-tag fast path
        # answers false without ever running the tree. That rejection is not an
        # answer about the call, so a second pass has to rescue it -- here with a
        # probe that misses the guarded global once and finds it after. The
        # rescuable result is [1], which the fall-through this dispatch replaced
        # never reached: it re-checked compiled_results[0] and raised [0]'s
        # L['mode'] == 0.
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
        probe, saved = self._install_global_probe("AOT_BRANCH_SCALE", misses=1)
        out = model(x, 1)
        compares = probe.compares
        self.assertEqual(out, x * saved)
        # Only [1]'s tree names the global and it looks the name up once per
        # evaluation, so the rescue cost exactly one more evaluation.
        self.assertEqual(compares, 2)

    def test_module_dispatch_rechecks_an_opted_out_result_whose_tree_accepts(self):
        # The same false rejection of [1], with both results opted out. A
        # re-check that skipped opted-out results would leave the call to the
        # last resort, which serves the FIRST opted-out result: [0], whose
        # L['mode'] == 0 guard genuinely fails this call.
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
        for result in model.forward.compiled_results:
            result.disable_guard_check()
        probe, saved = self._install_global_probe("AOT_BRANCH_SCALE", misses=1)
        out = model(x, 1)
        compares = probe.compares
        self.assertEqual(out, x * saved)
        # The second evaluation is the re-check reading [1]'s global again; the
        # last resort would have served [0] without one.
        self.assertEqual(compares, 2)

    def test_module_dispatch_serves_an_opted_out_result_from_any_position(self):
        # With [0] still checked and [1] opted out, a call neither guards is
        # served by [1]: the fall-through this dispatch replaced re-entered
        # compiled_results[0] alone and raised its guard error.
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

    def test_module_dispatch_rechecks_before_honouring_an_opt_out(self):
        # [0] opted out, [1] checked and falsely rejected once: the re-check
        # finds [1]'s real match before the last resort can hand the call to [0].
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
        model.forward.compiled_results[0].disable_guard_check()
        probe, saved = self._install_global_probe("AOT_BRANCH_SCALE", misses=1)
        out = model(x, 1)
        compares = probe.compares
        self.assertEqual(out, x * saved)
        self.assertEqual(compares, 2)

    def test_module_dispatch_shares_a_binding_past_a_tensor_default(self):
        # Every result's signature carries the same default object, so the
        # results share a binding; deciding that through Signature equality
        # raised `Boolean value of Tensor with more than one value is ambiguous`
        # on the first call, for _aot_compile and deserialize alike.
        mod = TensorDefaultModule()
        model = torch.compile(mod, fullgraph=True, backend="eager")
        xs = [torch.randn(3, 3), torch.randn(3, 3, dtype=torch.float64)]
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[]) for x in xs])
        compiled = model.forward
        self.assertTrue(compiled._binds_alike(tuple(compiled.compiled_results)))
        for x in xs:
            self.assertEqual(model(x), x * 2)
        # Each result unpickles on its own, so the loaded results hold distinct
        # default objects and bind per result: a false negative, not a false share.
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
        # result too.
        loaded = AOTCompiledModel.deserialize(mod, model.forward.serialize())
        self.assertFalse(loaded._binds_alike(tuple(loaded.compiled_results)))
        for x in xs:
            self.assertEqual(loaded(x), x * 3)

    def test_no_match_message_when_a_guard_answers_inconsistently(self):
        # Both dispatch passes ran [1]'s whole tree and both rejected the call,
        # so an accept while the report asks why contradicts them rather than
        # correcting them: neither the guards it just passed nor "add a
        # ModelInput" says anything true about that entry.
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
        g = globals()
        probe = CountedKey("AOT_BRANCH_SCALE", misses=2)
        saved = g.pop("AOT_BRANCH_SCALE")
        self.addCleanup(g.__setitem__, "AOT_BRANCH_SCALE", saved)
        self.addCleanup(g.pop, probe, None)
        g[probe] = saved
        with self.assertRaises(RuntimeError) as ctx:
            model(x, 1)
        message = str(ctx.exception)
        compares = probe.compares
        # Two rejections in dispatch, then the report's accept.
        self.assertEqual(compares, 3)
        self.assertIn(
            "[1] <guards did not accept this call in dispatch and accepted", message
        )
        # [0] is a real mismatch, so its advice still applies to the call.
        self.assertIn("[0] L['mode'] == 0", message)
        self.assertIn("Add a ModelInput", message)

    def test_no_match_message_after_a_raise_counts_no_second_rejection(self):
        # The same line is reachable with ONE rejection on record once a raise is
        # tolerated: pass 1 raised, pass 2 rejected, the report accepted. The
        # line has to describe what dispatch did without claiming a count it
        # never took -- the raise itself survives only in the chain. And the
        # caveat beside the ModelInput advice keys on what dispatch recorded,
        # not on the lines the report printed: the one rejection dispatch got
        # followed the raise, so the advice carries it although this report
        # quotes no rejection.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )

        class RaisesThenRejectsThenAccepts:
            def __init__(self):
                self.checks = 0

            def check(self, f_locals):
                self.checks += 1
                if self.checks == 1:
                    raise RuntimeError("the first pass is unhappy")
                return False

            def check_verbose(self, f_locals):
                return types.SimpleNamespace(result=True, verbose_code_parts=[])

        stub = RaisesThenRejectsThenAccepts()
        model.forward.compiled_results[0]._artifacts.guard_manager = stub
        with self.assertRaises(RuntimeError) as ctx:
            served = model(torch.randn(3, 3))
            self.fail(f"dispatch served {served[0, 0].item()}, not a raise")
        message = str(ctx.exception)
        # One raise and one rejection, so "twice" would be an invented count.
        self.assertEqual(stub.checks, 2)
        accepted = (
            "  [0] <guards did not accept this call in dispatch and accepted it "
            "here: a guard that does not answer consistently, or a tag-safe fast "
            "path that refused without running the tree>"
        )
        self.assertIn(accepted, message.splitlines())
        self.assertNotIn("twice", message)
        self.assertEqual(str(ctx.exception.__cause__), "the first pass is unhappy")
        self.assertIn("Add a ModelInput", message)
        self.assertIn("advice above rests only on rejections", message)
        self.assertNotIn("Every guard tree raised", message)

    def test_no_match_message_qualifies_the_advice_when_the_re_check_raises(self):
        # The other line the re-check can put where dispatch saw a rejection: it
        # raised. The report's only entry line is then that raise, the re-check's
        # own, and the caveat beside the ModelInput advice is still emitted: it
        # is about the rejection dispatch acted on, not about that line.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )

        class RaisesThenRejectsThenRaises:
            def __init__(self):
                self.checks = 0

            def check(self, f_locals):
                self.checks += 1
                if self.checks == 1:
                    raise RuntimeError("the first pass is unhappy")
                return False

            def check_verbose(self, f_locals):
                raise RuntimeError("the re-check is unhappy")

        stub = RaisesThenRejectsThenRaises()
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
        self.assertIn("advice above rests only on rejections", message)
        self.assertNotIn("Every guard tree raised", message)
        self.assertEqual(str(ctx.exception.__cause__), "the first pass is unhappy")

    def test_no_match_message_advises_an_input_for_a_missing_global(self):
        # A report every entry of which names a missing global still has to
        # advise an input: guards are installed in sorted(guards) order, so the
        # root's G accessor runs before an nn.Module attribute guard and the
        # KeyError is all the report gets to see -- the mismatch it hides is
        # self.mode, and an input captured for the mode == 2 branch would not
        # read AOT_BRANCH_SCALE at all. Withholding the advice here leaves the
        # call with no actionable line, and which line the report withholds must
        # not turn on whether an unrelated global happens to be defined.
        self._hide_leaked_dynamo_globals()
        mod = SelfModeBranchGlobalModule(1)
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        x = torch.randn(3, 3)
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        mod.mode = 2
        g = globals()
        saved = g.pop("AOT_BRANCH_SCALE")
        try:
            with self.assertRaises(RuntimeError) as ctx:
                model(x)
            message = str(ctx.exception)
        finally:
            g["AOT_BRANCH_SCALE"] = saved
        self.assertIn("[0] KeyError on G['AOT_BRANCH_SCALE']", message)
        self.assertNotIn("L['self'].mode", message)
        self.assertIn("For [0]: a guarded global is missing", message)
        # The footer is unconditional at this tree; this fences a future gate on
        # it (the mixed case is keeps_the_advice_for_every_entry's).
        self.assertIn("Add a ModelInput", message)
        # The mismatch the entry above hid: with the global back, the same call
        # fails on self.mode alone. Second, not first, because a failed check()
        # re-sorts the root's children by fail count, which would have put this
        # guard ahead of the G accessor for the call above.
        with self.assertRaises(RuntimeError) as ctx:
            model(x)
        self.assertIn("[0] L['self'].mode == 1", str(ctx.exception))

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
            with self.assertRaises(RuntimeError):
                reloaded(x)
            # The dict the load actually resolved: the guards hold it by
            # reference, so defining the name here is what lets them resolve.
            # The graph goes on reading the serialized copy, equal to this
            # tensor -- the bytecode globals were built once, at load, and a
            # live value is substituted only for a name the scope had then.
            ns["AOT_HERMETIC_WEIGHT"] = saved
            self.assertEqual(reloaded(x), x @ saved)
        finally:
            g["AOT_HERMETIC_WEIGHT"] = saved

    def test_no_match_message_hint_stays_neutral_for_a_supplied_scope(self):
        # deserialize skips _resolve_guard_scope when the caller passes
        # guard_globals=, so the guards hold THAT dict rather than the globals of
        # the function model.forward resolves to. Naming forward here would send
        # the reader to a dict where the name is already defined and dispatch
        # still fails -- the assertions after the wording measure both halves.
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
            "missing from the live scope this artifact was loaded against", message
        )
        self.assertNotIn("instance's forward", message)
        # The guards hold the supplied dict, not the one the class's forward
        # owns: this module's, where the name read below resolves.
        self.assertIs(compiled.compiled_results[0]._guard_globals, scope)
        self.assertIs(HermeticModule.forward.__globals__, globals())
        scope["AOT_HERMETIC_WEIGHT"] = AOT_HERMETIC_WEIGHT
        self.assertEqual(compiled(x), x @ AOT_HERMETIC_WEIGHT)

    def test_no_match_report_resolves_forward_only_for_a_supplied_scope(self):
        # An in-process capture keeps the CAPTURED scope, whose hint never names
        # forward, so the report has no reason to resolve it -- and resolving it
        # runs user code: get_traced_fn formats the forward it refuses, a partial
        # over this module, whose extra_repr raises past the (RuntimeError,
        # AttributeError) that _resolve_guard_scope catches. Dispatch itself
        # never calls the rebound forward, so the rebind reaches only the report.
        # The catch alone would keep this report arriving, so the gate is pinned
        # by counting resolves rather than by the wording.
        self._hide_leaked_dynamo_globals()
        mod = RaisingReprModule()
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        x = torch.randn(3, 3)
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        mod.forward = functools.partial(HermeticModule.forward, mod)
        with self.assertRaises(ValueError):
            repr(mod.forward)
        g = globals()
        saved = g.pop("AOT_HERMETIC_WEIGHT")
        resolve = patch(
            "torch._dynamo.aot_compile._resolve_guard_scope", wraps=_resolve_guard_scope
        )
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
            with self.assertRaises(RuntimeError) as ctx:
                compiled(x)
            message = str(ctx.exception)
        finally:
            g["AOT_HERMETIC_WEIGHT"] = saved
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

    def test_no_match_report_names_the_results_the_dispatch_judged(self):
        # compiled_results is public, and the report indexes the binding the
        # dispatch built, one per result the call began with. [0]'s check()
        # appends a result mid-call: a report that re-read the list would ask for
        # a binding the dispatch never made, and an IndexError would take its
        # place. The appended result is the next call's to judge.
        mod, x = ScaleModule(), torch.randn(3, 3)
        first = aot_compile_forward(mod, make_scaling_forward(2), x)
        later = aot_compile_forward(mod, make_scaling_forward(3), x)
        combined = AOTCompiledModel(mod, [first])
        manager = first._live_guard_manager()
        check = manager.check

        def appending_check(f_locals):
            combined.compiled_results[1:] = [later]
            return check(f_locals)

        with patch.object(manager, "check", appending_check):
            with self.assertRaises(RuntimeError) as ctx:
                combined(x.double())
        lines = str(ctx.exception).splitlines()
        self.assertIn("Tried 1 compiled input(s)", lines[0])
        self.assertEqual(sum(line.startswith("  [") for line in lines), 1)
        with self.assertRaises(RuntimeError) as ctx:
            combined(x.double())
        self.assertIn("Tried 2 compiled input(s)", str(ctx.exception))

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
        # serves the first result whose guards pass, in index order, and an
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
        # itself ordered: it serves the FIRST opted-out result, so a call no
        # artifact guards gets the graph the earliest opted-out ModelInput was
        # traced for. A scan that skipped opted-out results is not pinned here,
        # for two different reasons: in the first case the result that must
        # serve is the CHECKED one, which the skip never applies to, so [1] is
        # served by the scan itself; in the second both trees fail on "other"
        # anyway and the answer this ordering demands is the first opted-out
        # result, which the last resort serves either way. What tells a
        # skipping scan apart is an opted-out result whose tree the scan would
        # accept: test_aot_compile_module_opted_out_result_keeps_its_place_in_the_scan
        # below pins that with honest guards, and
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
                msg="the last resort must serve the first opted-out result",
            )

    def test_aot_compile_module_raising_tree_does_not_reach_an_opted_out_result(self):
        # A tree that raised never returned False, so it never rejected the
        # call; letting an opted-out result answer on the strength of that
        # serves a graph whose guards never passed. A plain call raises here, so
        # the only honest answers are that raise or a report naming it.
        x = torch.ones(3, 3)
        evil = {RaisesOnCompare(): 1}
        with self.assertRaisesRegex(ValueError, "boom from __eq__"):
            DictBranchModule()(x, evil)
        model = torch.compile(DictBranchModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [
                ModelInput(args=(x, {}), kwargs={}, contexts=[]),
                ModelInput(args=(x, None), kwargs={}, contexts=[]),
            ]
        )
        # [1] was traced with d=None, so opting it out arms x * 5 as the last
        # resort: without the _guard_check_enabled gate on the scan's raises,
        # reading the raise as a plain non-match would silently serve it. (The
        # parent never gets this far -- the raise leaves the scan.)
        model.forward.compiled_results[1].disable_guard_check()
        with self.assertRaises(RuntimeError) as ctx:
            served = model(x, evil)
            self.fail(f"dispatch served {served[0, 0].item()}, not a raise")
        message = str(ctx.exception)
        raised = "[0] <guard check raised ValueError: boom from __eq__ (through the guard tree's pybind boundary)>"
        self.assertIn(raised, message)
        # This is the only path that reports an opted-out result, and its guards
        # say nothing about why the call was refused: quoting them would name a
        # rejection nobody asked for, and a ModelInput cannot cover a raise. What
        # the report has to say is which artifact to fix.
        withheld = (
            "[1] <opted out of guard checks; withheld because [0]'s guard check raised>"
        )
        self.assertIn(withheld, message)
        self.assertIn("[0]'s raise, not a guard failure, is what withheld", message)
        self.assertNotIn("Add a ModelInput", message)
        # The withheld line above has already said what happened to [1], so the
        # every-tree-raised footer must not be emitted as well: with `not
        # withheld` dropped from its gate the report contradicts itself.
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

    def test_no_match_message_quotes_a_tree_that_raised_then_answered(self):
        # A raise still withholds the opted-out last resort: dispatch cannot tell
        # this flavour, which returns with an exception merely set and leaks
        # nothing, from a C++ throw that skips the non-RAII TorchFunction TLS
        # restore and makes a later check of a tree holding a GLOBAL_STATE guard
        # reject on that guard, so it holds the opt-out back on any raise. But the
        # raise is no longer all the report can say about that entry -- the tree
        # did reject this call on the second pass, and that rejection is a real
        # answer, so the report quotes the guard it rejected on and the advice
        # that rejection earns, and names the raise as what withheld the opt-out
        # and what to fix.
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
        chained = []
        cause: BaseException | None = ctx.exception
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
        # caller. Refusing would not repair what the raise may already have done
        # either -- a throw out of C++ leaves TorchFunction disabled on this
        # thread whichever way dispatch answers -- so the raise is served over,
        # and warned about, rather than becoming a second failure mode.
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

        class RaisingGuardManager:
            def check(self, f_locals):
                raise RuntimeError("guard tree is unhappy")

        # A stub rather than the DICT_NOT_CONTAINS leaf the tests above use: what
        # dispatch answers does not depend on how the tree raised.
        results = model.forward.compiled_results
        results[0]._artifacts.guard_manager = RaisingGuardManager()
        logger = "torch._dynamo.aot_compile"
        with self.assertLogs(logger, level="WARNING") as logs:
            with _set_pooling("mean"):
                served = model(x)
        self.assertEqual(served, expected)
        # Nothing else records the raise on this path: the report is never built.
        warned = "\n".join(logs.output)
        self.assertIn(
            "[0]'s guard check raised RuntimeError: guard tree is unhappy", warned
        )
        self.assertIn("dispatch served [1]", warned)
        self.assertIn("Fix or drop input [0]", warned)

    def test_aot_compile_module_last_resort_warns_that_its_own_tree_raised(self):
        # The other path that serves a graph after swallowing a raise: a raise
        # from the opted-out result itself vetoes nothing -- nobody asked about
        # its guards -- so it is still served, and nothing but the warning says
        # its tree was evaluated by the scan and threw.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )

        class RaisingGuardManager:
            def check(self, f_locals):
                raise RuntimeError("guard tree is unhappy")

        result = model.forward.compiled_results[0]
        result._artifacts.guard_manager = RaisingGuardManager()
        result.disable_guard_check()
        x = torch.randn(3, 3)
        logger = "torch._dynamo.aot_compile"
        with self.assertLogs(logger, level="WARNING") as logs:
            served = model(x)
        self.assertEqual(served, x * 2)
        warned = "\n".join(logs.output)
        self.assertIn(
            "[0]'s guard check raised RuntimeError: guard tree is unhappy", warned
        )
        self.assertIn("dispatch served [0]", warned)
        # Nobody acts on this result's rejections, so the stale-state clause an
        # enabled raiser gets would name a cost it cannot pay; what its raise
        # costs is the scan match.
        self.assertIn("reachable only through the last resort", warned)
        self.assertNotIn("relational guard state", warned)

    def test_aot_compile_module_last_resort_warns_about_another_index(self):
        # The last resort serves the FIRST opted-out result, which need not be
        # the one that raised, so the warning reads the raiser off `raised` and
        # the served index off the loop rather than assuming they coincide.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [
                ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[]),
                ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[]),
            ]
        )

        class Rejects:
            def check(self, f_locals):
                return False

        class Raises:
            def check(self, f_locals):
                raise RuntimeError("the second tree is unhappy")

        results = model.forward.compiled_results
        results[0]._artifacts.guard_manager = Rejects()
        results[1]._artifacts.guard_manager = Raises()
        for result in results:
            result.disable_guard_check()
        x = torch.randn(3, 3)
        logger = "torch._dynamo.aot_compile"
        with self.assertLogs(logger, level="WARNING") as logs:
            served = model(x)
        self.assertEqual(served, x * 2)
        warned = "\n".join(logs.output)
        self.assertIn(
            "[1]'s guard check raised RuntimeError: the second tree is unhappy",
            warned,
        )
        self.assertIn("dispatch served [0]", warned)
        self.assertNotIn("[0]'s guard check raised", warned)

    def test_aot_compile_module_veto_and_report_survive_without_the_dict_leaf(self):
        # The stub counterpart of the raised-then-answered test above, whose
        # raise needs guards.cpp to keep letting an exception survive that leaf.
        # Same shape, no C++: [0] raises in the scan and rejects on the second
        # pass, which still withholds opted-out [1], and the report quotes the
        # rejection while the `from` chain keeps the raise.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [
                ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[]),
                ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[]),
            ]
        )

        class RaisesThenRejects:
            def __init__(self):
                self.checks = 0

            def check(self, f_locals):
                self.checks += 1
                if self.checks == 1:
                    raise RuntimeError("guard tree is unhappy")
                return False

            def check_verbose(self, f_locals):
                return types.SimpleNamespace(
                    result=False, verbose_code_parts=["stub guard rejected"]
                )

        class Rejects:
            def check(self, f_locals):
                return False

            # Never reached while the opt-out is reported before the re-check,
            # which the entry-line count below pins; defined so a regression
            # there quotes this rejection rather than an AttributeError the
            # report's handler would dress up as a raise from the guard tree.
            def check_verbose(self, f_locals):
                return types.SimpleNamespace(
                    result=False, verbose_code_parts=["stub guard rejected"]
                )

        results = model.forward.compiled_results
        stub = RaisesThenRejects()
        results[0]._artifacts.guard_manager = stub
        # [1] would match this call on its real guards, so the scan has to see a
        # rejection from it for the last resort to be what serves it -- or would,
        # without [0]'s raise to withhold it.
        results[1]._artifacts.guard_manager = Rejects()
        results[1].disable_guard_check()
        with self.assertRaises(RuntimeError) as ctx:
            served = model(torch.randn(3, 3))
            self.fail(f"dispatch served {served[0, 0].item()}, not a raise")
        message = str(ctx.exception)
        # The scan raised and the second pass answered: two checks, not one.
        self.assertEqual(stub.checks, 2)
        self.assertIn("[0] stub guard rejected", message)
        self.assertNotIn("[0] <guard check raised", message)
        withheld = (
            "[1] <opted out of guard checks; withheld because [0]'s guard check raised>"
        )
        self.assertIn(withheld, message)
        self.assertIn("[0]'s raise, not a guard failure, is what withheld", message)
        self.assertIn("Add a ModelInput", message)
        # The withheld line says why [1] was withheld, not that [0]'s rejection
        # followed its raise; the advice standing on that rejection says so.
        self.assertIn("advice above rests only on rejections", message)
        self.assertEqual(str(ctx.exception.__cause__), "guard tree is unhappy")
        # One line per input: the opt-out is described once, and reaching the
        # re-check for it would add a second line about the guards nobody asked
        # about.
        self.assertEqual(sum(ln.startswith("  [") for ln in message.splitlines()), 2)

    def test_no_match_message_quotes_the_last_raise_at_one_index(self):
        # Both dispatch passes evaluate an enabled tree, so one index can raise
        # twice with two different exceptions. The report quotes the later one,
        # which is the raise `unanswered` recorded and the raise the report's own
        # re-check line quotes for an entry that answered and then raised: one
        # rule for both lines rather than one for each.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )

        class RaisesTwice(NeverReChecked):
            def __init__(self):
                self.checks = 0

            def check(self, f_locals):
                self.checks += 1
                raise RuntimeError(f"unhappy on pass {self.checks}")

        stub = RaisesTwice()
        model.forward.compiled_results[0]._artifacts.guard_manager = stub
        with self.assertRaises(RuntimeError) as ctx:
            model(torch.randn(3, 3))
        self.assertEqual(stub.checks, 2)
        message = str(ctx.exception)
        raised = "  [0] <guard check raised RuntimeError: unhappy on pass 2>"
        self.assertIn(raised, message.splitlines())
        self.assertNotIn("unhappy on pass 1", message)
        self.assertEqual(str(ctx.exception.__cause__), "unhappy on pass 2")

    def test_no_match_message_chains_a_raise_the_advice_does_not_name(self):
        # The chain carries the FIRST index that raised and the advice names the
        # first ENABLED one, which are different entries when an opted-out result
        # raised first: [0] is reported as the opt-out it is and its line quotes
        # no exception text, so the chain is the only surviving record of its
        # raise. Reading raised[raiser] instead would delete it.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [
                ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[]),
                ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[]),
            ]
        )

        class Raises(NeverReChecked):
            def __init__(self, message):
                self.message = message

            def check(self, f_locals):
                raise RuntimeError(self.message)

        results = model.forward.compiled_results
        results[0]._artifacts.guard_manager = Raises("the opted-out tree is unhappy")
        results[0].disable_guard_check()
        results[1]._artifacts.guard_manager = Raises("the checked tree is unhappy")
        with self.assertRaises(RuntimeError) as ctx:
            served = model(torch.randn(3, 3))
            self.fail(f"dispatch served {served[0, 0].item()}, not a raise")
        message = str(ctx.exception)
        lines = message.splitlines()
        withheld = (
            "  [0] <opted out of guard checks; withheld because [1]'s guard "
            "check raised>"
        )
        self.assertIn(withheld, lines)
        checked = "  [1] <guard check raised RuntimeError: the checked tree is unhappy>"
        self.assertIn(checked, lines)
        self.assertIn("[1]'s raise, not a guard failure, is what withheld", message)
        self.assertNotIn("the opted-out tree is unhappy", message)
        self.assertEqual(str(ctx.exception.__cause__), "the opted-out tree is unhappy")

    def test_aot_compile_module_warning_unwraps_a_pybind_boundary_raise(self):
        # This warning is the only record of a swallowed raise on the serving
        # paths, and a tree that returns to pybind with an exception merely set
        # arrives as a SystemError whose own str() is the bound method's repr:
        # naming that says nothing about what raised, and keying the per-index
        # dedup on it collapses every such raise at one index into one line.
        x = torch.ones(3, 3)
        model = torch.compile(DictBranchModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [
                ModelInput(args=(x, {}), kwargs={}, contexts=[]),
                ModelInput(args=(x, None), kwargs={}, contexts=[]),
            ]
        )
        # [0]'s DICT_NOT_CONTAINS probes "foo" and raises through that boundary;
        # [1] guards d is None and rejects without probing at all, so opting [0]
        # out arms the last resort, whose own raise vetoes nothing.
        model.forward.compiled_results[0].disable_guard_check()

        class RaisesTypeErrorOnCompare:
            # RaisesOnCompare's collision, raising a different type: two defects
            # at one index, distinguishable only once the SystemError is unwrapped.
            def __hash__(self):
                return hash("foo")

            def __eq__(self, other):
                raise TypeError("boom from __eq__")

        logger = "torch._dynamo.aot_compile"
        with self.assertLogs(logger, level="WARNING") as logs:
            self.assertEqual(model(x, {RaisesOnCompare(): 1}), x * 2)
            self.assertEqual(model(x, {RaisesTypeErrorOnCompare(): 1}), x * 2)
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
        # Dispatch puts the state back, so the report is about the raise and not
        # about what the raise left behind -- with the restore removed, [0]'s line
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
        self.addCleanup(torch._C._set_torch_function_state, state)
        with self.assertRaises(RuntimeError) as ctx:
            model(nested)
        self.assertEqual(torch._C._get_torch_function_state(), state)
        message = str(ctx.exception)
        lines = message.splitlines()
        # One entry line for the one input: this is the only report content in
        # this file a test did not author, so it is where a raise text that
        # splits into two entries would show up. The throw's own text is pinned
        # by a substring, as the function path's sibling does: what ATen says is
        # not this test's subject.
        self.assertEqual(sum(ln.startswith("  [") for ln in lines), 1, message)
        entry = next(ln for ln in lines if ln.startswith("  ["))
        self.assertIn("[0] <guard check raised RuntimeError: ", entry)
        self.assertIn("doesn't support strides", entry)
        self.assertNotIn("GLOBAL_STATE changed", message)
        self.assertNotIn("Add a ModelInput", message)

    def test_no_match_report_restores_torch_function_after_a_throw(self):
        # The report's re-check is the second place a tree can throw out of C++,
        # and it runs on the way to raising, so a state left disabled there would
        # travel out with the report. Stubbed rather than thrown for real: the
        # trees that leak this way stay unanswered and never reach the re-check.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )

        class LeaksThenRaises:
            def check(self, f_locals):
                # Leaked here too, so the state the report puts back is the one
                # DISPATCH found and not one dispatch itself left behind.
                torch._C._set_torch_function_state(
                    torch._C._TorchFunctionState.ALL_DISABLED
                )
                return False

            def check_verbose(self, f_locals):
                raise RuntimeError("the re-check is unhappy")

        model.forward.compiled_results[0]._artifacts.guard_manager = LeaksThenRaises()
        state = torch._C._get_torch_function_state()
        self.addCleanup(torch._C._set_torch_function_state, state)
        with self.assertRaises(RuntimeError) as ctx:
            model(torch.randn(3, 3))
        self.assertEqual(torch._C._get_torch_function_state(), state)
        raised = "[0] <guard check raised RuntimeError: the re-check is unhappy>"
        self.assertIn(f"  {raised}", str(ctx.exception).splitlines())

    def test_aot_compile_function_restores_torch_function_after_a_throw(self):
        # load_compiled_function returns an AOTCompiledFunction, whose guard
        # check runs the same non-RAII restore the module path repairs above, so
        # a C++ throw leaves TorchFunction disabled on this thread and silently
        # stops a __torch_function__ subclass from dispatching afterwards. This
        # path propagates the throw rather than serving over it, so the state is
        # put back on the way out.
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
        self.addCleanup(torch._C._set_torch_function_state, state)
        with self.assertRaisesRegex(RuntimeError, "doesn't support strides"):
            loaded(nested)
        self.assertEqual(torch._C._get_torch_function_state(), state)

    def test_aot_compile_function_restores_torch_function_after_a_verbose_throw(self):
        # The function path's second evaluation, which runs only to explain a
        # rejection, is the other place a tree can throw out of C++ there, and it
        # throws on the way to raising, so a state left disabled would travel out
        # with the message. Stubbed as in the report's counterpart: the trees that
        # leak this way throw on the first check and never reach the second.
        self._hide_leaked_dynamo_globals()

        def fn(x):
            return x * 2

        x = torch.randn(3, 3)
        compiled_fn = torch.compile(fn, fullgraph=True, backend="eager").aot_compile(
            ((x,), {})
        )

        class LeaksThenRaises:
            def check(self, f_locals):
                return False

            def check_verbose(self, f_locals):
                torch._C._set_torch_function_state(
                    torch._C._TorchFunctionState.ALL_DISABLED
                )
                raise RuntimeError("the re-check is unhappy")

        compiled_fn._artifacts.guard_manager = LeaksThenRaises()
        state = torch._C._get_torch_function_state()
        self.addCleanup(torch._C._set_torch_function_state, state)
        with self.assertRaisesRegex(RuntimeError, "the re-check is unhappy"):
            compiled_fn(x)
        self.assertEqual(torch._C._get_torch_function_state(), state)

    def test_aot_compile_module_restores_torch_function_after_an_interrupt(self):
        # A KeyboardInterrupt through the tree leaves the TLS disabled exactly as
        # a C++ throw does, and `except Exception` catches neither it nor the
        # SystemExit a guard could raise, so the restore has to sit under a
        # handler that sees a BaseException. It is also not an answer about this
        # call: dispatch puts the state back and lets it travel out rather than
        # reading it as a non-match.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )

        class LeaksThenInterrupts:
            def check(self, f_locals):
                torch._C._set_torch_function_state(
                    torch._C._TorchFunctionState.ALL_DISABLED
                )
                raise KeyboardInterrupt("ctrl-c inside the tree")

        results = model.forward.compiled_results
        results[0]._artifacts.guard_manager = LeaksThenInterrupts()
        state = torch._C._get_torch_function_state()
        self.addCleanup(torch._C._set_torch_function_state, state)
        with self.assertRaisesRegex(KeyboardInterrupt, "ctrl-c inside the tree"):
            model(torch.randn(3, 3))
        self.assertEqual(torch._C._get_torch_function_state(), state)

    def test_no_match_report_restores_torch_function_after_an_interrupt(self):
        # The report's re-check, same two obligations: an interrupt there must
        # not leave the state disabled, and must not be turned into a report
        # line -- the call is being interrupted, not explained.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )

        class LeaksThenInterrupts:
            def check(self, f_locals):
                return False

            def check_verbose(self, f_locals):
                torch._C._set_torch_function_state(
                    torch._C._TorchFunctionState.ALL_DISABLED
                )
                raise KeyboardInterrupt("ctrl-c inside the tree")

        results = model.forward.compiled_results
        results[0]._artifacts.guard_manager = LeaksThenInterrupts()
        state = torch._C._get_torch_function_state()
        self.addCleanup(torch._C._set_torch_function_state, state)
        with self.assertRaisesRegex(KeyboardInterrupt, "ctrl-c inside the tree"):
            model(torch.randn(3, 3))
        self.assertEqual(torch._C._get_torch_function_state(), state)

    def test_aot_compile_function_restores_torch_function_after_an_interrupt(self):
        # The function path's guard check, which propagates whatever the tree
        # produced: an interrupt travels out either way, so the state is all this
        # pins.
        self._hide_leaked_dynamo_globals()

        def fn(x):
            return x * 2

        x = torch.randn(3, 3)
        compiled_fn = torch.compile(fn, fullgraph=True, backend="eager").aot_compile(
            ((x,), {})
        )

        class LeaksThenInterrupts:
            def check(self, f_locals):
                torch._C._set_torch_function_state(
                    torch._C._TorchFunctionState.ALL_DISABLED
                )
                raise KeyboardInterrupt("ctrl-c inside the tree")

        compiled_fn._artifacts.guard_manager = LeaksThenInterrupts()
        state = torch._C._get_torch_function_state()
        self.addCleanup(torch._C._set_torch_function_state, state)
        with self.assertRaisesRegex(KeyboardInterrupt, "ctrl-c inside the tree"):
            compiled_fn(x)
        self.assertEqual(torch._C._get_torch_function_state(), state)

    def test_aot_compile_function_restores_torch_function_after_a_verbose_interrupt(
        self,
    ):
        # And its second evaluation, the one that runs only to explain a
        # rejection.
        self._hide_leaked_dynamo_globals()

        def fn(x):
            return x * 2

        x = torch.randn(3, 3)
        compiled_fn = torch.compile(fn, fullgraph=True, backend="eager").aot_compile(
            ((x,), {})
        )

        class LeaksThenInterrupts:
            def check(self, f_locals):
                return False

            def check_verbose(self, f_locals):
                torch._C._set_torch_function_state(
                    torch._C._TorchFunctionState.ALL_DISABLED
                )
                raise KeyboardInterrupt("ctrl-c inside the tree")

        compiled_fn._artifacts.guard_manager = LeaksThenInterrupts()
        state = torch._C._get_torch_function_state()
        self.addCleanup(torch._C._set_torch_function_state, state)
        with self.assertRaisesRegex(KeyboardInterrupt, "ctrl-c inside the tree"):
            compiled_fn(x)
        self.assertEqual(torch._C._get_torch_function_state(), state)

    def test_no_match_message_advises_an_input_after_an_answer_then_a_raise(self):
        # This tree's LAST evaluation raised, so its line is that raise and it has
        # no guard to quote -- but it did reject this call on the scan, so an
        # input covering the call is still on the table and "every guard tree
        # raised" would be false. The advice is keyed on whether a tree EVER
        # answered, not on what it last did.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )

        class RejectsThenRaises(NeverReChecked):
            def __init__(self):
                self.checks = 0

            def check(self, f_locals):
                self.checks += 1
                if self.checks == 1:
                    return False
                raise RuntimeError("the second pass is unhappy")

        stub = RejectsThenRaises()
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
        self.assertNotIn("advice above rests only on rejections", message)
        self.assertNotIn("Every guard tree raised", message)

    def test_no_match_message_qualifies_a_rejection_that_followed_a_raise(self):
        # The mirror image: this tree raised in the scan and rejected the call on
        # the second pass. That rejection is the one the veto refuses to release
        # the last resort on, so the report says both things it knows -- an
        # input covering the call is on the table, and the only rejection on
        # record followed a throw -- rather than keeping one line by dropping the
        # other. The rejection above, taken BEFORE its raise, earns no such line.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )

        class RaisesThenRejects:
            def __init__(self):
                self.checks = 0

            def check(self, f_locals):
                self.checks += 1
                if self.checks == 1:
                    raise RuntimeError("the scan is unhappy")
                return False

            def check_verbose(self, f_locals):
                return types.SimpleNamespace(
                    result=False, verbose_code_parts=["stub guard rejected"]
                )

        stub = RaisesThenRejects()
        model.forward.compiled_results[0]._artifacts.guard_manager = stub
        with self.assertRaises(RuntimeError) as ctx:
            model(torch.randn(3, 3))
        message = str(ctx.exception)
        self.assertEqual(stub.checks, 2)
        self.assertIn("  [0] stub guard rejected", message.splitlines())
        self.assertIn("[0]'s guard check raised while checking this call", message)
        self.assertIn("Add a ModelInput", message)
        caveat = (
            "The ModelInput advice above rests only on rejections dispatch took "
            "after the same tree had raised, so they can be about the relational "
            "guard state a C++ throw leaves stale rather than about this call."
        )
        self.assertIn(caveat, message.splitlines())
        # Not the wording for a report whose every line IS a raise: this one
        # quotes a guard.
        self.assertNotIn("Every guard tree raised", message)

    def test_aot_compile_module_second_pass_warns_that_it_served_over_a_raise(self):
        # The second serving path that has to record a swallowed raise: nothing
        # matched in the scan, and the second pass found a match. Only the second
        # pass reaches [1]'s accept, so this warning is that call site's.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [
                ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[]),
                ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[]),
            ]
        )

        class Raises:
            def check(self, f_locals):
                raise RuntimeError("both passes are unhappy")

        class RejectsThenAccepts:
            def __init__(self):
                self.checks = 0

            def check(self, f_locals):
                self.checks += 1
                return self.checks > 1

        results = model.forward.compiled_results
        results[0]._artifacts.guard_manager = Raises()
        stub = RejectsThenAccepts()
        results[1]._artifacts.guard_manager = stub
        x = torch.randn(3, 3)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            served = model(x)
        self.assertEqual(served, x * 2)
        # Rejected in the scan and accepted on the second pass, so the serve --
        # and the warning with it -- is the second pass's.
        self.assertEqual(stub.checks, 2)
        warned = "\n".join(logs.output)
        self.assertIn(
            "[0]'s guard check raised RuntimeError: both passes are unhappy", warned
        )
        self.assertIn("dispatch served [1]", warned)

    def test_aot_compile_module_serves_a_tree_that_raised_and_then_accepted(self):
        # The one direction the veto is NOT applied to: [0] raised in the scan
        # and its own accept on the second pass is what serves. The veto holds
        # the last resort back because a raise rejected nothing and an unguarded
        # graph needs real rejections; an accept is that tree's answer about this
        # call. Vetoing it would not contain the stale relational state either --
        # a scan raise whose call another result serves leaves it stale for the
        # NEXT call, which has no raise on record -- so the warning names both
        # directions a stale tree can bend a later answer, and the caller keeps
        # the graph.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
        )

        class RaisesThenAccepts:
            def __init__(self):
                self.checks = 0

            def check(self, f_locals):
                self.checks += 1
                if self.checks == 1:
                    raise RuntimeError("the scan is unhappy")
                return True

        stub = RaisesThenAccepts()
        model.forward.compiled_results[0]._artifacts.guard_manager = stub
        x = torch.randn(3, 3)
        logger = "torch._dynamo.aot_compile"
        with self.assertLogs(logger, level="WARNING") as logs:
            served = model(x)
        self.assertEqual(served, x * 2)
        # The scan raised and the second pass accepted, so the serve is the
        # second pass's and the raiser is the index it served.
        self.assertEqual(stub.checks, 2)
        warned = "\n".join(logs.output)
        raised = "[0]'s guard check raised RuntimeError: the scan is unhappy"
        self.assertIn(raised, warned)
        self.assertIn("dispatch served [0]", warned)
        self.assertIn("reject a call it fits or accept one it does not", warned)
        # The same tree's next answer is acted on with no raise on record at all,
        # which is why a per-call veto is not what keeps a stale tree honest.
        with self.assertNoLogs(logger, level="WARNING"):
            self.assertEqual(model(x), x * 2)

    def test_aot_compile_module_warns_about_every_input_that_raised(self):
        # Two broken artifacts are two things to fix, so the warning walks every
        # raise on record rather than the first.
        self._hide_leaked_dynamo_globals()
        model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
        model._aot_compile(
            [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])] * 3
        )

        class Raises:
            def __init__(self, message):
                self.message = message

            def check(self, f_locals):
                raise RuntimeError(self.message)

        results = model.forward.compiled_results
        results[0]._artifacts.guard_manager = Raises("the first tree is unhappy")
        results[1]._artifacts.guard_manager = Raises("the second tree is unhappy")
        x = torch.randn(3, 3)
        with self.assertLogs("torch._dynamo.aot_compile", level="WARNING") as logs:
            served = model(x)
        self.assertEqual(served, x * 2)
        self.assertEqual(len(logs.output), 2)
        warned = "\n".join(logs.output)
        self.assertIn(
            "[0]'s guard check raised RuntimeError: the first tree is unhappy", warned
        )
        self.assertIn(
            "[1]'s guard check raised RuntimeError: the second tree is unhappy", warned
        )
        self.assertEqual(warned.count("dispatch served [2]"), 2)
        # A raising tree raises on every call, so this is once per defect and not
        # once per call.
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            model(x)

    def test_aot_compiled_model_takes_no_dedup_set_from_its_caller(self):
        # The warning dedup set below is internal state, not a third constructor
        # parameter: a caller handing one in would share it across models and get
        # the process-global dedup the field exists to avoid.
        init_fields = [f.name for f in dataclasses.fields(AOTCompiledModel) if f.init]
        self.assertNotIn("_warned", init_fields)
        with self.assertRaisesRegex(TypeError, "takes 3 positional arguments but 4"):
            AOTCompiledModel(ScaleModule(), [], set())

    def test_aot_compile_module_binds_a_call_once_per_result(self):
        # Results whose signatures differ cannot share a binding, so each binds
        # on its own -- and still only once: the two dispatch passes and the
        # report all ask about the same call, so a rebind per pass would leave
        # the rest of this file green and pay for itself on every no-match.
        # test_module_dispatch_binds_a_call_once_for_results_sharing_a_signature
        # pins the shared case; this is the per-result count it cannot see.
        mod = ScaleModule()
        x = torch.randn(3, 3)
        model = torch.compile(mod, fullgraph=True, backend="eager")
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        doubled = model.forward.compiled_results

        def triple(self, y):
            return y * 3

        mod.forward = types.MethodType(triple, mod)
        model = torch.compile(mod, fullgraph=True, backend="eager")
        model._aot_compile([ModelInput(args=(x.double(),), kwargs={}, contexts=[])])
        combined = AOTCompiledModel(mod, doubled + model.forward.compiled_results)
        self.assertFalse(combined._binds_alike(tuple(combined.compiled_results)))
        binds = []
        bind = AOTCompiledFunction.prepare_f_locals

        def counted(result, *args, **kwargs):
            binds.append(result)
            return bind(result, *args, **kwargs)

        with patch.object(AOTCompiledFunction, "prepare_f_locals", counted):
            with self.assertRaises(RuntimeError) as ctx:
                combined(x.half())
        # Both passes and the report ran on both results, each from its own one
        # binding: one bind per result, in index order, and the same objects.
        results = combined.compiled_results
        self.assertEqual([id(b) for b in binds], [id(r) for r in results])
        lines = str(ctx.exception).splitlines()
        self.assertEqual(sum(ln.startswith("  [") for ln in lines), 2, lines)

    def test_aot_compile_module_warns_once_per_model_not_per_process(self):
        # The dedup set is a field on the model, which is the whole reason it is
        # not torch._logging.warning_once: that cache is process-global, so the
        # first model to log would silence every later one carrying the same
        # defect -- one broken artifact per process reported, and the rest quiet.
        self._hide_leaked_dynamo_globals()

        class Raises:
            def check(self, f_locals):
                raise RuntimeError("guard tree is unhappy")

        def broken_model():
            model = torch.compile(ScaleModule(), fullgraph=True, backend="eager")
            model._aot_compile(
                [ModelInput(args=(torch.randn(3, 3),), kwargs={}, contexts=[])]
            )
            result = model.forward.compiled_results[0]
            result._artifacts.guard_manager = Raises()
            result.disable_guard_check()
            return model

        x = torch.randn(3, 3)
        logger = "torch._dynamo.aot_compile"
        first, second = broken_model(), broken_model()
        with self.assertLogs(logger, level="WARNING") as logs:
            self.assertEqual(first(x), x * 2)
        self.assertEqual(len(logs.output), 1)
        with self.assertNoLogs(logger, level="WARNING"):
            first(x)
        with self.assertLogs(logger, level="WARNING") as logs:
            self.assertEqual(second(x), x * 2)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("[0]'s guard check raised RuntimeError", logs.output[0])

    def test_aot_compile_module_ordinary_dispatch_warns_about_nothing(self):
        # The swallowed-raise warning is about a raise, so an ordinary dispatch
        # that walks past a non-matching input to a matching one is silent.
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
        x = torch.randn(3, 3)
        with self.assertNoLogs("torch._dynamo.aot_compile", level="WARNING"):
            served = model(x)
        self.assertEqual(served, x * 2)

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
        # A module artifact's bytecode reads a guarded global's LIVE value as of
        # the load, not the value serialized at capture -- narrower than a
        # function artifact loaded with an f_globals, which merges the whole
        # dict. The load feeds the scope resolved from model.forward to the
        # guards and substitutes name by name, skipping the recorded
        # __builtins_dict___N key whether or not a guard reads it.
        # keep_global_guards is what makes that guard exist at all -- the default
        # aot_compile filter drops every global guard, which would leave nothing
        # guarding AOT_HERMETIC_WEIGHT. That scope is this module's dict, which
        # the capture leaks Dynamo's generated names into, and the load seeds
        # nothing into it: the only kept global guard is rooted at
        # AOT_HERMETIC_WEIGHT, not at an import alias or the builtins-dict key.
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
            # the snapshot the bytecode reads is taken at load, so it picks up a
            # rebind that happened before the load rather than serving the
            # capture-time product.
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
            # A rebind after the load: the call still answers with the
            # load-time value.
            AOT_HERMETIC_WEIGHT = saved * 3
            self.assertEqual(reloaded(x), live)
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
        # bytecode never reads it: the load-time snapshot carries a copy of the
        # name, but only the guards consult its value.
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
        saved_eps = EPS
        self.addCleanup(globals().__setitem__, "EPS", saved_eps)
        AOT_UNGUARDED_PARAM = torch.nn.Parameter(torch.ones(3))
        saved_param = AOT_UNGUARDED_PARAM

        class TwoGlobalsModule(torch.nn.Module):
            def forward(self, x):
                return x * EPS + AOT_UNGUARDED_PARAM

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
        self.assertNotIn("vars(", message)

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
        # The bytecode's half of the f_globals contract, which this commit
        # leaves alone: the dict is merged OVER the globals serialized with the
        # artifact into a load-time snapshot, so a name it omits still resolves
        # and a name it binds is what the graph computes with. This passes on
        # the parent too -- the default filter keeps no global guard, so nothing
        # here turns on which dict the guards resolve against. It is here
        # because the other half REPLACES the guard scope, and the two are easy
        # to conflate.
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
        with open(self.path(), "rb") as f:
            bound = torch.compiler.load_compiled_function(f, f_globals={"EPS": live})
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

    def test_load_compiled_function_f_globals_accepted_rebind_is_not_used(self):
        # The limitation the docstring and the guide record: the guards read
        # f_globals live while the bytecode reads the load-time snapshot, so a
        # rebind a kept TENSOR_MATCH accepts -- it compares metadata, not values
        # -- passes the check and the graph still computes with the value the
        # load snapshotted. test_..._guard_checks_the_tensor_type pins the
        # rejected end; this pins the accepted one, so either fix the message
        # weighs -- re-rooting the guards at the snapshot, or refreshing it once
        # a check passes -- has to change this test. Unlike the rejected end, it
        # answers the same way on the parent, where a rebind was invisible to
        # the guards too.
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
        # The rebind being ACCEPTED has to be a guard's answer rather than an
        # absent one: both the served value and guard_check returning True also
        # hold for an artifact that kept no guard on G['EPS'] at all.
        guards_state = load_guards_state(compiled_fn._artifacts.guards_state)
        kept = [str(guard) for guard in guards_state.output_graph.guards]
        self.assertTrue(any("G['EPS']" in guard for guard in kept), kept)
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()

        load_time = EPS.clone()
        scope = {"EPS": load_time}
        with open(self.path(), "rb") as f:
            loaded = torch.compiler.load_compiled_function(f, f_globals=scope)
        self.assertEqual(loaded(x), x * load_time)

        rebound = torch.tensor(5.0)
        self.assertNotEqual(rebound.item(), load_time.item())
        scope["EPS"] = rebound
        self.assertEqual(loaded(x), x * load_time)

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
        # resolves to is the dict the guards hold, as it is for any load that
        # passed no guard_globals= -- the rebound sibling's included -- so the
        # substring asserted below fits that sibling too; what pins this
        # non-rebound load is the recovery arm at the end -- defining the name in
        # globals() serves the call here, where the same binding in
        # test_no_match_message_hint_covers_a_rebound_forward still raises.
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
            # A missing global counts as the mismatch it is: the report cannot
            # know whether a new ModelInput would read that global -- one captured
            # for a branch that does not is served with the name still absent --
            # so it emits the hedged advice next to the hint.
            self.assertIn("Add a ModelInput", message)
            # Taking the advice restores dispatch but not the value: the
            # guards read the live scope, so the new binding passes the kept
            # TENSOR_MATCH, which checks metadata and not values. The bytecode's
            # globals are built once, at load, and substitute a live value only
            # for a name the scope has then, so the value serialized with the
            # artifact was never overwritten -- which is what asserting
            # `x @ saved` here pins.
            g["AOT_HERMETIC_WEIGHT"] = saved * 2
            self.assertEqual(reloaded(x), x @ saved)
        finally:
            g["AOT_HERMETIC_WEIGHT"] = saved

    def test_missing_key_inside_a_present_global_is_not_a_missing_global(self):
        # A guard on G['CONFIG']['key'] reports "KeyError on
        # G['GLOBAL_POOLING_CONFIG']['pooling']" when the KEY is gone but the
        # global itself resolved, which is an ordinary mismatch: the report keeps
        # the "Add a ModelInput" advice and withholds the missing-global hint,
        # whose advice (define the global) is wrong here. The whole-verbose-part
        # match that tells the two apart came in with the hint and is pinned
        # there; what this pins is which of the two the module report reaches for.
        self._hide_leaked_dynamo_globals()
        mod = GlobalConfigModule()
        x = torch.randn(4, 8)
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_global_guards},
        )
        model._aot_compile([ModelInput(args=(x,), kwargs={}, contexts=[])])
        saved = GLOBAL_POOLING_CONFIG.pop("pooling")
        try:
            with self.assertRaises(RuntimeError) as ctx:
                model(x)
            message = str(ctx.exception)
        finally:
            GLOBAL_POOLING_CONFIG["pooling"] = saved
        self.assertIn("KeyError on G['GLOBAL_POOLING_CONFIG']['pooling']", message)
        self.assertIn("Add a ModelInput", message)
        self.assertNotIn("a guarded global is missing", message)

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
        self.assertEqual(loaded(x), fn(x))

    def test_load_rederives_over_a_builtin_the_loading_process_lacks(self):
        # The re-derive only ever fires when the GENERATED bytecode reads the key,
        # since that is what makes get_runtime_env record it, so on the default
        # path it also decides what the bytecode subscripts -- and the recording
        # it replaces is filtered for picklability alone, never narrowed to names
        # the loading process has. A builtin missing here therefore stops being
        # readable: a kept guard on that name reports it, and with that guard
        # filtered out the read fails inside the generated bytecode instead.
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

        # Through the public wrapper the guards get the caller's dict, so the
        # re-derive lands there and fn.__globals__ keeps the recording -- the key
        # the bytecode subscripts. The artifact that just raised therefore serves
        # here, off a builtin this process no longer has, while the guard reads
        # the live builtins the caller's dict now holds.
        torch._dynamo.reset()
        scope: dict[str, object] = {}
        loaded = torch.compiler.load_compiled_function(
            io.BytesIO(unguarded), f_globals=scope
        )
        self.assertTrue(loaded.guard_check(x))
        self.assertEqual(loaded(x)[1], os.getcwd)
        self.assertIn("aot_probe", loaded.fn.__globals__[unguarded_key])
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
        # what lets the last arm see that a refused load wrote nothing.
        def fn(mod, x):
            if isinstance(x, torch.Tensor):
                return mod(x)
            return x

        lin, x = torch.nn.Linear(3, 3), torch.randn(3, 3)
        compiled_fn = torch.compile(
            fn,
            fullgraph=True,
            backend="eager",
            options={"guard_filter_fn": keep_builtin_guards},
        ).aot_compile(((lin, x), {}))
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
