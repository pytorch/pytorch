# Owner(s): ["module: dynamo"]

import builtins
import contextlib
import copy
import dataclasses
import functools
import gc
import importlib
import inspect
import itertools
import math
import math as _precompile_stdlib_alias
import os
import pickle
import queue
import re
import sys
import sysconfig
import textwrap
import threading
import types
import unittest
from unittest import mock

import torch
import torch._dynamo.package as dynamo_package
import torch._dynamo.precompile_package as dynamo_package_lint
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.nn.functional as F
from torch._C._dynamo.eval_frame import _debug_get_precompile_entries
from torch._dynamo.exc import PackageError, RecompileError
from torch._dynamo.package import (
    _defining_module_name,
    CompilePackage,
    DynamoCache,
    SystemInfo,
)
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.precompile_package import (
    _compose_with_default,
    _dynamo_alias_module,
    _fact_order,
    _GuardFact,
    _normalize,
    _SingleFileStore,
    default_guard_filter_fn,
    precompile_capture,
    serving,
    varying_guard_slots,
)
from torch._dynamo.types import GuardFilterEntry
from torch._functorch import config as functorch_config
from torch._inductor.runtime.runtime_utils import cache_dir
from torch.compiler._precompile_types import PrecompileSummary
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


def staged_with_graph_breaks(x):
    x = x * 2
    torch._dynamo.graph_break()
    x = x + 3
    torch._dynamo.graph_break()
    return x.sum()


def _graph_on(device):
    graph = torch.fx.Graph()
    graph.call_function(torch.ones, args=(torch.device(device),))
    return graph


@contextlib.contextmanager
def _counting_cpu_probe(toolchain=True):
    """Count pick_vec_isa calls; without a toolchain every probe raises."""
    import torch._inductor.cpu_vec_isa as cpu_vec_isa
    from torch._inductor.exc import InvalidCxxCompiler

    calls = []
    real_pick = cpu_vec_isa.pick_vec_isa

    def pick():
        calls.append(1)
        if not toolchain:
            raise InvalidCxxCompiler
        return real_pick()

    with mock.patch.object(cpu_vec_isa, "pick_vec_isa", pick):
        yield calls


_MIPS_TARGET = ("mips", "DEFAULT", None, "INVALID")

# recorded target, whether the host probe works, what the comparison raises
_CPU_TARGET_CASES = {
    "no_target_recorded": (None, True, None),
    "skewed_target": (_MIPS_TARGET, True, "built for machine 'mips'"),
    "host_probe_failed": (("x86_64", "AVX512", None, "avx512"), False, "no usable"),
}


# collections.abc, three lines of `from _collections_abc import *`, with the
# private file spoofing __name__ back to the public name so tracebacks read
# right. Every model that inlines through such a module hits both traps.
_SHIM_IMPL_SRC = """\
# Renamed out of the way for import-time reasons, exactly as _collections_abc.
__name__ = "shim_abc"


def helper(x):
    return x * 3.0
"""

_SHIM_SRC = """\
from _shim_impl import *
"""


def _rebind(scope):
    scope["g"] = "bystander"


# Values successive packages bind to one name, what a bystander does to the
# binding, and what each unload (last installed first) must leave bound. The
# binding stack is bookkeeping, not ownership of the namespace.
_BINDING_STACK_CASES = {
    # Rebinding by plain torch.compile, a CleanupHook or user code is theirs
    # to keep: restoring a surviving package's value hands them a compiled
    # value they never asked for, and the last unload would then delete it.
    "rebound": (("first", "second"), _rebind, ("bystander", "bystander")),
    # The "or gone" arm: a name deleted out from under a still-serving earlier
    # package is put back by the later package's unload.
    "deleted": (("first", "second"), lambda scope: scope.pop("g"), ("first", None)),
    # One value stacked twice with another between them: deregistering by
    # value alone finds the earlier frame and leaves an unloaded package's
    # value bound for good.
    "phantom": (("shared", "other", "shared"), lambda scope: None, ("other", "shared", None)),
}  # fmt: skip


class PrecompileBlock(torch.nn.Module):
    """With resume_work, the frame resumed after the break compiles too."""

    def __init__(self, i, resume_work):
        super().__init__()
        self.i = i
        self.resume_work = resume_work

    def forward(self, x):
        x = x * 2 + self.i
        torch._dynamo.graph_break()
        return x + 1.0 if self.resume_work else x


class PrecompileStack(torch.nn.Module):
    """All blocks share one forward code object, so variants pile onto it."""

    def __init__(self, n, resume_work=False):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [PrecompileBlock(i, resume_work) for i in range(n)]
        )

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x.sum()


PRECOMPILE_CONFIG = {"mode": "sum"}


def staged_with_global_dict_conditional(x):
    # The global is read on both sides of the break, so the entry frame and the
    # resume frame each carry a guard on it.
    if PRECOMPILE_CONFIG["mode"] == "sum":
        x = x * 2
    else:
        x = x * 3
    torch._dynamo.graph_break()
    if PRECOMPILE_CONFIG["mode"] == "sum":
        return x.sum()
    return x.mean() * 10.0


def _precompile_scale(t):
    return t * 2


class PrecompileNoDispatchSlot(torch.nn.Module):
    """Ordinary code with no swappable slot: a torch op, stdlib, a local def."""

    def forward(self, x):
        y = torch.relu(_precompile_scale(x)) * math.sqrt(2.0)
        torch._dynamo.graph_break()
        return (y + 1).sum()


PRECOMPILE_INV_CONFIG = {"mode": "sum"}


class PrecompileInvariantModel(torch.nn.Module):
    """Reads a global across a graph break, so the resume frame guards it."""

    def forward(self, x):
        y = x * 2
        torch._dynamo.graph_break()
        if PRECOMPILE_INV_CONFIG["mode"] == "sum":
            return y.sum()
        return y.mean()


_TWO_SHAPES = [(torch.ones(4, 8),), (torch.ones(5, 8),)]


class PrecompileSelfAct(torch.nn.Module):
    """self.act = <callable> -- how configurable activations are usually written."""

    def __init__(self, act):
        super().__init__()
        self.act = act

    def forward(self, x):
        y = self.act(x)
        torch._dynamo.graph_break()
        return (y + 1).sum()


class PrecompileEmptyGraph(torch.nn.Module):
    """Used to construct a legacy package with no guarded code in skip tests."""

    def forward(self, x):
        return x.sin()


class PrecompilePartialForward(torch.nn.Module):
    """self.forward = functools.partial(...) shadows the class method."""

    def __init__(self, scale):
        super().__init__()
        self.forward = functools.partial(self._impl, scale)

    def _impl(self, scale, x):
        return (x * scale).sum()


def _precompile_sin(t):
    return t.sin()


PRECOMPILE_ACTIVATION = _precompile_sin


def staged_with_global_function_ref(x):
    y = PRECOMPILE_ACTIVATION(x) + 1
    torch._dynamo.graph_break()
    return (y * 10).sum()


def staged_with_builtin_calls(x, cfg):
    # len() and sorted() are resolved through the builtins dict Dynamo installs
    # in this module, which is where a builtin read the ordinary way comes from.
    n = len(cfg)
    torch._dynamo.graph_break()
    return x.sum() * n * len(sorted(cfg))


def _precompile_house_op(t):
    return t * 5.0


_PRECOMPILE_OPS = {"len": len}


def staged_with_injected_builtin(x):
    # The test installs this into builtins, so it is read exactly the way len()
    # is, off a namespace anyone can write to at runtime.
    y = precompile_house_op(x)  # noqa: F821
    torch._dynamo.graph_break()
    return (y + 1).sum()


def _with_injected_builtin(test):
    builtins.precompile_house_op = _precompile_house_op
    test.addCleanup(lambda: delattr(builtins, "precompile_house_op"))
    return staged_with_injected_builtin


def staged_with_a_registry_keyed_by_a_builtin_name(x, cfg):
    n = _PRECOMPILE_OPS["len"](cfg)
    torch._dynamo.graph_break()
    return x.sum() * n


def _precompile_user_act(t):
    return -t


def _precompile_closure_over(fn):
    def inner(x):
        return fn(x).sum()

    return inner


class _PrecompileRegistry:
    def __init__(self, act):
        self.act = act


PRECOMPILE_DISPATCH = {"act": _precompile_user_act}
PRECOMPILE_REGISTRY = _PrecompileRegistry(_precompile_user_act)


def _in_inference_mode(tensor):
    """Tag a tensor so the call using it runs under inference_mode."""
    return _InferenceInput(tensor)


def _marked_static(tensor):
    torch._dynamo.mark_static(tensor, 0)
    return tensor


class _InferenceInput:
    """Marker: the corpus runs this input inside inference_mode."""

    def __init__(self, tensor):
        self.tensor = tensor


class PrecompileBuiltinReadingModel(torch.nn.Module):
    """Iterates a ModuleList, which reads ``iter`` off Dynamo's builtins dict."""

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(2)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x.sum()


# One source LINE, so the two entries agree on file AND lineno: what separates
# them can only come from the code body. This is the ACT2FN shape.
_LAMBDA_TABLE = {"a": lambda x: x.sin(), "b": lambda x: x.cos()}


class PrecompileSelfActPair(torch.nn.Module):
    """Two PrecompileSelfAct instances, so their shared forward compiles twice."""

    def __init__(self, *acts):
        super().__init__()
        self.ms = torch.nn.ModuleList([PrecompileSelfAct(act) for act in acts])

    def forward(self, x):
        out = x.sum()
        for m in self.ms:
            out = out + m(x)
        return out


def _pin_item_in_a_branch(x):
    scale = x.abs().max().item()
    y = x * 2 if scale > 0.5 else x * 3
    return y.sum()


def _pin_item_across_a_break(x):
    """.item() pins the stack slot AND the local the resume frame reads it from."""
    scale = x.abs().max().item()
    torch._dynamo.graph_break()
    return (x * 2 if scale > 0.5 else x * 3).sum()


def _pin_int_arg(x, k):
    y = x * k
    torch._dynamo.graph_break()
    return (y + k).sum()


def _pin_keys_arg(x, ks):
    """A dict_keys argument is pinned by EQUALS_MATCH, not CONSTANT_MATCH."""
    y = x * float(len(ks))
    torch._dynamo.graph_break()
    return (y + 1).sum()


def _precompile_break_then_cos(t):
    torch._dynamo.graph_break()
    return t.cos()


def _tensor_across_a_break(x):
    """x.sin() sits on the stack across the break, so it gets a ___stackN name."""
    return (x.sin() + _precompile_break_then_cos(x)).sum()


class PrecompileIntAttr(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        y = x * self.k
        torch._dynamo.graph_break()
        return (y + self.k).sum()


class PrecompileConfigConstants(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = torch.nn.LayerNorm(8)
        self.drop = torch.nn.Dropout(0.1)
        self.lin = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.drop(self.lin(self.ln(x))).relu().sum()


# The stateless corpus shapes are plain functions; precompile_capture takes
# either, and only the guard sources matter here.
def _aliased_torch_import(x):
    """import torch.nn.functional as F -- the global name is not the module's."""
    return F.gelu(x).sum()


def _fully_qualified(x):
    """torch.nn.functional.gelu spelled out -- the namespace is two hops down."""
    return torch.nn.functional.gelu(x).sum()


def _function_scoped_torch_import(x):
    """A function-scoped `import torch`, which transformers is full of. Dynamo
    reaches it through the `__import_torch` alias and guards the attributes read
    off that alias, never the alias itself."""
    import torch

    if isinstance(x, torch.Tensor):
        return torch.relu(x).sum()
    return x


class PrecompileModuleInAttribute(torch.nn.Module):
    """self.ns = importlib.import_module(cfg.backend) -- a config-picked module."""

    def __init__(self, ns):
        super().__init__()
        self.ns = ns

    def forward(self, x):
        return self.ns.gelu(x).sum()


def _dict_dispatch(x):
    """CFG["act"] -- the same dispatch slot spelled as a dict lookup."""
    return PRECOMPILE_DISPATCH["act"](x).sum()


def _registry_lookup(x):
    """REGISTRY.act -- a dispatch table parked in a module-level object."""
    return PRECOMPILE_REGISTRY.act(x).sum()


def _function_arg(x, fn):
    return fn(x).sum()


def _function_default(x, fn=_precompile_user_act):
    return fn(x).sum()


class PrecompileStockEncoderLayer(torch.nn.Module):
    """nn.TransformerEncoderLayer parks its activation in an attribute, and
    which callable lands there is a constructor keyword -- torch's own spelling
    of self.act = getattr(F, cfg.activation)."""

    def __init__(self):
        super().__init__()
        self.enc = torch.nn.TransformerEncoderLayer(
            8, 2, 16, batch_first=True, dropout=0.0
        )

    def forward(self, x):
        return self.enc(x)


class PrecompileStockSequential(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 8)
        )

    def forward(self, x):
        return self.net(x).sum()


def _aliased_stdlib_import(x):
    """import math as <alias> -- a stdlib module under a name it does not own."""
    return (x * _precompile_stdlib_alias.sqrt(2.0)).sum()


def _stdlib_attribute(x):
    return (x * math.sqrt(2.0)).sum()


def _same_module_helper(x):
    return _precompile_scale(x).sum()


# The modules the corpus needs on disk, because a dispatch read off another
# module cannot be spelled inside this file. Written under one directory so
# they import each other by plain name, exactly as a user package would.
_CORPUS_MODULES = {
    "cimpl_a": "def op(x):\n    return x + 1.0\n",
    "cimpl_b": "def op(x):\n    return x * 7.0\n",
    "lazy_helper": "def op(x):\n    return x * 3.0\n",
    "cdispatch": """\
import os

if os.environ.get("CORPUS_B") == "1":
    from cimpl_b import op
else:
    from cimpl_a import op
""",
    "chelpers": """\
import os

if os.environ.get("CORPUS_B") == "1":
    from cimpl_b import op
else:
    from cimpl_a import op

def call(x):
    return op(x)
""",
    "cconf": """\
import os

import cimpl_a
import cimpl_b

ACT = cimpl_b.op if os.environ.get("CORPUS_B") == "1" else cimpl_a.op
""",
    "calias": """\
import os

import torch

if os.environ.get("CORPUS_B") == "1":
    import cimpl_b as impl
else:
    import cimpl_a as impl

class Model(torch.nn.Module):
    def forward(self, x):
        return impl.op(x)
""",
    "cfrom": """\
import os

import torch

if os.environ.get("CORPUS_B") == "1":
    from cimpl_b import op
else:
    from cimpl_a import op

class Model(torch.nn.Module):
    def forward(self, x):
        return op(x)
""",
    "cown": """\
import torch

def _scale(x):
    return x * 2.0

def call(x):
    return torch.relu(_scale(x))
""",
    # Four spellings the def-name rule cannot clear on its own, so whether
    # they are refused rides entirely on recognising torch and the stdlib:
    # `from torch import relu` (owned by torch itself, not a torch.*
    # submodule), `from math import sqrt` (a def in a C stdlib module, so
    # there is no file to match the reader against), `import math as _math`
    # and `import collections.abc as _abc` (modules under names that are
    # not their own, one of them dotted).
    "clibspell": """\
import collections.abc as _abc
import math as _math
import torch
from math import sqrt
from torch import relu

class Model(torch.nn.Module):
    def forward(self, x):
        n = float(isinstance([], _abc.Sized))
        return (relu(x) * sqrt(2.0) * _math.fabs(-1.0) * n).sum()
""",
    "cpkg/cimpl": "def op(x):\n    return x + 1.0\n",
    "cpkg/__init__": "from . import cimpl as impl\n\n\nop = impl.op\n",
    "cmodels": """\
import os

import torch
from torch.nn.functional import gelu

import cconf
import cdispatch
import chelpers
import cimpl_a
import cimpl_b
import cown
import cpkg
import cpkg.cimpl

OPS = cimpl_b if os.environ.get("CORPUS_B") == "1" else cimpl_a

class ModuleAttr(torch.nn.Module):
    def forward(self, x):
        return cconf.ACT(x)

class ModuleSwitch(torch.nn.Module):
    def forward(self, x):
        return OPS.op(x)

class SiblingModule(torch.nn.Module):
    def forward(self, x):
        return cdispatch.op(x)

class InlinedHelper(torch.nn.Module):
    def forward(self, x):
        return chelpers.call(x)

class PackageReexport(torch.nn.Module):
    def forward(self, x):
        return cpkg.op(x)

class PackageSubmoduleAlias(torch.nn.Module):
    def forward(self, x):
        return cpkg.impl.op(x)

class RealSubmodule(torch.nn.Module):
    def forward(self, x):
        return cpkg.cimpl.op(x)

class OwnNameDef(torch.nn.Module):
    def forward(self, x):
        return gelu(cown.call(x))

class LazyUserImport(torch.nn.Module):
    def forward(self, x):
        import lazy_helper

        return lazy_helper.op(x).sum()
""",
}

_CORPUS_X = torch.randn(4, 8)
_CORPUS_SEQ = torch.randn(2, 4, 8)

# Every shape below was once a silent wrong answer on a serving machine that
# somebody had to find by hand. _is_risky_drop regressed three review rounds
# running -- each fix closed the previous round's false negative and opened a
# new one -- so the shapes live in a table rather than in prose: adding one is
# a single entry, and nothing here may ever stop being flagged. Each row names
# the guard sources (by suffix) the risky-drop report must carry.
_RISKY_DROP_CORPUS = {
    "aliased_module_import": (("G['impl']", "G['impl'].op"), lambda t: (t._corpus_model("calias"), (_CORPUS_X,))),
    # self.act is a dispatch slot whatever it holds, and classifying by the
    # binding site is what catches the torch-owned and builtin cases: an
    # ACT2FN-style table holds abs next to torch.relu, and which one lands in
    # the slot is exactly what config decides.
    "attribute_builtin_fn": (("self.act",), lambda t: (PrecompileSelfAct(abs), (_CORPUS_X,))),
    "attribute_torch_fn": (("self.act",), lambda t: (PrecompileSelfAct(F.gelu), (_CORPUS_X,))),
    "attribute_torch_relu": (("self.act",), lambda t: (PrecompileSelfAct(torch.relu), (_CORPUS_X,))),
    "attribute_user_fn": (("self.act",), lambda t: (PrecompileSelfAct(_precompile_user_act), (_CORPUS_X,))),
    "bare_global_fn": (("G['PRECOMPILE_ACTIVATION']",), lambda t: (staged_with_global_function_ref, (_CORPUS_X,))),
    "closure_cell": (("fn",), lambda t: (_precompile_closure_over(_precompile_user_act), (_CORPUS_X,))),
    "cross_module_from_import": (("G['op']",), lambda t: (t._corpus_model("cfrom"), (_CORPUS_X,))),
    "dict_lookup": (("G['PRECOMPILE_DISPATCH']['act']",), lambda t: (_dict_dispatch, (_CORPUS_X,))),
    "dispatch_in_inlined_helper": (("G['__import_chelpers'].op",), lambda t: (t._corpus("InlinedHelper"), (_CORPUS_X,))),
    "function_argument": (("fn",), lambda t: (_function_arg, (_CORPUS_X, _precompile_user_act))),
    "function_default_arg": (("fn",), lambda t: (_function_default, (_CORPUS_X,))),
    # builtins is writable and per-process, so a plugin doing `builtins.op =
    # impl_a` here and impl_b there produces a read that comes off the right
    # namespace but holds user code.
    "injected_builtin": (("['precompile_house_op']",), lambda t: (_with_injected_builtin(t), (_CORPUS_X,))),
    "module_in_attribute": (("self.ns", "self.ns.gelu"), lambda t: (PrecompileModuleInAttribute(F), (_CORPUS_X,))),
    "module_level_global": (("G['cconf'].ACT",), lambda t: (t._corpus("ModuleAttr"), (_CORPUS_X,))),
    "module_valued_global": (("G['OPS'].op",), lambda t: (t._corpus("ModuleSwitch"), (_CORPUS_X,))),
    "object_attribute": (("G['PRECOMPILE_REGISTRY'].act",), lambda t: (_registry_lookup, (_CORPUS_X,))),
    "package_reexport": (("G['cpkg'].op",), lambda t: (t._corpus("PackageReexport"), (_CORPUS_X,))),
    "package_submodule_alias": (("G['cpkg'].impl.op",), lambda t: (t._corpus("PackageSubmoduleAlias"), (_CORPUS_X,))),
    # A table under a user global has the same source shape as the builtins
    # dict while being a slot, whatever it happens to hold.
    "registry_keyed_by_builtin_name": (("G['_PRECOMPILE_OPS']['len']",), lambda t: (staged_with_a_registry_keyed_by_a_builtin_name, (_CORPUS_X, {"a": 1}))),
    "sibling_module": (("G['cdispatch'].op",), lambda t: (t._corpus("SiblingModule"), (_CORPUS_X,))),
    "stock_layer_activation": (("self._modules['enc'].activation",), lambda t: (PrecompileStockEncoderLayer(), (_CORPUS_SEQ,))),
}  # fmt: skip

# The other half of the corpus, and the half that keeps the report worth
# reading: the lint only warns by default, so if ordinary code trips it the
# warning is noise nobody audits and nobody ever opts into enforcement. Each
# row names the identity guards (by suffix) that ARE dropped without being
# flagged: torch internals, stdlib modules and their attributes, a global bound
# to a def of its own name, reads off Dynamo's import aliases and its builtins
# dict.
_BENIGN_DROP_CORPUS = {
    "aliased_stdlib_import": ((), lambda t: (_aliased_stdlib_import, (_CORPUS_X,))),
    "aliased_torch_import": (("G['F']",), lambda t: (_aliased_torch_import, (_CORPUS_X,))),
    "builtin_read_ordinary": (("['len']", "['sorted']"), lambda t: (staged_with_builtin_calls, (_CORPUS_X, {"alpha": 1, "beta": 2}))),
    "direct_functional_call": (("G['torch'].nn.functional.gelu",), lambda t: (_fully_qualified, (_CORPUS_X,))),
    "function_scoped_torch_import": (("G['__import_torch'].Tensor", "G['__import_torch'].relu"), lambda t: (_function_scoped_torch_import, (torch.ones(4),))),
    "function_scoped_user_import": (("G['__import_lazy_helper'].op",), lambda t: (t._corpus("LazyUserImport"), (_CORPUS_X,))),
    "library_spellings": (("G['relu']", "G['sqrt']", "G['_math']", "G['_abc']"), lambda t: (t._corpus_model("clibspell"), (torch.ones(4),))),
    "no_dispatch_slot": (("G['math']", "G['math'].sqrt", "G['_precompile_scale']"), lambda t: (PrecompileNoDispatchSlot(), (_CORPUS_X,))),
    "own_name_def_in_own_module": (("G['cown']", "G['gelu']", "G['__import_cown']._scale"), lambda t: (t._corpus("OwnNameDef"), (_CORPUS_X,))),
    "real_submodule": (("G['cpkg'].cimpl", "G['cpkg'].cimpl.op"), lambda t: (t._corpus("RealSubmodule"), (_CORPUS_X,))),
    "same_module_helper": ((), lambda t: (_same_module_helper, (_CORPUS_X,))),
    "stdlib_attribute": ((), lambda t: (_stdlib_attribute, (_CORPUS_X,))),
    "stock_linear_layernorm": ((), lambda t: (PrecompileConfigConstants(), (_CORPUS_X,))),
    "stock_sequential": ((), lambda t: (PrecompileStockSequential(), (_CORPUS_X,))),
}  # fmt: skip

# A value crossing a graph break or arriving as a non-tensor argument is
# guarded by equality, so the artifact only serves inputs reproducing it, and
# nothing else in the summary says so. Model config -- LayerNorm eps, Dropout p,
# a plain int attribute -- is CONSTANT_MATCH too but constant for the model the
# artifact is loaded onto; counting it would flag every model. Rows: builder,
# args, the wont_generalize report, and a (type, source substring) guard that
# must have been KEPT, so a row cannot pass by Dynamo not emitting the guard.
_VALUE_PIN_CORPUS = {
    "item_in_a_branch": (lambda: _pin_item_in_a_branch, (_CORPUS_X,), ("___stack0",), ("CONSTANT_MATCH", "___stack0")),
    "item_across_a_break": (lambda: _pin_item_across_a_break, (_CORPUS_X,), ("___stack0", "scale"), ("CONSTANT_MATCH", "scale")),
    "plain_argument": (lambda: _pin_int_arg, (_CORPUS_X, 7), ("k",), ("CONSTANT_MATCH", "k")),
    "equals_match_argument": (lambda: _pin_keys_arg, (_CORPUS_X, {"a": 1, "b": 2}.keys()), ("ks",), ("EQUALS_MATCH", "ks")),
    "tensor_across_a_break": (lambda: _tensor_across_a_break, (_CORPUS_X,), (), ("TENSOR_MATCH", "___stack")),
    "layer_config_constants": (lambda: PrecompileConfigConstants().eval(), (torch.randn(2, 8),), (), ("CONSTANT_MATCH", "")),
    "int_attribute": (lambda: PrecompileIntAttr(3), (torch.randn(2, 8),), (), ("CONSTANT_MATCH", "")),
}  # fmt: skip


def _sum_if_tensor(x):
    return x.sum() if type(x) is torch.Tensor else x


# Rows: builder, example_inputs, text the invariants file must contain, and a
# pattern it must not -- object ids, the per-process counter in Dynamo's
# builtins-dict name, and the address install_global_by_id bakes into an
# identifier ("<prefix>_<id(value)>_c<n>": `type(x) is torch.Tensor` installs
# one) are all normalized so the file is stable enough to commit and diff.
_INVARIANTS_FILE_CASES = {
    "config_read_across_a_break": (PrecompileInvariantModel, _TWO_SHAPES, ("frame forward", "invariant [enforced]", "varies"), r"\b\d{9,}\b"),
    "dynamo_global_counter": (PrecompileBuiltinReadingModel, _TWO_SHAPES, ("__builtins_dict___<n>",), r"__builtins_dict___\d"),
    "global_named_by_object_id": (lambda: _sum_if_tensor, [(torch.ones(4),)], ("_<id>_c<n>",), r"_\d{9,}_c\d"),
}  # fmt: skip


def _scale_by_first_value(x, d):
    return x * next(iter(d.values()))


def _mul_or_add(x, flag, k):
    return x * k if flag else x + k


# Shapes where the guard that SPLIT two compilations must show up as varying
# and never as an invariant of both. Rows: builder, capture kwargs, calls,
# substrings some varying fact must carry, substrings some invariant fact must
# carry, substrings no invariant fact may carry.
_VARYING_CORPUS = {
    # _normalize strips object ids so the file diffs clean. It must not strip a
    # user constant with it: these two variants pin the dict to different keys.
    "large_constant": (lambda: _scale_by_first_value, {"dynamic": False}, [(torch.ones(4), {1000000001: 2}), (torch.ones(4), {2000000002: 3})], ("[1000000001]", "[2000000002]"), (), ("dict.keys",)),
    # k is unspecialized, so its guard is "k is an int" in both variants and
    # is a real precondition. Fingerprinting the value it happened to hold
    # would split one shared guard into two indistinguishable varying facts.
    "shared_int_guard": (lambda: _mul_or_add, {"dynamic": True}, [(torch.ones(4), True, 1), (torch.ones(4), False, 2)], (), ("___check_type_id(L['k']",), ()),
    # The id in an identity guard's code is normalized away, so the object has
    # to be named some other way, or self.act is reported invariant.
    "identity_guard_named": (lambda: PrecompileSelfActPair(torch.relu, torch.sigmoid), {"dynamic": False}, [(torch.ones(4),)], ("relu on self.act", "sigmoid on self.act"), (), (".act",)),
    # Every entry of an ACT2FN-style table is "<lambda>" in one module on one
    # line; _object_identity must still tell them apart.
    "lambda_table_pair": (lambda: PrecompileSelfActPair(*_LAMBDA_TABLE.values()), {"dynamic": False}, [(torch.randn(3, 4),)], ("self.act",), (), ("self.act",)),
    # example_inputs run under no_grad and body calls do not, so the same call
    # made both ways compiles twice; global-state guards carry no value of
    # their own, so without a fingerprint nothing would be reported varying.
    "grad_mode": (PrecompileInvariantModel, {"dynamic": False, "example_inputs": [(torch.ones(4, 8),)]}, [(torch.ones(4, 8),)], ("grad_enabled=True", "grad_enabled=False"), (), ()),
}  # fmt: skip


@instantiate_parametrized_tests
class TestPrecompilePackage(torch._inductor.test_case.TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        DynamoCache.clear()
        PrecompileContext.clear()

    def dir(self):
        path = os.path.join(cache_dir(), f"precompile_{self.id()}")
        os.makedirs(path, exist_ok=True)
        return path

    def path(self, name="artifact.pt"):
        """An artifact FILE inside this test's scratch dir; save() writes files."""
        return os.path.join(self.dir(), name)

    def _write_module(self, dirname, name, src):
        pkg_dir = os.path.join(self.dir(), dirname)
        os.makedirs(pkg_dir, exist_ok=True)
        with open(os.path.join(pkg_dir, f"{name}.py"), "w") as f:
            f.write(src)
        return pkg_dir

    def _forget_modules(self, *names):
        for name in names:
            sys.modules.pop(name, None)
            self.addCleanup(lambda n=name: sys.modules.pop(n, None))

    def _import_module(self, pkg_dir, name):
        sys.path.insert(0, pkg_dir)
        self.addCleanup(lambda: pkg_dir in sys.path and sys.path.remove(pkg_dir))
        self.addCleanup(lambda: sys.modules.pop(name, None))
        sys.modules.pop(name, None)
        importlib.invalidate_caches()
        return importlib.import_module(name)

    def _bare_package(self):
        """A package that installs no codes: these exercise the name stack only."""

        def fn(x):
            return x + 1

        return CompilePackage(fn)

    def _corpus_module(self, name):
        """Write _CORPUS_MODULES to disk and import one of them fresh."""
        for path, src in _CORPUS_MODULES.items():
            head, _, leaf = path.rpartition("/")
            self._write_module(os.path.join("corpus", head), leaf, src)
        roots = {path.partition("/")[0] for path in _CORPUS_MODULES}

        def purge():
            for key in [k for k in sys.modules if k.partition(".")[0] in roots]:
                del sys.modules[key]

        purge()
        self.addCleanup(purge)
        return self._import_module(os.path.join(self.dir(), "corpus"), name)

    def _corpus(self, cls):
        return getattr(self._corpus_module("cmodels"), cls)()

    def _corpus_model(self, name):
        return self._corpus_module(name).Model()

    def _capture(self, fn, *args, no_grad=True):
        grad = torch.no_grad() if no_grad else contextlib.nullcontext()
        session = precompile_capture(fn, backend="eager", dynamic=False)
        with session as compiled, grad:
            compiled(*args)
        return session

    @parametrize("case", sorted(_CPU_TARGET_CASES))
    def test_cpu_codegen_target_is_compared_only_when_recorded(self, case):
        # None means "written before the field existed", not "no vector ISA".
        # Treating it as a mismatch would reject every artifact already on disk
        # over a target they never recorded. An artifact that DOES record one
        # must still be compared against this host, or an AVX-512 capture
        # silently miscomputes on AVX2 -- and when the host probe cannot run,
        # "current=None" alone reads like a precompile bug rather than a
        # missing compiler.
        recorded, toolchain, error = _CPU_TARGET_CASES[case]
        if toolchain and SystemInfo.current().cpu_codegen_target is None:
            self.skipTest("no C++ toolchain to determine a CPU codegen target")
        captured = dataclasses.replace(
            SystemInfo.current(), cpu_codegen_target=recorded
        )

        def expect():
            if error is None:
                return contextlib.nullcontext()
            return self.assertRaisesRegex(RuntimeError, error)

        with _counting_cpu_probe(toolchain):
            current = SystemInfo.current()
        self.assertEqual(current.cpu_codegen_target is None, not toolchain)
        with expect():
            captured.check_compatibility(current, "cpu")

        # check_versions is what a load runs. Computing the current target is
        # pure cost when the artifact recorded none -- and a hard failure with
        # no compiler -- so the probe must not run at all, not merely not crash.
        entry = dynamo_package._DynamoCacheEntry(
            codes=[],
            source_info=dynamo_package.SourceInfo(inlined_sources=set()),
            device_type="cpu",
            device_types=frozenset(("cpu",)),
            requires_native_backend_compatibility=True,
            system_info=captured,
        )
        with _counting_cpu_probe(toolchain) as calls, expect():
            entry.check_versions()
        self.assertEqual(len(calls), 0 if recorded is None else 1)

    @parametrize("backend", ("eager", "inductor"))
    def test_caching_precompile_probes_the_cpu_only_for_native_code(self, backend):
        # torch.compile(backend="eager") under caching_precompile builds its own
        # CompilePackage, and that package used to demand native backend
        # compatibility -- so an eager compile dry-compiled a C++ probe for a
        # vector width no eager artifact can ever bake. The relaxation must not
        # reach the backend that actually bakes one into the code it ships.
        seen = []
        real = dynamo_package.CompilePackage.cache_entry

        def spy(package):
            seen.append(real(package))
            return seen[-1]

        def f(x):
            return x.sin() + 1

        with (
            torch._dynamo.config.patch(caching_precompile=True),
            mock.patch.object(dynamo_package.CompilePackage, "cache_entry", spy),
            _counting_cpu_probe() as calls,
        ):
            torch.compile(f, backend=backend)(torch.randn(4))
        native = backend == "inductor"
        self.assertTrue(seen)
        self.assertEqual(
            {e.requires_native_backend_compatibility for e in seen}, {native}
        )
        if native:
            self.assertTrue(
                all(e.system_info.cpu_codegen_target is not None for e in seen)
            )
        else:
            self.assertEqual(calls, [])

    def test_accelerator_capture_defers_the_cpu_toolchain_probe(self):
        # A capture that has only produced accelerator graphs has no CPU vector
        # width to record, so it must not run the probe. The first cpu graph
        # backfills the field: the artifact still has to say what it baked.
        with _counting_cpu_probe() as calls:
            package = CompilePackage(staged_with_graph_breaks)
            package.update_device_type(_graph_on("cuda"))
            self.assertEqual(calls, [])
            self.assertIsNone(package.cache_entry().system_info.cpu_codegen_target)

            package.update_device_type(_graph_on("cpu"))
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                package.cache_entry().system_info.cpu_codegen_target,
                SystemInfo.current().cpu_codegen_target,
            )

    def test_device_type_records_the_accelerator_not_the_last_graph(self):
        # _DynamoCacheEntry.device_type gates every GPU check in
        # SystemInfo.check_compatibility, and one package holds many graphs, so
        # a cpu epilogue after a graph break must not erase the accelerator.
        package = CompilePackage(staged_with_graph_breaks)
        package.update_device_type(_graph_on("cuda"))
        package.update_device_type(_graph_on("cpu"))
        self.assertEqual(package.cache_entry().device_type, "cuda")
        self.assertEqual(package.cache_entry().device_types, frozenset(("cpu", "cuda")))

        # A load, a cpu recompile and a re-save must not downgrade it either.
        # Loading a "cuda" entry runs check_versions and so needs a GPU; the
        # restore path does not care which accelerator it is, so the round trip
        # uses one SystemInfo does not gate and stays runnable on a cpu host.
        accel = CompilePackage(staged_with_graph_breaks)
        accel.update_device_type(_graph_on("mtia"))
        reloaded = CompilePackage(staged_with_graph_breaks, accel.cache_entry())
        reloaded.update_device_type(_graph_on("cpu"))
        self.assertEqual(reloaded.cache_entry().device_type, "mtia")

        cpu_only = CompilePackage(staged_with_graph_breaks)
        cpu_only.update_device_type(_graph_on("cpu"))
        self.assertEqual(cpu_only.cache_entry().device_type, "cpu")

    def test_scan_sys_modules_retries_after_a_later_import(self):
        # A miss usually means the module is not imported YET. Caching that
        # permanently drops the source checksum for every lazily imported file
        # for the rest of the process.
        name = "_precompile_late_import"
        pkg_dir = self._write_module("late", name, "def f(x):\n    return x + 1\n")
        self.addCleanup(dynamo_package._MODULE_KEY_BY_FILE.clear)
        path = os.path.join(pkg_dir, f"{name}.py")
        self.assertIsNone(dynamo_package._scan_sys_modules_for_file(path))
        late = self._import_module(pkg_dir, name)
        self.assertEqual(dynamo_package._scan_sys_modules_for_file(late.__file__), name)

    def test_inlined_code_is_named_by_the_file_that_holds_it(self):
        # inspect.getmodule maps a file to the name the module knows itself by,
        # and a private implementation file that spoofs __name__ answers with
        # the shim re-exporting it. The key is what load-time revalidation
        # re-imports, and __name__ imports to the shim, so it has to be the key.
        pkg = self._write_module("shim", "_shim_impl", _SHIM_IMPL_SRC)
        self._write_module("shim", "shim_abc", _SHIM_SRC)
        self._forget_modules("_shim_impl", "shim_abc")
        shim = self._import_module(pkg, "shim_abc")
        impl = sys.modules["_shim_impl"]
        self.assertIsNot(inspect.getmodule(shim.helper.__code__), impl)
        name = _defining_module_name(shim.helper.__code__)
        self.assertEqual(name, "_shim_impl")
        self.assertIs(importlib.import_module(name), impl)

    def test_a_failed_load_leaves_no_device_type_behind(self):
        # eval_frame's caching_precompile path retries a failed load on the SAME
        # package object, and update_device_type only widens cpu -> accelerator,
        # so a device_type left behind by the failed load can never be corrected
        # and gets re-saved as this capture's. One outside CHECK_GPUS turns off
        # every GPU check for whoever loads the artifact next.
        drifted = dynamo_package.SourceInfo(
            inlined_sources={
                dynamo_package.InlinedSource(
                    module="torch._dynamo.package",
                    firstlineno=1,
                    lastlineno=3,
                    checksum="not-the-checksum-on-disk",
                    content="",
                )
            }
        )
        stale = dynamo_package._DynamoCacheEntry(
            codes=[], source_info=drifted, device_type="mtia"
        )

        package = CompilePackage(None)
        with self.assertRaisesRegex(RuntimeError, "Source code changes detected"):
            package.initialize(staged_with_graph_breaks, stale)
        self.assertFalse(package.is_initialized())

        package.initialize(staged_with_graph_breaks, None)
        package.update_device_type(_graph_on("cuda"))
        entry = package.cache_entry()
        self.assertEqual(entry.device_type, "cuda")

        # "mtia" is not in CHECK_GPUS, so had it survived it would have waved
        # this artifact onto any host; what survives instead is gated.
        other_host = dataclasses.replace(entry.system_info, gpu_name="Some Other GPU")
        entry.system_info.check_compatibility(other_host, "mtia")
        self.assertIn(entry.device_type, SystemInfo.CHECK_GPUS)
        # The GPU-name check is what a surviving "mtia" would have skipped; it
        # sits behind the availability check, so a cpu host reaches it mocked.
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            self.assertRaisesRegex(RuntimeError, "different GPU"),
        ):
            entry.system_info.check_compatibility(other_host, entry.device_type)

    @parametrize("interference", sorted(_BINDING_STACK_CASES))
    def test_unload_rebinds_a_global_through_the_binding_stack(self, interference):
        values, interfere, expected = _BINDING_STACK_CASES[interference]
        module = types.ModuleType(f"test_precompile_{interference}")
        packages = [self._bare_package() for _ in values]
        for package, value in zip(packages, values):
            package._install_global(module, "g", value)
        interfere(module.__dict__)
        for package, want in zip(reversed(packages), expected):
            package.uninstall()
            self.assertEqual(module.__dict__.get("g"), want)
        self.assertEqual(dynamo_package._GLOBAL_BINDINGS.get(module, {}), {})

    def test_joining_a_dead_owners_binding_survives_its_cleanup(self):
        # A package collected without uninstall() parks its cleanup when the
        # registry lock is busy. A later install of the same artifact writes the
        # same builtins dict under the same name; the drain must find the new
        # owner on the binding and keep the name, not delete it as the dead
        # package's leftover.
        module = types.ModuleType("test_precompile_dead_owner")
        shared = {}
        dead = self._bare_package()
        dead._install_global(module, "g", shared)
        installed = dead._installed_globals
        del dead
        gc.collect()
        dynamo_package._DEAD_PACKAGES.append(
            dynamo_package._DeadPackageState(installed, [])
        )
        live = self._bare_package()
        live._install_global(module, "g", shared)
        self.assertIs(module.__dict__.get("g"), shared)
        live.uninstall()
        self.assertNotIn("g", module.__dict__)

    def test_source_graph_module_copies_are_isolated(self):
        # _src and the exec'd forward are shared between copies; everything else
        # nn.Module keeps on an instance is state, hook dicts included, and a
        # copy sharing it lets an update on one copy silently edit the other.
        # __reduce__ used to pickle the SHARED _src, whose body aliases the
        # original's parameter/buffer containers -- so mutating the original
        # after a deepcopy round-tripped the mutated tensors into the copy's
        # pickle even though live calls were isolated. __reduce__ now
        # snapshots the instance's own state.
        from torch._dynamo.precompile_context import (
            _EagerGraphSource,
            _SourceGraphModule,
        )

        src = _EagerGraphSource(
            code="def forward(self, x):\n    return x + self.b\n",
            import_block="",
            body={"_buffers": {"b": torch.ones(3)}},
        )
        original = _SourceGraphModule(src)
        dup = copy.deepcopy(original)
        dup.register_forward_hook(lambda *args: None)
        dup._non_persistent_buffers_set.add("b")
        self.assertEqual(len(original._forward_hooks), 0)
        self.assertEqual(original._non_persistent_buffers_set, set())
        self.assertIs(dup._src, original._src)

        x = torch.randn(3)
        original._buffers["b"].mul_(100)
        self.assertEqual(original(x), x + 100)
        self.assertEqual(dup(x), x + 1)  # live isolation
        self.assertEqual(pickle.loads(pickle.dumps(dup))(x), x + 1)
        # And an instance pickles its CURRENT parameters/buffers/submodules,
        # not its load-time ones (other nn.Module state still comes from _src).
        self.assertEqual(pickle.loads(pickle.dumps(original))(x), x + 100)

    def test_eager_artifact_round_trips_a_hop_graph_as_source(self):
        # GraphModule.__reduce__ re-traces the generated source at load; cond
        # rejects the Proxy and autocast enter/exit EXECUTE and leave no node.
        # The top level must travel as source, its HOP bodies as real Graphs.
        from torch._dynamo.precompile_context import (
            _SourceGraphModule,
            EagerCacheArtifact,
        )

        def fn(x):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                y = torch.cond(x.sum() > 0, lambda t: t.sin(), lambda t: t.cos(), (x,))
            return y + 1

        gms = []

        def backend(gm, example_inputs):
            gms.append(gm)
            return gm.forward

        x = torch.randn(3)
        torch.compile(fn, backend=backend, fullgraph=True)(x)
        (gm,) = gms
        artifact = EagerCacheArtifact(key="k", content=gm.forward)
        loaded = pickle.loads(pickle.dumps(artifact)).after_deserialization()
        module = loaded.__self__
        self.assertIsInstance(module, _SourceGraphModule)
        self.assertFalse(hasattr(module, "graph"))
        self.assertTrue(module._modules)
        for sub in module._modules.values():
            self.assertIsInstance(sub, torch.fx.GraphModule)
            self.assertGreater(len(sub.graph.nodes), 0)
        self.assertEqual(loaded(x), gm.forward(x))
        self.assertEqual(loaded(-x.abs()), gm.forward(-x.abs()))
        # Re-serializing a loaded artifact goes through _SourceGraphModule.__reduce__.
        reloaded = pickle.loads(pickle.dumps(pickle.loads(pickle.dumps(artifact))))
        self.assertEqual(reloaded.after_deserialization()(x), gm.forward(x))

    def test_take_artifact_removes_the_staged_backend(self):
        from torch._dynamo.precompile_context import EagerCacheArtifact

        PrecompileContext.record_artifact(EagerCacheArtifact(key="k", content=None))
        self.assertEqual(PrecompileContext.take_artifact("k").key, "k")
        self.assertIsNone(PrecompileContext.take_artifact("k"))
        self.assertIsNone(PrecompileContext.serialize_artifact_by_key("k"))

    def test_library_module_requires_the_name_to_resolve_to_the_stdlib(self):
        # The risky-drop waiver keys on the OWNER's module name, and a name is
        # not an identity: graphlib, queue, code and distutils are all stdlib
        # names a third party ships, and purelib NESTS inside stdlib (conda) or
        # platstdlib (venv), so a __file__ prefix check waived every shadow.
        stdlib_root = sysconfig.get_paths()["stdlib"]
        shadowed = types.ModuleType("graphlib")
        shadowed.__file__ = os.path.join(
            stdlib_root, "site-packages", "graphlib", "__init__.py"
        )
        unlocated = types.ModuleType("graphlib")  # no __file__, no __spec__

        for module in (shadowed, unlocated):
            with mock.patch.dict(sys.modules, {"graphlib": module}):
                self.assertFalse(dynamo_package_lint._is_library_module("graphlib"))

        # Real stdlib and real torch, including the shapes with no file at all
        # (torch._C._nn owns F.gelu; pyexpat.errors is a stdlib submodule with
        # no location evidence of its own), must keep their waiver, so
        # descendants only have to not be located somewhere else.
        import xml.parsers.expat  # noqa: F401

        for name in (
            "torch",
            "torch._C",
            "torch._C._nn",
            "torch.ops",
            "os.path",
            "collections.abc",
            "sys",
            "zipimport",
            "pyexpat.errors",
            "xml.parsers.expat.model",
        ):
            self.assertTrue(
                dynamo_package_lint._is_library_module(name), f"{name} lost its waiver"
            )
        self.assertFalse(dynamo_package_lint._is_library_module("numpy"))
        self.assertFalse(dynamo_package_lint._is_library_module(None))

    def test_an_import_alias_decodes_to_the_module_it_names(self):
        self.assertIs(_dynamo_alias_module("__import_torch"), torch)
        self.assertIs(
            _dynamo_alias_module("__import_torch_dot_nn_dot_functional"),
            torch.nn.functional,
        )
        self.assertIsNone(_dynamo_alias_module("__import_not_a_module_at_all"))
        self.assertIsNone(_dynamo_alias_module("CFG"))

    def test_facts_differing_only_in_value_sort_apart(self):
        # Once the boilerplate code parts are filtered a TENSOR_MATCH renders no
        # code at all, so two shape specializations tie on every other component
        # of the sort key and their order falls to set iteration, which is hash
        # seeded: the file then differs between PROCESSES, which two captures in
        # one process cannot show.
        def fact(shape):
            return _GuardFact("TENSOR_MATCH", "x", (), f"shape={shape}", True)

        self.assertNotEqual(_fact_order(fact((4, 8))), _fact_order(fact((5, 8))))

    def test_code_fingerprint_recurses_into_container_and_nested_consts(self):
        # _code_fingerprint names a callable by its body so an ACT2FN-style table
        # can be told apart. Two lambdas can differ ONLY inside a constant the
        # outer co_code does not distinguish: a tuple, a frozenset, or a nested
        # code object. Filtering those out whole -- rather than recursing -- gives
        # both the same digest, _object_identity names them identically, and the
        # guard that split the two compilations is reported as an invariant of
        # each.
        from torch._dynamo.precompile_package import _code_fingerprint

        pairs = {
            "tuple const": (lambda x: x * (1, 2), lambda x: x * (1, 3)),
            "frozenset const": (lambda x: x in {1, 2}, lambda x: x in {1, 3}),
            # Not called: what matters is the nested code object in co_consts.
            "nested code": (lambda x: (lambda y: y + 1), lambda x: (lambda y: y + 2)),
        }
        for label, (left, right) in pairs.items():
            self.assertEqual(
                left.__code__.co_code,
                right.__code__.co_code,
                f"{label}: the pair must differ only in co_consts",
            )
            self.assertNotEqual(
                _code_fingerprint(left.__code__),
                _code_fingerprint(right.__code__),
                f"{label}: two different bodies share a fingerprint",
            )

    def test_guard_policy_classification_is_total(self):
        # A guard type in no set is KEPT, so a drop policy can only ever
        # drop what _INVARIANT_DROPPABLE_GUARD_TYPES names. This test is
        # what makes the never-drop claim enforceable:
        # a guard type added to GuardBuilder fails here until someone triages
        # it into exactly one of the four sets.
        from torch._dynamo.guards import GuardBuilder
        from torch._dynamo.precompile_package import (
            _INVARIANT_DROPPABLE_GUARD_TYPES,
            _NOOP_GUARD_TYPES,
            _SHAPE_BEARING_GUARD_TYPES,
            _UNMODELLED_GUARD_TYPES,
        )

        guard_types = {
            name
            for name, value in vars(GuardBuilder).items()
            if name.isupper() and callable(value)
        }
        self.assertGreater(len(guard_types), 40)  # the enumeration itself works
        sets = {
            "_SHAPE_BEARING_GUARD_TYPES": _SHAPE_BEARING_GUARD_TYPES,
            "_UNMODELLED_GUARD_TYPES": _UNMODELLED_GUARD_TYPES,
            "_INVARIANT_DROPPABLE_GUARD_TYPES": _INVARIANT_DROPPABLE_GUARD_TYPES,
            "_NOOP_GUARD_TYPES": _NOOP_GUARD_TYPES,
        }
        classified: frozenset[str] = frozenset().union(*sets.values())
        self.assertEqual(
            sorted(guard_types - classified),
            [],
            "unclassified GuardBuilder guard type(s): add each to exactly one "
            "policy set in torch/_dynamo/precompile_package.py (KEPT until then)",
        )
        self.assertEqual(
            sorted(classified - guard_types),
            [],
            "phantom entries: no GuardBuilder method by these names",
        )
        for (a_name, a), (b_name, b) in itertools.combinations(sets.items(), 2):
            self.assertEqual(sorted(a & b), [], f"{a_name} overlaps {b_name}")

    def test_varying_guard_slots_counts_presence_as_variation(self):
        def fact(guard_type, source, value):
            return _GuardFact(guard_type, source, (), value, True)

        both = fact("TENSOR_MATCH", "x", "shape=(4,)")
        differs_a = fact("CONSTANT_MATCH", "n", "1")
        differs_b = fact("CONSTANT_MATCH", "n", "2")
        only_one = fact("ID_MATCH", "G['fn']", "is m.f")
        guard_sets = {
            ("f", "f.py", 1): [
                frozenset({both, differs_a, only_one}),
                frozenset({both, differs_b}),
            ]
        }
        self.assertEqual(
            varying_guard_slots(guard_sets),
            frozenset({("CONSTANT_MATCH", "n"), ("ID_MATCH", "G['fn']")}),
        )

    def test_composed_guard_filter_ands_with_the_default_and_checks_length(self):
        def entry(guard_type, derived=()):
            return GuardFilterEntry(
                name="x",
                has_value=False,
                value=None,
                guard_type=guard_type,
                derived_guard_types=tuple(derived),
                is_global=False,
                orig_guard=None,
            )

        entries = [
            entry("TENSOR_MATCH"),
            entry("ID_MATCH"),
            entry("TYPE_MATCH", ["NN_MODULE"]),
        ]
        self.assertEqual(default_guard_filter_fn(entries), [True, False, False])
        drop_first = _compose_with_default(lambda es: [False] + [True] * (len(es) - 1))
        self.assertEqual(drop_first(entries), [False, False, False])
        with self.assertRaisesRegex(ValueError, "returned 1 decisions for 3 guards"):
            _compose_with_default(lambda es: [True])(entries)

    def test_summary_is_incomplete_without_a_backend_graph(self):
        def summary(**kw):
            base = dict(
                frames=1,
                resume_functions=0,
                guarded_codes=1,
                backend_graphs=1,
                bypassed=(),
            )
            base.update(kw)
            return PrecompileSummary(**base)

        self.assertTrue(summary().complete)
        self.assertFalse(summary(backend_graphs=0).complete)
        self.assertFalse(summary(guarded_codes=0).complete)
        self.assertFalse(summary(capture_errors=("boom",)).complete)

    def test_normalize_scrubs_addresses_and_compile_counters(self):
        self.assertEqual(
            _normalize(
                "___check_obj_id(G['fn'], 140234567890123), type=<class 'function'>"
            ),
            "___check_obj_id(G['fn'], <id>), type=<class 'function'>",
        )
        self.assertEqual(
            _normalize("G['__builtins_dict___6']['len']"),
            "G['__builtins_dict___<n>']['len']",
        )
        self.assertEqual(_normalize("__compiled_fn_3_0"), "__compiled_fn_<n>")
        self.assertEqual(
            _normalize("G['__tmp_140234567890123_c7']"), "G['__tmp_<id>_c<n>']"
        )
        self.assertEqual(_normalize("x[1234567890]"), "x[1234567890]")

    def test_subgraph_renaming_leaves_lookalikes_alone(self):
        # Per-subgraph renaming is driven by AST positions. An attribute
        # (runner.call), a nested def (call inside class Runner) and a dotted
        # import path (torch._inductor.async_compile) each contain a renamed
        # name as text and must not be rewritten.
        from torch._dynamo.precompile_package import _namespace_module_names

        source = textwrap.dedent(
            """
            import torch._inductor.async_compile
            from torch._inductor import async_compile


            class Runner:
                def call(self, x):
                    return x


            runner = Runner()


            def call(x):
                return runner.call(x) + torch._inductor.async_compile.__name__
            """
        )
        (renamed,) = _namespace_module_names({"b0": source}).values()
        self.assertIn("class Runner_s0:", renamed)
        self.assertIn("runner_s0 = Runner_s0()", renamed)
        self.assertIn("def call_s0(x):", renamed)
        self.assertIn("    def call(self, x):", renamed)
        self.assertIn(
            "return runner_s0.call(x) + torch._inductor.async_compile.__name__", renamed
        )
        self.assertIn("import torch._inductor.async_compile\n", renamed)

    def test_summary_reports_dropped_guards(self):
        # Guard types the filter discards are recorded with their source name
        # rather than silently disappearing; dropping one widens what a graph
        # gets reused for, and only the name says whether that matters.
        session = precompile_capture(
            staged_with_global_dict_conditional, backend="eager", dynamic=False
        )
        with session as compiled:
            compiled(torch.randn(4, 8))
        summary = session.summary()
        # The filter cannot serialize identity guards, so referencing the torch
        # module at all produces at least one drop, reported as (type, source).
        self.assertTrue(summary.dropped_guards)
        for guard_type, source in summary.dropped_guards:
            self.assertIsInstance(guard_type, str)
            self.assertIsInstance(source, str)
        self.assertEqual(
            sum(summary.dropped_guard_types().values()), len(summary.dropped_guards)
        )
        self.assertIn("dropped guards", str(summary))
        # save() can be made to enforce that nothing was dropped.
        with self.assertRaisesRegex(PackageError, "dropped .* guard"):
            session.save(self.path(), require_no_dropped_guards=True)

    def test_strict_and_risky_drop_requirements_fail_closed(self):
        # Strict save rejects every drop. Relaxing that still keeps the risky
        # lint as a second gate unless the caller acknowledges it separately.
        clean = precompile_capture(
            PrecompileNoDispatchSlot(), backend="eager", dynamic=False
        )
        with clean as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        clean.save(self.path())
        with self.assertRaisesRegex(PackageError, "not serialized"):
            clean.save(self.path(), require_no_dropped_guards=True)

        torch._dynamo.reset()
        session = precompile_capture(
            staged_with_global_function_ref, backend="eager", dynamic=False
        )
        with session as compiled:
            compiled(torch.randn(4, 8))
        with (
            self.assertNoLogs("torch._dynamo.precompile_package", "WARNING"),
            self.assertRaisesRegex(PackageError, "PRECOMPILE_ACTIVATION"),
        ):
            session.save(self.path(), require_no_dropped_guards=False)
        with self.assertLogs("torch._dynamo.precompile_package", "WARNING") as logs:
            session.save(
                self.path(),
                require_no_risky_drops=False,
                require_no_dropped_guards=False,
            )
        reported = [m for m in logs.output if "dropped guard" in m]
        self.assertEqual(len(reported), 1, logs.output)
        self.assertIn("PRECOMPILE_ACTIVATION", reported[0])
        self.assertIn("require_no_risky_drops=False", logs.output[0])

    def test_example_inputs_drive_the_capture(self):
        # example_inputs is just "run these for me": capture is by execution, so
        # the block body becomes optional.
        session = precompile_capture(
            PrecompileInvariantModel(),
            backend="eager",
            dynamic=False,
            example_inputs=_TWO_SHAPES,
        )
        with session:
            pass
        summary = session.summary()
        self.assertEqual(summary.frames, 2)
        self.assertEqual(summary.guarded_codes, 4)

    def test_a_failing_example_input_does_not_wedge_the_session(self):
        # __enter__ runs example_inputs, and a __enter__ that raises never gets
        # its __exit__, so the config patch the session holds would stay on for
        # the life of the process and the session would refuse to save what it
        # did capture. A bare tensor instead of a 1-tuple is the likely way in.
        before = functorch_config.bundled_autograd_cache
        session = precompile_capture(
            PrecompileInvariantModel(),
            backend="eager",
            dynamic=False,
            example_inputs=[torch.ones(4, 8)],
        )
        with self.assertRaisesRegex(TypeError, "tuples of positional args"):
            with session:
                pass
        self.assertEqual(functorch_config.bundled_autograd_cache, before)
        with self.assertRaisesRegex(PackageError, "capture raised"):
            session.save(self.path())

    def test_save_rejects_capture_that_ran_nothing(self):
        # Capture is by execution. A session whose callable was never run has
        # nothing to serve, and install() would just skip the frame, so
        # serving() could not report the gap either.
        session = precompile_capture(
            staged_with_graph_breaks, backend="eager", dynamic=False
        )
        with session:
            pass
        summary = session.summary()
        self.assertEqual(summary.guarded_codes, 0)
        self.assertFalse(summary.complete)
        with self.assertRaisesRegex(PackageError, "captured no compiled code"):
            session.save(self.path())

    def test_save_refuses_capture_that_compiled_no_graph(self):
        # allow_empty_graphs turns a frame that compiled nothing into one guarded
        # code, so guarded_codes alone cannot tell this from a real capture.
        @torch._dynamo.disable
        def body(x):
            return x.sin()

        def outer(x):
            return body(x)

        session = self._capture(outer, torch.randn(4))
        summary = session.summary()
        self.assertGreater(summary.guarded_codes, 0)
        self.assertEqual(summary.backend_graphs, 0)
        self.assertFalse(summary.complete)
        with self.assertRaisesRegex(PackageError, "compiled no graph"):
            session.save(self.path())
        with self.assertRaisesRegex(PackageError, "compiled no graph"):
            session.artifact()
        session.save(self.path(), require_complete=False)

    def test_truncation_report_is_a_lower_bound(self):
        # Hitting recompile_limit sets FrameExecStrategy(RUN_ONLY, RUN_ONLY),
        # whose recursive half silences every frame called beneath the offender,
        # yet only the offender is recorded. Pin the gap the message must admit.
        inputs = [torch.randn(*s) for s in [(4, 8), (5, 8), (6, 8)]]

        def capture(limit):
            torch._dynamo.reset()
            session = precompile_capture(
                PrecompileStack(5, resume_work=True),
                backend="eager",
                recompile_limit=limit,
                dynamic=False,
            )
            with session as compiled:
                for x in inputs:
                    compiled(x)
            return session

        self.assertEqual(capture(64).summary().truncated, ())
        session = capture(8)
        summary = session.summary()
        # One frame named, but the resume frames beneath it also lost variants.
        self.assertEqual(len(summary.truncated), 1)
        self.assertIn(">=1 TRUNCATED", str(summary))
        with self.assertRaisesRegex(PackageError, "lower bound"):
            session.save(self.path())

    def test_capture_rejects_a_callable_with_no_code_object(self):
        # An nn.Module reaches the same dead end one level down: self.forward =
        # functools.partial(...) in __init__ shadows the class method, so the
        # entry function has no __code__ either. Saying so is the whole point of
        # the check; without it this died on a bare AttributeError from inside
        # CompilePackage.
        def scaled(scale, x):
            return x * scale

        class OnlyCall:
            def __call__(self, x):
                return x + 1.0

        for fn in (
            functools.partial(scaled, 2.0),
            OnlyCall(),
            PrecompilePartialForward(2.0),
        ):
            with self.assertRaisesRegex(TypeError, "no __code__"):
                precompile_capture(fn, backend="eager")

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_truncated_frame_is_not_also_reported_uncovered(self):
        # Hitting the recompile limit happens INSIDE code_context, whose exit
        # saw an attempt that added no guarded code and called the frame
        # uncovered. But it has working variants: "uncovered" means no guarded
        # code at all, which is what install() skip_code()s and what save()'s
        # message describes.
        session = precompile_capture(
            PrecompileSelfAct(torch.relu),
            backend="eager",
            dynamic=False,
            recompile_limit=2,
        )
        with session as compiled, torch.no_grad():
            for n in (3, 4, 5, 6, 7, 8):
                compiled(torch.randn(n, 4))
        summary = session.summary()
        self.assertTrue(summary.truncated)
        self.assertGreater(summary.guarded_codes, 0)
        self.assertEqual(summary.uncovered_frames, ())

    def test_install_skips_backends_only_a_bypassed_entry_references(self):
        # Deserializing an inductor artifact is expensive and can fail on a
        # serving host, so install() must not touch one for an entry it will
        # skip anyway.
        class _Exploding:
            def after_deserialization(self):
                raise AssertionError("install deserialized an unusable backend")

        model = PrecompileSelfAct(torch.relu)
        session = precompile_capture(model, backend="eager", dynamic=False)
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        package = session._package
        backend_ids = {
            backend_id
            for entry in package._codes.values()
            for backend_id in entry.backend_ids
        }
        self.assertTrue(backend_ids)
        for entry in package._codes.values():
            entry.bypassed = True
        package.install({backend_id: _Exploding() for backend_id in backend_ids})
        package.uninstall()

    def test_inductor_artifact_records_compile_time_cpu_target(self):
        from torch._inductor import cpu_vec_isa

        isa = cpu_vec_isa.pick_vec_isa()
        if not isa:
            raise unittest.SkipTest("no vector ISA on this host")
        model = PrecompileEmptyGraph()
        x = torch.randn(16)
        with torch._inductor.config.patch({"cpp.simdlen": isa.bit_width()}):
            session = precompile_capture(
                model,
                backend="inductor",
                dynamic=False,
                example_inputs=[(x,)],
            )
            with session:
                pass
            capture_target = session._package.cache_entry().system_info
        session.save(self.path())
        saved = _SingleFileStore().read(self.path()).dynamo.system_info

        self.assertEqual(capture_target.cpu_codegen_target[2], isa.bit_width())
        self.assertEqual(saved.cpu_codegen_target, capture_target.cpu_codegen_target)

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_failed_cache_install_cold_falls_back_on_same_package(self):
        from torch._dynamo.precompile_context import EagerCacheArtifact

        x = torch.randn(4, 8)
        session = precompile_capture(
            staged_with_graph_breaks, backend="eager", dynamic=False
        )
        with session as compiled:
            compiled(x)
        session.save(self.path())
        entry = _SingleFileStore().read(self.path())

        torch._dynamo.reset()
        with (
            mock.patch.object(DynamoCache, "load", return_value=entry),
            mock.patch.object(
                EagerCacheArtifact,
                "after_deserialization",
                side_effect=RuntimeError("cache install failed"),
            ),
            self.assertLogs("torch._dynamo.eval_frame", level="WARNING"),
        ):
            compiled = torch.compile(
                staged_with_graph_breaks, backend="eager", dynamic=False
            )
        self.assertEqual(compiled(x), staged_with_graph_breaks(x))

    @parametrize("shape", sorted(_RISKY_DROP_CORPUS))
    def test_risky_drop_corpus_is_flagged(self, shape):
        # See _RISKY_DROP_CORPUS: this predicate has failed open three times,
        # so every shape ever found is asserted here rather than described.
        expected, build = _RISKY_DROP_CORPUS[shape]
        model, args = build(self)
        session = self._capture(model, *args)
        risky = [name for _, name in session.summary().risky_dropped_guards]
        for name in expected:
            self.assertTrue(any(r.endswith(name) for r in risky), (name, risky))
        # Guards on the torch module itself are dropped too but are not risky.
        self.assertNotIn("G['torch']", risky)
        # Enforcement is the default; the corpus opts out only to prove that
        # every risky shape remains serializable when explicitly acknowledged.
        with self.assertRaisesRegex(PackageError, re.escape(expected[0])):
            session.save(self.path(), require_no_dropped_guards=False)
        session.save(
            self.path(),
            require_no_risky_drops=False,
            require_no_dropped_guards=False,
        )

    @parametrize("shape", sorted(_BENIGN_DROP_CORPUS))
    def test_benign_drop_corpus_is_not_flagged(self, shape):
        expected_dropped, build = _BENIGN_DROP_CORPUS[shape]
        model, args = build(self)
        session = self._capture(model, *args)
        summary = session.summary()
        dropped = [name for _, name in summary.dropped_guards]
        for name in expected_dropped:
            self.assertTrue(any(d.endswith(name) for d in dropped), (name, dropped))
        self.assertEqual(summary.risky_dropped_guards, ())
        session.save(self.path())
        with self.assertRaisesRegex(PackageError, "not serialized"):
            session.save(self.path(), require_no_dropped_guards=True)

    @parametrize("shape", sorted(_VALUE_PIN_CORPUS))
    def test_value_pinned_guards_are_reported(self, shape):
        build, args, pinned, (kept_type, kept_source) = _VALUE_PIN_CORPUS[shape]
        session = self._capture(build(), *args)
        summary = session.summary()
        self.assertTrue(summary.complete)
        self.assertEqual(summary.wont_generalize, pinned)
        self.assertTrue(
            any(t == kept_type and kept_source in n for t, n in summary.kept_guards),
            summary.kept_guards,
        )
        if pinned:
            self.assertIn("value-pinned", str(summary))

    def test_invariants_separate_what_holds_from_what_varies(self):
        # Two shapes, one config value. The config read is on both sides of the
        # break, so it is invariant; the shapes are what tell the graphs apart.
        session = precompile_capture(
            PrecompileInvariantModel(),
            backend="eager",
            dynamic=False,
            example_inputs=_TWO_SHAPES,
        )
        with session:
            pass

        frames = {f.frame: f for f in session.invariants()}
        resume = next(k for k in frames if k.startswith("torch_dynamo_resume_in"))
        entry = frames["forward"]
        self.assertEqual(entry.variants, 2)
        self.assertEqual(frames[resume].variants, 2)

        def rendered(facts):
            return [f.render() for f in facts]

        # The global read survives into every variant of the resume frame.
        self.assertTrue(
            any("['mode'] == 'sum'" in r for r in rendered(frames[resume].invariant)),
            rendered(frames[resume].invariant),
        )
        # The shapes differ between variants, so they are NOT invariant. This is
        # the half that needs the value fingerprint: TENSOR_MATCH's code_list
        # carries no shape, so without it both variants would look identical and
        # land in the intersection.
        for frame in (entry, frames[resume]):
            varies = rendered(frame.varying)
            # Rendered as guard code, so sizes read size=[4, 8].
            self.assertTrue(any("size=[4, 8]" in r for r in varies), varies)
            self.assertTrue(any("size=[5, 8]" in r for r in varies), varies)
            self.assertFalse(any("size=[" in r for r in rendered(frame.invariant)))

    def test_invariants_hold_back_the_guards_no_fingerprint_models(self):
        # GLOBAL_STATE and friends compare things this report cannot fingerprint,
        # so calling two of them equal would assert a precondition that was never
        # checked. They must land in `undetermined`, never in `invariant` --
        # emptying _UNMODELLED_GUARD_TYPES makes them all compare equal and
        # silently promotes them into the intersection.
        from torch._dynamo.precompile_package import _UNMODELLED_GUARD_TYPES

        session = precompile_capture(
            PrecompileInvariantModel(),
            backend="eager",
            dynamic=False,
            example_inputs=_TWO_SHAPES,
        )
        with session:
            pass

        frames = {f.frame: f for f in session.invariants()}
        undetermined_types = {f.guard_type for f in frames["forward"].undetermined}
        self.assertIn("GLOBAL_STATE", undetermined_types)
        self.assertIn("TORCH_FUNCTION_STATE", undetermined_types)
        determined = {
            f.guard_type
            for frame in frames.values()
            for f in (*frame.invariant, *frame.varying)
        }
        self.assertFalse(determined & _UNMODELLED_GUARD_TYPES, determined)

    @parametrize("case", sorted(_INVARIANTS_FILE_CASES))
    def test_invariants_file_is_written_and_reproducible(self, case):
        # Same contract as save(): the path names a file, written exactly where
        # asked with its parent directories created.
        build, inputs, must_contain, forbidden = _INVARIANTS_FILE_CASES[case]
        texts = []
        for name in ("a", "b"):
            path = os.path.join(self.dir(), "snapshots", name, "invariants.txt")
            self.assertFalse(os.path.exists(os.path.dirname(path)))
            torch._dynamo.reset()
            with precompile_capture(
                build(),
                backend="eager",
                dynamic=False,
                example_inputs=inputs,
                invariants=path,
            ):
                pass
            self.assertTrue(os.path.isfile(path))
            with open(path, encoding="utf-8") as handle:
                texts.append(handle.read())
        text = texts[0]
        self.assertIn("# precompile invariants for", text)
        for needle in must_contain:
            self.assertIn(needle, text)
        self.assertNotRegex(text, forbidden)
        # Object ids are normalized out, so the file is stable enough to commit
        # and diff across runs.
        self.assertEqual(texts[0], texts[1])

    def test_invariants_report_saved_tensor_hooks_by_content(self):
        # AUTOGRAD_SAVED_TENSORS_HOOKS renders a raw tuple(map(id, hooks)), so
        # the ids must be normalized or the file churns -- but erasing them
        # alone merges two variants that differ ONLY in their hooks, and the
        # guard that split them gets printed as an invariant of both. Both
        # directions at once: stable text, still discriminating.
        #
        # The hooks must be fx GraphModules or are_inline_hooks() rejects them,
        # the guard renders "ids == None", and this passes without exercising
        # anything. That is how an earlier version of this test was vacuous.
        first = tuple(
            torch.fx.symbolic_trace(f) for f in (lambda x: x + 1, lambda x: x - 1)
        )
        second = tuple(
            torch.fx.symbolic_trace(f) for f in (lambda x: x * 2, lambda x: x / 2)
        )

        def f(x):
            return (x * 2).sum()

        session = precompile_capture(f, backend="eager", dynamic=False)
        with session as compiled:
            for hooks in (first, second):
                with torch.autograd.graph.saved_tensors_hooks(*hooks):
                    compiled(torch.ones(4, 8, requires_grad=True)).backward()
        path = self.path("hooks.invariants")
        session.write_invariants(path)

        frame = session.invariants()[0]
        self.assertEqual(frame.variants, 2)
        hook_facts = [
            fact.render()
            for fact in frame.varying
            if "top_saved_tensors_hooks" in fact.render()
        ]
        # The guard really did split the two compilations, so it must be here
        # and not in the invariant set.
        self.assertEqual(len(hook_facts), 2, frame.varying)
        self.assertFalse(
            any("top_saved_tensors_hooks" in f.render() for f in frame.invariant)
        )
        with open(path, encoding="utf-8") as handle:
            self.assertNotRegex(handle.read(), r"\b\d{9,}\b")

    # Every shape found to split a compilation while the report called it an
    # invariant. The fingerprint has failed open three times -- shapes, then
    # python type and conj/neg, then the dispatch key set -- each fix revealing
    # the next, so the shapes are asserted here rather than described. A new
    # one is one line. See _value_fingerprint.
    _SPLIT_CORPUS = {
        "shape": lambda: (torch.ones(4, 8), torch.ones(5, 8)),
        "dtype": lambda: (torch.ones(4, 8), torch.ones(4, 8, dtype=torch.float64)),
        "requires_grad": lambda: (
            torch.ones(4, 8),
            torch.ones(4, 8, requires_grad=True),
        ),
        "python_type": lambda: (
            torch.nn.Parameter(torch.ones(4, 8)),
            torch.ones(4, 8),
        ),
        "conjugate_key": lambda: (
            torch.ones(4, dtype=torch.complex64),
            torch.ones(4, dtype=torch.complex64).conj(),
        ),
        "negative_key": lambda: (torch.ones(4), torch.ones(4)._neg_view()),
        "memory_format": lambda: (
            torch.ones(2, 3, 4, 5),
            torch.ones(2, 3, 4, 5).to(memory_format=torch.channels_last),
        ),
        # TensorCheck stores the TLS-ADJUSTED key set, so a tensor built
        # OUTSIDE inference_mode still splits the guard when the call is made
        # inside it. Rendering the tensor's own key set missed exactly this.
        "tls_dispatch_keys": lambda: (
            torch.ones(4, 8),
            _in_inference_mode(torch.ones(4, 8)),
        ),
        # The dimension-marking guards live entirely in code_list, which an
        # earlier version discarded as boilerplate.
        "dimension_marking": lambda: (
            torch.ones(4, 8),
            _marked_static(torch.ones(4, 8)),
        ),
    }

    @parametrize("shape", sorted(_SPLIT_CORPUS))
    def test_a_tensor_property_that_splits_a_compilation_is_reported_varying(
        self, shape
    ):
        first, second = self._SPLIT_CORPUS[shape]()

        def f(x):
            return x * 2

        torch._dynamo.reset()
        session = precompile_capture(f, backend="eager", dynamic=False)
        with session as compiled:
            for arg in (first, second):
                if isinstance(arg, _InferenceInput):
                    with torch.inference_mode():
                        compiled(arg.tensor)
                else:
                    with torch.no_grad():
                        compiled(arg)
        frame = session.invariants()[0]
        # If Dynamo did not actually split, the case is not exercising anything
        # and the corpus entry is wrong -- say so rather than passing quietly.
        self.assertEqual(frame.variants, 2, f"{shape}: dynamo did not recompile")
        # Key on the field, not the rendered text: the render is authoritative
        # guard code now and does not spell the type.
        self.assertTrue(
            any(fact.guard_type == "TENSOR_MATCH" for fact in frame.varying),
            f"{shape}: the guard that split the compilations is reported as an "
            f"invariant of both. varying={[f.render() for f in frame.varying]}",
        )
        self.assertFalse(
            any(fact.guard_type == "TENSOR_MATCH" for fact in frame.invariant)
        )

    def test_invariants_marks_unenforced_preconditions(self):
        # An invariant whose guard could not be serialized is a precondition
        # nothing rechecks at load. It has to be visibly distinct.
        session = precompile_capture(
            PrecompileInvariantModel(),
            backend="eager",
            dynamic=False,
            example_inputs=_TWO_SHAPES,
        )
        with session:
            pass
        rendered = [
            f.render() for frame in session.invariants() for f in frame.invariant
        ]
        self.assertTrue(any(r.startswith("[dropped ]") for r in rendered))
        self.assertTrue(any(r.startswith("[enforced]") for r in rendered))

    @parametrize("shape", sorted(_VARYING_CORPUS))
    def test_invariants_report_what_split_the_variants_as_varying(self, shape):
        build, kwargs, calls, must_vary, must_hold, must_not_hold = _VARYING_CORPUS[
            shape
        ]
        session = precompile_capture(build(), backend="eager", **kwargs)
        with session as compiled:
            for args in calls:
                compiled(*args)
        split = [f for f in session.invariants() if f.variants > 1]
        self.assertTrue(split, "expected a frame compiled more than once")
        for frame in split:
            invariant = [f.render() for f in frame.invariant]
            varying = [f.render() for f in frame.varying]
            self.assertEqual(len(varying), len(set(varying)), varying)
            for needle in must_vary:
                self.assertTrue(
                    any(needle in r for r in varying), (frame.frame, needle, varying)
                )
            for needle in must_hold:
                self.assertTrue(
                    any(needle in r for r in invariant),
                    (frame.frame, needle, invariant),
                )
            for needle in must_not_hold:
                self.assertFalse(
                    any(needle in r for r in invariant),
                    (frame.frame, needle, invariant),
                )

    def test_policy_keeps_input_pins_on_a_single_example(self):
        # A constant passed as an argument is a CONSTANT_MATCH on an input; one
        # example makes it invariant, but dropping it would change a served
        # answer, so the fail-closed policy must keep it.
        def f(x, k):
            return x * k

        session = precompile_capture(
            f,
            backend="eager",
            dynamic=False,
            example_inputs=[(torch.ones(4), 3)],
            prune_invariant_guards=True,
        )
        with session:
            pass
        kept = {(t, n) for t, n in session.summary().kept_guards}
        self.assertIn(("CONSTANT_MATCH", "k"), kept)
        self.assertNotIn(
            ("CONSTANT_MATCH", "k"),
            set(session.summary().policy_dropped_guards),
        )
        self.assertTrue(any(t == "TENSOR_MATCH" for t, _ in kept))

    def test_policy_drops_are_absent_from_the_reserialized_guards(self):
        from torch._dynamo.guards import strip_local_scope
        from torch._dynamo.package import load_guards_state

        session = precompile_capture(
            PrecompileStockSequential(),
            backend="eager",
            dynamic=False,
            example_inputs=[(torch.randn(n, 8),) for n in (4, 5)],
            prune_invariant_guards=True,
        )
        with session:
            pass
        dropped = set(session.summary().policy_dropped_guards)
        self.assertTrue(dropped)
        for entry in session._package.code_entries():
            for guarded in entry.guarded_codes:
                state = load_guards_state(guarded.guards_state)
                names = {
                    (g.create_fn_name(), strip_local_scope(g.name))
                    for g in state.output_graph.guards
                }
                self.assertEqual(names & dropped, set())
        # The report re-marks a policy-dropped fact rather than deleting it.
        facts = [f for fr in session.invariants() for f in fr.invariant]
        for slot in dropped:
            self.assertTrue(
                any(not f.enforced and (f.guard_type, f.source) == slot for f in facts),
                slot,
            )

    def test_policy_reserialization_failure_leaves_the_package_unpruned(self):
        session = precompile_capture(
            PrecompileStockSequential(),
            backend="eager",
            dynamic=False,
            example_inputs=[(torch.randn(4, 8),)],
            prune_invariant_guards=True,
        )
        with (
            mock.patch(
                "torch._dynamo.package.load_guards_state",
                side_effect=RuntimeError("boom"),
            ),
            self.assertRaisesRegex(
                PackageError, "cannot be rebuilt from their serialized form"
            ),
        ):
            with session:
                pass
        # Nothing was pruned, and the accounting still says so.
        summary = session.summary()
        self.assertEqual(summary.policy_dropped_guards, ())
        self.assertIn(("AUTOGRAD_SAVED_TENSORS_HOOKS", ""), set(summary.kept_guards))

    def test_hook_guards_dynamo_skips_are_not_policy_drops(self):
        # Under skip_nnmodule_hook_guards, the default, GuardBuilder emits
        # nothing for EMPTY_NN_MODULE_HOOKS_DICT, so there is no check for the
        # invariance policy to drop and none to report as dropped. With the
        # guards on, they are rooted at the served module -- an input -- so
        # the policy keeps them: only environment-rooted guards may be dropped.
        def hook_slots(session):
            return [
                slot
                for slot in session.summary().policy_dropped_guards
                if slot[0] == "EMPTY_NN_MODULE_HOOKS_DICT"
            ]

        def capture():
            session = precompile_capture(
                PrecompileStockSequential(),
                backend="eager",
                dynamic=False,
                example_inputs=[(torch.randn(n, 8),) for n in (4, 5)],
                prune_invariant_guards=True,
            )
            with session:
                pass
            return session

        session = capture()
        self.assertEqual(hook_slots(session), [])
        self.assertIn(
            ("AUTOGRAD_SAVED_TENSORS_HOOKS", ""),
            session.summary().policy_dropped_guards,
        )
        torch._dynamo.reset()
        with torch._dynamo.config.patch(skip_nnmodule_hook_guards=False):
            session = capture()
        self.assertEqual(hook_slots(session), [])
        kept_types = {t for t, _ in session.summary().kept_guards}
        self.assertIn("EMPTY_NN_MODULE_HOOKS_DICT", kept_types)

    @torch.compiler.set_stance("default")
    def test_overlapping_serving_contexts_keep_compilation_disabled(self):
        torch._dynamo.reset()

        @torch.compile(backend="eager", dynamic=False)
        def compiled(x):
            return x + 1

        compiled(torch.randn(2))
        both_serving = threading.Barrier(2, timeout=10)
        a_exited = threading.Event()
        errors = queue.SimpleQueue()
        rejected = queue.SimpleQueue()

        def serve_a():
            try:
                with serving():
                    both_serving.wait()
                a_exited.set()
            except Exception as e:
                errors.put(e)

        def serve_b():
            try:
                with serving():
                    both_serving.wait()
                    if not a_exited.wait(10):
                        raise RuntimeError("timed out waiting for serving thread")
                    try:
                        compiled(torch.randn(3))
                    except RuntimeError as e:
                        rejected.put("fail_on_recompile" in str(e))
                    else:
                        rejected.put(False)
            except Exception as e:
                errors.put(e)

        threads = [threading.Thread(target=fn) for fn in (serve_a, serve_b)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertTrue(errors.empty())
        self.assertTrue(rejected.get_nowait())
        self.assertTrue(rejected.empty())
        self.assertEqual(torch._dynamo.eval_frame._stance.stance, "default")

    def test_precompile_entries_are_region_scoped_in_both_directions(self):
        # Two rails that pull against each other. A precompile entry installed
        # for the DEFAULT region must not be served to an isolated one (the
        # identity guards that would tell two artifacts apart are the ones
        # precompile drops), and the caching_precompile loader must therefore
        # install into the region its own context looks up in.
        x = torch.randn(3, 4)
        model = PrecompileSelfAct(torch.relu)
        self._capture(model, x).save(self.path(), require_no_risky_drops=False)

        def installed(into_region):
            torch._dynamo.reset()
            cache_entry = _SingleFileStore().load_cache_entry(self.path())
            package = CompilePackage(model.forward, cache_entry.dynamo)
            ctx = torch._dynamo.optimize(
                "eager", dynamic=False, isolate_recompiles=True
            )
            region = {"isolate_recompiles_id": ctx._isolate_recompiles_id}
            package.install(cache_entry.backends, **(region if into_region else {}))
            return package, ctx(model)

        # Rail 1: default-region install is NOT visible from an isolated region.
        package, isolated = installed(into_region=False)
        try:
            with (
                torch.no_grad(),
                serving(),
                self.assertRaisesRegex(RecompileError, "Detected recompile"),
            ):
                isolated(x)
        finally:
            package.uninstall()

        # Rail 2: installing into the region the context uses does serve.
        package, isolated = installed(into_region=True)
        try:
            with torch.no_grad(), serving():
                self.assertEqual(isolated(x), model(x))
        finally:
            package.uninstall()

    @parametrize("error_type", (RuntimeError, KeyboardInterrupt))
    def test_a_failed_install_leaves_nothing_installed(self, error_type):
        self._capture(staged_with_graph_breaks, torch.randn(3, 4)).save(self.path())

        torch._dynamo.reset()
        cache_entry = _SingleFileStore().load_cache_entry(self.path())
        backends = cache_entry.backends

        class _Boom:
            def after_deserialization(self):
                raise error_type("artifact will not deserialize on this host")

        # The last frame's backend is the one that fails, so the earlier frames
        # are already installed when install() gives up. A backend rejected by
        # the serving host is the realistic way into this.
        backends[cache_entry.dynamo.codes[-1].backend_ids[-1]] = _Boom()

        scope = staged_with_graph_breaks.__globals__
        before = set(scope)
        package = CompilePackage(staged_with_graph_breaks, cache_entry.dynamo)
        torch._dynamo.optimize("eager", package=package, dynamic=False)(
            staged_with_graph_breaks
        )
        with self.assertRaisesRegex(error_type, "will not deserialize"):
            package.install(backends)

        # install() raising leaves the caller no handle to unload with, so a
        # partial install would be permanent: some frames served, some not.
        leftover = [n for n in set(scope) - before if not n.startswith("__import_")]
        self.assertEqual(sorted(leftover), [])
        self.assertEqual(
            _debug_get_precompile_entries(staged_with_graph_breaks.__code__), []
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
