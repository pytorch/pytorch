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
import tempfile
import textwrap
import threading
import types
import unittest
import weakref
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
    precompile_load,
    serving,
    varying_guard_slots,
)
from torch._dynamo.types import FrameAction, FrameExecStrategy, GuardFilterEntry
from torch._dynamo.utils import CleanupHook
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


_MIPS_TARGET = ("mips", "DEFAULT", 128, ("INVALID",), None, "INVALID")

# recorded target, whether the host probe works, what the comparison raises
_CPU_TARGET_CASES = {
    "no_target_recorded": (None, True, None),
    "skewed_target": (_MIPS_TARGET, True, "built for machine 'mips'"),
    "host_probe_failed": (
        ("x86_64", "AVX512", 512, ("CPU_CAPABILITY_AVX512",), None, "avx512"),
        False,
        "no usable",
    ),
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


class _PackageDescriptorHolder:
    """Code defined under descriptors, which getattr on the class hides."""

    @property
    def getter_only(self):
        def inner(x):
            return x + 1

        return inner(2)

    @functools.cached_property
    def cached(self):
        return 3

    @property
    def pair(self):
        return 1

    @pair.setter
    def pair(self, v):
        self._v = v


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


# Rows: builder, input calls, text the invariants file must contain, and a
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
# carry, substrings no invariant fact may carry, and whether the single call
# must be made once under no_grad and once under enable_grad to split it.
_VARYING_CORPUS = {
    # _normalize strips object ids so the file diffs clean. It must not strip a
    # user constant with it: these two variants pin the dict to different keys.
    "large_constant": (lambda: _scale_by_first_value, {"dynamic": False}, [(torch.ones(4), {1000000001: 2}), (torch.ones(4), {2000000002: 3})], ("[1000000001]", "[2000000002]"), (), ("dict.keys",), False),
    # k is unspecialized, so its guard is "k is an int" in both variants and
    # is a real precondition. Fingerprinting the value it happened to hold
    # would split one shared guard into two indistinguishable varying facts.
    "shared_int_guard": (lambda: _mul_or_add, {"dynamic": True}, [(torch.ones(4), True, 1), (torch.ones(4), False, 2)], (), ("___check_type_id(L['k']",), (), False),
    # The id in an identity guard's code is normalized away, so the object has
    # to be named some other way, or self.act is reported invariant.
    "identity_guard_named": (lambda: PrecompileSelfActPair(torch.relu, torch.sigmoid), {"dynamic": False}, [(torch.ones(4),)], ("relu on self.act", "sigmoid on self.act"), (), (".act",), False),
    # Every entry of an ACT2FN-style table is "<lambda>" in one module on one
    # line; _object_identity must still tell them apart.
    "lambda_table_pair": (lambda: PrecompileSelfActPair(*_LAMBDA_TABLE.values()), {"dynamic": False}, [(torch.randn(3, 4),)], ("self.act",), (), ("self.act",), False),
    # The same call made once under no_grad and once under enable_grad compiles
    # twice; global-state guards carry no value of their own, so without a
    # fingerprint nothing would be reported varying.
    "grad_mode": (PrecompileInvariantModel, {"dynamic": False}, [(torch.ones(4, 8),)], ("grad_enabled=True", "grad_enabled=False"), (), (), True),
}  # fmt: skip


# A pair that differs only after the break, so a resume function borrowed from
# the wrong artifact still runs and still returns a number.
def staged_break_then_add_one(x):
    y = x * 2
    torch._dynamo.graph_break()
    return (y + 1).sum()


def staged_break_then_add_thousand(x):
    y = x * 2
    torch._dynamo.graph_break()
    return (y + 1000).sum()


def _resume_names_in(path):
    with open(path, "rb") as f:
        entry = pickle.load(f)
    return [
        name
        for code in entry.dynamo.codes
        if code.install_to_global
        for name in code.function_names
    ]


def _rename_resume_function(path, old, new):
    """
    Give a saved artifact the resume-function name a SEPARATE capture process
    would have minted for it. The name comes from a counter that restarts per
    process, so two captures in one process cannot collide, but two capture
    processes routinely do.
    """
    with open(path, "rb") as f:
        entry = pickle.load(f)
    for code in entry.dynamo.codes:
        code.function_names = [new if n == old else n for n in code.function_names]
        for guarded in code.guarded_codes:
            guarded.dynamo_code = dataclasses.replace(
                guarded.dynamo_code,
                co_names=tuple(
                    new if n == old else n for n in guarded.dynamo_code.co_names
                ),
            )
    with open(path, "wb") as f:
        pickle.dump(entry, f)


# Content-addressing the resume code tells the first pair apart, but not two
# instances of ONE model class: the same script captured in two processes
# mints the same __resume_at_<offset>_<n> AND byte-identical resume code.
_COLLIDING_RESUME_PAIRS = {
    "separate_captures": (lambda: (staged_break_then_add_one, staged_break_then_add_thousand), {}),
    "identical_captures": (lambda: (PrecompileSelfAct(torch.relu), PrecompileSelfAct(torch.sigmoid)), {"require_no_risky_drops": False}),
}  # fmt: skip


def staged_with_nested_dict_conditional(x, cfg):
    # membership, nested lookup, and iteration over the key set, which produce
    # DICT_CONTAINS / DICT_NOT_CONTAINS / DICT_KEYS_MATCH rather than a plain
    # value comparison.
    if "alpha" in cfg and cfg["alpha"]["kind"] == "wide":
        x = x * len(cfg["alpha"]["dims"])
    else:
        x = x + 1
    torch._dynamo.graph_break()
    total = 0
    for k in sorted(cfg):
        total += cfg[k]["weight"]
    return x.sum() * total


def staged_with_local_dict_conditional(x, cfg):
    if cfg["op"] == "sin":
        x = x.sin()
    else:
        x = x.cos()
    torch._dynamo.graph_break()
    return x.sum() * cfg["scale"]


@contextlib.contextmanager
def _precompile_mode(mode):
    old = PRECOMPILE_CONFIG["mode"]
    PRECOMPILE_CONFIG["mode"] = mode
    try:
        yield
    finally:
        PRECOMPILE_CONFIG["mode"] = old


# Rows: fn, the captured variants as (PRECOMPILE_CONFIG mode, extra args), one
# uncaptured variant, guard types that must be emitted AND kept, and whether
# save() has to be told the drops it reports are acknowledged risky ones.
_DICT_GUARD_CORPUS = {
    "global_dict": (staged_with_global_dict_conditional, [("sum", ()), ("mean", ())], ("uncaptured", ()), (), False),
    "local_dict": (staged_with_local_dict_conditional, [(None, ({"op": "sin", "scale": 2},)), (None, ({"op": "cos", "scale": 5},))], (None, ({"op": "tan", "scale": 1},)), (), False),
    "nested_dict": (staged_with_nested_dict_conditional, [(None, ({"alpha": {"kind": "wide", "dims": [1, 2], "weight": 3}},)), (None, ({"beta": {"kind": "narrow", "dims": [1], "weight": 7}},))], (None, ({"gamma": {"kind": "wide", "dims": [1], "weight": 2}},)), ("DICT_KEYS_MATCH", "DICT_CONTAINS"), True),
}  # fmt: skip

_BUILTIN_ACROSS_BREAK_SRC = """\
import torch

class Model(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x, cfg):
        # `f` holds a builtin across the break, so the dynamo bytecode puts it
        # back by READING Dynamo's builtins-dict global rather than by
        # resolving the name again through the ordinary lookup.
        f = len
        y = x * self.scale
        torch._dynamo.graph_break()
        return y.sum() * f(cfg)
"""

_SHARED_FRAME_SRC = """\
import torch

class SharedBlock(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        y = x * 2
        marker = y.sum().item()
        return y * self.scale + marker * 0.0

class ModelOne(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.block = SharedBlock(scale)

    def forward(self, x):
        return self.block(x).sum()

class ModelTwo(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.block = SharedBlock(scale)

    def forward(self, x):
        return self.block(x).sum() + 0.0
"""


class PrecompileBreakOnlyWhenFalse(torch.nn.Module):
    """The uncovered branch breaks AGAIN, so a fallback compile mints new globals."""

    def forward(self, x, flag):
        y = x * 2
        torch._dynamo.graph_break()
        if flag:
            return y + 1
        z = y - 1
        torch._dynamo.graph_break()
        return z * 3


class _PrecompileForwardsCode:
    """A wrapper forwarding __code__ with a safe default -- the usual decorator
    spelling -- around a builtin, so the attribute is present and None."""

    def __init__(self, fn):
        self.__code__ = getattr(fn, "__code__", None)
        self._fn = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


# functools.wraps copies __qualname__, so an instrumented wrapper around a
# different callable passes the qualname check while being a different code
# object.
@functools.wraps(staged_with_graph_breaks)
def _instrumented_staged_with_graph_breaks(x):
    return staged_with_local_dict_conditional(x, {"op": "sin", "scale": 2})


# CompilePackage rebinds the stored guards onto whatever callable it is given,
# so load has to refuse anything but the captured callable itself; a __code__
# that is present and None gets past the capture-side hasattr check.
_FOREIGN_CALLABLES = {
    "different_function": (staged_with_local_dict_conditional, "captured from"),
    "wrapper_sharing_the_qualname": (_instrumented_staged_with_graph_breaks, "captured from code object"),
    "code_attribute_is_none": (_PrecompileForwardsCode(len), "no __code__"),
}  # fmt: skip


def _forget_torch_version(entry):
    entry.dynamo.system_info = dataclasses.replace(
        entry.dynamo.system_info, torch_version="0.0.0"
    )


def _forget_codes(entry):
    entry.dynamo.codes = []


# How the saved artifact is damaged (None: the path is a directory, the
# pre-single-file layout) and the error precompile_load must name it with.
_DAMAGED_ARTIFACTS = {
    "different_torch": (_forget_torch_version, RuntimeError, "different PyTorch version"),
    "no_code_entries": (_forget_codes, PackageError, "no code entries"),
    "directory": (None, PackageError, "is a directory"),
}  # fmt: skip

_ENCODER_SRC = """\
import torch


class Encoder(torch.nn.Module):
    def forward(self, x):
        return x {op} 1.0
"""

_FLIPPED_ENCODER_SRC = """\
import os

import torch


if os.environ.get("FLIP_V2") == "1":
    class Encoder(torch.nn.Module):
        def forward(self, x):
            return x * 7.0
else:
    class Encoder(torch.nn.Module):
        def forward(self, x):
            return x + 1.0
"""

_SHIM_MODEL_SRC = """\
import shim_abc
import torch

class Model(torch.nn.Module):
    def forward(self, x):
        return shim_abc.helper(x).sum()
"""

_DECORATOR_SRC = """\
import functools

ENABLE = True

def deco(fn):
    @functools.wraps(fn)
    def wrapper(x):
        y = x * 2 if ENABLE else x * 3
        scalar = x.sum().item()
        return y + scalar
    return wrapper
"""

_WRAPPED_ENTRY_SRC = """\
from precompile_entry_decorators import deco

@deco
def staged(x):
    return x
"""


def _run_only(code):
    strategy = FrameExecStrategy(FrameAction.RUN_ONLY, FrameAction.RUN_ONLY)
    torch._dynamo.eval_frame.set_code_exec_strategy(code, strategy)


# A legacy package's empty frame installs a skip only in its region; whatever
# global strategy the frame had before the load, or gained while loaded, has
# to be what unload leaves. Rows: applied before the load, applied after it,
# the (cur, recursive) strategy expected after unload.
_SKIP_INTERFERENCE = {
    "none": (None, None, (FrameAction.DEFAULT, FrameAction.DEFAULT)),
    "skip_before_load": (torch._dynamo.eval_frame.skip_code, None, (FrameAction.SKIP, FrameAction.DEFAULT)),
    "skip_after_load": (None, torch._dynamo.eval_frame.skip_code, (FrameAction.SKIP, FrameAction.DEFAULT)),
    "run_only_after_load": (None, _run_only, (FrameAction.RUN_ONLY, FrameAction.RUN_ONLY)),
}  # fmt: skip


def _names(scope, *prefixes):
    return sorted(name for name in scope if name.startswith(prefixes))


def _precompile_unreachable_helper(y):
    z = y * 3
    torch._dynamo.graph_break()
    return z.sum()


def _precompile_unreachable_entry(x, scale=2.0):
    # Without nested_graph_breaks the helper's break stops its inlining: its
    # frame is entered by an ordinary call no name in the entry reaches.
    return _precompile_unreachable_helper(x * 2) * scale


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

    def _load(self, fn, path=None, **kwargs):
        return precompile_load(
            fn, path or self.path(), backend="eager", dynamic=False, **kwargs
        )

    def _save_relu_and_sigmoid_artifacts(self, x):
        # Two instances of one class share a forward code object. self.act is a
        # dispatch slot, which these tests are not about, so its drop is waived.
        paths = []
        for act in (torch.relu, torch.sigmoid):
            torch._dynamo.reset()
            session = precompile_capture(
                PrecompileSelfAct(act), backend="eager", dynamic=False
            )
            with session as compiled, torch.no_grad():
                compiled(x)
            path = self.path(f"{act.__name__}.pt")
            session.save(path, require_no_risky_drops=False)
            paths.append(path)
        return paths

    def _save_legacy_empty_graph_package(self):
        session = precompile_capture(PrecompileEmptyGraph(), backend="eager")
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        entry = session._package.cache_entry().codes[0]
        entry.guarded_codes.clear()
        entry.backend_ids.clear()
        session._package.cached_backends.clear()
        summary = session.save(self.path(), require_complete=False)
        self.assertEqual(summary.guarded_codes, 0)

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
        from torch._functorch._aot_autograd.to_standalone_python import (
            namespace_module_names,
        )

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
        (renamed,) = namespace_module_names([source])
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

    def test_caller_driven_calls_drive_the_capture(self):
        # Capture is by execution: the calls the caller makes inside the block
        # are what fold into the artifact.
        session = precompile_capture(
            PrecompileInvariantModel(),
            backend="eager",
            dynamic=False,
        )
        with session as compiled:
            for args in _TWO_SHAPES:
                compiled(*args)
        summary = session.summary()
        self.assertEqual(summary.frames, 2)
        self.assertEqual(summary.guarded_codes, 4)

    def test_a_failing_call_does_not_wedge_the_session(self):
        # A call that raises inside the block still triggers __exit__ (normal
        # context-manager semantics), so the config patch the session holds is
        # restored rather than staying on for the life of the process, and the
        # session refuses to save what it did not finish capturing.
        before = functorch_config.bundled_autograd_cache
        session = precompile_capture(
            PrecompileInvariantModel(),
            backend="eager",
            dynamic=False,
        )
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with session as compiled:
                compiled(torch.ones(4, 8))
                raise RuntimeError("boom")
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
            )
            with session as compiled:
                compiled(x)
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
        )
        with session as compiled:
            for args in _TWO_SHAPES:
                compiled(*args)

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
        )
        with session as compiled:
            for args in _TWO_SHAPES:
                compiled(*args)

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
                invariants=path,
            ) as compiled:
                for args in inputs:
                    compiled(*args)
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
        )
        with session as compiled:
            for args in _TWO_SHAPES:
                compiled(*args)
        rendered = [
            f.render() for frame in session.invariants() for f in frame.invariant
        ]
        self.assertTrue(any(r.startswith("[dropped ]") for r in rendered))
        self.assertTrue(any(r.startswith("[enforced]") for r in rendered))

    @parametrize("shape", sorted(_VARYING_CORPUS))
    def test_invariants_report_what_split_the_variants_as_varying(self, shape):
        build, kwargs, calls, must_vary, must_hold, must_not_hold, grad_split = (
            _VARYING_CORPUS[shape]
        )
        session = precompile_capture(build(), backend="eager", **kwargs)
        with session as compiled:
            if grad_split:
                with torch.no_grad():
                    compiled(*calls[0])
                with torch.enable_grad():
                    compiled(*calls[0])
            else:
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
            prune_invariant_guards=True,
        )
        with session as compiled:
            compiled(torch.ones(4), 3)
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
            prune_invariant_guards=True,
        )
        with session as compiled:
            for n in (4, 5):
                compiled(torch.randn(n, 8))
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
            with session as compiled:
                compiled(torch.randn(4, 8))
        # Nothing was pruned, and the accounting still says so.
        summary = session.summary()
        self.assertEqual(summary.policy_dropped_guards, ())
        self.assertIn(("AUTOGRAD_SAVED_TENSORS_HOOKS", ""), set(summary.kept_guards))

    @torch._dynamo.config.patch(nested_graph_breaks=False)
    def test_unreachable_frame_capture_is_served_by_installing(self):
        from torch._dynamo.utils import counters
        from torch._precompile import (
            _parse_artifact_metadata,
            PrecompiledCallable,
            PrecompileError,
        )

        x = torch.randn(4)
        session = precompile_capture(
            _precompile_unreachable_entry,
            backend="eager",
            dynamic=False,
        )
        with session as compiled, torch.no_grad():
            compiled(x)
        code, cache = session.artifact()
        meta = _parse_artifact_metadata(code)
        self.assertEqual(meta["SERVING_MODE"], "installed")
        self.assertIn(
            "_precompile_unreachable_helper", str(meta["UNREACHABLE_WITHOUT_INSTALL"])
        )

        torch._dynamo.reset()
        with tempfile.TemporaryDirectory() as d:
            artifact_path = os.path.join(d, "m.py")
            cache_path = os.path.join(d, "m.cache")
            with open(artifact_path, "w", encoding="utf-8") as f:
                f.write(code)
            with open(cache_path, "wb") as f:
                f.write(cache)
            loaded = torch.compiler.precompile.load(artifact_path, cache_path)
        self.assertIsInstance(loaded, PrecompiledCallable)
        helper_code = _precompile_unreachable_helper.__code__
        self.assertEqual(_debug_get_precompile_entries(helper_code), [])
        counters.clear()
        with loaded, torch.no_grad():
            # The omitted default comes back from the artifact.
            self.assertEqual(loaded(x), _precompile_unreachable_entry(x))
            self.assertTrue(_debug_get_precompile_entries(helper_code))
            self.assertEqual(counters["stats"]["unique_graphs"], 0)
        self.assertEqual(_debug_get_precompile_entries(helper_code), [])
        with self.assertRaisesRegex(PrecompileError, "has been unloaded"):
            loaded(x)
        with loaded, torch.no_grad():
            self.assertEqual(loaded(x), _precompile_unreachable_entry(x))

    def test_hook_guards_dynamo_skips_are_not_policy_drops(self):
        # Under skip_nnmodule_hook_guards, the default, GuardBuilder emits
        # nothing for EMPTY_NN_MODULE_HOOKS_DICT, so there is no check for the
        # invariance policy to drop and none to report as dropped. Turn the
        # guards on and they DO appear -- rooted at the served module, which
        # the policy classes as environment by value, so it drops them. Their
        # absence by default is Dynamo skipping the guard, not a policy drop.
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
                prune_invariant_guards=True,
            )
            with session as compiled:
                for n in (4, 5):
                    compiled(torch.randn(n, 8))
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
        self.assertTrue(hook_slots(session))

    def test_graph_breaks_and_recompiles_round_trip(self):
        shapes = [(4, 8), (5, 8), (6, 8)]
        inputs = [torch.randn(*s) for s in shapes]
        expected = [staged_with_graph_breaks(x) for x in inputs]

        session = precompile_capture(
            staged_with_graph_breaks, backend="eager", dynamic=False
        )
        with session as compiled:
            for x in inputs:
                compiled(x)
        summary = session.summary()
        # entry frame plus one resume frame per graph break, each specialized
        # once per input shape.
        self.assertEqual(summary.frames, 3)
        self.assertEqual(summary.resume_functions, 2)
        self.assertEqual(summary.guarded_codes, 3 * len(shapes))
        self.assertTrue(summary.complete)
        session.save(self.path(), require_no_dropped_guards=False)

        torch._dynamo.reset()
        with (
            self._load(staged_with_graph_breaks) as loaded,
            serving(),
        ):
            for x, want in zip(inputs, expected):
                self.assertEqual(loaded(x), want)
            with self.assertRaisesRegex(RuntimeError, "fail_on_recompile"):
                loaded(torch.randn(9, 8))

    @parametrize("shape", sorted(_DICT_GUARD_CORPUS))
    def test_dict_guards_round_trip(self, shape):
        fn, variants, uncaptured, must_keep, risky_drops = _DICT_GUARD_CORPUS[shape]
        x = torch.randn(4, 8)

        def run(call, mode, args):
            with _precompile_mode(mode) if mode else contextlib.nullcontext():
                return call(x, *args)

        expected = [run(fn, *variant) for variant in variants]
        self.assertNotEqual(expected[0].item(), expected[1].item())

        session = precompile_capture(fn, backend="eager", dynamic=False)
        with session as compiled:
            for variant in variants:
                run(compiled, *variant)
        summary = session.summary()
        # entry frame + one resume frame, each specialized per variant
        self.assertEqual(summary.frames, 2)
        self.assertEqual(summary.resume_functions, 1)
        self.assertEqual(summary.guarded_codes, 2 * len(variants))
        self.assertTrue(summary.complete)
        # Assert positively that the key-set and membership guards were emitted
        # AND retained. Checking only that they are absent from dropped_guards
        # would pass just as well if Dynamo never emitted them at all.
        for guard_type in must_keep:
            self.assertIn(guard_type, summary.kept_guard_types())
            self.assertNotIn(guard_type, summary.dropped_guard_types())
        session.save(self.path(), require_no_risky_drops=not risky_drops)

        torch._dynamo.reset()
        with (
            self._load(fn) as loaded,
            serving(),
        ):
            # The guard must be load-bearing: flipping it has to select the
            # other graph rather than silently reusing the first, and a variant
            # never captured must not match either graph.
            for variant, want in zip(variants, expected):
                self.assertEqual(run(loaded, *variant), want)
            with self.assertRaisesRegex(RuntimeError, "fail_on_recompile"):
                run(loaded, *uncaptured)

    @parametrize("limit", (8, 64))
    def test_an_incomplete_capture_saves_only_when_acknowledged(self, limit):
        # 5 blocks x 3 shapes = 15 variants on one shared forward code object,
        # which overruns a recompile_limit of 8; raising the limit captures
        # every variant, but the top-level forward only dispatches to
        # submodules and still has no guarded code of its own.
        n = 5
        model = PrecompileStack(n)
        inputs = [torch.randn(*s) for s in [(4, 8), (5, 8), (6, 8)]]
        expected = [model(x) for x in inputs]

        session = precompile_capture(
            model, backend="eager", recompile_limit=limit, dynamic=False
        )
        with session as compiled:
            for x in inputs:
                compiled(x)

        summary = session.summary()
        self.assertFalse(summary.complete)
        self.assertEqual(summary.bypassed, ())
        self.assertEqual(summary.uncovered_frames, ("forward",))
        if limit == 8:
            self.assertTrue(summary.truncated)
            self.assertGreater(summary.guarded_codes, 0)
            refusal = "exceeded recompile_limit"
        else:
            self.assertEqual(summary.truncated, ())
            # Explicit precompile enables empty graphs, so each no-op resume
            # after a block's graph break is guarded alongside every block variant.
            self.assertEqual(summary.guarded_codes, (n + 1) * len(inputs))
            refusal = "produced NO guarded code at all"
        with self.assertRaisesRegex(PackageError, refusal):
            session.save(self.path())

        # Opting in to a partial artifact is still allowed, and the variants
        # that WERE captured must still serve -- truncation records a gap, it
        # does not throw away the coverage already obtained.
        session.save(self.path(), require_complete=False)
        torch._dynamo.reset()
        with (
            self._load(model, recompile_limit=limit) as loaded,
            serving(),
        ):
            served = inputs if limit == 64 else inputs[:1]
            for x, want in zip(served, expected):
                self.assertEqual(loaded(x), want)
            with self.assertRaisesRegex(RuntimeError, "fail_on_recompile"):
                loaded(torch.randn(7, 8))

    def test_shared_frame_artifacts_coexist(self):
        # Two instances of one class share a forward code object, but each
        # loaded callable dispatches only through its own compile region, so
        # unloading either package must leave the other's entries alone.
        x = torch.randn(3, 4)
        paths = self._save_relu_and_sigmoid_artifacts(x)
        relu_model = PrecompileSelfAct(torch.relu)
        sigmoid_model = PrecompileSelfAct(torch.sigmoid)
        with torch.no_grad():
            expected_relu, expected_sigmoid = relu_model(x), sigmoid_model(x)

        torch._dynamo.reset()
        first = self._load(relu_model, paths[0])
        with torch.no_grad(), serving():
            self.assertEqual(first(x), expected_relu)
        with self.assertNoLogs("torch._dynamo.package", level="WARNING"):
            second = self._load(sigmoid_model, paths[1])
        entries = _debug_get_precompile_entries(PrecompileSelfAct.forward.__code__)
        self.assertEqual(
            {entry.isolate_recompiles_id for entry in entries},
            {first._isolate_recompiles_id, second._isolate_recompiles_id},
        )
        with torch.no_grad(), serving():
            self.assertEqual(first(x), expected_relu)
            self.assertEqual(second(x), expected_sigmoid)
        with self.assertNoLogs("torch._dynamo.package", level="WARNING"):
            first.unload()
        with torch.no_grad(), serving():
            self.assertEqual(second(x), expected_sigmoid)
        with self.assertNoLogs("torch._dynamo.package", level="WARNING"):
            second.unload()

        # A repeated unload must not clear a package loaded after it either.
        third = self._load(relu_model, paths[0])
        with third:
            third.unload()
            fourth = self._load(relu_model, paths[0])
        with fourth, torch.no_grad(), serving():
            self.assertEqual(fourth(x), expected_relu)

    def test_stale_artifact_rejected_when_source_drifts(self):
        # The deployment shape is capture on one machine, serve on another. The
        # dangerous version of that is an artifact outliving a code change, so
        # the source checksum has to fire even though the module is found by
        # name and its path differs between the two machines.
        src = "import torch\n\n\ndef staged(x):\n    y = x * 2\n    torch._dynamo.graph_break()\n    return (y + 1).sum()\n"
        pkg_dir = self._write_module("srcdrift", "drift_mod", src)
        mod = self._import_module(pkg_dir, "drift_mod")
        self._capture(mod.staged, torch.randn(4, 8)).save(self.path())

        # The serving machine runs a slightly different build.
        self._write_module("srcdrift", "drift_mod", src.replace("y + 1", "y + 2"))
        mod2 = self._import_module(pkg_dir, "drift_mod")
        torch._dynamo.reset()
        with self.assertRaisesRegex(RuntimeError, "Source code changes detected"):
            self._load(mod2.staged)

    def test_eager_precompile_does_not_need_a_cxx_toolchain(self):
        # cpu_codegen_target only guards inductor's baked CPU vector width, but
        # computing it dry-compiles a probe, so on a host with no compiler an
        # eager capture -- and, worse, an eager LOAD in a serve-only container --
        # died with InvalidCxxCompiler on a field nothing will read.
        model = PrecompileEmptyGraph()
        x = torch.randn(4)
        with _counting_cpu_probe(toolchain=False) as calls:
            session = precompile_capture(model, backend="eager", dynamic=False)
            with session as compiled, torch.no_grad():
                compiled(x)
            entry = session._package.cache_entry()
            self.assertFalse(entry.requires_native_backend_compatibility)
            self.assertIsNone(entry.system_info.cpu_codegen_target)
            session.save(self.path())

            torch._dynamo.reset()
            loaded = self._load(model)
            with loaded, torch.no_grad(), serving():
                self.assertEqual(loaded(x), model(x))
        self.assertEqual(calls, [])

        # Nor does an eager artifact care what CPU target the serving host has.
        skewed = dataclasses.replace(
            SystemInfo.current(), cpu_codegen_target=_MIPS_TARGET
        )
        torch._dynamo.reset()
        with mock.patch.object(SystemInfo, "current", return_value=skewed):
            loaded = self._load(model)
        with loaded, torch.no_grad(), serving():
            self.assertEqual(loaded(x), model(x))

    def test_reset_probe_ignores_an_installed_code_with_no_entries(self):
        # The probe has to be a code that really RECEIVED entries. A top-level
        # forward that only dispatches to submodules produces no guarded code of
        # its own, is installed anyway so its children can serve, and gets zero
        # entries. Probing "the first installed code" would call that a dropped
        # install and raise PackageError on every call, forever.
        model = PrecompileStack(5)
        inputs = [torch.randn(*shape) for shape in ((4, 8), (5, 8), (6, 8))]
        expected = [model(x) for x in inputs]

        session = precompile_capture(
            model, backend="eager", recompile_limit=64, dynamic=False
        )
        with session as compiled:
            for x in inputs:
                compiled(x)
        self.assertEqual(session.summary().uncovered_frames, ("forward",))
        session.save(
            self.path(), require_complete=False, require_no_dropped_guards=False
        )

        torch._dynamo.reset()
        with (
            self._load(model, recompile_limit=64) as loaded,
            serving(),
        ):
            self.assertFalse(loaded._package.installed_entries_dropped())
            for x, want in zip(inputs, expected):
                self.assertEqual(loaded(x), want)

    def test_reset_invalidates_a_loaded_artifact_loudly(self):
        # torch._dynamo.reset() clears the precompile entries install() loaded
        # while leaving the installed globals in place; neither a silent
        # recompile nor a RecompileError says the artifact is gone. The probe
        # asks for entries in THIS package's region, so a second artifact
        # loaded onto the same function cannot answer for the first.
        x = torch.randn(4, 8)
        self._capture(staged_with_graph_breaks, x).save(
            self.path(), require_no_dropped_guards=False
        )

        def assert_gone(loaded):
            self.assertTrue(loaded._package.installed_entries_dropped())
            for ctx in (contextlib.nullcontext(), serving()):
                with (
                    self.assertRaisesRegex(PackageError, r"torch\._dynamo\.reset\(\)"),
                    ctx,
                    torch.no_grad(),
                ):
                    loaded(x)

        torch._dynamo.reset()
        first = self._load(staged_with_graph_breaks)
        second = None
        try:
            with serving(), torch.no_grad():
                first(x)
            torch._dynamo.reset()
            assert_gone(first)
            second = self._load(staged_with_graph_breaks)
            assert_gone(first)
            self.assertFalse(second._package.installed_entries_dropped())
            with serving(), torch.no_grad():
                second(x)
        finally:
            if second is not None:
                second.unload()
            first.unload()

    def test_two_artifacts_sharing_an_inner_frame_both_serve(self):
        # The shared entry frame has no scale guard; scale is checked only by
        # its resume. Region-scoped dispatch must keep each entry paired with
        # the continuation from the same artifact instead of taking the first
        # globally matching entry.
        shared = self._import_module(
            self._write_module("shared_frame", "shared_frame", _SHARED_FRAME_SRC),
            "shared_frame",
        )
        x = torch.ones(3, 4)
        paths = []
        for cls, scale in ((shared.ModelOne, 3.0), (shared.ModelTwo, 7.0)):
            torch._dynamo.reset()
            session = precompile_capture(
                cls(scale),
                backend="eager",
                dynamic=False,
            )
            with session as compiled, torch.no_grad():
                compiled(x)
            self.assertTrue(session.summary().complete)
            self.assertEqual(session.summary().risky_dropped_guards, ())
            path = self.path(f"shared_{cls.__name__}.pt")
            session.save(path, require_no_dropped_guards=False)
            paths.append(path)

        torch._dynamo.reset()
        model_a, model_b = shared.ModelOne(3.0), shared.ModelTwo(7.0)
        with torch.no_grad():
            want_a, want_b = model_a(x), model_b(x)
        with (
            self._load(model_a, paths[0]) as a,
            self._load(model_b, paths[1]) as b,
            torch.no_grad(),
            serving(),
        ):
            self.assertEqual(a(x), want_a)
            self.assertEqual(b(x), want_b)

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

    @parametrize("victim", sorted(_FOREIGN_CALLABLES))
    def test_load_rejects_a_callable_the_artifact_was_not_captured_from(self, victim):
        callable_, message = _FOREIGN_CALLABLES[victim]
        self._capture(staged_with_graph_breaks, torch.randn(4, 8)).save(self.path())

        torch._dynamo.reset()
        with self.assertRaisesRegex(PackageError, message):
            self._load(callable_)

    @parametrize("damage", sorted(_DAMAGED_ARTIFACTS))
    def test_load_rejects_a_damaged_artifact(self, damage):
        # Capture on one machine, serve on another: the version check is only
        # worth anything if precompile_load runs it, and an artifact carrying
        # no code entries has to be named as such rather than falling through
        # to the unpack in CompilePackage.initialize.
        damage_entry, error, message = _DAMAGED_ARTIFACTS[damage]
        if damage_entry is None:
            os.makedirs(self.path(), exist_ok=True)
        else:
            self._capture(staged_with_graph_breaks, torch.randn(4, 8)).save(self.path())
            with open(self.path(), "rb") as f:
                entry = pickle.load(f)
            damage_entry(entry)
            with open(self.path(), "wb") as f:
                pickle.dump(entry, f)

        torch._dynamo.reset()
        with self.assertRaisesRegex(error, message):
            self._load(staged_with_graph_breaks)

    def test_load_rejects_a_same_named_callable_from_another_module(self):
        # Two modules each defining `class Encoder` agree on qualname and on
        # co_name, and each is self-consistent so the source checksum passes.
        a = self._import_module(
            self._write_module("a", "enc_a", _ENCODER_SRC.format(op="+")), "enc_a"
        )
        b = self._import_module(
            self._write_module("b", "enc_b", _ENCODER_SRC.format(op="*")), "enc_b"
        )
        self._capture(a.Encoder(), torch.randn(4)).save(self.path())

        torch._dynamo.reset()
        with self.assertRaisesRegex(PackageError, "defined in 'enc_b'"):
            self._load(b.Encoder())

    def test_load_rejects_a_different_definition_of_the_same_name(self):
        # The deployment shape is capture here, serve there. A module that
        # defines the same class twice under an `if` gives the two machines
        # matching module, qualname and co_name, and the source checksum covers
        # a line range that is identical in both builds, so only the first line
        # of the definition separates them.
        pkg_dir = self._write_module("flip", "flip_mod", _FLIPPED_ENCODER_SRC)
        os.environ.pop("FLIP_V2", None)
        self.addCleanup(lambda: os.environ.pop("FLIP_V2", None))
        v1 = self._import_module(pkg_dir, "flip_mod")
        x = torch.ones(4) * 2.0
        session = precompile_capture(v1.Encoder(), backend="eager", dynamic=False)
        with session as compiled, torch.no_grad():
            self.assertEqual(compiled(x), x + 1.0)
        session.save(self.path())

        os.environ["FLIP_V2"] = "1"
        sys.modules.pop("flip_mod", None)
        v2 = importlib.import_module("flip_mod")
        model = v2.Encoder()
        self.assertEqual(model(x), x * 7.0)
        torch._dynamo.reset()
        with self.assertRaisesRegex(PackageError, "different definition"):
            self._load(model)

    def test_load_survives_a_checkout_at_a_different_path(self):
        # The capture and serving machines check out to different absolute
        # paths, so co_filename must NOT be part of artifact identity. This
        # test exists to fail if someone tightens the check with it.
        src = _ENCODER_SRC.format(op="+")
        a = self._import_module(
            self._write_module("here", "moved_mod", src), "moved_mod"
        )
        x = torch.ones(4)
        self._capture(a.Encoder(), x).save(self.path())

        other = self._write_module("there", "moved_mod", src)
        sys.path.remove(os.path.dirname(a.__file__))
        b = self._import_module(other, "moved_mod")
        self.assertNotEqual(a.__file__, b.__file__)
        torch._dynamo.reset()
        with (
            self._load(b.Encoder()) as loaded,
            torch.no_grad(),
        ):
            self.assertEqual(loaded(x), x + 1.0)

    def test_inlining_through_a_re_exporting_shim_round_trips(self):
        # End to end because the two halves fail at different times: hashing
        # the inlined line range against the shim raises "Source mismatch"
        # during capture, and recording the shim's name rather than the key
        # raises "Source code changes detected" at load, against source that
        # nothing changed.
        pkg = self._write_module("shim", "_shim_impl", _SHIM_IMPL_SRC)
        self._write_module("shim", "shim_abc", _SHIM_SRC)
        self._write_module("shim", "shim_model", _SHIM_MODEL_SRC)
        self._forget_modules("_shim_impl", "shim_abc")
        model = self._import_module(pkg, "shim_model").Model()
        x = torch.ones(4)
        expected = model(x)

        session = precompile_capture(model, backend="eager", dynamic=False)
        with session as compiled, torch.no_grad():
            compiled(x)
        sources = session._package.cache_entry().source_info.inlined_sources
        recorded = {s.module for s in sources}
        self.assertIn("_shim_impl", recorded)
        self.assertNotIn("shim_abc", recorded)
        session.save(self.path())

        torch._dynamo.reset()
        with self._load(model) as loaded:
            self.assertEqual(loaded(x), expected)

    def test_an_acknowledged_risky_drop_serves_the_captured_dispatch(self):
        # What the refusal buys, spelled out: which module `impl` holds is an
        # env var's choice, so capture and serve disagree while every other
        # rail passes -- the artifact records the capture-time module as its
        # inlined source and that module is untouched on the serving box, so
        # the checksum revalidates. Opt out and the serving machine runs the
        # other backend eagerly and the captured one under the artifact.
        os.environ.pop("CORPUS_B", None)
        self.addCleanup(lambda: os.environ.pop("CORPUS_B", None))
        x = torch.ones(4)
        session = self._capture(self._corpus_model("calias"), x)
        session.save(self.path(), require_no_risky_drops=False)

        os.environ["CORPUS_B"] = "1"
        flipped = self._corpus_model("calias")
        self.assertEqual(flipped(x), x * 7.0)
        torch._dynamo.reset()
        with (
            self._load(flipped) as l,
            serving(),
            torch.no_grad(),
        ):
            self.assertEqual(l(x), x + 1.0)

    def test_wrapped_entry_uses_its_defining_module_globals(self):
        pkg_dir = self._write_module(
            "wrapped_entry", "precompile_entry_decorators", _DECORATOR_SRC
        )
        self._write_module(
            "wrapped_entry", "precompile_wrapped_entry", _WRAPPED_ENTRY_SRC
        )
        self._forget_modules("precompile_entry_decorators", "precompile_wrapped_entry")
        mod = self._import_module(pkg_dir, "precompile_wrapped_entry")
        x = torch.arange(4.0)
        expected = mod.staged(x)
        session = precompile_capture(mod.staged, backend="eager", dynamic=False)
        with session as compiled, torch.no_grad():
            self.assertEqual(compiled(x), expected)
        self.assertEqual(session.summary().dropped_guards, ())
        session.save(self.path())

        torch._dynamo.reset()
        loaded = self._load(mod.staged)
        with loaded, torch.no_grad(), serving():
            self.assertEqual(loaded(x), expected)

    def test_unload_clears_resume_function_entries(self):
        # uninstall() used to clear precompile entries only for the entry frame,
        # leaving resume functions installed on module-level code objects for
        # the rest of the process.
        x = torch.randn(4, 8)
        self._capture(staged_with_graph_breaks, x, no_grad=False).save(self.path())

        torch._dynamo.reset()
        loaded = self._load(staged_with_graph_breaks)
        installed = [
            code
            for code in loaded._package._installed_precompile_codes
            if code.co_name.startswith("torch_dynamo_resume_in")
        ]
        self.assertTrue(installed, "expected resume frames to be installed")
        self.assertTrue(all(_debug_get_precompile_entries(c) for c in installed))

        loaded.unload()
        for code in installed:
            self.assertEqual(_debug_get_precompile_entries(code), [])

    @torch._dynamo.config.patch(accumulated_recompile_limit=4)
    def test_unload_clears_fallback_entries_on_the_live_frames(self):
        # Uncovered calls compile into the region on the LIVE code, a
        # DIFFERENT object from the reconstructed twin the package holds even
        # though the two compare equal, and a fallback compile mints its OWN
        # resume code object too. Unload has to clear both by identity or every
        # load/serve/unload cycle leaks an entry toward accumulated_recompile_limit.
        from torch._dynamo.eval_frame import _get_total_cache_entry_count
        from torch._dynamo.resume_execution import ContinueExecutionCache

        inner = PrecompileBlock.forward.__code__
        model = PrecompileStack(1, resume_work=True)
        self._capture(model, torch.randn(4, 8), no_grad=False).save(
            self.path(), require_complete=False
        )

        torch._dynamo.reset()
        for n in range(5, 10):
            loaded = self._load(model)
            loaded(torch.randn(n, 8))
            self.assertGreater(_get_total_cache_entry_count(inner), 0)
            self.assertTrue(
                any(code is inner for code in loaded._package.region_codes())
            )
            loaded.unload()
            self.assertEqual(_get_total_cache_entry_count(inner), 0)

        resumes = list(ContinueExecutionCache.cache.get(inner, {}).values())
        self.assertTrue(resumes, "expected a live fallback resume frame")
        for code in resumes:
            self.assertEqual(_get_total_cache_entry_count(code), 0)

        counter = torch._dynamo.testing.CompileCounter()
        x = torch.randn(11, 8)
        ordinary = torch.compile(model, backend=counter, dynamic=False)
        self.assertEqual(ordinary(x), model(x))
        self.assertEqual(counter.frame_count, 2)

    def test_unload_leaves_another_packages_region_entries_on_a_shared_frame(self):
        # The clear above reaches the LIVE code object, which every package
        # loaded for another instance of the same class installed onto too.
        # Only this package's region may go: the other one is still dispatching
        # out of its own bucket on that same code, and a fallback it already
        # compiled is what keeps its next call from recompiling under serving().
        from torch._dynamo.eval_frame import _get_cache_entries_for_region

        covered, uncovered = torch.randn(4, 8), torch.randn(5, 8)
        session = precompile_capture(
            PrecompileStack(1, resume_work=True), backend="eager", dynamic=False
        )
        with session as compiled:
            compiled(covered)
        session.save(self.path(), require_complete=False)

        torch._dynamo.reset()
        inner = PrecompileBlock.forward.__code__
        first = self._load(PrecompileStack(1, resume_work=True))
        second = self._load(PrecompileStack(1, resume_work=True))
        try:
            first(uncovered)
            second(uncovered)
            region = second._isolate_recompiles_id
            self.assertTrue(_get_cache_entries_for_region(inner, region))
            first.unload()
            self.assertFalse(
                _get_cache_entries_for_region(inner, first._isolate_recompiles_id)
            )
            self.assertTrue(_get_cache_entries_for_region(inner, region))
            with serving():
                second(uncovered)
        finally:
            second.unload()
            first.unload()

    def test_a_served_call_allows_empty_graphs_only_for_its_own_frames(self):
        # The package's callback compiles an uncovered no-op branch as a guarded
        # variant instead of Dynamo's eager-only SkipFrame. An unrelated
        # torch.compile'd function running eagerly inside the served call
        # compiles on its own callback and must keep the SkipFrame.
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list

        def unrelated(n):
            return n + 1

        compiled_unrelated = torch.compile(unrelated, backend="eager")

        @torch._dynamo.disable
        def eager_helper(n):
            return compiled_unrelated(n)

        class Model(torch.nn.Module):
            def forward(self, x):
                return (x + 1) * eager_helper(2)

        model = Model()
        x = torch.randn(4)
        self._capture(model, x, no_grad=False).save(
            self.path(), require_no_dropped_guards=False
        )

        torch._dynamo.reset()
        with self._load(model) as loaded:
            self.assertEqual(loaded(x), model(x))
            self.assertEqual(len(_debug_get_cache_entry_list(unrelated.__code__)), 0)

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

    def test_concurrent_calls_do_not_deadlock_on_the_cache_lock(self):
        # Stress test: lookup() holds the ExtraState cache lock across guard
        # evaluation, which can drop the GIL, so the lock must release the GIL
        # before it waits or a second thread wedges the first. A deadlock here
        # shows up as the harness timeout.
        x = torch.randn(3, 4)
        model = PrecompileSelfAct(torch.relu)
        self._capture(model, x).save(self.path(), require_no_risky_drops=False)

        torch._dynamo.reset()
        loaded = self._load(model)
        errors = queue.SimpleQueue()

        def hammer():
            try:
                with torch.no_grad():
                    for _ in range(200):
                        loaded(x)
            except BaseException as e:
                errors.put(e)

        threads = [threading.Thread(target=hammer, daemon=True) for _ in range(4)]
        self.addCleanup(sys.setswitchinterval, sys.getswitchinterval())
        sys.setswitchinterval(1e-6)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        raised = []
        while not errors.empty():
            raised.append(errors.get_nowait())
        self.assertEqual(raised, [])
        loaded.unload()

    @parametrize("interference", sorted(_SKIP_INTERFERENCE))
    def test_unload_restores_the_skip_strategy_a_legacy_frame_had(self, interference):
        from torch._C._dynamo.eval_frame import (
            get_code_exec_strategy,
            get_code_region_exec_strategy,
        )

        before_load, after_load, expected = _SKIP_INTERFERENCE[interference]
        self._save_legacy_empty_graph_package()

        torch._dynamo.reset()
        code = PrecompileEmptyGraph.forward.__code__
        if before_load is not None:
            before_load(code)
        loaded = precompile_load(PrecompileEmptyGraph(), self.path(), backend="eager")
        self.assertIn(code, loaded._package._region_skipped_codes)
        self.assertEqual(
            get_code_exec_strategy(code).cur_action,
            FrameAction.SKIP if before_load is not None else FrameAction.DEFAULT,
        )
        region = get_code_region_exec_strategy(code, loaded._isolate_recompiles_id)
        self.assertEqual(region.cur_action, FrameAction.SKIP)
        if after_load is not None:
            after_load(code)

        loaded.unload()
        self.assertEqual(loaded._package._region_skipped_codes, [])
        restored = get_code_exec_strategy(code)
        self.assertEqual((restored.cur_action, restored.recursive_action), expected)

    @parametrize("first_is", ("unloaded", "garbage"))
    def test_a_region_skip_outlives_the_other_loads_of_its_frame(self, first_is):
        # Legacy packages can contain a frame with no guarded codes. Each
        # loaded callable gets an independent region-local skip, whether the
        # other load was unloaded or merely abandoned to the collector.
        from torch._C._dynamo.eval_frame import get_code_region_exec_strategy

        self._save_legacy_empty_graph_package()

        def load():
            return precompile_load(PrecompileEmptyGraph(), self.path(), backend="eager")

        torch._dynamo.reset()
        code = PrecompileEmptyGraph.forward.__code__
        first = load()
        if first_is == "garbage":
            package_ref = weakref.ref(first._package)
            del first
            gc.collect()
            self.assertIsNone(package_ref())
            torch._dynamo.reset()
            second = load()
        else:
            second = load()
            first.unload()

        def region_action():
            return get_code_region_exec_strategy(
                code, second._isolate_recompiles_id
            ).cur_action

        self.assertEqual(region_action(), FrameAction.SKIP)
        second.unload()
        self.assertEqual(region_action(), FrameAction.DEFAULT)

    def test_concurrent_compile_does_not_change_region_skip(self):
        from torch._C._dynamo.eval_frame import (
            get_code_exec_strategy,
            get_code_region_exec_strategy,
        )

        self._save_legacy_empty_graph_package()
        torch._dynamo.reset()

        entered = threading.Event()
        release = threading.Event()
        errors = []

        def backend(gm, example_inputs):
            entered.set()
            if not release.wait(20):
                raise AssertionError("timed out waiting to release backend")
            return gm.forward

        model = PrecompileEmptyGraph()
        compiled = torch.compile(model, backend=backend, dynamic=False)

        def run_compile():
            try:
                compiled(torch.randn(3, 4))
            except BaseException as e:
                errors.append(e)

        code = PrecompileEmptyGraph.forward.__code__

        def region_action():
            return get_code_region_exec_strategy(
                code, loaded._isolate_recompiles_id
            ).cur_action

        thread = threading.Thread(target=run_compile)
        thread.start()
        try:
            self.assertTrue(entered.wait(20))
            loaded = precompile_load(model, self.path(), backend="eager")
            self.assertEqual(region_action(), FrameAction.SKIP)
        finally:
            release.set()
        thread.join(20)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(region_action(), FrameAction.SKIP)
        strategy = get_code_exec_strategy(code)

        loaded.unload()
        after_unload = get_code_exec_strategy(code)
        self.assertEqual(after_unload.cur_action, strategy.cur_action)
        self.assertEqual(after_unload.recursive_action, strategy.recursive_action)

    @parametrize("pair", sorted(_COLLIDING_RESUME_PAIRS))
    def test_resume_names_from_other_captures_do_not_collide(self, pair):
        # Two artifacts captured in different processes carry the same
        # __resume_at_<offset>_<n> name, and a serving process installs both
        # into one module dict. Without a rename the loser of that write
        # silently runs the winner's continuation -- no error, right shape,
        # wrong numbers.
        build, save_kwargs = _COLLIDING_RESUME_PAIRS[pair]
        fns = build()
        x = torch.randn(3, 4)
        expected = [fn(x) for fn in fns]
        self.assertNotEqual(expected[0], expected[1])

        paths = [self.path(f"{i}.pt") for i in range(2)]
        for fn, path in zip(fns, paths):
            torch._dynamo.reset()
            self._capture(fn, x, no_grad=False).save(path, **save_kwargs)

        names = [_resume_names_in(path) for path in paths]
        self.assertEqual(len(names[0]), 1)
        _rename_resume_function(paths[1], names[1][0], names[0][0])
        self.assertEqual(_resume_names_in(paths[1]), names[0])

        torch._dynamo.reset()
        with contextlib.ExitStack() as stack:
            loaded = [
                stack.enter_context(self._load(fn, path))
                for fn, path in zip(fns, paths)
            ]
            # Both continuations have to be reachable at once. Pre-fix the
            # second load overwrote the first's binding and both callables
            # returned the second artifact's answer.
            stack.enter_context(serving())
            for call, want in zip(loaded, expected):
                self.assertEqual(call(x), want)

    def test_two_loads_of_one_artifact_both_serve(self):
        # A serving process holds one loaded package per model instance, so the
        # same artifact is routinely loaded twice, and both loads install a
        # resume function under a name derived from the resume code.
        x = torch.randn(3, 4)
        expected = staged_break_then_add_one(x)
        self._capture(staged_break_then_add_one, x, no_grad=False).save(self.path())

        torch._dynamo.reset()
        scope = staged_break_then_add_one.__globals__
        before = set(_names(scope, "__resume_at"))
        first, second = (self._load(staged_break_then_add_one) for _ in range(2))
        with first, second, serving():
            self.assertEqual(first(x), expected)
            self.assertEqual(second(x), expected)
            # Backstop on the mechanism: one installed name per load, not one
            # name both loads fought over.
            self.assertEqual(len(set(_names(scope, "__resume_at")) - before), 2)

    def test_unload_keeps_globals_a_bystander_compile_needs(self):
        # A serving process shares a module namespace with plain torch.compile.
        # Import aliases are minted from the module name, so both writers pick
        # the same one; popping it on unload takes it out from under whatever
        # else in the module resolved it.
        self._capture(staged_with_graph_breaks, torch.randn(3, 4)).save(self.path())

        torch._dynamo.reset()
        scope = staged_with_graph_breaks.__globals__
        # Other tests in this file compiled functions from this module and left
        # their aliases here. Drop them for the state a serving process starts
        # in, where the load is what installs them.
        for name in _names(scope, "__import_"):
            del scope[name]

        x = torch.randn(3, 4)
        with torch.no_grad():
            expected = staged_with_global_function_ref(x)
            loaded = self._load(staged_with_graph_breaks)
            bystander = torch.compile(
                staged_with_global_function_ref, backend="eager", dynamic=False
            )
            self.assertEqual(bystander(x), expected)
            loaded.unload()
            self.assertEqual(bystander(x), expected)

    def test_a_second_load_keeps_the_builtins_key_the_first_installed(self):
        # Two loads of one artifact -- the replica shape -- record the same
        # builtins-dict name, and only the first finds it unbound. Unless the
        # second joins the owner set, the first unload deletes a global the
        # second's bytecode reads and every later call is a NameError.
        mod = self._import_module(
            self._write_module(
                "builtin_break", "builtin_break", _BUILTIN_ACROSS_BREAK_SRC
            ),
            "builtin_break",
        )
        x, cfg = torch.ones(3, 4), {"a": 1, "b": 2}
        self._capture(mod.Model(2.0), x, cfg).save(
            self.path(), require_no_risky_drops=False
        )

        torch._dynamo.reset()
        # A serving process that never compiled this module holds no builtins
        # key, so the load is what creates it. Capturing in this process bound
        # one here, so drop it to get to that state.
        scope = mod.__dict__
        for name in _names(scope, "__builtins_dict__"):
            del scope[name]

        model_a, model_b = mod.Model(2.0), mod.Model(2.0)
        with torch.no_grad():
            expected = model_b(x, cfg)
        first = self._load(model_a)
        self.addCleanup(first.unload)
        second = self._load(model_b)
        self.addCleanup(second.unload)
        installed = _names(scope, "__builtins_dict__")
        self.assertTrue(installed)

        first.unload()
        with torch.no_grad(), serving():
            self.assertEqual(second(x, cfg), expected)
        self.assertEqual(_names(scope, "__builtins_dict__"), installed)

    def test_unload_keeps_a_builtins_key_a_plain_compile_minted(self):
        # The counter naming the builtins dict is per process, so a serving
        # process that compiles before it loads can mint exactly the name the
        # artifact recorded. Nothing displaced that binding, so the load must
        # leave it alone: claiming it makes this unload delete the key the
        # plain compile's own bytecode still reads.
        import torch._dynamo.bytecode_transformation as bytecode_transformation

        mod = self._import_module(
            self._write_module(
                "builtin_break_local", "builtin_break_local", _BUILTIN_ACROSS_BREAK_SRC
            ),
            "builtin_break_local",
        )
        x, cfg = torch.ones(3, 4), {"a": 1, "b": 2}
        scope = mod.__dict__
        with mock.patch.object(
            bytecode_transformation, "_unique_id_counter", itertools.count()
        ):
            self._capture(mod.Model(2.0), x, cfg).save(
                self.path(), require_no_risky_drops=False
            )

            torch._dynamo.reset()
            for name in _names(scope, "__builtins_dict__", "__resume_at"):
                CleanupHook.disown(scope, name)
                del scope[name]

            # Same fresh counter the capture ran on, so the plain compile mints
            # the names the artifact recorded.
            bytecode_transformation._unique_id_counter = itertools.count()
            bystander = torch.compile(mod.Model(2.0), backend="eager", dynamic=False)
            with torch.no_grad():
                expected = bystander(x, cfg)
            minted = _names(scope, "__builtins_dict__")
            self.assertTrue(minted)

            loaded = self._load(mod.Model(2.0))
            self.assertEqual(
                _names(scope, "__builtins_dict__"),
                minted,
                "the load minted a name of its own, so nothing collided",
            )
            loaded.unload()
            with torch.no_grad(), torch.compiler.set_stance("fail_on_recompile"):
                self.assertEqual(bystander(x, cfg), expected)
            self.assertEqual(_names(scope, "__builtins_dict__"), minted)

    def test_loaded_builtins_names_do_not_collide_with_local_compiles(self):
        # The other order: the load installs the names the artifact recorded,
        # and a plain compile that follows mints from the same per-process
        # counter, so it has to skip past them rather than rebind them.
        import torch._dynamo.bytecode_transformation as bytecode_transformation

        scope = staged_with_graph_breaks.__globals__

        def forget_minted_names():
            for name in _names(scope, "__builtins_dict__", "__resume_at"):
                CleanupHook.disown(scope, name)
                del scope[name]

        with mock.patch.object(
            bytecode_transformation, "_unique_id_counter", itertools.count()
        ):
            forget_minted_names()
            self._capture(staged_with_graph_breaks, torch.randn(3, 4)).save(self.path())

            torch._dynamo.reset()
            forget_minted_names()
            bytecode_transformation._unique_id_counter = itertools.count()
            self.addCleanup(torch._dynamo.reset)
            with self._load(staged_with_graph_breaks):
                for i in range(10):
                    name = f"_precompile_bystander_{i}"
                    self.addCleanup(scope.pop, name, None)
                    exec(
                        compile(
                            f"def {name}(x):\n    return x + {i}\n", __file__, "exec"
                        ),
                        scope,
                    )
                    x = torch.ones(2)
                    later = torch.compile(scope[name], backend="eager", dynamic=False)
                    self.assertEqual(later(x), x + i)

    @parametrize("uncovered_call", (False, True))
    def test_unload_removes_the_globals_a_load_added(self, uncovered_call):
        # A call the artifact does not cover falls back to an ordinary Dynamo
        # compile inside the loaded region, and that compile installs globals of
        # its own that the package never sees -- unclaimed, they stay in the
        # served module for the life of the process, four more on every load.
        model = PrecompileBreakOnlyWhenFalse()
        x = torch.randn(3, 4)
        with torch.no_grad():
            expected = [model(x, flag) for flag in (True, False)]
        self._capture(model, x, True).save(self.path())

        torch._dynamo.reset()
        scope = PrecompileBreakOnlyWhenFalse.forward.__globals__
        # A serving process that never compiled this module holds no
        # builtins-dict key, so install() is the one that creates it. Capturing
        # in this same process leaves one behind, so drop it to get there.
        for name in _names(scope, "__builtins_dict__"):
            del scope[name]
        before = set(scope)

        for _ in range(2):
            loaded = self._load(model)
            with torch.no_grad(), serving():
                self.assertEqual(loaded(x, True), expected[0])
            self.assertTrue(_names(scope, "__builtins_dict__"))
            if uncovered_call:
                covered = set(scope)
                with torch.no_grad():
                    self.assertEqual(loaded(x, False), expected[1])
                # The uncovered branch breaks at a line the artifact never
                # captured, so the fallback compile has to mint globals of its own.
                self.assertTrue(set(scope) - covered)
            loaded.unload()
        # Import aliases are the one thing install() is expected to leave:
        # plain torch.compile installs them permanently too.
        leftover = [n for n in set(scope) - before if not n.startswith("__import_")]
        self.assertEqual(sorted(leftover), [])

    def test_save_writes_a_single_file(self):
        # save() names a FILE, written exactly as given with parent directories
        # created, and precompile_load takes that same path back. A parent that
        # is a file has to arrive as a PackageError naming the path the caller
        # passed, and a refused write must leave nothing behind.
        session = precompile_capture(
            staged_with_graph_breaks, backend="eager", dynamic=False
        )
        with session as compiled:
            compiled(torch.randn(4, 8))
        tmp = os.path.join(self.dir(), "store")
        path = os.path.join(tmp, "nested", "model.pt")
        session.save(path)
        self.assertTrue(os.path.isfile(path))
        self.assertFalse(os.path.isdir(path))

        as_dir = os.path.join(tmp, "adir")
        os.makedirs(as_dir)
        with self.assertRaisesRegex(PackageError, "single files"):
            session.save(as_dir)
        parent = os.path.join(tmp, "plain_file")
        with open(parent, "w"):
            pass
        target = os.path.join(parent, "model.pt")
        with self.assertRaisesRegex(PackageError, re.escape(target)):
            session.save(target)
        self.assertEqual(sorted(os.listdir(tmp)), ["adir", "nested", "plain_file"])

        torch._dynamo.reset()
        with self._load(staged_with_graph_breaks, path) as loaded:
            self.assertEqual(loaded(torch.randn(4, 8)).shape, torch.Size([]))

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

    def test_unload_leaves_a_global_a_later_package_rebound(self):
        # A serving process loads one artifact per model instance, so two
        # packages can install the same names -- the backend uuid and the
        # resume-function name both come from the artifact, not the loader.
        # Popping them on the first unload leaves the second package's entry
        # frame reaching for a continuation that is gone.
        torch._dynamo.reset()
        session = precompile_capture(
            PrecompileSelfAct(torch.relu), backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        # self.act is a dispatch slot; this test is about unloading, not it.
        session.save(self.path(), require_no_risky_drops=False)

        def load():
            return self._load(PrecompileSelfAct(torch.relu))

        torch._dynamo.reset()
        scope = PrecompileSelfAct.forward.__globals__
        before = set(scope)
        first = load()
        # Ask the package what it installed rather than diffing the namespace:
        # a diff is a GC-timing observation, and on 3.14t it dropped the one
        # name that IS rebound.
        installed = {
            g.name: g.value
            for entries in first._package._installed_globals.values()
            for g in entries
        }
        second = load()
        rebound = {
            name: scope[name]
            for name, value in installed.items()
            if scope.get(name) is not value
        }
        # Two separate preconditions, asserted separately: an empty `rebound` can
        # mean the first load installed nothing OR that the second load produced
        # objects identical to the first, and those are different bugs.
        self.assertTrue(installed, "the first load installed no globals")
        self.assertTrue(rebound, f"the second load rebound none of {sorted(installed)}")

        first.unload()
        for name, value in rebound.items():
            self.assertIs(scope.get(name), value)
        second.unload()
        # And unloading in load order still has to leave the namespace clean:
        # the values `first` installed were orphaned the moment `second`
        # rebound them, so putting them back would leak a compiled backend per load.
        leftover = sorted(
            name
            for name in set(scope) - before
            if not name.startswith("__import_") and name in installed
        )
        self.assertEqual(leftover, [])

    def test_code_under_a_descriptor_resolves(self):
        # getattr on the CLASS returns the descriptor, not the function inside
        # it, so a frame defined under @property had no resolvable name and the
        # whole frame silently fell back to eager. Real capture of a large model
        # lost four frames to one such property.
        import ast as _ast

        holder = _PackageDescriptorHolder
        nested = next(
            c
            for c in holder.getter_only.fget.__code__.co_consts
            if isinstance(c, types.CodeType)
        )
        cases = {
            "getter": holder.getter_only.fget.__code__,
            "nested in getter": nested,
            "cached_property": holder.cached.func.__code__,
            "setter": holder.pair.fset.__code__,
        }
        for label, code in cases.items():
            qualname, source = dynamo_package._get_code_source(code)
            # Replay exactly what the loader does, so the path round-trips
            # rather than merely being produced.
            obj = sys.modules[holder.__module__]
            for part in qualname.split("."):
                obj = getattr(obj, part)
            for part in source.split("."):
                if not part:
                    continue
                if part.endswith("]"):
                    at = part.rfind("[")
                    obj = getattr(obj, part[:at])[_ast.literal_eval(part[at + 1 : -1])]
                else:
                    obj = getattr(obj, part)
            self.assertIs(obj, code, f"{label}: {qualname!r} + {source!r}")

    def test_stale_module_memo_is_revalidated(self):
        # The filename -> module memo caches a hit outright, and its ABA check
        # on len(sys.modules) cannot see a plain delete. Handing back the dead
        # name made add_code raise KeyError on it.
        name = "_precompile_stale_memo_probe"
        source = "def probe(t):\n    return t + 1\n"
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, name + ".py")
            with open(path, "w") as f:
                f.write(source)
            sys.path.insert(0, d)
            try:
                module = importlib.import_module(name)
                scan = dynamo_package._scan_sys_modules_for_file
                self.assertEqual(scan(module.__file__), name)
                del sys.modules[name]
                self.assertIsNone(scan(module.__file__))
                # No KeyError: an unresolvable module contributes no source.
                info = dynamo_package.SourceInfo(inlined_sources=set())
                info.add_code(module.probe.__code__)
            finally:
                sys.path.remove(d)
                sys.modules.pop(name, None)

    def test_risky_drop_warning_leads_with_the_shape_bearing_ones(self):
        # A flat cut is dominated by whichever guard type is most numerous. On a
        # real capture that meant one SEQUENCE_LENGTH and three CONSTANT_MATCHes
        # -- the only drops that can change what shape the graph computes --
        # were invisible behind 392 CLOSURE_MATCHes on function identities.
        risky = (
            [("CLOSURE_MATCH", f"c{i}") for i in range(392)]
            + [("ID_MATCH", f"i{i}") for i in range(81)]
            + [("SEQUENCE_LENGTH", "impl.__defaults__")]
            + [("CONSTANT_MATCH", "impl.__defaults__[4]")]
            + [("TYPE_MATCH", "L['pg'].group_name")]
        )
        with self.assertLogs("torch._dynamo.precompile_package", "WARNING") as cm:
            dynamo_package_lint._warn_risky_drops(risky)
        message = "\n".join(cm.output)

        self.assertIn("476 dropped guard(s)", message)
        self.assertIn("COULD BEAR ON SHAPE (3)", message)
        # Each shape-bearing name survives the truncation, and each is named
        # ahead of the bulk that used to crowd it out.
        for name in ("impl.__defaults__", "impl.__defaults__[4]", "group_name"):
            self.assertIn(name, message)
            self.assertLess(message.index(name), message.index("CLOSURE_MATCH"))
        self.assertIn("CLOSURE_MATCH x392", message)

    def test_unserializable_backend_says_so_rather_than_reporting_a_gap(self):
        # A session takes any backend Dynamo resolves -- tests register their
        # own -- so the refusal belongs at artifact(), not at construction. But
        # the generic "recorded 0 of N compiled backend(s)" reads as a defect in
        # the model, when the real answer is that this backend records nothing.
        # aot_eager is the one people reach for, since it is how you isolate
        # AOTAutograd.
        session = precompile_capture(
            _precompile_scale, backend="aot_eager", dynamic=False
        )
        with session as compiled:
            compiled(torch.randn(2))
        with self.assertRaisesRegex(Exception, "does not produce anything") as ctx:
            session.artifact(require_complete=False, require_no_risky_drops=False)
        message = str(ctx.exception)
        self.assertIn("aot_eager", message)
        self.assertIn("'inductor' or 'eager'", message)
        # The grad-mode and backward advice the generic message carries would be
        # a wrong lead here.
        self.assertNotIn("training=True", message)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
