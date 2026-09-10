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
import sys
import sysconfig
import tempfile
import textwrap
import types
from unittest import mock

import torch
import torch._dynamo.package as dynamo_package
import torch._dynamo.precompile_package as dynamo_package_lint
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.nn.functional as F
from torch._dynamo.exc import PackageError
from torch._dynamo.package import (
    _defining_module_name,
    CompilePackage,
    DynamoCache,
    SystemInfo,
)
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.types import FrameAction, FrameExecStrategy
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
    graph.call_function(torch.ones, ((2,),), {"device": torch.device(device)})
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
        "reports no CPU codegen target",
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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
