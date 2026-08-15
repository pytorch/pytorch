# Owner(s): ["module: dynamo"]

import builtins
import contextlib
import dataclasses
import functools
import gc
import importlib
import inspect
import math
import math as _precompile_stdlib_alias
import os
import pickle
import queue
import re
import sys
import tempfile
import threading
import unittest
from unittest import mock

import torch
import torch._dynamo.package as dynamo_package
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.nn.functional as F
import torch.onnx.operators
import torch.utils.cpp_extension
from torch._dynamo.exc import PackageError
from torch._dynamo.package import (
    _defining_module_name,
    CompilePackage,
    DiskDynamoStore,
    DynamoCache,
    SystemInfo,
)
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.precompile_package import (
    _dynamo_alias_module,
    _fact_order,
    _GuardFact,
    _SingleFileStore,
    precompile_capture,
    precompile_load,
    serving,
)
from torch._dynamo.testing import reduce_to_scalar_loss
from torch._dynamo.utils import CleanupManager, counters
from torch._functorch import config as functorch_config
from torch._inductor.runtime.runtime_utils import cache_dir
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
    TEST_WITH_TORCHDYNAMO,
)
from torch.testing._internal.inductor_utils import (
    HAS_CUDA_AND_TRITON,
    HAS_XPU_AND_TRITON,
)


def staged_with_graph_breaks(x):
    x = x * 2
    torch._dynamo.graph_break()
    x = x + 3
    torch._dynamo.graph_break()
    return x.sum()


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


class PrecompileBlock(torch.nn.Module):
    def __init__(self, i):
        super().__init__()
        self.i = i

    def forward(self, x):
        x = x * 2 + self.i
        torch._dynamo.graph_break()
        return x


class PrecompileStack(torch.nn.Module):
    """All blocks share one forward code object, so variants pile onto it."""

    def __init__(self, n):
        super().__init__()
        self.blocks = torch.nn.ModuleList([PrecompileBlock(i) for i in range(n)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x.sum()


class PrecompileResumingBlock(torch.nn.Module):
    """Like PrecompileBlock, but the frame resumed after the break compiles too."""

    def __init__(self, i):
        super().__init__()
        self.i = i

    def forward(self, x):
        x = x * 2 + self.i
        torch._dynamo.graph_break()
        return x + 1.0


class PrecompileResumingStack(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [PrecompileResumingBlock(i) for i in range(n)]
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


def staged_with_a_registry_keyed_by_a_builtin_name(x, cfg):
    n = _PRECOMPILE_OPS["len"](cfg)
    torch._dynamo.graph_break()
    return x.sum() * n


def _precompile_user_act(t):
    return -t


def _precompile_scale(t):
    return t * 2


def _precompile_closure_over(fn):
    def inner(x):
        return fn(x).sum()

    return inner


class _PrecompileRegistry:
    def __init__(self, act):
        self.act = act


PRECOMPILE_DISPATCH = {"act": _precompile_user_act}
PRECOMPILE_REGISTRY = _PrecompileRegistry(_precompile_user_act)


class PrecompileNoDispatchSlot(torch.nn.Module):
    """Ordinary code with no swappable slot: a torch op, stdlib, a local def."""

    def forward(self, x):
        y = torch.relu(_precompile_scale(x)) * math.sqrt(2.0)
        torch._dynamo.graph_break()
        return (y + 1).sum()


PRECOMPILE_INV_CONFIG = {"mode": "sum"}


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


class PrecompileInvariantModel(torch.nn.Module):
    """Reads a global across a graph break, so the resume frame guards it."""

    def forward(self, x):
        y = x * 2
        torch._dynamo.graph_break()
        if PRECOMPILE_INV_CONFIG["mode"] == "sum":
            return y.sum()
        return y.mean()


class PrecompileBuiltinReadingModel(torch.nn.Module):
    """Iterates a ModuleList, which reads ``iter`` off Dynamo's builtins dict."""

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(2)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x.sum()


class PrecompileSharedBlock(torch.nn.Module):
    """A library block reused by two different models; its frame graph-breaks."""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        y = x * 2
        torch._dynamo.graph_break()
        return y * self.scale


class PrecompileSharedUserA(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.block = PrecompileSharedBlock(scale)

    def forward(self, x):
        return self.block(x).sum()


class PrecompileSharedUserB(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.block = PrecompileSharedBlock(scale)

    def forward(self, x):
        return self.block(x).sum() + 0.0


class PrecompileSelfAct(torch.nn.Module):
    """self.act = <callable> -- how configurable activations are usually written."""

    def __init__(self, act):
        super().__init__()
        self.act = act

    def forward(self, x):
        y = self.act(x)
        torch._dynamo.graph_break()
        return (y + 1).sum()


_SHARED_RACE_SRC = """\
import torch


class SharedBlock(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        y = x * 2
        torch._dynamo.graph_break()
        return y * self.scale


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


class PrecompileEmptyGraph(torch.nn.Module):
    """Dynamo traces this to an empty graph, so install() skip_code()s it."""

    def forward(self, x):
        return x


class PrecompileValuePinned(torch.nn.Module):
    def forward(self, x):
        scale = x.abs().max().item()
        y = x * 2 if scale > 0.5 else x * 3
        return y.sum()


class PrecompileItemThenBreak(torch.nn.Module):
    """.item() pins the stack slot AND the local the resume frame reads it from."""

    def forward(self, x):
        scale = x.abs().max().item()
        torch._dynamo.graph_break()
        return (x * 2 if scale > 0.5 else x * 3).sum()


class PrecompileIntArg(torch.nn.Module):
    def forward(self, x, k):
        y = x * k
        torch._dynamo.graph_break()
        return (y + k).sum()


class PrecompileKeysArg(torch.nn.Module):
    """A dict_keys argument is pinned by EQUALS_MATCH, not CONSTANT_MATCH."""

    def forward(self, x, ks):
        y = x * float(len(ks))
        torch._dynamo.graph_break()
        return (y + 1).sum()


class PrecompileIntAttr(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        y = x * self.k
        torch._dynamo.graph_break()
        return (y + self.k).sum()


def _precompile_break_then_cos(t):
    torch._dynamo.graph_break()
    return t.cos()


class PrecompileTensorAcrossBreak(torch.nn.Module):
    """x.sin() sits on the stack across the break, so it gets a ___stackN name."""

    def forward(self, x):
        return (x.sin() + _precompile_break_then_cos(x)).sum()


class PrecompileConfigConstants(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = torch.nn.LayerNorm(8)
        self.drop = torch.nn.Dropout(0.1)
        self.lin = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.drop(self.lin(self.ln(x))).relu().sum()


class PrecompileAliasedImport(torch.nn.Module):
    """import torch.nn.functional as F -- the global name is not the module's."""

    def forward(self, x):
        return F.gelu(x).sum()


class PrecompileFullyQualified(torch.nn.Module):
    """torch.nn.functional.gelu spelled out -- the namespace is two hops down."""

    def forward(self, x):
        return torch.nn.functional.gelu(x).sum()


class PrecompileFunctionScopedImport(torch.nn.Module):
    """A function-scoped `import torch`, which transformers is full of. Dynamo
    reaches it through the `__import_torch` alias and guards the attributes read
    off that alias, never the alias itself."""

    def forward(self, x):
        import torch

        if isinstance(x, torch.Tensor):
            return torch.relu(x).sum()
        return x


class PrecompileFunctionScopedUserImport(torch.nn.Module):
    """The same shape for a user module, which clears on the other arm of the
    namespace rule: the namespace owns the def rather than torch owning the
    namespace."""

    def forward(self, x):
        import lazy_helper

        return lazy_helper.op(x).sum()


class PrecompileModuleInAttribute(torch.nn.Module):
    """self.ns = importlib.import_module(cfg.backend) -- a config-picked module."""

    def __init__(self, ns):
        super().__init__()
        self.ns = ns

    def forward(self, x):
        return self.ns.gelu(x).sum()


class PrecompileDictDispatch(torch.nn.Module):
    """CFG["act"] -- the same dispatch slot spelled as a dict lookup."""

    def forward(self, x):
        return PRECOMPILE_DISPATCH["act"](x).sum()


class PrecompileRegistryLookup(torch.nn.Module):
    """REGISTRY.act -- a dispatch table parked in a module-level object."""

    def forward(self, x):
        return PRECOMPILE_REGISTRY.act(x).sum()


class PrecompileFunctionArg(torch.nn.Module):
    def forward(self, x, fn):
        return fn(x).sum()


class PrecompileFunctionDefault(torch.nn.Module):
    def forward(self, x, fn=_precompile_user_act):
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


class PrecompileAliasedStdlibImport(torch.nn.Module):
    """import math as <alias> -- a stdlib module under a name it does not own."""

    def forward(self, x):
        return (x * _precompile_stdlib_alias.sqrt(2.0)).sum()


class PrecompileStdlibAttribute(torch.nn.Module):
    def forward(self, x):
        return (x * math.sqrt(2.0)).sum()


class PrecompileSameModuleHelper(torch.nn.Module):
    def forward(self, x):
        return _precompile_scale(x).sum()


class PrecompilePartialForward(torch.nn.Module):
    """self.forward = functools.partial(...) shadows the class method."""

    def __init__(self, scale):
        super().__init__()
        self.forward = functools.partial(self._impl, scale)

    def _impl(self, scale, x):
        return (x * scale).sum()


class _PrecompileForwardsCode:
    """A wrapper forwarding __code__ with a safe default -- the usual decorator
    spelling -- around a builtin, so the attribute is present and None."""

    def __init__(self, fn):
        self.__code__ = getattr(fn, "__code__", None)
        self._fn = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


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


_PLUGGABLE_OP_SRC = """\
def op(x):
    return x * {f}
"""

_ALIAS_DISPATCH_SRC = """\
import os

import torch


if os.environ.get("PICK_B") == "1":
    import pick_b as impl
else:
    import pick_a as impl


class Model(torch.nn.Module):
    def forward(self, x):
        return impl.op(x)
"""

_FROM_IMPORT_DISPATCH_SRC = """\
import os

import torch


if os.environ.get("PICK_B") == "1":
    from pick_b import op
else:
    from pick_a import op


class Model(torch.nn.Module):
    def forward(self, x):
        return op(x)
"""

_OWN_HELPERS_SRC = """\
import torch.nn.functional as F


def _scale(x):
    return x * 2.0


def call(x):
    return F.relu(_scale(x))
"""

_OWN_MODEL_SRC = """\
import own_helpers
import torch
from torch.nn.functional import gelu


class Model(torch.nn.Module):
    def forward(self, x):
        return gelu(own_helpers.call(x))
"""

_OWN_SUB_SRC = """\
def op(x):
    return x * 3.0
"""

_OWN_PARENT_SRC = """\
import own_sub
"""

_OWN_NESTED_MODEL_SRC = """\
import own_parent
import torch


class Model(torch.nn.Module):
    def forward(self, x):
        return own_parent.own_sub.op(x)
"""

_LIBRARY_SPELLINGS_SRC = """\
import collections.abc as _abc
import math as _math
import torch
from math import sqrt
from torch import relu


class Model(torch.nn.Module):
    def forward(self, x):
        n = float(isinstance([], _abc.Sized))
        return (relu(x) * sqrt(2.0) * _math.fabs(-1.0) * n).sum()
"""

_LAZY_HELPER_SRC = """\
def op(x):
    return x * 3.0
"""

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

_SHIM_MODEL_SRC = """\
import shim_abc
import torch


class Model(torch.nn.Module):
    def forward(self, x):
        return shim_abc.helper(x).sum()
"""


def _precompile_sin(t):
    return t.sin()


PRECOMPILE_ACTIVATION = _precompile_sin


def staged_with_global_function_ref(x):
    y = PRECOMPILE_ACTIVATION(x) + 1
    torch._dynamo.graph_break()
    return (y * 10).sum()


@contextlib.contextmanager
def _precompile_mode(mode):
    old = PRECOMPILE_CONFIG["mode"]
    PRECOMPILE_CONFIG["mode"] = mode
    try:
        yield
    finally:
        PRECOMPILE_CONFIG["mode"] = old


def compute_loss_helper(x):
    return reduce_to_scalar_loss(x)


def compiled_region_with_backend_id_for_package_test():
    return __compiled_fn_0_00000000_0000_0000_0000_000000000000()  # noqa: F821


# The modules the corpus needs on disk, because a dispatch read off another
# module cannot be spelled inside this file. Written under one directory so
# they import each other by plain name, exactly as a user package would.
_CORPUS_MODULES = {
    "cimpl_a": "def op(x):\n    return x + 1.0\n",
    "cimpl_b": "def op(x):\n    return x * 7.0\n",
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
    "cpkg/cimpl": "def op(x):\n    return x + 1.0\n",
    "cpkg/__init__": "from . import cimpl as impl\n\n\nop = impl.op\n",
    "cmodels": """\
import os

import torch

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
        return cown.call(x)
""",
}

_CORPUS_X = torch.randn(4, 8)
_CORPUS_SEQ = torch.randn(2, 4, 8)

# Every shape below was once a silent wrong answer on a serving machine that
# somebody had to find by hand. _is_risky_drop regressed three review rounds
# running -- each fix closed the previous round's false negative and opened a
# new one -- so the shapes live in a table rather than in prose: adding one is
# a single entry, and nothing here may ever stop being flagged.
_RISKY_DROP_CORPUS = {
    "aliased_module_import": (
        "G['impl'].op",
        lambda t: (t._corpus_model("calias"), (_CORPUS_X,)),
    ),
    "attribute_builtin_fn": (
        "self.act",
        lambda t: (PrecompileSelfAct(abs), (_CORPUS_X,)),
    ),
    "attribute_torch_fn": (
        "self.act",
        lambda t: (PrecompileSelfAct(F.gelu), (_CORPUS_X,)),
    ),
    "attribute_user_fn": (
        "self.act",
        lambda t: (PrecompileSelfAct(_precompile_user_act), (_CORPUS_X,)),
    ),
    "bare_global_fn": (
        "G['PRECOMPILE_ACTIVATION']",
        lambda t: (staged_with_global_function_ref, (_CORPUS_X,)),
    ),
    "closure_cell": (
        "fn",
        lambda t: (_precompile_closure_over(_precompile_user_act), (_CORPUS_X,)),
    ),
    "cross_module_from_import": (
        "G['op']",
        lambda t: (t._corpus_model("cfrom"), (_CORPUS_X,)),
    ),
    "dict_lookup": (
        "G['PRECOMPILE_DISPATCH']['act']",
        lambda t: (PrecompileDictDispatch(), (_CORPUS_X,)),
    ),
    "dispatch_in_inlined_helper": (
        "G['__import_chelpers'].op",
        lambda t: (t._corpus("InlinedHelper"), (_CORPUS_X,)),
    ),
    "function_argument": (
        "fn",
        lambda t: (PrecompileFunctionArg(), (_CORPUS_X, _precompile_user_act)),
    ),
    "function_default_arg": (
        "fn",
        lambda t: (PrecompileFunctionDefault(), (_CORPUS_X,)),
    ),
    "module_in_attribute": (
        "self.ns.gelu",
        lambda t: (PrecompileModuleInAttribute(F), (_CORPUS_X,)),
    ),
    "module_level_global": (
        "G['cconf'].ACT",
        lambda t: (t._corpus("ModuleAttr"), (_CORPUS_X,)),
    ),
    "module_valued_global": (
        "G['OPS'].op",
        lambda t: (t._corpus("ModuleSwitch"), (_CORPUS_X,)),
    ),
    "object_attribute": (
        "G['PRECOMPILE_REGISTRY'].act",
        lambda t: (PrecompileRegistryLookup(), (_CORPUS_X,)),
    ),
    "package_reexport": (
        "G['cpkg'].op",
        lambda t: (t._corpus("PackageReexport"), (_CORPUS_X,)),
    ),
    "package_submodule_alias": (
        "G['cpkg'].impl.op",
        lambda t: (t._corpus("PackageSubmoduleAlias"), (_CORPUS_X,)),
    ),
    "sibling_module": (
        "G['cdispatch'].op",
        lambda t: (t._corpus("SiblingModule"), (_CORPUS_X,)),
    ),
    "stock_layer_activation": (
        "self._modules['enc'].activation",
        lambda t: (PrecompileStockEncoderLayer(), (_CORPUS_SEQ,)),
    ),
}

# The other half of the corpus, and the half that keeps the report worth
# reading: the lint only warns by default, so if ordinary code trips it the
# warning is noise nobody audits and nobody ever opts into enforcement. Each of
# these pairs with a positive above.
_BENIGN_DROP_CORPUS = {
    "aliased_stdlib_import": lambda t: (PrecompileAliasedStdlibImport(), (_CORPUS_X,)),
    "aliased_torch_import": lambda t: (PrecompileAliasedImport(), (_CORPUS_X,)),
    "direct_functional_call": lambda t: (PrecompileFullyQualified(), (_CORPUS_X,)),
    "own_name_def_in_own_module": lambda t: (t._corpus("OwnNameDef"), (_CORPUS_X,)),
    "real_submodule": lambda t: (t._corpus("RealSubmodule"), (_CORPUS_X,)),
    "same_module_helper": lambda t: (PrecompileSameModuleHelper(), (_CORPUS_X,)),
    "stdlib_attribute": lambda t: (PrecompileStdlibAttribute(), (_CORPUS_X,)),
    "stock_linear_layernorm": lambda t: (PrecompileConfigConstants(), (_CORPUS_X,)),
    "stock_sequential": lambda t: (PrecompileStockSequential(), (_CORPUS_X,)),
}


@functorch_config.patch("bundled_autograd_cache", True)
@torch._dynamo.config.patch({"strict_precompile": True})
@instantiate_parametrized_tests
class TestPackage(torch._inductor.test_case.TestCase):
    def path(self):
        path = os.path.join(cache_dir(), f"package_{self.id()}")
        os.makedirs(path, exist_ok=True)
        return path

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        DynamoCache.clear()
        PrecompileContext.clear()

    def _save_and_reload(self, expected_backends, expected_dynamo):
        """
        Serializes all artifacts, clears all caches, then reloads the serialized artifact
        Simulates a new process.

        Args:
            expected_backends: Expected number of precompile_aot_autograd_artifacts
            expected_dynamo: Expected number of precompile_dynamo_artifacts
        """
        debug_info = PrecompileContext.save_to_dynamo_cache()
        self.assertEqual(len(debug_info["dynamo"]), expected_dynamo)
        self.assertEqual(len(debug_info["backends"]), expected_backends)
        torch._dynamo.reset()
        PrecompileContext.clear()

    def test_guarded_code_records_backend_ids_from_bytecode(self):
        def fn(x):
            return x + 1

        (backend_id,) = (
            compiled_region_with_backend_id_for_package_test.__code__.co_names
        )
        package = CompilePackage(fn)
        with package.code_context(fn.__code__):
            package.add_guarded_code(
                b"", compiled_region_with_backend_id_for_package_test.__code__
            )

        cache_entry = package.cache_entry()
        self.assertEqual(cache_entry.codes[0].backend_ids, [backend_id])

    @unittest.expectedFailure  # FUNCTION_MATCH guard not serializable today
    def test_nn_module(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10, device="cuda")

            def forward(self, x):
                return self.linear(x)

        fn = MyModule()
        package = CompilePackage(fn.forward)
        compiled_fn = torch._dynamo.optimize("inductor", package=package)(fn)
        x = torch.randn(10, 10, device="cuda")
        compiled_fn(x)

    @parametrize("backend", ("eager", "inductor"))
    @parametrize("device", ("cpu", "cuda", "xpu"))
    def test_basic_fn(self, backend, device):
        if device == "cuda" and not HAS_CUDA_AND_TRITON:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU_AND_TRITON:
            raise unittest.SkipTest("Requires XPU/Triton")

        ctx = DiskDynamoStore()

        def fn(x):
            return x + 1

        args = (
            torch.randn(
                3,
                2,
                device=device,
            ),
        )

        # Saving
        package = CompilePackage(fn)
        compiled_fn = torch._dynamo.optimize(backend, package=package)(fn)
        expected = compiled_fn(*args)
        if backend == "eager":
            for backend_id, backend in package.cached_backends.items():
                ctx.record_eager_backend(backend_id, backend)

        ctx.save_package(package, self.path())
        # Loading
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with self.assertRaisesRegex(
                RuntimeError,
                "Detected recompile when torch.compile stance is 'fail_on_recompile'",
            ):
                compiled_fn(*args)

            package, backends = ctx.load_package(fn, self.path())
            compiled_fn = torch._dynamo.optimize(package=package)(fn)
            package.install(backends)
            self.assertEqual(expected, compiled_fn(*args))

    @parametrize("backend", ("eager", "inductor"))
    @parametrize("device", ("cpu", "cuda", "xpu"))
    def test_lazy_backward(self, backend, device):
        if device == "cuda" and not HAS_CUDA_AND_TRITON:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU_AND_TRITON:
            raise unittest.SkipTest("Requires XPU/Triton")

        ctx = DiskDynamoStore()

        def fn(x):
            return x.sin() + x.cos()

        args = (
            torch.zeros(
                3,
                2,
                device=device,
                requires_grad=True,
            ),
        )

        # Saving
        package = CompilePackage(fn)
        compiled_fn = torch._dynamo.optimize(backend, package=package)(fn)
        expected = compiled_fn(*args)
        expected.sum().backward()

        if backend == "eager":
            for backend_id, backend in package.cached_backends.items():
                ctx.record_eager_backend(backend_id, backend)

        ctx.save_package(package, self.path())
        # Loading
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with self.assertRaisesRegex(
                RuntimeError,
                "Detected recompile when torch.compile stance is 'fail_on_recompile'",
            ):
                compiled_fn(*args)

            package, backends = ctx.load_package(fn, self.path())
            compiled_fn = torch._dynamo.optimize(package=package)(fn)
            package.install(backends)
            self.assertEqual(expected, compiled_fn(*args))

    @parametrize("backend", ("eager", "inductor"))
    @parametrize("device", ("cpu", "cuda", "xpu"))
    def test_graph_break_bomb(self, backend, device):
        if device == "cuda" and not HAS_CUDA_AND_TRITON:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU_AND_TRITON:
            raise unittest.SkipTest("Requires XPU/Triton")

        ctx = DiskDynamoStore()

        def fn(x, l, r):
            if l > r:
                return x.sum()
            mid = (l + r) // 2
            if x.sum() == mid:
                return x.sum()
            elif x.sum() < mid:
                return fn(x, l, mid)
            else:
                return fn(x, mid + 1, r)

        def guard_filter_fn(guards):
            return [
                guard.guard_type not in ("CLOSURE_MATCH", "FUNCTION_MATCH")
                for guard in guards
            ]

        # Saving
        package = CompilePackage(fn)
        compiled_fn = torch._dynamo.optimize(
            backend=backend, package=package, guard_filter_fn=guard_filter_fn
        )(fn)
        N = 10
        args_list = [(torch.tensor(x, device=device), 0, N - 1) for x in range(N)]
        for args in args_list:
            compiled_fn(*args)
        if backend == "eager":
            for backend_id, backend in package.cached_backends.items():
                ctx.record_eager_backend(backend_id, backend)
        ctx.save_package(package, self.path())

        # Loading
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            for args in args_list:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Detected recompile when torch.compile stance is 'fail_on_recompile'",
                ):
                    compiled_fn(*args)
            package, backends = ctx.load_package(fn, self.path())
            compiled_fn = torch._dynamo.optimize(
                backend="eager", package=package, guard_filter_fn=guard_filter_fn
            )(fn)
            package.install(backends)
            for args in args_list:
                self.assertEqual(compiled_fn(*args), args[0].sum())

            with self.assertRaisesRegex(
                RuntimeError,
                "Detected recompile when torch.compile stance is 'fail_on_recompile'",
            ):
                compiled_fn(torch.tensor(N), 0, N - 1)

    @parametrize("backend", ("eager", "inductor"))
    @parametrize("device", ("cpu", "cuda", "xpu"))
    def test_dynamic_shape(self, backend, device):
        if device == "cuda" and not HAS_CUDA_AND_TRITON:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU_AND_TRITON:
            raise unittest.SkipTest("Requires XPU/Triton")

        ctx = DiskDynamoStore()

        def fn(x):
            return x + x.shape[0]

        args = (torch.randn(3, 2, device=device),)
        args1 = (torch.randn(5, 2, device=device),)
        args2 = (torch.randn(7, 2, device=device),)
        expected1 = fn(*args1)

        torch._dynamo.mark_dynamic(args[0], 0, min=3, max=5)

        # Saving
        package = CompilePackage(fn)
        compiled_fn = torch._dynamo.optimize(backend=backend, package=package)(fn)
        compiled_fn(*args)
        if backend == "eager":
            for backend_id, backend in package.cached_backends.items():
                ctx.record_eager_backend(backend_id, backend)
        ctx.save_package(package, self.path())

        # Loading
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with self.assertRaisesRegex(
                RuntimeError,
                "Detected recompile when torch.compile stance is 'fail_on_recompile'",
            ):
                compiled_fn(*args1)

            package, backends = ctx.load_package(fn, self.path())
            compiled_fn = torch._dynamo.optimize(package=package)(fn)
            package.install(backends)

            self.assertEqual(expected1, compiled_fn(*args1))

            with self.assertRaisesRegex(
                RuntimeError,
                "Detected recompile when torch.compile stance is 'fail_on_recompile'",
            ):
                compiled_fn(*args2)

    def test_install_survives_stale_cleanup_hooks(self):
        # The first compile installs its generated functions -- and, on every
        # compile, a builtins-dict global (see install_builtins_dict_in_fglobals)
        # -- into the module globals behind a CleanupHook keyed on the generated
        # code object. install() rebinds __compiled_fn/__resume_at names to fresh
        # values, but leaves the builtins-dict binding alone when it's already
        # correct, since it's the same dict object on every compile in this
        # module. Either way, a hook firing afterwards must not delete the
        # binding install() is now responsible for.
        ctx = DiskDynamoStore()

        def fn(x):
            y = x + x.shape[0]
            if y.sum() > 0:  # data-dependent branch, forces a resume function
                return y * 2
            return y

        args = (torch.randn(3, 2),)
        expected = fn(*args)

        # Other tests in this file compile functions defined in this same
        # module, so ignore what they left behind in the shared globals, and
        # hold their code objects alive so ids stay unambiguous below.
        prefixes = ("__compiled_fn", "__resume_at", "__builtins_dict__")
        scope = fn.__globals__
        preexisting = {name for name in scope if name.startswith(prefixes)}
        # Plain loops with an explicit del, rather than a walrus in a list
        # comprehension: a walrus target leaks into this method's own frame,
        # which would pin the last code object seen and defeat the gc.collect()
        # below.
        others = []
        code = None
        for ref in list(CleanupManager.instance.refs.values()):
            code = ref()
            if code is not None:
                others.append(code)
        del code
        other_ids = {id(o) for o in others}

        package = CompilePackage(fn)
        compiled_fn = torch._dynamo.optimize(backend="eager", package=package)(fn)
        compiled_fn(*args)
        for backend_id, backend in package.cached_backends.items():
            ctx.record_eager_backend(backend_id, backend)
        ctx.save_package(package, self.path())

        # Whether the hooks fire before or after install() is left to the
        # garbage collector, so pin the code objects they are keyed on to pick
        # the losing order deterministically.
        pinned = []
        code = None
        for idx, ref in list(CleanupManager.instance.refs.items()):
            if idx in other_ids:
                continue
            code = ref()
            if code is not None:
                pinned.append(code)
        del code
        pinned_ids = {id(p) for p in pinned}
        self.assertTrue(pinned_ids)

        torch._dynamo.reset()
        package, backends = ctx.load_package(fn, self.path())
        compiled_fn = torch._dynamo.optimize(package=package)(fn)
        package.install(backends)

        installed = {name for name in scope if name.startswith(prefixes)} - preexisting
        self.assertTrue(installed)

        del pinned
        gc.collect()

        # Without this the assert below can pass without any hook ever running.
        self.assertTrue(pinned_ids - set(CleanupManager.instance.refs))
        self.assertEqual(installed - set(scope), set())
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(expected, compiled_fn(*args))

    def test_file_change(self):
        ctx = DiskDynamoStore()

        def import_from_path(module_name, file_path):
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        mock_module_add_original = """
def add(x, y):
    return x + y
"""

        mock_module_add_modified = """
def add(x, y):
    return x - y
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_module_add_original_path = os.path.join(
                tmp_dir, "mock_module_add_original.py"
            )
            mock_module_add_modified_path = os.path.join(
                tmp_dir, "mock_module_add_modified.py"
            )
            with open(mock_module_add_original_path, "w") as f:
                f.write(mock_module_add_original)
            with open(mock_module_add_modified_path, "w") as f:
                f.write(mock_module_add_modified)

            module = import_from_path(
                "torch.test_package_helper",
                mock_module_add_original_path,
            )

            def fn(x):
                return module.add(x, 1)

            args = (torch.randn(3, 2),)

            def guard_filter_fn(guards):
                return [
                    guard.guard_type
                    not in ("CLOSURE_MATCH", "FUNCTION_MATCH", "MODULE_MATCH")
                    for guard in guards
                ]

            # Saving
            package = CompilePackage(fn)
            compiled_fn = torch._dynamo.optimize(
                backend="eager", package=package, guard_filter_fn=guard_filter_fn
            )(fn)
            compiled_fn(*args)
            for backend_id, backend in package.cached_backends.items():
                ctx.record_eager_backend(backend_id, backend)
            ctx.save_package(package, self.path())

            module = import_from_path(
                "torch.test_package_helper",
                mock_module_add_modified_path,
            )
            with self.assertRaisesRegex(RuntimeError, "Source code changes detected"):
                ctx.load_package(fn, self.path())

            module = import_from_path(
                "torch.test_package_helper",
                mock_module_add_original_path,
            )
            ctx.load_package(fn, self.path())

    @parametrize("device", ("cpu", "cuda", "xpu"))
    def test_dynamo_cache_manual_load(self, device):
        if device == "cuda" and not HAS_CUDA_AND_TRITON:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU_AND_TRITON:
            raise unittest.SkipTest("Requires XPU/Triton")

        def fn(x):
            return x.sin() + x.cos()

        def fn2(x):
            return x.cos() + x

        package1 = CompilePackage(fn)
        package2 = CompilePackage(fn2)
        compiled_fn1 = torch._dynamo.optimize(backend="inductor", package=package1)(fn)
        compiled_fn2 = torch._dynamo.optimize(backend="inductor", package=package2)(fn2)
        arg1 = torch.randn(3, 2, device=device)
        arg2 = torch.randn(5, 2, device=device)
        expected = [compiled_fn1(arg1), compiled_fn2(arg2)]

        DynamoCache.save(package1)
        DynamoCache.save(package2)
        total_frames = torch._dynamo.convert_frame.FRAME_COUNTER
        self._save_and_reload(expected_backends=2, expected_dynamo=2)

        # These should exist because of populate_caches
        package1 = DynamoCache.load_and_install_package(fn)
        package2 = DynamoCache.load_and_install_package(fn2)

        with torch.compiler.set_stance("fail_on_recompile"):
            result1 = compiled_fn1(arg1)
            result2 = compiled_fn2(arg2)
            self.assertEqual(expected, [result1, result2])
        self.assertEqual(torch._dynamo.convert_frame.FRAME_COUNTER, total_frames)

    @parametrize("backend", ("eager", "inductor"))
    def test_reset_clears_installed_package(self, backend):
        # Regression test for https://github.com/pytorch/pytorch/issues/190664.
        # package.install() must register target_code in input_codes so that
        # torch._dynamo.reset() clears precompile entries on the installed code.
        from torch._C._dynamo.eval_frame import _debug_get_precompile_entries

        ctx = DiskDynamoStore()

        def fn(x):
            return x.sin() + x.cos()

        package = CompilePackage(fn)
        compiled_fn = torch._dynamo.optimize(backend=backend, package=package)(fn)
        compiled_fn(torch.randn(3, 2))
        if backend == "eager":
            for backend_id, bknd in package.cached_backends.items():
                ctx.record_eager_backend(backend_id, bknd)
        ctx.save_package(package, self.path())

        torch._dynamo.reset()
        package, backends = ctx.load_package(fn, self.path())
        package.install(backends)
        self.assertGreater(len(_debug_get_precompile_entries(fn.__code__)), 0)

        torch._dynamo.reset()
        self.assertEqual(len(_debug_get_precompile_entries(fn.__code__)), 0)

    @parametrize("device", ("cpu", "cuda", "xpu"))
    @torch._dynamo.config.patch(caching_precompile=True)
    def test_automatic_dynamo_serialize(self, device):
        if device == "cuda" and not HAS_CUDA_AND_TRITON:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU_AND_TRITON:
            raise unittest.SkipTest("Requires XPU/Triton")

        def fn(x):
            return x.sin() + x.cos()

        def fn2(x):
            return x.cos() + x

        arg1 = torch.randn(3, 2, device=device)
        arg2 = torch.randn(5, 2, device=device)
        expected = [fn(arg1), fn2(arg2)]
        compiled_fn1 = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        compiled_fn2 = torch.compile(fn2)  # noqa: UNSPECIFIED_BACKEND
        result = [compiled_fn1(arg1), compiled_fn2(arg2)]
        self.assertEqual(expected, result)
        DynamoCache.clear()
        total_frames = torch._dynamo.convert_frame.FRAME_COUNTER

        self._save_and_reload(expected_backends=2, expected_dynamo=2)

        compiled_fn1 = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        compiled_fn2 = torch.compile(fn2)  # noqa: UNSPECIFIED_BACKEND
        with torch.compiler.set_stance("fail_on_recompile"):
            result1 = compiled_fn1(arg1)
            result2 = compiled_fn2(arg2)
            self.assertEqual(expected, [result1, result2])
        self.assertEqual(torch._dynamo.convert_frame.FRAME_COUNTER, total_frames)

    def test_import_source_unpickle_without_trace(self):
        # Deserializing an ImportSource happens at torch.compile() time with no
        # active TracingContext (e.g. precompile warm-load). Reconstructing the
        # source must not install a guard (which would require a tracing
        # context), so the round-trip must not raise.
        import pickle

        from torch._dynamo.source import ImportSource

        source = ImportSource("torch")
        reloaded = pickle.loads(pickle.dumps(source))
        self.assertEqual(reloaded, source)

    @parametrize("device", ("cpu", "cuda", "xpu"))
    @torch._dynamo.config.patch(caching_precompile=True)
    def test_automatic_dynamo_import_source_guard(self, device):
        # Warm-loading a guard state whose serialized sources include an
        # ImportSource must not raise. `pytree.tree_is_leaf` routes through
        # `get_pytree_SUPPORTED_NODES_source`, which builds an
        # `ImportSource("torch")` that ends up in the serialized guard state.
        if device == "cuda" and not HAS_CUDA_AND_TRITON:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU_AND_TRITON:
            raise unittest.SkipTest("Requires XPU/Triton")

        def fn(x):
            if torch.utils._pytree.tree_is_leaf(x):
                return torch.nn.functional.relu(x) + x.sin()
            return x

        arg = torch.randn(3, 2, device=device)
        expected = fn(arg)
        compiled_fn = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(compiled_fn(arg), expected)
        total_frames = torch._dynamo.convert_frame.FRAME_COUNTER

        self._save_and_reload(expected_backends=1, expected_dynamo=1)

        compiled_fn = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        with torch.compiler.set_stance("fail_on_recompile"):
            result = compiled_fn(arg)
            self.assertEqual(result, expected)
        self.assertEqual(torch._dynamo.convert_frame.FRAME_COUNTER, total_frames)

    @parametrize("device", ("cpu", "cuda", "xpu"))
    @torch._dynamo.config.patch(caching_precompile=True)
    def test_automatic_dynamo_recompiles(self, device):
        if device == "cuda" and not HAS_CUDA_AND_TRITON:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU_AND_TRITON:
            raise unittest.SkipTest("Requires XPU/Triton")

        def fn(x):
            return x.sin() + x.cos()

        arg1 = torch.randn(3, 2, device=device)
        arg2 = torch.randn(5, 2, device=device)
        compiled_fn = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        expected1 = compiled_fn(arg1)

        # Should cause a recompile
        expected2 = compiled_fn(arg2)
        total_frames = torch._dynamo.convert_frame.FRAME_COUNTER

        self._save_and_reload(expected_backends=2, expected_dynamo=1)

        compiled_fn = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        with torch.compiler.set_stance("fail_on_recompile"):
            result1 = compiled_fn(arg1)
            result2 = compiled_fn(arg2)
            # Because of automatic dynamic, a third random shape should also not cause a recompile
            arg3 = torch.randn(7, 2, device=device)
            compiled_fn(arg3)
        self.assertEqual(result1, expected1)
        self.assertEqual(result2, expected2)
        self.assertEqual(torch._dynamo.convert_frame.FRAME_COUNTER, total_frames)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO or IS_LINUX,
        "https://github.com/pytorch/pytorch/issues/183810",
    )
    @parametrize("device", ("cpu", "cuda", "xpu"))
    @torch._dynamo.config.patch(caching_precompile=True)
    def test_automatic_dynamo_graph_breaks(self, device):
        if device == "cuda" and not HAS_CUDA_AND_TRITON:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU_AND_TRITON:
            raise unittest.SkipTest("Requires XPU/Triton")

        def fn(x, l, r):
            if l > r:
                return x.sum()
            mid = (l + r) // 2
            if x.sum() == mid:
                return x.sum()
            elif x.sum() < mid:
                return fn(x, l, mid)
            else:
                return fn(x, mid + 1, r)

        def guard_filter_fn(guards):
            return [
                guard.guard_type not in ("CLOSURE_MATCH", "FUNCTION_MATCH")
                for guard in guards
            ]

        # Saving
        compiled_fn = torch._dynamo.optimize(
            backend="inductor", guard_filter_fn=guard_filter_fn
        )(fn)
        N = 10
        args_list = [(torch.tensor(x, device=device), 0, N - 1) for x in range(N)]
        for args in args_list:
            compiled_fn(*args)

        total_frames = torch._dynamo.convert_frame.FRAME_COUNTER
        self._save_and_reload(expected_backends=9, expected_dynamo=1)

        compiled_fn = torch._dynamo.optimize(
            backend="inductor", guard_filter_fn=guard_filter_fn
        )(fn)
        with torch.compiler.set_stance("fail_on_recompile"):
            for args in args_list:
                self.assertEqual(compiled_fn(*args), args[0].sum())
            # Should have same number of frames as on cold start
            self.assertEqual(torch._dynamo.convert_frame.FRAME_COUNTER, total_frames)

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/184832")
    @parametrize("device", ("cpu", "cuda", "xpu"))
    @torch._dynamo.config.patch(caching_precompile=True)
    def test_automatic_dynamo_lazy_backward(self, device):
        if device == "cuda" and not HAS_CUDA_AND_TRITON:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU_AND_TRITON:
            raise unittest.SkipTest("Requires XPU/Triton")

        def fn(x):
            return x.sin() + x.cos()

        arg1 = torch.randn(3, 2, device=device, requires_grad=True)
        arg2 = arg1.clone().detach_().requires_grad_(True)

        compiled_fn = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        expected1 = compiled_fn(arg1)
        expected1.sum().backward()
        total_frames = torch._dynamo.convert_frame.FRAME_COUNTER

        self._save_and_reload(expected_backends=1, expected_dynamo=1)

        compiled_fn = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        # Run it again, no recompile needed
        with torch.compiler.set_stance("fail_on_recompile"):
            expected2 = compiled_fn(arg2)
            expected2.sum().backward()

        self.assertEqual(torch._dynamo.convert_frame.FRAME_COUNTER, total_frames)

    @parametrize("device", ("cpu", "cuda", "xpu"))
    @torch._dynamo.config.patch(caching_precompile=True)
    def test_graph_break_partial_backend(self, device):
        if device == "cuda" and not HAS_CUDA_AND_TRITON:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU_AND_TRITON:
            raise unittest.SkipTest("Requires XPU/Triton")

        def fn(x):
            y = x.sin()
            torch._dynamo.graph_break()
            return x.sin() + y

        arg1 = torch.randn(3, 2, device=device, requires_grad=True)
        arg2 = arg1.clone().detach_().requires_grad_(True)
        compiled_fn = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        expected1 = compiled_fn(arg1)
        expected1.sum().backward()
        total_frames = torch._dynamo.convert_frame.FRAME_COUNTER

        # Remove backends related to resume functions
        dynamo_entry = next(iter(PrecompileContext._dynamo_cache_entries.values()))
        for code in dynamo_entry.codes:
            module = sys.modules[code.python_module]
            if code.install_to_global:
                # Clear the fn_names from global scope, to simulate a new environment
                for fn_name in code.function_names:
                    module.__dict__.pop(fn_name)
            for fn_name in code.function_names:
                if "resume" in fn_name:
                    self.assertEqual(len(code.backend_ids), 1)
                    # delete the fn from the global scope to simulate a new
                    backend = code.backend_ids[0]
                    # Delete the backend associated with the resume function
                    del PrecompileContext._backend_artifacts_by_key[backend]

        self._save_and_reload(expected_backends=1, expected_dynamo=1)

        compiled_fn = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        # Run it again. There will be a recompile because one of the backends is deleted, but it should
        # still work.
        expected2 = compiled_fn(arg2)
        expected2.sum().backward()
        self.assertEqual(expected1, expected2)
        # One recompile on a new frame, so total_frames should increase by 1
        self.assertEqual(torch._dynamo.convert_frame.FRAME_COUNTER, total_frames + 1)

    @parametrize("device", ("cpu", "cuda", "xpu"))
    @torch._dynamo.config.patch(caching_precompile=True)
    def test_call_function_from_resume(self, device):
        if device == "cuda" and not HAS_CUDA_AND_TRITON:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU_AND_TRITON:
            raise unittest.SkipTest("Requires XPU/Triton")
        mod = torch.nn.Linear(2, 3, device=device)

        def foo(x, mod):
            pred = mod(x)
            compute_loss_helper(pred).backward()
            return None

        args = (torch.randn(3, 2, device=device), mod)
        compiled_fn = torch.compile(foo)  # noqa: UNSPECIFIED_BACKEND
        compiled_fn(*args)
        total_frames = torch._dynamo.convert_frame.FRAME_COUNTER

        self._save_and_reload(expected_backends=1, expected_dynamo=1)

        compiled_fn = torch.compile(foo)  # noqa: UNSPECIFIED_BACKEND
        # Run it again, no recompile needed
        with torch.compiler.set_stance("fail_on_recompile"):
            compiled_fn(*args)

        self.assertEqual(torch._dynamo.convert_frame.FRAME_COUNTER, total_frames)

    @parametrize("device", ("cpu", "cuda", "xpu"))
    @torch._dynamo.config.patch(caching_precompile=True)
    def test_code_with_generator(self, device):
        if device == "cuda" and not HAS_CUDA_AND_TRITON:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU_AND_TRITON:
            raise unittest.SkipTest("Requires XPU/Triton")

        def foo(set_of_x):
            if not all(isinstance(s, torch.Tensor) for s in set_of_x):
                raise TypeError(
                    f"Expected all elements of set_of_x to be tensors, got {set_of_x}"
                )

            return torch.cat(set_of_x, dim=0)

        args = ([torch.randn(3, 2, device=device) for _ in range(3)],)
        compiled_fn = torch.compile(foo)  # noqa: UNSPECIFIED_BACKEND
        compiled_fn(*args)
        self._save_and_reload(expected_backends=1, expected_dynamo=1)

    @parametrize("device", ("cpu", "cuda", "xpu"))
    @torch._dynamo.config.patch(caching_precompile=True)
    def test_automatic_dynamo_graph_breaks_from_print_model_as_fn(self, device):
        if device == "cuda" and not HAS_CUDA_AND_TRITON:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU_AND_TRITON:
            raise unittest.SkipTest("Requires XPU/Triton")

        def guard_filter_fn(guards):
            return [
                guard.guard_type not in ("CLOSURE_MATCH", "FUNCTION_MATCH")
                for guard in guards
            ]

        class TempNN(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = torch.nn.functional.relu(x)
                x *= x
                x /= 2
                print(x.sum().item())
                x += 1
                return x

        # Saving
        x = torch.rand(10, device=device)
        model = TempNN()
        model(x)
        compiled_fn = torch.compile(
            model,
            backend="inductor",
            options=dict(guard_filter_fn=guard_filter_fn),
        )

        compiled_fn(x)
        total_frames = torch._dynamo.convert_frame.FRAME_COUNTER
        self._save_and_reload(expected_backends=2, expected_dynamo=1)

        del compiled_fn

        with torch.compiler.set_stance("fail_on_recompile"):
            compiled_fn = torch.compile(
                model, backend="inductor", options=dict(guard_filter_fn=guard_filter_fn)
            )
            compiled_fn(x)
            self.assertEqual(torch._dynamo.convert_frame.FRAME_COUNTER, total_frames)

    class _tempTensorSamplerForQualName:
        def __init__(self, val, mask, prob):
            self.val = val
            self.mask = mask
            self.prob = prob

        @classmethod
        def class_method_that_is_used(cls, x):
            prob = torch.sigmoid(x)
            thresh = torch.rand(1, device=x.device)
            mask = (prob > thresh).to(torch.bool)
            return cls(x, mask, prob)

        @classmethod
        def class_method_that_is_not_used(cls, x):
            prob = torch.sigmoid(x)
            thresh = torch.rand(1, device=x.device)
            mask = (prob > thresh).to(torch.bool)
            return cls(x, mask, prob)

        def instance_method_that_is_used(self, x):
            return x / 2

    class _tempNetForQualName(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def instance_method_without_args(self):
            shape = [1, 2, 3, 4]
            x = torch.randn(shape)
            return x

        def instance_method_with_args(self, x):
            return x + 1

        def forward(self, x):
            x *= x
            with torch.device(x.device):
                y = self.instance_method_without_args()
            # test classmethod called from class
            sampler = (
                TestPackage._tempTensorSamplerForQualName.class_method_that_is_used(x)
            )
            x = torch.where(torch.rand_like(x) < sampler.prob, sampler.val, x) + y.sum()
            # test instance method called from instance
            x = sampler.instance_method_that_is_used(x)
            # test classmethod called from instance
            another_sampler = sampler.class_method_that_is_not_used(x)
            # test instance method called from instance
            x = another_sampler.instance_method_that_is_used(x)
            # test classmethod called from instance
            x += y.sum()
            x = self.instance_method_with_args(x)
            return x

    @parametrize("device", ("cpu", "cuda", "xpu"))
    @torch._dynamo.config.patch(caching_precompile=True)
    def test_classmethod_qualname(self, device):
        if device == "cuda" and not HAS_CUDA_AND_TRITON:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU_AND_TRITON:
            raise unittest.SkipTest("Requires XPU/Triton")

        x = torch.rand(10, device=device)
        model = TestPackage._tempNetForQualName()
        model.forward(x)
        compiled_fn = torch.compile(  # noqa: UNSPECIFIED_BACKEND
            model.forward,
            options=dict(guard_filter_fn=torch.compiler.skip_guard_on_globals_unsafe),
        )
        compiled_fn(x)


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

    @parametrize("backend", ("eager", "inductor"))
    def test_graph_breaks_and_recompiles_round_trip(self, backend):
        shapes = [(4, 8), (5, 8), (6, 8)]
        inputs = [torch.randn(*s) for s in shapes]
        expected = [staged_with_graph_breaks(x) for x in inputs]

        session = precompile_capture(
            staged_with_graph_breaks, backend=backend, dynamic=False
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
            precompile_load(
                staged_with_graph_breaks, self.path(), backend=backend, dynamic=False
            ) as loaded,
            serving(),
        ):
            for x, want in zip(inputs, expected):
                self.assertEqual(loaded(x), want)
            with self.assertRaisesRegex(RuntimeError, "fail_on_recompile"):
                loaded(torch.randn(9, 8))

    def test_save_refuses_incomplete_package(self):
        # 5 blocks x 3 shapes = 15 variants on one shared forward code object,
        # which overruns a recompile_limit of 8. Before, the truncated package
        # saved happily and only stopped matching at serving time.
        n = 5
        model = PrecompileStack(n)
        inputs = [torch.randn(*s) for s in [(4, 8), (5, 8), (6, 8)]]
        expected_first = model(inputs[0])

        session = precompile_capture(
            model, backend="eager", recompile_limit=8, dynamic=False
        )
        with session as compiled:
            for x in inputs:
                compiled(x)

        summary = session.summary()
        self.assertFalse(summary.complete)
        self.assertTrue(summary.truncated)
        self.assertEqual(summary.bypassed, ())
        with self.assertRaisesRegex(PackageError, "exceeded recompile_limit"):
            session.save(self.path())

        # Opting in to a partial artifact is still allowed, and the variants
        # that WERE captured must still serve -- truncation records a gap, it
        # does not throw away the coverage already obtained.
        self.assertGreater(summary.guarded_codes, 0)
        session.save(self.path(), require_complete=False)
        torch._dynamo.reset()
        with (
            precompile_load(
                model, self.path(), backend="eager", recompile_limit=8, dynamic=False
            ) as loaded,
            serving(),
        ):
            self.assertEqual(loaded(inputs[0]), expected_first)

    def test_global_dict_conditional_guard_round_trip(self):
        modes = ["sum", "mean"]
        x = torch.randn(4, 8)
        expected = {}
        for mode in modes:
            with _precompile_mode(mode):
                expected[mode] = staged_with_global_dict_conditional(x)
        self.assertNotEqual(expected["sum"].item(), expected["mean"].item())

        session = precompile_capture(
            staged_with_global_dict_conditional, backend="eager", dynamic=False
        )
        with session as compiled:
            for mode in modes:
                with _precompile_mode(mode):
                    compiled(x)
        summary = session.summary()
        # entry frame + one resume frame, each specialized per mode
        self.assertEqual(summary.frames, 2)
        self.assertEqual(summary.resume_functions, 1)
        self.assertEqual(summary.guarded_codes, 2 * len(modes))
        self.assertTrue(summary.complete)
        session.save(self.path())

        torch._dynamo.reset()
        with (
            precompile_load(
                staged_with_global_dict_conditional,
                self.path(),
                backend="eager",
                dynamic=False,
            ) as loaded,
            serving(),
        ):
            # The global guard must be load-bearing: flipping it has to select
            # the other graph rather than silently reusing the first.
            for mode in modes:
                with _precompile_mode(mode):
                    self.assertEqual(loaded(x), expected[mode])
            with _precompile_mode("uncaptured"):
                with self.assertRaisesRegex(RuntimeError, "fail_on_recompile"):
                    loaded(x)

    def test_local_dict_conditional_guard_round_trip(self):
        configs = [{"op": "sin", "scale": 2}, {"op": "cos", "scale": 5}]
        x = torch.randn(4, 8)
        expected = [staged_with_local_dict_conditional(x, c) for c in configs]
        self.assertNotEqual(expected[0].item(), expected[1].item())

        session = precompile_capture(
            staged_with_local_dict_conditional, backend="eager", dynamic=False
        )
        with session as compiled:
            for cfg in configs:
                compiled(x, cfg)
        summary = session.summary()
        self.assertEqual(summary.frames, 2)
        self.assertEqual(summary.resume_functions, 1)
        self.assertTrue(summary.complete)
        session.save(self.path())

        torch._dynamo.reset()
        with (
            precompile_load(
                staged_with_local_dict_conditional,
                self.path(),
                backend="eager",
                dynamic=False,
            ) as loaded,
            serving(),
        ):
            for cfg, want in zip(configs, expected):
                self.assertEqual(loaded(x, cfg), want)
            with self.assertRaisesRegex(RuntimeError, "fail_on_recompile"):
                loaded(x, {"op": "tan", "scale": 1})

    def test_nested_dict_guards_round_trip(self):
        configs = [
            {"alpha": {"kind": "wide", "dims": [1, 2], "weight": 3}},
            {"beta": {"kind": "narrow", "dims": [1], "weight": 7}},
        ]
        x = torch.randn(4, 8)
        expected = [staged_with_nested_dict_conditional(x, c) for c in configs]
        self.assertNotEqual(expected[0].item(), expected[1].item())

        session = precompile_capture(
            staged_with_nested_dict_conditional, backend="eager", dynamic=False
        )
        with session as compiled:
            for cfg in configs:
                compiled(x, cfg)
        summary = session.summary()
        self.assertTrue(summary.complete)
        # Assert positively that the key-set and membership guards were emitted
        # AND retained. Checking only that they are absent from dropped_guards
        # would pass just as well if Dynamo never emitted them at all.
        kept = summary.kept_guard_types()
        self.assertIn("DICT_KEYS_MATCH", kept)
        self.assertIn("DICT_CONTAINS", kept)
        self.assertNotIn("DICT_KEYS_MATCH", summary.dropped_guard_types())
        self.assertNotIn("DICT_CONTAINS", summary.dropped_guard_types())
        session.save(
            self.path(),
            require_no_risky_drops=False,
            require_no_dropped_guards=False,
        )

        torch._dynamo.reset()
        with (
            precompile_load(
                staged_with_nested_dict_conditional,
                self.path(),
                backend="eager",
                dynamic=False,
            ) as loaded,
            serving(),
        ):
            for cfg, want in zip(configs, expected):
                self.assertEqual(loaded(x, cfg), want)
            # a key set never captured must not match either graph
            with self.assertRaisesRegex(RuntimeError, "fail_on_recompile"):
                loaded(x, {"gamma": {"kind": "wide", "dims": [1], "weight": 2}})

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

    def test_raised_recompile_limit_captures_every_variant(self):
        n = 5
        model = PrecompileStack(n)
        inputs = [torch.randn(*s) for s in [(4, 8), (5, 8), (6, 8)]]
        expected = [model(x) for x in inputs]

        session = precompile_capture(
            model, backend="eager", recompile_limit=64, dynamic=False
        )
        with session as compiled:
            for x in inputs:
                compiled(x)
        summary = session.summary()
        self.assertEqual(summary.truncated, ())
        self.assertEqual(summary.guarded_codes, n * len(inputs))
        # The top-level forward only dispatches to submodules, so Dynamo keeps
        # no graph for it and install() will skip it. That is reported rather
        # than counted as complete, because it is indistinguishable from a frame
        # Dynamo gave up on.
        self.assertEqual(summary.uncovered_frames, ("forward",))
        self.assertFalse(summary.complete)
        with self.assertRaisesRegex(PackageError, "no compiled code for entry frame"):
            session.save(self.path())
        session.save(self.path(), require_complete=False)

        torch._dynamo.reset()
        with (
            precompile_load(
                model, self.path(), backend="eager", recompile_limit=64, dynamic=False
            ) as loaded,
            serving(),
        ):
            for x, want in zip(inputs, expected):
                self.assertEqual(loaded(x), want)
            with self.assertRaisesRegex(RuntimeError, "fail_on_recompile"):
                loaded(torch.randn(7, 8))

    def test_save_refuses_risky_dropped_identity_guard(self):
        # Identity guards cannot be serialized, so a bare global holding a
        # function loses its guard. Rebinding it between capture and load would
        # then silently serve the graph traced against the old value.
        session = precompile_capture(
            staged_with_global_function_ref, backend="eager", dynamic=False
        )
        with session as compiled:
            compiled(torch.randn(4, 8))
        summary = session.summary()
        risky = [name for _, name in summary.risky_dropped_guards]
        self.assertIn("G['PRECOMPILE_ACTIVATION']", risky)
        # Guards on the torch module itself are dropped too but are not risky.
        self.assertNotIn("G['torch']", risky)
        with self.assertRaisesRegex(PackageError, "PRECOMPILE_ACTIVATION"):
            session.save(self.path(), require_no_dropped_guards=False)
        # The risk is acknowledgeable, not a hard block.
        session.save(
            self.path(),
            require_no_risky_drops=False,
            require_no_dropped_guards=False,
        )

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

    @parametrize("owner", ("user", "torch", "torch_functional", "builtin"))
    def test_risky_drop_detected_through_a_module_attribute(self, owner):
        # self.act is a dispatch slot whatever it holds. Its guard is dropped as
        # an unserializable identity guard and the source is reported with local
        # scope stripped ("self.act"), so it cannot be recognised by matching
        # against a global pattern -- classify by the binding site. Keying on who
        # owns the value instead would wave through the torch-owned cases, which
        # are the common ones (self.act = getattr(F, cfg.activation)), and the
        # builtin case too -- an ACT2FN-style table holds abs next to torch.relu,
        # and which one lands in the slot is exactly what config decides.
        act = {
            "user": _precompile_user_act,
            "torch": torch.relu,
            "torch_functional": torch.nn.functional.gelu,
            "builtin": abs,
        }[owner]
        session = precompile_capture(
            PrecompileSelfAct(act), backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        risky = [name for _, name in session.summary().risky_dropped_guards]
        self.assertIn("self.act", risky)
        with self.assertRaisesRegex(PackageError, "self.act"):
            session.save(self.path(), require_no_dropped_guards=False)

    def test_library_and_def_site_drops_need_relaxed_save(self):
        # Ordinary code with no dispatch slot still drops identity guards: on
        # torch internals, on stdlib modules and their attributes, and on a
        # global bound to a def of its own name. The relaxed lint waives these
        # common shapes to stay usable, even though runtime rebinding can still
        # invalidate them; the strict all-drops default is the sound gate.
        session = precompile_capture(
            PrecompileNoDispatchSlot(), backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        summary = session.summary()
        dropped = [name for _, name in summary.dropped_guards]
        self.assertIn("G['math']", dropped)
        self.assertIn("G['math'].sqrt", dropped)
        self.assertIn("G['_precompile_scale']", dropped)
        self.assertEqual(summary.risky_dropped_guards, ())
        session.save(self.path())
        with self.assertRaisesRegex(PackageError, "not serialized"):
            session.save(self.path(), require_no_dropped_guards=True)

    def test_a_builtin_read_the_ordinary_way_is_not_a_risky_drop(self):
        # The other side of the "builtin" case above. len() and sorted() are
        # reached through the builtins dict Dynamo installs, with no binding in
        # front of them for config to repoint, so the exemption has to survive
        # being narrowed to that read -- otherwise every model calling len()
        # warns on every save and the report stops being read.
        cfg = {"alpha": 1, "beta": 2}
        session = precompile_capture(
            staged_with_builtin_calls, backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4), cfg)
        summary = session.summary()
        dropped = [name for _, name in summary.dropped_guards]
        self.assertTrue(any(n.endswith("['len']") for n in dropped), dropped)
        self.assertTrue(any(n.endswith("['sorted']") for n in dropped), dropped)
        self.assertEqual(summary.risky_dropped_guards, ())
        session.save(self.path())
        with self.assertRaisesRegex(PackageError, "not serialized"):
            session.save(self.path(), require_no_dropped_guards=True)

    def test_a_builtin_shaped_read_that_is_still_a_slot_is_risky(self):
        # Both halves of that exemption carry weight, so neither can be dropped
        # for being redundant. builtins is writable and per-process, so a plugin
        # doing `builtins.op = impl_a` here and impl_b there produces a read
        # that comes off the right namespace but holds user code. And a table
        # under a user global has the same source shape as the builtins dict
        # while being a slot, whatever it happens to hold.
        builtins.precompile_house_op = _precompile_house_op
        self.addCleanup(lambda: delattr(builtins, "precompile_house_op"))
        session = precompile_capture(
            staged_with_injected_builtin, backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        risky = [name for _, name in session.summary().risky_dropped_guards]
        injected = [n for n in risky if n.endswith("['precompile_house_op']")]
        self.assertEqual(len(injected), 1, risky)

        torch._dynamo.reset()
        session = precompile_capture(
            staged_with_a_registry_keyed_by_a_builtin_name,
            backend="eager",
            dynamic=False,
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4), {"a": 1})
        risky = [name for _, name in session.summary().risky_dropped_guards]
        self.assertIn("G['_PRECOMPILE_OPS']['len']", risky)

    def test_a_fully_qualified_torch_op_is_not_a_risky_drop(self):
        # torch.nn.functional.gelu spelled out guards every module on the way
        # down, so the namespace the call comes off is two attribute hops from
        # the global rather than one. Walking only one hop back to look for a
        # global root drops it out of the trusted set and refuses the most
        # ordinary spelling there is.
        session = precompile_capture(
            PrecompileFullyQualified(), backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        summary = session.summary()
        dropped = [name for _, name in summary.dropped_guards]
        self.assertIn("G['torch'].nn.functional.gelu", dropped)
        self.assertEqual(summary.risky_dropped_guards, ())
        session.save(self.path())

    def test_a_submodule_of_a_self_named_package_is_not_a_risky_drop(self):
        # own_parent is bound under its own name, which no import statement can
        # repoint, and own_parent.own_sub inherits that: the trust has to carry
        # down the attribute chain, or reaching a helper through the package
        # that owns it counts as dispatch.
        pkg = self._write_module("nest", "own_sub", _OWN_SUB_SRC)
        self._write_module("nest", "own_parent", _OWN_PARENT_SRC)
        self._write_module("nest", "own_nested_model", _OWN_NESTED_MODEL_SRC)
        self._forget_modules("own_sub", "own_parent")
        model = self._import_module(pkg, "own_nested_model").Model()

        session = precompile_capture(model, backend="eager", dynamic=False)
        with session as compiled, torch.no_grad():
            compiled(torch.ones(4))
        summary = session.summary()
        dropped = [name for _, name in summary.dropped_guards]
        self.assertIn("G['own_parent'].own_sub", dropped)
        self.assertIn("G['own_parent'].own_sub.op", dropped)
        self.assertEqual(summary.risky_dropped_guards, ())
        session.save(self.path())

    def test_library_spellings_that_only_the_owner_test_clears(self):
        # Four spellings the def-name rule cannot clear on its own, so whether
        # they are refused rides entirely on recognising torch and the stdlib:
        # `from torch import relu` (owned by torch itself, not a torch.*
        # submodule), `from math import sqrt` (a def in a C stdlib module, so
        # there is no file to match the reader against), `import math as _math`
        # and `import collections.abc as _abc` (modules under names that are
        # not their own, one of them dotted).
        pkg = self._write_module("libspell", "lib_spellings", _LIBRARY_SPELLINGS_SRC)
        model = self._import_module(pkg, "lib_spellings").Model()

        session = precompile_capture(model, backend="eager", dynamic=False)
        with session as compiled, torch.no_grad():
            compiled(torch.ones(4))
        summary = session.summary()
        dropped = [name for _, name in summary.dropped_guards]
        for name in ("G['relu']", "G['sqrt']", "G['_math']", "G['_abc']"):
            self.assertIn(name, dropped)
        self.assertEqual(summary.risky_dropped_guards, ())
        session.save(self.path())

    def test_an_aliased_module_import_is_not_a_risky_drop(self):
        # `import torch.nn.functional as F` binds a global whose name is nothing
        # like the module's __name__, so the def-name rule that clears
        # `G['math']` does not clear `G['F']`. What clears it is that torch owns
        # the module: there is one torch.nn.functional and no config picks
        # another, and a check that flagged every model spelling its imports the
        # ordinary way would be ignored from day one.
        session = precompile_capture(
            PrecompileAliasedImport(), backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        summary = session.summary()
        self.assertIn("G['F']", [name for _, name in summary.dropped_guards])
        self.assertEqual(summary.risky_dropped_guards, ())
        session.save(self.path())

    def test_a_user_module_aliased_by_a_conditional_import_is_a_risky_drop(self):
        # The same read as F.gelu -- an attribute off a module namespace -- but
        # which module `impl` holds is an env var's choice, so capture and serve
        # disagree while every other rail passes: the artifact records the
        # capture-time module as its inlined source and that module is untouched
        # on the serving box, so the checksum revalidates.
        pkg = self._write_module("plug", "pick_a", _PLUGGABLE_OP_SRC.format(f="2.0"))
        self._write_module("plug", "pick_b", _PLUGGABLE_OP_SRC.format(f="7.0"))
        self._write_module("plug", "alias_dispatch", _ALIAS_DISPATCH_SRC)
        self._forget_modules("pick_a", "pick_b")
        os.environ.pop("PICK_B", None)
        self.addCleanup(lambda: os.environ.pop("PICK_B", None))
        model = self._import_module(pkg, "alias_dispatch").Model()

        x = torch.ones(4)
        session = precompile_capture(model, backend="eager", dynamic=False)
        with session as compiled, torch.no_grad():
            compiled(x)
        risky = [name for _, name in session.summary().risky_dropped_guards]
        self.assertIn("G['impl']", risky)
        self.assertIn("G['impl'].op", risky)
        with self.assertRaisesRegex(PackageError, r"G\['impl'\]"):
            session.save(self.path(), require_no_risky_drops=True)

        # What the refusal buys, spelled out: opt out and the serving machine
        # runs the other backend eagerly and the captured one under the artifact.
        session.save(self.path(), require_no_risky_drops=False)
        os.environ["PICK_B"] = "1"
        sys.modules.pop("alias_dispatch", None)
        flipped = importlib.import_module("alias_dispatch").Model()
        self.assertEqual(flipped(x), x * 7.0)
        torch._dynamo.reset()
        with (
            precompile_load(flipped, self.path(), backend="eager", dynamic=False) as l,
            serving(),
            torch.no_grad(),
        ):
            self.assertEqual(l(x), x * 2.0)

    def test_a_cross_module_from_import_is_a_risky_drop(self):
        # `from pick_a import op` binds a global to a def of that same name, so
        # the def-name rule would clear it, but the def lives in another file:
        # repointing it takes a conditional import in THIS file, which is not an
        # edit anywhere and which no checksum covers.
        pkg = self._write_module("frm", "pick_a", _PLUGGABLE_OP_SRC.format(f="2.0"))
        self._write_module("frm", "pick_b", _PLUGGABLE_OP_SRC.format(f="7.0"))
        self._write_module("frm", "from_dispatch", _FROM_IMPORT_DISPATCH_SRC)
        self._forget_modules("pick_a", "pick_b")
        os.environ.pop("PICK_B", None)
        self.addCleanup(lambda: os.environ.pop("PICK_B", None))
        model = self._import_module(pkg, "from_dispatch").Model()

        session = precompile_capture(model, backend="eager", dynamic=False)
        with session as compiled, torch.no_grad():
            compiled(torch.ones(4))
        risky = [name for _, name in session.summary().risky_dropped_guards]
        self.assertIn("G['op']", risky)
        with self.assertRaisesRegex(PackageError, r"G\['op'\]"):
            session.save(self.path(), require_no_risky_drops=True)

    def test_a_self_named_module_import_is_not_a_risky_drop(self):
        # The other side of that line. `import own_helpers` binds the module
        # under its own name, which no import statement can repoint elsewhere,
        # and `from torch.nn.functional import gelu` names a def torch owns.
        # Dynamo also reaches the helper's own globals through the
        # `__import_own_helpers` alias it installs, which follows the value
        # rather than any user binding. Refusing these would refuse most code.
        pkg = self._write_module("own", "own_helpers", _OWN_HELPERS_SRC)
        self._write_module("own", "own_model", _OWN_MODEL_SRC)
        self._forget_modules("own_helpers")
        model = self._import_module(pkg, "own_model").Model()

        session = precompile_capture(model, backend="eager", dynamic=False)
        with session as compiled, torch.no_grad():
            compiled(torch.ones(4))
        summary = session.summary()
        dropped = [name for _, name in summary.dropped_guards]
        self.assertIn("G['own_helpers']", dropped)
        self.assertIn("G['gelu']", dropped)
        self.assertIn("G['__import_own_helpers']._scale", dropped)
        self.assertEqual(summary.risky_dropped_guards, ())
        session.save(self.path())

    def test_reads_off_an_import_alias_are_anchored_to_its_module(self):
        # A function-scoped `import torch` is reached through the alias Dynamo
        # installs for it, and Dynamo guards the attributes read off that alias
        # but never the bare alias, so nothing in the guards holds the module.
        # Left unanchored, torch.Tensor reads as a config-swappable slot.
        session = precompile_capture(
            PrecompileFunctionScopedImport(), backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.ones(4))
        summary = session.summary()
        dropped = [name for _, name in summary.dropped_guards]
        self.assertIn("G['__import_torch'].Tensor", dropped)
        self.assertIn("G['__import_torch'].relu", dropped)
        self.assertEqual(summary.risky_dropped_guards, ())
        session.save(self.path(), require_no_risky_drops=True)

    def test_reads_off_a_user_module_import_alias_are_anchored(self):
        # The same shape clearing on the other arm: the alias names a user
        # module, so nothing about torch or the stdlib waives the read, and
        # what does is that the namespace it decodes to owns the def.
        pkg = self._write_module("lazy", "lazy_helper", _LAZY_HELPER_SRC)
        self._forget_modules("lazy_helper")
        self._import_module(pkg, "lazy_helper")

        session = precompile_capture(
            PrecompileFunctionScopedUserImport(), backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.ones(4))
        summary = session.summary()
        dropped = [name for _, name in summary.dropped_guards]
        self.assertIn("G['__import_lazy_helper'].op", dropped)
        self.assertEqual(summary.risky_dropped_guards, ())
        session.save(self.path(), require_no_risky_drops=True)

    def test_an_import_alias_decodes_to_the_module_it_names(self):
        self.assertIs(_dynamo_alias_module("__import_torch"), torch)
        self.assertIs(
            _dynamo_alias_module("__import_torch_dot_nn_dot_functional"),
            torch.nn.functional,
        )
        self.assertIsNone(_dynamo_alias_module("__import_not_a_module_at_all"))
        self.assertIsNone(_dynamo_alias_module("CFG"))

    def test_a_module_held_in_an_attribute_is_a_risky_drop(self):
        # A module read off a global is fixed by the import statement, but one
        # parked in an instance attribute -- self.ns =
        # importlib.import_module(cfg.backend) -- is a slot config picks, and
        # the serving machine's config can pick another. Both the attribute and
        # everything reached through it have to stay risky.
        session = precompile_capture(
            PrecompileModuleInAttribute(torch.nn.functional),
            backend="eager",
            dynamic=False,
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        risky = [name for _, name in session.summary().risky_dropped_guards]
        self.assertIn("self.ns", risky)
        self.assertIn("self.ns.gelu", risky)
        with self.assertRaisesRegex(PackageError, "self.ns"):
            session.save(self.path(), require_no_risky_drops=True)

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

    def _capture_corpus_shape(self, model, args):
        session = precompile_capture(model, backend="eager", dynamic=False)
        with session as compiled, torch.no_grad():
            compiled(*args)
        return session

    @parametrize("shape", sorted(_RISKY_DROP_CORPUS))
    def test_risky_drop_corpus_is_flagged(self, shape):
        # See _RISKY_DROP_CORPUS: this predicate has failed open three times,
        # so every shape ever found is asserted here rather than described.
        expected, build = _RISKY_DROP_CORPUS[shape]
        session = self._capture_corpus_shape(*build(self))
        risky = [name for _, name in session.summary().risky_dropped_guards]
        self.assertIn(expected, risky)
        # Enforcement is the default; the corpus opts out only to prove that
        # every risky shape remains serializable when explicitly acknowledged.
        with self.assertRaisesRegex(PackageError, re.escape(expected)):
            session.save(self.path(), require_no_dropped_guards=False)
        session.save(
            self.path(),
            require_no_risky_drops=False,
            require_no_dropped_guards=False,
        )

    @parametrize("shape", sorted(_BENIGN_DROP_CORPUS))
    def test_benign_drop_corpus_is_not_flagged(self, shape):
        session = self._capture_corpus_shape(*_BENIGN_DROP_CORPUS[shape](self))
        self.assertEqual(session.summary().risky_dropped_guards, ())
        session.save(self.path())
        with self.assertRaisesRegex(PackageError, "not serialized"):
            session.save(self.path(), require_no_dropped_guards=True)

    def test_summary_reports_value_pinned_guards(self):
        # A value crossing a graph break is guarded by equality, so the artifact
        # only serves inputs reproducing it. Nothing else in the summary says so.
        session = precompile_capture(
            PrecompileValuePinned(), backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        summary = session.summary()
        self.assertTrue(summary.wont_generalize)
        self.assertTrue(any("___stack" in n for n in summary.wont_generalize))
        self.assertIn("value-pinned", str(summary))

    def test_two_packages_on_a_shared_frame_can_both_unload(self):
        # Two instances of one class share a forward code object. Refusing to
        # uninstall while another package is installed would deadlock them,
        # since neither could go first.
        paths = []
        for act in (torch.relu, torch.sigmoid):
            torch._dynamo.reset()
            session = precompile_capture(
                PrecompileSelfAct(act), backend="eager", dynamic=False
            )
            with session as compiled, torch.no_grad():
                compiled(torch.randn(3, 4))
            path = self.path(f"pkg_{act.__name__}.pt")
            # self.act is a dispatch slot; this test is about unloading, not it.
            session.save(path, require_no_risky_drops=False)
            paths.append(path)

        torch._dynamo.reset()
        first = precompile_load(
            PrecompileSelfAct(torch.relu), paths[0], backend="eager", dynamic=False
        )
        second = precompile_load(
            PrecompileSelfAct(torch.sigmoid), paths[1], backend="eager", dynamic=False
        )
        # The registry is what makes the clobber visible; without it the second
        # package silently stops serving this frame with nothing said.
        with self.assertLogs("torch._dynamo.package", level="WARNING") as logs:
            first.unload()
        self.assertTrue(any("other loaded package" in m for m in logs.output))
        # first is out of the registry now, so the last one out has nothing to say.
        with self.assertNoLogs("torch._dynamo.package", level="WARNING"):
            second.unload()

    def test_stale_artifact_rejected_when_source_drifts(self):
        # The deployment shape is capture on one machine, serve on another. The
        # dangerous version of that is an artifact outliving a code change, so
        # the source checksum has to fire even though the module is found by
        # name and its path differs between the two machines.
        src = "import torch\n\n\ndef staged(x):\n    y = x * 2\n    torch._dynamo.graph_break()\n    return (y + 1).sum()\n"
        pkg_dir = os.path.join(self.dir(), "srcdrift")
        os.makedirs(pkg_dir, exist_ok=True)
        mod_path = os.path.join(pkg_dir, "drift_mod.py")
        with open(mod_path, "w") as f:
            f.write(src)

        sys.path.insert(0, pkg_dir)
        try:
            mod = importlib.import_module("drift_mod")
            session = precompile_capture(mod.staged, backend="eager", dynamic=False)
            with session as compiled, torch.no_grad():
                compiled(torch.randn(4, 8))
            session.save(self.path())

            # The serving machine runs a slightly different build.
            with open(mod_path, "w") as f:
                f.write(src.replace("y + 1", "y + 2"))
            importlib.invalidate_caches()
            del sys.modules["drift_mod"]
            mod2 = importlib.import_module("drift_mod")
            torch._dynamo.reset()
            with self.assertRaisesRegex(RuntimeError, "Source code changes detected"):
                precompile_load(
                    mod2.staged, self.path(), backend="eager", dynamic=False
                )
        finally:
            sys.path.remove(pkg_dir)
            sys.modules.pop("drift_mod", None)

    def test_artifact_rejected_on_version_skew(self):
        # Guards and bytecode are version specific, so an artifact must not load
        # onto a machine running a different Python or PyTorch.
        current = SystemInfo.current()
        for field, bad in (
            ("python_version", "3.0.0"),
            ("torch_version", "0.0.0"),
        ):
            skewed = dataclasses.replace(current, **{field: bad})
            with self.assertRaisesRegex(RuntimeError, "different"):
                skewed.check_compatibility(current, "cpu")

    def test_two_artifacts_sharing_an_inner_frame_both_serve(self):
        # Two DIFFERENT models containing the same library block share that
        # block's frame and its resume function. Unlike two artifacts for one
        # class, which collide on the entry frame and evict each other, these
        # coexist: precompile entries accumulate on the shared code object and
        # the guards pick the right one. Pin that, because the alternative --
        # the second load evicting the first -- is silent here, the eviction
        # warning covering entry frames only.
        x = torch.ones(3, 4)
        paths = []
        for cls, scale in ((PrecompileSharedUserA, 3.0), (PrecompileSharedUserB, 7.0)):
            torch._dynamo.reset()
            session = precompile_capture(cls(scale), backend="eager", dynamic=False)
            with session as compiled, torch.no_grad():
                compiled(x)
            # No opt-out: a float attribute is a serializable guard.
            self.assertEqual(session.summary().risky_dropped_guards, ())
            path = self.path(f"shared_{cls.__name__}.pt")
            session.save(path, require_complete=False)
            paths.append(path)

        torch._dynamo.reset()
        model_a, model_b = PrecompileSharedUserA(3.0), PrecompileSharedUserB(7.0)
        with torch.no_grad():
            want_a, want_b = model_a(x), model_b(x)
        with (
            precompile_load(model_a, paths[0], backend="eager", dynamic=False) as a,
            precompile_load(model_b, paths[1], backend="eager", dynamic=False) as b,
            torch.no_grad(),
            serving(),
        ):
            # The surviving artifact still serves its own model correctly.
            self.assertEqual(b(x), want_b)
            # The evicted one must not be served the survivor's graph. Its
            # entries are gone, so this is a miss, and fail_on_recompile makes
            # the miss loud instead of letting it recompile.
            self.assertEqual(a(x), want_a)

    @torch._dynamo.config.patch(nested_graph_breaks=False)
    def test_concurrent_unload_does_not_clear_a_later_package(self):
        shared = self._import_module(
            self._write_module("shared_race", "shared_race", _SHARED_RACE_SRC),
            "shared_race",
        )
        x = torch.randn(3, 4)
        paths = []
        for cls, scale in ((shared.ModelOne, 3.0), (shared.ModelTwo, 7.0)):
            torch._dynamo.reset()
            session = precompile_capture(cls(scale), backend="eager", dynamic=False)
            with session as compiled, torch.no_grad():
                compiled(x)
            path = self.path(f"concurrent_{cls.__name__}.pt")
            session.save(path, require_complete=False)
            paths.append(path)

        torch._dynamo.reset()
        model_a = shared.ModelOne(3.0)
        model_b = shared.ModelTwo(7.0)
        with torch.no_grad():
            want_b = model_b(x)
        loaded_a = precompile_load(model_a, paths[0], backend="eager", dynamic=False)

        eval_frame = torch._C._dynamo.eval_frame
        real_reset = eval_frame._reset_precompile_entries
        shared_code = shared.SharedBlock.forward.__code__
        self.assertIn(shared_code, loaded_a._package._installed_precompile_codes)
        at_shared_reset = threading.Event()
        allow_reset = threading.Event()
        b_waiting_for_lock = threading.Event()
        b_loaded = threading.Event()
        errors = queue.SimpleQueue()
        loaded_b = queue.SimpleQueue()
        unload_thread = None
        load_thread = None

        class ObservedLock:
            def __init__(self, lock):
                self.lock = lock

            def __enter__(self):
                if threading.current_thread() is load_thread:
                    b_waiting_for_lock.set()
                self.lock.acquire()
                return self

            def __exit__(self, *exc):
                self.lock.release()

        def delayed_reset(code):
            if code is shared_code and threading.current_thread() is unload_thread:
                at_shared_reset.set()
                if not allow_reset.wait(10):
                    raise RuntimeError("timed out waiting to reset precompile entries")
            real_reset(code)

        def unload_a():
            try:
                loaded_a.unload()
            except Exception as e:
                errors.put(e)

        def load_b():
            try:
                loaded_b.put(
                    precompile_load(model_b, paths[1], backend="eager", dynamic=False)
                )
                b_loaded.set()
            except Exception as e:
                errors.put(e)

        observed_lock = ObservedLock(dynamo_package._PACKAGE_INSTALL_LOCK)
        with (
            mock.patch.object(dynamo_package, "_PACKAGE_INSTALL_LOCK", observed_lock),
            mock.patch.object(eval_frame, "_reset_precompile_entries", delayed_reset),
        ):
            unload_thread = threading.Thread(target=unload_a)
            unload_thread.start()
            try:
                self.assertTrue(at_shared_reset.wait(10))
                load_thread = threading.Thread(target=load_b)
                load_thread.start()
                self.assertTrue(b_waiting_for_lock.wait(10))
                self.assertFalse(b_loaded.is_set())
            finally:
                allow_reset.set()
                unload_thread.join(10)
                if load_thread is not None:
                    load_thread.join(10)

        self.assertFalse(unload_thread.is_alive())
        self.assertIsNotNone(load_thread)
        self.assertFalse(load_thread.is_alive())
        self.assertTrue(errors.empty())
        loaded = loaded_b.get_nowait()
        self.assertTrue(loaded_b.empty())
        with loaded, torch.no_grad(), serving():
            self.assertEqual(loaded(x), want_b)

    @torch.compiler.set_stance("default")
    def test_overlapping_serving_contexts_keep_compilation_disabled(self):
        torch._dynamo.reset()

        @torch.compile(backend="eager", dynamic=False)
        def compiled(x):
            return x + 1

        compiled(torch.randn(2))
        a_entered = threading.Event()
        b_entered = threading.Event()
        a_exited = threading.Event()
        errors = queue.SimpleQueue()
        rejected = queue.SimpleQueue()

        def serve_a():
            try:
                with serving():
                    a_entered.set()
                    if not b_entered.wait(10):
                        raise RuntimeError("timed out waiting for serving thread")
                a_exited.set()
            except Exception as e:
                errors.put(e)

        def serve_b():
            try:
                if not a_entered.wait(10):
                    raise RuntimeError("timed out waiting for serving thread")
                with serving():
                    b_entered.set()
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

        a = threading.Thread(target=serve_a)
        b = threading.Thread(target=serve_b)
        try:
            a.start()
            b.start()
            a.join(10)
            b.join(10)
        finally:
            a_entered.set()
            b_entered.set()
            a_exited.set()

        self.assertFalse(a.is_alive())
        self.assertFalse(b.is_alive())
        self.assertTrue(errors.empty())
        self.assertTrue(rejected.get_nowait())
        self.assertTrue(rejected.empty())
        self.assertEqual(torch._dynamo.eval_frame._stance.stance, "default")

    def test_example_inputs_drive_the_capture(self):
        # example_inputs is just "run these for me": capture is by execution, so
        # the block body becomes optional.
        session = precompile_capture(
            PrecompileInvariantModel(),
            backend="eager",
            dynamic=False,
            example_inputs=[(torch.ones(4, 8),), (torch.ones(5, 8),)],
        )
        with session:
            pass
        summary = session.summary()
        self.assertEqual(summary.frames, 2)
        self.assertEqual(summary.guarded_codes, 4)

    def test_invariants_separate_what_holds_from_what_varies(self):
        # Two shapes, one config value. The config read is on both sides of the
        # break, so it is invariant; the shapes are what tell the graphs apart.
        session = precompile_capture(
            PrecompileInvariantModel(),
            backend="eager",
            dynamic=False,
            example_inputs=[(torch.ones(4, 8),), (torch.ones(5, 8),)],
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

    def test_invariants_file_is_written_and_reproducible(self):
        def capture(path):
            torch._dynamo.reset()
            with precompile_capture(
                PrecompileInvariantModel(),
                backend="eager",
                dynamic=False,
                example_inputs=[(torch.ones(4, 8),), (torch.ones(5, 8),)],
                invariants=path,
            ):
                pass

        first = self.path("a.invariants")
        second = self.path("b.invariants")
        capture(first)
        capture(second)
        with open(first) as handle:
            text = handle.read()
        self.assertIn("frame forward", text)
        self.assertIn("invariant [enforced]", text)
        self.assertIn("varies", text)
        # Object ids are normalized out, so the file is stable enough to commit
        # and diff across runs.
        self.assertNotRegex(text, r"\b\d{9,}\b")
        with open(second) as handle:
            self.assertEqual(text, handle.read())

    def test_invariants_path_is_a_file_not_a_directory(self):
        # Same contract as save(): the path names a file, written exactly where
        # asked with its parent directories created, which
        # test_save_writes_a_single_file pins for save() itself. A path kwarg
        # that names a directory to fill is the easy confusion, so pin it.
        path = os.path.join(self.dir(), "snapshots", "invariants.txt")
        self.assertFalse(os.path.exists(os.path.dirname(path)))
        with precompile_capture(
            PrecompileInvariantModel(),
            backend="eager",
            dynamic=False,
            example_inputs=[(torch.ones(4, 8),)],
            invariants=path,
        ):
            pass
        self.assertTrue(os.path.isfile(path))
        self.assertFalse(os.path.isdir(path))
        with open(path) as handle:
            self.assertIn("# precompile invariants for", handle.read())

    def test_invariants_report_tensor_properties_that_split_a_compilation(self):
        # TensorCheck compares the python type and the conj/neg bits as well as
        # dtype/shape/stride/device. A fingerprint missing them renders two
        # compilations as one fact, so the guard that SPLIT them gets printed as
        # an invariant of both -- the report lying, which is its worst failure.
        def f(x):
            return x * 2

        cases = (
            ("type", torch.nn.Parameter(torch.ones(4)), torch.ones(4)),
            (
                "conj",
                torch.ones(4, dtype=torch.complex64),
                torch.ones(4, dtype=torch.complex64).conj(),
            ),
        )
        for label, first, second in cases:
            with self.subTest(label):
                torch._dynamo.reset()
                session = precompile_capture(f, backend="eager", dynamic=False)
                with session as compiled, torch.no_grad():
                    compiled(first)
                    compiled(second)
                frame = session.invariants()[0]
                self.assertEqual(frame.variants, 2)
                self.assertTrue(
                    any(fact.guard_type == "TENSOR_MATCH" for fact in frame.varying),
                    f"{label}: the guard that split the compilations is not "
                    f"reported as varying: {[f.render() for f in frame.varying]}",
                )
                self.assertFalse(
                    any(fact.guard_type == "TENSOR_MATCH" for fact in frame.invariant)
                )

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
        def pack_a(x):
            return x + 1

        def unpack_a(x):
            return x - 1

        def pack_b(x):
            return x * 2

        def unpack_b(x):
            return x / 2

        first = (torch.fx.symbolic_trace(pack_a), torch.fx.symbolic_trace(unpack_a))
        second = (torch.fx.symbolic_trace(pack_b), torch.fx.symbolic_trace(unpack_b))

        def f(x):
            return (x * 2).sum()

        def capture(path):
            torch._dynamo.reset()
            session = precompile_capture(f, backend="eager", dynamic=False)
            with session as compiled:
                for hooks in (first, second):
                    with torch.autograd.graph.saved_tensors_hooks(*hooks):
                        compiled(torch.ones(4, 8, requires_grad=True)).backward()
            session.write_invariants(path)
            return session

        path = self.path("hooks.invariants")
        session = capture(path)

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

        with open(path) as handle:
            text = handle.read()
        self.assertNotRegex(text, r"\b\d{9,}\b")

        # ...and the discriminator is content-derived, so it survives a second
        # process where the addresses are different.
        other = self.path("hooks_again.invariants")
        capture(other)
        with open(other) as handle:
            self.assertEqual(text, handle.read())

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
            example_inputs=[(torch.ones(4, 8),), (torch.ones(5, 8),)],
        )
        with session:
            pass
        rendered = [
            f.render() for frame in session.invariants() for f in frame.invariant
        ]
        self.assertTrue(any(r.startswith("[dropped ]") for r in rendered))
        self.assertTrue(any(r.startswith("[enforced]") for r in rendered))

    def test_invariants_do_not_collapse_a_large_constant(self):
        # _normalize strips object ids so the file diffs clean. It must not
        # strip a user constant with it: these two variants pin the dict to
        # different keys, and collapsing them promotes a condition NEITHER
        # variant holds into the intersection.
        def fn(x, d):
            return x * next(iter(d.values()))

        session = precompile_capture(fn, backend="eager", dynamic=False)
        with session as compiled:
            compiled(torch.ones(4), {1000000001: 2})
            compiled(torch.ones(4), {2000000002: 3})
        (frame,) = session.invariants()
        self.assertEqual(frame.variants, 2)
        invariant = [f.render() for f in frame.invariant]
        varying = [f.render() for f in frame.varying]
        self.assertFalse(any("dict.keys" in r for r in invariant), invariant)
        self.assertTrue(any("[1000000001]" in r for r in varying), varying)
        self.assertTrue(any("[2000000002]" in r for r in varying), varying)

    def test_invariants_keep_a_guard_every_variant_shares(self):
        # k is unspecialized, so its guard is "k is an int" in both variants and
        # is a real precondition. Fingerprinting the value it happened to hold
        # would split one shared guard into two indistinguishable varying facts.
        def fn(x, flag, k):
            return x * k if flag else x + k

        session = precompile_capture(fn, backend="eager", dynamic=True)
        with session as compiled:
            compiled(torch.ones(4), True, 1)
            compiled(torch.ones(4), False, 2)
        (frame,) = session.invariants()
        self.assertEqual(frame.variants, 2)
        invariant = [f.render() for f in frame.invariant]
        varying = [f.render() for f in frame.varying]
        self.assertTrue(
            any("___check_type_id(L['k']" in r for r in invariant), invariant
        )
        self.assertEqual(len(varying), len(set(varying)), varying)

    def test_invariants_name_the_object_an_identity_guard_pinned(self):
        # The id in an identity guard's code is normalized away, so the object
        # has to be named some other way: without it the two variants -- traced
        # against relu and against sigmoid -- render one fact and self.act is
        # reported invariant, hiding the only thing that tells them apart.
        model = PrecompileSelfAct(torch.relu)
        session = precompile_capture(model, backend="eager", dynamic=False)
        with session as compiled:
            compiled(torch.ones(4))
            model.act = torch.sigmoid
            compiled(torch.ones(4, dtype=torch.float64))
        entry = {f.frame: f for f in session.invariants()}["forward"]
        self.assertEqual(entry.variants, 2)
        invariant = [f.render() for f in entry.invariant]
        varying = [f.render() for f in entry.varying]
        self.assertFalse(any(".act" in r for r in invariant), invariant)
        self.assertTrue(any(r.endswith("relu on self.act") for r in varying), varying)
        self.assertTrue(
            any(r.endswith("sigmoid on self.act") for r in varying), varying
        )

    def test_invariants_are_stable_across_dynamo_global_counters(self):
        # The builtins dict Dynamo installs carries a per-process counter, so
        # the same guard reads __builtins_dict___0 in one capture and ___4 in
        # the next. Un-normalized, a committed report changes every run and the
        # guard reports as varying rather than invariant. The model the other
        # invariants tests use reads no builtin, so nothing covers this today.
        def capture(path):
            torch._dynamo.reset()
            with precompile_capture(
                PrecompileBuiltinReadingModel(),
                backend="eager",
                dynamic=False,
                example_inputs=[(torch.ones(4, 8),), (torch.ones(5, 8),)],
                invariants=path,
            ):
                pass

        first, second = self.path("first.invariants"), self.path("second.invariants")
        capture(first)
        capture(second)
        with open(first) as handle:
            text = handle.read()
        self.assertIn("__builtins_dict___<n>", text)
        self.assertNotRegex(text, r"__builtins_dict___\d")
        with open(second) as handle:
            self.assertEqual(text, handle.read())

    def test_invariants_are_stable_across_globals_named_by_object_id(self):
        # install_global_by_id names a global "<prefix>_<id(value)>_c<n>", so a
        # guard reading one carries an address inside an identifier, where
        # neither the id nor the counter pattern can see it. `type(x) is
        # torch.Tensor` installs one; transformers' Qwen2 installs three, and
        # its report differed between processes before this was normalized.
        def fn(x):
            return x.sum() if type(x) is torch.Tensor else x

        path = self.path("byid.invariants")
        with precompile_capture(
            fn,
            backend="eager",
            dynamic=False,
            example_inputs=[(torch.ones(4),)],
            invariants=path,
        ):
            pass
        with open(path) as handle:
            text = handle.read()
        self.assertIn("_<id>_c<n>", text)
        self.assertNotRegex(text, r"_\d{9,}_c\d")

    def test_facts_differing_only_in_value_sort_apart(self):
        # Once the boilerplate code parts are filtered a TENSOR_MATCH renders no
        # code at all, so two shape specializations tie on every other component
        # of the sort key and their order falls to set iteration, which is hash
        # seeded: the file then differs between PROCESSES, which two captures in
        # one process cannot show.
        def fact(shape):
            return _GuardFact("TENSOR_MATCH", "x", (), f"shape={shape}", True)

        self.assertNotEqual(_fact_order(fact((4, 8))), _fact_order(fact((5, 8))))

    def test_grad_mode_is_reported_when_variants_differ_in_it(self):
        # example_inputs run under no_grad and body calls do not, so the same
        # call made both ways compiles twice. Global-state guards carry no value
        # of their own, so without a fingerprint the report shows two variants
        # and nothing varying -- the half that is supposed to tell them apart.
        x = torch.ones(4, 8)
        session = precompile_capture(
            PrecompileInvariantModel(),
            backend="eager",
            dynamic=False,
            example_inputs=[(x,)],
        )
        with session as compiled:
            compiled(x)
        (frame,) = [f for f in session.invariants() if f.frame == "forward"]
        self.assertEqual(frame.variants, 2)
        varies = [f.render() for f in frame.varying]
        self.assertTrue(any("grad_enabled=True" in r for r in varies), varies)
        self.assertTrue(any("grad_enabled=False" in r for r in varies), varies)

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

    def test_repeated_guard_facts_are_stored_once_per_session(self):
        # A recompiled frame repeats nearly all of its guards, so storing one
        # object per compilation makes the session grow with the number of
        # variants instead of with the number of distinct facts. Measured on
        # resnet18, 200 variants retained 83MB to describe 1281 facts.
        session = precompile_capture(
            PrecompileIntArg(),
            backend="eager",
            recompile_limit=64,
            dynamic=False,
            example_inputs=[(torch.ones(4, 8), k) for k in range(20)],
        )
        with session:
            pass
        facts = [f for sets in session._guard_sets.values() for s in sets for f in s]
        self.assertGreater(len(facts), 5 * len(set(facts)))
        self.assertEqual(len({id(f) for f in facts}), len(set(facts)))

    def test_load_rejects_artifact_from_a_different_callable(self):
        x = torch.randn(4, 8)
        session = precompile_capture(
            staged_with_graph_breaks, backend="eager", dynamic=False
        )
        with session as compiled:
            compiled(x)
        session.save(self.path())

        # CompilePackage rebinds the stored guards onto whatever callable it is
        # given, and the source checksum only covers the captured function, so
        # without an explicit check this silently serves the wrong graphs.
        torch._dynamo.reset()
        with self.assertRaisesRegex(PackageError, "captured from"):
            precompile_load(
                staged_with_local_dict_conditional,
                self.path(),
                backend="eager",
                dynamic=False,
            )

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

    def test_unload_clears_resume_function_entries(self):
        # uninstall() used to clear precompile entries only for the entry frame,
        # leaving resume functions installed on module-level code objects for
        # the rest of the process.
        from torch._C._dynamo.eval_frame import _debug_get_precompile_entries

        x = torch.randn(4, 8)
        session = precompile_capture(
            staged_with_graph_breaks, backend="eager", dynamic=False
        )
        with session as compiled:
            compiled(x)
        session.save(self.path())

        torch._dynamo.reset()
        loaded = precompile_load(
            staged_with_graph_breaks, self.path(), backend="eager", dynamic=False
        )
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

    def test_truncation_report_is_a_lower_bound(self):
        # Hitting recompile_limit sets FrameExecStrategy(RUN_ONLY, RUN_ONLY),
        # whose recursive half silences every frame called beneath the offender,
        # yet only the offender is recorded. Pin the gap the message must admit.
        inputs = [torch.randn(*s) for s in [(4, 8), (5, 8), (6, 8)]]

        def capture(limit):
            torch._dynamo.reset()
            session = precompile_capture(
                PrecompileResumingStack(5),
                backend="eager",
                recompile_limit=limit,
                dynamic=False,
            )
            with session as compiled:
                for x in inputs:
                    compiled(x)
            resumes = {
                c.python_code.co_name: len(c.guarded_codes)
                for c in session._package.cache_entry().codes
                if c.python_code.co_name.startswith("torch_dynamo_resume")
            }
            return session, resumes

        full_session, full = capture(64)
        self.assertEqual(full_session.summary().truncated, ())
        self.assertTrue(full and all(n > 1 for n in full.values()))

        session, cut = capture(8)
        summary = session.summary()
        # One frame named, but the resume frames beneath it also lost variants.
        self.assertEqual(len(summary.truncated), 1)
        self.assertEqual(set(cut), set(full))
        self.assertTrue(all(cut[name] < full[name] for name in full))
        self.assertFalse(
            any(name in entry for name in full for entry in summary.truncated)
        )
        self.assertIn(">=1 TRUNCATED", str(summary))
        with self.assertRaisesRegex(PackageError, "lower bound"):
            session.save(self.path())

    def test_value_pinned_guard_on_a_plain_argument(self):
        # A non-tensor argument is pinned by equality exactly like a graph-break
        # stack slot, but gets no ___stackN name, so a name-keyed report misses
        # it and a complete-looking artifact silently serves only k == 7.
        session = precompile_capture(PrecompileIntArg(), backend="eager", dynamic=False)
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4), 7)
        summary = session.summary()
        self.assertEqual(summary.wont_generalize, ("k",))
        self.assertTrue(summary.complete)

    def test_a_value_pin_that_arrives_as_equals_match(self):
        # Which equality guard Dynamo picks depends on the value's type, not on
        # whether it pins anything: a dict_keys argument is compared with == and
        # lands on EQUALS_MATCH rather than CONSTANT_MATCH. Recognising only one
        # of the two reports a clean artifact that serves one key set.
        session = precompile_capture(
            PrecompileKeysArg(), backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4), {"a": 1, "b": 2}.keys())
        summary = session.summary()
        self.assertIn(("EQUALS_MATCH", "ks"), summary.kept_guards)
        self.assertEqual(summary.wont_generalize, ("ks",))

    def test_tensor_crossing_a_graph_break_is_not_value_pinned(self):
        # A tensor left on the stack across a break also gets a ___stackN
        # source, but under TENSOR_MATCH, which generalizes like any other
        # input. Reporting it would fire on ordinary tensor-only models.
        session = precompile_capture(
            PrecompileTensorAcrossBreak(), backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        summary = session.summary()
        self.assertTrue(any("___stack" in n for _, n in summary.kept_guards))
        self.assertEqual(summary.wont_generalize, ())

    def test_value_pinned_guards_cover_the_stack_slot_and_the_local(self):
        # The .item() result is guarded twice: once on the stack slot it crossed
        # the break in, once on the resume frame's local. Both pin the artifact.
        session = precompile_capture(
            PrecompileItemThenBreak(), backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        self.assertEqual(session.summary().wont_generalize, ("___stack0", "scale"))

    def test_module_config_constants_are_not_value_pinned(self):
        # LayerNorm eps, Dropout p and a plain int attribute are all
        # CONSTANT_MATCH but are model config, constant for the model the
        # artifact is loaded onto. Counting them would flag every model.
        for model in (PrecompileConfigConstants().eval(), PrecompileIntAttr(3)):
            torch._dynamo.reset()
            session = precompile_capture(model, backend="eager", dynamic=False)
            with session as compiled, torch.no_grad():
                compiled(torch.randn(2, 8))
            summary = session.summary()
            self.assertTrue(any(t == "CONSTANT_MATCH" for t, _ in summary.kept_guards))
            self.assertEqual(summary.wont_generalize, ())

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

    def test_load_rejects_a_same_named_callable_from_another_module(self):
        # Two modules each defining `class Encoder` agree on qualname and on
        # co_name, and each is self-consistent so the source checksum passes.
        a = self._import_module(
            self._write_module("a", "enc_a", _ENCODER_SRC.format(op="+")), "enc_a"
        )
        b = self._import_module(
            self._write_module("b", "enc_b", _ENCODER_SRC.format(op="*")), "enc_b"
        )
        session = precompile_capture(a.Encoder(), backend="eager", dynamic=False)
        with session as compiled, torch.no_grad():
            compiled(torch.randn(4))
        session.save(self.path())

        torch._dynamo.reset()
        with self.assertRaisesRegex(PackageError, "defined in 'enc_b'"):
            precompile_load(b.Encoder(), self.path(), backend="eager", dynamic=False)

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
            precompile_load(model, self.path(), backend="eager", dynamic=False)

    def test_load_survives_a_checkout_at_a_different_path(self):
        # The capture and serving machines check out to different absolute
        # paths, so co_filename must NOT be part of artifact identity. This
        # test exists to fail if someone tightens the check with it.
        src = _ENCODER_SRC.format(op="+")
        a = self._import_module(
            self._write_module("here", "moved_mod", src), "moved_mod"
        )
        x = torch.ones(4)
        session = precompile_capture(a.Encoder(), backend="eager", dynamic=False)
        with session as compiled, torch.no_grad():
            compiled(x)
        session.save(self.path())

        other = self._write_module("there", "moved_mod", src)
        sys.path.remove(os.path.dirname(a.__file__))
        b = self._import_module(other, "moved_mod")
        self.assertNotEqual(a.__file__, b.__file__)
        torch._dynamo.reset()
        with (
            precompile_load(
                b.Encoder(), self.path(), backend="eager", dynamic=False
            ) as loaded,
            torch.no_grad(),
        ):
            self.assertEqual(loaded(x), x + 1.0)

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
        with precompile_load(
            model, self.path(), backend="eager", dynamic=False
        ) as loaded:
            self.assertEqual(loaded(x), expected)

    def test_capture_rejects_a_callable_with_no_code_object(self):
        def scaled(scale, x):
            return x * scale

        class OnlyCall:
            def __call__(self, x):
                return x + 1.0

        for fn in (functools.partial(scaled, 2.0), OnlyCall()):
            with self.assertRaisesRegex(TypeError, "no __code__"):
                precompile_capture(fn, backend="eager")

    def test_capture_rejects_a_module_whose_forward_has_no_code_object(self):
        # An nn.Module reaches the same dead end one level down: self.forward =
        # functools.partial(...) in __init__ shadows the class method, so the
        # entry function has no __code__ either. Saying so is the whole point of
        # the check above; without it this died on a bare AttributeError from
        # inside CompilePackage.
        with self.assertRaisesRegex(TypeError, "no __code__"):
            precompile_capture(
                PrecompilePartialForward(2.0), backend="eager", dynamic=False
            )

    def test_load_refuses_a_callable_whose_code_attribute_is_none(self):
        # The capture-side check asks hasattr, so a callable carrying a __code__
        # that is present and None gets past it -- and load rebinds the stored
        # guards and bytecode onto whatever it is handed, so nothing downstream
        # is looking either. Refuse it at load with the same sentence rather
        # than somewhere deep in CompilePackage with a weakref TypeError.
        session = precompile_capture(
            PrecompileSelfAct(torch.relu), backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        session.save(self.path(), require_no_risky_drops=False)
        torch._dynamo.reset()
        with self.assertRaisesRegex(PackageError, "no __code__"):
            precompile_load(
                _PrecompileForwardsCode(len),
                self.path(),
                backend="eager",
                dynamic=False,
            )

    def test_device_type_records_the_accelerator_not_the_last_graph(self):
        # _DynamoCacheEntry.device_type gates every GPU check in
        # SystemInfo.check_compatibility, and one package holds many graphs, so
        # a cpu epilogue after a graph break must not erase the accelerator.
        def graph_on(device):
            graph = torch.fx.Graph()
            graph.call_function(torch.ones, args=(torch.device(device),))
            return graph

        package = CompilePackage(staged_with_graph_breaks)
        package.update_device_type(graph_on("cuda"))
        package.update_device_type(graph_on("cpu"))
        self.assertEqual(package.cache_entry().device_type, "cuda")

        # A load, a cpu recompile and a re-save must not downgrade it either.
        # Loading a "cuda" entry runs check_versions and so needs a GPU; the
        # restore path does not care which accelerator it is, so the round trip
        # uses one SystemInfo does not gate and stays runnable on a cpu host.
        accel = CompilePackage(staged_with_graph_breaks)
        accel.update_device_type(graph_on("mtia"))
        reloaded = CompilePackage(staged_with_graph_breaks, accel.cache_entry())
        reloaded.update_device_type(graph_on("cpu"))
        self.assertEqual(reloaded.cache_entry().device_type, "mtia")

        cpu_only = CompilePackage(staged_with_graph_breaks)
        cpu_only.update_device_type(graph_on("cpu"))
        self.assertEqual(cpu_only.cache_entry().device_type, "cpu")

    def test_a_failed_load_leaves_no_device_type_behind(self):
        # eval_frame's caching_precompile path retries a failed load on the SAME
        # package object, and update_device_type only widens cpu -> accelerator,
        # so a device_type left behind by the failed load can never be corrected
        # and gets re-saved as this capture's. One outside CHECK_GPUS turns off
        # every GPU check for whoever loads the artifact next.
        from torch._dynamo.package import _DynamoCacheEntry, InlinedSource, SourceInfo

        def graph_on(device):
            graph = torch.fx.Graph()
            graph.call_function(torch.ones, args=(torch.device(device),))
            return graph

        drifted = SourceInfo(
            inlined_sources={
                InlinedSource(
                    module="torch._dynamo.package",
                    firstlineno=1,
                    lastlineno=3,
                    checksum="not-the-checksum-on-disk",
                    content="",
                )
            }
        )
        stale = _DynamoCacheEntry(codes=[], source_info=drifted, device_type="mtia")

        package = CompilePackage(None)
        with self.assertRaisesRegex(RuntimeError, "Source code changes detected"):
            package.initialize(staged_with_graph_breaks, stale)
        self.assertFalse(package.is_initialized())

        package.initialize(staged_with_graph_breaks, None)
        package.update_device_type(graph_on("cuda"))
        entry = package.cache_entry()
        self.assertEqual(entry.device_type, "cuda")

        # "mtia" is not in CHECK_GPUS, so had it survived it would have waved
        # this artifact onto any host; what survives instead is gated.
        other_host = dataclasses.replace(entry.system_info, gpu_name="Some Other GPU")
        entry.system_info.check_compatibility(other_host, "mtia")
        self.assertIn(entry.device_type, SystemInfo.CHECK_GPUS)
        if torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "different GPU"):
                entry.system_info.check_compatibility(other_host, entry.device_type)

    @unittest.skipIf(not HAS_CUDA_AND_TRITON, "Requires CUDA/Triton")
    def test_device_type_of_a_cuda_capture_with_a_cpu_epilogue(self):
        from torch._dynamo.precompile_package import default_guard_filter_fn

        def cuda_then_cpu_epilogue(x):
            v = (x.sin() + 1.0).sum().item()
            return torch.tensor([v]) * 2

        package = CompilePackage(cuda_then_cpu_epilogue)
        compiled = torch._dynamo.optimize(
            "eager", package=package, guard_filter_fn=default_guard_filter_fn
        )(cuda_then_cpu_epilogue)
        compiled(torch.randn(8, device="cuda"))
        entry = package.cache_entry()
        # entry frame (cuda) plus the resume frame after .item() (cpu)
        self.assertEqual(len(entry.codes), 2)
        self.assertEqual(entry.device_type, "cuda")

    def test_second_artifact_for_a_shared_frame_warns_before_evicting(self):
        # Two instances of one class share a forward code object, and precompile
        # entries can only be cleared en masse. __init__ calls uninstall() to
        # start from a clean slate, before this package has installed anything,
        # so loading a second artifact took the first one's entries with it
        # without a word.
        x = torch.randn(3, 4)
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

        torch._dynamo.reset()
        relu_model = PrecompileSelfAct(torch.relu)
        with torch.no_grad():
            expected = relu_model(x)
        first = precompile_load(relu_model, paths[0], backend="eager", dynamic=False)
        with torch.no_grad(), serving():
            self.assertEqual(first(x), expected)

        with self.assertLogs("torch._dynamo.package", level="WARNING") as logs:
            second = precompile_load(
                PrecompileSelfAct(torch.sigmoid),
                paths[1],
                backend="eager",
                dynamic=False,
            )
        self.assertTrue(any("also installed on" in m for m in logs.output))
        # Regression guard on the wording: what follows is a silent
        # substitution, not the recompile the warning used to promise.
        self.assertTrue(any("silently" in m for m in logs.output))
        # The eviction itself is not preventable, so the first artifact now
        # dispatches into the second's graph -- the identity guard that told
        # them apart is not serializable. The warning is what makes it visible.
        with torch.no_grad(), serving():
            self.assertNotEqual(first(x), expected)
        second.unload()
        first.unload()

    def test_eviction_warns_once_and_not_when_nothing_is_left_to_evict(self):
        # precompile_load runs uninstall() twice on the shared frame -- once from
        # CompilePackage.__init__ and once from install() -- and only the first
        # evicts anything. A warning per call cries wolf about an eviction that
        # already happened, and cries at all after torch._dynamo.reset() has
        # cleared the entries out from under the neighbour, when there is
        # nothing left to take. Both make the real one easy to tune out.
        x = torch.randn(3, 4)
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

        def load(path, act):
            return precompile_load(
                PrecompileSelfAct(act), path, backend="eager", dynamic=False
            )

        torch._dynamo.reset()
        first = load(paths[0], torch.relu)
        with self.assertLogs("torch._dynamo.package", level="WARNING") as logs:
            second = load(paths[1], torch.sigmoid)
        evicted = [r for r in logs.output if "also installed on" in r]
        self.assertEqual(len(evicted), 1, logs.output)
        second.unload()
        first.unload()

        torch._dynamo.reset()
        first = load(paths[0], torch.relu)
        torch._dynamo.reset()
        with self.assertNoLogs("torch._dynamo.package", level="WARNING"):
            second = load(paths[1], torch.sigmoid)
        second.unload()
        first.unload()

    def test_repeated_unload_does_not_clear_a_later_package(self):
        x = torch.randn(3, 4)
        model = PrecompileSelfAct(torch.relu)
        session = precompile_capture(model, backend="eager", dynamic=False)
        with session as compiled, torch.no_grad():
            compiled(x)
        session.save(self.path(), require_no_risky_drops=False)

        torch._dynamo.reset()
        first = precompile_load(model, self.path(), backend="eager", dynamic=False)
        with first:
            first.unload()
            second = precompile_load(model, self.path(), backend="eager", dynamic=False)
        with second, torch.no_grad(), serving():
            self.assertEqual(second(x), model(x))

    def test_unload_keeps_a_skip_another_package_still_holds(self):
        # install() skip_code()s a frame with no guarded codes, and the strategy
        # it had before cannot be read back, so restoring it unconditionally
        # un-skips a frame a second loaded package still needs skipped.
        session = precompile_capture(PrecompileEmptyGraph(), backend="eager")
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        summary = session.save(self.path(), require_complete=False)
        self.assertEqual(summary.guarded_codes, 0)

        torch._dynamo.reset()
        first = precompile_load(PrecompileEmptyGraph(), self.path(), backend="eager")
        second = precompile_load(PrecompileEmptyGraph(), self.path(), backend="eager")

        first.unload()
        counters.clear()
        torch.compile(PrecompileEmptyGraph(), backend="eager")(torch.randn(3, 4))
        self.assertEqual(counters["frames"]["total"], 0)

        second.unload()
        counters.clear()
        torch.compile(PrecompileEmptyGraph(), backend="eager")(torch.randn(3, 4))
        self.assertEqual(counters["frames"]["total"], 1)

    def test_unload_restores_a_skipped_frame(self):
        # install() skip_code()s a frame it has no compiled code for, which is
        # global state on the code object. Without a restore the frame stays
        # unskippable for the rest of the process, so anything else that shares
        # it silently runs eager.
        model = PrecompileStack(5)
        x = torch.randn(4, 8)
        session = precompile_capture(model, backend="eager", dynamic=False)
        with session as compiled, torch.no_grad():
            compiled(x)
        self.assertEqual(session.summary().uncovered_frames, ("forward",))
        session.save(self.path(), require_complete=False)

        torch._dynamo.reset()
        loaded = precompile_load(model, self.path(), backend="eager", dynamic=False)
        self.assertIn(PrecompileStack.forward.__code__, loaded._package._skipped_codes)

        # Same code object, no blocks: it compiles a graph unless it is skipped.
        def frames_for_empty_stack():
            counter = torch._dynamo.testing.CompileCounter()
            out = torch.compile(
                PrecompileStack.forward, backend=counter, dynamic=False
            )(PrecompileStack(0), x)
            self.assertEqual(out, x.sum())
            return counter.frame_count

        self.assertEqual(frames_for_empty_stack(), 0)
        loaded.unload()
        self.assertEqual(loaded._package._skipped_codes, [])
        self.assertEqual(frames_for_empty_stack(), 1)

    def test_load_rejects_artifact_recorded_on_a_different_torch(self):
        # Capture on one machine, serve on another: the version check is only
        # worth anything if precompile_load runs it. Calling
        # SystemInfo.check_compatibility directly passes either way.
        session = precompile_capture(
            staged_with_graph_breaks, backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(4, 8))
        session.save(self.path())

        entry_path = self.path()
        with open(entry_path, "rb") as f:
            entry = pickle.load(f)
        entry.dynamo.system_info = dataclasses.replace(
            entry.dynamo.system_info, torch_version="0.0.0"
        )
        with open(entry_path, "wb") as f:
            pickle.dump(entry, f)

        torch._dynamo.reset()
        with self.assertRaisesRegex(RuntimeError, "different PyTorch version"):
            precompile_load(
                staged_with_graph_breaks, self.path(), backend="eager", dynamic=False
            )

    def test_load_rejects_an_artifact_with_no_code_entries(self):
        # Every other identity check reads dynamo.codes[0]. An artifact that
        # carries none has to be named as such here rather than falling through
        # to the unpack in CompilePackage.initialize, which reports a bare
        # "not enough values to unpack" with no mention of the path.
        session = precompile_capture(
            staged_with_graph_breaks, backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(4, 8))
        session.save(self.path())

        entry_path = self.path()
        with open(entry_path, "rb") as f:
            entry = pickle.load(f)
        entry.dynamo.codes = []
        with open(entry_path, "wb") as f:
            pickle.dump(entry, f)

        torch._dynamo.reset()
        with self.assertRaisesRegex(PackageError, "no code entries"):
            precompile_load(
                staged_with_graph_breaks, self.path(), backend="eager", dynamic=False
            )

    def test_load_rejects_wrapper_sharing_the_captured_qualname(self):
        # functools.wraps copies __qualname__, so an instrumented wrapper around
        # a different callable passes the qualname check while being a different
        # code object. Installing there serves the captured graphs for the
        # wrapper's frame.
        session = precompile_capture(
            staged_with_graph_breaks, backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(4, 8))
        session.save(self.path())

        @functools.wraps(staged_with_graph_breaks)
        def instrumented(x):
            return staged_with_local_dict_conditional(x, {"op": "sin", "scale": 2})

        self.assertEqual(
            instrumented.__qualname__, staged_with_graph_breaks.__qualname__
        )
        torch._dynamo.reset()
        with self.assertRaisesRegex(PackageError, "captured from code object"):
            precompile_load(instrumented, self.path(), backend="eager", dynamic=False)

    def test_save_writes_a_single_file(self):
        # save() names a FILE, written exactly as given with parent directories
        # created, and precompile_load takes that same path back. Matches
        # `invariants`; deliberately unlike DiskDynamoStore, whose path is a
        # directory because the transparent cache owns its own layout.
        session = precompile_capture(
            staged_with_graph_breaks, backend="eager", dynamic=False
        )
        with session as compiled:
            compiled(torch.randn(4, 8))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "model.pt")
            session.save(path)
            self.assertTrue(os.path.isfile(path))
            self.assertFalse(os.path.isdir(path))
            torch._dynamo.reset()
            with precompile_load(
                staged_with_graph_breaks, path, backend="eager", dynamic=False
            ) as loaded:
                self.assertEqual(loaded(torch.randn(4, 8)).shape, torch.Size([]))

    def test_save_reports_write_failures_as_package_errors(self):
        # Writing the artifact includes creating its parent, so a parent that
        # is a file has to arrive as a PackageError naming the path the caller
        # passed rather than as a bare FileExistsError naming the parent, and a
        # refused write must leave nothing behind.
        session = precompile_capture(
            staged_with_graph_breaks, backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(4, 8))
        with tempfile.TemporaryDirectory() as tmp:
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
            self.assertEqual(sorted(os.listdir(tmp)), ["adir", "plain_file"])

    def test_load_rejects_a_directory(self):
        # An artifact captured before save() became single-file is a directory
        # holding "entry", so this is what a migration hits. Nothing exercises
        # the branch today.
        stale = self.path("stale_layout.pt")
        os.makedirs(stale, exist_ok=True)
        with self.assertRaisesRegex(PackageError, "is a directory"):
            precompile_load(
                staged_with_graph_breaks, stale, backend="eager", dynamic=False
            )

    def test_resume_names_from_separate_captures_do_not_collide(self):
        # Two artifacts captured in different processes carry the same
        # __resume_at_<offset>_<n> name, and a serving process installs both
        # into one module dict. Without a rename the loser of that write
        # silently runs the winner's continuation -- no error, right shape,
        # wrong numbers.
        fns = (staged_break_then_add_one, staged_break_then_add_thousand)
        x = torch.randn(3, 4)
        expected = [fn(x) for fn in fns]
        self.assertNotEqual(expected[0], expected[1])

        paths = [self.path(fn.__name__ + ".pt") for fn in fns]
        for fn, path in zip(fns, paths):
            torch._dynamo.reset()
            session = precompile_capture(fn, backend="eager", dynamic=False)
            with session as compiled:
                compiled(x)
            session.save(path)

        shared_name = _resume_names_in(paths[0])
        self.assertEqual(len(shared_name), 1)
        other_name = _resume_names_in(paths[1])
        self.assertNotEqual(shared_name, other_name)
        _rename_resume_function(paths[1], other_name[0], shared_name[0])
        self.assertEqual(_resume_names_in(paths[1]), shared_name)

        torch._dynamo.reset()
        with contextlib.ExitStack() as stack:
            loaded = [
                stack.enter_context(
                    precompile_load(fn, path, backend="eager", dynamic=False)
                )
                for fn, path in zip(fns, paths)
            ]
            # Both continuations have to be reachable at once. Pre-fix the
            # second load overwrote the first's binding and both callables
            # returned the second artifact's answer.
            stack.enter_context(serving())
            for call, want in zip(loaded, expected):
                self.assertEqual(call(x), want)

    def test_unload_keeps_globals_a_bystander_compile_needs(self):
        # A serving process shares a module namespace with plain torch.compile.
        # Import aliases are minted from the module name, so both writers pick
        # the same one; popping it on unload takes it out from under whatever
        # else in the module resolved it.
        session = precompile_capture(
            staged_with_graph_breaks, backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        session.save(self.path())

        torch._dynamo.reset()
        scope = staged_with_graph_breaks.__globals__
        # Other tests in this file compiled functions from this module and left
        # their aliases here. Drop them for the state a serving process starts
        # in, where the load is what installs them.
        for name in [n for n in scope if n.startswith("__import_")]:
            del scope[name]

        x = torch.randn(3, 4)
        with torch.no_grad():
            expected = staged_with_global_function_ref(x)
            loaded = precompile_load(
                staged_with_graph_breaks, self.path(), backend="eager", dynamic=False
            )
            bystander = torch.compile(
                staged_with_global_function_ref, backend="eager", dynamic=False
            )
            self.assertEqual(bystander(x), expected)
            loaded.unload()
            self.assertEqual(bystander(x), expected)

    def test_unload_removes_the_builtins_key_install_added(self):
        session = precompile_capture(
            staged_with_graph_breaks, backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        session.save(self.path())

        torch._dynamo.reset()
        scope = staged_with_graph_breaks.__globals__
        # A serving process that never compiled this module holds no
        # builtins-dict key, so install() is the one that creates it. Capturing
        # in this same process leaves one behind, so drop it to get there.
        for name in [n for n in scope if n.startswith("__builtins_dict__")]:
            del scope[name]
        before = set(scope)

        loaded = precompile_load(
            staged_with_graph_breaks, self.path(), backend="eager", dynamic=False
        )
        with torch.no_grad(), serving():
            loaded(torch.randn(3, 4))
        self.assertTrue(any(n.startswith("__builtins_dict__") for n in scope))

        loaded.unload()
        # The key carries the capture process's unique_id counter, so a leaked
        # one collides with the first local compile that mints the same name,
        # which then dies in CleanupHook.create. Import aliases are the one
        # thing install() is expected to leave: plain torch.compile installs
        # them permanently too.
        leftover = sorted(
            name for name in set(scope) - before if not name.startswith("__import_")
        )
        self.assertEqual(leftover, [])

    def test_a_failed_install_leaves_nothing_installed(self):
        from torch._C._dynamo.eval_frame import _debug_get_precompile_entries

        session = precompile_capture(
            staged_with_graph_breaks, backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        session.save(self.path())

        torch._dynamo.reset()
        cache_entry = _SingleFileStore().load_cache_entry(self.path())
        backends = cache_entry.backends

        class _Boom:
            def after_deserialization(self):
                raise RuntimeError("artifact will not deserialize on this host")

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
        with self.assertRaisesRegex(RuntimeError, "will not deserialize"):
            package.install(backends)

        # install() raising leaves the caller no handle to unload with, so a
        # partial install would be permanent: some frames served, some not.
        leftover = sorted(
            name for name in set(scope) - before if not name.startswith("__import_")
        )
        self.assertEqual(leftover, [])
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

        torch._dynamo.reset()
        scope = PrecompileSelfAct.forward.__globals__
        before = set(scope)
        first = precompile_load(
            PrecompileSelfAct(torch.relu), self.path(), backend="eager", dynamic=False
        )
        installed = {name: scope[name] for name in set(scope) - before}
        second = precompile_load(
            PrecompileSelfAct(torch.relu), self.path(), backend="eager", dynamic=False
        )
        rebound = {
            name: scope[name]
            for name, value in installed.items()
            if scope.get(name) is not value
        }
        self.assertTrue(rebound)

        first.unload()
        for name, value in rebound.items():
            self.assertIs(scope.get(name), value)
        second.unload()
        # And unloading in load order still has to leave the namespace clean.
        # The values `first` installed were orphaned the moment `second` rebound
        # them, so putting them back here would leave a compiled backend bound
        # in the module for the life of the process, one set per load.
        leftover = sorted(
            name for name in set(scope) - before if not name.startswith("__import_")
        )
        self.assertEqual(leftover, [])


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
