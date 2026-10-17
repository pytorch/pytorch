# Owner(s): ["module: dynamo"]

import collections
import contextlib
import dataclasses
import functools
import gc
import importlib
import inspect
import os
import pickle
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.onnx.operators
import torch.utils.cpp_extension
from torch._C._dynamo.eval_frame import _debug_get_precompile_entries
from torch._dynamo.exc import PackageError, Unsupported
from torch._dynamo.guards import CheckFunctionManager
from torch._dynamo.package import (
    _collapse_device_types,
    _lookup_code,
    CompilePackage,
    DiskDynamoStore,
    DynamoCache,
    load_guards_state,
)
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.precompile_package import default_guard_filter_fn
from torch._dynamo.symbolic_convert import _import_module, UNNAMED_SCOPE_PREFIX
from torch._dynamo.testing import reduce_to_scalar_loss
from torch._dynamo.utils import CleanupManager
from torch._functorch import config as functorch_config
from torch._functorch._aot_autograd.aot_autograd_result import GenericAOTAutogradResult
from torch._functorch._aot_autograd.runtime_wrappers import AOTDispatchAutograd
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
    TEST_WITH_TORCHDYNAMO,
)
from torch.utils._triton import has_triton


def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def hooked_forward(x, w):
    y = x.sin() * w
    torch._dynamo.graph_break()
    return y.cos().sum()


# Inlineable (GraphModule) saved tensors hooks whose pack is lossy, so packing a
# saved tensor twice changes the gradients.
def doubling_pack(x):
    return x * 2


def identity_unpack(x):
    return x


INLINEABLE_HOOKS = (
    torch.fx.symbolic_trace(doubling_pack),
    torch.fx.symbolic_trace(identity_unpack),
)


def compute_loss_helper(x):
    return reduce_to_scalar_loss(x)


def compiled_region_with_backend_id_for_package_test():
    return __compiled_fn_0_00000000_0000_0000_0000_000000000000()  # noqa: F821


PKG_SCALE = 3

_TENSOR_FREE_RESUME_STATE = {"value": 1}
# Called through an alias so the continuation below names no torch module and
# has_tensor_in_frame treats it as tensor-free.
_tensor_free_resume_ones = torch.ones


@torch.compiler.disable
def _tensor_free_resume_value():
    return _TENSOR_FREE_RESUME_STATE["value"]


def _tensor_free_continuation(y):
    value = _tensor_free_resume_value()
    # Only the int crosses the graph break.
    return _tensor_free_resume_ones(3) * value


def _tensor_free_resume_fn(x):
    y = x.sin()
    return y.cos(), _tensor_free_continuation(y)


def _empty_graph_continuation(y):
    step = _tensor_free_resume_value()
    return step + 1


def _empty_graph_resume_fn(x):
    y = x.sin()
    return y.cos(), _empty_graph_continuation(y)


@torch.compiler.disable
def _tensor_or_int(y):
    return y if _TENSOR_FREE_RESUME_STATE["value"] else 3


def _resume_maybe_tensor_free_fn(x):
    y = _tensor_or_int(x.sin())
    return y + 1


def _skip_closure_match(entries):
    # The torch.compiler.disable wrapper's CLOSURE_MATCH cannot be serialized.
    return [e.guard_type != "CLOSURE_MATCH" for e in entries]


class UnpicklableConfig:
    # Like ConfigThatCannotPickle, but the failure is not an AttributeError.
    scale = 2.0

    def __reduce__(self):
        raise RuntimeError("config cannot pickle")


def _bound_method_guard_target(self, x):
    return x


@functools.wraps(_bound_method_guard_target)
def _bound_method_guard_wrapper(self, x):
    return x * 3


class _BoundMethodGuardRecv:
    pass


class BoundMethodNameGuardModule(torch.nn.Module):
    # self.cb binds an fqn-mismatched function; the guard reads through its
    # __func__. Unless that function is seeded into guard_tree_values when the
    # method's guard manager is built, it is pruned to an fqn-mismatch _Missing
    # and the __name__ guard AttributeErrors at load (that reproduces with
    # self.cb alone). self.other pins the seed SITE: it reaches the same
    # function first, so a seed placed in the reducer, when it finally sees the
    # method, is too late -- pickle has memoized the function by then.
    def __init__(self):
        super().__init__()
        self.other = _bound_method_guard_wrapper  # must precede self.cb, see above
        self.cb = types.MethodType(_bound_method_guard_wrapper, _BoundMethodGuardRecv())

    def forward(self, x):
        if self.cb.__name__ == "_bound_method_guard_target":
            x = x + 1
        return x * 2


class ConfigThatCannotPickle:
    scale = 2.0

    def __reduce__(self):
        # AttributeError was the one exception the bypass mapped to a
        # PackageError before #196470 widened it; UnpicklableConfig covers the
        # rest.
        raise AttributeError("config cannot pickle")


class StaticParamModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(3))

    def forward(self, x, use_w=False):
        if use_w:
            return (x * self.w).sin()
        return x.sin()


class _DescriptorCodeOwner:
    @property
    def prop(self):
        def inner():
            return 1

        return inner

    @prop.setter
    def prop(self, value):
        pass

    @prop.deleter
    def prop(self):
        pass

    @functools.cached_property
    def cached(self):
        return 2


class _RaisingClassAttribute:
    def __get__(self, obj, owner=None):
        raise RuntimeError("not available on the class")


class _ClassWithRaisingAttribute:
    # Visited by the module search before _ExecGeneratedCodeOwner.
    value = _RaisingClassAttribute()


@dataclasses.dataclass
class _ExecGeneratedCodeOwner:
    x: int = 0

    @property
    def doubled(self):
        return self.x * 2


def _exec_without_renaming():
    namespace = {}
    exec(
        "def __create_fn__():\n    def f(self):\n        return 1\n    return f\n",
        globals(),
        namespace,
    )
    return namespace["__create_fn__"]()


class _UnrenamedExecOwner:
    # exec()'d like a dataclass __init__, but its __qualname__ is left as
    # "__create_fn__.<locals>.f", which is not a path from this module.
    f = _exec_without_renaming()


def _exec_function():
    namespace = {}
    exec("def g(self):\n    return 1\n", globals(), namespace)
    return namespace["g"]


def _closureless():
    return None


def _hold(fn):
    def holder():
        return fn

    # Found through holder's closure, but the recorded qualname leads to a
    # function with no closure, so replaying the path raises PackageError.
    holder.__qualname__ = "_closureless"
    return holder


_mislabeled_holder = _hold(_exec_function())


def _package_op(x):
    return x + 1


@dataclasses.dataclass
class _GraphBreakingDataclass:
    x: torch.Tensor

    def __post_init__(self):
        torch._dynamo.graph_break()


def _make_generated_method():
    def method(self):
        return 1

    method.__qualname__ = "_GeneratedMethodOwner.method"
    return method


def _make_generated_property():
    def prop(self):
        return 1

    prop.__qualname__ = "_GeneratedMethodOwner.prop"
    return property(prop)


class _GeneratedMethodOwner:
    method = _make_generated_method()
    prop = _make_generated_property()


def _make_stale_method():
    def method(self):
        return 1

    return method


class _StaleMethodOwner:
    method = _make_stale_method()


def _unwrapped_decorator(f):
    def wrapper(*args):
        return f(*args)

    return wrapper


@_unwrapped_decorator
def _decorated_without_wraps(x):
    return x


# The co_qualnames, _make_*.<locals>.*, now name nothing in this module; only
# the rewritten __qualname__s on _GeneratedMethodOwner do.
del _make_generated_method, _make_generated_property, _make_stale_method


# A dynamic dim on a module-level tensor is what makes a SHAPE_ENV guard read a
# global -- as a literal G['PKG_DYN_ROWS'] inside a Python lambda by default.
PKG_DYN_ROWS = torch.randn(4, 3)
torch._dynamo.mark_dynamic(PKG_DYN_ROWS, 0)

# namedtuple builds __new__ with exec into a namespace no module owns.
PkgRecord = collections.namedtuple("PkgRecord", ("value", "weight"))

PKG_LOG_SCOPE = {"log": []}
exec("def scale(self, x):\n    log.append(1)\n    return x * 3\n", PKG_LOG_SCOPE)


class PkgLogger:
    scale = PKG_LOG_SCOPE["scale"]


PKG_LOGGER = PkgLogger()

PKG_FACTOR_SCOPE = {"factor": 3}
exec("def scale(self, x):\n    return x * factor\n", PKG_FACTOR_SCOPE)


class PkgScaler:
    scale = PKG_FACTOR_SCOPE["scale"]


PKG_SCALER = PkgScaler()

# scale reaches helper through the namespace itself, so helper's source is rooted
# at the unnamed scope key that stands for it.
PKG_CHAIN_SCOPE = {"factor": 3}
exec(
    "def helper(x):\n    return x * factor\n"
    "helpers = [helper]\n"
    "def scale(self, x):\n    return helpers[0](x)\n",
    PKG_CHAIN_SCOPE,
)


class PkgChainScaler:
    scale = PKG_CHAIN_SCOPE["scale"]


PKG_HANDLER_SCOPE = {"factor": 3}
exec("def scale(x):\n    return x * factor\n", PKG_HANDLER_SCOPE)
PKG_HANDLERS = [PKG_HANDLER_SCOPE["scale"]]
PKG_HANDLER_SET = {PKG_HANDLER_SCOPE["scale"]}


def nested_break_sources(prefix, shape):
    """Modules, in import order, whose last one's ``fn`` reaches a graph break
    inlined from another module: ``flat`` breaks in the helper, ``closure`` in a
    closure the helper defines, ``two_levels`` in a third module the helper calls.
    """
    helper, inner = f"{prefix}_helper", f"{prefix}_inner"
    breaking = (
        "    y = y.sin() + OFF\n"
        "    torch._dynamo.graph_break()\n"
        "    return y.cos() * OFF\n"
    )
    sources = {}
    if shape == "flat":
        sources[helper] = f"import torch\n\nOFF = 3.0\n\ndef helper(y):\n{breaking}"
    elif shape == "closure":
        sources[helper] = (
            "import torch\n\n"
            "def helper(y):\n"
            "    OFF = y.shape[0] + 1.0\n\n"
            "    def inner(y):\n"
            "        y = y.sin() + OFF\n"
            "        torch._dynamo.graph_break()\n"
            "        return y.cos() * OFF\n\n"
            "    return inner(y) - OFF\n"
        )
    else:
        sources[inner] = f"import torch\n\nOFF = 3.0\n\ndef inner(y):\n{breaking}"
        sources[helper] = (
            f"import {inner}\n\nSCALE = 2.0\n\n"
            f"def helper(y):\n    return {inner}.inner(y * SCALE) + SCALE\n"
        )
    sources[f"{prefix}_entry"] = (
        f"import {helper}\n\ndef fn(x):\n    return {helper}.helper(x * 2) + 1\n"
    )
    return sources


@functorch_config.patch("bundled_autograd_cache", True)
@torch._dynamo.config.patch({"strict_precompile": True})
@instantiate_parametrized_tests
class TestPackage(torch._inductor.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

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

    def test_collapse_device_types_prefers_an_accelerator(self):
        # The single string both callers record. Among several accelerators
        # one SystemInfo.check_compatibility checks wins: alphabetical order
        # would record "mps" for {"mps", "xpu"}, and a name outside CHECK_GPUS
        # skips every host check the way the old "cpu" did.
        self.assertEqual(_collapse_device_types(frozenset()), "cpu")
        self.assertEqual(_collapse_device_types(frozenset(("cpu",))), "cpu")
        self.assertEqual(_collapse_device_types(frozenset(("cpu", "cuda"))), "cuda")
        self.assertEqual(_collapse_device_types(frozenset(("cuda", "xpu"))), "cuda")
        self.assertEqual(_collapse_device_types(frozenset(("mps", "xpu"))), "xpu")
        self.assertEqual(_collapse_device_types(frozenset(("hpu", "mps"))), "hpu")

    def test_code_source_walk_skips_non_code_constants(self):
        # Python 3.10 has no co_qualname, so the walk visits every constant of the
        # module's functions. b"" and 2**61 - 1 both hash to 0, and putting both in
        # one set compares them: a BytesWarning, which CI's python -bb raises. (0
        # would too, but newer Pythons keep small ints out of co_consts.)
        source = """
def f():
    a = b""
    b = 2305843009213693951

    class C:
        def g(self):
            return a, b

    return C
"""
        script = """
import sys
from unittest import mock

import torch._dynamo.package as package
import bbmod

code = bbmod.f().g.__code__
with mock.patch.object(package.sys, "version_info", (3, 10, 0)):
    name, path = package._get_code_source(code, bbmod)
print(eval(f"bbmod.{name}.{path}") is code)
"""
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "bbmod.py"), "w") as f:
                f.write(source)
            out = subprocess.run(
                [sys.executable, "-bb", "-c", script],
                cwd=d,
                capture_output=True,
                text=True,
            )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "True")

    def test_code_source_resolves_through_descriptors(self):
        # getattr on the class returns the descriptor, not the function it wraps.
        owner = vars(_DescriptorCodeOwner)
        prop = owner["prop"]
        getter = prop.fget.__code__
        inner = next(c for c in getter.co_consts if inspect.iscode(c))
        # The setter and deleter share the getter's qualname, so they are only
        # found after the search through fget comes back empty.
        codes = (getter, inner, prop.fset.__code__, prop.fdel.__code__)
        for code in (*codes, owner["cached"].func.__code__):
            package = CompilePackage(None)
            with package.code_context(code):
                pass
            self.assertIs(_lookup_code(package._codes[code]), code)

    def test_code_source_resolves_exec_generated_code_from_globals(self):
        # dataclasses exec() __init__, so its code names no source file and its
        # co_qualname is synthetic; only the frame's globals name its module.
        init = _ExecGeneratedCodeOwner.__init__
        code = init.__code__
        self.assertIsNone(inspect.getmodule(code))
        package = CompilePackage(None)
        with package.code_context(code, init.__globals__):
            pass
        entry = package._codes[code]
        self.assertIs(sys.modules[entry.python_module], inspect.getmodule(init))
        self.assertIs(_lookup_code(entry), code)

    def test_code_source_refuses_exec_generated_code_it_cannot_replay(self):
        fn = _UnrenamedExecOwner.f
        self.assertIsNone(inspect.getmodule(fn.__code__))
        package = CompilePackage(None)
        with self.assertRaisesRegex(PackageError, "Cannot resolve"):
            with package.code_context(fn.__code__, fn.__globals__):
                pass
        self.assertNotIn(fn.__code__, package._codes)

    def test_code_source_refuses_a_path_whose_replay_raises(self):
        fn = _mislabeled_holder()
        self.assertIsNone(inspect.getmodule(fn.__code__))
        package = CompilePackage(None)
        with self.assertRaisesRegex(
            PackageError, r"'_closureless' does not resolve back to it.*fails with"
        ) as cm:
            with package.code_context(fn.__code__, fn.__globals__):
                pass
        self.assertIsInstance(cm.exception.__cause__, PackageError)
        self.assertNotIn(fn.__code__, package._codes)

    def test_code_source_keeps_the_codes_own_module_over_its_globals(self):
        code = _closureless.__code__
        package = CompilePackage(None)
        with package.code_context(code, vars(functools)):
            pass
        entry = package._codes[code]
        self.assertEqual(entry.python_module, __name__)
        self.assertIs(_lookup_code(entry), code)

    def test_code_source_search_skips_class_attributes_that_raise(self):
        with self.assertRaises(RuntimeError):
            _ClassWithRaisingAttribute.value
        init = _ExecGeneratedCodeOwner.__init__
        package = CompilePackage(None)
        with package.code_context(init.__code__, init.__globals__):
            pass
        self.assertIs(_lookup_code(package._codes[init.__code__]), init.__code__)

    def test_dataclass_init_frame_saves_and_loads(self):
        # The graph break under the constructor makes the exec()'d __init__ a
        # frame of its own, which convert_frame records with its globals.
        ctx = DiskDynamoStore()

        def fn(x):
            return _GraphBreakingDataclass(x + 1).x * 2

        def guard_filter_fn(guards):
            # Identity guards on the globals are beside the point here.
            unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
            return [guard.guard_type not in unsupported for guard in guards]

        x = torch.randn(3)
        package = CompilePackage(fn)
        compiled_fn = torch._dynamo.optimize(
            "eager", package=package, guard_filter_fn=guard_filter_fn
        )(fn)
        expected = compiled_fn(x)
        self.assertIn(_GraphBreakingDataclass.__init__.__code__, package._codes)
        for backend_id, backend in package.cached_backends.items():
            ctx.record_eager_backend(backend_id, backend)
        ctx.save_package(package, self.path())

        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            package, backends = ctx.load_package(fn, self.path())
            compiled_fn = torch._dynamo.optimize(
                package=package, guard_filter_fn=guard_filter_fn
            )(fn)
            package.install(backends)
            self.assertEqual(expected, compiled_fn(x))

    def test_module_function_guard_saves_unfiltered_and_reports_a_miss(self):
        # No guard_filter_fn: the CLOSURE_MATCH on _package_op is still saved,
        # as a FUNCTION_CODE_MATCH, and a rebinding names it in the refusal.
        ctx = DiskDynamoStore()

        def fn(x):
            return _package_op(x) * 2

        x = torch.randn(3)
        package = CompilePackage(fn)
        compiled_fn = torch._dynamo.optimize("eager", package=package)(fn)
        expected = compiled_fn(x)
        for backend_id, backend in package.cached_backends.items():
            ctx.record_eager_backend(backend_id, backend)
        ctx.save_package(package, self.path())

        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            package, backends = ctx.load_package(fn, self.path())
            compiled_fn = torch._dynamo.optimize(package=package)(fn)
            package.install(backends)
            self.assertEqual(expected, compiled_fn(x))
            with patch.dict(globals(), {"_package_op": lambda x: x - 1}):
                with self.assertRaisesRegex(RuntimeError, "___check_function_code"):
                    compiled_fn(x)

    def test_code_source_searches_the_module_when_the_qualname_is_stale(self):
        # Python >= 3.11 walks co_qualname from the module and used to refuse
        # the code at the first missing part, while < 3.11 searches the module.
        for name, code in [
            ("method", _GeneratedMethodOwner.method.__code__),
            ("prop", _GeneratedMethodOwner.prop.fget.__code__),
        ]:
            package = CompilePackage(None)
            with package.code_context(code):
                pass
            entry = package._codes[code]
            self.assertEqual(entry.function_names, [f"_GeneratedMethodOwner.{name}"])
            self.assertIs(_lookup_code(entry), code)

    def test_code_source_refuses_a_stale_qualname_the_search_cannot_fix(self):
        # The search finds _StaleMethodOwner.method, but its __qualname__ is as
        # stale as co_qualname, so recording it would only fail at load.
        code = _StaleMethodOwner.method.__code__
        package = CompilePackage(None)
        with self.assertRaisesRegex(
            PackageError, r"__qualname__ '_make_stale_method\.<locals>\.method'"
        ):
            with package.code_context(code):
                pass

    def test_code_source_refuses_a_decorator_wrapper_without_wraps(self):
        # On Python >= 3.11 the co_qualname walk reaches the wrapper and finds
        # the code in its closure, recording the wrapper's stale __qualname__.
        code = _decorated_without_wraps.__closure__[0].cell_contents.__code__
        package = CompilePackage(None)
        with self.assertRaisesRegex(
            PackageError,
            r"__qualname__ '_unwrapped_decorator\.<locals>\.wrapper'",
        ):
            with package.code_context(code):
                pass

    def test_package_records_the_devices_a_graph_names(self):
        # The recording side of the scan, which is what the artifact carries. A
        # stand-in for a dynamic-shape cuda capture, whose first meta value is a
        # SymInt with no device, has to record cuda: reading the first leaf
        # recorded "cpu", which skips every GPU check at load. Both graphs are
        # fake and never run, so this needs no accelerator.
        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env):
            cuda = torch.empty(2, device="cuda")
            s0 = shape_env.create_unbacked_symint()
            meta = torch.empty(2, device="meta")

        def fn(x):
            return x + 1

        graph = torch.fx.Graph()
        graph.placeholder("s0").meta["val"] = s0
        x = graph.placeholder("x")
        x.meta["val"] = cuda
        graph.call_function(torch.ops.aten.add.Tensor, (x, 1)).meta["val"] = cuda

        package = CompilePackage(fn)
        # A package that has scanned no graph starts at cpu, so the flip below
        # is this scan's answer rather than that initial value.
        self.assertEqual(package.cache_entry().device_type, "cpu")
        package.update_device_type(graph)
        self.assertEqual(package.cache_entry().device_type, "cuda")

        meta_graph = torch.fx.Graph()
        meta_graph.placeholder("x").meta["val"] = meta
        package = CompilePackage(fn)
        package.update_device_type(meta_graph)
        # Dropping meta leaves no device named, which reads as cpu rather than
        # as no answer.
        self.assertEqual(package.cache_entry().device_type, "cpu")

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_package_keeps_a_device_a_later_frame_does_not_name(self):
        # update_device_type runs once per compiled frame and a package spans
        # frames, so recording only the last answer let the cpu-only resume
        # frame of this cuda compile erase the cuda the first frame named. The
        # input is fake, so the compile needs no accelerator.
        def fn(x):
            _y = x.sin()
            torch._dynamo.graph_break()
            return torch.ones(2) + 1

        with FakeTensorMode(allow_non_fake_inputs=True):
            torch.compile(fn, backend="eager")(torch.randn(3, 2, device="cuda"))

        (entry,) = PrecompileContext._dynamo_cache_entries.values()
        names = [n for code in entry.codes for n in code.function_names]
        self.assertTrue(any("resume" in n for n in names))
        self.assertEqual(entry.device_type, "cuda")

    def test_package_keeps_a_loaded_device_a_recompile_does_not_name(self):
        # A package rebuilt from a saved entry keeps the entry's codes, so its
        # union has to start from the device those codes recorded: started at
        # frozenset(), one cpu-only recompile after a reload re-snapshotted the
        # entry as "cpu" and the cuda code still in it lost its GPU load check.
        # The graphs are fake and never run; is_available is patched so
        # check_versions accepts the cuda entry on a host without one. Save and
        # reload share the patch so both SystemInfo snapshots see the same
        # accelerator state.
        with FakeTensorMode():
            cuda = torch.empty(2, device="cuda")
            cpu = torch.empty(2)

        def fn(x):
            return x + 1

        cuda_graph = torch.fx.Graph()
        cuda_graph.placeholder("x").meta["val"] = cuda
        cpu_graph = torch.fx.Graph()
        cpu_graph.placeholder("x").meta["val"] = cpu

        package = CompilePackage(fn)
        package.update_device_type(cuda_graph)
        with patch.object(torch.cuda, "is_available", return_value=True):
            saved = pickle.loads(pickle.dumps(package.cache_entry()))
            self.assertEqual(saved.device_type, "cuda")
            package = CompilePackage(fn, dynamo=saved)
        self.assertEqual(package.cache_entry().device_type, "cuda")
        package.update_device_type(cpu_graph)
        self.assertEqual(package.cache_entry().device_type, "cuda")

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

    def test_bypass_drops_only_the_current_compiles_backend(self):
        # A bypass discards what the compile that failed to serialize registered
        # on its entry, not what earlier compiles of the same code object did.
        def fn(x):
            return x + 1

        first_code = compiled_region_with_backend_id_for_package_test.__code__
        (first_id,) = first_code.co_names
        second_id = "__compiled_fn_1_00000000_0000_0000_0000_000000000000"
        package = CompilePackage(fn)
        with package.code_context(fn.__code__):
            package.add_guarded_code(b"", first_code)
        with package.code_context(fn.__code__):
            package.add_backend_id(second_id, object())
            package.bypass_current_compile()

        entry = package.cache_entry().codes[0]
        self.assertFalse(entry.bypassed)
        self.assertEqual(entry.backend_ids, [first_id])
        self.assertEqual(len(entry.guarded_codes), 1)
        self.assertNotIn(second_id, package.cached_backends)

    def test_bypass_of_every_compile_marks_the_entry_bypassed(self):
        # With nothing installable the entry must NOT look like a trivial
        # function that install() would skip_code; a later compile that does
        # record a guarded code makes it installable again.
        def fn(x):
            return x + 1

        code = compiled_region_with_backend_id_for_package_test.__code__
        (backend_id,) = code.co_names
        package = CompilePackage(fn)
        with package.code_context(fn.__code__):
            package.add_backend_id(backend_id)
            package.bypass_current_compile()
        entry = package.cache_entry().codes[0]
        self.assertTrue(entry.bypassed)
        self.assertEqual(entry.backend_ids, [])
        with package.code_context(fn.__code__):
            package.add_guarded_code(b"", code)
            # A bypass used to stick to the entry and suppress this too.
            package.add_inlined_source([fn.__code__])
        self.assertFalse(entry.bypassed)
        self.assertEqual(entry.backend_ids, [backend_id])
        self.assertTrue(package.cache_entry().source_info.inlined_sources)

    @parametrize("config_cls", (ConfigThatCannotPickle, UnpicklableConfig))
    @torch._dynamo.config.patch(caching_precompile=True, strict_precompile=False)
    def test_bypassed_guards_keep_the_frames_earlier_variant(self, config_cls):
        # The title path: the second variant guards on a value whose __reduce__
        # raises, so serializing its guards bypasses the compile. The first
        # variant is still saved, installed on reload and hit; the second is
        # traced fresh. On main this tripped `check_fn.guards_state must not be
        # None` in convert_frame; a __reduce__ raising anything other than
        # AttributeError escaped as an internal error.
        def fn(x, cfg=None):
            if cfg is not None:
                return x.sin() * cfg.scale
            return x.sin()

        x = torch.randn(3)
        cfg = config_cls()
        compiled = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(compiled(x), fn(x))
        with self.assertLogs("torch._dynamo", level="WARNING") as logs:
            self.assertEqual(compiled(x, cfg), fn(x, cfg))
        self.assertTrue(any("config cannot pickle" in line for line in logs.output))
        # The bypassed compile's backend id is referenced by no entry, so the
        # written cache entry carries exactly the surviving compile's backend.
        compiles = torch._dynamo.utils.counters["frames"]["total"]
        info = PrecompileContext.save_to_dynamo_cache()
        (entry,) = info["dynamo"]
        self.assertEqual(len(entry["backend_ids"]), 1)
        written = DynamoCache.load(fn)
        self.assertEqual(list(written.backends), entry["backend_ids"])
        torch._dynamo.reset()
        PrecompileContext.clear()
        # Wrapping reloads from the on-disk DynamoCache (per-test fresh_cache dir).
        compiled = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(len(_debug_get_precompile_entries(fn.__code__)), 1)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled(x), fn(x))
        # Deliberate: re-triggers the bypass in the loading process against an
        # entry that already holds one installed guarded code. The installed
        # variant served the first call; only this one compiled (counted by
        # actual compiles, since FRAME_COUNTER also advances for an installed
        # entry and is zeroed by reset()).
        with self.assertLogs("torch._dynamo", level="WARNING") as logs:
            self.assertEqual(compiled(x, cfg), fn(x, cfg))
        self.assertTrue(any("config cannot pickle" in line for line in logs.output))
        self.assertEqual(torch._dynamo.utils.counters["frames"]["total"], compiles + 1)

    @torch._dynamo.config.patch(
        caching_precompile=True, strict_precompile=False, prepare_freezing=True
    )
    def test_bypass_before_guards_keeps_the_frames_earlier_variant(self):
        # A bypass raised before guards are built (a graph holding a named
        # parameter under prepare_freezing) drops only that compile. It also
        # pins that convert_frame reads the package off the output graph, which
        # the bypass cleared: reading its own local instead records the bypassed
        # compile's guarded code and a backend id nothing cached, and the save
        # then fails or drops the whole frame.
        mod = StaticParamModule()
        torch._dynamo.mark_static_address(mod.w, guard=False)
        x = torch.randn(3)
        compiled = torch.compile(mod)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(compiled(x), mod(x))
        with self.assertLogs("torch._dynamo", level="WARNING") as logs:
            self.assertEqual(compiled(x, use_w=True), mod(x, use_w=True))
        self.assertTrue(any("named parameters" in line for line in logs.output))
        (entry,) = PrecompileContext.save_to_dynamo_cache()["dynamo"]
        self.assertEqual(len(entry["backend_ids"]), 1)
        torch._dynamo.reset()
        PrecompileContext.clear()
        compiled = torch.compile(mod)  # noqa: UNSPECIFIED_BACKEND
        code = StaticParamModule.forward.__code__
        self.assertEqual(len(_debug_get_precompile_entries(code)), 1)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled(x), mod(x))

    def _hooked_grads(self, fn, under_hooks):
        x = torch.linspace(0.1, 1.0, 4, requires_grad=True)
        w = torch.linspace(1.0, 2.0, 4, requires_grad=True)
        if under_hooks:
            with torch.autograd.graph.saved_tensors_hooks(*INLINEABLE_HOOKS):
                fn(x, w).backward()
        else:
            fn(x, w).backward()
        return x.grad, w.grad

    def _save_and_load_hooked_forward(
        self, guard_filter_fn=None, compile_under_hooks=True
    ):
        ctx = DiskDynamoStore()
        package = CompilePackage(hooked_forward)
        compiled_fn = torch._dynamo.optimize(
            "inductor", package=package, guard_filter_fn=guard_filter_fn
        )(hooked_forward)
        self._hooked_grads(compiled_fn, under_hooks=compile_under_hooks)
        expected = self._hooked_grads(compiled_fn, under_hooks=True)
        ctx.save_package(package, self.path())

        torch._dynamo.reset()
        # Load outside any hooks context, the way setup code does.
        package, backends = ctx.load_package(hooked_forward, self.path())
        compiled_fn = torch._dynamo.optimize(
            package=package, guard_filter_fn=guard_filter_fn
        )(hooked_forward)
        package.install(backends)
        return compiled_fn, expected

    def test_loaded_graphs_do_not_rerun_inlined_saved_tensors_hooks(self):
        # The runtime wrappers read the ambient hooks and autocast state, so
        # loading must leave them to the first call.
        built = []
        apply = GenericAOTAutogradResult._apply_runtime_wrappers

        def recording_apply(result, *args):
            built.append(result)
            return apply(result, *args)

        with patch.object(
            GenericAOTAutogradResult, "_apply_runtime_wrappers", recording_apply
        ):
            compiled_fn, expected = self._save_and_load_hooked_forward()
            self.assertEqual(built, [])
            with torch.compiler.set_stance("fail_on_recompile"):
                grads = self._hooked_grads(compiled_fn, under_hooks=True)
        self.assertEqual(len(built), 2)
        self.assertEqual(grads, expected)

    @staticmethod
    def _drop_hooks_guard(guards):
        # Without its hooks guard, one loaded graph serves calls with and
        # without the hooks.
        return [g.guard_type != "AUTOGRAD_SAVED_TENSORS_HOOKS" for g in guards]

    def test_a_first_call_outside_the_hooks_still_disables_them_later(self):
        compiled_fn, expected = self._save_and_load_hooked_forward(
            self._drop_hooks_guard
        )
        with torch.compiler.set_stance("fail_on_recompile"):
            self._hooked_grads(compiled_fn, under_hooks=False)
            grads = self._hooked_grads(compiled_fn, under_hooks=True)
        self.assertEqual(grads, expected)

    def test_a_graph_compiled_without_hooks_keeps_them_on_a_first_call_under_them(
        self,
    ):
        # Nothing was inlined, so the eager hooks must run, as they did in the
        # capturing process, even though the wrappers are built under them.
        compiled_fn, expected = self._save_and_load_hooked_forward(
            self._drop_hooks_guard, compile_under_hooks=False
        )
        self.assertNotEqual(self._hooked_grads(hooked_forward, False), expected)
        with torch.compiler.set_stance("fail_on_recompile"):
            grads = self._hooked_grads(compiled_fn, under_hooks=True)
        self.assertEqual(grads, expected)

    def test_a_loaded_graph_disables_the_autocast_it_was_compiled_under(self):
        # disable_amp is read from the ambient autocast state when the runtime
        # wrappers are built, and the graph is loaded outside autocast.
        disable_amp = []
        post_compile = AOTDispatchAutograd.post_compile

        def recording_post_compile(spec):
            disable_amp.append(spec.disable_amp)
            return post_compile(spec)

        def run(fn):
            x = torch.linspace(0.1, 1.0, 4, requires_grad=True)
            w = torch.linspace(1.0, 2.0, 4, requires_grad=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                fn(x, w).backward()

        ctx = DiskDynamoStore()
        package = CompilePackage(hooked_forward)
        run(torch._dynamo.optimize("inductor", package=package)(hooked_forward))
        ctx.save_package(package, self.path())

        torch._dynamo.reset()
        with patch.object(AOTDispatchAutograd, "post_compile", recording_post_compile):
            package, backends = ctx.load_package(hooked_forward, self.path())
            compiled_fn = torch._dynamo.optimize(package=package)(hooked_forward)
            package.install(backends)
            self.assertEqual(disable_amp, [])
            with torch.compiler.set_stance("fail_on_recompile"):
                run(compiled_fn)
        self.assertEqual(disable_amp, [True, True])

    def test_installed_shape_guard_on_a_global_reads_the_live_module_dict(self):
        # install() roots the guards at sys.modules[...].__dict__, and a package
        # keeps every guard, so the serialized scope holds the global as a
        # FakeTensor with the traced sizes. A SHAPE_ENV lambda over that scope
        # agreed with the artifact whatever the module bound: a one-row global
        # was served the graph built for 2 <= rows. The lambda has to read the
        # same dict the C++ tree does. Fresh tensors on both arms, since the
        # loaded TENSOR_MATCH rejects the marked original.
        ctx = DiskDynamoStore()

        def fn(x):
            return x * 2 + PKG_DYN_ROWS.sum(0)

        x = torch.randn(3)
        module_dict = sys.modules[__name__].__dict__
        self.addCleanup(module_dict.__setitem__, "PKG_DYN_ROWS", PKG_DYN_ROWS)

        package = CompilePackage(fn)
        compiled_fn = torch._dynamo.optimize(backend="eager", package=package)(fn)
        compiled_fn(x)
        for backend_id, backend in package.cached_backends.items():
            ctx.record_eager_backend(backend_id, backend)
        ctx.save_package(package, self.path())

        torch._dynamo.reset()
        package, backends = ctx.load_package(fn, self.path())
        compiled_fn = torch._dynamo.optimize(package=package)(fn)
        package.install(backends)
        with torch.compiler.set_stance("fail_on_recompile"):
            module_dict["PKG_DYN_ROWS"] = torch.randn(7, 3)
            self.assertEqual(fn(x), compiled_fn(x))

            module_dict["PKG_DYN_ROWS"] = torch.randn(1, 3)
            # The stance message dumps the whole tree, LAMBDA_GUARD line included;
            # only the failed parts follow verbose_code_parts=.
            failed_part = r"verbose_code_parts=\[\"2 <= G\['PKG_DYN_ROWS'\]"
            with self.assertRaisesRegex(RuntimeError, failed_part):
                compiled_fn(x)

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

    def _save_record_package(self, ctx):
        # Inlining PkgRecord.__new__ roots guards at the ___unnamed_scope_<id>_c<n>
        # key Dynamo mints for its globals. The key embeds an id() from the
        # compiling process, so a loading process has no binding for it and every
        # call misses on a KeyError unless install() seeds it.
        def fn(x):
            record = PkgRecord(x.sin(), 2)
            return record.value * record.weight

        return self._save_unnamed_scope_package(ctx, fn)

    def _save_unnamed_scope_package(self, ctx, fn, *args):
        package = CompilePackage(fn)
        torch._dynamo.optimize(
            backend="eager", package=package, guard_filter_fn=default_guard_filter_fn
        )(fn)(*(args or (torch.randn(4),)))
        names = {
            name
            for entry in package.cache_entry().codes
            for guarded in entry.guarded_codes
            for name in load_guards_state(
                guarded.guards_state
            ).output_graph.global_scope
            if name.startswith(UNNAMED_SCOPE_PREFIX)
        }
        self.assertTrue(names)
        for backend_id, backend in package.cached_backends.items():
            ctx.record_eager_backend(backend_id, backend)
        ctx.save_package(package, self.path())
        return fn, names

    def _recorded_unnamed_scope_sources(self, package):
        return {
            name: source
            for entry in package.cache_entry().codes
            for guarded in entry.guarded_codes
            for name, source in load_guards_state(
                guarded.guards_state
            ).output_graph.unnamed_scope_sources.items()
        }

    def _install_unnamed_scope_package(self, ctx, fn, names):
        # Drop the keys the save bound, as a fresh process would not have them.
        torch._dynamo.reset()
        scope = fn.__globals__
        for name in names:
            scope.pop(name, None)
        self.addCleanup(lambda: [scope.pop(name, None) for name in names])
        package, backends = ctx.load_package(fn, self.path())
        compiled_fn = torch._dynamo.optimize(package=package)(fn)
        package.install(backends)
        self.addCleanup(package.uninstall)
        return package, backends, compiled_fn

    def test_install_seeds_unnamed_scope_globals(self):
        ctx = DiskDynamoStore()
        fn, names = self._save_record_package(ctx)
        package, backends, compiled_fn = self._install_unnamed_scope_package(
            ctx, fn, names
        )
        scope = fn.__globals__
        for name in names:
            self.assertIs(scope[name], PkgRecord.__new__.__globals__)
        x = torch.randn(4)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled_fn(x), fn(x))
        package.uninstall()
        self.assertFalse(names & scope.keys())

        # A binding someone else wrote is neither replaced nor removed.
        foreign = {name: {} for name in names}
        scope.update(foreign)
        package.install(backends)
        package.uninstall()
        for name in names:
            self.assertIs(scope.pop(name), foreign[name])

    def test_install_keeps_unnamed_scope_globals_of_a_live_compile(self):
        # The live compile bound the key behind a CleanupHook keyed on its
        # generated code. install() keeps that binding, and the hook firing
        # afterwards must not delete it from under the installed guards.
        ctx = DiskDynamoStore()
        # Hold the code objects already registered, so none is collected and its
        # id reused by a code object of the compile below.
        held = [ref() for ref in list(CleanupManager.instance.refs.values())]
        before = set(CleanupManager.instance.refs)
        fn, names = self._save_record_package(ctx)
        self.addCleanup(lambda: [fn.__globals__.pop(name, None) for name in names])
        # Pin the live compile's code objects so its hooks fire after install().
        pinned = [
            ref()
            for idx, ref in list(CleanupManager.instance.refs.items())
            if idx not in before
        ]
        pinned_ids = {id(p) for p in pinned if p is not None}
        self.assertTrue(pinned_ids)

        torch._dynamo.reset()
        live = {name: fn.__globals__[name] for name in names}
        package, backends = ctx.load_package(fn, self.path())
        compiled_fn = torch._dynamo.optimize(package=package)(fn)
        package.install(backends)
        self.addCleanup(package.uninstall)
        del pinned
        gc.collect()

        self.assertFalse(pinned_ids & set(CleanupManager.instance.refs))
        for name in names:
            self.assertIs(fn.__globals__[name], live[name])
        x = torch.randn(4)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled_fn(x), fn(x))
        # Taken over from the hook, so nothing would drop it if uninstall() didn't.
        package.uninstall()
        self.assertFalse(names & fn.__globals__.keys())
        del held

    def test_install_binds_unnamed_scope_globals_to_the_live_namespace(self):
        # The bytecode replays log.append through the key, so binding the
        # artifact's copy of the namespace would append to the copy.
        def fn(x):
            return PKG_LOGGER.scale(x.sin())

        # Trace against an empty list, and leave it empty for a rerun.
        log = PKG_LOG_SCOPE["log"]
        log.clear()
        self.addCleanup(log.clear)
        ctx = DiskDynamoStore()
        fn, names = self._save_unnamed_scope_package(ctx, fn)
        _, _, compiled_fn = self._install_unnamed_scope_package(ctx, fn, names)
        for name in names:
            self.assertIs(fn.__globals__[name], PKG_LOG_SCOPE)
        # The guards want the list as it was while tracing; the save appended.
        log.clear()
        x = torch.randn(4)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled_fn(x), x.sin() * 3)
        self.assertEqual(log, [1])

    def test_installed_unnamed_scope_guards_check_the_live_namespace(self):
        # EQUALS_MATCH on factor must fail once the namespace changes; checking
        # the artifact's frozen copy would keep serving x * 3.
        def fn(x):
            return PKG_SCALER.scale(x.sin())

        ctx = DiskDynamoStore()
        fn, names = self._save_unnamed_scope_package(ctx, fn)
        _, _, compiled_fn = self._install_unnamed_scope_package(ctx, fn, names)
        x = torch.randn(4)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled_fn(x), x.sin() * 3)
        PKG_FACTOR_SCOPE["factor"] = 5
        self.addCleanup(PKG_FACTOR_SCOPE.__setitem__, "factor", 3)
        self.assertEqual(compiled_fn(x), x.sin() * 5)

    @parametrize("handlers", ["emptied", "rebound"])
    def test_install_leaves_unbound_an_unnamed_scope_whose_source_raises(
        self, handlers
    ):
        # The namespace's source is G['PKG_HANDLERS'][0].__globals__. Against an
        # emptied or rebound list it raises IndexError or TypeError, and
        # install() must skip the key rather than fail partway through, leaving
        # a call that misses.
        def fn(x):
            return PKG_HANDLERS[0](x.sin())

        ctx = DiskDynamoStore()
        fn, names = self._save_unnamed_scope_package(ctx, fn)
        stale = [] if handlers == "emptied" else None
        with patch.object(sys.modules[fn.__module__], "PKG_HANDLERS", stale):
            package, _, compiled_fn = self._install_unnamed_scope_package(
                ctx, fn, names
            )
        self.assertEqual(self._recorded_unnamed_scope_sources(package).keys(), names)
        self.assertFalse(names & fn.__globals__.keys())
        with torch.compiler.set_stance("fail_on_recompile"):
            with self.assertRaisesRegex(RuntimeError, "fail_on_recompile"):
                compiled_fn(torch.randn(4))

    def test_install_binds_an_unnamed_scope_reached_through_a_set(self):
        # The namespace's source goes through Dynamo's ___set_getitem helper,
        # which a bare eval of the source's name does not have.
        def fn(x):
            for handler in PKG_HANDLER_SET:
                return handler(x.sin())

        ctx = DiskDynamoStore()
        fn, names = self._save_unnamed_scope_package(ctx, fn)
        package, _, compiled_fn = self._install_unnamed_scope_package(ctx, fn, names)
        sources = self._recorded_unnamed_scope_sources(package)
        self.assertIn("___set_getitem", "".join(s.name for s in sources.values()))
        for name in names:
            self.assertIs(fn.__globals__[name], PKG_HANDLER_SCOPE)
        x = torch.randn(4)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled_fn(x), x.sin() * 3)

    def test_install_leaves_unbound_an_unnamed_scope_reached_through_a_local(self):
        # The namespace is found through the frame's argument, which install()
        # does not have, so the key stays unbound and the call misses.
        def fn(scaler, x):
            return scaler.scale(x.sin())

        ctx = DiskDynamoStore()
        fn, names = self._save_unnamed_scope_package(
            ctx, fn, PKG_SCALER, torch.randn(4)
        )
        package, _, compiled_fn = self._install_unnamed_scope_package(ctx, fn, names)
        self.assertFalse(self._recorded_unnamed_scope_sources(package))
        self.assertFalse(names & fn.__globals__.keys())
        x = torch.randn(4)
        with torch.compiler.set_stance("fail_on_recompile"):
            with self.assertRaisesRegex(RuntimeError, "fail_on_recompile"):
                compiled_fn(PKG_SCALER, x)
        self.assertEqual(compiled_fn(PKG_SCALER, x), x.sin() * 3)

    def test_install_records_no_source_rooted_at_an_unnamed_scope_key(self):
        # scale is reached through the frame's argument and helper through
        # G[<key>]['helpers'][0]. Recording the latter would bind the key through
        # itself, so neither is recorded and the key stays unbound.
        def fn(scaler, x):
            return scaler.scale(x.sin())

        ctx = DiskDynamoStore()
        fn, names = self._save_unnamed_scope_package(
            ctx, fn, PkgChainScaler(), torch.randn(4)
        )
        package, _, compiled_fn = self._install_unnamed_scope_package(ctx, fn, names)
        self.assertFalse(self._recorded_unnamed_scope_sources(package))
        self.assertFalse(names & fn.__globals__.keys())
        with torch.compiler.set_stance("fail_on_recompile"):
            with self.assertRaisesRegex(RuntimeError, "fail_on_recompile"):
                compiled_fn(PkgChainScaler(), torch.randn(4))

    def test_uninstall_keeps_a_global_it_did_not_bind(self):
        # An aot_compile load seeds the __import_* aliases its kept guards are
        # rooted at into the live module scope those guards then hold BY
        # REFERENCE, and leaves an already-bound name alone. A package that
        # installs the same alias afterwards did not create the binding:
        # deleting it on uninstall() breaks the loaded artifact's guards for
        # good, since nothing re-seeds them.
        alias = "__import_torch_dot_nn_dot_modules_dot_module"
        module_name = "torch.test_package_alias_helper"
        # Calling through nn.Module.__call__ is what roots a kept guard at the
        # alias. The module lives in a file so that re-importing it gives a
        # scope with none of the names a load has to seed.
        source = """
import torch



class Child(torch.nn.Module):
    def forward(self, x):
        return x.sin()


CHILD = Child()


def fn(x):
    return CHILD(x)
"""

        def guard_filter_fn(guards):
            # Keep the global guards, which is what puts the alias in the
            # artifact, minus the types the serializer rejects.
            unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
            return [
                guard.guard_type not in unsupported
                and not any(d in unsupported for d in guard.derived_guard_types)
                for guard in guards
            ]

        ctx = DiskDynamoStore()
        self.addCleanup(sys.modules.pop, module_name, None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            helper_path = os.path.join(tmp_dir, "package_alias_helper.py")
            with open(helper_path, "w") as f:
                f.write(source)
            module = import_from_path(module_name, helper_path)
            args = (torch.randn(3),)
            expected = module.fn(*args)

            aot_path = os.path.join(tmp_dir, "aot_fn.pt")
            torch.compile(
                module.fn,
                fullgraph=True,
                backend="eager",
                options={"guard_filter_fn": guard_filter_fn},
            ).aot_compile((args, {})).save_compiled_function(aot_path)

            torch._dynamo.reset()
            package = CompilePackage(module.fn)
            compiled_fn = torch._dynamo.optimize(
                backend="eager", package=package, guard_filter_fn=guard_filter_fn
            )(module.fn)
            compiled_fn(*args)
            for backend_id, backend in package.cached_backends.items():
                ctx.record_eager_backend(backend_id, backend)
            ctx.save_package(package, self.path())

            torch._dynamo.reset()
            # A fresh import, as the loading process would see the module: the
            # alias is unbound there until the load seeds it.
            module = import_from_path(module_name, helper_path)
            scope = vars(module)
            self.assertNotIn(alias, set(scope))
            with open(aot_path, "rb") as f:
                loaded = torch.compiler.load_compiled_function(f, f_globals=scope)
            self.assertIn(alias, set(scope))

            package, backends = ctx.load_package(module.fn, self.path())
            # Not a vacuous test: install() really does write this alias.
            installs_alias = any(
                alias in entry.import_sources for entry in package._codes.values()
            )
            self.assertTrue(installs_alias)
            # The gate is scoped to the aliases: the backend ids go through
            # the default record_only_if_new=False, so they are recorded and
            # removed however the module scope looked beforehand.
            backend_ids = set(backends)
            self.assertTrue(backend_ids)
            self.assertEqual(backend_ids & set(scope), set())
            package.install(backends)
            self.assertIn(alias, set(scope))
            self.assertTrue(backend_ids <= set(scope))
            package.uninstall()
            self.assertIn(alias, set(scope))
            self.assertEqual(backend_ids & set(scope), set())
            self.assertEqual(loaded(*args), expected)

            # The other arm of the record, which needs a scope where the alias
            # is still unbound when install() runs: another fresh import gives
            # one, and there the package IS the first binder, so uninstall()
            # takes the alias back out.
            module = import_from_path(module_name, helper_path)
            unseeded_scope = vars(module)
            self.assertNotIn(alias, set(unseeded_scope))
            package, backends = ctx.load_package(module.fn, self.path())
            package.install(backends)
            self.assertIn(alias, set(unseeded_scope))
            package.uninstall()
            self.assertNotIn(alias, set(unseeded_scope))

    @parametrize("shape", ["flat", "closure", "two_levels"])
    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_package_binds_an_inlined_frames_resume_in_the_root_module(self, shape):
        # Under nested graph breaks, the continuation of a graph break inside an
        # inlined helper runs with the helper's globals but is bound in, and
        # loaded by LOAD_GLOBAL from, the root frame's module. Binding it in the
        # helper's module instead leaves the root's bytecode with a NameError.
        sources = nested_break_sources(f"test_package_ngb_{shape}", shape)
        entry_name = list(sources)[-1]
        ctx = DiskDynamoStore()
        with tempfile.TemporaryDirectory() as tmp_dir:
            for name in sources:
                self.addCleanup(sys.modules.pop, name, None)

            def fresh_import():
                for name, source in sources.items():
                    path = os.path.join(tmp_dir, name + ".py")
                    with open(path, "w") as f:
                        f.write(source)
                    import_from_path(name, path)
                return sys.modules[entry_name]

            entry = fresh_import()
            args = (torch.randn(3),)
            expected = entry.fn(*args)
            package = CompilePackage(entry.fn)
            torch._dynamo.optimize(backend="eager", package=package)(entry.fn)(*args)
            for backend_id, backend in package.cached_backends.items():
                ctx.record_eager_backend(backend_id, backend)
            ctx.save_package(package, self.path())
            bound_in = {
                name: module
                for code in package._codes.values()
                for name, module in code.bound_in.items()
            }
            self.assertEqual(set(bound_in.values()), {entry_name})
            self.assertEqual(len(bound_in), 1 if shape == "flat" else 2)
            has_closure = any(code.co_freevars for code in package._codes)
            self.assertEqual(has_closure, shape == "closure")

            torch._dynamo.reset()
            entry = fresh_import()
            others = [vars(sys.modules[name]) for name in list(sources)[:-1]]
            package, backends = ctx.load_package(entry.fn, self.path())
            with torch.compiler.set_stance("fail_on_recompile"):
                compiled_fn = torch._dynamo.optimize(package=package)(entry.fn)
                package.install(backends)
                self.assertEqual(compiled_fn(*args), expected)
            self.assertTrue(set(bound_in) <= set(vars(entry)))
            for scope in others:
                self.assertEqual(set(bound_in) & set(scope), set())
            package.uninstall()
            self.assertEqual(set(bound_in) & set(vars(entry)), set())

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_bound_method_name_guard_survives_func_reached_first(self):
        # Regression: a guard on a bound method's __name__ where the method's
        # __func__ is ALSO reachable (self.other) and inserted first, so the
        # pickle memoizes the fqn-mismatched function as _Missing before it
        # reaches the method. Seeding the method's __func__ into
        # guard_tree_values makes the save order-independent; without it the
        # method's __func__ loads back as _Missing and the __name__ guard
        # AttributeErrors at torch.compile() wrap time in the reloading process.
        mod = BoundMethodNameGuardModule()
        keys = list(mod.__dict__)
        self.assertLess(keys.index("other"), keys.index("cb"))
        x = torch.randn(3)
        expected = mod(x)
        self.assertEqual(torch.compile(mod)(x), expected)  # noqa: UNSPECIFIED_BACKEND
        (entry,) = PrecompileContext.save_to_dynamo_cache()["dynamo"]
        self.assertEqual(len(entry["backend_ids"]), 1)
        torch._dynamo.reset()
        PrecompileContext.clear()
        compiled = torch.compile(mod)  # noqa: UNSPECIFIED_BACKEND
        code = BoundMethodNameGuardModule.forward.__code__
        self.assertEqual(len(_debug_get_precompile_entries(code)), 1)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled(x), expected)
            # The reloaded guard really reads __name__ off the rebuilt function.
            _bound_method_guard_wrapper.__name__ = "renamed"
            try:
                with self.assertRaisesRegex(RuntimeError, "fail_on_recompile"):
                    compiled(x)
            finally:
                _bound_method_guard_wrapper.__name__ = "_bound_method_guard_target"

    @torch._dynamo.config.patch(caching_precompile=True, strict_precompile=False)
    def test_unserializable_guard_bypasses_the_package(self):
        # A guarded value that cannot be pickled is a package bypass, not a
        # compile failure: the frame still compiles and runs, and its entry is
        # saved bypassed with no backend, so nothing is installed on reload.
        def fn(x, cfg=UnpicklableConfig()):
            if cfg.scale == 2.0:
                x = x + 1
            return x.sin()

        x = torch.randn(3)
        expected = fn(x)
        with self.assertLogs("torch._dynamo", level="WARNING") as logs:
            self.assertEqual(torch.compile(fn)(x), expected)  # noqa: UNSPECIFIED_BACKEND
        self.assertTrue(any("config cannot pickle" in line for line in logs.output))
        (entry,) = PrecompileContext.save_to_dynamo_cache()["dynamo"]
        self.assertEqual(entry["backend_ids"], [])
        torch._dynamo.reset()
        PrecompileContext.clear()
        # Wrapping is what reloads the cache; the bypassed entry installs nothing.
        compiled = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(len(_debug_get_precompile_entries(fn.__code__)), 0)
        with self.assertLogs("torch._dynamo", level="WARNING") as logs:
            self.assertEqual(compiled(x), expected)
        self.assertTrue(any("config cannot pickle" in line for line in logs.output))

    def test_import_alias_is_not_bound_to_a_non_module_import(self):
        # sys.modules accepts any object and __import__ hands it back verbatim.
        # IMPORT_NAME rejects it before import_source binds the alias, so the
        # traced globals never hold the non-module, and a later trace after a
        # real module has replaced the entry binds the alias to that module
        # rather than tracing it through a slot still holding the non-module.
        name = "torch_test_package_import_alias_non_module"
        alias = f"__import_{name}"
        module = types.ModuleType(name)
        module.VALUE = 1
        args = (torch.randn(3, 2),)

        def fn(x):
            import torch_test_package_import_alias_non_module as taken

            return x + taken.VALUE

        try:
            sys.modules[name] = object()
            with self.assertRaisesRegex(Unsupported, "Bad import result"):
                torch.compile(fn, backend="eager", fullgraph=True)(*args)
            self.assertNotIn(alias, fn.__globals__)
            torch._dynamo.reset()
            sys.modules[name] = module
            compiled = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(fn(*args), compiled(*args))
            self.assertIs(fn.__globals__[alias], module)
        finally:
            sys.modules.pop(name, None)
            fn.__globals__.pop(alias, None)
            # The memo outlives the sys.modules entry: a same-process rerun would
            # otherwise resolve this run's module from it.
            _import_module.cache_clear()
            torch._dynamo.reset()

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_a_poisoned_entry_is_reset_on_load_instead_of_growing(self):
        # A resume frame whose backend artifact is missing at save time is
        # written bypassed. install() skips it and the frame is traced fresh;
        # that compile used to append its guarded code to the stale one and
        # re-register the missing backend id, so every reload/save cycle
        # re-poisoned the entry and grew it. Loading a bypassed entry now drops
        # its stale codes and ids: the entry stops growing and the next save is
        # installable, so the third process hits without a recompile.
        def fn(x):
            y = x.sin()
            torch._dynamo.graph_break()
            return x.sin() + y

        x = torch.randn(3, 2)
        expected = torch.compile(fn)(x)  # noqa: UNSPECIFIED_BACKEND
        dynamo_entry = next(iter(PrecompileContext._dynamo_cache_entries.values()))
        for code in dynamo_entry.codes:
            if any("resume" in name for name in code.function_names):
                (backend,) = code.backend_ids
                del PrecompileContext._backend_artifacts_by_key[backend]
        self._save_and_reload(expected_backends=1, expected_dynamo=1)

        def resume_of(entry):
            (code,) = [
                c for c in entry.codes if any("resume" in n for n in c.function_names)
            ]
            return code

        def resume_entry():
            return resume_of(DynamoCache.load(fn).dynamo)

        self.assertTrue(resume_entry().bypassed)
        self.assertEqual(len(resume_entry().guarded_codes), 1)
        # Loading resets the package's copy, not the caller's entry.
        loaded = DynamoCache.load(fn).dynamo
        package = CompilePackage(fn, dynamo=loaded)
        self.assertEqual(len(resume_of(loaded).guarded_codes), 1)
        reset = resume_of(package.cache_entry())
        self.assertEqual(reset.guarded_codes, [])
        self.assertEqual(reset.backend_ids, [])
        # The containers the fresh compile writes to are detached as well.
        self.assertIsNot(reset.import_sources, resume_of(loaded).import_sources)
        self.assertIsNot(reset.function_names, resume_of(loaded).function_names)
        self.assertEqual(torch.compile(fn)(x), expected)  # noqa: UNSPECIFIED_BACKEND
        self._save_and_reload(expected_backends=2, expected_dynamo=1)
        # One guarded code and one backend id, not two of each; and installable:
        # the third process compiles nothing (FRAME_COUNTER also advances for
        # installed entries, so count actual compiles).
        self.assertFalse(resume_entry().bypassed)
        self.assertEqual(len(resume_entry().guarded_codes), 1)
        self.assertEqual(len(resume_entry().backend_ids), 1)
        compiles = torch._dynamo.utils.counters["frames"]["total"]
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(torch.compile(fn)(x), expected)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(torch._dynamo.utils.counters["frames"]["total"], compiles)

    def test_file_change(self):
        ctx = DiskDynamoStore()

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

    def _tensor_free_resume_entry(self, package, name="_tensor_free_continuation"):
        (entry,) = (
            entry
            for code, entry in package._codes.items()
            if code.co_name.startswith(f"torch_dynamo_resume_in_{name}_at_")
        )
        return entry

    def _capture_tensor_free_resume(
        self, values, fn=_tensor_free_resume_fn, name="_tensor_free_continuation"
    ):
        ctx = DiskDynamoStore()
        x = torch.randn(4)
        package = CompilePackage(fn)
        compiled = torch._dynamo.optimize(
            backend="eager",
            package=package,
            guard_filter_fn=_skip_closure_match,
            dynamic=False,
        )(fn)
        for value in values:
            _TENSOR_FREE_RESUME_STATE["value"] = value
            self.assertEqual(compiled(x), fn(x))
        self.assertEqual(
            len(self._tensor_free_resume_entry(package, name).guarded_codes),
            len(values),
        )
        for backend_id, bknd in package.cached_backends.items():
            ctx.record_eager_backend(backend_id, bknd)
        ctx.save_package(package, self.path())

        torch._dynamo.reset()
        package, backends = ctx.load_package(fn, self.path())
        compiled = torch._dynamo.optimize("eager", package=package, dynamic=False)(fn)
        package.install(backends)
        return package, compiled, x

    # Nested graph breaks would resume inside a tensor-bearing parent frame.
    @torch._dynamo.config.patch(nested_graph_breaks=False)
    @patch.dict(_TENSOR_FREE_RESUME_STATE)
    def test_tensor_free_resume_guard_miss_fails_on_recompile(self):
        # The continuation after the break in _tensor_free_continuation holds
        # only an int, yet its graph depends on it: the package must record
        # its guards, and a miss on it must raise under fail_on_recompile
        # instead of silently running eagerly.
        _, compiled, x = self._capture_tensor_free_resume((1, 2))
        fn = _tensor_free_resume_fn
        with torch.compiler.set_stance("fail_on_recompile"):
            for value in (1, 2):
                _TENSOR_FREE_RESUME_STATE["value"] = value
                self.assertEqual(compiled(x), fn(x))
            _TENSOR_FREE_RESUME_STATE["value"] = 3
            with self.assertRaisesRegex(
                RuntimeError,
                "function name: 'torch_dynamo_resume_in__tensor_free_continuation_at_",
            ):
                compiled(x)

    # Nested graph breaks would resume inside a tensor-bearing parent frame.
    @torch._dynamo.config.patch(nested_graph_breaks=False)
    @patch.dict(_TENSOR_FREE_RESUME_STATE)
    def test_tensor_free_resume_recompile_after_load_keeps_guards(self):
        # A loaded package still knows its resume functions, so a miss on a
        # tensor-free continuation compiles a new guarded variant into it.
        package, compiled, x = self._capture_tensor_free_resume((1,))
        fn = _tensor_free_resume_fn
        _TENSOR_FREE_RESUME_STATE["value"] = 2
        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(self._tensor_free_resume_entry(package).guarded_codes), 2)
        with torch.compiler.set_stance("fail_on_recompile"):
            for value in (1, 2):
                _TENSOR_FREE_RESUME_STATE["value"] = value
                self.assertEqual(compiled(x), fn(x))

    # Precompile capture keeps empty graphs as compiled frames.
    @torch._dynamo.config.patch(nested_graph_breaks=False, allow_empty_graphs=True)
    @patch.dict(_TENSOR_FREE_RESUME_STATE)
    def test_tensor_free_resume_with_an_empty_graph_is_refused_on_a_new_value(self):
        # The continuation only computes step + 1, yet it is still guarded:
        # a call no captured variant covers is refused, even without tensor work.
        fn = _empty_graph_resume_fn
        name = "_empty_graph_continuation"
        _, compiled, x = self._capture_tensor_free_resume((1,), fn, name)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled(x), fn(x))
            _TENSOR_FREE_RESUME_STATE["value"] = 2
            with self.assertRaisesRegex(
                RuntimeError, f"function name: 'torch_dynamo_resume_in_{name}_at_"
            ):
                compiled(x)

    # Nested graph breaks would resume inside a tensor-bearing parent frame.
    @torch._dynamo.config.patch(nested_graph_breaks=False)
    @patch.dict(_TENSOR_FREE_RESUME_STATE)
    def test_tensor_free_resume_is_skipped_without_an_explicit_package(self):
        # Without a package, or with only the one caching_precompile creates,
        # the tensor-free continuation stays skipped: it gets no cache entry
        # and a new value under fail_on_recompile is no recompile.
        from torch._C._dynamo.eval_frame import _debug_get_cache_entry_list, reset_code
        from torch._dynamo.resume_execution import ContinueExecutionCache

        fn = _tensor_free_resume_fn
        x = torch.randn(4)
        for caching_precompile in (False, True):
            with (
                self.subTest(caching_precompile=caching_precompile),
                torch._dynamo.config.patch(caching_precompile=caching_precompile),
            ):
                torch._dynamo.reset()
                DynamoCache.clear()
                _TENSOR_FREE_RESUME_STATE["value"] = 1
                compiled = torch._dynamo.optimize("eager", dynamic=False)(fn)
                self.assertEqual(compiled(x), fn(x))
                (resume,) = ContinueExecutionCache.cache[
                    _tensor_free_continuation.__code__
                ].values()
                self.assertEqual(len(_debug_get_cache_entry_list(resume)), 0)
                # Drop the cached skip so the next run reaches the
                # fail_on_recompile callback.
                reset_code(resume)
                _TENSOR_FREE_RESUME_STATE["value"] = 2
                with torch.compiler.set_stance("fail_on_recompile"):
                    self.assertEqual(compiled(x), fn(x))

    @torch._dynamo.config.patch(caching_precompile=True)
    @patch.dict(_TENSOR_FREE_RESUME_STATE)
    def test_warm_loaded_resume_entries_do_not_fail_a_tensor_free_call(self):
        # The warm load installs entries on fn's resume function, but the
        # caching_precompile package is not explicit: once that frame holds no
        # tensor it is skipped and runs eagerly, so under fail_on_recompile it
        # is no recompile either.
        fn = _resume_maybe_tensor_free_fn
        x = torch.randn(3)
        options = {"guard_filter_fn": _skip_closure_match}
        torch.compile(fn, options=options)(x)  # noqa: UNSPECIFIED_BACKEND
        self._save_and_reload(expected_backends=2, expected_dynamo=1)
        compiled = torch.compile(fn, options=options)  # noqa: UNSPECIFIED_BACKEND
        _TENSOR_FREE_RESUME_STATE["value"] = 0
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled(x), fn(x))

    def _round_trip_under_fail_on_recompile(self, fn, x):
        ctx = DiskDynamoStore()
        package = CompilePackage(fn)
        compiled_fn = torch._dynamo.optimize(
            backend="eager", package=package, guard_filter_fn=default_guard_filter_fn
        )(fn)
        expected = compiled_fn(x)
        for backend_id, backend in package.cached_backends.items():
            ctx.record_eager_backend(backend_id, backend)
        ctx.save_package(package, self.path())

        torch._dynamo.reset()
        package, backends = ctx.load_package(fn, self.path())
        compiled_fn = torch._dynamo.optimize(
            package=package, guard_filter_fn=default_guard_filter_fn
        )(fn)
        package.install(backends)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled_fn(x), expected)
        return package

    # Without nested graph breaks, the break makes helper a frame of its own.
    @torch._dynamo.config.patch(nested_graph_breaks=False)
    def test_package_records_an_aliased_modules_own_frames(self):
        # One file imported under two names is two modules, and
        # inspect.getmodule maps a code object to whichever one it saw last.
        # Recording helper's frame under the other alias looked for its code
        # there and failed the compiled call.
        source = """
import torch


def helper(y):
    y = y.sin()
    torch._dynamo.graph_break()
    return y.cos()


def entry(x):
    return helper(x + 1) * 2
"""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "aliased.py")
        with open(path, "w") as f:
            f.write(source)
        names = [f"test_package_alias_{i}" for i in range(2)]
        for name in names:
            self.addCleanup(sys.modules.pop, name, None)
        first, second = (import_from_path(name, path) for name in names)
        self.assertIs(inspect.getmodule(first.helper.__code__), second)

        package = self._round_trip_under_fail_on_recompile(first.entry, torch.randn(4))
        modules = {entry.python_module for entry in package.cache_entry().codes}
        self.assertEqual(modules, {first.__name__})

    def test_package_records_the_entry_under_its_globals_module(self):
        # functools.wraps copies __module__ from the wrapped function, so the
        # wrapper's __module__ names a module whose namespace is not the one
        # its guards and installed globals belong in.
        @functools.wraps(torch.sin)
        def wrapper(x):
            return torch.sin(x) + PKG_SCALE

        self.assertEqual(wrapper.__module__, "torch")
        package = self._round_trip_under_fail_on_recompile(wrapper, torch.randn(4))
        self.assertEqual(package.cache_entry().codes[0].python_module, __name__)

    def test_package_keeps_an_entry_module_alias_that_owns_its_globals(self):
        # A __module__ naming another sys.modules key for the same module (here
        # an importable alias of this test module) is kept, not replaced by the
        # globals' own __name__.
        def entry(x):
            return x.sin() + PKG_SCALE

        alias = "test_package_entry_alias"
        sys.modules[alias] = sys.modules[__name__]
        self.addCleanup(sys.modules.pop, alias, None)
        entry.__module__ = alias
        package = self._round_trip_under_fail_on_recompile(entry, torch.randn(4))
        self.assertEqual(package.cache_entry().codes[0].python_module, alias)

    # Without nested graph breaks, hooked_forward becomes a frame of its own.
    @torch._dynamo.config.patch(nested_graph_breaks=False)
    def test_package_keeps_the_entry_module_alias_for_frames_sharing_its_globals(
        self,
    ):
        # The child frame (hooked_forward) and the resume functions run with the
        # entry's globals, so they are recorded under the entry's alias too.
        def entry(x):
            y = hooked_forward(x, x) + x
            torch._dynamo.graph_break()
            return y + PKG_SCALE

        alias = "test_package_entry_alias"
        sys.modules[alias] = sys.modules[__name__]
        self.addCleanup(sys.modules.pop, alias, None)
        entry.__module__ = alias
        package = self._round_trip_under_fail_on_recompile(entry, torch.randn(4))
        codes = package.cache_entry().codes
        self.assertGreater(len(codes), 2)
        self.assertEqual({code.python_module for code in codes}, {alias})

    def test_package_records_a_renamed_module_under_its_spec_name(self):
        # A module that rebinds __name__ is still registered under its spec
        # name, and that is the only key a loading process can find it by.
        source = """
import torch

__name__ = "test_package_renamed_public"


def entry(x):
    x = x.sin()
    torch._dynamo.graph_break()
    return x * 2
"""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "renamed.py")
        with open(path, "w") as f:
            f.write(source)
        name = "test_package_renamed"
        self.addCleanup(sys.modules.pop, name, None)
        module = import_from_path(name, path)
        self.assertNotIn(module.entry.__module__, sys.modules)

        package = self._round_trip_under_fail_on_recompile(module.entry, torch.randn(4))
        codes = package.cache_entry().codes
        self.assertEqual(len(codes), 2)
        self.assertEqual({entry.python_module for entry in codes}, {name})

    def test_reset_keeps_nonrecursive_disable_wrapper_skipped(self):
        # A bound method's innermost fn is the shared non-recursive disable
        # wrapper, so this package's top-level code is the wrapper's code.
        # Installing it must not let reset() strip its import-time skip.
        class Mod(torch.nn.Module):
            @torch._dynamo.disable(recursive=False)
            def forward(self, x):
                return self.inner(x) * 2

            def inner(self, x):
                return x.cos() + 1

        ctx = DiskDynamoStore()
        mod = Mod()
        x = torch.randn(4)
        package = CompilePackage(mod.forward)
        torch._dynamo.optimize(backend="eager", package=package)(mod)(x)
        for backend_id, bknd in package.cached_backends.items():
            ctx.record_eager_backend(backend_id, bknd)
        ctx.save_package(package, self.path())

        torch._dynamo.reset()
        package, backends = ctx.load_package(mod.forward, self.path())
        self.assertIs(
            next(iter(package._codes)),
            torch._dynamo.decorators._nonrecursive_disable_wrapper_code,
        )
        package.install(backends)
        torch._dynamo.reset()

        cnt = torch._dynamo.testing.CompileCounter()
        self.assertEqual(torch.compile(mod, backend=cnt)(x), mod(x))
        # Only inner() is compiled; forward() itself stays eager.
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

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
        sampler = _tempTensorSamplerForQualName.class_method_that_is_used(x)
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


class TestPackageAccelerator(torch._inductor.test_case.TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    _package_config_stack = None

    @classmethod
    def setUpClass(cls):
        cls._package_config_stack = contextlib.ExitStack()
        cls._package_config_stack.enter_context(
            functorch_config.patch("bundled_autograd_cache", True)
        )
        cls._package_config_stack.enter_context(
            torch._dynamo.config.patch({"strict_precompile": True})
        )
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            cls._package_config_stack.close()
            cls._package_config_stack = None

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

    @parametrize("backend", ("eager", "inductor"))
    def test_basic_fn(self, device, backend):
        if device != "cpu" and backend == "inductor" and not has_triton():
            raise unittest.SkipTest("Requires Triton")
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
    def test_lazy_backward(self, device, backend):
        if device != "cpu" and backend == "inductor" and not has_triton():
            raise unittest.SkipTest("Requires Triton")
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
    def test_graph_break_bomb(self, device, backend):
        if device != "cpu" and backend == "inductor" and not has_triton():
            raise unittest.SkipTest("Requires Triton")
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
    def test_dynamic_shape(self, device, backend):
        if device != "cpu" and backend == "inductor" and not has_triton():
            raise unittest.SkipTest("Requires Triton")
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

    def test_dynamo_cache_manual_load(self, device):
        if device != "cpu" and not has_triton():
            raise unittest.SkipTest("Requires Triton")

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

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_automatic_dynamo_serialize(self, device):
        if device != "cpu" and not has_triton():
            raise unittest.SkipTest("Requires Triton")

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

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_automatic_dynamo_import_source_guard(self, device):
        if device != "cpu" and not has_triton():
            raise unittest.SkipTest("Requires Triton")

        # Warm-loading a guard state whose serialized sources include an
        # ImportSource must not raise. `pytree.tree_is_leaf` routes through
        # `get_pytree_SUPPORTED_NODES_source`, which builds an
        # `ImportSource("torch")` that ends up in the serialized guard state.
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

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_automatic_dynamo_recompiles(self, device):
        if device != "cpu" and not has_triton():
            raise unittest.SkipTest("Requires Triton")

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
    @torch._dynamo.config.patch(caching_precompile=True)
    def test_automatic_dynamo_graph_breaks(self, device):
        if device != "cpu" and not has_triton():
            raise unittest.SkipTest("Requires Triton")

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
    @torch._dynamo.config.patch(caching_precompile=True)
    def test_automatic_dynamo_lazy_backward(self, device):
        if device != "cpu" and not has_triton():
            raise unittest.SkipTest("Requires Triton")

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

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_graph_break_partial_backend(self, device):
        if device != "cpu" and not has_triton():
            raise unittest.SkipTest("Requires Triton")

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

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_call_function_from_resume(self, device):
        if device != "cpu" and not has_triton():
            raise unittest.SkipTest("Requires Triton")
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

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_code_with_generator(self, device):
        if device != "cpu" and not has_triton():
            raise unittest.SkipTest("Requires Triton")

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

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_automatic_dynamo_graph_breaks_from_print_model_as_fn(self, device):
        if device != "cpu" and not has_triton():
            raise unittest.SkipTest("Requires Triton")

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

    @unittest.expectedFailure  # FUNCTION_MATCH guard not serializable today
    def test_nn_module(self, device):
        if device != "cpu" and not has_triton():
            raise unittest.SkipTest("Requires Triton")

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10, device=device)

            def forward(self, x):
                return self.linear(x)

        fn = MyModule()
        package = CompilePackage(fn.forward)
        compiled_fn = torch._dynamo.optimize("inductor", package=package)(fn)
        x = torch.randn(10, 10, device=device)
        compiled_fn(x)

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_classmethod_qualname(self, device):
        if device != "cpu" and not has_triton():
            raise unittest.SkipTest("Requires Triton")
        x = torch.rand(10, device=device)
        model = _tempNetForQualName()
        model.forward(x)
        compiled_fn = torch.compile(  # noqa: UNSPECIFIED_BACKEND
            model.forward,
            options=dict(guard_filter_fn=torch.compiler.skip_guard_on_globals_unsafe),
        )
        compiled_fn(x)


instantiate_device_type_tests(
    TestPackageAccelerator,
    globals(),
    allow_xpu=True,
    allow_mps=True,
)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
