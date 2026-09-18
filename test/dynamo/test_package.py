# Owner(s): ["module: dynamo"]

import builtins
import functools
import gc
import importlib
import os
import pickle
import re
import sys
import tempfile
import types
import unittest
from unittest import mock
from unittest.mock import patch

import torch
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.onnx.operators
import torch.utils.cpp_extension
from torch._C._dynamo.eval_frame import _debug_get_precompile_entries
from torch._dynamo.comptime import comptime
from torch._dynamo.exc import InternalTorchDynamoError, Unsupported
from torch._dynamo.guards import CheckFunctionManager
from torch._dynamo.package import (
    _collapse_device_types,
    CompilePackage,
    DiskDynamoStore,
    DynamoCache,
)
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.symbolic_convert import (
    _import_source_cache,
    InstructionTranslatorBase,
)
from torch._dynamo.testing import CompileCounter, reduce_to_scalar_loss
from torch._dynamo.utils import CleanupManager
from torch._functorch import config as functorch_config
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
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


def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def compute_loss_helper(x):
    return reduce_to_scalar_loss(x)


def compiled_region_with_backend_id_for_package_test():
    return __compiled_fn_0_00000000_0000_0000_0000_000000000000()  # noqa: F821


class UnpicklableConfig:
    # Like ConfigThatCannotPickle, but the failure is not an AttributeError.
    scale = 2.0

    def __reduce__(self):
        raise RuntimeError("config cannot pickle")


def _import_alias_getattr_boom(name):
    # A PEP 562 module __getattr__ that must never run inside a trace.
    raise RuntimeError(f"module __getattr__ ran inside a trace for {name}")


def _import_source_on_stack():
    # InstructionTranslatorBase.import_source, matched by module as well as by
    # name; the getattribute test pins that the pair still names it.
    frame = sys._getframe(1)
    while frame is not None:
        where = (frame.f_code.co_name, frame.f_globals.get("__name__"))
        if where == ("import_source", "torch._dynamo.symbolic_convert"):
            return True
        frame = frame.f_back
    return False


class _ImportAliasHookedModule(types.ModuleType):
    # A class-level __getattribute__ intercepts every read, __dict__ included,
    # as importlib.util._LazyModule's does, and that one runs the module body
    # on any read. __import__ and PythonModuleVariable read __spec__ and
    # __name__ off a module by design, so only a read made by import_source
    # raises: its resolution makes none, and its alias check takes the
    # instance dict through object.__getattribute__ instead.
    def __getattribute__(self, name):
        if _import_source_on_stack():
            raise RuntimeError(f"module __getattribute__ read {name} inside a trace")
        return object.__getattribute__(self, name)


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


# A dynamic dim on a module-level tensor is what makes a SHAPE_ENV guard read a
# global -- as a literal G['PKG_DYN_ROWS'] inside a Python lambda by default.
PKG_DYN_ROWS = torch.randn(4, 3)
torch._dynamo.mark_dynamic(PKG_DYN_ROWS, 0)


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
        # check_versions accepts the cuda entry on a host without one.
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
        saved = pickle.loads(pickle.dumps(package.cache_entry()))
        self.assertEqual(saved.device_type, "cuda")
        with patch.object(torch.cuda, "is_available", return_value=True):
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

    def test_import_alias_binds_the_live_module(self):
        # The alias roots the guards for every attribute read off the module,
        # while the graph is specialized on what IMPORT_NAME pushed, which is
        # what __import__ returned: live sys.modules. Binding the module an
        # earlier compile resolved instead guards an object the graph never
        # saw, and the guard is then blind to every later change to the one
        # the program does use.
        name = "torch_test_package_import_alias_live"
        alias = f"__import_{name}"
        stale = types.ModuleType(name)
        stale.VALUE = 2
        args = (torch.randn(3, 2),)

        def fn(x):
            import torch_test_package_import_alias_live as mod

            return x * mod.VALUE

        try:
            sys.modules[name] = stale
            torch.compile(fn, backend="eager", fullgraph=True)(*args)
            torch._dynamo.reset()
            fresh = types.ModuleType(name)
            fresh.VALUE = 3
            sys.modules[name] = fresh

            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(fn(*args), compiled_fn(*args))
            self.assertIs(fn.__globals__[alias], fresh)

            fresh.VALUE = 5
            self.assertEqual(fn(*args), compiled_fn(*args))
        finally:
            sys.modules.pop(name, None)
            fn.__globals__.pop(alias, None)
            _import_source_cache.pop(name, None)
            torch._dynamo.reset()

    def test_import_alias_survives_a_sys_modules_rebind(self):
        # install() binds every recorded __import_* alias to whatever sys.modules
        # holds at load time, in the traced module's live globals, and nothing
        # unbinds it afterwards. A shim that hands over again leaves the alias
        # holding a module object the next trace does not resolve the name to,
        # and that trace has to rebind it instead of dying on it. All three
        # objects agree on VALUE, so the installed artifact stays valid across
        # the rebind.
        ctx = DiskDynamoStore()
        name = "torch_test_package_import_alias_shim"
        alias = f"__import_{name}"
        placeholder = types.ModuleType(name)
        placeholder.VALUE = 2

        def fn(x):
            import torch_test_package_import_alias_shim as shim

            return x + shim.VALUE

        def fn2(x):
            import torch_test_package_import_alias_shim as shim

            return x * shim.VALUE

        args = (torch.randn(3, 2),)
        try:
            sys.modules[name] = placeholder
            package = CompilePackage(fn)
            compiled_fn = torch._dynamo.optimize(backend="eager", package=package)(fn)
            self.assertEqual(fn(*args), compiled_fn(*args))
            for backend_id, backend in package.cached_backends.items():
                ctx.record_eager_backend(backend_id, backend)
            ctx.save_package(package, self.path())

            # The shim hands over to the real module: same name, new object.
            real = types.ModuleType(name)
            real.VALUE = 2
            sys.modules[name] = real
            torch._dynamo.reset()
            package, backends = ctx.load_package(fn, self.path())
            loaded_fn = torch._dynamo.optimize(backend="eager", package=package)(fn)
            package.install(backends)
            self.assertIs(fn.__globals__[alias], real)

            # And hands over once more, after the install pinned the alias.
            newer = types.ModuleType(name)
            newer.VALUE = 2
            sys.modules[name] = newer
            compiled_fn2 = torch.compile(fn2, backend="eager", fullgraph=True)
            self.assertEqual(fn2(*args), compiled_fn2(*args))
            self.assertIs(fn.__globals__[alias], newer)

            with torch.compiler.set_stance("fail_on_recompile"):
                self.assertEqual(fn(*args), loaded_fn(*args))
        finally:
            sys.modules.pop(name, None)
            fn.__globals__.pop(alias, None)
            _import_source_cache.pop(name, None)
            torch._dynamo.reset()

    def test_import_alias_rebind_makes_a_stale_artifact_visible(self):
        # The rebind hands the alias to the live module, so a loaded artifact
        # whose guards read it sees the value the program itself now reads. When
        # the handover changes that value the artifact is stale and its guard
        # says so: a stance that forbids recompiling reports it, and the default
        # stance recompiles and serves the live answer. Left on the abandoned
        # module the same guard passes and the artifact answers out of a module
        # nobody holds.
        ctx = DiskDynamoStore()
        name = "torch_test_package_import_alias_recompile"
        alias = f"__import_{name}"
        placeholder = types.ModuleType(name)
        placeholder.VALUE = 2

        def fn(x):
            import torch_test_package_import_alias_recompile as shim

            return x + shim.VALUE

        def fn2(x):
            import torch_test_package_import_alias_recompile as shim

            return x * shim.VALUE

        args = (torch.randn(3, 2),)
        try:
            sys.modules[name] = placeholder
            package = CompilePackage(fn)
            compiled_fn = torch._dynamo.optimize(backend="eager", package=package)(fn)
            self.assertEqual(fn(*args), compiled_fn(*args))
            for backend_id, backend in package.cached_backends.items():
                ctx.record_eager_backend(backend_id, backend)
            ctx.save_package(package, self.path())
            torch._dynamo.reset()
            package, backends = ctx.load_package(fn, self.path())
            loaded_fn = torch._dynamo.optimize(backend="eager", package=package)(fn)
            package.install(backends)

            # This handover answers differently, so the artifact is now stale.
            newer = types.ModuleType(name)
            newer.VALUE = 7
            sys.modules[name] = newer
            compiled_fn2 = torch.compile(fn2, backend="eager", fullgraph=True)
            self.assertEqual(fn2(*args), compiled_fn2(*args))
            self.assertIs(fn.__globals__[alias], newer)

            with torch.compiler.set_stance("fail_on_recompile"):
                with self.assertRaisesRegex(RuntimeError, "Detected recompile"):
                    loaded_fn(*args)
            self.assertEqual(fn(*args), loaded_fn(*args))
        finally:
            sys.modules.pop(name, None)
            fn.__globals__.pop(alias, None)
            _import_source_cache.pop(name, None)
            torch._dynamo.reset()

    @parametrize("stale_named_for", ("target", "key"))
    def test_import_alias_accepts_a_stale_module_of_either_accepted_name(
        self, stale_named_for
    ):
        # A sys.modules key need not equal the module's own __name__: os.path is
        # named posixpath, and torch's own BC shims (torch.distributed._shard.
        # checkpoint, torch._inductor.template_heuristics.triton) are all such
        # entries. So the stale module one more handover after an install leaves
        # in the alias slot is recognized by either accepted name. One is the
        # name the resolved module answers to, since under such a key that is
        # never the key itself. The other is the key on its own, whichever name
        # the resolved module has: the copy a BC-shim key held before a handover
        # put the shim's target under it. A module of some third name still
        # graph breaks: that is what two module names mangling onto one alias
        # leave behind.
        key = "torch_test_package_import_alias_shim_key"
        target = "torch_test_package_import_alias_shim_target"
        alias = f"__import_{key}"
        stale = types.ModuleType(target if stale_named_for == "target" else key)
        stale.VALUE = 2
        live = types.ModuleType(target)
        live.VALUE = 3
        args = (torch.randn(3, 2),)

        def fn(x):
            import torch_test_package_import_alias_shim_key as shim

            return x + shim.VALUE

        try:
            sys.modules[key] = live
            fn.__globals__[alias] = stale
            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(fn(*args), compiled_fn(*args))
            self.assertIs(fn.__globals__[alias], live)

            torch._dynamo.reset()
            fn.__globals__[alias] = types.ModuleType("some.other.name")
            with self.assertRaisesRegex(
                Unsupported, rf"alias {alias} for {key}.*named some\.other\.name"
            ):
                torch.compile(fn, backend="eager", fullgraph=True)(*args)
        finally:
            sys.modules.pop(key, None)
            fn.__globals__.pop(alias, None)
            _import_source_cache.pop(key, None)
            torch._dynamo.reset()

    def test_import_alias_keeps_the_installed_live_module(self):
        # The first trace of the name binds the alias; install() then rebinds
        # it to the live entry a handover has since put in sys.modules. A later
        # trace in the same globals finds the two disagreeing and must not put
        # the first module back: its graph is specialized on the live one,
        # which is what __import__ hands IMPORT_NAME, and the alias roots its
        # guards, so a change to the live module has to fail them. The sibling
        # below pins the first module written back and the change unseen.
        ctx = DiskDynamoStore()
        name = "torch_test_package_import_alias_installed"
        alias = f"__import_{name}"
        old = types.ModuleType(name)
        old.VALUE = 2
        new = types.ModuleType(name)
        new.VALUE = 7

        def fn(x):
            import torch_test_package_import_alias_installed as shim

            return x + shim.VALUE

        def fn2(x):
            import torch_test_package_import_alias_installed as shim

            return x * shim.VALUE

        args = (torch.randn(3, 2),)
        try:
            sys.modules[name] = old
            package = CompilePackage(fn)
            compiled_fn = torch._dynamo.optimize(backend="eager", package=package)(fn)
            self.assertEqual(fn(*args), compiled_fn(*args))
            for backend_id, backend in package.cached_backends.items():
                ctx.record_eager_backend(backend_id, backend)
            ctx.save_package(package, self.path())
            torch._dynamo.reset()

            sys.modules[name] = new
            package, backends = ctx.load_package(fn, self.path())
            package.install(backends)
            self.assertIs(fn.__globals__[alias], new)

            cnt = CompileCounter()
            compiled_fn2 = torch.compile(fn2, backend=cnt, fullgraph=True)
            self.assertEqual(fn2(*args), compiled_fn2(*args))
            self.assertIs(fn.__globals__[alias], new)
            self.assertEqual(cnt.frame_count, 1)
            new.VALUE = 8
            self.assertEqual(fn2(*args), compiled_fn2(*args))
            self.assertEqual(cnt.frame_count, 2)
        finally:
            sys.modules.pop(name, None)
            fn.__globals__.pop(alias, None)
            _import_source_cache.pop(name, None)
            torch._dynamo.reset()

    def test_import_alias_rebinds_the_installed_module_to_the_live_entry(self):
        # install() binds the module a handover put in sys.modules, and a
        # second handover precedes the next trace. The alias is rebound to the
        # live entry, the module the graph is specialized on, so a change to
        # the live module recompiles and a change to the installed one is not
        # read by anything.
        ctx = DiskDynamoStore()
        name = "torch_test_package_import_alias_three_way"
        alias = f"__import_{name}"
        old = types.ModuleType(name)
        old.VALUE = 2
        mid = types.ModuleType(name)
        mid.VALUE = 7
        new = types.ModuleType(name)
        new.VALUE = 9

        def fn(x):
            import torch_test_package_import_alias_three_way as shim

            return x + shim.VALUE

        def fn2(x):
            import torch_test_package_import_alias_three_way as shim

            return x * shim.VALUE

        args = (torch.randn(3, 2),)
        try:
            sys.modules[name] = old
            package = CompilePackage(fn)
            compiled_fn = torch._dynamo.optimize(backend="eager", package=package)(fn)
            self.assertEqual(fn(*args), compiled_fn(*args))
            for backend_id, backend in package.cached_backends.items():
                ctx.record_eager_backend(backend_id, backend)
            ctx.save_package(package, self.path())
            torch._dynamo.reset()

            sys.modules[name] = mid
            package, backends = ctx.load_package(fn, self.path())
            package.install(backends)
            self.assertIs(fn.__globals__[alias], mid)
            sys.modules[name] = new

            cnt = CompileCounter()
            compiled_fn2 = torch.compile(fn2, backend=cnt, fullgraph=True)
            self.assertEqual(fn2(*args), compiled_fn2(*args))
            self.assertIs(fn.__globals__[alias], new)
            self.assertEqual(cnt.frame_count, 1)
            new.VALUE = 10
            self.assertEqual(fn2(*args), compiled_fn2(*args))
            self.assertEqual(cnt.frame_count, 2)
            mid.VALUE = 11
            self.assertEqual(fn2(*args), compiled_fn2(*args))
            self.assertEqual(cnt.frame_count, 2)
        finally:
            sys.modules.pop(name, None)
            fn.__globals__.pop(alias, None)
            _import_source_cache.pop(name, None)
            torch._dynamo.reset()

    def test_import_alias_of_a_dotted_name_is_the_module_import_name_resolves(self):
        # IMPORT_NAME hands import_source the top-level name for an empty
        # fromlist, the module __import__ returns then, and the full dotted
        # name otherwise. Each arm's slot holds a stale module of that arm's
        # name; each is rebound to the live module under the matching
        # sys.modules key, and the first arm leaves the second's alias unbound.
        pkg_name = "torch_test_package_import_alias_pkg"
        helper_name = f"{pkg_name}.helper"
        pkg_alias = f"__import_{pkg_name}"
        helper_alias = f"__import_{pkg_name}_dot_helper"
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = []
        helper = types.ModuleType(helper_name)
        helper.VALUE = 3
        pkg.helper = helper
        args = (torch.randn(3, 2),)

        def fn_pkg(x):
            import torch_test_package_import_alias_pkg.helper

            return x + torch_test_package_import_alias_pkg.helper.VALUE

        def fn_from(x):
            from torch_test_package_import_alias_pkg.helper import VALUE

            return x + VALUE

        try:
            sys.modules[pkg_name] = pkg
            sys.modules[helper_name] = helper
            fn_pkg.__globals__[pkg_alias] = types.ModuleType(pkg_name)
            compiled_fn = torch.compile(fn_pkg, backend="eager", fullgraph=True)
            self.assertEqual(fn_pkg(*args), compiled_fn(*args))
            self.assertIs(fn_pkg.__globals__[pkg_alias], pkg)
            self.assertNotIn(helper_alias, fn_pkg.__globals__)

            torch._dynamo.reset()
            fn_from.__globals__[helper_alias] = types.ModuleType(helper_name)
            compiled_fn = torch.compile(fn_from, backend="eager", fullgraph=True)
            self.assertEqual(fn_from(*args), compiled_fn(*args))
            self.assertIs(fn_from.__globals__[helper_alias], helper)
        finally:
            sys.modules.pop(pkg_name, None)
            sys.modules.pop(helper_name, None)
            fn_pkg.__globals__.pop(pkg_alias, None)
            fn_pkg.__globals__.pop(helper_alias, None)
            _import_source_cache.pop(pkg_name, None)
            _import_source_cache.pop(helper_name, None)
            torch._dynamo.reset()

    def test_import_alias_accepts_a_stale_torch_package_module(self):
        # A torch.package module is named by its mangled <torch_package_N>.name,
        # which is never a sys.modules key; import_source resolves it through
        # the importer registry keyed by that same name, so the registry's
        # module is the one the name resolves to now, and a same-named module a
        # writer left in the slot is accepted and replaced by it. The alias is
        # reached through an inlined call: the packaged function's global read
        # roots at its own module, not the frame's.
        name = "torch_test_package_import_alias_packaged"
        path = os.path.join(self.path(), "alias.pt")
        src = "SCALE = 2\n\ndef helper(x):\n    return x * SCALE\n"
        with torch.package.PackageExporter(path) as exp:
            exp.save_source_string(name, src)
        packaged = torch.package.PackageImporter(path).import_module(name)
        mangled = packaged.__name__
        self.assertNotIn(mangled, sys.modules)
        alias = mangled.replace(">", "_").replace("<", "_").replace(".", "_dot_")
        args = (torch.randn(3, 2),)

        def fn(x):
            return packaged.helper(x) + 1

        try:
            fn.__globals__[alias] = types.ModuleType(mangled)
            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(fn(*args), compiled_fn(*args))
            self.assertIs(fn.__globals__[alias], packaged)
            # Never cached: the registry is a WeakValueDictionary and the cache
            # is never cleared, so an entry would pin the packaged module.
            self.assertNotIn(mangled, _import_source_cache)
        finally:
            fn.__globals__.pop(alias, None)
            torch._dynamo.reset()

    def test_import_alias_the_trace_left_alone_is_recorded_for_install(self):
        # Under a CompilePackage: install() bound the live module after a
        # handover, the trace resolves that same live entry and its write
        # leaves the binding as it found it. The alias is recorded all the
        # same, because the graph's guards read through it whoever bound it,
        # and a fresh process has no writer but install(): there it binds
        # importlib.import_module(name), that process's own live entry -- here
        # a third module, distinct from the one this trace found bound and
        # built its graph from -- and the loaded guards are rebuilt against
        # it, so the loaded function serves with no recompile. Without the
        # record install() binds nothing and the loaded guards read an alias
        # no writer has bound.
        ctx = DiskDynamoStore()
        name = "torch_test_package_import_alias_recorded"
        alias = f"__import_{name}"
        old = types.ModuleType(name)
        old.VALUE = 2
        new = types.ModuleType(name)
        new.VALUE = 7
        third = types.ModuleType(name)
        third.VALUE = 99

        def fn(x):
            import torch_test_package_import_alias_recorded as shim

            return x + shim.VALUE

        def fn2(x):
            import torch_test_package_import_alias_recorded as shim

            return x * shim.VALUE

        args = (torch.randn(3, 2),)
        second = os.path.join(self.path(), "second")
        try:
            sys.modules[name] = old
            package = CompilePackage(fn)
            compiled_fn = torch._dynamo.optimize(backend="eager", package=package)(fn)
            self.assertEqual(fn(*args), compiled_fn(*args))
            for backend_id, backend in package.cached_backends.items():
                ctx.record_eager_backend(backend_id, backend)
            ctx.save_package(package, self.path())
            torch._dynamo.reset()

            sys.modules[name] = new
            package, backends = ctx.load_package(fn, self.path())
            package.install(backends)
            self.assertIs(fn.__globals__[alias], new)

            pkg2 = CompilePackage(fn2)
            compiled_fn2 = torch._dynamo.optimize(backend="eager", package=pkg2)(fn2)
            self.assertEqual(fn2(*args), compiled_fn2(*args))
            self.assertIs(fn.__globals__[alias], new)
            self.assertEqual(pkg2._codes[fn2.__code__].import_sources[alias], name)
            for backend_id, backend in pkg2.cached_backends.items():
                ctx.record_eager_backend(backend_id, backend)
            ctx.save_package(pkg2, second)
            torch._dynamo.reset()

            del fn.__globals__[alias]
            sys.modules[name] = third
            pkg2, backends = ctx.load_package(fn2, second)
            compiled_fn2 = torch._dynamo.optimize(backend="eager", package=pkg2)(fn2)
            pkg2.install(backends)
            self.assertIs(fn.__globals__[alias], third)
            with torch.compiler.set_stance("fail_on_recompile"):
                # The artifact carries the VALUE its graph was built from and
                # not the module: the loaded EQUALS_MATCH takes its expected
                # value from the module install() bound, so the loaded function
                # serves x * 7 against third.VALUE == 99 with no recompile.
                # That is what a module pickled by name gives any loaded guard;
                # the record's part is that third is behind the alias at all,
                # where the guard reads it live and a change to it fails it.
                self.assertEqual(compiled_fn2(*args), args[0] * 7)
                self.assertNotEqual(fn2(*args), compiled_fn2(*args))
                third.VALUE = 5
                guard = re.escape(f"G['{alias}'].VALUE == 99")
                with self.assertRaisesRegex(RuntimeError, guard):
                    compiled_fn2(*args)
        finally:
            sys.modules.pop(name, None)
            fn.__globals__.pop(alias, None)
            _import_source_cache.pop(name, None)
            torch._dynamo.reset()

    def test_import_alias_of_an_empty_slot_reads_an_attribute_the_cached_module_lacks(
        self,
    ):
        # The loud form of the parent's staleness, from an empty slot: the
        # trace reads an attribute off the live module, and the guard on it is
        # built by reading that attribute through the alias. The parent bound
        # the cached module there, which lacks the attribute, so Guard.create
        # raised AttributeError, _compile_inner wrapped it, and the compile died
        # with no eager fallback. The alias now binds the live entry, and the
        # read compiles and agrees with eager.
        name = "torch_test_package_import_alias_stale_attr"
        alias = f"__import_{name}"
        old = types.ModuleType(name)
        old.VALUE = 2
        new = types.ModuleType(name)
        new.VALUE = 7
        new.EXTRA = 3

        def fn(x):
            import torch_test_package_import_alias_stale_attr as shim

            return x + shim.VALUE

        def fn3(x):
            import torch_test_package_import_alias_stale_attr as shim

            return x - shim.EXTRA

        args = (torch.randn(3, 2),)
        try:
            sys.modules[name] = old
            torch.compile(fn, backend="eager", fullgraph=True)(*args)
            self.assertIs(fn.__globals__[alias], old)
            del fn.__globals__[alias]
            torch._dynamo.reset()

            sys.modules[name] = new
            self.assertEqual(fn3(*args), torch.compile(fn3, backend="eager")(*args))
            self.assertIs(fn.__globals__[alias], new)
        finally:
            sys.modules.pop(name, None)
            fn.__globals__.pop(alias, None)
            _import_source_cache.pop(name, None)
            torch._dynamo.reset()

    def test_import_alias_nn_hook_guards_read_the_live_entry_the_alias_binds(self):
        # nn_modules_globals_vt's value is the defining module, whose dicts
        # _call_impl reads through its own __globals__, rooted at the alias,
        # which binds the live entry: a shim under the name, so the hook-dict
        # guards read the shim. With no hook registered they agree with eager;
        # with a forward hook on the defining module the CLOSURE_MATCH on the
        # handle's id subscripts the shim's empty dict as the guard is built
        # and raises, with no eager fallback. Pinned as the state the commit
        # directly above this one changes; the cache entry the compile writes
        # is popped in finally so the shim cannot be left in it.
        name = "torch.nn.modules.module"
        alias = "__import_torch_dot_nn_dot_modules_dot_module"
        real = sys.modules[name]

        class M(torch.nn.Module):
            def forward(self, x):
                return x + 1

        mod = M()

        def fn(x):
            return mod(x)

        args = (torch.randn(3, 2),)
        torch.compile(fn, backend="eager", fullgraph=True)(*args)
        torch._dynamo.reset()
        shim = types.ModuleType(name)
        for attr in (
            "_global_backward_pre_hooks",
            "_global_backward_hooks",
            "_global_forward_hooks",
            "_global_forward_pre_hooks",
        ):
            setattr(shim, attr, {})
        handle = None
        try:
            sys.modules[name] = shim
            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(fn(*args), compiled_fn(*args))
            self.assertIs(fn.__globals__[alias], shim)
            torch._dynamo.reset()
            handle = real.register_module_forward_hook(lambda m, i, o: o * 2)
            # The dict key is the handle's id, a process-global counter.
            with (
                self.assertLogs("torch._guards", level="ERROR") as logs,
                self.assertRaisesRegex(InternalTorchDynamoError, r"KeyError: \d+"),
            ):
                torch.compile(fn, backend="eager", fullgraph=True)(*args)
            guard = (
                rf"Error while creating guard:\nName: \"G\['{alias}'\]"
                rf"\._global_forward_hooks\[{handle.id}\]\"\n\s+Source: global"
                r"\n\s+Create Function: CLOSURE_MATCH"
            )
            self.assertRegex("\n".join(logs.output), guard)
            self.assertIs(fn.__globals__[alias], shim)
        finally:
            if handle is not None:
                handle.remove()
            sys.modules[name] = real
            fn.__globals__.pop(alias, None)
            _import_source_cache.pop(name, None)
            torch._dynamo.reset()

    def test_import_alias_of_an_empty_slot_is_the_live_entry_not_the_cache(self):
        # An empty slot binds the live entry, not what the process's earlier
        # trace of the name bound, which _import_source_cache still holds after
        # a handover: the graph is specialized on the live module's VALUE, which
        # __import__ pushed, and the guards this alias roots must read the same
        # module. A change to the live module recompiles; a change to the cached
        # one, which nothing guards, does not. The parent bound the cached
        # module into an empty slot, and there the first change went unseen
        # while the second recompiled. Once the entry is gone the cache serves
        # the module the second compile bound, not the first the process
        # resolved: it holds what import_source last bound under the name.
        name = "torch_test_package_import_alias_handed_over"
        alias = f"__import_{name}"
        old = types.ModuleType(name)
        old.VALUE = 2
        new = types.ModuleType(name)
        new.VALUE = 7

        def fn(x):
            import torch_test_package_import_alias_handed_over as shim

            return x + shim.VALUE

        args = (torch.randn(3, 2),)
        try:
            sys.modules[name] = old
            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(fn(*args), compiled_fn(*args))
            self.assertIs(fn.__globals__[alias], old)
            del fn.__globals__[alias]
            torch._dynamo.reset()

            sys.modules[name] = new
            cnt = CompileCounter()
            compiled_fn = torch.compile(fn, backend=cnt, fullgraph=True)
            self.assertEqual(fn(*args), compiled_fn(*args))
            self.assertIs(fn.__globals__[alias], new)
            self.assertEqual(cnt.frame_count, 1)
            new.VALUE = 8
            self.assertEqual(fn(*args), compiled_fn(*args))
            self.assertEqual(cnt.frame_count, 2)
            old.VALUE = 11
            self.assertEqual(fn(*args), compiled_fn(*args))
            self.assertEqual(cnt.frame_count, 2)

            # fn's import statement cannot run with the entry gone, in eager or
            # in the trace, so a probe whose bytecode never imports the name
            # drives import_source from a comptime callback on the live
            # translator, as IMPORT_NAME would.
            self.assertIs(_import_source_cache[name], new)
            sys.modules.pop(name)
            del fn.__globals__[alias]
            torch._dynamo.reset()
            seen = []

            def resolve(ctx):
                tx = ctx._i_will_not_complain_if_bc_breaks_InstructionTranslator()
                seen.append(tx.import_source(name, True))

            def probe(x):
                comptime(resolve)
                return x + 1

            compiled_probe = torch.compile(probe, backend="eager", fullgraph=True)
            self.assertEqual(probe(*args), compiled_probe(*args))
            (source,) = seen
            self.assertEqual(source.global_name, alias)
            self.assertIs(fn.__globals__[alias], new)
        finally:
            sys.modules.pop(name, None)
            fn.__globals__.pop(alias, None)
            _import_source_cache.pop(name, None)
            torch._dynamo.reset()

    def test_import_alias_taken_by_a_non_module_graph_breaks(self):
        # The relaxed check still catches what it was written for: the alias name
        # holding something other than the module it names -- a non-module, or a
        # module of another name, which is the state two module names mangling
        # onto one alias leave it in. The condition is the user's globals, so it
        # is a graph break, not an internal error, and the slot is left alone;
        # a refusal does not remember the name in _import_source_cache either.
        name = "torch_test_package_import_alias_taken"
        alias = f"__import_{name}"
        module = types.ModuleType(name)
        module.VALUE = 1
        nameless = types.ModuleType("nameless")
        del nameless.__dict__["__name__"]
        nameless.__dict__["__getattr__"] = _import_alias_getattr_boom

        def fn(x):
            import torch_test_package_import_alias_taken as taken

            return x + taken.VALUE

        scope = fn.__globals__["__name__"]
        cases = (
            ("not a module", f"already bound to a str in the globals of {scope}"),
            (types.ModuleType("other.name"), "bound to a module named other.name"),
            (nameless, "already bound to a module in the globals"),
        )
        # The first hint names the module whose globals hold the slot -- the
        # root frame's, which an inlined callee's own module is not.
        hint = f"Remove or rename the global {alias} in module {scope}."
        args = (torch.randn(3, 2),)
        try:
            sys.modules[name] = module
            for bound, expected in cases:
                with self.subTest(expected=expected):
                    torch._dynamo.reset()
                    _import_source_cache.pop(name, None)
                    fn.__globals__[alias] = bound
                    with self.assertRaisesRegex(
                        Unsupported, f"alias {alias} for {name}.*{re.escape(expected)}"
                    ) as cm:
                        torch.compile(fn, backend="eager", fullgraph=True)(*args)
                    self.assertIn(hint, str(cm.exception))
                    self.assertIs(fn.__globals__[alias], bound)
                    self.assertNotIn(name, _import_source_cache)
        finally:
            sys.modules.pop(name, None)
            fn.__globals__.pop(alias, None)
            _import_source_cache.pop(name, None)
            torch._dynamo.reset()

    def test_import_alias_graph_break_is_cached_until_reset(self):
        # Without fullgraph the import is the frame's first work, so there is
        # no checkpoint to compile up to and the whole frame is skipped -- and
        # stays skipped, the alias left alone, after the global is removed:
        # nothing guards it, so only torch._dynamo.reset() makes Dynamo trace
        # the frame again, as the registry entry's third hint says. The refusal
        # does not remember the name in _import_source_cache; the compile that
        # binds the alias does, which is why the pop in finally stays.
        name = "torch_test_package_import_alias_cached"
        alias = f"__import_{name}"
        module = types.ModuleType(name)
        module.VALUE = 1

        def fn(x):
            import torch_test_package_import_alias_cached as taken

            return x + taken.VALUE

        args = (torch.randn(3, 2),)
        try:
            sys.modules[name] = module
            fn.__globals__[alias] = "not a module"
            cnt = CompileCounter()
            skipped = torch.compile(fn, backend=cnt)
            self.assertEqual(fn(*args), skipped(*args))
            self.assertEqual(cnt.frame_count, 0)
            self.assertEqual(fn.__globals__[alias], "not a module")
            self.assertNotIn(name, _import_source_cache)
            del fn.__globals__[alias]
            self.assertEqual(fn(*args), skipped(*args))
            self.assertEqual(cnt.frame_count, 0)
            self.assertNotIn(alias, fn.__globals__)
            torch._dynamo.reset()
            self.assertEqual(fn(*args), skipped(*args))
            self.assertEqual(cnt.frame_count, 1)
            self.assertIs(fn.__globals__[alias], module)
            self.assertIs(_import_source_cache[name], module)
        finally:
            sys.modules.pop(name, None)
            _import_source_cache.pop(name, None)
            fn.__globals__.pop(alias, None)
            torch._dynamo.reset()

    @torch._dynamo.config.patch(record_runtime_overhead=True)
    def test_import_alias_taken_at_codegen_is_a_hard_error(self):
        # import_source is also called after tracing, from codegen: with
        # record_runtime_overhead on (its default, patched so this keeps
        # driving that site if the default moves), make_call_generated_code
        # resolves torch.autograd.profiler for the pregraph marker. compile_subgraph
        # has no instruction left to break at, so an Unsupported raised there
        # would be no graph break: under the default stance the frame would be
        # skipped to eager, the backend having already run, with nothing said
        # at default log levels. So a collision found by any caller but
        # IMPORT_NAME stays the hard error it was, naming alias, module and
        # offender. The function runs over a throwaway globals dict, so the
        # real alias is planted without touching this module's globals.
        alias = "__import_torch_dot_autograd_dot_profiler"
        scope = {"__builtins__": builtins, "__name__": "throwaway"}
        scope[alias] = "not a module"

        def template(x):
            return x + 1

        fn = types.FunctionType(template.__code__, scope, "fn")
        cnt = CompileCounter()
        args = (torch.randn(3, 2),)
        try:
            refused = f"alias {alias} for torch.autograd.profiler is already bound to a str in the globals of throwaway"
            with self.assertRaisesRegex(AssertionError, refused):
                torch.compile(fn, backend=cnt)(*args)
            self.assertEqual(cnt.frame_count, 1)
            self.assertEqual(scope[alias], "not a module")
        finally:
            torch._dynamo.reset()

    def test_import_alias_taken_in_a_torch_package_slot_is_a_hard_error(self):
        # The torch_package arm is reached through get_globals_source_and_value,
        # for an inlined call into a packaged module: the alias is minted from
        # the mangled name in the outer frame's globals, and nothing but
        # import_source binds one (a mangled name is not importable, so
        # install() cannot). That caller does not pass graph_break_ok, so a
        # collision there is the hard error, with the three facts in it.
        name = "torch_test_package_import_alias_packaged_taken"
        path = os.path.join(self.path(), "alias.pt")
        src = "SCALE = 2\n\ndef helper(x):\n    return x * SCALE\n"
        with torch.package.PackageExporter(path) as exp:
            exp.save_source_string(name, src)
        packaged = torch.package.PackageImporter(path).import_module(name)
        mangled = packaged.__name__
        alias = mangled.replace(">", "_").replace("<", "_").replace(".", "_dot_")
        args = (torch.randn(3, 2),)

        def fn(x):
            return packaged.helper(x) + 1

        try:
            fn.__globals__[alias] = "not a module"
            refused = f"alias {re.escape(alias)} for {re.escape(mangled)} is already bound to a str"
            with self.assertRaisesRegex(AssertionError, refused):
                torch.compile(fn, backend="eager")(*args)
            self.assertEqual(fn.__globals__[alias], "not a module")
        finally:
            fn.__globals__.pop(alias, None)
            torch._dynamo.reset()

    def test_import_alias_the_trace_refused_is_not_recorded_for_install(self):
        # Two ops precede the import, so the graph break has a checkpoint: the
        # trace restarts and the frame is compiled up to it, and the
        # CompilePackage entry made for the frame outlives the abandoned trace
        # and is what gets saved. A record written before the check would ship
        # in that entry, and install() binds every recorded alias
        # unconditionally -- record_only_if_new gates only the uninstall
        # bookkeeping -- over the very global the check refused to touch.
        # Nothing is recorded for an alias the trace did not bind, in the
        # package or in _import_source_cache, so there is nothing to pop.
        ctx = DiskDynamoStore()
        name = "torch_test_package_import_alias_refused"
        alias = f"__import_{name}"
        module = types.ModuleType(name)
        module.VALUE = 1
        foreign = "not a module"

        def fn(x):
            y = x + 1
            y = y * 2
            import torch_test_package_import_alias_refused as taken

            return y + taken.VALUE

        args = (torch.randn(3, 2),)
        try:
            sys.modules[name] = module
            fn.__globals__[alias] = foreign
            package = CompilePackage(fn)
            compiled_fn = torch._dynamo.optimize(backend="eager", package=package)(fn)
            self.assertEqual(fn(*args), compiled_fn(*args))
            self.assertIs(fn.__globals__[alias], foreign)
            self.assertNotIn(name, _import_source_cache)
            self.assertEqual(len(package._codes[fn.__code__].guarded_codes), 1)
            for entry in package._codes.values():
                self.assertNotIn(alias, entry.import_sources)
            for backend_id, backend in package.cached_backends.items():
                ctx.record_eager_backend(backend_id, backend)
            ctx.save_package(package, self.path())
            torch._dynamo.reset()
            package, backends = ctx.load_package(fn, self.path())
            package.install(backends)
            self.assertIs(fn.__globals__[alias], foreign)
            self.assertNotIn(name, _import_source_cache)
        finally:
            sys.modules.pop(name, None)
            fn.__globals__.pop(alias, None)
            torch._dynamo.reset()

    def test_import_alias_check_does_not_run_a_module_getattribute(self):
        # Both slots hold a module whose class raises on any read import_source
        # makes: the one __import__ resolves and the stale one a prior writer
        # left. The resolution reads nothing off the live one, and the check
        # reads each module's name out of its instance dict, so neither runs
        # the hook; __import__ and PythonModuleVariable read __spec__ and
        # __name__ off the live one by design, and the frame reads nothing else
        # off it. The hook recognizes import_source by the frame's function
        # name and module, pinned against the real function so a rename fails
        # here instead of letting every read through.
        walked = InstructionTranslatorBase.import_source.__wrapped__
        where = (walked.__code__.co_name, walked.__module__)
        self.assertEqual(where, ("import_source", "torch._dynamo.symbolic_convert"))
        name = "torch_test_package_import_alias_hooked"
        alias = f"__import_{name}"
        live = _ImportAliasHookedModule(name)
        stale = _ImportAliasHookedModule(name)
        args = (torch.randn(3, 2),)

        def fn(x):
            import torch_test_package_import_alias_hooked as hooked

            del hooked
            return x + 1

        try:
            sys.modules[name] = live
            fn.__globals__[alias] = stale
            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(fn(*args), compiled_fn(*args))
            self.assertIs(fn.__globals__[alias], live)
        finally:
            sys.modules.pop(name, None)
            fn.__globals__.pop(alias, None)
            _import_source_cache.pop(name, None)
            torch._dynamo.reset()

    def test_import_alias_is_not_bound_to_a_non_module_import(self):
        # sys.modules accepts any object and __import__ hands it back verbatim.
        # IMPORT_NAME rejects it before import_source binds the alias, so the
        # traced globals never hold the non-module and a second trace after the
        # entry is swapped for another non-module graph breaks the same way
        # instead of tripping the alias check on two nameless objects.
        name = "torch_test_package_import_alias_non_module"
        alias = f"__import_{name}"
        args = (torch.randn(3, 2),)

        def fn(x):
            import torch_test_package_import_alias_non_module as taken

            return x + taken.VALUE

        try:
            for entry in (object(), object()):
                torch._dynamo.reset()
                sys.modules[name] = entry
                with self.assertRaisesRegex(Unsupported, "Bad import result"):
                    torch.compile(fn, backend="eager", fullgraph=True)(*args)
                self.assertNotIn(alias, fn.__globals__)
        finally:
            sys.modules.pop(name, None)
            fn.__globals__.pop(alias, None)
            torch._dynamo.reset()

    def test_import_alias_check_with_a_non_module_value_accepts_the_key_alone(self):
        # sys.modules accepts any object and import_source serves the live entry
        # as is, and the callers with no module-type test in front of it pass
        # torch's and the stdlib's own names or a class's __module__, so a
        # non-module value reaches it only once one of those entries has been
        # replaced by a non-module. That value has no __name__ to accept, so the
        # accepted names are the key alone: a nameless module in the slot is
        # refused rather than matched against None, and a module named for the
        # key is accepted and, the value being the live entry, replaced by it.
        # No traced bytecode reaches that arm, so a comptime callback calls
        # import_source on the live translator mid-trace, as IMPORT_NAME does
        # (graph_break_ok), with the frame's real globals and output behind it,
        # and snapshots the record there: the translator is not read once the
        # compile has returned. The two arms run on two translators, as in
        # production: a refusal raises Unsupported out to skip or restart the
        # whole frame, so one translator never reaches the name twice.
        key = "torch_test_package_import_alias_non_module_value"
        alias = f"__import_{key}"
        value = types.SimpleNamespace(VALUE=1)
        nameless = types.ModuleType("nameless")
        del nameless.__dict__["__name__"]
        seen = []

        def resolve(ctx):
            tx = ctx._i_will_not_complain_if_bc_breaks_InstructionTranslator()
            try:
                source = tx.import_source(key, True)
            finally:
                seen.append(dict(tx.output.import_sources))
            seen.append(source)

        def fn(x):
            comptime(resolve)
            return x + 1

        args = (torch.randn(3, 2),)
        try:
            sys.modules[key] = value
            fn.__globals__[alias] = nameless
            refused = f"alias {alias} for {key}.*bound to a module in the globals"
            with self.assertRaisesRegex(Unsupported, refused):
                torch.compile(fn, backend="eager", fullgraph=True)(*args)
            self.assertEqual(len(seen), 1)
            (recorded,) = seen
            self.assertIs(fn.__globals__[alias], nameless)
            self.assertNotIn(alias, recorded)

            seen.clear()
            torch._dynamo.reset()
            fn.__globals__[alias] = types.ModuleType(key)
            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(fn(*args), compiled_fn(*args))
            self.assertEqual(len(seen), 2)
            recorded, source = seen
            self.assertEqual(source.global_name, alias)
            self.assertIs(fn.__globals__[alias], value)
            self.assertEqual(recorded[alias], key)
            # Bound but not remembered: a later trace with the entry gone must
            # not be served a non-module from the cache.
            self.assertNotIn(key, _import_source_cache)
        finally:
            sys.modules.pop(key, None)
            _import_source_cache.pop(key, None)
            fn.__globals__.pop(alias, None)
            torch._dynamo.reset()

    @torch._dynamo.config.patch(record_runtime_overhead=True)
    def test_import_alias_of_a_name_removed_from_sys_modules_keeps_its_module(self):
        # Codegen resolves names the traced bytecode never imported -- here
        # torch.autograd.profiler, for the runtime-overhead record in every
        # compiled frame's prologue -- so nothing has put a removed entry back
        # by the time import_source runs. The alias keeps the module the program
        # holds, what import_source last bound while the name was live, instead
        # of re-importing inside the trace (a second copy of the module) or, for
        # None, the blocked-module sentinel, raising. The spy pins the mechanism
        # itself: whatever else the compile imports, it never asks for this
        # name. The alias is deleted before each compile so the binding checked
        # is that compile's write, which is also what pins that a compile ran.
        name = "torch.autograd.profiler"
        alias = "__import_torch_dot_autograd_dot_profiler"
        profiler = sys.modules[name]

        def fn(x):
            return x + 1

        args = (torch.randn(3, 2),)
        real_import = importlib.import_module
        spy = mock.patch.object(importlib, "import_module", wraps=real_import)
        try:
            compiled_fn = torch.compile(fn, backend="eager")
            self.assertEqual(fn(*args), compiled_fn(*args))
            self.assertIs(fn.__globals__[alias], profiler)
            for blocked in (False, True):
                with self.subTest(blocked=blocked):
                    torch._dynamo.reset()
                    del fn.__globals__[alias]
                    with spy as import_module:
                        if blocked:
                            sys.modules[name] = None
                        else:
                            del sys.modules[name]
                        self.assertEqual(fn(*args), compiled_fn(*args))
                        asked = [call.args[0] for call in import_module.call_args_list]
                        self.assertNotIn(name, asked)
                        if blocked:
                            self.assertIsNone(sys.modules[name])
                        else:
                            self.assertNotIn(name, sys.modules)
                    self.assertIs(fn.__globals__[alias], profiler)
        finally:
            sys.modules[name] = profiler
            fn.__globals__.pop(alias, None)
            torch._dynamo.reset()

    def test_import_alias_of_a_name_gone_rebinds_the_installed_module_to_the_cache(
        self,
    ):
        # install() and a trace resolve the name independently, so the alias
        # can hold install()'s same-named module while _import_source_cache
        # holds the one the trace bound and built its guards against. With the
        # name then gone from sys.modules, or blocked with None, neither is the
        # live entry and the cached one replaces the installed one; the parent
        # left the installed one in the slot. fn's import statement cannot run
        # with the entry gone, so a probe whose bytecode never imports the name
        # drives import_source from a comptime callback on the live translator.
        ctx = DiskDynamoStore()
        name = "torch_test_package_import_alias_gone"
        alias = f"__import_{name}"
        old = types.ModuleType(name)
        old.VALUE = 2
        mid = types.ModuleType(name)
        mid.VALUE = 7

        def fn(x):
            import torch_test_package_import_alias_gone as shim

            return x + shim.VALUE

        def resolve(ctx):
            tx = ctx._i_will_not_complain_if_bc_breaks_InstructionTranslator()
            tx.import_source(name, True)

        def probe(x):
            comptime(resolve)
            return x + 1

        args = (torch.randn(3, 2),)
        try:
            sys.modules[name] = old
            package = CompilePackage(fn)
            compiled_fn = torch._dynamo.optimize(backend="eager", package=package)(fn)
            self.assertEqual(fn(*args), compiled_fn(*args))
            for backend_id, backend in package.cached_backends.items():
                ctx.record_eager_backend(backend_id, backend)
            ctx.save_package(package, self.path())
            torch._dynamo.reset()

            sys.modules[name] = mid
            package, backends = ctx.load_package(fn, self.path())
            package.install(backends)
            self.assertIs(fn.__globals__[alias], mid)
            for blocked in (False, True):
                with self.subTest(blocked=blocked):
                    torch._dynamo.reset()
                    fn.__globals__[alias] = mid
                    if blocked:
                        sys.modules[name] = None
                    else:
                        sys.modules.pop(name)
                    compiled = torch.compile(probe, backend="eager", fullgraph=True)
                    self.assertEqual(probe(*args), compiled(*args))
                    self.assertIs(fn.__globals__[alias], old)
            self.assertIs(_import_source_cache[name], old)
        finally:
            sys.modules.pop(name, None)
            fn.__globals__.pop(alias, None)
            _import_source_cache.pop(name, None)
            torch._dynamo.reset()

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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
