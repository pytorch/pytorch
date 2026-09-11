# Owner(s): ["module: dynamo"]

import functools
import gc
import importlib
import os
import sys
import tempfile
import types
import unittest

import torch
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.onnx.operators
import torch.utils.cpp_extension
from torch._C._dynamo.eval_frame import _debug_get_precompile_entries
from torch._dynamo.package import CompilePackage, DiskDynamoStore, DynamoCache
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.testing import reduce_to_scalar_loss
from torch._dynamo.utils import CleanupManager
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


def compute_loss_helper(x):
    return reduce_to_scalar_loss(x)


def compiled_region_with_backend_id_for_package_test():
    return __compiled_fn_0_00000000_0000_0000_0000_000000000000()  # noqa: F821


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
        self.assertEqual(resume_of(package.cache_entry()).guarded_codes, [])
        self.assertEqual(resume_of(package.cache_entry()).backend_ids, [])
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
