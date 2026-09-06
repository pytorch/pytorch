# Owner(s): ["module: dynamo"]

import dataclasses
import gc
import importlib
import inspect
import os
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
from torch._C._dynamo.eval_frame import (
    _debug_get_precompile_entries,
    get_code_exec_strategy,
)
from torch._dynamo.exc import PackageError
from torch._dynamo.package import (
    _current_cpu_codegen_target,
    _defining_module_name,
    _MODULE_KEY_BY_FILE,
    _rename_globals,
    _scan_sys_modules_for_file,
    CompilePackage,
    DiskDynamoStore,
    DynamoCache,
    load_guards_state,
    SourceInfo,
    SystemInfo,
)
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.testing import reduce_to_scalar_loss
from torch._dynamo.types import FrameAction
from torch._dynamo.utils import CleanupManager, counters
from torch._functorch import config as functorch_config
from torch._inductor import cpu_vec_isa
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
    def __init__(self):
        self.flag = 2.0

    def __reduce__(self):
        raise RuntimeError("config cannot pickle")


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

    def test_mixed_device_capture_records_cpu_codegen_target(self):
        # A mixed cpu+accelerator capture still holds native CPU code, so the
        # codegen target must be recorded and compared even though the
        # collapsed device_type reads as the accelerator. The graph is
        # fabricated (never run), so no accelerator is needed.
        if _current_cpu_codegen_target() is None:
            self.skipTest("no CPU codegen target on this host")

        def fn(x):
            return x + 1

        package = CompilePackage(fn)
        graph = torch.fx.Graph()
        node = graph.placeholder("x")
        node.meta["val"] = torch.empty(2)
        graph.call_function(torch.ops.aten.add.Tensor, (node, torch.device("cuda")))
        package.update_device_type(graph)
        self.assertEqual(package._device_types, {"cpu", "cuda"})

        entry = package.cache_entry()
        self.assertEqual(entry.device_type, "cuda")
        self.assertEqual(entry.device_types, frozenset(("cpu", "cuda")))
        self.assertIsNotNone(entry.system_info.cpu_codegen_target)

        stale = ("mips", "DEFAULT", 128, ("INVALID",), None, "INVALID")
        entry.system_info = dataclasses.replace(
            entry.system_info, cpu_codegen_target=stale
        )
        with self.assertRaisesRegex(RuntimeError, "CPU codegen target"):
            entry.check_versions()

    def test_cpu_codegen_target_requires_the_host_to_pick_the_same_isa(self):
        # The kernel source is tiled for the ISA picked at codegen and compiled
        # with the ISA picked on the loading host, so the two must be equal. A
        # wider host is not a superset: its masked loads zero-fill the lanes the
        # narrower tiling never touches, and unmasked reductions read them.
        def check(cached_target, host_target):
            base = SystemInfo.current(cpu_codegen=False)
            cached = dataclasses.replace(base, cpu_codegen_target=cached_target)
            with patch(
                "torch._dynamo.package._current_cpu_codegen_target",
                return_value=host_target,
            ):
                cached.check_compatibility(SystemInfo.current())

        avx2 = ("x86_64", "avx2", 256, ("CPU_CAPABILITY_AVX2",), None, None)
        avx512 = ("x86_64", "avx512", 512, ("CPU_CAPABILITY_AVX512",), None, None)
        neon = (
            "aarch64",
            "asimd",
            128,
            ("CPU_CAPABILITY_NEON", "AT_BUILD_ARM_VEC256_WITH_SLEEF"),
            None,
            None,
        )
        sve128 = ("aarch64", "asimd", 128, ("CPU_CAPABILITY_SVE128",), None, None)
        check(avx2, avx2)
        with self.assertRaisesRegex(
            RuntimeError, "generated for vector ISA 'avx2'.*for 'avx512'"
        ):
            check(avx2, avx512)
        with self.assertRaisesRegex(
            RuntimeError, "generated for vector ISA 'avx512'.*for 'avx2'"
        ):
            check(avx512, avx2)
        with self.assertRaisesRegex(
            RuntimeError, "machine 'aarch64', this host is 'x86_64'"
        ):
            check(neon, avx2)
        # NEON and SVE128 share both the name "asimd" and a 128-bit width, so
        # only the build macro tells them apart.
        with self.assertRaisesRegex(RuntimeError, "vector ISA 'asimd'"):
            check(neon, sve128)
        with self.assertRaisesRegex(RuntimeError, "simdlen=256, this host uses None"):
            check(("x86_64", "avx2", 256, ("CPU_CAPABILITY_AVX2",), 256, None), avx2)
        with self.assertRaisesRegex(RuntimeError, "no usable CPU codegen target"):
            check(avx2, None)

    def test_no_valid_vec_isa_records_no_cpu_codegen_target(self):
        # pick_vec_isa never raises for a missing compiler; it returns
        # invalid_vec_isa, which must read as "no target", not as a target
        # named INVALID_VEC_ISA that only an equally broken host would match.
        with patch.object(cpu_vec_isa, "valid_vec_isa_list", return_value=[]):
            self.assertIsNone(_current_cpu_codegen_target())

    def test_sve_widths_do_not_collide_in_the_codegen_fingerprint(self):
        # VecSVE(128) and VecSVE(256) both stringify to "asimd", so the ISA name
        # alone cannot tell a 128-bit tiling from a 256-bit one. The fingerprint
        # records bit_width() so the two do not compare equal and a kernel tiled
        # for one width is refused on a host that picks the other.
        narrow = cpu_vec_isa.VecSVE(_bit_width=128)
        wide = cpu_vec_isa.VecSVE(_bit_width=256)
        self.assertEqual(str(narrow), str(wide))
        with patch.object(cpu_vec_isa, "pick_vec_isa", return_value=narrow):
            narrow_target = _current_cpu_codegen_target()
        with patch.object(cpu_vec_isa, "pick_vec_isa", return_value=wide):
            wide_target = _current_cpu_codegen_target()
        self.assertEqual(narrow_target[1], wide_target[1])
        self.assertNotEqual(narrow_target[2], wide_target[2])
        self.assertNotEqual(narrow_target, wide_target)

    @torch._dynamo.config.patch(caching_precompile=True, strict_precompile=False)
    def test_eager_backend_entry_is_exempt_from_the_codegen_target(self):
        def fn(x):
            return x + 1

        def custom_backend(gm, example_inputs):
            return gm

        with patch(
            "torch._dynamo.package._current_cpu_codegen_target",
            side_effect=AssertionError("toolchain probe ran for an eager backend"),
        ):
            torch.compile(fn, backend="eager")(torch.randn(3))
            (entry,) = PrecompileContext._dynamo_cache_entries.values()
        self.assertFalse(entry.requires_native_backend_compatibility)
        self.assertIsNone(entry.system_info.cpu_codegen_target)

        torch._dynamo.reset()
        PrecompileContext.clear()
        # A user's own callable may emit anything, so it counts as native.
        torch.compile(fn, backend=custom_backend)(torch.randn(3))
        (entry,) = PrecompileContext._dynamo_cache_entries.values()
        self.assertTrue(entry.requires_native_backend_compatibility)

    def test_loaded_eager_package_stays_exempt_on_resave(self):
        def fn(x):
            return x + 1

        package = CompilePackage(fn, requires_native_backend_compatibility=False)
        torch._dynamo.optimize(backend="eager", package=package)(fn)(torch.randn(3))
        with patch(
            "torch._dynamo.package._current_cpu_codegen_target",
            side_effect=AssertionError("toolchain probe ran for an eager backend"),
        ):
            entry = package.cache_entry()
            self.assertFalse(entry.requires_native_backend_compatibility)
            self.assertIsNone(entry.system_info.cpu_codegen_target)
            # Reload under an eager session (native_backend=False, as eval_frame
            # passes it): an eager artifact reloaded to be served again stays
            # exempt, so the resave never runs the toolchain probe.
            reloaded = CompilePackage(
                fn, entry, requires_native_backend_compatibility=False
            )
            resaved = reloaded.cache_entry()
        self.assertFalse(resaved.requires_native_backend_compatibility)
        self.assertIsNone(resaved.system_info.cpu_codegen_target)

    def test_loaded_eager_entry_does_not_disable_the_gate_on_an_inductor_run(self):
        # The flag is a floor, not a replacement: reloading an eager artifact
        # (requires=False) into a session whose backend emits native code must
        # not clear the gate, or a CPU kernel compiled after the load is saved
        # with no ISA fingerprint and reloads on any host (fail open).
        def fn(x):
            return x + 1

        eager = CompilePackage(fn, requires_native_backend_compatibility=False)
        torch._dynamo.optimize(backend="eager", package=eager)(fn)(torch.randn(3))
        entry = eager.cache_entry()
        self.assertFalse(entry.requires_native_backend_compatibility)

        # A native-backend session (native_backend=True) reloads that entry.
        reloaded = CompilePackage(fn, entry, requires_native_backend_compatibility=True)
        self.assertTrue(reloaded._requires_native_backend_compatibility)

    def test_codegen_drift_refuses_serialization_not_introspection(self):
        # A drifted package can never be serialized, but building a
        # cache_entry() for introspection (summary(), backend enumeration,
        # session teardown) must keep working -- a refusal there would erupt
        # out of __exit__ and mask the in-flight capture exception.
        def fn(x):
            return x + 1

        graph = torch.fx.Graph()
        graph.placeholder("x").meta["example_value"] = torch.ones(2)
        base = SystemInfo.current(cpu_codegen=False)
        target = ("x86_64", "avx2", 256, 256, None)
        first = dataclasses.replace(base, cpu_codegen_target=target)
        package = CompilePackage(fn)
        with (
            mock.patch.object(SystemInfo, "current", return_value=first),
            mock.patch(
                "torch._dynamo.package._current_cpu_codegen_target",
                return_value=("x86_64", "avx512", 512, 256, None),
            ),
            self.assertLogs("torch._dynamo.package", level="WARNING") as logs,
        ):
            package.update_device_type(graph)
            package.update_device_type(graph)
        self.assertIn("CPU codegen target changed during capture", logs.output[0])
        self.assertIsNotNone(package.cache_entry())
        with self.assertRaisesRegex(PackageError, "cannot be serialized"):
            package.refuse_unserializable()
        with self.assertRaisesRegex(PackageError, "cannot be serialized"):
            DynamoCache.record_package(package)

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

        # The bindings install() is responsible for: its per-install names plus
        # the builtins dict. The capture-time __compiled_fn global is not among
        # them (install binds a renamed twin), so its hook popping it is fine.
        installed = {
            g.name for g in package._installed_globals[sys.modules[fn.__module__]]
        }
        self.assertTrue(installed)
        self.assertTrue(installed - preexisting)

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
    def test_held_autocast_object_survives_the_package_round_trip(self):
        # An ID_MATCH guard on the autocast object was dropped (with a warning)
        # under caching_precompile, so a differently configured object could
        # reuse the graph. The value guards serialize, so the entry installs on
        # reload and still tells configurations apart -- kept in strict mode so
        # a regression fails loudly instead of bypassing the package silently.
        def fn(x, ac):
            with ac:
                return torch.mm(x, x)

        x = torch.randn(4, 4)
        warm = torch.autocast("cpu", dtype=torch.bfloat16)
        self.assertEqual(torch.compile(fn)(x, warm).dtype, torch.bfloat16)  # noqa: UNSPECIFIED_BACKEND
        (entry,) = PrecompileContext.save_to_dynamo_cache()["dynamo"]
        self.assertTrue(entry["backend_ids"])
        torch._dynamo.reset()
        PrecompileContext.clear()
        compiled = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        self.assertGreater(len(_debug_get_precompile_entries(fn.__code__)), 0)
        with torch.compiler.set_stance("fail_on_recompile"):
            fresh = torch.autocast("cpu", dtype=torch.bfloat16)
            self.assertEqual(compiled(x, fresh).dtype, torch.bfloat16)
            other = torch.autocast("cpu", dtype=torch.bfloat16, enabled=False)
            with self.assertRaisesRegex(
                RuntimeError,
                "Detected recompile when torch.compile stance is 'fail_on_recompile'",
            ):
                compiled(x, other)
            # _cache_enabled is the fourth guarded field; diverging it alone must
            # also miss, or dropping it from the tuple would go unnoticed.
            other_cache = torch.autocast(
                "cpu", dtype=torch.bfloat16, cache_enabled=False
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "Detected recompile when torch.compile stance is 'fail_on_recompile'",
            ):
                compiled(x, other_cache)

    @torch._dynamo.config.patch(caching_precompile=True, strict_precompile=False)
    def test_unserializable_guard_bypasses_the_package(self):
        # A guarded value that cannot be pickled is a package bypass, not a
        # compile failure: the frame still compiles and runs, and its entry is
        # saved bypassed with no backend, so nothing is installed on reload.
        # convert_frame used to assert on the missing guards_state because it
        # checked the package it was handed, not the one the bypass had
        # cleared on the output graph.
        def fn(x, cfg=UnpicklableConfig()):
            if cfg.flag == 2.0:
                x = x + 1
            return x.sin()

        x = torch.randn(3)
        expected = fn(x)
        with self.assertLogs("torch._dynamo", level="WARNING") as logs:
            self.assertEqual(torch.compile(fn)(x), expected)  # noqa: UNSPECIFIED_BACKEND
        self.assertTrue(any("package bypass" in line for line in logs.output))
        (entry,) = PrecompileContext.save_to_dynamo_cache()["dynamo"]
        self.assertEqual(entry["backend_ids"], [])
        torch._dynamo.reset()
        PrecompileContext.clear()
        # Wrapping is what reloads the cache; the bypassed entry installs nothing.
        compiled = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(len(_debug_get_precompile_entries(fn.__code__)), 0)
        with self.assertLogs("torch._dynamo", level="WARNING") as logs:
            self.assertEqual(compiled(x), expected)
        self.assertTrue(any("package bypass" in line for line in logs.output))

    @torch._dynamo.config.patch(caching_precompile=True, strict_precompile=False)
    def test_bypassed_recompile_drops_the_frames_earlier_variants(self):
        # A bypass marks the frame's whole entry, so a variant that serialized
        # fine earlier goes with it and install() skips the frame.
        def fn(x, cfg=None):
            if cfg is not None and cfg.flag == 2.0:
                x = x + 1
            return x.sin()

        x = torch.randn(3)
        expected = fn(x)
        compiled = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(compiled(x), expected)
        with self.assertLogs("torch._dynamo", level="WARNING") as logs:
            compiled(x, UnpicklableConfig())
        self.assertTrue(any("package bypass" in line for line in logs.output))
        (entry,) = PrecompileContext.save_to_dynamo_cache()["dynamo"]
        self.assertEqual(entry["backend_ids"], [])
        torch._dynamo.reset()
        PrecompileContext.clear()
        compiled = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(len(_debug_get_precompile_entries(fn.__code__)), 0)
        with torch.compiler.set_stance("fail_on_recompile"):
            with self.assertRaisesRegex(RuntimeError, "Detected recompile"):
                compiled(x)
        self.assertEqual(compiled(x), expected)

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_saving_does_not_bypass_the_live_entry(self):
        # from_cache_entry marks a code whose backend it cannot find as bypassed
        # on the entry it is handed. Saving must work on a copy: the live entry
        # keeps serving this process, and a save that came up short on a
        # backend must not flip it to bypassed.
        def fn(x):
            return x.sin()

        x = torch.randn(3)
        self.assertEqual(torch.compile(fn)(x), fn(x))  # noqa: UNSPECIFIED_BACKEND
        ((key, live),) = PrecompileContext._dynamo_cache_entries.items()
        self.assertTrue(live.codes[0].backend_ids)
        PrecompileContext._backend_artifacts_by_key.clear()
        saved, _ = PrecompileContext.create_cache_entries()
        self.assertTrue(saved[key].dynamo.codes[0].bypassed)
        self.assertFalse(live.codes[0].bypassed)

    def test_abandoned_package_uninstalls_on_gc(self):
        # Without the finalizer, each load+install of one artifact would leave
        # behind its per-owner entries and per-install uuid-named resume globals.
        ctx = DiskDynamoStore()

        def fn(x):
            y = x.sin()
            torch._dynamo.graph_break()
            return y + x.cos()

        def guard_filter_fn(guards):
            # A nested fn with a graph break produces guards that cannot be
            # serialized.
            unserializable = ("MODULE_MATCH", "CLOSURE_MATCH", "FUNCTION_MATCH")
            return [guard.guard_type not in unserializable for guard in guards]

        package = CompilePackage(fn)
        compiled_fn = torch._dynamo.optimize(
            backend="eager", package=package, guard_filter_fn=guard_filter_fn
        )(fn)
        compiled_fn(torch.randn(3, 2))
        for backend_id, bknd in package.cached_backends.items():
            ctx.record_eager_backend(backend_id, bknd)
        ctx.save_package(package, self.path())
        torch._dynamo.reset()
        del package, compiled_fn
        gc.collect()

        module_keys = set(sys.modules[fn.__module__].__dict__)
        counts = []
        for _ in range(4):
            pkg, backends = ctx.load_package(fn, self.path())
            pkg.install(backends)
            counts.append(len(_debug_get_precompile_entries(fn.__code__)))
            del pkg, backends
            gc.collect()
        # Each reload sees only its own entries (no growth across the four
        # generations), the last dead owner's entries are gone, and nothing
        # the dead packages installed is left in the module globals -- the
        # shared builtins dict is deliberately left in place.
        self.assertGreater(counts[0], 0)
        self.assertEqual(counts, [counts[0]] * 4)
        self.assertEqual(len(_debug_get_precompile_entries(fn.__code__)), 0)
        leaked = set(sys.modules[fn.__module__].__dict__) - module_keys
        self.assertTrue(all(k.startswith("__builtins_dict") for k in leaked), leaked)

        # A LIVE package is untouched by garbage collection.
        pkg, backends = ctx.load_package(fn, self.path())
        pkg.install(backends)
        gc.collect()
        self.assertEqual(len(_debug_get_precompile_entries(fn.__code__)), counts[0])
        pkg.uninstall()

    def test_failed_install_is_torn_down_when_the_package_dies(self):
        # install() registers its teardown finalizer BEFORE binding any global,
        # so a mid-install failure leaves nothing behind: whatever it bound is
        # gone once the package dies, even though install() raised and handed
        # the caller no handle to undo it. Force the failure by handing
        # install() a backends dict missing a required backend.
        ctx = DiskDynamoStore()

        def fn(x):
            y = x.sin()
            torch._dynamo.graph_break()
            return y + x.cos()

        def guard_filter_fn(guards):
            unserializable = ("MODULE_MATCH", "CLOSURE_MATCH", "FUNCTION_MATCH")
            return [guard.guard_type not in unserializable for guard in guards]

        self._save_eager_package(fn, ctx, (torch.randn(3, 2),), guard_filter_fn)
        module_dict = sys.modules[fn.__module__].__dict__
        before = set(module_dict)

        pkg, backends = ctx.load_package(fn, self.path())
        # Drop the resume entry's backend: install() binds that entry's renamed
        # resume global and installs the earlier entry's precompile entry, then
        # raises on the missing backend -- a genuinely partial install.
        resume_entry = next(e for e in pkg._codes.values() if e.install_to_global)
        del backends[resume_entry.backend_ids[0]]
        with self.assertRaisesRegex(RuntimeError, "is not found in the given backends"):
            pkg.install(backends)
        # Reaching that error means install() bound the resume global and an
        # entry before it raised: a genuinely partial install to tear down.

        del pkg, backends
        gc.collect()
        # Nothing partial survives -- only the shared builtins dict, left in
        # place by design, may remain -- and no entries are left.
        leaked = set(module_dict) - before
        self.assertTrue(all(k.startswith("__builtins_dict") for k in leaked), leaked)
        self.assertEqual(len(_debug_get_precompile_entries(fn.__code__)), 0)

    def test_rename_globals_rewrites_nested_code(self):
        def outer(x):
            def inner(y):
                return resume_at_16_3(y)  # noqa: F821

            return inner(x) + resume_at_16_3(x)  # noqa: F821

        old, new = "resume_at_16_3", "resume_at_16_3_0123456789abcdef_tok"
        code = _rename_globals(outer.__code__, {old: new})
        (inner_code,) = [c for c in code.co_consts if isinstance(c, types.CodeType)]
        self.assertIn(new, code.co_names)
        self.assertNotIn(old, code.co_names)
        self.assertIn(new, inner_code.co_names)
        self.assertNotIn(old, inner_code.co_names)
        # Indices into co_names are preserved, so the bytecode is untouched and
        # the renamed code follows the new binding.
        self.assertEqual(code.co_code, outer.__code__.co_code)
        renamed = types.FunctionType(code, {new: lambda y: y + 1})
        self.assertEqual(renamed(1), 4)
        # The original is not mutated, and renames that apply nowhere hand the
        # same object back.
        self.assertIn(old, outer.__code__.co_names)
        self.assertIs(_rename_globals(outer.__code__, {"absent": "x"}), outer.__code__)

    def _save_eager_package(self, fn, ctx, args, guard_filter_fn=None):
        package = CompilePackage(fn)
        compiled_fn = torch._dynamo.optimize(
            backend="eager", package=package, guard_filter_fn=guard_filter_fn
        )(fn)
        compiled_fn(*args)
        for backend_id, bknd in package.cached_backends.items():
            ctx.record_eager_backend(backend_id, bknd)
        ctx.save_package(package, self.path())
        torch._dynamo.reset()

    def test_two_packages_from_one_artifact_coexist(self):
        # Two loads of one artifact serve the same frame at once: their
        # precompile entries are told apart by owner and their resume functions
        # by per-install names, so one can be served while the other is live and
        # each unloads without disturbing the other.
        ctx = DiskDynamoStore()

        def fn(x):
            y = x.sin()
            torch._dynamo.graph_break()
            return y + x.cos()

        def guard_filter_fn(guards):
            unserializable = ("MODULE_MATCH", "CLOSURE_MATCH", "FUNCTION_MATCH")
            return [guard.guard_type not in unserializable for guard in guards]

        x = torch.randn(3, 2)
        expected = fn(x)
        self._save_eager_package(fn, ctx, (x,), guard_filter_fn)
        module_dict = sys.modules[fn.__module__].__dict__
        before = set(module_dict)

        pkg_a, backends_a = ctx.load_package(fn, self.path())
        pkg_a.install(backends_a)
        count = len(_debug_get_precompile_entries(fn.__code__))
        self.assertGreater(count, 0)
        resume_a = {k for k in set(module_dict) - before if k.startswith("__resume_at")}
        pkg_b, backends_b = ctx.load_package(fn, self.path())
        pkg_b.install(backends_b)
        resume_b = {k for k in set(module_dict) - before if k.startswith("__resume_at")}
        resume_b -= resume_a
        self.assertTrue(resume_a)
        self.assertTrue(resume_b)
        self.assertEqual(len(_debug_get_precompile_entries(fn.__code__)), 2 * count)

        # Either package serves the frame while the other is live: both loads'
        # entries and resume functions coexist on the one code object.
        compiled_fn = torch._dynamo.optimize(package=pkg_a)(fn)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled_fn(x), expected)
        compiled_b = torch._dynamo.optimize(package=pkg_b)(fn)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled_b(x), expected)
        # a's unload takes only a's entries and a's resume functions. Names
        # both loads share go with the FIRST unload: an import alias, whose
        # loss is a failed guard and a silent cache miss, and the capture-time
        # __compiled_fn, whose loss is a NameError in live user code -- so b is
        # NOT served after this. Teardown by owner count for shared names is
        # #195915, a separate change.
        pkg_a.uninstall()
        self.assertEqual(len(_debug_get_precompile_entries(fn.__code__)), count)
        self.assertFalse(resume_a & set(module_dict))
        self.assertTrue(resume_b <= set(module_dict))
        pkg_b.uninstall()
        self.assertEqual(len(_debug_get_precompile_entries(fn.__code__)), 0)
        self.assertFalse(resume_b & set(module_dict))
        with torch.compiler.set_stance("fail_on_recompile"):
            with self.assertRaisesRegex(RuntimeError, "Detected recompile"):
                compiled_fn(x)

    def test_uninstall_leaves_a_users_rebinding_alone(self):
        # uninstall() pops a global only while it still holds the value this
        # package installed, as the GC finalizer already did.
        ctx = DiskDynamoStore()

        def fn(x):
            return x + 1

        self._save_eager_package(fn, ctx, (torch.randn(3, 2),))
        module = sys.modules[fn.__module__]
        module_dict = module.__dict__
        pkg, backends = ctx.load_package(fn, self.path())
        pkg.install(backends)
        # From the package's own bookkeeping, not a module-dict diff: on a
        # free-threaded build the capture-time CleanupHook (keyed on a code
        # object, deferred-refcounted there) may not have popped the previous
        # compile's name yet, so a diff can be empty.
        (name,) = [
            g.name
            for g in pkg._installed_globals[module]
            if g.name.startswith("__compiled_fn")
        ]
        sentinel = object()
        module_dict[name] = sentinel
        self.addCleanup(module_dict.pop, name, None)
        pkg.uninstall()
        self.assertIs(module_dict[name], sentinel)
        self.assertEqual(len(_debug_get_precompile_entries(fn.__code__)), 0)

    def test_system_info_is_read_once_per_package(self):
        # SystemInfo.current probes the accelerator and the C++ toolchain, and
        # update_device_type runs on every compile under caching_precompile.
        def fn(x):
            return x + 1

        graph = torch.fx.Graph()
        graph.placeholder("x").meta["example_value"] = torch.ones(2)
        package = CompilePackage(fn)
        with mock.patch.object(
            SystemInfo, "current", wraps=SystemInfo.current
        ) as current:
            for _ in range(3):
                package.update_device_type(graph)
        self.assertEqual(current.call_count, 1)
        self.assertIsNone(package._cpu_codegen_target_drift)

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_unrecordable_package_warns_and_still_compiles(self):
        def fn(x):
            return x.sin()

        x = torch.randn(3)
        with (
            mock.patch.object(
                DynamoCache, "record_package", side_effect=PackageError("drifted")
            ),
            self.assertLogs("torch._dynamo.convert_frame", level="WARNING") as logs,
        ):
            self.assertEqual(torch.compile(fn, backend="eager")(x), fn(x))
        self.assertTrue(
            any("Not recording compile package: drifted" in m for m in logs.output)
        )

    def test_defining_module_name_prefers_the_file_over_a_reexporting_shim(self):
        # The _collections_abc idiom: the implementation file sets __name__ to
        # the public name, so inspect.getmodule(code) lands on the shim whose
        # file does not contain the code.
        with tempfile.TemporaryDirectory() as tmp:
            impl_path = os.path.join(tmp, "_package_shim_impl.py")
            with open(impl_path, "w") as f:
                f.write('__name__ = "package_shim"\n\n\ndef f(x):\n    return x + 1\n')
            shim_path = os.path.join(tmp, "package_shim.py")
            with open(shim_path, "w") as f:
                f.write("from _package_shim_impl import *\n")
            spec = importlib.util.spec_from_file_location(
                "_package_shim_impl", impl_path
            )
            impl = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(impl)
            shim = types.ModuleType("package_shim")
            shim.__file__ = shim_path
            shim.f = impl.f
            sys.modules["_package_shim_impl"] = impl
            sys.modules["package_shim"] = shim
            try:
                code = impl.f.__code__
                self.assertIs(inspect.getmodule(code), shim)
                self.assertEqual(_defining_module_name(code), "_package_shim_impl")
                info = SourceInfo(inlined_sources=set())
                info.add_code(code)
                self.assertEqual(
                    {s.module for s in info.inlined_sources}, {"_package_shim_impl"}
                )
            finally:
                sys.modules.pop("_package_shim_impl", None)
                sys.modules.pop("package_shim", None)
                _MODULE_KEY_BY_FILE.pop(impl_path, None)

    def test_scan_sys_modules_revalidates_a_stale_hit(self):
        # Renaming a module's sys.modules key keeps len(sys.modules) equal, so
        # the ABA check alone would keep returning the dead key.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "_package_stale_hit.py")
            with open(path, "w") as f:
                f.write("def f(x):\n    return x + 1\n")
            spec = importlib.util.spec_from_file_location("_package_stale_hit", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules["_package_stale_hit"] = module
            try:
                self.assertEqual(_scan_sys_modules_for_file(path), "_package_stale_hit")
                sys.modules["_package_stale_hit_renamed"] = sys.modules.pop(
                    "_package_stale_hit"
                )
                self.assertEqual(
                    _scan_sys_modules_for_file(path), "_package_stale_hit_renamed"
                )
                del sys.modules["_package_stale_hit_renamed"]
                self.assertIsNone(_scan_sys_modules_for_file(path))
            finally:
                sys.modules.pop("_package_stale_hit", None)
                sys.modules.pop("_package_stale_hit_renamed", None)
                _MODULE_KEY_BY_FILE.pop(path, None)

    def test_reserve_unique_id_through_skips_past_a_loaded_artifacts_counter(self):
        from torch._dynamo.bytecode_transformation import (
            _reserve_unique_id_through,
            unique_id,
        )

        current = int(unique_id("probe").rsplit("_", 1)[1])
        _reserve_unique_id_through(current + 50)
        self.assertGreater(int(unique_id("probe").rsplit("_", 1)[1]), current + 50)
        # Reserving below the counter must not move it backwards.
        after = int(unique_id("probe").rsplit("_", 1)[1])
        _reserve_unique_id_through(0)
        self.assertGreater(int(unique_id("probe").rsplit("_", 1)[1]), after)

    def test_abandoned_package_restores_skipped_frames_on_gc(self):
        ctx = DiskDynamoStore()

        def fn(x):
            return x.sin()

        package = CompilePackage(fn)
        torch._dynamo.optimize(backend="eager", package=package)(fn)(torch.randn(3))
        # A frame with no guarded code is what install() skip_code()s.
        entry = package.cache_entry().codes[0]
        entry.guarded_codes.clear()
        entry.backend_ids.clear()
        package.cached_backends.clear()
        ctx.save_package(package, self.path())
        torch._dynamo.reset()
        del package
        gc.collect()

        code = fn.__code__
        pkg, backends = ctx.load_package(fn, self.path())
        pkg.install(backends)
        self.assertEqual(get_code_exec_strategy(code).cur_action, FrameAction.SKIP)
        del pkg, backends
        gc.collect()
        self.assertEqual(get_code_exec_strategy(code).cur_action, FrameAction.DEFAULT)

    def test_explicit_capture_is_not_inferred_from_the_serialization_filter(self):
        # The serialization filter and the capture mode are independent: a
        # package can carry a filter without being an explicit capture, and be
        # an explicit capture without one.
        def fn(x):
            return x + 1

        def keep_all(entries):
            return [True] * len(entries)

        filtered = CompilePackage(fn, serialization_guard_filter_fn=keep_all)
        self.assertFalse(filtered.explicit_capture)
        self.assertIs(filtered.serialization_guard_filter_fn, keep_all)
        explicit = CompilePackage(fn, explicit_capture=True)
        self.assertTrue(explicit.explicit_capture)
        self.assertIsNone(explicit.serialization_guard_filter_fn)

    def _saved_guard_names(self, package):
        names = set()
        for guarded in package.cache_entry().codes[0].guarded_codes:
            state = load_guards_state(guarded.guards_state)
            names |= {g.create_fn_name() for g in state.output_graph.guards}
        return names

    def test_serialization_filter_applies_to_the_saved_guards_only(self):
        # The live guards keep checking what they check, so an explicit capture
        # still recompiles on a dtype change; only the serialized copy is
        # filtered. The same filter on a non-explicit package does the same, and
        # an explicit package without a filter saves its guards unfiltered.
        def fn(x):
            return x + 1

        def drop_tensor_match(entries):
            return [e.guard_type != "TENSOR_MATCH" for e in entries]

        for explicit_capture in (True, False):
            torch._dynamo.reset()
            pkg = CompilePackage(
                fn,
                explicit_capture=explicit_capture,
                serialization_guard_filter_fn=drop_tensor_match,
            )
            counter = torch._dynamo.testing.CompileCounter()
            compiled = torch._dynamo.optimize(backend=counter, package=pkg)(fn)
            compiled(torch.randn(3))
            compiled(torch.randint(0, 5, (3,)))
            self.assertEqual(counter.frame_count, 2)
            self.assertNotIn("TENSOR_MATCH", self._saved_guard_names(pkg))

        torch._dynamo.reset()
        bare = CompilePackage(fn, explicit_capture=True)
        torch._dynamo.optimize(backend="eager", package=bare)(fn)(torch.randn(3))
        self.assertIn("TENSOR_MATCH", self._saved_guard_names(bare))

    @torch._dynamo.config.patch(recompile_limit=1)
    def test_truncated_frames_names_the_frame_that_hit_the_recompile_limit(self):
        def fn(x):
            return x + 1

        pkg = CompilePackage(fn, explicit_capture=True)
        compiled = torch._dynamo.optimize(backend="eager", package=pkg)(fn)
        compiled(torch.randn(3))
        self.assertEqual(pkg.truncated_frames, frozenset())
        compiled(torch.randint(0, 5, (3,)))
        code = fn.__code__
        location = f"fn ({code.co_filename}:{code.co_firstlineno})"
        self.assertEqual(pkg.truncated_frames, frozenset({location}))
        # The variant captured before the limit stays in the package.
        self.assertEqual(len(pkg.cache_entry().codes[0].guarded_codes), 1)

    def test_uncovered_frames_follows_the_entries(self):
        # A frame that entered Dynamo without producing guarded code is a gap
        # only while that stays true: a later variant that compiles covers it,
        # whichever order the variants ran in.
        def fn(x):
            return x

        pkg = CompilePackage(fn)
        torch._dynamo.optimize(backend="eager", package=pkg)(fn)(torch.randn(3))
        self.assertEqual(pkg.uncovered_frames, frozenset({"fn"}))
        with pkg.code_context(fn.__code__):
            pkg.add_guarded_code(b"", fn.__code__)
        self.assertEqual(pkg.uncovered_frames, frozenset())

    def test_serving_package_records_nothing_and_still_recompiles(self):
        def fn(x):
            return x + 1

        pkg = CompilePackage(fn, serving=True)
        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch._dynamo.optimize(backend=counter, package=pkg)(fn)
        compiled(torch.randn(3))
        compiled(torch.randint(0, 5, (3,)))
        self.assertEqual(counter.frame_count, 2)
        self.assertEqual(pkg.cache_entry().codes[0].guarded_codes, [])

    @parametrize("device", ("cpu", "cuda", "xpu"))
    @parametrize("isolate_recompiles", (False, True))
    @torch._dynamo.config.patch(caching_precompile=True)
    def test_automatic_dynamo_serves_an_isolate_recompiles_context(
        self, device, isolate_recompiles
    ):
        # The transparent cache installs the package it loads, and precompile
        # entries match their own region only, so installing into the default
        # bucket while the context looks up in its own region loaded the
        # artifact and then served nothing -- every call recompiled, and under
        # fail_on_recompile it raised. Nothing combined these two before.
        def fn(x):
            return x.sin() + x.cos()

        if device == "cuda" and not HAS_CUDA_AND_TRITON:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU_AND_TRITON:
            raise unittest.SkipTest("Requires XPU/Triton")

        arg = torch.randn(3, 2, device=device)
        expected = fn(arg)
        torch.compile(  # noqa: UNSPECIFIED_BACKEND
            fn, isolate_recompiles=isolate_recompiles
        )(arg)
        DynamoCache.clear()
        self._save_and_reload(expected_backends=1, expected_dynamo=1)

        counters.clear()
        warm = torch.compile(  # noqa: UNSPECIFIED_BACKEND
            fn, isolate_recompiles=isolate_recompiles
        )
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(warm(arg), expected)
        # The warm call loaded the package from the transparent cache (a hit,
        # not a fresh compile); fail_on_recompile above proves it then served
        # the frame rather than loading and serving nothing.
        self.assertEqual(counters["dynamo_cache"]["dynamo_cache_hit"], 1)

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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
