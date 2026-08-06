# Owner(s): ["module: dynamo"]

import contextlib
import dataclasses
import importlib
import os
import sys
import tempfile
import unittest

import torch
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.onnx.operators
import torch.utils.cpp_extension
from torch._dynamo.exc import PackageError
from torch._dynamo.package import (
    CompilePackage,
    DiskDynamoStore,
    DynamoCache,
    SystemInfo,
)
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.precompile_package import (
    precompile_capture,
    precompile_load,
    serving,
)
from torch._dynamo.testing import reduce_to_scalar_loss
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


def _precompile_user_act(t):
    return -t


class PrecompileSelfAct(torch.nn.Module):
    """self.act = <callable> -- how configurable activations are usually written."""

    def __init__(self, act):
        super().__init__()
        self.act = act

    def forward(self, x):
        y = self.act(x)
        torch._dynamo.graph_break()
        return (y + 1).sum()


class PrecompileValuePinned(torch.nn.Module):
    def forward(self, x):
        scale = x.abs().max().item()
        y = x * 2 if scale > 0.5 else x * 3
        return y.sum()


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

    def path(self):
        path = os.path.join(cache_dir(), f"precompile_{self.id()}")
        os.makedirs(path, exist_ok=True)
        return path

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
        session.save(self.path())

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
        session.save(self.path())

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

    def test_raised_recompile_limit_is_complete(self):
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
        self.assertTrue(summary.complete)
        self.assertEqual(summary.guarded_codes, n * len(inputs))
        session.save(self.path())

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
        global PRECOMPILE_ACTIVATION
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
            session.save(self.path())
        # The risk is acknowledgeable, not a hard block.
        session.save(self.path(), require_no_risky_drops=False)

    def test_risky_drop_detected_through_a_module_attribute(self):
        # The guard on self.act is dropped as an unserializable identity guard,
        # and its source is reported with local scope stripped ("self.act"), so
        # it cannot be recognised by matching the source against a global
        # pattern. Classify by what the guarded value is instead.
        session = precompile_capture(
            PrecompileSelfAct(_precompile_user_act), backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        risky = [name for _, name in session.summary().risky_dropped_guards]
        self.assertIn("self.act", risky)
        with self.assertRaisesRegex(PackageError, "self.act"):
            session.save(self.path())

    def test_torch_owned_drops_are_not_risky(self):
        # Identity guards on torch internals are dropped for every model. If
        # those counted as risky the check would refuse ordinary code and get
        # switched off.
        session = precompile_capture(
            PrecompileSelfAct(torch.relu), backend="eager", dynamic=False
        )
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        summary = session.summary()
        self.assertTrue(summary.dropped_guards)
        self.assertEqual(summary.risky_dropped_guards, ())
        session.save(self.path())

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
            path = os.path.join(self.path(), f"pkg_{act.__name__}")
            os.makedirs(path, exist_ok=True)
            session.save(path)
            paths.append(path)

        torch._dynamo.reset()
        first = precompile_load(
            PrecompileSelfAct(torch.relu), paths[0], backend="eager", dynamic=False
        )
        second = precompile_load(
            PrecompileSelfAct(torch.sigmoid), paths[1], backend="eager", dynamic=False
        )
        first.unload()
        second.unload()

    def test_stale_artifact_rejected_when_source_drifts(self):
        # The deployment shape is capture on one machine, serve on another. The
        # dangerous version of that is an artifact outliving a code change, so
        # the source checksum has to fire even though the module is found by
        # name and its path differs between the two machines.
        src = "import torch\n\n\ndef staged(x):\n    y = x * 2\n    torch._dynamo.graph_break()\n    return (y + 1).sum()\n"
        pkg_dir = os.path.join(self.path(), "srcdrift")
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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
