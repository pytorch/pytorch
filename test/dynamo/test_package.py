# Owner(s): ["module: dynamo"]

import os
import unittest

import torch
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.onnx.operators
import torch.utils.cpp_extension
from torch._dynamo.package import CompilePackage, DiskDynamoStore, DynamoCache
from torch._dynamo.precompile_context import PrecompileContext
from torch._functorch import config as functorch_config
from torch._inductor.runtime.runtime_utils import cache_dir
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_CUDA, HAS_XPU


@functorch_config.patch("bundled_autograd_cache", True)
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

    @parametrize("backend", ("eager", "inductor"))
    @parametrize("device", ("cpu", "cuda", "xpu"))
    def test_basic_fn(self, backend, device):
        if device == "cuda" and not HAS_CUDA:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU:
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
    def test_graph_break_bomb(self, backend, device):
        if device == "cuda" and not HAS_CUDA:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU:
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
        if device == "cuda" and not HAS_CUDA:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU:
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

    @parametrize("device", ("cpu", "cuda", "xpu"))
    def test_dynamo_cache_manual_load(self, device):
        if device == "cuda" and not HAS_CUDA:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU:
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

        result = PrecompileContext.serialize()
        assert result is not None
        bytes_, cache_info = result
        self.assertEqual(len(cache_info.precompile_aot_autograd_artifacts), 2)
        self.assertEqual(len(cache_info.precompile_dynamo_artifacts), 2)

        DynamoCache.clear()

        deserialized = PrecompileContext.deserialize(bytes_)
        assert deserialized is not None
        PrecompileContext.populate_caches(deserialized)
        torch._dynamo.reset()

        # These should exist because of populate_caches
        package1 = DynamoCache.load_and_install_package(fn)
        package2 = DynamoCache.load_and_install_package(fn2)

        with torch.compiler.set_stance("fail_on_recompile"):
            result1 = compiled_fn1(arg1)
            result2 = compiled_fn2(arg2)
            self.assertEqual(expected, [result1, result2])

    @parametrize("device", ("cpu", "cuda", "xpu"))
    @torch._dynamo.config.patch(caching_precompile=True)
    def test_automatic_dynamo_cache(self, device):
        if device == "cuda" and not HAS_CUDA:
            raise unittest.SkipTest("Requires CUDA/Triton")
        if device == "xpu" and not HAS_XPU:
            raise unittest.SkipTest("Requires XPU/Triton")

        def fn(x):
            return x.sin() + x.cos()

        def fn2(x):
            return x.cos() + x

        arg1 = torch.randn(3, 2, device=device)
        arg2 = torch.randn(5, 2, device=device)
        expected = [fn(arg1), fn2(arg2)]
        compiled_fn1 = torch._dynamo.optimize(backend="inductor")(fn)
        compiled_fn2 = torch._dynamo.optimize(backend="inductor")(fn2)
        result = [compiled_fn1(arg1), compiled_fn2(arg2)]
        self.assertEqual(expected, result)

        torch._dynamo.reset()
        # Compile again
        compiled_fn1 = torch._dynamo.optimize(backend="inductor")(fn)
        compiled_fn2 = torch._dynamo.optimize(backend="inductor")(fn2)
        with torch.compiler.set_stance("fail_on_recompile"):
            result1 = compiled_fn1(arg1)
            result2 = compiled_fn2(arg2)
            self.assertEqual(expected, [result1, result2])


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
