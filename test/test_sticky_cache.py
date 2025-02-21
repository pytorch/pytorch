# Owner(s): ["oncall: export"]

import glob
import inspect
import os
import shutil
import types

import torch
from torch._inductor.output_code import CompiledAOTI
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda,
    run_tests,
)


class TestStickyCache(TestCase):
    def path(self):
        return f"sticky_cache_{self.id()}"

    def _test_save(self, compiled_fn, args, expected, path, prefix=None) -> int:
        # From a clean state, save_sticky_cache() can produce some archive
        # files and the files loaded back can reproduce the expected result.
        # Returns the index of compilation hit in the cache.
        prefix = prefix or cache_dir()

        load_dir = os.path.join(prefix, self.path())
        load_paths = glob.glob(os.path.join(load_dir, "*.pt2"))
        self.assertEqual(len(load_paths), 0)

        compiled_fn.save_sticky_cache(prefix=prefix)
        load_paths = glob.glob(os.path.join(load_dir, "*.pt2"))
        self.assertGreaterEqual(len(load_paths), 1)

        precompiles = []
        for i in range(len(load_paths)):
            precompile = torch._dynamo.sticky_cache._load_precompile(load_dir, i)
            self.assertIsInstance(precompile.dynamo_code, types.CodeType)
            self.assertIsInstance(precompile.aoti, CompiledAOTI)
            precompiles.append(precompile)

        for i, precompile in enumerate(precompiles):
            if precompile.match_inputs(args, {}):
                if isinstance(compiled_fn, torch.nn.Module):
                    self.assertEqual(expected, precompile(compiled_fn._orig_mod, *args))
                elif inspect.ismethod(compiled_fn._torchdynamo_orig_callable):
                    self.assertEqual(
                        expected,
                        precompile(
                            compiled_fn._torchdynamo_orig_callable.__self__, *args
                        ),
                    )
                else:
                    self.assertEqual(expected, precompile(*args))
                return i
        else:
            self.fail()

    def _test_load(self, compiled_fn, args, expected, path, prefix=None) -> None:
        # From a clean state with recompile disabled, we can load back the
        # sticky cache and reproduce expected results.
        # Will clean up the loaded package on disk.
        prefix = prefix or cache_dir()

        state = compiled_fn.reset_sticky_cache()
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with self.assertRaisesRegex(
                RuntimeError, "Detected recompile when torch.compile"
            ):
                compiled_fn(*args)

            compiled_fn.load_sticky_cache(prefix=prefix)
            results = compiled_fn(*args)
            self.assertEqual(expected, results)

            shutil.rmtree(os.path.join(prefix, path))
            with self.assertRaisesRegex(
                RuntimeError, "Sticky cache path .* doesn't exist"
            ):
                compiled_fn.load_sticky_cache(prefix=prefix)
        compiled_fn.reset_sticky_cache(state)

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_function_forward_staic(self, device):
        torch.set_default_device(device)

        def f(a, b):
            for i in range(100):
                a = a + b * i
            return a

        args = torch.randn(3), torch.randn(3)
        args1 = torch.randn((4, 2)), torch.randn((4, 2))
        expected = f(*args)
        expected1 = f(*args1)
        f = torch.compile(f, fullgraph=True, dynamic=False, sticky_cache=self.path())
        f(*args)
        compile_index = self._test_save(f, args, expected, self.path())
        self.assertEqual(compile_index, 0)
        self._test_load(f, args, expected, self.path())

        f(*args1)
        compile_index = self._test_save(f, args1, expected1, self.path())
        self.assertEqual(compile_index, 1)
        self._test_load(f, args1, expected1, self.path())

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_module_forward_staic(self, device):
        torch.set_default_device(device)

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("p", torch.tensor(10))

            def forward(self, a, b):
                for i in range(100):
                    a = a + b * i + self.p
                return a

        f = Module()
        args = torch.randn(3), torch.randn(3)
        args1 = torch.randn((4, 2)), torch.randn((4, 2))
        expected = f(*args)
        expected1 = f(*args1)
        f = torch.compile(f, fullgraph=True, dynamic=False, sticky_cache=self.path())
        f(*args)
        compile_index = self._test_save(f, args, expected, self.path())
        self.assertEqual(compile_index, 0)
        self._test_load(f, args, expected, self.path())

        f(*args1)
        compile_index = self._test_save(f, args1, expected1, self.path())
        self.assertEqual(compile_index, 1)
        self._test_load(f, args1, expected1, self.path())

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_method_forward_staic(self, device):
        torch.set_default_device(device)

        class Module:
            def __init__(self):
                self.p = torch.tensor(10)

            def g(self, a, b):
                for i in range(100):
                    a = a + b * i + self.p
                return a

        m = Module()
        args = torch.randn(3), torch.randn(3)
        args1 = torch.randn((4, 2)), torch.randn((4, 2))
        expected = m.g(*args)
        expected1 = m.g(*args1)
        f = torch.compile(m.g, fullgraph=True, dynamic=False, sticky_cache=self.path())
        f(*args)
        compile_index = self._test_save(f, args, expected, self.path())
        self.assertEqual(compile_index, 0)
        self._test_load(f, args, expected, self.path())

        f(*args1)
        compile_index = self._test_save(f, args1, expected1, self.path())
        self.assertEqual(compile_index, 1)
        self._test_load(f, args1, expected1, self.path())

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_function_forward_non_fullgraph(self, device):
        torch.set_default_device(device)

        def f(a, b):
            for i in range(100):
                a = a + b * i
            return a

        with self.assertRaisesRegex(
            RuntimeError, "Sticky cache is only supported .*fullgraph=True"
        ):
            torch.compile(f, dynamic=False, sticky_cache=self.path())

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_function_forward_dynamic(self, device):
        torch.set_default_device(device)

        def f(a, b):
            for i in range(100):
                a = a + b * i
            return a

        with self.assertRaisesRegex(
            NotImplementedError,
            "dynamic shape",
        ):
            torch.compile(f, fullgraph=True, sticky_cache=self.path())

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_function_forward_backward(self, device):
        torch.set_default_device(device)

        def f(a, b):
            for i in range(100):
                a = a + b * i
            return a.sum()

        args = torch.randn(3, requires_grad=True), torch.randn(3, requires_grad=True)
        f = torch.compile(f, fullgraph=True, dynamic=False, sticky_cache=self.path())
        with self.assertRaisesRegex(
            NotImplementedError,
            "backward",
        ):
            f(*args).backward()

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_function_input_mutation(self, device):
        torch.set_default_device(device)

        def f(a, b):
            for i in range(100):
                a.add_(b * i)
            return a

        args = torch.randn(3), torch.randn(3)
        f = torch.compile(f, fullgraph=True, dynamic=False, sticky_cache=self.path())
        with self.assertRaisesRegex(
            NotImplementedError,
            "mutations on input tensors",
        ):
            f(*args)

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_function_container_inputs(self, device):
        torch.set_default_device(device)

        def f(a, b):
            for i in range(100):
                a = a + b[0] * i
            return a

        args = torch.randn(3), [torch.randn(3)]
        f = torch.compile(f, fullgraph=True, dynamic=False, sticky_cache=self.path())
        with self.assertRaisesRegex(
            NotImplementedError,
            "structured inputs",
        ):
            f(*args)

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_function_container_outputs(self, device):
        torch.set_default_device(device)

        def f(a, b):
            for i in range(100):
                a = a + b * i
            return [a]

        args = torch.randn(3), torch.randn(3)
        f = torch.compile(f, fullgraph=True, dynamic=False, sticky_cache=self.path())
        from torch._dynamo.exc import InternalTorchDynamoError

        with self.assertRaisesRegex(
            InternalTorchDynamoError,
            "structured outputs",
        ):
            f(*args)

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_function_wrong_shape(self, device):
        torch.set_default_device(device)

        def f(a, b):
            for i in range(100):
                a = a + b * i
            return a

        args = torch.randn(3), torch.randn(3)
        args1 = torch.randn(4), torch.randn(4)
        f(*args)
        expected1 = f(*args1)
        f = torch.compile(f, fullgraph=True, dynamic=False, sticky_cache=self.path())
        f(*args)
        f.save_sticky_cache(prefix=cache_dir())
        with self.assertRaisesRegex(RuntimeError, "Detected recompile"):
            self._test_load(f, args1, expected1, self.path())

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_function_different_dir(self, device):
        torch.set_default_device(device)

        def f(a, b):
            for i in range(100):
                a = a + b * i
            return a

        args = torch.randn(3), torch.randn(3)
        expected = f(*args)
        f = torch.compile(f, fullgraph=True, dynamic=False, sticky_cache=self.path())
        f(*args)
        f.save_sticky_cache(prefix=cache_dir())
        new_prefix = os.path.join(cache_dir(), "new")
        os.makedirs(new_prefix, exist_ok=False)
        shutil.move(os.path.join(cache_dir(), self.path()), new_prefix)

        with self.assertRaises(RuntimeError):
            f.load_sticky_cache(prefix=cache_dir())

        self._test_load(f, args, expected, self.path(), new_prefix)


instantiate_parametrized_tests(TestStickyCache)


if __name__ == "__main__":
    run_tests()
