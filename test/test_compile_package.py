# Owner(s): ["oncall: export"]

import copy
import glob
import inspect
import os
import shutil
import types
import unittest

import torch
from torch._dynamo.compile_package import _CompilePackage
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda,
    run_tests,
)


class ClassForTest:
    def __init__(self):
        self.p = torch.tensor(10)

    def g(self, a, b):
        for i in range(30):
            a = a + b * i + self.p
        return a


class TestCompilePackage(TestCase):
    def path(self):
        return os.path.join(cache_dir(), f"package_{self.id()}")

    def _test_save(self, compiled_fn, args, expected, path, index) -> int:
        # From a clean state, save_package() can produce some archive
        # files and the files loaded back can reproduce the expected result.
        # Returns the index of compilation hit in the cache.
        load_dir = path
        load_paths = glob.glob(os.path.join(load_dir, "*.dynamo_code"))
        self.assertEqual(len(load_paths), 0)

        compiled_fn.save_package()
        load_paths = glob.glob(os.path.join(load_dir, "*.dynamo_code"))
        self.assertGreaterEqual(len(load_paths), 1)

        precompiles = []
        for i in range(len(load_paths)):
            precompile = torch._dynamo.compile_package._load_precompile(
                _CompilePackage(), load_dir, i
            )
            self.assertIsInstance(precompile.dynamo_code, types.CodeType)
            precompiles.append(precompile)

        precompile = precompiles[index]
        if isinstance(compiled_fn, torch.nn.Module):
            inputs = (compiled_fn._orig_mod, *args)
        elif inspect.ismethod(compiled_fn._torchdynamo_orig_callable):
            inputs = (compiled_fn._torchdynamo_orig_callable.__self__, *args)
        else:
            inputs = args

        self.assertEqual(expected, precompile(*inputs))

    def _test_load(self, compiled_fn, args, expected, path) -> None:
        # From a clean state with recompile disabled, we can load back the
        # compile package and reproduce expected results.
        # Will clean up the loaded package on disk.
        state = compiled_fn.reset_package()
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with self.assertRaisesRegex(
                RuntimeError, "Detected recompile when torch.compile"
            ):
                compiled_fn(*copy.deepcopy(args))

            compiled_fn.load_package()
            results = compiled_fn(*args)
            self.assertEqual(expected, results)

            shutil.rmtree(path)
            with self.assertRaisesRegex(
                RuntimeError, "Compile package path .* doesn't exist"
            ):
                compiled_fn.load_package()
        compiled_fn.reset_package(state)

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_function_forward_static(self, device):
        torch.set_default_device(device)

        def f(a, b):
            for i in range(30):
                a = a + b * i
            return a

        args = torch.randn(3), torch.randn(3)
        args1 = torch.randn((4, 2)), torch.randn((4, 2))
        expected = f(*args)
        expected1 = f(*args1)
        f = torch.compile(f, fullgraph=True, dynamic=False, package=self.path())
        f(*args)
        self._test_save(f, args, expected, self.path(), 0)
        self._test_load(f, args, expected, self.path())

        f(*args1)
        self._test_save(f, args1, expected1, self.path(), 1)
        self._test_load(f, args1, expected1, self.path())

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_module_forward_static(self, device):
        torch.set_default_device(device)

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("p", torch.tensor(10))

            def forward(self, a, b, c):
                assert c is None
                for i in range(30):
                    a = a + b * i + self.p
                return a

        f = Module()
        args = torch.randn(3), torch.randn(3), None
        args1 = torch.randn((4, 2)), torch.randn((4, 2)), None
        expected = f(*args)
        expected1 = f(*args1)
        f = torch.compile(f, fullgraph=True, dynamic=False, package=self.path())
        f(*args)
        self._test_save(f, args, expected, self.path(), 0)
        self._test_load(f, args, expected, self.path())

        f(*args1)
        self._test_save(f, args1, expected1, self.path(), 1)
        self._test_load(f, args1, expected1, self.path())

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_method_forward_static(self, device):
        torch.set_default_device(device)

        m = ClassForTest()
        args = torch.randn(3), torch.randn(3)
        args1 = torch.randn((4, 2)), torch.randn((4, 2))
        expected = m.g(*args)
        expected1 = m.g(*args1)
        f = torch.compile(m.g, fullgraph=True, dynamic=False, package=self.path())
        f(*args)
        self._test_save(f, args, expected, self.path(), 0)
        self._test_load(f, args, expected, self.path())

        f(*args1)
        self._test_save(f, args1, expected1, self.path(), 1)
        self._test_load(f, args1, expected1, self.path())

    def test_basic_class_method(self):
        class Module:
            def __init__(self):
                self.p = torch.tensor(10)

            def g(self, a, b):
                for i in range(30):
                    a = a + b * i + self.p
                return a

        m = Module()
        args = torch.randn(3), torch.randn(3)
        f = torch.compile(m.g, fullgraph=True, dynamic=False, package=self.path())
        with self.assertRaisesRegex(
            RuntimeError, "Please define the class at global scope"
        ):
            f(*args)

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_function_forward_non_fullgraph(self, device):
        torch.set_default_device(device)

        def f(a, b):
            for i in range(30):
                a = a + b * i
            return a

        with self.assertRaisesRegex(
            RuntimeError, "Compile package is only supported .*fullgraph=True"
        ):
            torch.compile(f, dynamic=False, package=self.path())

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_function_forward_backward(self, device):
        torch.set_default_device(device)

        def f(a, b):
            for i in range(30):
                a = a + b * i
            return a.sum()

        args = torch.randn(3, requires_grad=True), torch.randn(3, requires_grad=True)
        f = torch.compile(f, fullgraph=True, dynamic=False, package=self.path())
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
            for i in range(30):
                a.add_(b * i)
            return a

        args = torch.randn(3), torch.randn(3)
        expected = f(*copy.deepcopy(args))
        f = torch.compile(f, fullgraph=True, dynamic=False, package=self.path())
        f(*copy.deepcopy(args))
        self._test_save(f, copy.deepcopy(args), expected, self.path(), 0)
        self._test_load(f, copy.deepcopy(args), expected, self.path())

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_function_container_inputs(self, device):
        torch.set_default_device(device)

        def f(a, b):
            for i in range(30):
                a = a + b[0] * i
            return a

        args = torch.randn(3), [torch.randn(3)]
        expected = f(*args)
        f = torch.compile(f, fullgraph=True, dynamic=False, package=self.path())
        f(*args)
        self._test_save(f, copy.deepcopy(args), expected, self.path(), 0)
        self._test_load(f, copy.deepcopy(args), expected, self.path())

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_function_container_outputs(self, device):
        torch.set_default_device(device)

        def f(a, b):
            for i in range(30):
                a = a + b * i
            return [a]

        args = torch.randn(3), torch.randn(3)
        f = torch.compile(f, fullgraph=True, dynamic=False, package=self.path())
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
            for i in range(30):
                a = a + b * i
            return a

        args = torch.randn(3), torch.randn(3)
        args1 = torch.randn(4), torch.randn(4)
        f(*args)
        expected1 = f(*args1)
        f = torch.compile(f, fullgraph=True, dynamic=False, package=self.path())
        f(*args)
        f.save_package()
        with self.assertRaisesRegex(RuntimeError, "Detected recompile"):
            self._test_load(f, args1, expected1, self.path())

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_function_different_dir(self, device):
        torch.set_default_device(device)

        def f(a, b):
            for i in range(30):
                a = a + b * i
            return a

        args = torch.randn(3), torch.randn(3)
        expected = f(*args)
        f = torch.compile(f, fullgraph=True, dynamic=False, package=self.path())
        f(*args)
        f.save_package()
        new_prefix = os.path.join(cache_dir(), "new")
        os.makedirs(new_prefix, exist_ok=False)
        shutil.move(self.path(), new_prefix)

        with self.assertRaises(RuntimeError):
            f.load_package()

        new_path = os.path.join(new_prefix, os.path.split(self.path())[-1])
        f = torch.compile(f, fullgraph=True, dynamic=False, package=new_path)
        self._test_load(f, args, expected, new_path)

    def test_basic_function_tensor_meta_mismatch(self):
        def f(a, b):
            for i in range(30):
                a = a + b * i
            return a

        args = torch.randn(3), torch.randn(3)
        expected = f(*args)
        f = torch.compile(f, fullgraph=True, dynamic=False, package=self.path())
        f(*args)
        self._test_save(f, args, expected, self.path(), 0)
        with self.assertRaisesRegex(RuntimeError, "Detected recompile"):
            self._test_load(f, (torch.randn(3), torch.randn(4)), None, self.path())
        with self.assertRaisesRegex(RuntimeError, "Detected recompile"):
            self._test_load(
                f, (torch.zeros(3), torch.zeros(3, dtype=torch.int)), None, self.path()
            )

    @unittest.expectedFailure  # failing with AOTI right now
    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_basic_function_dynamic_shape(self, device):
        torch.set_default_device(device)

        def f(a, b):
            for i in range(30):
                a = a + b * i
            return a

        args = torch.randn(3), torch.randn(3)
        args1 = torch.randn(4, 2), torch.randn(4, 2)
        args2 = torch.randn(4), torch.randn(4)
        args3 = torch.randn(5), torch.randn(5)
        expected = f(*args)
        expected1 = f(*args1)
        expected2 = f(*args2)
        expected3 = f(*args3)
        f = torch.compile(f, fullgraph=True, package=self.path())
        f(*args)
        f(*args1)
        f(*args2)

        self._test_save(f, args3, expected3, self.path(), 2)
        self._test_load(f, args3, expected3, self.path())

        # self._test_save(f, (torch.randn(4, 2), torch.randn(4, 3)), None, self.path(), None)


instantiate_parametrized_tests(TestCompilePackage)


if __name__ == "__main__":
    run_tests()
