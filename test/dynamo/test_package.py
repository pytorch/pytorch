# Owner(s): ["module: dynamo"]

import os
import pickle

import torch
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.onnx.operators
import torch.utils.cpp_extension
from torch._dynamo.package import CompilePackage
from torch._inductor.runtime.runtime_utils import cache_dir


class StorageForTesting:
    def __init__(self, path: str):
        self.path = path
        self.backends = {}

    def _write_pickle(self, data, *path: str):
        with open(os.path.join(self.path, *path) + ".pickle", "wb") as f:
            pickle.dump(data, f)

    def write_dynamo(self, dynamo):
        self._write_pickle(dynamo, "dynamo")

    def write_backend(self, backend_id):
        os.makedirs(os.path.join(self.path, backend_id), exist_ok=True)
        self._write_pickle(self.backends[backend_id], backend_id, "fx_graph")

    def _read_pickle(self, *path):
        with open(os.path.join(self.path, *path) + ".pickle", "rb") as f:
            return pickle.load(f)

    def read_backend(self, backend_id):
        return self._read_pickle(backend_id, "fx_graph")

    def read_dynamo(self):
        return self._read_pickle("dynamo")

    def add_backend(self, backend_id, backend):
        self.backends[backend_id] = backend

    def save_package(self, dynamo_cache_entry):
        self.write_dynamo(dynamo_cache_entry)
        for backend_id in dynamo_cache_entry.backend_ids:
            self.write_backend(backend_id)

    def load_package(self):
        dynamo = self.read_dynamo()
        self.backends = {}
        for backend_id in dynamo.backend_ids:
            self.backends[backend_id] = self.read_backend(backend_id)
        return dynamo


class TestPackage(torch._inductor.test_case.TestCase):
    def storage(self):
        path = os.path.join(cache_dir(), f"package_{self.id()}")
        os.makedirs(path, exist_ok=True)
        return StorageForTesting(path)

    def test_basic_fn(self):
        storage = self.storage()

        def fn(x):
            return x + 1

        args = (torch.randn(3, 2),)

        # Saving
        package = CompilePackage(fn)
        compiled_fn = torch._dynamo.optimize(backend="eager", package=package)(fn)
        expected = compiled_fn(*args)
        for backend_id, backend in package.cached_backends.items():
            storage.add_backend(backend_id, backend)
        storage.save_package(package.save())

        # Loading
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with self.assertRaisesRegex(
                RuntimeError,
                "Detected recompile when torch.compile stance is 'fail_on_recompile'",
            ):
                compiled_fn(*args)

            package = CompilePackage(fn, storage.load_package())
            compiled_fn = torch._dynamo.optimize(package=package)(fn)
            package.install(storage.backends)
            self.assertEqual(expected, compiled_fn(*args))

    def test_graph_break_bomb(self):
        storage = self.storage()

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
            backend="eager", package=package, guard_filter_fn=guard_filter_fn
        )(fn)
        N = 10
        args_list = [(torch.tensor(x), 0, N - 1) for x in range(N)]
        for args in args_list:
            compiled_fn(*args)
        for backend_id, backend in package.cached_backends.items():
            storage.add_backend(backend_id, backend)
        storage.save_package(package.save())

        # Loading
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            for args in args_list:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Detected recompile when torch.compile stance is 'fail_on_recompile'",
                ):
                    compiled_fn(*args)
            package = CompilePackage(fn, storage.load_package())
            compiled_fn = torch._dynamo.optimize(
                backend="eager", package=package, guard_filter_fn=guard_filter_fn
            )(fn)
            package.install(storage.backends)
            for args in args_list:
                self.assertEqual(compiled_fn(*args), args[0].sum())

            with self.assertRaisesRegex(
                RuntimeError,
                "Detected recompile when torch.compile stance is 'fail_on_recompile'",
            ):
                compiled_fn(torch.tensor(N), 0, N - 1)

    def test_dynamic_shape(self):
        storage = self.storage()

        def fn(x):
            return x + x.shape[0]

        args = (torch.randn(3, 2),)
        args1 = (torch.randn(5, 2),)
        args2 = (torch.randn(7, 2),)
        expected1 = fn(*args1)

        torch._dynamo.mark_dynamic(args[0], 0, min=3, max=5)

        # Saving
        package = CompilePackage(fn)
        compiled_fn = torch._dynamo.optimize(backend="eager", package=package)(fn)
        compiled_fn(*args)
        for backend_id, backend in package.cached_backends.items():
            storage.add_backend(backend_id, backend)
        storage.save_package(package.save())

        # Loading
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with self.assertRaisesRegex(
                RuntimeError,
                "Detected recompile when torch.compile stance is 'fail_on_recompile'",
            ):
                compiled_fn(*args1)

            package = CompilePackage(fn, storage.load_package())
            compiled_fn = torch._dynamo.optimize(package=package)(fn)
            package.install(storage.backends)

            self.assertEqual(expected1, compiled_fn(*args1))

            with self.assertRaisesRegex(
                RuntimeError,
                "Detected recompile when torch.compile stance is 'fail_on_recompile'",
            ):
                compiled_fn(*args2)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
