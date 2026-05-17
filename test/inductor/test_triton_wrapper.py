# Owner(s): ["module: inductor"]

import ast
import inspect
import os
import pickle
import re
import subprocess
import sys

import torch
import torch._inductor.async_compile
from torch._functorch import config as functorch_config
from torch._inductor import config as inductor_config
from torch._inductor.codecache import PyCodeCache
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


class TestTritonWrapper(TestCase):
    def setUp(self):
        super().setUp()
        PyCodeCache.cache_clear()

    def get_compiled_module(self):
        compiled_module = None
        for v in PyCodeCache.modules:
            if hasattr(v, "benchmark_compiled_module"):
                self.assertTrue(
                    compiled_module is None, "Found multiple compiled modules"
                )
                compiled_module = v

        self.assertTrue(compiled_module is not None)
        return compiled_module

    def test_wrapper_using_gpu_seed(self):
        """
        Make sure the subprocess.check_output does not throw.
        """

        @torch.compile
        def f(x, y):
            # dropout will result in usage of cuda_seed
            z = torch.nn.functional.dropout(x, 0.5)
            return z + y

        N = 10
        x = torch.rand(N).to(device=GPU_TYPE)
        y = torch.rand(N).to(device=GPU_TYPE)
        out = f(x, y)  # noqa: F841
        compiled_module = self.get_compiled_module()
        # to make sure the subprocess runs on the exact same path as the parent process
        # we augment the PYTHONPATH env var
        augmented_pp = ":".join(sys.path)
        if os.environ.get("PYTHONPATH"):
            augmented_pp = f"{os.environ.get('PYTHONPATH')}:{augmented_pp}"
        # now run the compiled module in subprocess and check its output
        bench_out = subprocess.check_output(
            f"{sys.executable} {compiled_module.__file__}".split(),
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": augmented_pp},
        ).decode()

        self.assertTrue(len(bench_out) > 0)

    def test_get_args_and_benchmark_compiled_module(self):
        @torch.compile
        def f(x, y):
            return x * y + x

        N = 10
        x = torch.rand(N).to(device=GPU_TYPE)
        y = torch.rand(N).to(device=GPU_TYPE)
        f(x, y)

        compiled_module = self.get_compiled_module()

        # Verify get_args function exists and is callable
        self.assertTrue(hasattr(compiled_module, "get_args"))
        self.assertTrue(callable(compiled_module.get_args))

        self.assertTrue(hasattr(compiled_module, "benchmark_compiled_module"))
        sig = inspect.signature(compiled_module.benchmark_compiled_module)
        self.assertIn("args", sig.parameters)

        args = compiled_module.get_args()
        self.assertIsInstance(args, list)
        self.assertTrue(len(args) > 0)

        # Verify that running the compiled module as a subprocess works
        augmented_pp = ":".join(sys.path)
        if os.environ.get("PYTHONPATH"):
            augmented_pp = f"{os.environ.get('PYTHONPATH')}:{augmented_pp}"
        bench_out = subprocess.check_output(
            f"{sys.executable} {compiled_module.__file__}".split(),
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": augmented_pp},
        ).decode()
        self.assertTrue(len(bench_out) > 0)

    @functorch_config.patch({"enable_autograd_cache": False})
    def test_get_args_preserves_aliased_inputs(self):
        @torch.compile
        def f(x, y, empty_bool, empty_long):
            return x + y, empty_bool.logical_not(), empty_long + 1

        base = torch.arange(64, device=GPU_TYPE, dtype=torch.long)
        x = torch.as_strided(base, (3, 4), (4, 1), 2)
        y = torch.as_strided(base, (3, 4), (4, 1), 10)
        empty_bool = torch.empty((0, 3), device=GPU_TYPE, dtype=torch.bool)
        empty_long = torch.empty((0, 2), device=GPU_TYPE, dtype=torch.long)
        f(x, y, empty_bool, empty_long)

        compiled_module = self.get_compiled_module()
        get_args_src = inspect.getsource(compiled_module.get_args)
        shared_storage_names = re.findall(
            r"(_shared_storage_\d+) = rand_strided\(",
            get_args_src,
        )
        self.assertEqual(len(shared_storage_names), 1, get_args_src)

        shared_storage = shared_storage_names[0]
        aliased_view_lines = re.findall(
            rf"^\s+\w+ = torch\.as_strided\({shared_storage}, .*$",
            get_args_src,
            re.MULTILINE,
        )
        self.assertEqual(len(aliased_view_lines), 2, get_args_src)

        args = compiled_module.get_args()
        recreated_x, recreated_y, _, _ = args

        self.assertEqual(
            recreated_x.untyped_storage().data_ptr(),
            recreated_y.untyped_storage().data_ptr(),
        )
        self.assertEqual(len(args), 4)

    @functorch_config.patch({"enable_autograd_cache": False})
    @inductor_config.patch({"benchmark_harness_preserve_input_values": False})
    def test_benchmark_does_not_embed_integer_bool_values_by_default(self):
        @torch.compile
        def f(index, mask):
            return index + 1, mask.logical_not()

        index = torch.tensor(
            [-10, -9, -8, -7, -10, -9], device=GPU_TYPE, dtype=torch.int64
        )
        mask = torch.tensor(
            [[True], [False], [True], [False]], device=GPU_TYPE, dtype=torch.bool
        )
        f(index, mask)

        compiled_module = self.get_compiled_module()
        get_args_src = inspect.getsource(compiled_module.get_args)
        args = compiled_module.get_args()

        self.assertNotIn("pickle.loads", get_args_src)
        self.assertEqual(args[0], torch.zeros_like(index))
        self.assertEqual(args[1], torch.zeros_like(mask))

    @functorch_config.patch({"enable_autograd_cache": False})
    @inductor_config.patch({"benchmark_harness_preserve_input_values": True})
    def test_benchmark_preserves_integer_and_bool_input_values_when_enabled(self):
        @torch.compile
        def f(values, index, mask):
            base = torch.zeros((4, values.shape[1]), device=values.device)
            out = torch.index_add(base, 0, index + 10, values)
            return torch.where(mask, out, -out)

        values = torch.randn(6, 3, device=GPU_TYPE)
        index = torch.tensor(
            [-10, -9, -8, -7, -10, -9], device=GPU_TYPE, dtype=torch.int64
        )
        mask = torch.tensor(
            [[True], [False], [True], [False]], device=GPU_TYPE, dtype=torch.bool
        )
        f(values, index, mask)

        compiled_module = self.get_compiled_module()
        get_args_src = inspect.getsource(compiled_module.get_args)
        args = compiled_module.get_args()

        self.assertIn("pickle.loads", get_args_src)
        self.assertEqual(args[1], index)
        self.assertEqual(args[2], mask)
        compiled_module.benchmark_compiled_module(args, times=1, repeat=1)

    @functorch_config.patch({"enable_autograd_cache": False})
    @inductor_config.patch(
        {"benchmark_harness_preserve_input_values": True, "fx_graph_cache": True}
    )
    def test_benchmark_preserved_input_values_bypass_fx_graph_cache(self):
        @torch.compile
        def f(index):
            return index + 10

        index1 = torch.tensor([-10, -9], device=GPU_TYPE, dtype=torch.int64)
        f(index1)
        compiled_module = self.get_compiled_module()
        args = compiled_module.get_args()
        self.assertEqual(args[0], index1)

        torch._dynamo.reset()
        PyCodeCache.cache_clear()

        index2 = torch.tensor([-8, -7], device=GPU_TYPE, dtype=torch.int64)
        f(index2)
        compiled_module = self.get_compiled_module()
        args = compiled_module.get_args()
        self.assertEqual(args[0], index2)

    @functorch_config.patch({"enable_autograd_cache": False})
    @inductor_config.patch({"benchmark_harness_preserve_input_values": True})
    def test_benchmark_preserved_cpu_view_is_compact(self):
        @torch.compile
        def f(index):
            return index + 10

        storage = torch.tensor([1111, -10, -9, 2222], dtype=torch.int64)
        index = storage[1:3]
        f(index)

        compiled_module = self.get_compiled_module()
        get_args_src = inspect.getsource(compiled_module.get_args)
        args = compiled_module.get_args()
        payload_bytes = None
        for node in ast.walk(ast.parse(get_args_src)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "loads"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pickle"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, bytes)
            ):
                payload_bytes = node.args[0].value
                break
        self.assertIsNotNone(payload_bytes, get_args_src)
        payload = pickle.loads(payload_bytes)

        self.assertEqual(args[0], index)
        self.assertEqual(payload, index)
        self.assertEqual(payload.untyped_storage().nbytes(), index.nbytes)

    @functorch_config.patch({"enable_autograd_cache": False})
    @inductor_config.patch({"benchmark_harness_preserve_input_values": True})
    def test_benchmark_skips_overlapping_integer_input_values(self):
        @torch.compile
        def f(index):
            return index + 10

        index = torch.tensor([-10], device=GPU_TYPE, dtype=torch.int64).expand(6)
        f(index)

        compiled_module = self.get_compiled_module()
        get_args_src = inspect.getsource(compiled_module.get_args)
        args = compiled_module.get_args()

        self.assertNotIn("pickle.loads", get_args_src)
        self.assertEqual(args[0], torch.zeros_like(index))


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
