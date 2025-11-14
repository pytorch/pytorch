# Owner(s): ["module: cpp"]

import math
import re
import unittest
from pathlib import Path

import torch
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    install_cpp_extension,
    IS_WINDOWS,
    run_tests,
    TestCase,
    xfailIfTorchDynamo,
)


def get_pytorch_version():
    """Get the PyTorch version as a tuple (major, minor, patch)"""
    version_str = torch.__version__.split("+")[0]  # Remove git hash
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
    if match:
        return tuple(int(x) for x in match.groups())
    return (2, 10, 0)  # Default to 2.10.0


PYTORCH_VERSION = get_pytorch_version()
IS_PYTORCH_2_9 = PYTORCH_VERSION < (2, 10, 0)
IS_PYTORCH_2_10_OR_LATER = PYTORCH_VERSION >= (2, 10, 0)


def skipIfPyTorch2_9(reason):
    """Skip test if running on PyTorch 2.9"""
    return unittest.skipIf(IS_PYTORCH_2_9, reason)


# TODO: Fix this error in Windows:
# LINK : error LNK2001: unresolved external symbol PyInit__C
if not IS_WINDOWS:

    class TestLibtorchAgnosticVersioned(TestCase):
        """
        Tests for versioned libtorch_agnostic extensions.

        This test class supports testing both:
        - libtorch_agnostic_2_9: Extension built with TORCH_TARGET_VERSION=2.9.0
        - libtorch_agnostic_2_10: Extension built with TORCH_TARGET_VERSION=2.10.0

        Both extensions must be available for the tests to run.
        """

        ops_2_9 = None
        ops_2_10 = None

        @classmethod
        def setUpClass(cls):
            print(f"Running tests with PyTorch {'.'.join(map(str, PYTORCH_VERSION))}")

            # Install and import the 2.9 extension (always needed)
            try:
                import libtorch_agnostic_2_9

                cls.ops_2_9 = libtorch_agnostic_2_9.ops
            except ImportError:
                extension_root = (
                    Path(__file__).parent / "libtorch_agnostic_2_9_extension"
                )
                install_cpp_extension(extension_root=extension_root)
                import libtorch_agnostic_2_9

                cls.ops_2_9 = libtorch_agnostic_2_9.ops

            # Install and import the 2.10 extension (only if on PyTorch 2.10+)
            if IS_PYTORCH_2_10_OR_LATER:
                try:
                    import libtorch_agnostic_2_10

                    cls.ops_2_10 = libtorch_agnostic_2_10.ops
                except ImportError:
                    extension_root = (
                        Path(__file__).parent / "libtorch_agnostic_2_10_extension"
                    )
                    install_cpp_extension(extension_root=extension_root)
                    import libtorch_agnostic_2_10

                    cls.ops_2_10 = libtorch_agnostic_2_10.ops
            else:
                print("Skipping 2.10 extension (running on PyTorch 2.9)")
                cls.ops_2_10 = None

        # ============================================================================
        # Tests for 2.9 features
        # ============================================================================

        @onlyCPU
        def test_2_9_slow_sgd(self, device):
            ops = self.ops_2_9

            param = torch.rand(5, device=device)
            grad = torch.rand_like(param)
            weight_decay = 0.01
            lr = 0.001
            maximize = False

            new_param = ops.sgd_out_of_place(param, grad, weight_decay, lr, maximize)
            torch._fused_sgd_(
                (param,),
                (grad,),
                (),
                weight_decay=weight_decay,
                momentum=0.0,
                lr=lr,
                dampening=0.0,
                nesterov=False,
                maximize=maximize,
                is_first_step=False,
            )
            self.assertEqual(new_param, param)

        @onlyCUDA
        def test_2_9_identity_does_not_hog_memory(self, device):
            ops = self.ops_2_9

            def _run_identity(prior_mem):
                t = torch.rand(32, 32, device=device)
                self.assertGreater(torch.cuda.memory_allocated(device), prior_mem)
                identi_t = ops.identity(t)
                assert identi_t is t

            init_mem = torch.cuda.memory_allocated(device)

            for _ in range(3):
                _run_identity(init_mem)
                curr_mem = torch.cuda.memory_allocated(device)
                self.assertEqual(curr_mem, init_mem)

        def test_2_9_exp_neg_is_leaf(self, device):
            ops = self.ops_2_9

            t1 = torch.rand(2, 3, device=device)
            t2 = torch.rand(3, 2, device=device)
            t3 = torch.rand(2, device=device)

            exp, neg, is_leaf = ops.exp_neg_is_leaf(t1, t2, t3)
            self.assertEqual(exp, torch.exp(t1))
            self.assertEqual(neg, torch.neg(t2))
            self.assertEqual(is_leaf, t3.is_leaf)

        def test_2_9_my_abs(self, device):
            ops = self.ops_2_9

            t = torch.rand(32, 16, device=device) - 0.5
            res = ops.my_abs(t)
            self.assertEqual(res, torch.abs(t))

            def _make_cuda_tensors(prior_mem):
                cuda_t = ops.my_abs(t)
                self.assertGreater(torch.cuda.memory_allocated(device), prior_mem)
                self.assertEqual(cuda_t, torch.abs(t))

            if t.is_cuda:
                init_mem = torch.cuda.memory_allocated(device)
                for _ in range(3):
                    _make_cuda_tensors(init_mem)
                    curr_mem = torch.cuda.memory_allocated(device)
                    self.assertEqual(curr_mem, init_mem)

        def test_2_9_neg_exp(self, device):
            ops = self.ops_2_9

            t = torch.rand(32, 16, device=device) - 0.5
            res = ops.neg_exp(t)
            self.assertEqual(res, torch.neg(torch.exp(t)))

            def _make_cuda_tensors(prior_mem):
                cuda_res = ops.neg_exp(t)
                self.assertGreater(torch.cuda.memory_allocated(device), prior_mem)
                self.assertEqual(cuda_res, torch.neg(torch.exp(t)))

            if t.is_cuda:
                init_mem = torch.cuda.memory_allocated(device)
                for _ in range(3):
                    _make_cuda_tensors(init_mem)
                    curr_mem = torch.cuda.memory_allocated(device)
                    self.assertEqual(curr_mem, init_mem)

        def test_2_9_divide_neg_exp(self, device):
            ops = self.ops_2_9

            t = torch.zeros(2, 3, device=device) - 0.5
            res = ops.divide_neg_exp(t)
            self.assertEqual(res, torch.neg(t) / torch.exp(t))

            def _make_cuda_tensors(prior_mem):
                cuda_res = ops.divide_neg_exp(t)
                self.assertGreater(torch.cuda.memory_allocated(device), prior_mem)
                self.assertEqual(cuda_res, torch.neg(t) / torch.exp(t))

            if t.is_cuda:
                init_mem = torch.cuda.memory_allocated(device)
                for _ in range(3):
                    _make_cuda_tensors(init_mem)
                    curr_mem = torch.cuda.memory_allocated(device)
                    self.assertEqual(curr_mem, init_mem)

        def test_2_9_is_contiguous(self, device):
            ops = self.ops_2_9

            t = torch.rand(2, 7, device=device)
            self.assertTrue(ops.is_contiguous(t))
            self.assertFalse(ops.is_contiguous(t.transpose(0, 1)))

        @xfailIfTorchDynamo
        def test_2_9_my_ones_like(self, device):
            ops = self.ops_2_9

            t = torch.rand(3, 1, device=device) - 0.5
            cpu_t = ops.my_ones_like(t, "cpu")
            self.assertEqual(cpu_t, torch.ones_like(t, device="cpu"))

            def _make_cuda_tensors(prior_mem):
                cuda_t = ops.my_ones_like(t, device)
                self.assertGreater(torch.cuda.memory_allocated(device), prior_mem)
                self.assertEqual(cuda_t, torch.ones_like(t, device=device))

            if t.is_cuda:
                init_mem = torch.cuda.memory_allocated(device)
                for _ in range(3):
                    _make_cuda_tensors(init_mem)
                    curr_mem = torch.cuda.memory_allocated(device)
                    self.assertEqual(curr_mem, init_mem)

        def test_2_9_my_transpose(self, device):
            ops = self.ops_2_9

            t = torch.rand(2, 7, device=device)
            out = ops.my_transpose(t, 0, 1)
            self.assertEqual(out, torch.transpose(t, 0, 1))

            with self.assertRaisesRegex(RuntimeError, "API call failed"):
                ops.my_transpose(t, 1, 2)

        def test_2_9_my_empty_like(self, device):
            ops = self.ops_2_9

            deterministic = torch.are_deterministic_algorithms_enabled()
            try:
                torch.use_deterministic_algorithms(True)

                t = torch.rand(2, 7, device=device)
                out = ops.my_empty_like(t)
                self.assertTrue(id(out != id(t)))
                self.assertEqual(out, torch.empty_like(t))
            finally:
                torch.use_deterministic_algorithms(deterministic)

        @onlyCPU
        def test_2_9_my_zero_(self, device):
            ops = self.ops_2_9

            t = torch.rand(2, 7, device=device)
            out = ops.my_zero_(t)
            self.assertEqual(id(out), id(t))
            self.assertEqual(out, torch.zeros_like(t))

        def test_2_9_my_amax(self, device):
            ops = self.ops_2_9

            t = torch.rand(2, 7, device=device)
            out = ops.my_amax(t)
            self.assertEqual(out, torch.amax(t, 0))

        def test_2_9_my_amax_vec(self, device):
            ops = self.ops_2_9

            t = torch.rand(2, 7, 5, device=device)
            out = ops.my_amax_vec(t)
            self.assertEqual(out, torch.amax(t, (0, 1)))

        def test_2_9_my_is_cpu(self, device):
            ops = self.ops_2_9

            t = torch.rand(2, 7, device=device)
            out = ops.my_is_cpu(t)
            self.assertEqual(out, t.is_cpu)

        def test_2_9_fill_infinity(self, device):
            ops = self.ops_2_9

            t = torch.rand(3, 4, device=device)
            out = ops.fill_infinity(t)

            self.assertEqual(id(out), id(t))
            expected = torch.full_like(t, math.inf)
            self.assertEqual(out, expected)

        @onlyCPU
        def test_2_9_default_constructor(self):
            ops = self.ops_2_9

            defined_tensor_is_defined = ops.test_default_constructor(True)
            self.assertTrue(defined_tensor_is_defined)

            undefined_tensor_is_defined = ops.test_default_constructor(False)
            self.assertFalse(undefined_tensor_is_defined)

        def test_2_9_my_pad(self, device):
            ops = self.ops_2_9

            t = torch.rand(2, 3, device=device)
            out = ops.my_pad(t)
            expected = torch.nn.functional.pad(t, [1, 2, 2, 1], "constant", 0.0)
            self.assertEqual(out, expected)

        def test_2_9_my_narrow(self, device):
            ops = self.ops_2_9

            t = torch.randn(2, 5, device=device)

            dim0 = 0
            start0 = 0
            length0 = 1
            out0 = ops.my_narrow(t, dim0, start0, length0)
            expected0 = torch.narrow(t, dim0, start0, length0)
            self.assertEqual(out0, expected0)

        @onlyCUDA
        @deviceCountAtLeast(2)
        def test_2_9_device_guard(self, device):
            ops = self.ops_2_9

            device_index = 1
            out = ops.test_device_guard(device_index)
            self.assertEqual(out, device_index)

        @onlyCUDA
        @deviceCountAtLeast(2)
        def test_2_9_device_guard_set_index(self, device):
            ops = self.ops_2_9

            out = ops.test_device_guard_set_index()
            self.assertEqual(out, 0)

        @onlyCUDA
        def test_2_9_stream(self, device):
            ops = self.ops_2_9

            stream = torch.cuda.Stream()
            device = torch.cuda.current_device()

            with stream:
                expected_stream_id = torch.cuda.current_stream(0).stream_id
                stream_id = ops.test_stream(device)

            self.assertEqual(stream_id, expected_stream_id)

        @onlyCUDA
        @deviceCountAtLeast(2)
        def test_2_9_get_current_device_index(self, device):
            ops = self.ops_2_9

            prev_device = torch.cuda.current_device()

            try:
                expected_device = 1
                torch.cuda.set_device(expected_device)

                current_device = ops.test_get_current_device_index()
                self.assertEqual(current_device, expected_device)
            finally:
                torch.cuda.set_device(prev_device)

        def test_2_9_my_new_empty_dtype_variant(self, device):
            ops = self.ops_2_9

            deterministic = torch.are_deterministic_algorithms_enabled()
            try:
                torch.use_deterministic_algorithms(True)
                t = torch.randn(3, 4, device=device)
                out = ops.my_new_empty_dtype_variant(t)
                ref_out = t.new_empty((2, 5), dtype=torch.bfloat16)

                self.assertEqual(out, ref_out, exact_device=True)
            finally:
                torch.use_deterministic_algorithms(deterministic)

        def test_2_9_my_new_zeros_dtype_variant(self, device):
            ops = self.ops_2_9

            t = torch.randn(3, 4, device=device)
            out = ops.my_new_zeros_dtype_variant(t)
            ref_out = t.new_zeros((2, 5), dtype=torch.float)
            self.assertEqual(out, ref_out, exact_device=True)

        # ============================================================================
        # Tests for 2.10 features (only work with 2.10 extension)
        # These tests are skipped when running on PyTorch 2.9 runtime
        # ============================================================================

        @skipIfPyTorch2_9("Requires PyTorch 2.10+ runtime")
        def test_2_10_my_copy_(self, device):
            ops = self.ops_2_10

            dst = torch.empty(2, 5, device=device)
            src = torch.randn(2, 5, device=device)

            result = ops.my_copy_(dst, src, False)
            expected = src
            self.assertEqual(result, expected)
            self.assertEqual(result.data_ptr(), dst.data_ptr())

        @skipIfPyTorch2_9("Requires PyTorch 2.10+ runtime")
        def test_2_10_my_clone(self, device):
            ops = self.ops_2_10

            t = torch.randn(2, 5, device=device)

            result = ops.my_clone(t)
            expected = t.clone()
            self.assertEqual(result, expected)
            self.assertNotEqual(result.data_ptr(), expected.data_ptr())
            self.assertEqual(result.stride(), expected.stride())

        @skipIfPyTorch2_9("Requires PyTorch 2.10+ runtime")
        def test_2_10_my__foreach_mul_(self, device):
            ops = self.ops_2_10

            N = 5
            tensors = [torch.rand(32, 16, device=device) for _ in range(N)]
            tensors_c = [t.clone() for t in tensors]
            others = [torch.rand(32, 16, device=device) for _ in range(N)]

            ops.my__foreach_mul_(tensors, others)
            expected_values = torch._foreach_mul(tensors_c, others)

            for tensor_t, expected_t in zip(tensors, expected_values):
                self.assertEqual(tensor_t, expected_t)

        @skipIfPyTorch2_9("Requires PyTorch 2.10+ runtime")
        def test_2_10_my__foreach_mul(self, device):
            ops = self.ops_2_10

            N = 5
            tensors = [torch.rand(32, 16, device=device) for _ in range(N)]
            others = [torch.rand(32, 16, device=device) for _ in range(N)]

            result = ops.my__foreach_mul(tensors, others)
            expected = torch._foreach_mul(tensors, others)

            for result_t, expected_t in zip(result, expected):
                self.assertEqual(result_t, expected_t)

            def _make_cuda_tensors(prior_mem):
                cuda_res = ops.my__foreach_mul(tensors, others)
                self.assertGreater(torch.cuda.memory_allocated(device), prior_mem)

                expected = torch._foreach_mul(tensors, others)
                for result_t, expected_t in zip(cuda_res, expected):
                    self.assertEqual(result_t, expected_t)

            if tensors[0].is_cuda:
                init_mem = torch.cuda.memory_allocated(device)
                for _ in range(3):
                    _make_cuda_tensors(init_mem)
                    curr_mem = torch.cuda.memory_allocated(device)
                    self.assertEqual(curr_mem, init_mem)

        @skipIfPyTorch2_9("Requires PyTorch 2.10+ runtime")
        def test_2_10_make_tensor_clones_and_call_foreach(self, device):
            ops = self.ops_2_10

            t1 = torch.rand(2, 5, device=device)
            t2 = torch.rand(3, 4, device=device)
            result = ops.make_tensor_clones_and_call_foreach(t1, t2)
            self.assertEqual(result[0], t1 * t1)
            self.assertEqual(result[1], t2 * t2)

        @skipIfPyTorch2_9("Requires PyTorch 2.10+ runtime")
        @onlyCUDA
        def test_2_10_device(self, device):
            ops = self.ops_2_10

            cuda_device = ops.test_device_constructor(
                is_cuda=True, index=1, use_str=False
            )
            self.assertEqual(cuda_device, torch.device("cuda:1"))
            cuda_device = ops.test_device_constructor(
                is_cuda=True, index=1, use_str=True
            )
            self.assertEqual(cuda_device, torch.device("cuda:1"))

            self.assertEqual(ops.test_device_index(cuda_device), 1)
            self.assertTrue(
                ops.test_device_equality(cuda_device, torch.device("cuda:1"))
            )
            self.assertFalse(
                ops.test_device_equality(cuda_device, torch.device("cuda:0"))
            )
            self.assertFalse(ops.test_device_is_cpu(cuda_device))
            self.assertTrue(ops.test_device_is_cuda(cuda_device))

            cuda_0_device = ops.test_device_set_index(cuda_device, 0)
            self.assertEqual(cuda_0_device, torch.device("cuda:0"))

            cpu_device = ops.test_device_constructor(False, 0, False)
            self.assertEqual(cpu_device, torch.device("cpu"))
            self.assertTrue(ops.test_device_equality(cpu_device, torch.device("cpu")))
            self.assertTrue(ops.test_device_is_cpu(cpu_device))
            self.assertFalse(ops.test_device_is_cuda(cpu_device))
            self.assertFalse(ops.test_device_equality(cpu_device, cuda_device))

            with self.assertRaisesRegex(
                RuntimeError, "Device index 129 is out of range for int8_t"
            ):
                ops.test_device_constructor(is_cuda=True, index=129, use_str=False)

            with self.assertRaisesRegex(
                RuntimeError, "Device index 129 is out of range for int8_t"
            ):
                ops.test_device_set_index(cuda_device, 129)

        @skipIfPyTorch2_9("Requires PyTorch 2.10+ runtime")
        @onlyCUDA
        @deviceCountAtLeast(2)
        def test_2_10_tensor_device(self, device):
            ops = self.ops_2_10

            t = torch.randn(2, 3)
            self.assertEqual(ops.test_tensor_device(t), t.device)

            t_cuda = torch.randn(2, 3, device="cuda")
            self.assertEqual(ops.test_tensor_device(t_cuda), t_cuda.device)

            t_cuda_1 = torch.randn(2, 3, device="cuda:1")
            self.assertEqual(ops.test_tensor_device(t_cuda_1), t_cuda_1.device)

        @skipIfPyTorch2_9("Requires PyTorch 2.10+ runtime")
        @onlyCPU
        @xfailIfTorchDynamo
        def test_2_10_parallel_for(self, device):
            ops = self.ops_2_10

            num_threads = torch.get_num_threads()
            size = 100
            grain_size = 10
            expected_num_threads_used = min(
                (size + grain_size - 1) // grain_size, num_threads
            )

            result = ops.test_parallel_for(size, grain_size)
            result_thread_ids = torch.unique(torch.bitwise_right_shift(result, 32))
            result_values = torch.bitwise_and(result, 0xFFFFFFFF)
            expected = torch.arange(size, dtype=torch.int64)

            self.assertEqual(result_values, expected)
            self.assertEqual(result_thread_ids, torch.arange(expected_num_threads_used))

        @skipIfPyTorch2_9("Requires PyTorch 2.10+ runtime")
        @onlyCPU
        def test_2_10_get_num_threads(self, device):
            ops = self.ops_2_10

            num_threads = ops.test_get_num_threads()
            expected_num_threads = torch.get_num_threads()
            self.assertEqual(num_threads, expected_num_threads)

        @skipIfPyTorch2_9("Requires PyTorch 2.10+ runtime")
        def test_2_10_my_empty(self, device):
            ops = self.ops_2_10

            deterministic = torch.are_deterministic_algorithms_enabled()
            try:
                torch.use_deterministic_algorithms(True)

                size = [2, 3]
                result = ops.my_empty(size, None, None, None)
                expected = torch.empty(size)
                self.assertEqual(result, expected, exact_device=True)

                result_float = ops.my_empty(size, torch.float32, None, None)
                expected_float = torch.empty(size, dtype=torch.float32)
                self.assertEqual(result_float, expected_float, exact_device=True)

                result_with_device = ops.my_empty(size, torch.float64, device, None)
                expected_with_device = torch.empty(
                    size, dtype=torch.float64, device=device
                )
                self.assertEqual(
                    result_with_device, expected_with_device, exact_device=True
                )

                if device == "cuda":
                    result_pinned = ops.my_empty(size, torch.float32, "cpu", True)
                    expected_pinned = torch.empty(
                        size, dtype=torch.float32, device="cpu", pin_memory=True
                    )
                    self.assertEqual(result_pinned, expected_pinned, exact_device=True)
                    self.assertTrue(result_pinned.is_pinned())
            finally:
                torch.use_deterministic_algorithms(deterministic)

        @skipIfPyTorch2_9("Requires PyTorch 2.10+ runtime")
        def test_2_10_my_flatten(self, device):
            ops = self.ops_2_10

            t = torch.randn(2, 3, 4, device=device)
            result = ops.my_flatten(t, 0, 1)
            expected = torch.flatten(t, 0, 1)
            self.assertEqual(result, expected)

            result_all = ops.my_flatten(t, 0, -1)
            expected_all = torch.flatten(t, 0, -1)
            self.assertEqual(result_all, expected_all)

        @skipIfPyTorch2_9("Requires PyTorch 2.10+ runtime")
        def test_2_10_my_reshape(self, device):
            ops = self.ops_2_10

            t = torch.randn(2, 3, 4, device=device)
            shape = [6, 4]
            result = ops.my_reshape(t, shape)
            expected = torch.reshape(t, shape)
            self.assertEqual(result, expected)

        @skipIfPyTorch2_9("Requires PyTorch 2.10+ runtime")
        def test_2_10_my_view(self, device):
            ops = self.ops_2_10

            t = torch.randn(2, 3, 4, device=device)
            size = [6, 4]
            result = ops.my_view(t, size)
            expected = t.view(size)
            self.assertEqual(result, expected)

    instantiate_device_type_tests(TestLibtorchAgnosticVersioned, globals())


if __name__ == "__main__":
    run_tests()
