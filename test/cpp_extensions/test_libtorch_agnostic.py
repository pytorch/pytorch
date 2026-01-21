# Owner(s): ["module: cpp"]

import math
import sysconfig
import unittest
from pathlib import Path

import torch
from torch.testing._internal.common_cuda import _get_torch_cuda_version
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    dtypes,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
)
from torch.testing._internal.common_dtype import all_types_and
from torch.testing._internal.common_utils import (
    install_cpp_extension,
    parametrize,
    run_tests,
    skipIfRocm,
    skipIfTorchDynamo,
    skipIfWindows,
    TestCase,
    xfailIfTorchDynamo,
)


def get_supported_dtypes():
    """Return a list of dtypes that are supported by torch stable ABI."""
    return [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.float8_e5m2,
        torch.float8_e4m3fn,
        torch.float8_e5m2fnuz,
        torch.float8_e4m3fnuz,
        torch.complex32,
        torch.complex64,
        torch.complex128,
        torch.bool,
    ]


def skipIfTorchVersionLessThan(major, minor):
    """Skip test if PyTorch version is less than specified version."""

    def decorator(func):
        version_parts = torch.__version__.split(".")
        current_major = int(version_parts[0])
        current_minor = int(
            version_parts[1].split("+")[0].split("a")[0].split("b")[0].split("rc")[0]
        )

        should_skip = (current_major < major) or (
            current_major == major and current_minor < minor
        )
        reason = f"Test requires PyTorch >= {major}.{minor}, current version is {torch.__version__}"

        return unittest.skipIf(should_skip, reason)(func)

    return decorator


@unittest.skipIf(
    sysconfig.get_config_var("Py_GIL_DISABLED") == 1,
    "Cpython limited API not available, see https://github.com/python/cpython/issues/111506",
)
class TestLibtorchAgnostic(TestCase):
    """
    Tests for versioned libtorch_agnostic extensions.

    This test class supports testing both:

    - libtorch_agn_2_9: Extension built with TORCH_TARGET_VERSION=2.9.0
    - libtorch_agn_2_10: Extension built with TORCH_TARGET_VERSION=2.10.0

    Tests should be decorated with @skipIfTorchVersionLessThan to indicate the
    version that they target.
    """

    @classmethod
    def setUpClass(cls):
        # Build both 2.9 and 2.10 extensions
        base_dir = Path(__file__).parent

        try:
            import libtorch_agn_2_9  # noqa: F401
        except Exception:
            install_cpp_extension(
                extension_root=base_dir / "libtorch_agn_2_9_extension"
            )

        # Only build 2.10 extension if running on PyTorch 2.10+
        import re

        version_parts = torch.__version__.split(".")
        current_major = int(version_parts[0])
        # Extract just the numeric part of the minor version (handles "10+git", "10a1", etc.)
        current_minor = int(re.match(r"\d+", version_parts[1]).group())

        if (current_major > 2) or (current_major == 2 and current_minor >= 10):
            try:
                import libtorch_agn_2_10  # noqa: F401
            except Exception:
                install_cpp_extension(
                    extension_root=base_dir / "libtorch_agn_2_10_extension"
                )
        else:
            print(f"Skipping 2.10 extension (running on PyTorch {torch.__version__})")

    @onlyCPU
    def test_slow_sgd(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        param = torch.rand(5, device=device)
        grad = torch.rand_like(param)
        weight_decay = 0.01
        lr = 0.001
        maximize = False

        new_param = libtorch_agnostic.ops.sgd_out_of_place(
            param, grad, weight_decay, lr, maximize
        )
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
    def test_identity_does_not_hog_memory(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        def _run_identity(prior_mem):
            t = torch.rand(32, 32, device=device)
            self.assertGreater(torch.cuda.memory_allocated(device), prior_mem)
            identi_t = libtorch_agnostic.ops.identity(t)
            assert identi_t is t

        init_mem = torch.cuda.memory_allocated(device)

        for _ in range(3):
            _run_identity(init_mem)
            curr_mem = torch.cuda.memory_allocated(device)
            self.assertEqual(curr_mem, init_mem)

    def test_exp_neg_is_leaf(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        t1 = torch.rand(2, 3, device=device)
        t2 = torch.rand(3, 2, device=device)
        t3 = torch.rand(2, device=device)

        exp, neg, is_leaf = libtorch_agnostic.ops.exp_neg_is_leaf(t1, t2, t3)
        self.assertEqual(exp, torch.exp(t1))
        self.assertEqual(neg, torch.neg(t2))
        self.assertEqual(is_leaf, t3.is_leaf)

    def test_my_abs(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.rand(32, 16, device=device) - 0.5
        res = libtorch_agnostic.ops.my_abs(t)
        self.assertEqual(res, torch.abs(t))

        def _make_cuda_tensors(prior_mem):
            cuda_t = libtorch_agnostic.ops.my_abs(t)
            self.assertGreater(torch.cuda.memory_allocated(device), prior_mem)
            self.assertEqual(cuda_t, torch.abs(t))

        if t.is_cuda:
            init_mem = torch.cuda.memory_allocated(device)
            for _ in range(3):
                _make_cuda_tensors(init_mem)
                curr_mem = torch.cuda.memory_allocated(device)
                self.assertEqual(curr_mem, init_mem)

    def test_neg_exp(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.rand(32, 16, device=device) - 0.5
        res = libtorch_agnostic.ops.neg_exp(t)
        self.assertEqual(res, torch.neg(torch.exp(t)))

        def _make_cuda_tensors(prior_mem):
            cuda_res = libtorch_agnostic.ops.neg_exp(t)
            self.assertGreater(torch.cuda.memory_allocated(device), prior_mem)
            self.assertEqual(cuda_res, torch.neg(torch.exp(t)))

        if t.is_cuda:
            init_mem = torch.cuda.memory_allocated(device)
            for _ in range(3):
                _make_cuda_tensors(init_mem)
                curr_mem = torch.cuda.memory_allocated(device)
                self.assertEqual(curr_mem, init_mem)

    def test_divide_neg_exp(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.zeros(2, 3, device=device) - 0.5
        res = libtorch_agnostic.ops.divide_neg_exp(t)
        self.assertEqual(res, torch.neg(t) / torch.exp(t))

        def _make_cuda_tensors(prior_mem):
            cuda_res = libtorch_agnostic.ops.divide_neg_exp(t)
            self.assertGreater(torch.cuda.memory_allocated(device), prior_mem)
            self.assertEqual(cuda_res, torch.neg(t) / torch.exp(t))

        if t.is_cuda:
            init_mem = torch.cuda.memory_allocated(device)
            for _ in range(3):
                _make_cuda_tensors(init_mem)
                curr_mem = torch.cuda.memory_allocated(device)
                self.assertEqual(curr_mem, init_mem)

    def test_is_contiguous(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.rand(2, 7, device=device)
        self.assertTrue(libtorch_agnostic.ops.is_contiguous(t))
        self.assertFalse(libtorch_agnostic.ops.is_contiguous(t.transpose(0, 1)))

    # TODO: Debug this:
    # torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
    # call_function libtorch_agnostic.my_ones_like.default(*(FakeTensor(..., size=(3, 1)), 'cpu'),
    # **{}): got AssertionError("tensor's device must be `meta`, got cpu instead")
    @xfailIfTorchDynamo
    def test_my_ones_like(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.rand(3, 1, device=device) - 0.5
        cpu_t = libtorch_agnostic.ops.my_ones_like(t, "cpu")
        self.assertEqual(cpu_t, torch.ones_like(t, device="cpu"))

        def _make_cuda_tensors(prior_mem):
            cuda_t = libtorch_agnostic.ops.my_ones_like(t, device)
            self.assertGreater(torch.cuda.memory_allocated(device), prior_mem)
            self.assertEqual(cuda_t, torch.ones_like(t, device=device))

        if t.is_cuda:
            init_mem = torch.cuda.memory_allocated(device)
            for _ in range(3):
                _make_cuda_tensors(init_mem)
                curr_mem = torch.cuda.memory_allocated(device)
                self.assertEqual(curr_mem, init_mem)

    def test_my_transpose(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.rand(2, 7, device=device)
        out = libtorch_agnostic.ops.my_transpose(t, 0, 1)
        self.assertEqual(out, torch.transpose(t, 0, 1))

        with self.assertRaisesRegex(RuntimeError, "API call failed"):
            libtorch_agnostic.ops.my_transpose(t, 1, 2)

    def test_my_empty_like(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            # set use_deterministic_algorithms to fill uninitialized memory
            torch.use_deterministic_algorithms(True)

            t = torch.rand(2, 7, device=device)
            out = libtorch_agnostic.ops.my_empty_like(t)
            self.assertTrue(id(out != id(t)))
            self.assertEqual(out, torch.empty_like(t))
        finally:
            torch.use_deterministic_algorithms(deterministic)

    @onlyCPU
    def test_my_zero_(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.rand(2, 7, device=device)
        out = libtorch_agnostic.ops.my_zero_(t)
        self.assertEqual(id(out), id(t))
        self.assertEqual(out, torch.zeros_like(t))

    def test_my_amax(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.rand(2, 7, device=device)
        out = libtorch_agnostic.ops.my_amax(t)
        self.assertEqual(out, torch.amax(t, 0))

    def test_my_amax_vec(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.rand(2, 7, 5, device=device)
        out = libtorch_agnostic.ops.my_amax_vec(t)
        self.assertEqual(out, torch.amax(t, (0, 1)))

    def test_my_is_cpu(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.rand(2, 7, device=device)
        out = libtorch_agnostic.ops.my_is_cpu(t)
        self.assertEqual(out, t.is_cpu)

    def test_fill_infinity(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.rand(3, 4, device=device)
        out = libtorch_agnostic.ops.fill_infinity(t)

        self.assertEqual(id(out), id(t))
        expected = torch.full_like(t, math.inf)
        self.assertEqual(out, expected)

    @onlyCPU
    def test_default_constructor(self):
        import libtorch_agn_2_9 as libtorch_agnostic

        defined_tensor_is_defined = libtorch_agnostic.ops.test_default_constructor(True)
        self.assertTrue(defined_tensor_is_defined)

        undefined_tensor_is_defined = libtorch_agnostic.ops.test_default_constructor(
            False
        )
        self.assertFalse(undefined_tensor_is_defined)

    def test_my_pad(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.rand(2, 3, device=device)
        out = libtorch_agnostic.ops.my_pad(t)
        expected = torch.nn.functional.pad(t, [1, 2, 2, 1], "constant", 0.0)
        self.assertEqual(out, expected)

    def test_my_narrow(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.randn(2, 5, device=device)

        dim0 = 0
        start0 = 0
        length0 = 1
        out0 = libtorch_agnostic.ops.my_narrow(t, dim0, start0, length0)
        expected0 = torch.narrow(t, dim0, start0, length0)
        self.assertEqual(out0, expected0)

    @onlyCUDA
    @deviceCountAtLeast(2)
    def test_device_guard(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        device_index = 1
        out = libtorch_agnostic.ops.test_device_guard(device_index)
        self.assertEqual(out, device_index)

    @onlyCUDA
    @deviceCountAtLeast(2)
    def test_device_guard_set_index(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        # This test creates a DeviceGuard with index 1, then sets it to index 0
        # and returns the current device (should be 0)
        out = libtorch_agnostic.ops.test_device_guard_set_index()
        self.assertEqual(out, 0)

    @onlyCUDA
    def test_stream(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        stream = torch.cuda.Stream()
        device = torch.cuda.current_device()

        with stream:
            expected_stream_id = torch.cuda.current_stream(0).stream_id
            stream_id = libtorch_agnostic.ops.test_stream(device)

        self.assertEqual(stream_id, expected_stream_id)

    @onlyCUDA
    @deviceCountAtLeast(2)
    def test_get_current_device_index(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        prev_device = torch.cuda.current_device()

        try:
            expected_device = 1
            torch.cuda.set_device(expected_device)

            current_device = libtorch_agnostic.ops.test_get_current_device_index()
            self.assertEqual(current_device, expected_device)
        finally:
            torch.cuda.set_device(prev_device)

    def test_my_new_empty_dtype_variant(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            # set use_deterministic_algorithms to fill uninitialized memory
            torch.use_deterministic_algorithms(True)
            t = torch.randn(3, 4, device=device)
            out = libtorch_agnostic.ops.my_new_empty_dtype_variant(t)
            ref_out = t.new_empty((2, 5), dtype=torch.bfloat16)

            self.assertEqual(out, ref_out, exact_device=True)
        finally:
            torch.use_deterministic_algorithms(deterministic)

    def test_my_new_zeros_dtype_variant(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.randn(3, 4, device=device)
        out = libtorch_agnostic.ops.my_new_zeros_dtype_variant(t)
        ref_out = t.new_zeros((2, 5), dtype=torch.float)
        self.assertEqual(out, ref_out, exact_device=True)

    def test_my_copy_(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        dst = torch.empty(2, 5, device=device)
        src = torch.randn(2, 5, device=device)

        result = libtorch_agnostic.ops.my_copy_(dst, src, False)
        expected = src
        self.assertEqual(result, expected)
        self.assertEqual(result.data_ptr(), dst.data_ptr())

    def test_my_clone(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.randn(2, 5, device=device)

        result = libtorch_agnostic.ops.my_clone(t)
        expected = t.clone()
        self.assertEqual(result, expected)
        self.assertNotEqual(result.data_ptr(), expected.data_ptr())
        self.assertEqual(result.stride(), expected.stride())

    @skipIfTorchVersionLessThan(2, 10)
    def test_my__foreach_mul_(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        N = 5
        tensors = [torch.rand(32, 16, device=device) for _ in range(N)]
        tensors_c = [t.clone() for t in tensors]
        others = [torch.rand(32, 16, device=device) for _ in range(N)]

        libtorch_agnostic.ops.my__foreach_mul_(tensors, others)
        expected_values = torch._foreach_mul(tensors_c, others)

        for tensor_t, expected_t in zip(tensors, expected_values):
            self.assertEqual(tensor_t, expected_t)

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfTorchVersionLessThan(2, 10)
    def test_my__foreach_mul(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        N = 5
        tensors = [torch.rand(32, 16, device=device) for _ in range(N)]
        others = [torch.rand(32, 16, device=device) for _ in range(N)]

        result = libtorch_agnostic.ops.my__foreach_mul(tensors, others)
        expected = torch._foreach_mul(tensors, others)

        for result_t, expected_t in zip(result, expected):
            self.assertEqual(result_t, expected_t)

        def _make_cuda_tensors(prior_mem):
            cuda_res = libtorch_agnostic.ops.my__foreach_mul(tensors, others)
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

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfTorchVersionLessThan(2, 10)
    def test_make_tensor_clones_and_call_foreach(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        t1 = torch.rand(2, 5, device=device)
        t2 = torch.rand(3, 4, device=device)
        result = libtorch_agnostic.ops.make_tensor_clones_and_call_foreach(t1, t2)
        self.assertEqual(result[0], t1 * t1)
        self.assertEqual(result[1], t2 * t2)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyCUDA
    def test_device(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        cuda_device = libtorch_agnostic.ops.test_device_constructor(
            is_cuda=True, index=1, use_str=False
        )
        self.assertEqual(cuda_device, torch.device("cuda:1"))
        cuda_device = libtorch_agnostic.ops.test_device_constructor(
            is_cuda=True, index=1, use_str=True
        )
        self.assertEqual(cuda_device, torch.device("cuda:1"))

        self.assertEqual(libtorch_agnostic.ops.test_device_index(cuda_device), 1)
        self.assertTrue(
            libtorch_agnostic.ops.test_device_equality(
                cuda_device, torch.device("cuda:1")
            )
        )
        self.assertFalse(
            libtorch_agnostic.ops.test_device_equality(
                cuda_device, torch.device("cuda:0")
            )
        )
        self.assertFalse(libtorch_agnostic.ops.test_device_is_cpu(cuda_device))
        self.assertTrue(libtorch_agnostic.ops.test_device_is_cuda(cuda_device))

        cuda_0_device = libtorch_agnostic.ops.test_device_set_index(cuda_device, 0)
        self.assertEqual(cuda_0_device, torch.device("cuda:0"))

        cpu_device = libtorch_agnostic.ops.test_device_constructor(False, 0, False)
        self.assertEqual(cpu_device, torch.device("cpu"))
        self.assertTrue(
            libtorch_agnostic.ops.test_device_equality(cpu_device, torch.device("cpu"))
        )
        self.assertTrue(libtorch_agnostic.ops.test_device_is_cpu(cpu_device))
        self.assertFalse(libtorch_agnostic.ops.test_device_is_cuda(cpu_device))
        self.assertFalse(
            libtorch_agnostic.ops.test_device_equality(cpu_device, cuda_device)
        )

        with self.assertRaisesRegex(
            RuntimeError, "Device index 129 is out of range for int8_t"
        ):
            libtorch_agnostic.ops.test_device_constructor(
                is_cuda=True, index=129, use_str=False
            )

        with self.assertRaisesRegex(
            RuntimeError, "Device index 129 is out of range for int8_t"
        ):
            libtorch_agnostic.ops.test_device_set_index(cuda_device, 129)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyCUDA
    @deviceCountAtLeast(2)
    def test_tensor_device(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        t = torch.randn(2, 3)
        self.assertEqual(libtorch_agnostic.ops.test_tensor_device(t), t.device)

        t_cuda = torch.randn(2, 3, device="cuda")
        self.assertEqual(
            libtorch_agnostic.ops.test_tensor_device(t_cuda), t_cuda.device
        )

        t_cuda_1 = torch.randn(2, 3, device="cuda:1")
        self.assertEqual(
            libtorch_agnostic.ops.test_tensor_device(t_cuda_1), t_cuda_1.device
        )

    @skipIfTorchVersionLessThan(2, 10)
    @onlyCPU
    # TODO: Debug this:
    # Dynamo failed to run FX node with fake tensors:
    # call_function libtorch_agnostic.test_parallel_for.default(*(100, 10), **{}):
    # got RuntimeError('libtorch_agnostic::test_parallel_for() expected at most
    # 2 argument(s) but received 3 argument(s).
    # Declaration: libtorch_agnostic::test_parallel_for(int size, int grain_size) -> Tensor')
    @xfailIfTorchDynamo
    def test_parallel_for(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        num_threads = torch.get_num_threads()
        size = 100
        grain_size = 10
        expected_num_threads_used = min(
            (size + grain_size - 1) // grain_size, num_threads
        )

        result = libtorch_agnostic.ops.test_parallel_for(size, grain_size)
        result_thread_ids = torch.unique(torch.bitwise_right_shift(result, 32))
        result_values = torch.bitwise_and(result, 0xFFFFFFFF)
        expected = torch.arange(size, dtype=torch.int64)

        self.assertEqual(result_values, expected)
        self.assertEqual(result_thread_ids, torch.arange(expected_num_threads_used))

    @skipIfTorchVersionLessThan(2, 10)
    @onlyCPU
    def test_get_num_threads(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        num_threads = libtorch_agnostic.ops.test_get_num_threads()
        expected_num_threads = torch.get_num_threads()
        self.assertEqual(num_threads, expected_num_threads)

    @skipIfTorchVersionLessThan(2, 10)
    @parametrize("layout", [None, torch.strided, torch.sparse_coo])
    @parametrize("memory_format", [None, torch.channels_last, torch.contiguous_format])
    def test_my_empty(self, device, layout, memory_format):
        import libtorch_agn_2_10 as libtorch_agnostic

        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            # set use_deterministic_algorithms to fill uninitialized memory
            torch.use_deterministic_algorithms(True)

            # Use 4D size for channels_last, 2D otherwise
            size = [2, 3, 4, 5] if memory_format == torch.channels_last else [2, 3]

            # sparse_coo layout doesn't support memory_format parameter
            if layout == torch.sparse_coo and memory_format is not None:
                return

            # Test default parameters
            result = libtorch_agnostic.ops.my_empty(
                size, None, layout, None, None, memory_format
            )
            expected = torch.empty(size, layout=layout, memory_format=memory_format)
            self.assertEqual(result, expected, exact_device=True, exact_layout=True)

            # Test with dtype
            result_float = libtorch_agnostic.ops.my_empty(
                size, torch.float32, layout, None, None, memory_format
            )
            expected_float = torch.empty(
                size,
                dtype=torch.float32,
                layout=layout,
                memory_format=memory_format,
            )
            self.assertEqual(
                result_float, expected_float, exact_device=True, exact_layout=True
            )

            # Test with dtype and device
            result_with_device = libtorch_agnostic.ops.my_empty(
                size, torch.float64, layout, device, None, memory_format
            )
            expected_with_device = torch.empty(
                size,
                dtype=torch.float64,
                layout=layout,
                device=device,
                memory_format=memory_format,
            )
            self.assertEqual(
                result_with_device,
                expected_with_device,
                exact_device=True,
                exact_layout=True,
            )

            # Verify layout if specified
            if layout is not None:
                self.assertEqual(result_with_device.layout, layout)

            # Verify memory format if specified
            if memory_format == torch.channels_last:
                self.assertTrue(
                    result_with_device.is_contiguous(memory_format=torch.channels_last)
                )
            elif memory_format == torch.contiguous_format:
                self.assertTrue(result_with_device.is_contiguous())

            # Test pin_memory on CUDA (only once, not for every parameter combination)
            if device == "cuda" and layout is None and memory_format is None:
                result_pinned = libtorch_agnostic.ops.my_empty(
                    [2, 3], torch.float32, None, "cpu", True, None
                )
                expected_pinned = torch.empty(
                    [2, 3], dtype=torch.float32, device="cpu", pin_memory=True
                )
                self.assertEqual(
                    result_pinned,
                    expected_pinned,
                    exact_device=True,
                    exact_layout=True,
                )
                self.assertTrue(result_pinned.is_pinned())
        finally:
            torch.use_deterministic_algorithms(deterministic)

    def test_my_flatten(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.randn(2, 3, 4, device=device)
        result = libtorch_agnostic.ops.my_flatten(t)
        expected = torch.flatten(t)
        self.assertEqual(result, expected)

        result_start = libtorch_agnostic.ops.my_flatten(t, 1)
        expected_start = torch.flatten(t, 1)
        self.assertEqual(result_start, expected_start)

        result_range = libtorch_agnostic.ops.my_flatten(t, 2, -1)
        expected_range = torch.flatten(t, 2, -1)
        self.assertEqual(result_range, expected_range)

    @onlyCPU
    @xfailIfTorchDynamo
    def test_my_optional_tensor_ref(self, device):
        """Test TORCH_BOX with const std::optional<Tensor>& parameter."""
        import libtorch_agn_2_9 as libtorch_agnostic

        # Test with a tensor provided
        t = torch.randn(5, device=device)
        result = libtorch_agnostic.ops.my_optional_tensor_ref(t, 10)
        self.assertEqual(result, t)

        # Test with None (should return zeros tensor of specified size)
        result_none = libtorch_agnostic.ops.my_optional_tensor_ref(None, 7)
        expected_zeros = torch.zeros(7)
        self.assertEqual(result_none, expected_zeros)
        self.assertEqual(result_none.shape, (7,))

    @skipIfTorchDynamo("no data pointer defined for FakeTensor, FunctionalTensor")
    def test_my_storage_offset(self, device):
        """Test storage_offset method on Tensor."""
        import libtorch_agn_2_9 as libtorch_agnostic

        # Test with a regular tensor (storage_offset should be 0)
        t = torch.randn(3, 4, device=device)
        result = libtorch_agnostic.ops.my_storage_offset(t)
        self.assertEqual(result, t.storage_offset())
        self.assertEqual(result, 0)

        # Test with a sliced tensor (storage_offset should be non-zero)
        t_sliced = t[1:]
        result_sliced = libtorch_agnostic.ops.my_storage_offset(t_sliced)
        self.assertEqual(result_sliced, t_sliced.storage_offset())
        self.assertEqual(result_sliced, 4)  # 1 row * 4 columns

        # Test with a view with offset
        t_view = t.view(-1)[2:]
        result_view = libtorch_agnostic.ops.my_storage_offset(t_view)
        self.assertEqual(result_view, t_view.storage_offset())
        self.assertEqual(result_view, 2)

    @dtypes(*all_types_and(torch.float16, torch.bool))
    def test_my_element_size(self, device, dtype):
        """Test element_size method on Tensor."""
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.zeros(2, 3, device=device, dtype=dtype)
        result = libtorch_agnostic.ops.my_element_size(t)
        self.assertEqual(result, t.element_size())

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_reshape(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        t = torch.randn(2, 3, 4, device=device)

        result = libtorch_agnostic.ops.my_reshape(t, [6, 4])
        expected = torch.reshape(t, [6, 4])
        self.assertEqual(result, expected)

        result_infer = libtorch_agnostic.ops.my_reshape(t, [-1, 4])
        expected_infer = torch.reshape(t, [-1, 4])
        self.assertEqual(result_infer, expected_infer)

        result_flat = libtorch_agnostic.ops.my_reshape(t, [-1])
        expected_flat = torch.reshape(t, [-1])
        self.assertEqual(result_flat, expected_flat)

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_view(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        t = torch.randn(2, 3, 4, device=device)

        result = libtorch_agnostic.ops.my_view(t, [6, 4])
        expected = t.view([6, 4])
        self.assertEqual(result, expected)

        result_infer = libtorch_agnostic.ops.my_view(t, [-1, 4])
        expected_infer = t.view([-1, 4])
        self.assertEqual(result_infer, expected_infer)

        result_flat = libtorch_agnostic.ops.my_view(t, [-1])
        expected_flat = t.view([-1])
        self.assertEqual(result_flat, expected_flat)

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_shape(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        expected = (3, 5)
        t = torch.rand(*expected, device=device)
        shape = libtorch_agnostic.ops.my_shape(t)
        self.assertEqual(shape, expected)

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_sum(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        t = torch.randn(3, 4, 5, device=device)

        result = libtorch_agnostic.ops.my_sum(t, [0])
        expected = torch.sum(t, [0])
        self.assertEqual(result, expected)

        result_multi = libtorch_agnostic.ops.my_sum(t, [0, 2])
        expected_multi = torch.sum(t, [0, 2])
        self.assertEqual(result_multi, expected_multi)

        result_keepdim = libtorch_agnostic.ops.my_sum(t, [1], True)
        expected_keepdim = torch.sum(t, [1], keepdim=True)
        self.assertEqual(result_keepdim, expected_keepdim)

        result_dtype = libtorch_agnostic.ops.my_sum(t, [0], False, torch.float64)
        expected_dtype = torch.sum(t, [0], dtype=torch.float64)
        self.assertEqual(result_dtype, expected_dtype)

        # Test sum without dim (sum all elements)
        result_all = libtorch_agnostic.ops.my_sum(t)
        expected_all = torch.sum(t)
        self.assertEqual(result_all, expected_all)

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_sum_out(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        t = torch.randn(3, 4, 5, device=device)

        out = torch.empty(4, 5, device=device)
        result = libtorch_agnostic.ops.my_sum_out(out, t, [0])
        expected = torch.sum(t, [0])
        self.assertEqual(out, expected)
        self.assertEqual(id(result), id(out))

        out_keepdim = torch.empty(3, 1, 5, device=device)
        libtorch_agnostic.ops.my_sum_out(out_keepdim, t, [1], True)
        expected_keepdim = torch.sum(t, [1], keepdim=True)
        self.assertEqual(out_keepdim, expected_keepdim)

        out_dtype = torch.empty(4, 5, dtype=torch.float64, device=device)
        libtorch_agnostic.ops.my_sum_out(out_dtype, t, [0], False, torch.float64)
        expected_dtype = torch.sum(t, [0], dtype=torch.float64)
        self.assertEqual(out_dtype, expected_dtype)

        out_all = torch.empty([], device=device)
        libtorch_agnostic.ops.my_sum_out(out_all, t)
        expected_all = torch.sum(t)
        self.assertEqual(out_all, expected_all)

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_sum_all(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        t = torch.randn(3, 4, 5, device=device)

        # Test my_sum_all (sums all elements, returns scalar)
        result = libtorch_agnostic.ops.my_sum_all(t)
        expected = torch.sum(t)
        self.assertEqual(result, expected)
        self.assertEqual(result.shape, torch.Size([]))

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_sum_dim1(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        t = torch.randn(3, 4, 5, device=device)

        # Test my_sum_dim1 (sums along dimension 1)
        result = libtorch_agnostic.ops.my_sum_dim1(t)
        expected = torch.sum(t, dim=1)
        self.assertEqual(result, expected)
        self.assertEqual(result.shape, torch.Size([3, 5]))

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_full(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        # Test basic full with default parameters
        result = libtorch_agnostic.ops.my_full([2, 3], 3.14)
        expected = torch.full([2, 3], 3.14)
        self.assertEqual(result, expected)

        # Test with dtype
        result_dtype = libtorch_agnostic.ops.my_full([3, 4], 42.0, dtype=torch.int64)
        expected_dtype = torch.full([3, 4], 42, dtype=torch.int64)
        self.assertEqual(result_dtype, expected_dtype)

        # Test with device
        result_device = libtorch_agnostic.ops.my_full([2, 2], 1.5, device=device)
        expected_device = torch.full([2, 2], 1.5, device=device)
        self.assertEqual(result_device, expected_device, exact_device=True)

        # Test with dtype and device
        result_both = libtorch_agnostic.ops.my_full(
            [4, 5], 2.5, dtype=torch.float64, device=device
        )
        expected_both = torch.full([4, 5], 2.5, dtype=torch.float64, device=device)
        self.assertEqual(result_both, expected_both, exact_device=True)

    def test_mv_tensor_accessor(self, device):
        import libtorch_agn_2_9 as libtorch_agnostic

        m = torch.rand(3, 5, device=device)
        v = torch.rand(5, device=device)
        result = libtorch_agnostic.ops.mv_tensor_accessor(m, v)
        expected = torch.mv(m, v)
        self.assertEqual(result, expected)

        # non-contiguous inputs
        m = torch.rand(3 * 2, 5 * 3, device=device)[::2, ::3]
        v = torch.rand(5 * 4, device=device)[::4]
        result = libtorch_agnostic.ops.mv_tensor_accessor(m, v)
        expected = torch.mv(m, v)
        self.assertEqual(result, expected)

    @skipIfTorchVersionLessThan(2, 10)
    @skipIfTorchDynamo("no data pointer defined for FakeTensor, FunctionalTensor")
    def test_get_any_data_ptr(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        t = torch.empty(2, 5, device=device, dtype=torch.float32)
        expected_p = t.data_ptr()

        for mutable in [True, False]:
            p = libtorch_agnostic.ops.get_any_data_ptr(t, mutable)
            self.assertEqual(p, expected_p)

    @skipIfTorchVersionLessThan(2, 10)
    @skipIfTorchDynamo("no data pointer defined for FakeTensor, FunctionalTensor")
    def test_get_template_any_data_ptr(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        supported_dtypes = get_supported_dtypes()

        for dtype in supported_dtypes:
            t = torch.empty(2, 5, device=device, dtype=dtype)
            expected_p = t.data_ptr()

            for rdtype in supported_dtypes:
                if dtype == rdtype:
                    for mutable in [True, False]:
                        p = libtorch_agnostic.ops.get_template_any_data_ptr(
                            t, rdtype, mutable
                        )
                        self.assertEqual(p, expected_p)
                else:
                    for mutable in [True, False]:
                        with self.assertRaisesRegex(
                            RuntimeError, "expected scalar type.* but found"
                        ):
                            libtorch_agnostic.ops.get_template_any_data_ptr(
                                t, rdtype, mutable
                            )

    @skipIfTorchVersionLessThan(2, 10)
    @onlyCUDA
    def test_my_get_curr_cuda_blas_handle(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        res = libtorch_agnostic.ops.my_get_curr_cuda_blas_handle()
        expected = torch.cuda.current_blas_handle()
        self.assertEqual(res, expected)

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfTorchVersionLessThan(2, 10)
    def test_my_string_op(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        t = torch.empty(3, 4, 5, device=device)

        dim_vec, result_dim = libtorch_agnostic.ops.my_string_op(t, "dim", "ice")
        self.assertEqual(dim_vec, ["dim", str(t.dim()), "ice"])
        self.assertEqual(result_dim, t.dim())

        size_vec, result_size = libtorch_agnostic.ops.my_string_op(t, "size", "cream")
        self.assertEqual(size_vec, ["size", str(t.size(0)), "cream"])
        self.assertEqual(result_size, t.size(0))

        stride_vec, result_stride = libtorch_agnostic.ops.my_string_op(
            t, "stride", "cake"
        )
        self.assertEqual(stride_vec, ["stride", str(t.stride(0)), "cake"])
        self.assertEqual(result_stride, t.stride(0))

        with self.assertRaisesRegex(RuntimeError, "Unsupported accessor value: "):
            libtorch_agnostic.ops.my_string_op(t, "invalid", "")

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfTorchVersionLessThan(2, 10)
    def test_my__foreach_mul_vec(self, device):
        """Test my__foreach_mul_vec which uses const std::vector<Tensor>& parameters."""
        import libtorch_agn_2_10 as libtorch_agnostic

        N = 5
        tensors = [torch.rand(32, 16, device=device) for _ in range(N)]
        others = [torch.rand(32, 16, device=device) for _ in range(N)]

        result = libtorch_agnostic.ops.my__foreach_mul_vec(tensors, others)
        expected = torch._foreach_mul(tensors, others)

        for result_t, expected_t in zip(result, expected):
            self.assertEqual(result_t, expected_t)

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfTorchVersionLessThan(2, 10)
    def test_my_string_op_const_string_ref(self, device):
        """Test my_string_op_const_string_ref which uses const std::string& parameters."""
        import libtorch_agn_2_10 as libtorch_agnostic

        t = torch.empty(3, 4, 5, device=device)

        dim_vec, result_dim = libtorch_agnostic.ops.my_string_op_const_string_ref(
            t, "dim", "test1"
        )
        self.assertEqual(dim_vec, ["dim", str(t.dim()), "test1"])
        self.assertEqual(result_dim, t.dim())

        size_vec, result_size = libtorch_agnostic.ops.my_string_op_const_string_ref(
            t, "size", "test2"
        )
        self.assertEqual(size_vec, ["size", str(t.size(0)), "test2"])
        self.assertEqual(result_size, t.size(0))

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfTorchVersionLessThan(2, 10)
    def test_my_string_op_const_string_view_ref(self, device):
        """Test my_string_op_const_string_view_ref which uses const std::string_view& parameters."""
        import libtorch_agn_2_10 as libtorch_agnostic

        t = torch.empty(3, 4, 5, device=device)

        dim_vec, result_dim = libtorch_agnostic.ops.my_string_op_const_string_view_ref(
            t, "dim", "view1"
        )
        self.assertEqual(dim_vec, ["dim", str(t.dim()), "view1"])
        self.assertEqual(result_dim, t.dim())

        stride_vec, result_stride = (
            libtorch_agnostic.ops.my_string_op_const_string_view_ref(
                t, "stride", "view2"
            )
        )
        self.assertEqual(stride_vec, ["stride", str(t.stride(0)), "view2"])
        self.assertEqual(result_stride, t.stride(0))

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfTorchVersionLessThan(2, 10)
    def test_my_string_op_string_ref(self, device):
        """Test my_string_op_string_ref which uses std::string& (non-const) parameters."""
        import libtorch_agn_2_10 as libtorch_agnostic

        t = torch.empty(3, 4, 5, device=device)

        dim_vec, result_dim = libtorch_agnostic.ops.my_string_op_string_ref(
            t, "dim", "ref1"
        )
        self.assertEqual(dim_vec, ["dim", str(t.dim()), "ref1"])
        self.assertEqual(result_dim, t.dim())

        size_vec, result_size = libtorch_agnostic.ops.my_string_op_string_ref(
            t, "size", "ref2"
        )
        self.assertEqual(size_vec, ["size", str(t.size(0)), "ref2"])
        self.assertEqual(result_size, t.size(0))

    @skipIfTorchVersionLessThan(2, 10)
    @onlyCPU
    def test_my_set_requires_grad(self, device):
        """Test set_requires_grad method on Tensor."""
        import libtorch_agn_2_10 as libtorch_agnostic

        # Use torch.no_grad() to prevent autograd from wrapping the output
        # tensor with a grad_fn. When a tensor with requires_grad=True goes
        # through a custom op, PyTorch wraps the output with a grad_fn
        # (e.g., WarnNotImplemented), making requires_grad computed based on
        # inputs rather than directly settable.
        t = torch.randn(3, 4, device=device)
        self.assertFalse(t.requires_grad)

        with torch.no_grad():
            libtorch_agnostic.ops.my_set_requires_grad(t, True)
        self.assertTrue(t.requires_grad)

        with torch.no_grad():
            libtorch_agnostic.ops.my_set_requires_grad(t, False)
        self.assertFalse(t.requires_grad)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyCUDA
    def test_my_get_current_cuda_stream(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        device_index = torch.device(device).index
        res = libtorch_agnostic.ops.my_get_current_cuda_stream(device_index)
        expected = torch.cuda.current_stream(device_index).cuda_stream
        self.assertEqual(res, expected)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyCUDA
    def test_my_set_current_cuda_stream(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        device_index = torch.device(device).index
        prev_stream = torch.cuda.current_stream(device_index).cuda_stream
        new_stream = torch.cuda.streams.Stream(device_index).cuda_stream

        try:
            libtorch_agnostic.ops.my_set_current_cuda_stream(new_stream, device_index)
            expected = torch.cuda.current_stream(device_index).cuda_stream
            self.assertEqual(new_stream, expected)
        finally:
            libtorch_agnostic.ops.my_set_current_cuda_stream(prev_stream, device_index)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyCUDA
    def test_my_get_cuda_stream_from_pool(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        device_index = torch.device(device).index
        prev_stream = torch.cuda.current_stream(device_index).cuda_stream

        try:
            for high_priority in [False, True]:
                stream = libtorch_agnostic.ops.my_get_cuda_stream_from_pool(
                    high_priority, device_index
                )
                libtorch_agnostic.ops.my_set_current_cuda_stream(stream, device_index)
                expected = torch.cuda.current_stream(device_index).cuda_stream
                self.assertEqual(stream, expected)
        finally:
            libtorch_agnostic.ops.my_set_current_cuda_stream(prev_stream, device_index)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyCUDA
    def test_my_cuda_stream_synchronize(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        device_index = torch.device(device).index
        stream = torch.cuda.current_stream(device_index).cuda_stream
        # sanity check for torch_cuda_stream_synchronize:
        libtorch_agnostic.ops.my_cuda_stream_synchronize(stream, device_index)

    @skipIfTorchVersionLessThan(2, 10)
    @skipIfTorchDynamo("no data pointer defined for FakeTensor, FunctionalTensor")
    def test_my_from_blob(self, device):
        import libtorch_agn_2_10 as libtorch_agnostic

        # Create reference implementation using unstable torch::from_blob via load_inline
        source = """
        #include <torch/extension.h>

        at::Tensor reference_from_blob(at::Tensor t) {
            void* data_ptr = t.storage().data_ptr().get();
            auto options = torch::TensorOptions()
                .dtype(t.dtype())
                .device(t.device());

            return torch::from_blob(
                data_ptr,
                t.sizes(),
                t.strides(),
                options);
        }
        """

        module = torch.utils.cpp_extension.load_inline(
            name="test_from_blob_reference",
            cpp_sources=[source],
            functions=["reference_from_blob"],
        )

        # Test basic from_blob with contiguous tensor
        original = torch.rand(2, 3, device=device, dtype=torch.float32)
        stable_result = libtorch_agnostic.ops.my_from_blob(
            original.data_ptr(),
            original.size(),
            original.stride(),
            device,
            torch.float32,
        )
        reference_result = module.reference_from_blob(original)
        self.assertEqual(stable_result, reference_result)
        self.assertEqual(stable_result.data_ptr(), original.data_ptr())

        # Test with non-contiguous strides
        transposed = torch.rand(4, 6, device=device, dtype=torch.float32).t()

        stable_transposed = libtorch_agnostic.ops.my_from_blob(
            transposed.data_ptr(),
            transposed.size(),
            transposed.stride(),
            device,
            transposed.dtype,
        )

        reference_transposed = module.reference_from_blob(transposed)
        self.assertEqual(stable_transposed, reference_transposed)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyCUDA
    def test_std_cuda_check_success(self, device):
        """Test that STD_CUDA_CHECK works correctly for successful CUDA calls."""
        import libtorch_agn_2_10 as libtorch_agnostic

        result = libtorch_agnostic.ops.test_std_cuda_check_success()
        expected_device = torch.cuda.current_device()
        self.assertEqual(result, expected_device)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyCUDA
    @skipIfRocm(msg="TODO: @mikaylagawarecki fix after branch cut")
    @parametrize("show_cpp_stacktraces", [False, True])
    def test_std_cuda_check_error(self, device, show_cpp_stacktraces):
        """Test that STD_CUDA_CHECK throws std::runtime_error with CUDA error message.

        When TORCH_SHOW_CPP_STACKTRACES=1, the error should include a C++ stack trace.
        Since this env var is cached on first use, we use subprocess to test both cases.
        """
        import os
        import subprocess
        import sys

        test_script = """
import torch
import libtorch_agn_2_10 as libtorch_agnostic

try:
    libtorch_agnostic.ops.test_std_cuda_check_error()
except RuntimeError as e:
    print(str(e))
"""
        env = os.environ.copy()
        env["TORCH_SHOW_CPP_STACKTRACES"] = "1" if show_cpp_stacktraces else "0"
        # Pass the current sys.path to subprocess so it can find the locally installed extension
        env["PYTHONPATH"] = os.pathsep.join(sys.path)

        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            env=env,
        )

        error_message = result.stdout + result.stderr

        self.assertTrue(
            "CUDA error: invalid device ordinal" in error_message
            or "HIP error: invalid device ordinal" in error_message,
            f"Expected 'CUDA/HIP error: invalid device ordinal' in error message, got: {error_message}",
        )
        self.assertIn(
            "GPU device may be out of range, do you have enough GPUs?",
            error_message,
        )

        if show_cpp_stacktraces:
            self.assertIn("C++ CapturedTraceback:", error_message)
            self.assertRegex(
                error_message,
                r"Exception raised from test_std_.*_check_error at .*test_std_.*check\..*:\d+",
            )
        else:
            self.assertNotIn("C++ CapturedTraceback:", error_message)

    @skipIfTorchVersionLessThan(2, 10)
    @skipIfTorchDynamo(" Dynamo failed to run FX node with fake tensors")
    def test_my_to_device(self, device):
        """Test to(device) convenience overload."""
        import libtorch_agn_2_10 as libtorch_agnostic

        t = torch.randn(3, 4, device="cpu")

        # Move to current device
        result = libtorch_agnostic.ops.my_to_device(t, device)
        expected = t.to(device)
        self.assertEqual(result, expected, exact_device=True)

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_to_dtype(self, device):
        """Test to(dtype) via the main to function."""
        import libtorch_agn_2_10 as libtorch_agnostic

        t = torch.randn(3, 4, device=device, dtype=torch.float32)

        # Convert to float64
        result = libtorch_agnostic.ops.my_to_dtype(t, torch.float64)
        expected = t.to(torch.float64)
        self.assertEqual(result, expected, exact_device=True)

        # Convert to int32
        t2 = torch.randn(2, 3, device=device, dtype=torch.float32)
        result2 = libtorch_agnostic.ops.my_to_dtype(t2, torch.int32)
        expected2 = t2.to(torch.int32)
        self.assertEqual(result2, expected2, exact_device=True)

    @skipIfTorchVersionLessThan(2, 10)
    @skipIfTorchDynamo(" Dynamo failed to run FX node with fake tensors")
    def test_my_to_dtype_layout(self, device):
        """Test the full to.dtype_layout op with various parameter combinations."""
        import libtorch_agn_2_10 as libtorch_agnostic

        # Test dtype conversion
        t = torch.randn(3, 4, device=device, dtype=torch.float32)
        result = libtorch_agnostic.ops.my_to_dtype_layout(t, dtype=torch.float64)
        expected = t.to(dtype=torch.float64)
        self.assertEqual(result, expected, exact_device=True)

        # Test device conversion (move to CPU if on CUDA, or stay on CPU)
        result_cpu = libtorch_agnostic.ops.my_to_dtype_layout(t, device="cpu")
        expected_cpu = t.to(device="cpu")
        self.assertEqual(result_cpu, expected_cpu, exact_device=True)

        # Test copy=True (should always create a copy)
        t_copy = torch.randn(2, 3, device=device)
        result_copy = libtorch_agnostic.ops.my_to_dtype_layout(t_copy, copy=True)
        expected_copy = t_copy.to(copy=True)
        self.assertEqual(result_copy, expected_copy, exact_device=True)
        self.assertNotEqual(result_copy.data_ptr(), t_copy.data_ptr())

        # Test dtype + device together
        t3 = torch.randn(2, 2, device=device, dtype=torch.float32)
        result_both = libtorch_agnostic.ops.my_to_dtype_layout(
            t3, dtype=torch.float64, device="cpu"
        )
        expected_both = t3.to(dtype=torch.float64, device="cpu")
        self.assertEqual(result_both, expected_both, exact_device=True)

        # Test memory_format (channels_last for 4D tensor)
        t4d = torch.randn(2, 3, 4, 5, device=device)
        result_channels_last = libtorch_agnostic.ops.my_to_dtype_layout(
            t4d, memory_format=torch.channels_last
        )
        expected_channels_last = t4d.to(memory_format=torch.channels_last)
        self.assertEqual(
            result_channels_last, expected_channels_last, exact_device=True
        )
        self.assertTrue(
            result_channels_last.is_contiguous(memory_format=torch.channels_last)
        )

        # Test with all None (should return equivalent tensor)
        t_none = torch.randn(2, 3, device=device)
        result_none = libtorch_agnostic.ops.my_to_dtype_layout(t_none)
        expected_none = t_none.to()
        self.assertEqual(result_none, expected_none, exact_device=True)

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_contiguous(self, device):
        """Test contiguous with default memory format."""
        import libtorch_agn_2_10 as libtorch_agnostic

        t = torch.randn(3, 4, device=device).t()
        self.assertFalse(t.is_contiguous())

        result = libtorch_agnostic.ops.my_contiguous(t)
        self.assertTrue(result.is_contiguous())

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_contiguous_memory_format(self, device):
        """Test contiguous with specified memory format."""
        import libtorch_agn_2_10 as libtorch_agnostic

        # Create a 4D tensor (N, C, H, W)
        t = torch.randn(2, 3, 4, 5, device=device)

        # Convert to channels_last format
        result = libtorch_agnostic.ops.my_contiguous_memory_format(
            t, torch.channels_last
        )
        self.assertTrue(result.is_contiguous(memory_format=torch.channels_last))

    @skipIfTorchVersionLessThan(2, 10)
    @onlyCUDA
    def test_std_cuda_kernel_launch_check_success(self, device):
        """Test that STD_CUDA_KERNEL_LAUNCH_CHECK works correctly for successful kernel launches."""
        import libtorch_agn_2_10 as libtorch_agnostic

        libtorch_agnostic.ops.test_std_cuda_kernel_launch_check_success()

    @skipIfTorchVersionLessThan(2, 10)
    @onlyCUDA
    @parametrize("show_cpp_stacktraces", [False, True])
    @skipIfRocm(msg="TODO: @mikaylagawarecki fix after branch cut")
    @unittest.skipIf(
        _get_torch_cuda_version() >= (13, 0), "To be resolved after branch cut"
    )
    def test_std_cuda_kernel_launch_check_error(self, device, show_cpp_stacktraces):
        """Test that STD_CUDA_KERNEL_LAUNCH_CHECK throws std::runtime_error for invalid kernel launches.

        When TORCH_SHOW_CPP_STACKTRACES=1, the error should include a C++ stack trace.
        Since this env var is cached on first use, we use subprocess to test both cases.
        """
        import os
        import subprocess
        import sys

        test_script = """
import torch
import libtorch_agn_2_10 as libtorch_agnostic

try:
    libtorch_agnostic.ops.test_std_cuda_kernel_launch_check_error()
except RuntimeError as e:
    print(str(e))
"""
        env = os.environ.copy()
        env["TORCH_SHOW_CPP_STACKTRACES"] = "1" if show_cpp_stacktraces else "0"
        # Pass the current sys.path to subprocess so it can find the locally installed extension
        env["PYTHONPATH"] = os.pathsep.join(sys.path)

        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            env=env,
        )

        error_message = result.stdout + result.stderr

        self.assertTrue(
            "CUDA error: invalid configuration argument" in error_message
            or "HIP error: invalid configuration argument" in error_message,
            f"Expected 'CUDA|HIP error: invalid configuration argument' in error message, got: {error_message}",
        )

        if show_cpp_stacktraces:
            self.assertIn("C++ CapturedTraceback:", error_message)
            self.assertRegex(
                error_message,
                r"Exception raised from test_std_.*_kernel_launch_check_error at .*test_std_.*_check\..*:\d+",
            )
        else:
            self.assertNotIn("C++ CapturedTraceback:", error_message)

    @skipIfTorchVersionLessThan(2, 10)
    @skipIfTorchDynamo(
        "AssertionError(tensor's device must be `meta`, got cpu instead)"
    )
    def test_my_new_empty(self, device):
        """Test new_empty with all kwargs."""
        import libtorch_agn_2_10 as libtorch_agnostic

        t = torch.randn(3, 4, device=device, dtype=torch.float32)

        # Test with default args (should inherit from self)
        result = libtorch_agnostic.ops.my_new_empty(t, [2, 3])
        expected = t.new_empty([2, 3])
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.device, expected.device)

        # Test with different dtype
        result_dtype = libtorch_agnostic.ops.my_new_empty(
            t, [2, 3], dtype=torch.float64
        )
        expected_dtype = t.new_empty([2, 3], dtype=torch.float64)
        self.assertEqual(result_dtype.shape, expected_dtype.shape)
        self.assertEqual(result_dtype.dtype, torch.float64)

        # Test with different device (move to CPU)
        result_device = libtorch_agnostic.ops.my_new_empty(t, [2, 3], device="cpu")
        expected_device = t.new_empty([2, 3], device="cpu")
        self.assertEqual(result_device.shape, expected_device.shape)
        self.assertEqual(result_device.device.type, "cpu")

        # Test with dtype and device together
        result_both = libtorch_agnostic.ops.my_new_empty(
            t, [4, 5], dtype=torch.int64, device="cpu"
        )
        expected_both = t.new_empty([4, 5], dtype=torch.int64, device="cpu")
        self.assertEqual(result_both.shape, expected_both.shape)
        self.assertEqual(result_both.dtype, torch.int64)
        self.assertEqual(result_both.device.type, "cpu")

    @skipIfTorchVersionLessThan(2, 10)
    @skipIfTorchDynamo(
        "AssertionError(tensor's device must be `meta`, got cpu instead)"
    )
    def test_my_new_zeros(self, device):
        """Test new_zeros with all kwargs."""
        import libtorch_agn_2_10 as libtorch_agnostic

        t = torch.randn(3, 4, device=device, dtype=torch.float32)

        # Test with default args (should inherit from self)
        result = libtorch_agnostic.ops.my_new_zeros(t, [2, 3])
        expected = t.new_zeros([2, 3])
        self.assertEqual(result, expected, exact_device=True)

        # Test with different dtype
        result_dtype = libtorch_agnostic.ops.my_new_zeros(
            t, [2, 3], dtype=torch.float64
        )
        expected_dtype = t.new_zeros([2, 3], dtype=torch.float64)
        self.assertEqual(result_dtype, expected_dtype, exact_device=True)

        # Test with different device (move to CPU)
        result_device = libtorch_agnostic.ops.my_new_zeros(t, [2, 3], device="cpu")
        expected_device = t.new_zeros([2, 3], device="cpu")
        self.assertEqual(result_device, expected_device, exact_device=True)

        # Test with dtype and device together
        result_both = libtorch_agnostic.ops.my_new_zeros(
            t, [4, 5], dtype=torch.int64, device="cpu"
        )
        expected_both = t.new_zeros([4, 5], dtype=torch.int64, device="cpu")
        self.assertEqual(result_both, expected_both, exact_device=True)

    def test_my_unsqueeze(self, device):
        """Test unsqueeze op."""
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.randn(3, 4, device=device)

        # Test unsqueeze at dim 0
        result = libtorch_agnostic.ops.my_unsqueeze(t, 0)
        expected = torch.unsqueeze(t, 0)
        self.assertEqual(result, expected)
        self.assertEqual(result.shape, torch.Size([1, 3, 4]))

        # Test unsqueeze at dim 1
        result1 = libtorch_agnostic.ops.my_unsqueeze(t, 1)
        expected1 = torch.unsqueeze(t, 1)
        self.assertEqual(result1, expected1)
        self.assertEqual(result1.shape, torch.Size([3, 1, 4]))

        # Test unsqueeze at dim -1
        result_neg = libtorch_agnostic.ops.my_unsqueeze(t, -1)
        expected_neg = torch.unsqueeze(t, -1)
        self.assertEqual(result_neg, expected_neg)
        self.assertEqual(result_neg.shape, torch.Size([3, 4, 1]))

    def test_my_squeeze(self, device):
        """Test squeeze.dim op."""
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.randn(3, 1, 4, device=device)

        # Test squeeze at dim 1 (the dimension of size 1)
        result = libtorch_agnostic.ops.my_squeeze(t, 1)
        expected = torch.squeeze(t, 1)
        self.assertEqual(result, expected)
        self.assertEqual(result.shape, torch.Size([3, 4]))

        # Test squeeze at dim 0 (not size 1, should be no-op)
        result0 = libtorch_agnostic.ops.my_squeeze(t, 0)
        expected0 = torch.squeeze(t, 0)
        self.assertEqual(result0, expected0)
        self.assertEqual(result0.shape, torch.Size([3, 1, 4]))

        # Test squeeze at dim -2 (same as dim 1)
        result_neg = libtorch_agnostic.ops.my_squeeze(t, -2)
        expected_neg = torch.squeeze(t, -2)
        self.assertEqual(result_neg, expected_neg)
        self.assertEqual(result_neg.shape, torch.Size([3, 4]))

    def test_my_select(self, device):
        """Test select.int op."""
        import libtorch_agn_2_9 as libtorch_agnostic

        t = torch.randn(3, 4, 5, device=device)

        # Test select at dim 0, index 1
        result = libtorch_agnostic.ops.my_select(t, 0, 1)
        expected = torch.select(t, 0, 1)
        self.assertEqual(result, expected)
        self.assertEqual(result.shape, torch.Size([4, 5]))

        # Test select at dim 1, index 2
        result1 = libtorch_agnostic.ops.my_select(t, 1, 2)
        expected1 = torch.select(t, 1, 2)
        self.assertEqual(result1, expected1)
        self.assertEqual(result1.shape, torch.Size([3, 5]))

        # Test select at dim -1, index 0
        result_neg = libtorch_agnostic.ops.my_select(t, -1, 0)
        expected_neg = torch.select(t, -1, 0)
        self.assertEqual(result_neg, expected_neg)
        self.assertEqual(result_neg.shape, torch.Size([3, 4]))

    def test_my_matmul(self, device):
        """Test matmul op."""
        import libtorch_agn_2_9 as libtorch_agnostic

        # Test 2D x 2D matrix multiplication
        a = torch.randn(3, 4, device=device)
        b = torch.randn(4, 5, device=device)
        result = libtorch_agnostic.ops.my_matmul(a, b)
        expected = torch.matmul(a, b)
        self.assertEqual(result, expected)
        self.assertEqual(result.shape, torch.Size([3, 5]))

        # Test 1D x 2D (vector-matrix)
        v = torch.randn(4, device=device)
        m = torch.randn(4, 5, device=device)
        result_vm = libtorch_agnostic.ops.my_matmul(v, m)
        expected_vm = torch.matmul(v, m)
        self.assertEqual(result_vm, expected_vm)

        # Test 2D x 1D (matrix-vector)
        m2 = torch.randn(3, 4, device=device)
        v2 = torch.randn(4, device=device)
        result_mv = libtorch_agnostic.ops.my_matmul(m2, v2)
        expected_mv = torch.matmul(m2, v2)
        self.assertEqual(result_mv, expected_mv)

        # Test batched matmul
        batch_a = torch.randn(2, 3, 4, device=device)
        batch_b = torch.randn(2, 4, 5, device=device)
        result_batch = libtorch_agnostic.ops.my_matmul(batch_a, batch_b)
        expected_batch = torch.matmul(batch_a, batch_b)
        self.assertEqual(result_batch, expected_batch)

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_subtract(self, device):
        """Test subtract.Tensor op."""
        import libtorch_agn_2_10 as libtorch_agnostic

        a = torch.randn(3, 4, device=device)
        b = torch.randn(3, 4, device=device)

        # Test basic subtraction (alpha=1.0)
        result = libtorch_agnostic.ops.my_subtract(a, b)
        expected = torch.subtract(a, b)
        self.assertEqual(result, expected)

        # Test subtraction with alpha=2.0
        result_alpha = libtorch_agnostic.ops.my_subtract(a, b, alpha=2.0)
        expected_alpha = torch.subtract(a, b, alpha=2.0)
        self.assertEqual(result_alpha, expected_alpha)

        # Test subtraction with alpha=0.5
        result_half = libtorch_agnostic.ops.my_subtract(a, b, alpha=0.5)
        expected_half = torch.subtract(a, b, alpha=0.5)
        self.assertEqual(result_half, expected_half)

        # Test subtraction with broadcasting
        c = torch.randn(4, device=device)
        result_broadcast = libtorch_agnostic.ops.my_subtract(a, c)
        expected_broadcast = torch.subtract(a, c)
        self.assertEqual(result_broadcast, expected_broadcast)


instantiate_device_type_tests(TestLibtorchAgnostic, globals(), except_for=None)

if __name__ == "__main__":
    run_tests()
