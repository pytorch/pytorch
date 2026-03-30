# Owner(s): ["module: inductor"]
import itertools
import os
import subprocess
import sys
import tempfile
import unittest
from typing import NamedTuple

import torch
from torch._inductor import config
from torch._inductor.codegen.common import TritonScratchWorkspace
from torch._inductor.codegen.cpp_wrapper_gpu import DeferredTritonCallWrapper
from torch._inductor.codegen.cuda.device_op_overrides import CUDADeviceOpOverrides
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import IndentedBuffer
from torch.testing._internal.common_utils import slowTest
from torch.testing._internal.inductor_utils import GPU_TYPE, RUN_GPU


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"

try:
    try:
        from . import (
            test_combo_kernels,
            test_foreach,
            test_pattern_matcher,
            test_select_algorithm,
            test_torchinductor,
            test_torchinductor_dynamic_shapes,
        )
    except ImportError:
        import test_combo_kernels  # @manual=fbcode//caffe2/test/inductor:combo_kernels-library

        import test_foreach  # @manual=fbcode//caffe2/test/inductor:foreach-library
        import test_pattern_matcher  # @manual=fbcode//caffe2/test/inductor:pattern_matcher-library
        import test_select_algorithm  # @manual=fbcode//caffe2/test/inductor:select_algorithm-library
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
        import test_torchinductor_dynamic_shapes  # @manual=fbcode//caffe2/test/inductor:test_inductor-library_dynamic_shapes
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


class GpuWrapperTemplate:
    pass


class TestGpuWrapper(InductorTestCase):
    device = GPU_TYPE

    def test_aoti_debug_printer_works_on_constants(self):
        batch_size = 32
        seq_length = 50
        hidden_size = 768

        def test_fn():
            inp = torch.randn(batch_size, seq_length, hidden_size, device=self.device)
            weight = torch.randn(hidden_size, hidden_size, device=self.device)
            matmul_output = inp @ weight
            torch.nn.LayerNorm(hidden_size, device=self.device)(matmul_output)
            return True

        comp = torch.compile(
            options={
                "cpp_wrapper": True,
                "aot_inductor.debug_intermediate_value_printer": "2",
            }
        )(test_fn)
        comp()

    def test_non_tensor_args_wrapped_on_cpu(self):
        if not RUN_GPU:
            self.skipTest("GPU not available")

        def test_fn(x, s):
            return (x + s).sum()

        compiled = torch.compile(options={"cpp_wrapper": True})(test_fn)
        x = torch.randn(4, device=self.device)
        with torch.utils._device.DeviceContext(self.device):
            _, code = test_torchinductor.run_and_get_cpp_code(compiled, x, 3)
        self.assertIn("torch.tensor(arg, device='cpu')", code)

    def test_cpp_scratch_scales_with_grid_size_for_tma(self):
        if GPU_TYPE != "cuda" or torch.version.hip:
            self.skipTest("CUDA-only codegen test")

        scratch_def, scratch_var = CUDADeviceOpOverrides().cpp_scratch(
            0,
            TritonScratchWorkspace(
                size=256, generate_dtype_str=lambda: "at::ScalarType::Byte"
            ),
            prefix="global_scratch",
        )
        self.assertEqual(scratch_var, "global_scratch_scratch_0")
        self.assertIn(
            "static_cast<int64_t>(256) * grid_0 * grid_1 * grid_2", scratch_def[0]
        )

    def test_triton_wrapper_scales_scratch_with_num_ctas(self):
        if GPU_TYPE != "cuda" or torch.version.hip:
            self.skipTest("CUDA-only codegen test")

        class FakeWrapper:
            device = "cuda"

            def __init__(self):
                self.scratch_spaces = None

            def generate_args_decl(
                self,
                prefix,
                call_args,
                arg_types,
                arg_signatures,
                is_triton_kernel=True,
                scratch_spaces=None,
            ):
                self.scratch_spaces = scratch_spaces

                return ""

        wrapper = FakeWrapper()
        prefix = IndentedBuffer()
        params = {
            "triton_meta": {"signature": {"x": "*fp32"}, "constants": {}},
            "def_args": ["x"],
            "call_args": ["x"],
            "config": {"num_ctas": 8},
            "num_warps": 4,
            "shared_mem": 0,
            "global_scratch": 256,
        }

        DeferredTritonCallWrapper(
            wrapper_name="wrapper",
            kernel_name="kernel",
            kernel_name_to_body={},
            arg_types=[torch.float32],
        ).generate_launch_kernel(prefix, wrapper, "kernel_var", params)

        self.assertEqual(wrapper.scratch_spaces, {"global_scratch": 256 * 8})


# Helper script for test_lazy_compile_kernel_name_collision_across_modules.
# Run as a subprocess so dlopen truly re-runs .so static initializers.
_LAZY_COMPILE_COLLISION_SCRIPT = """\
import torch
from torch._inductor import config

config.cpp_wrapper = True
config.triton.autotune_at_compile_time = False

def fn(x, y, z, w):
    a = x.sin()
    torch._dynamo.graph_break()
    b = (a * y).cos()
    torch._dynamo.graph_break()
    c = (b * z).sin()
    torch._dynamo.graph_break()
    d = (c * w).cos()
    return d.sum()

args = [torch.randn(32, device="cuda", requires_grad=True) for _ in range(4)]
ref_args = [a.detach().clone().requires_grad_(True) for a in args]
ref = fn(*ref_args)
ref.backward()

compiled_fn = torch.compile(fn)
res = compiled_fn(*args)
res.backward()

assert torch.allclose(res.detach(), ref.detach()), f"Forward mismatch: {res} vs {ref}"
for i, (a, r) in enumerate(zip(args, ref_args)):
    assert torch.allclose(a.grad, r.grad), f"Grad mismatch for arg {i}"
"""


class TestLazyCompileKernelCollision(InductorTestCase):
    device = GPU_TYPE

    def test_lazy_compile_kernel_name_collision_across_modules(self):
        """The collision manifests when a fresh process loads .so modules from
        warm on-disk caches: AOTAutograd cache hits cause both forward and
        backward .so to be loaded (static initializers register kernels in
        _pending_kernels) before either executes.  If two modules share a
        kernel name, the global dict collision corrupts the mapping.

        This requires two process invocations because dlopen within a single
        process reuses loaded libraries without re-running static initializers.
        """
        if not RUN_GPU:
            self.skipTest("GPU not available")

        with tempfile.TemporaryDirectory() as cache_dir:
            env = {
                **os.environ,
                "TORCHINDUCTOR_CACHE_DIR": cache_dir,
                "INDUCTOR_TEST_DISABLE_FRESH_CACHE": "1",
            }
            # First run: cold compile, populates on-disk caches.
            r1 = subprocess.run(
                [sys.executable, "-c", _LAZY_COMPILE_COLLISION_SCRIPT],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(r1.returncode, 0, f"Cold run failed:\n{r1.stderr[-2000:]}")
            # Second run: warm caches trigger the collision without the fix.
            r2 = subprocess.run(
                [sys.executable, "-c", _LAZY_COMPILE_COLLISION_SCRIPT],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(r2.returncode, 0, f"Warm run failed:\n{r2.stderr[-2000:]}")


class DynamicShapesGpuWrapperGpuTests(InductorTestCase):
    device = GPU_TYPE

    def test_annotation_training(self):
        batch_size = 32
        seq_length = 50
        hidden_size = 768

        def create_test_fn():
            def test_fn():
                inp = torch.randn(
                    batch_size, seq_length, hidden_size, device=self.device
                )
                weight = torch.randn(hidden_size, hidden_size, device=self.device)
                matmul_output = inp @ weight
                torch.nn.LayerNorm(hidden_size, device=self.device)(matmul_output)
                return True

            return test_fn

        fn = torch.compile(options={"annotate_training": True, "cpp_wrapper": True})(
            create_test_fn()
        )
        fn()


test_failures_gpu_wrapper = {
    "test_mm_plus_mm2_dynamic_shapes": test_torchinductor.TestFailure(
        ("gpu_wrapper",), is_skip=True
    ),
}

# XPU: complex add decomposition can return NotImplemented in cpp_wrapper path,
# which currently surfaces as InductorError in test_add_complex4_xpu_gpu_wrapper.
# Keep this targeted skip to XPU only.
if device_type == "xpu":
    test_failures_gpu_wrapper["test_add_complex4_xpu"] = test_torchinductor.TestFailure(
        ("gpu_wrapper",), is_skip=True
    )
    test_failures_gpu_wrapper["test_add_complex_xpu"] = test_torchinductor.TestFailure(
        ("gpu_wrapper",), is_skip=True
    )
    test_failures_gpu_wrapper["test_adding_tensor_offsets_xpu"] = (
        test_torchinductor.TestFailure(("gpu_wrapper",), is_skip=True)
    )

# Skip only on CUDA as wrapper dynamic shapes passes on ROCm.
# Per https://github.com/pytorch/pytorch/pull/172780
if not torch.version.hip:
    test_failures_gpu_wrapper["test_mm_plus_mm3_dynamic_shapes"] = (
        test_torchinductor.TestFailure(("gpu_wrapper",), is_skip=False)
    )


def make_test_case(
    name,
    device,
    tests,
    condition=True,
    slow=False,
    func_inputs=None,
    code_string_count=None,
    check_code=True,
):
    test_name = f"{name}_{device}" if device else name
    if code_string_count is None:
        code_string_count = {}

    func = getattr(tests, test_name)
    if not callable(func):
        raise AssertionError("not a callable")
    func = slowTest(func) if slow else func

    @config.patch(cpp_wrapper=True)
    def fn(self):
        tests.setUpClass()
        tests.setUp()
        try:
            with torch._C._PreserveDispatchKeyGuard():
                torch._C._dispatch_tls_set_dispatch_key_included(
                    torch._C.DispatchKey.Dense, True
                )

                _, code = test_torchinductor.run_and_get_cpp_code(
                    func, *func_inputs if func_inputs else []
                )
                if check_code:
                    self.assertEqual("CppWrapperCodeCache" in code, True)
                    self.assertTrue(
                        all(
                            code.count(string) == code_string_count[string]
                            for string in code_string_count
                        )
                    )
        finally:
            tests.tearDown()
            tests.tearDownClass()

    fn.__name__ = test_name
    import copy

    fn.__dict__ = copy.deepcopy(func.__dict__)
    if condition:
        setattr(
            GpuWrapperTemplate,
            test_name,
            fn,
        )


if RUN_GPU:

    class BaseTest(NamedTuple):
        name: str
        device: str = GPU_TYPE
        tests: InductorTestCase = test_torchinductor.GPUTests()
        check_code: bool = True

    # Maintain two separate test lists for cuda and cpp for now
    for item in [
        BaseTest("test_add_complex"),
        BaseTest("test_add_complex4"),
        BaseTest("test_as_strided"),  # buffer reuse
        BaseTest("test_batch_norm_2d_2"),
        BaseTest("test_bernoulli1_combo_kernels_False"),
        BaseTest("test_bernoulli1_combo_kernels_True"),
        BaseTest("test_bitwise"),  # int32
        BaseTest("test_bmm1"),
        BaseTest("test_bmm2"),
        BaseTest("test_buffer_use_after_remove"),
        BaseTest("test_cat"),  # alias
        BaseTest("test_convolution1"),
        BaseTest("test_conv_backward"),
        BaseTest("test_custom_op_1"),
        BaseTest("test_custom_op_2"),
        BaseTest("test_custom_op_3"),
        BaseTest("test_embedding_bag"),  # test default FallbackKernel
        BaseTest("test_index_put_deterministic_fallback"),
        BaseTest("test_adding_tensor_offsets"),
        BaseTest("test_index_tensor"),
        BaseTest("test_inductor_layout_optimization_input_mutations"),
        BaseTest("test_insignificant_strides"),
        BaseTest("test_layer_norm"),
        BaseTest("test_linear1"),
        BaseTest("test_linear2"),
        BaseTest("test_mm_views"),
        BaseTest("test_multi_device"),
        BaseTest("test_multi_threading"),
        BaseTest("test_pow3"),
        BaseTest("test_profiler_mark_wrapper_call"),
        BaseTest("test_randint"),
        BaseTest("test_reduction1"),  # Reduction
        BaseTest("test_relu"),  # multiple inputs
        BaseTest("test_repeat_interleave_2"),
        BaseTest("test_roi_align"),
        BaseTest("test_scalar_input"),
        BaseTest("test_scaled_dot_product_attention"),
        BaseTest("test_scaled_dot_product_efficient_attention"),
        BaseTest("test_sort"),
        BaseTest("test_silu"),  # single input, single output
        BaseTest("test_sum_dtype"),  # float64
        BaseTest("test_sum_int"),  # bool, int64, int8, uint8
        BaseTest("test_transpose"),  # multiple outputs, buffer clear
        *[
            BaseTest(f"test_unspec_inputs_{str(dtype)[6:]}")
            for dtype in test_torchinductor.test_dtypes
        ],
        BaseTest("test_consecutive_split_cumprod"),
        BaseTest("test_pointwise_hermite_polynomial_he"),
        BaseTest("test_pointwise_hermite_polynomial_h"),
        BaseTest(
            "test_foreach_cpp_wrapper",
            tests=test_foreach.ForeachTests(),
        ),  # test foreach
        BaseTest(
            "test_enable_dynamic_shapes_cpp_wrapper",
            tests=test_foreach.ForeachTests(),
        ),
        BaseTest(
            "test_dynamic_shapes_persistent_reduction_mixed_x_dim",
            tests=test_combo_kernels.ComboKernelDynamicShapesTests(),
        ),
        BaseTest(
            "test_cat_slice_cat",
            tests=test_pattern_matcher.TestPatternMatcher(),
        ),
        # TODO: Re-enable this test after fixing cuda wrapper for conv Triton templates with dynamic shapes.
        # This test is unstable: it succeeds when an ATEN kernel is used, and fails when a Triton kernel is used.
        # Currently it passes on CI (an ATEN kernel is chosen) and fails locally (a Triton kernel is chosen).
        # Ideally, it should succeed for whatever kernels.
        # BaseTest(
        #     "test_convolution1",
        #     device=None,
        #     tests=test_select_algorithm.TestSelectAlgorithm(),
        # ),
        BaseTest(
            "test_mm_plus_mm2",
            device=None,
            tests=test_select_algorithm.TestSelectAlgorithm(),
        ),
        BaseTest(
            "test_mm_plus_mm3",
            device=None,
            tests=test_select_algorithm.TestSelectAlgorithm(),
        ),
        BaseTest("test_fft_real_input"),
        BaseTest("test_fft_real_input_real_output"),
        *[
            # some dtypes may raise exception and be skipped in test_dtypeview, so set check_code to False here
            BaseTest(
                f"test_dtypeview_{str(dtype_x)[6:]}_{str(dtype_y)[6:]}",
                check_code=False,
            )
            for dtype_x, dtype_y in itertools.product(
                test_torchinductor.test_dtypes, test_torchinductor.test_dtypes
            )
        ],
        BaseTest("test_dtypeview_fusion"),
        # skip the next two tests if not enough SMs, logic for this is handled by TestSelectAlgorithm.setUp()
        BaseTest(
            "test_addmm",
            device=None,
            tests=test_select_algorithm.TestSelectAlgorithm(),
        ),
        BaseTest(
            "test_linear_relu",
            device=None,
            tests=test_select_algorithm.TestSelectAlgorithm(),
        ),
    ]:
        make_test_case(item.name, item.device, item.tests, check_code=item.check_code)

    test_torchinductor.copy_tests(
        GpuWrapperTemplate, TestGpuWrapper, "gpu_wrapper", test_failures_gpu_wrapper
    )

    DynamicShapesGpuWrapperTemplate = (
        test_torchinductor_dynamic_shapes.make_dynamic_cls(GpuWrapperTemplate)
    )

    test_torchinductor.copy_tests(
        DynamicShapesGpuWrapperTemplate,
        DynamicShapesGpuWrapperGpuTests,
        "gpu_wrapper",
        test_failures_gpu_wrapper,
        xfail_prop="_expected_failure_dynamic_wrapper",
    )

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if RUN_GPU:
        run_tests(needs="filelock")
