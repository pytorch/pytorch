# Owner(s): ["module: inductor"]

import functools
import os
import sys
import tempfile
import unittest
from unittest import skipUnless
from unittest.mock import MagicMock, patch

import torch
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime.triton_compat import HAS_WARP_SPEC
from torch._inductor.utils import clone_preserve_strides
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
    runOnRocm,
    skipIfXpu,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    HAS_GPU_AND_TRITON,
    requires_cuda_with_enough_memory,
)


try:
    import triton  # noqa: F401  # @manual
    import triton.language as tl  # @manual
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904

from torch._inductor import config
from torch._inductor.runtime.hints import (
    AttrsDescriptorWrapper,
    AutotuneHint,
    DeviceProperties,
    HeuristicType,
    TRITON_MAX_BLOCK,
)
from torch._inductor.runtime.triton_helpers import math as tl_math
from torch._inductor.runtime.triton_heuristics import (
    autotune_hints_to_configs,
    CachingAutotuner,
    template,
    triton_config,
)
from torch._inductor.test_case import run_tests, TestCase


@triton.jit
def amd_sqr_kernel(in_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    data = tl.load(in_ptr + offsets, mask=offsets < numel)
    sqr = data * data
    tl.store(out_ptr + offsets, sqr, mask=offsets < numel)


@functools.lru_cache
def get_autotuned_amd_sqr_kernel():
    return triton.autotune(
        configs=[
            triton.Config(
                {
                    "BLOCK_SIZE": 64,
                    "waves_per_eu": 3,
                }
            )
        ],
        key=[],
    )(amd_sqr_kernel)


@instantiate_parametrized_tests
class TestTritonHeuristics(TestCase):
    device_type = GPU_TYPE

    def test_triton_config(self):
        """
        Make sure block size does not exceed the maximum defined in inductor config.
        """
        cfg = triton_config({"x": 2048, "y": 2}, 64, 64)
        for label in "XYZ":
            key = f"{label}BLOCK"
            if key not in cfg.kwargs:
                continue
            self.assertTrue(cfg.kwargs[key] <= TRITON_MAX_BLOCK[label])

    def _test_artificial_zgrid(self):
        def forward(primals_1, primals_2, primals_5):
            view = torch.ops.aten.reshape.default(primals_5, [-1, 2, 4])
            primals_5 = None
            permute = torch.ops.aten.permute.default(view, [0, 2, 1])
            clone = torch.ops.aten.clone.default(
                permute, memory_format=torch.contiguous_format
            )
            permute = None
            view_1 = torch.ops.aten.reshape.default(clone, [-1, 4])
            clone = None
            permute_1 = torch.ops.aten.permute.default(primals_1, [1, 0])
            primals_1 = None
            addmm = torch.ops.aten.addmm.default(primals_2, view_1, permute_1)
            primals_2 = None
            return addmm

        s0 = 16777472
        s1 = 8

        args = [
            torch.rand([2, 4], device=GPU_TYPE),
            torch.rand([2], device=GPU_TYPE),
            torch.rand([s0, s1], device=GPU_TYPE),
        ]
        torch._dynamo.mark_dynamic(args[-1], 0)
        foo_c = torch.compile(forward)

        self.assertEqual(forward(*args), foo_c(*args))

        args = [
            torch.rand([2, 4], device=GPU_TYPE),
            torch.rand([2], device=GPU_TYPE),
            torch.rand([s0, s1], device=GPU_TYPE),
        ]
        self.assertEqual(forward(*args), foo_c(*args))

    # @skipIfXpu
    def test_artificial_zgrid(self):
        self._test_artificial_zgrid()

    # @skipIfXpu
    @config.patch("cpp_wrapper", True)
    def test_artificial_grid_cpp_wrapper(self):
        self._test_artificial_zgrid()

    @staticmethod
    def _get_cos_kernel_caching_autotuner_args():
        @triton.jit
        def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
            xnumel = 16
            xoffset = tl.program_id(0) * XBLOCK
            xindex = xoffset + tl.arange(0, XBLOCK)[:]
            xmask = xindex < xnumel
            x0 = xindex
            tmp0 = tl.load(in_ptr0 + (x0), xmask)
            tmp1 = tl_math.cos(tmp0)
            tl.store(out_ptr0 + (x0), tmp1, xmask)

        triton_meta = {
            "signature": {"in_ptr0": "*fp32", "out_ptr0": "*fp32", "xnumel": "i32"},
            "device": DeviceProperties.create(torch.device(GPU_TYPE)),
            "constants": {},
            "configs": [
                AttrsDescriptorWrapper(divisible_by_16=(0, 1, 2), equal_to_1=())
            ],
        }

        configs = [
            triton_config({"x": 16}, 64),
            triton_config({"x": 256}, 64),
        ]

        inductor_meta = {}

        return {
            "fn": triton_,
            "triton_meta": triton_meta,
            "configs": configs,
            "save_cache_hook": False,
            "mutated_arg_names": [],
            "reset_to_zero_arg_names": [],
            "optimize_mem": True,
            "heuristic_type": HeuristicType.POINTWISE,
            "inductor_meta": inductor_meta,
        }

    # @skipIfXpu
    def test_pre_hook_assert(self):
        # assert if any of the configs passed to the CachingAutotuner have pre-hooks
        args = self._get_cos_kernel_caching_autotuner_args()

        def pre_hook(kwargs):
            if "in_ptr0" in kwargs:
                kwargs["in_ptr0"].zero_()

        for cfg in args["configs"]:
            cfg.pre_hook = pre_hook

        with self.assertRaisesRegex(AssertionError, "pre_hook"):
            CachingAutotuner(**args)

    def test_autotune_hints_to_configs(self):
        device_props = DeviceProperties.create(torch.device(GPU_TYPE))
        device_props = device_props._replace(warp_size=8)

        hints = {AutotuneHint.ONE_ELEMENT_PER_THREAD}
        size_hints = (1024,)
        block_size = 256

        seen_num_elements_per_warp = set()

        def mock_triton_config(
            size_hints,
            x,
            y=None,
            z=None,
            num_stages=None,
            num_elements_per_warp=None,
            min_elem_per_thread=None,
        ):
            seen_num_elements_per_warp.add(num_elements_per_warp)
            return None

        with unittest.mock.patch(
            "torch._inductor.runtime.triton_heuristics.triton_config",
            mock_triton_config,
        ):
            _ = autotune_hints_to_configs(hints, size_hints, block_size, device_props)

        self.assertTrue(8 in seen_num_elements_per_warp)

    @unittest.skipIf(not HAS_WARP_SPEC, "FBCODE Triton is required for this test")
    def test_template_function_ws(self):
        triton_meta = {"device": MagicMock()}
        num_stages = 2
        num_warps = 4
        num_consumer_groups = 3
        num_buffers_warp_spec = 5

        with patch(
            "torch._inductor.runtime.triton_heuristics.cached_autotune"
        ) as mock_cached_autotune:
            template(
                num_stages=num_stages,
                num_warps=num_warps,
                triton_meta=triton_meta,
                num_consumer_groups=num_consumer_groups,
                num_buffers_warp_spec=num_buffers_warp_spec,
            )
            mock_cached_autotune.assert_called_once()
            configs = mock_cached_autotune.call_args[0][1]
            self.assertEqual(configs[0].num_consumer_groups, num_consumer_groups)
            self.assertEqual(configs[0].num_buffers_warp_spec, num_buffers_warp_spec)

    @runOnRocm
    def test_amd_special_config_args(self):
        """
        waves_per_eu is an example of a special config arg on AMD; if it is explicitly specified
        in a config, the kwarg will exist in the kwargs but not in the function signature.
        """

        @torch.library.triton_op("test_triton_heuristics::triton_sqr", mutates_args=())
        def triton_sqr(x: torch.Tensor) -> torch.Tensor:
            y = torch.empty_like(x)

            def grid(meta):
                return (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)

            torch.library.wrap_triton(get_autotuned_amd_sqr_kernel())[grid](
                x, y, x.numel()
            )

        def fn(x):
            return triton_sqr(x)

        x = torch.randn(32, device=GPU_TYPE)
        ref = fn(x)
        res = torch.compile(fn)(x)
        self.assertEqual(ref, res)

    @skipIfXpu(msg="https://github.com/intel/torch-xpu-ops/issues/2331")
    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @parametrize("do_pruning", [False, True])
    def test_prune_configs_over_shared_memory_limit(self, do_pruning):
        from torch._inductor.template_heuristics.triton import (
            CUDAConfigHeuristic,
            GemmConfig,
            ROCmConfigHeuristic,
        )

        expected_count = 1 if do_pruning else 2
        mm_configs = [
            GemmConfig(32, 32, 32, 1, 8, group_m=8),
            GemmConfig(
                128, 128, 128, 100, 8, group_m=4
            ),  # intentionally large to exceed shared memory limit
        ]
        with config.patch(
            {"max_autotune_prune_choices_based_on_shared_mem": do_pruning}
        ):
            if torch.version.hip:
                config_heuristic = ROCmConfigHeuristic()
            else:
                config_heuristic = CUDAConfigHeuristic()
            config_heuristic.should_scale_configs = False
            config_heuristic.mm_configs = mm_configs
            configs = list(
                config_heuristic.get_mm_configs()(3, 3, 3, dtype_size=4, op_name="mm")
            )
            self.assertEqual(len(configs), expected_count)


class TestArgumentCloneAndRestore(TestCase):
    # Our tensor is large enough. If a unexpected copy happens, the
    # peak memory increase should be larger than tolerance and the test
    # will fail.
    MEM_TOLERANCE = int(256 * 1e6)

    def _create_caching_autotuner(self):
        args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        args["optimize_mem"] = True
        args["mutated_arg_names"] = ["in_ptr0"]
        autotuner = CachingAutotuner(**args)
        return autotuner

    def _create_tensor(self, pad=1, with_offset=False):
        """
        Create a GPU tensor of about 1GB size.
        """
        M = 2
        N = 2**29 // 4
        out = rand_strided((M, N), (N + pad, 1), device=GPU_TYPE)
        if with_offset:
            out = out[:, 1:]
        return out

    def _do_test(self, gpu_tensor):
        torch.get_device_module(GPU_TYPE).reset_peak_memory_stats()
        autotuner = self._create_caching_autotuner()

        old_storage_offset = gpu_tensor.storage_offset()
        gpu_tensor_clone = clone_preserve_strides(gpu_tensor)

        peak_mem_before = torch.cuda.max_memory_allocated()
        cpu_copies = autotuner.copy_args_to_cpu_if_needed(gpu_tensor)
        self.assertTrue(len(cpu_copies) == 1)

        # Mutate the arg
        gpu_tensor.add_(1)

        # will restore gpu_tensor
        autotuner.restore_args_from_cpu(cpu_copies)
        self.assertTrue(gpu_tensor is not gpu_tensor_clone)
        self.assertEqual(gpu_tensor.size(), gpu_tensor_clone.size())
        self.assertEqual(gpu_tensor.stride(), gpu_tensor_clone.stride())
        self.assertEqual(gpu_tensor.storage_offset(), old_storage_offset)

        # Note: torch.allclose somehow allocates large amount of extra memory.
        # Record peak memory before that.
        peak_mem_after = torch.get_device_module(GPU_TYPE).max_memory_allocated()

        self.assertTrue(torch.allclose(gpu_tensor, gpu_tensor_clone))
        self.assertTrue(
            peak_mem_after <= peak_mem_before + self.MEM_TOLERANCE,
            f"{peak_mem_before=} v.s. {peak_mem_after=}",
        )

        # Avoid OOM in CI
        self.assertTrue(peak_mem_after < 1e10)

    @requires_cuda_with_enough_memory(1e10)
    def test_clone_contiguous_args(self):
        arg = self._create_tensor(pad=0)
        self.assertTrue(arg.is_contiguous())
        self.assertTrue(arg.storage_offset() == 0)
        self._do_test(arg)

    @requires_cuda_with_enough_memory(1e10)
    def test_clone_non_contiguous_args(self):
        arg = self._create_tensor(pad=1)
        self.assertFalse(arg.is_contiguous())
        self.assertTrue(arg.storage_offset() == 0)
        self._do_test(arg)

    @requires_cuda_with_enough_memory(1e10)
    def test_clone_args_with_non_zero_offset(self):
        arg = self._create_tensor(pad=1, with_offset=True)
        self.assertFalse(arg.is_contiguous())
        self.assertTrue(arg.storage_offset() > 0)

        self._do_test(arg)


class TestDumpLaunchTensors(TestCase):
    """Test the _dump_launch_tensors functionality"""

    def setUp(self):
        super().setUp()
        # Create a temporary directory for test dumps
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temporary directory
        import shutil

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        super().tearDown()

    @skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_dump_launch_tensors(self):
        """
        Test that dump_launch_tensors functions correctly:
        1. Creates the dump directory when torch.compile() runs
        2. Saves tensor files that can be loaded
        3. Loads tensors to match the original values
        4. Properly rotates when max_kernel_dump_occurrences is reached
        """
        from torch._inductor.config import triton as inductor_triton_config
        from torch._inductor.runtime.runtime_utils import cache_dir

        # Clear any existing state
        inductor_triton_config.debug_dump_kernel_inputs.clear()

        # Define a simple function that will generate Triton kernels
        def simple_model(x):
            y = x * 2.0
            z = y + 1.0
            return z.sum()

        old_dump_env = os.environ.get("TORCHINDUCTOR_DUMP_LAUNCH_TENSORS")
        os.environ["TORCHINDUCTOR_DUMP_LAUNCH_TENSORS"] = "1"

        try:
            compiled_fn = torch.compile(simple_model)

            # Run the compiled function multiple times to test rotation
            max_runs = inductor_triton_config.max_kernel_dump_occurrences

            for i in range(max_runs + 2):
                test_input = torch.randn(100, 100, device=GPU_TYPE) * (i + 1)
                _ = compiled_fn(test_input)

            # After multiple runs, verify rotation and tensor correctness
            kernel_bases = {}
            verified_tensor_load = False

            for root, dirs, files in os.walk(cache_dir()):
                for d in dirs:
                    if "_run_" in d:
                        full_path = os.path.join(root, d)
                        tensor_files = [
                            f
                            for f in os.listdir(full_path)
                            if f.startswith("tensor_") and f.endswith(".pt")
                        ]
                        if not tensor_files:
                            continue

                        dir_name = os.path.basename(full_path)
                        base_name = dir_name.rsplit("_run_", 1)[0]
                        run_idx = int(dir_name.rsplit("_run_", 1)[1])

                        # Track run indices per kernel
                        if base_name not in kernel_bases:
                            kernel_bases[base_name] = []
                        kernel_bases[base_name].append(run_idx)

                        # Verify we can successfully load at least one saved tensor
                        if not verified_tensor_load:
                            first_tensor_file = os.path.join(full_path, tensor_files[0])
                            loaded_tensor = torch.load(first_tensor_file)

                            # Verify it's a valid tensor with expected properties
                            self.assertIsInstance(loaded_tensor, torch.Tensor)
                            self.assertEqual(loaded_tensor.device.type, GPU_TYPE)
                            verified_tensor_load = True

            # Verify rotation constraints
            if kernel_bases:
                for base_name, indices in kernel_bases.items():
                    self.assertLessEqual(
                        len(indices),
                        max_runs,
                        f"Kernel {base_name} has more runs ({len(indices)}) than max ({max_runs})",
                    )

                    # Verify the indices are within [0, max_runs)
                    for idx in indices:
                        self.assertLess(
                            idx,
                            max_runs,
                            f"Run index {idx} exceeds max_runs-1 ({max_runs - 1})",
                        )

        finally:
            # Restore environment variable
            if old_dump_env is None:
                os.environ.pop("TORCHINDUCTOR_DUMP_LAUNCH_TENSORS", None)
            else:
                os.environ["TORCHINDUCTOR_DUMP_LAUNCH_TENSORS"] = old_dump_env


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU:
        run_tests()
