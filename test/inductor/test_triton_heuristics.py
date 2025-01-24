# Owner(s): ["module: inductor"]

import sys
import unittest

import torch
from torch._dynamo.testing import rand_strided
from torch._inductor.utils import clone_preserve_strides
from torch.testing._internal.common_utils import IS_LINUX, skipIfXpu
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
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
    triton_config,
)
from torch._inductor.test_case import run_tests, TestCase


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

    @skipIfXpu
    def test_artificial_zgrid(self):
        self._test_artificial_zgrid()

    @skipIfXpu
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
            "device": DeviceProperties.create(torch.device("cuda")),
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

    @skipIfXpu
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
        torch.cuda.reset_peak_memory_stats()
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
        peak_mem_after = torch.cuda.max_memory_allocated()

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


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU:
        run_tests()
