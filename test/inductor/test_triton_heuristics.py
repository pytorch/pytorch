# Owner(s): ["module: inductor"]

import functools
import math
import os
import sys
import tempfile
import types
import unittest
from unittest import skipUnless
from unittest.mock import MagicMock, patch, PropertyMock

import sympy

import torch
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime.triton_compat import Config, HAS_WARP_SPEC
from torch._inductor.utils import clone_preserve_strides
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
    runOnRocm,
    skipIfRocm,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU_AND_TRITON,
    requires_gpu_with_enough_memory,
)


try:
    import triton  # @manual
    import triton.language as tl  # @manual
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904

from torch._inductor import config
from torch._inductor.heuristics.registry import CodegenConfigHeuristics
from torch._inductor.heuristics.triton_codegen.pointwise import PointwiseHeuristic
from torch._inductor.runtime.coordinate_descent_tuner import CoordescTuner
from torch._inductor.runtime.hints import (
    AttrsDescriptorWrapper,
    AutotuneHint,
    DeviceProperties,
    HeuristicType,
    native_matmul_block_numel,
    native_matmul_persistent_rblock,
    ReductionHint,
    TRITON_MAX_BLOCK,
    TRITON_MAX_TENSOR_NUMEL,
)
from torch._inductor.runtime.triton_helpers import math as tl_math
from torch._inductor.runtime.triton_heuristics import (
    _check_max_grid_x,
    _enforce_reduction_config_block_minimums,
    _find_names,
    _num_warps,
    _persistent_reduction_configs,
    _reduction_configs,
    _warn_ignored_pin,
    autotune_hints_to_configs,
    cached_autotune,
    CachingAutotuner,
    CachingAutotunerPlugin,
    CappedGrid1D,
    check_autotune_cache,
    check_capped_grid_config,
    DEFER,
    generate_lookup_hash_from_source_code,
    GridExpr,
    lookup_autotune_config,
    make_matmul_triton_config,
    pointwise,
    reduction,
    template,
    triton_config,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache


class _FindNamesProbe:
    # A gc-tracked stand-in for a CachingAutotuner; object() is untracked, so a
    # dict holding one can itself stay untracked and escape gc.get_referrers.
    pass


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

    def test_find_names_ignores_frame_locals(self):
        """
        A kernel held only by frame locals has no name, so DebugAutotuner can
        fall back to inductor_meta["kernel_name"]. The stack walk used to leak
        _find_names' own `obj` and the caller's `self` into the result.
        """
        probe = _FindNamesProbe()
        self.assertEqual(_find_names(probe), [])

    def test_find_names_finds_namespace_binding(self):
        probe = _FindNamesProbe()
        namespace = {"triton_poi_fused_add_0": probe}
        try:
            self.assertIn("triton_poi_fused_add_0", _find_names(probe))
        finally:
            del namespace

    def test_unbound_kernel_falls_back_to_inductor_meta_name(self):
        """
        The label DebugAutotuner.run derives for an unbound kernel. The stack
        walk made _find_names return ['obj', 'self'], so max(..., key=len)
        picked "self" instead of reaching the inductor_meta fallback.
        """

        class FakeAutotuner:
            inductor_meta = {"kernel_name": "triton_poi_fused_add_0"}

            def kernel_name(self):
                possible_names = _find_names(self)
                if possible_names:
                    return f"{max(possible_names, key=len)}"
                return self.inductor_meta["kernel_name"]

        self.assertEqual(FakeAutotuner().kernel_name(), "triton_poi_fused_add_0")

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

    def test_native_matmul_config_block_numel_limit(self):
        device = DeviceProperties(
            type="cuda",
            index=0,
            multi_processor_count=1,
            cc=80,
            major=8,
            max_threads_per_block=1024,
            warp_size=32,
        )
        triton_meta = {"native_matmul": True, "device": device}

        for size_hints in (
            {"x": 1, "y": 1, "r0_": 1},
            {"x": 1, "y": 1, "z": 1, "r0_": 1},
        ):
            cfgs = _reduction_configs(
                size_hints=size_hints,
                inductor_meta={},
                triton_meta=triton_meta,
            )
            self.assertTrue(cfgs)
            for cfg in cfgs:
                self.assertLessEqual(
                    native_matmul_block_numel(cfg.kwargs),
                    TRITON_MAX_TENSOR_NUMEL,
                )

        for size_hints, inductor_meta in (
            ({"x": 4096, "y": 4096, "r0_": 1}, {}),
            ({"x": 16384, "y": 2048, "r0_": 2048}, {}),
            (
                {"x": 4096, "y": 4096, "r0_": 64},
                {"native_matmul_persistent_rblock": 1024},
            ),
            ({"x": 4096, "y": 4096, "z": 128, "r0_": 1}, {}),
            ({"x": 16384, "y": 2048, "z": 128, "r0_": 2048}, {}),
            (
                {"x": 4096, "y": 4096, "z": 128, "r0_": 64},
                {"native_matmul_persistent_rblock": 1024},
            ),
        ):
            cfgs = _persistent_reduction_configs(
                size_hints=size_hints,
                inductor_meta=inductor_meta,
                triton_meta=triton_meta,
            )
            rblock = inductor_meta.get(
                "native_matmul_persistent_rblock",
                native_matmul_persistent_rblock(size_hints["r0_"]),
            )
            self.assertTrue(cfgs)
            for cfg in cfgs:
                self.assertLessEqual(
                    native_matmul_block_numel(cfg.kwargs, r0_block=rblock),
                    TRITON_MAX_TENSOR_NUMEL,
                )

        with self.assertRaisesRegex(AssertionError, "exceeds Triton maximum"):
            make_matmul_triton_config({"x": 256, "y": 128, "r": 64}, 8, 1)

    def test_reduction_min_block_preserves_tile_product(self):
        cfg = _enforce_reduction_config_block_minimums(
            [triton.Config({"XBLOCK": 64, "R0_BLOCK": 1024})],
            {"x": 4096, "r0_": 4096},
            {"min_xblock": 128},
        )[0]
        self.assertEqual(cfg.kwargs["XBLOCK"], 128)
        self.assertEqual(cfg.kwargs["R0_BLOCK"], 512)

        cfg = _enforce_reduction_config_block_minimums(
            [triton.Config({"XBLOCK": 1024, "R0_BLOCK": 64})],
            {"x": 4096, "r0_": 4096},
            {"min_rblock": 128},
        )[0]
        self.assertEqual(cfg.kwargs["XBLOCK"], 512)
        self.assertEqual(cfg.kwargs["R0_BLOCK"], 128)

    @parametrize(
        "major,cc,expected_baseline_configs,expected_scalar_configs",
        [
            (
                8,
                80,
                [(1, 2048, 16, 1)],
                [(1, 4096, 16, 1), (1, 8192, 4, 1), (1, 16384, 8, 1)],
            ),
            (
                10,
                100,
                [(1, 1024, 8, 1)],
                [(1, 4096, 16, 1), (1, 8192, 4, 1), (1, 16384, 8, 1)],
            ),
            (
                10,
                103,
                [(1, 1024, 8, 1), (1, 4096, 8, 1)],
                [
                    (1, 4096, 16, 1),
                    (1, 8192, 4, 1),
                    (1, 16384, 8, 1),
                    (1, 4096, 8, 1),
                ],
            ),
        ],
    )
    def test_scalar_accumulator_reduction_configs(
        self, major, cc, expected_baseline_configs, expected_scalar_configs
    ):
        device = self._fake_cuda_device_properties()._replace(major=major, cc=cc)
        size_hints = {"x": 128, "r0_": 32768}
        triton_meta = {"device": device}

        def config_values(autotune_hints):
            configs = _reduction_configs(
                size_hints=size_hints,
                inductor_meta={
                    "autotune_hints": autotune_hints,
                    "reduction_hint": ReductionHint.INNER,
                },
                triton_meta=triton_meta,
            )
            return [
                (
                    config.kwargs["XBLOCK"],
                    config.kwargs["R0_BLOCK"],
                    config.num_warps,
                    config.num_stages,
                )
                for config in configs
            ]

        def tiled_block_products(autotune_hints):
            configs = _reduction_configs(
                size_hints={"x": 128, "y": 8, "r0_": 32768},
                inductor_meta={
                    "autotune_hints": autotune_hints,
                    "reduction_hint": ReductionHint.INNER,
                },
                triton_meta=triton_meta,
            )
            return [
                math.prod(
                    value
                    for name, value in config.kwargs.items()
                    if name.endswith("BLOCK")
                )
                for config in configs
            ]

        self.assertEqual(
            config_values(set()),
            expected_baseline_configs,
        )
        self.assertEqual(
            config_values({AutotuneHint.SCALAR_ACCUMULATORS}),
            expected_scalar_configs,
        )
        baseline_rblock = expected_baseline_configs[0][1]
        self.assertEqual(tiled_block_products(set())[0], baseline_rblock)
        scalar_tiled_products = tiled_block_products({AutotuneHint.SCALAR_ACCUMULATORS})
        self.assertEqual(scalar_tiled_products[0], 4096)
        self.assertIn(baseline_rblock, scalar_tiled_products)

    def test_inner_reduction_no_split_threshold_is_nvidia_only(self):
        from torch._inductor.choices import InductorChoices

        choices = InductorChoices()
        props = self._fake_cuda_device_properties()._replace(major=12)
        self.assertEqual(
            choices._inner_reduction_no_split_threshold(
                props._replace(type="hip"), xnumel=1, num_sm=16
            ),
            8192,
        )
        self.assertEqual(
            choices._inner_reduction_no_split_threshold(props, xnumel=1, num_sm=16),
            32768,
        )
        self.assertEqual(
            choices._inner_reduction_no_split_threshold(props, xnumel=64, num_sm=16),
            40960,
        )
        self.assertEqual(
            choices._inner_reduction_no_split_threshold(
                props._replace(major=None), xnumel=1, num_sm=16
            ),
            8192,
        )

    def test_reduction_r0_block_cap_is_nvidia_only(self):
        from torch._inductor.heuristics.triton_codegen.reduction import (
            ReductionHeuristic,
            ROCmReductionHeuristic,
        )

        def first_r0_block(heuristic, major, cc):
            configs = heuristic.get_configs(
                size_hints={"x": 128, "r0_": 2048},
                inductor_meta={"reduction_hint": ReductionHint.INNER},
                triton_meta={
                    "device": self._fake_cuda_device_properties()._replace(
                        major=major, cc=cc
                    )
                },
            )
            return configs[0].kwargs["R0_BLOCK"]

        self.assertEqual(first_r0_block(ReductionHeuristic(), 8, 80), 2048)
        self.assertEqual(first_r0_block(ReductionHeuristic(), 10, 100), 1024)
        self.assertEqual(first_r0_block(ROCmReductionHeuristic(), 12, 120), 2048)
        self.assertEqual(first_r0_block(ROCmReductionHeuristic(), 9, 95), 2048)

    def test_cached_autotune_enforces_reduction_min_block(self):
        def triton_fn(XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
            pass

        class FakeJitFunction:
            def __init__(self):
                self.fn = triton_fn

        class CaptureAutotuner:
            def __init__(self, *args, configs, **kwargs):
                self.configs = configs

        autotuner = cached_autotune(
            {"x": 4096, "r0_": 4096},
            [triton.Config({"XBLOCK": 64, "R0_BLOCK": 1024})],
            triton_meta={},
            heuristic_type=HeuristicType.REDUCTION,
            inductor_meta={"min_xblock": 128},
            caching_autotuner_cls=CaptureAutotuner,
        )(FakeJitFunction())

        cfg = autotuner.configs[0]
        self.assertEqual(cfg.kwargs["XBLOCK"], 128)
        self.assertEqual(cfg.kwargs["R0_BLOCK"], 512)

    @staticmethod
    def _fake_cuda_device_properties():
        return DeviceProperties(
            type="cuda",
            index=0,
            multi_processor_count=1,
            cc=80,
            major=8,
            regs_per_multiprocessor=65536,
            max_threads_per_multi_processor=2048,
            max_threads_per_block=1024,
            warp_size=32,
        )

    @staticmethod
    def _run_single_dsr_reduction_cached_autotune(configs, fake_cache):
        def triton_fn(XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
            pass

        class FakeJitFunction:
            def __init__(self):
                self.fn = triton_fn

        class CaptureAutotuner:
            def __init__(
                self, *args, configs, save_cache_hook, autotune_cache_info, **kwargs
            ):
                self.configs = configs
                self.save_cache_hook = save_cache_hook
                self.autotune_cache_info = autotune_cache_info

        with patch(
            "torch._inductor.runtime.triton_heuristics.AutotuneCache.create",
            return_value=fake_cache,
        ) as create:
            autotuner = cached_autotune(
                {"x": 4096, "r0_": 4096},
                configs,
                triton_meta={
                    "device": TestTritonHeuristics._fake_cuda_device_properties()
                },
                heuristic_type=HeuristicType.REDUCTION,
                filename="/tmp/kernel.py",
                inductor_meta={},
                caching_autotuner_cls=CaptureAutotuner,
            )(FakeJitFunction())

        return autotuner, create

    def test_cached_autotune_uses_cache_for_single_dsr_reduction_config(self):
        cfg = triton.Config({"XBLOCK": 1, "R0_BLOCK": 2048}, num_warps=16)
        fake_cache = MagicMock()
        fake_cache.read_best.return_value = None

        autotuner, create = self._run_single_dsr_reduction_cached_autotune(
            [cfg], fake_cache
        )

        create.assert_called_once()
        self.assertIs(autotuner.save_cache_hook, fake_cache.save)
        self.assertEqual(autotuner.autotune_cache_info["autotune_cache_state"], "miss")
        self.assertEqual(autotuner.autotune_cache_info["num_configs"], 1)

    def test_cached_autotune_loads_single_dsr_reduction_cache_hit(self):
        original_cfg = triton.Config({"XBLOCK": 1, "R0_BLOCK": 2048}, num_warps=16)
        dynamic_cfg = triton.Config({"XBLOCK": 1, "R0_BLOCK": 1024}, num_warps=16)

        fake_cache = MagicMock()
        fake_cache.read_best.return_value = dynamic_cfg

        autotuner, _ = self._run_single_dsr_reduction_cached_autotune(
            [original_cfg], fake_cache
        )

        self.assertEqual(autotuner.configs, [dynamic_cfg])
        self.assertEqual(autotuner.autotune_cache_info["autotune_cache_state"], "hit")

    def test_check_autotune_cache_skips_single_config_without_dsr(self):
        cfg = triton.Config({"XBLOCK": 1, "R0_BLOCK": 2048}, num_warps=16)

        with patch(
            "torch._inductor.runtime.triton_heuristics.AutotuneCache.create"
        ) as create:
            configs, autotune_cache, autotune_cache_info = check_autotune_cache(
                [cfg],
                "/tmp/kernel.py",
                {},
                dynamic_scale_rblock_eligible=False,
            )

        create.assert_not_called()
        self.assertEqual(configs, [cfg])
        self.assertIsNone(autotune_cache)
        self.assertEqual(autotune_cache_info["autotune_cache_state"], "only 1 config")

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

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_artificial_zgrid(self):
        self._test_artificial_zgrid()

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
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
            "signature": {
                "in_ptr0": "*fp32",
                "out_ptr0": "*fp32",
                "xnumel": "i32",
                "XBLOCK": "constexpr",
            },
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

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
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

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
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
            *,
            warp_size=32,
        ):
            seen_num_elements_per_warp.add(num_elements_per_warp)
            return None

        with unittest.mock.patch(
            "torch._inductor.runtime.triton_heuristics.triton_config",
            mock_triton_config,
        ):
            _ = autotune_hints_to_configs(hints, size_hints, block_size, device_props)

        self.assertTrue(8 in seen_num_elements_per_warp)
        self.assertEqual(
            autotune_hints_to_configs(
                {AutotuneHint.SCALAR_ACCUMULATORS},
                size_hints,
                block_size,
                device_props,
            ),
            [],
        )

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

    @parametrize(
        "hip_version, tlx_options, waves_per_eu, expected_kwargs",
        [
            (
                "7.2",
                ("matrix_instr_nonkdim", "waves_per_eu", "kpack", "test_extra_option"),
                0,
                {
                    "matrix_instr_nonkdim": 16,
                    "waves_per_eu": 0,
                    "kpack": 1,
                    "test_extra_option": 7,
                },
            ),
            (
                "7.2",
                (),
                0,
                {
                    "matrix_instr_nonkdim": 16,
                    "waves_per_eu": 0,
                    "kpack": 1,
                },
            ),
            (
                "7.2",
                (),
                2,
                {
                    "matrix_instr_nonkdim": 16,
                    "waves_per_eu": 2,
                    "kpack": 1,
                },
            ),
            (
                None,
                ("matrix_instr_nonkdim", "waves_per_eu", "kpack"),
                0,
                {},
            ),
        ],
    )
    def test_template_function_preserves_hip_options(
        self, hip_version, tlx_options, waves_per_eu, expected_kwargs
    ):
        triton_meta = {
            "device": MagicMock(),
            "matrix_instr_nonkdim": 16,
            "waves_per_eu": waves_per_eu,
            "kpack": 1,
            "test_extra_option": 7,
        }

        def capture_configs(_size_hints, configs, **_kwargs):
            return configs

        with (
            patch.object(torch.version, "hip", hip_version),
            patch(
                "torch._inductor.utils.tlx_only_hip_options",
                return_value=tlx_options,
            ),
            patch(
                "torch._inductor.runtime.triton_heuristics.cached_autotune",
                side_effect=capture_configs,
            ),
        ):
            configs = template(
                num_stages=1,
                num_warps=4,
                triton_meta=triton_meta,
            )

        self.assertEqual(configs[0].kwargs, expected_kwargs)

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

    @runOnRocm
    def test_rocm_exhaustive_configs_do_not_enumerate_num_stages(self):
        """ROCmConfigHeuristic._filter_configs forces num_stages to
        default_num_stages, so enumerating num_stages in exhaustive_configs would
        only yield duplicates that get deduped again in _finalize_mm_configs.
        """
        from torch._inductor.heuristics.template.triton import (
            GemmConfig,
            ROCmConfigHeuristic,
        )

        heuristic = ROCmConfigHeuristic()
        self.assertEqual(
            {c.num_stages for c in heuristic.exhaustive_configs},
            {heuristic.default_num_stages},
        )

        # the invariant above is only safe because _filter_configs clobbers it
        filtered = heuristic._filter_configs([GemmConfig(32, 32, 32, 1, 4, group_m=8)])
        self.assertEqual(filtered[0].num_stages, heuristic.default_num_stages)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @parametrize("do_pruning", [False, True])
    def test_prune_configs_over_shared_memory_limit(self, do_pruning):
        from torch._inductor.heuristics.template.triton import (
            CUDAConfigHeuristic,
            GemmConfig,
            ROCmConfigHeuristic,
            XPUConfigHeuristic,
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
            if GPU_TYPE == "xpu":
                config_heuristic = XPUConfigHeuristic()
            elif torch.version.hip:
                config_heuristic = ROCmConfigHeuristic()
            else:
                config_heuristic = CUDAConfigHeuristic()
            config_heuristic.should_scale_configs = False
            config_heuristic.mm_configs = mm_configs
            configs = list(
                config_heuristic.get_mm_configs()(3, 3, 3, dtype_size=4, op_name="mm")
            )
            self.assertEqual(len(configs), expected_count)

    @runOnRocm
    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_flex_default_configs_gated_on_arch_name(self):
        """A gfx target gets tuned flex defaults only if registered for them.

        get_device_capability() on ROCm reports the gfx major/minor, which
        neither orders by capability nor identifies the product line, so gating
        on `>= (9, 5)` also matches gfx1030, gfx1100 and gfx1250 (MI450) and
        hands them tables measured on gfx950.
        """
        from torch._inductor.heuristics.template.triton import ROCmConfigHeuristic
        from torch._inductor.utils import rocm_gfx_arch

        heuristic = ROCmConfigHeuristic()
        dtype, head_dim = torch.float16, 128

        # the registries are keyed on the bare target, so the features that
        # gcnArchName carries ("gfx950:sramecc+:xnack-") have to be stripped
        self.assertNotIn(":", rocm_gfx_arch())
        self.assertTrue(
            torch.cuda.get_device_properties(0).gcnArchName.startswith(rocm_gfx_arch())
        )

        # stands in for any target that has no measured values of its own, and
        # gives the fallback to compare against without pinning its values here
        unregistered = "gfx000"
        # RDNA3 is deliberately absent: it has its own branch, keyed by sequence
        # length rather than by target, so it is not part of this dispatch.
        targets = ("gfx942", "gfx950", "gfx1250", unregistered)

        def configs_for(arch, get_configs):
            with (
                patch(
                    "torch._inductor.heuristics.template.triton.rocm_gfx_arch",
                    lambda: arch,
                ),
                patch(
                    "torch._inductor.heuristics.template.triton.using_rocm_rdna3",
                    lambda: False,
                ),
            ):
                return get_configs()

        paths = (
            (
                "forward",
                heuristic.flex_fwd_config_by_arch,
                lambda: heuristic.get_flex_attn_fwd_configs(
                    head_dim, sympy.Integer(1024), dtype
                ),
            ),
            (
                "backward",
                heuristic.flex_bwd_config_by_arch,
                lambda: heuristic.get_flex_attn_bwd_configs(head_dim, dtype),
            ),
        )
        for path, registry, get_configs in paths:
            with self.subTest(path=path):
                self.assertIn("gfx950", registry)
                self.assertNotIn(unregistered, registry)
                fallback = configs_for(unregistered, get_configs)
                for arch in targets:
                    configs = configs_for(arch, get_configs)
                    tuned = registry.get(arch, {}).get((dtype, head_dim))
                    if tuned is None:
                        self.assertEqual(configs, fallback, msg=arch)
                    else:
                        self.assertEqual(configs, [tuned], msg=arch)
                        self.assertNotEqual(configs, fallback, msg=arch)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @parametrize(
        "arch,f16_count,int8_count,int8_kpack",
        [
            ("gfx908", 0, 1, 2),
            ("gfx90a", 0, 1, 2),
            ("gfx942", 0, 0, 1),
            ("gfx950", 1, 1, 1),
        ],
    )
    def test_rocm_prune_block_k_underfills_mfma(
        self, arch, f16_count, int8_count, int8_kpack
    ):
        if not torch.version.hip:
            self.skipTest("ROCm-specific MFMA config pruning")
        from torch._inductor.heuristics.template.triton import (
            BaseHeuristicSingleton,
            GemmConfig,
            ROCmConfigHeuristic,
            ROCmGemmConfig,
        )
        from torch._inductor.utils import kpack_supported

        with (
            patch("torch._inductor.utils.rocm_gfx_arch", return_value=arch),
            patch.object(BaseHeuristicSingleton, "_instances", {}),
        ):
            heuristic = ROCmConfigHeuristic()
            heuristic.should_scale_configs = False
            self.assertEqual(kpack_supported(), arch != "gfx950")

            def configs_for(mm_configs, dsize):
                heuristic.mm_configs = mm_configs
                return list(
                    heuristic.get_mm_configs()(
                        1, 256, 128, dtype_size=dsize, op_name="mm"
                    )
                )

            underfilled = ROCmGemmConfig(
                16, 64, 16, 2, 4, group_m=8, matrix_instr_nonkdim=16, kpack=2
            )
            valid = ROCmGemmConfig(
                16, 64, 32, 2, 4, group_m=8, matrix_instr_nonkdim=16, kpack=2
            )
            # f16 has kdim=16 on CDNA1/2/3; gfx950 skips underfill pruning.
            self.assertEqual(len(configs_for([underfilled], 2)), f16_count)
            self.assertEqual(len(configs_for([valid], 2)), 1)

            # int8 has kdim=16 on CDNA1/2, but kdim=32 on CDNA3.
            self.assertEqual(len(configs_for([valid], 1)), int8_count)

            # Unknown dtype kdim skips pruning on every target.
            self.assertEqual(len(configs_for([underfilled], 8)), 1)

            # Implicit kpack is clamped on CDNA3, preserving both int8 configs.
            int8_configs = [
                GemmConfig(64, 64, 32, 2, 4),
                GemmConfig(128, 128, 32, 2, 8),
            ]
            configs = configs_for(int8_configs, 1)
            self.assertEqual(len(configs), len(int8_configs))
            self.assertEqual([c.kwargs["kpack"] for c in configs], [int8_kpack] * 2)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_compile_time_autotune_not_repeated_at_runtime(self):
        def fn(x):
            return (x + 1).sum(dim=1)

        x = torch.randn(2048, 2048, device=GPU_TYPE)
        benchmark_calls = []

        def fake_benchmark_all_configs(autotuner, *args, **kwargs):
            benchmark_calls.append(autotuner.inductor_meta.get("kernel_name"))
            return {
                launcher: float(idx) for idx, launcher in enumerate(autotuner.launchers)
            }

        torch._dynamo.reset()
        with (
            fresh_cache(),
            config.patch(
                {
                    "triton.autotune_at_compile_time": True,
                    "max_autotune_pointwise": True,
                    "compile_threads": 1,
                }
            ),
            patch.object(
                CachingAutotuner,
                "benchmark_all_configs",
                fake_benchmark_all_configs,
            ),
        ):
            compiled = torch.compile(fn, backend="inductor")
            self.assertEqual(compiled(x), fn(x))
            self.assertEqual(len(benchmark_calls), 1)

            self.assertEqual(compiled(x), fn(x))
            self.assertEqual(len(benchmark_calls), 1)


_PLUGIN_FACTORY_PATH = (
    "torch._inductor.runtime.triton_heuristics.get_caching_autotuner_plugins"
)


class TestCachingAutotunerPrecompileDriverSetup(TestCase):
    @skipIfRocm
    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_warm_cache_only_precompile_skips_driver_setup_in_context(self):
        from torch._inductor.runtime import triton_helpers

        args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        num_configs = len(args["configs"])
        self.assertGreaterEqual(num_configs, 2)
        autotuner = CachingAutotuner(**args)

        with (
            triton_helpers.skip_gpu_driver_setup(),
            patch.object(
                type(triton.runtime.driver),
                "active",
                new_callable=PropertyMock,
                side_effect=RuntimeError("driver.active should not be read"),
            ) as mock_driver_active,
        ):
            autotuner.precompile(warm_cache_only=True)

        # Covers every synchronous _precompile_config() iteration.
        mock_driver_active.assert_not_called()
        self.assertEqual(len(autotuner.compile_results), num_configs)


@skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
class TestCachingAutotunerPlugin(TestCase):
    device_type = GPU_TYPE

    @staticmethod
    def _get_stream():
        from torch._dynamo.device_interface import get_interface_for_device

        device_interface = get_interface_for_device(GPU_TYPE)
        return device_interface.get_raw_stream(device_interface.current_device())

    @staticmethod
    def _make_kernel_inputs():
        in_ptr = torch.zeros(16, device=GPU_TYPE, dtype=torch.float32)
        out_ptr = torch.zeros(16, device=GPU_TYPE, dtype=torch.float32)
        return (in_ptr, out_ptr, 16)

    def _make_autotuner(self, plugins, configs=None):
        args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        args["inductor_meta"] = {**args["inductor_meta"], "grid_type": "Grid1D"}
        if configs is not None:
            args["configs"] = configs
        with patch(_PLUGIN_FACTORY_PATH, return_value=list(plugins)):
            return CachingAutotuner(**args)

    def test_pre_dispatch_runs_before_precompile_and_autotune(self):
        sentinel = object()

        class _Plugin(CachingAutotunerPlugin):
            def pre_dispatch(self, autotuner, *args, stream, **kwargs):
                return sentinel

        autotuner = self._make_autotuner([_Plugin()])
        with (
            patch.object(autotuner, "precompile") as mock_precompile,
            patch.object(autotuner, "autotune_to_one_config") as mock_autotune,
        ):
            result = autotuner.run(
                *self._make_kernel_inputs(), stream=self._get_stream()
            )

        self.assertIs(result, sentinel)
        mock_precompile.assert_not_called()
        mock_autotune.assert_not_called()

    def test_pre_autotune_runs_before_default_autotune(self):
        sentinel = object()

        class _Plugin(CachingAutotunerPlugin):
            def pre_autotune(self, autotuner, *args, stream, **kwargs):
                return sentinel

        autotuner = self._make_autotuner([_Plugin()])
        with patch.object(autotuner, "autotune_to_one_config") as mock_autotune:
            result = autotuner.run(
                *self._make_kernel_inputs(), stream=self._get_stream()
            )

        self.assertIs(result, sentinel)
        mock_autotune.assert_not_called()

    def test_hooks_fire_in_registration_order(self):
        sentinel = object()
        seen = []

        class _Plugin(CachingAutotunerPlugin):
            def __init__(self, name, return_value=DEFER):
                self.name = name
                self.return_value = return_value

            def pre_dispatch(self, autotuner, *args, stream, **kwargs):
                seen.append(self.name)
                return self.return_value

        autotuner = self._make_autotuner(
            [_Plugin("a"), _Plugin("b"), _Plugin("c", sentinel), _Plugin("d")]
        )
        result = autotuner.run(*self._make_kernel_inputs(), stream=self._get_stream())

        self.assertIs(result, sentinel)
        self.assertEqual(seen, ["a", "b", "c"])

    def test_defer_falls_through_to_default(self):
        seen = []

        class _Plugin(CachingAutotunerPlugin):
            def pre_dispatch(self, autotuner, *args, stream, **kwargs):
                seen.append("pre_dispatch")
                return DEFER

        # Single config so precompile produces one launcher and the kernel
        # runs to completion without the autotune branch firing.
        full_args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        autotuner = self._make_autotuner([_Plugin()], configs=full_args["configs"][:1])

        autotuner.run(*self._make_kernel_inputs(), stream=self._get_stream())

        self.assertEqual(seen, ["pre_dispatch"])
        self.assertEqual(len(autotuner.launchers), 1)

    def test_pre_compile_defer_runs_default_precompile_flow(self):
        """``DEFER`` from pre_compile lets the standard precompile flow
        run (``_precompile_worker`` → ``_make_launchers`` →
        ``_dynamic_scale_rblock``)."""

        class _Plugin(CachingAutotunerPlugin):
            def pre_compile(self, autotuner):
                return DEFER

        full_args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        autotuner = self._make_autotuner([_Plugin()], configs=full_args["configs"][:1])
        with (
            patch.object(autotuner, "_precompile_worker") as mock_pcw,
            patch.object(autotuner, "_make_launchers") as mock_mkl,
            patch.object(autotuner, "_dynamic_scale_rblock") as mock_dsr,
        ):
            autotuner.precompile()
        mock_pcw.assert_called_once()
        mock_mkl.assert_called_once()
        mock_dsr.assert_called_once()

    def test_pre_compile_non_defer_short_circuits_precompile(self):
        """A non-``DEFER`` return from pre_compile means the plugin
        owns the entire compile pipeline; ``precompile`` returns early
        without running ``_precompile_worker`` / ``_make_launchers`` /
        ``_dynamic_scale_rblock``."""

        class _Plugin(CachingAutotunerPlugin):
            def pre_compile(self, autotuner):
                return object()  # any non-DEFER value

        full_args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        autotuner = self._make_autotuner([_Plugin()], configs=full_args["configs"][:1])
        with (
            patch.object(autotuner, "_precompile_worker") as mock_pcw,
            patch.object(autotuner, "_make_launchers") as mock_mkl,
            patch.object(autotuner, "_dynamic_scale_rblock") as mock_dsr,
        ):
            autotuner.precompile()
        mock_pcw.assert_not_called()
        mock_mkl.assert_not_called()
        mock_dsr.assert_not_called()

    def test_getstate_clears_plugins_setstate_invokes_factory(self):
        """``__getstate__`` drops the live plugin instances so they
        don't pickle into workers; ``__setstate__`` re-runs
        ``get_caching_autotuner_plugins`` so the revived autotuner
        gets a fresh, factory-driven list (whose contents reflect the
        unpickling process's config flags, not the pickling process's)."""

        class _PluginA(CachingAutotunerPlugin):
            pass

        class _PluginB(CachingAutotunerPlugin):
            pass

        autotuner = self._make_autotuner([_PluginA(), _PluginB()])
        self.assertEqual(len(autotuner._plugins), 2)

        # State excludes the live plugin instances.
        state = autotuner.__getstate__()
        self.assertEqual(state["_plugins"], [])

        # __setstate__ re-invokes the factory; with no patch active the
        # default factory returns an empty list.
        revived = CachingAutotuner.__new__(CachingAutotuner)
        revived.__setstate__(state)
        self.assertEqual(revived._plugins, [])


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

        peak_mem_before = torch.get_device_module(GPU_TYPE).max_memory_allocated()
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
            lambda msg: f"{msg}\n{peak_mem_before=} v.s. {peak_mem_after=}",
        )

        # Avoid OOM in CI
        self.assertTrue(peak_mem_after < 1e10)

    @requires_gpu_with_enough_memory(1e10)
    def test_clone_contiguous_args(self):
        arg = self._create_tensor(pad=0)
        self.assertTrue(arg.is_contiguous())
        self.assertTrue(arg.storage_offset() == 0)
        self._do_test(arg)

    @requires_gpu_with_enough_memory(1e10)
    def test_clone_non_contiguous_args(self):
        arg = self._create_tensor(pad=1)
        self.assertFalse(arg.is_contiguous())
        self.assertTrue(arg.storage_offset() == 0)
        self._do_test(arg)

    @requires_gpu_with_enough_memory(1e10)
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
                        lambda msg: f"{msg}\nKernel {base_name} has more runs ({len(indices)}) than max ({max_runs})",
                    )

                    # Verify the indices are within [0, max_runs)
                    for idx in indices:
                        self.assertLess(
                            idx,
                            max_runs,
                            lambda msg: f"{msg}\nRun index {idx} exceeds max_runs-1 ({max_runs - 1})",
                        )

        finally:
            # Restore environment variable
            if old_dump_env is None:
                os.environ.pop("TORCHINDUCTOR_DUMP_LAUNCH_TENSORS", None)
            else:
                os.environ["TORCHINDUCTOR_DUMP_LAUNCH_TENSORS"] = old_dump_env


class TestRecheckAutotuneCache(TestCase):
    """Tests for CachingAutotuner.recheck_autotune_cache"""

    @staticmethod
    def _make_compile_result(cfg):
        """Create a mock StaticTritonCompileResult with the given config."""
        from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult

        result = MagicMock(spec=StaticTritonCompileResult)
        result.config = cfg
        return result

    @staticmethod
    def _make_autotuner_with_results(configs, compile_results):
        """
        Create a CachingAutotuner and inject compile_results directly,
        bypassing actual Triton compilation.
        """
        args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        args["configs"] = configs
        autotuner = CachingAutotuner(**args)
        autotuner.compile_results = compile_results
        return autotuner

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_recheck_single_config_enters_cache_hit_block(self):
        """
        When there is exactly 1 config and the autotune cache returns a hit,
        recheck_autotune_cache should narrow compile_results to that single
        matching result (not skip the block due to len(configs) == 1).
        """
        cfg = triton_config({"x": 16}, 64)
        cfg.found_by_coordesc = False
        compile_result = self._make_compile_result(cfg)

        autotuner = self._make_autotuner_with_results([cfg], [compile_result])

        # Cache returns the same config as the best
        cached_cfg = triton_config({"x": 16}, 64)
        cached_cfg.found_by_coordesc = True

        with patch(
            "torch._inductor.runtime.triton_heuristics.check_autotune_cache",
            return_value=([cached_cfg], None, {"autotune_cache_state": "hit"}),
        ):
            autotuner.recheck_autotune_cache(reload_kernel_from_src=MagicMock())

        # The compile_results should be narrowed to just the matching one
        self.assertEqual(len(autotuner.compile_results), 1)
        self.assertIs(autotuner.compile_results[0], compile_result)
        # And found_by_coordesc must be propagated
        self.assertTrue(autotuner.compile_results[0].config.found_by_coordesc)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_recheck_propagates_found_by_coordesc_true(self):
        """
        When the cached best config has found_by_coordesc=True,
        it must be propagated to the compile result's config.
        """
        cfg_a = triton_config({"x": 16}, 64)
        cfg_b = triton_config({"x": 256}, 64)
        cfg_a.found_by_coordesc = False
        cfg_b.found_by_coordesc = False
        result_a = self._make_compile_result(cfg_a)
        result_b = self._make_compile_result(cfg_b)

        autotuner = self._make_autotuner_with_results(
            [cfg_a, cfg_b], [result_a, result_b]
        )

        # Cache says cfg_b is the best, found via coordesc
        cached_cfg = triton_config({"x": 256}, 64)
        cached_cfg.found_by_coordesc = True

        with patch(
            "torch._inductor.runtime.triton_heuristics.check_autotune_cache",
            return_value=([cached_cfg], None, {"autotune_cache_state": "hit"}),
        ):
            autotuner.recheck_autotune_cache(reload_kernel_from_src=MagicMock())

        self.assertEqual(len(autotuner.compile_results), 1)
        self.assertIs(autotuner.compile_results[0], result_b)
        # The flag must be propagated from the cached config
        self.assertTrue(autotuner.compile_results[0].config.found_by_coordesc)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_recheck_propagates_found_by_coordesc_false(self):
        """
        When the cached best config has found_by_coordesc=False, it must be
        propagated so that coordinate descent can still run if enabled.
        """
        cfg = triton_config({"x": 16}, 64)
        cfg.found_by_coordesc = True
        compile_result = self._make_compile_result(cfg)

        autotuner = self._make_autotuner_with_results([cfg], [compile_result])

        cached_cfg = triton_config({"x": 16}, 64)
        cached_cfg.found_by_coordesc = False

        with patch(
            "torch._inductor.runtime.triton_heuristics.check_autotune_cache",
            return_value=([cached_cfg], None, {"autotune_cache_state": "hit"}),
        ):
            autotuner.recheck_autotune_cache(reload_kernel_from_src=MagicMock())

        self.assertEqual(len(autotuner.compile_results), 1)
        self.assertFalse(autotuner.compile_results[0].config.found_by_coordesc)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_recheck_no_cache_hit_leaves_results_unchanged(self):
        """
        When there's no autotune cache hit, compile_results should not change.
        """
        cfg_a = triton_config({"x": 16}, 64)
        cfg_b = triton_config({"x": 256}, 64)
        result_a = self._make_compile_result(cfg_a)
        result_b = self._make_compile_result(cfg_b)

        autotuner = self._make_autotuner_with_results(
            [cfg_a, cfg_b], [result_a, result_b]
        )

        # Cache returns no hit (empty list)
        with patch(
            "torch._inductor.runtime.triton_heuristics.check_autotune_cache",
            return_value=([], None, {"autotune_cache_state": "miss"}),
        ):
            autotuner.recheck_autotune_cache(reload_kernel_from_src=MagicMock())

        # Nothing should change
        self.assertEqual(len(autotuner.compile_results), 2)


@triton.jit
def hip_autotune_kernel(
    in_ptr,
    out_ptr,
    numel,
    XBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    mask = offsets < numel
    data = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, data * 2.0, mask=mask)


@functools.lru_cache
def get_hip_autotune_kernel_with_invalid_config():
    # num_warps=32 with HIP warp_size=64 = 2048 threads, exceeds 1024 limit
    return triton.autotune(
        configs=[
            triton.Config({"XBLOCK": 128}, num_warps=32, num_stages=1),  # invalid
            triton.Config({"XBLOCK": 256}, num_warps=1, num_stages=1),  # valid
        ],
        key=["numel"],
    )(hip_autotune_kernel)


class TestHIPInvalidConfigHandling(TestCase):
    @runOnRocm
    @skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_benchmark_returns_inf_on_invalid_config(self):
        from torch._inductor.runtime.benchmarking import TritonBenchmarker

        benchmarker = TritonBenchmarker()

        def failing_callable():
            raise RuntimeError(
                "Triton Error [HIP]: Code: 9, Message: invalid configuration argument"
            )

        result = benchmarker.benchmark(
            fn=failing_callable,
            device=GPU_TYPE,
            is_vetted_benchmarking=True,
        )
        self.assertEqual(result, float("inf"))

    @runOnRocm
    @skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_autotune_skips_invalid_hip_config_and_succeeds(self):
        numel = 1024 * 1024
        x = torch.randn(numel, device=GPU_TYPE, dtype=torch.float32)
        y = torch.empty_like(x)

        kernel = get_hip_autotune_kernel_with_invalid_config()

        def grid(meta):
            return (triton.cdiv(numel, meta["XBLOCK"]),)

        kernel[grid](x, y, numel)

        expected = x * 2.0
        torch.testing.assert_close(y, expected)


class TestGridExprMaximum(TestCase):
    def test_maximum_cpp_mode_casts_int_constants_to_long(self):
        from torch._inductor.runtime.triton_heuristics import Grid1D

        grid = Grid1D(inductor_meta={}, mode="cpp")
        # Mixed str/int: int constants must be cast to (long) for std::max
        result = grid.maximum(["ynumel_0", "ynumel_1", 4480])
        self.assertIn("(long)4480", result)
        self.assertIn("std::max", result)
        # All strings: no cast needed
        result = grid.maximum(["xnumel", "ynumel"])
        self.assertNotIn("(long)", result)
        # All ints: constant-folds
        self.assertEqual(grid.maximum([10, 20, 5]), 20)


@instantiate_parametrized_tests
class TestCappedGrid1D(TestCase):
    def test_grid_is_capped(self):
        for mode, x_grid in (
            ("python", "min(-((xnumel) // -(1024)), 66)"),
            ("cpp", "std::min<int64_t>(((xnumel + (1024 - 1)) / (1024)), 66)"),
        ):
            grid = CappedGrid1D(inductor_meta={"grid_cap": 66}, mode=mode)
            grid.generate({"XBLOCK": 1024})
            self.assertEqual((grid.x_grid, grid.y_grid, grid.z_grid), (x_grid, 1, 1))
        # GridExpr.minimum folds the int constants into one, after the other items
        nested = "std::min<int64_t>(std::min<int64_t>(a, b), 3)"
        self.assertEqual(grid.minimum(["a", "b", 3, 5]), nested)

    def test_symbolic_xnumel_gives_a_symbolic_grid(self):
        # The FX IR wrapper evaluates grids on symbolic sizes
        from torch.utils._sympy.functions import Min

        s0 = sympy.Symbol("s0", integer=True, positive=True)
        grid = CappedGrid1D(inductor_meta={"grid_cap": 66}, mode="python")
        grid.generate({"XBLOCK": 1024})
        x, y, z = grid.eval_slow({"xnumel": s0})
        self.assertIsInstance(x, Min)
        self.assertEqual([x.subs(s0, n) for n in (10, 10000, 1 << 24)], [1, 10, 66])
        self.assertEqual((y, z), (1, 1))

    @parametrize("grid_cap,block", [(0, "XBLOCK"), (None, "XBLOCK"), (44, "YBLOCK")])
    def test_invalid_grids_are_errors(self, grid_cap, block):
        grid = CappedGrid1D(inductor_meta={"grid_cap": grid_cap}, mode="python")
        with self.assertRaisesRegex(AssertionError, "at least 1, got|along x only"):
            grid.generate({"XBLOCK": 1024, block: 16})

    def test_from_meta_resolves_the_capped_grid(self):
        meta = {"grid_type": "CappedGrid1D", "grid_cap": 44}
        grid = GridExpr.from_meta(meta, {"XBLOCK": 512})
        self.assertEqual(grid.eval_slow({"xnumel": 1 << 20, "XBLOCK": 512}), (44, 1, 1))
        # A lazy grid's meta names every block; only resolved configs are checked
        lazy = GridExpr.from_meta_lazy(meta, "triton_poi_fused_0")
        xblock = "triton_poi_fused_0_result.xblocks[0]"
        self.assertEqual(
            lazy.x_grid,
            f"std::min<int64_t>(((xnumel + ({xblock} - 1)) / ({xblock})), 44)",
        )


# Configs, stock and capped, as outside ROCm, whatever the build
@patch.object(torch.version, "hip", None)
@instantiate_parametrized_tests
class TestCappedGridAutotuneConfigs(TestCase):
    _ALL = [(16384, 16), (16384, 32), (32768, 8), (32768, 16)]
    _F = [(8192, 32), (8192, 16), (8192, 8)]
    _OFF = {"autotune_pointwise": False}
    _X, _R = {"x": 1 << 24}, {"x": 128, "r0_": 32768}
    _TH = "torch._inductor.runtime.triton_heuristics"
    _cuda = TestTritonHeuristics._fake_cuda_device_properties()

    def _configs(
        self, size_hints, device=None, elements=0, types=(), threads=None, **meta
    ):
        size_hints = dict(size_hints)  # reduction() sets x to 1 for no_x_dim
        # A grid_cap gets its grid type unless the test sets one
        if "grid_cap" in meta:
            meta.setdefault("grid_type", "CappedGrid1D")
        device = device or self._cuda
        if threads:
            device = device._replace(max_threads_per_block=threads)
        signature = {f"arg{i}": arg_type for i, arg_type in enumerate(types)}
        common = {"inductor_meta": meta, "return_configs": True}
        common["triton_meta"] = {"device": device, "signature": signature}
        if "r0_" in size_hints:
            configs = reduction(size_hints, ReductionHint.INNER, **common)
        else:
            configs = pointwise(size_hints, min_elem_per_thread=elements, **common)
        return [(c.kwargs, c.num_warps, c.num_stages) for c in configs]

    @parametrize(
        "numel,elements,meta,expected",
        [
            # The table, of which grid_max_xblock keeps the blocks up to it
            (1 << 24, 0, {}, _ALL),
            (1 << 24, 0, {"grid_max_xblock": 16384}, _ALL[:2]),
            # Only the first without pointwise autotuning or max-autotune
            (1 << 24, 0, _OFF, _ALL[:1]),
            (1 << 24, 0, {**_OFF, "max_autotune": True}, _ALL),
            # A block per program, or 128 elements, or the rounded hint if smaller
            (1 << 20, 0, {}, _F),
            (32768, 0, {"grid_cap": 66}, [(128, 4)]),
            (1, 0, {}, [(1, 1)]),
            # The fallback: 4 elements per thread, then the table's smaller warps
            (1 << 24, 0, {"grid_max_xblock": 8192}, _F),
            (1 << 24, 0, {"grid_max_xblock": 1024}, [(1024, 8)]),
            # Raised for min_elem_per_thread; at the max block the warps drop instead
            (1 << 24, 16, {"grid_max_xblock": 8192}, [(8192, 16), (8192, 8)]),
            (1 << 24, 4, {"grid_max_xblock": 64}, [(512, 4)]),
            (100, 4, {}, [(128, 4)]),
            (1 << 17, 5, {}, [(2048, 8)]),
            # A dynamic xnumel's size hint bounds nothing
            (1, 0, {"grid_dynamic_xnumel": True}, _ALL),
            (1 << 16, 4, {"grid_dynamic_xnumel": True}, _ALL),
            # Warps within the device's thread limit, in powers of 2
            (1 << 24, 0, {"threads": 768}, [(16384, 16), (32768, 8), (32768, 16)]),
        ],
    )
    def test_pointwise_candidates(self, numel, elements, meta, expected):
        meta = {"grid_cap": 44, **meta}
        configs = self._configs({"x": numel}, elements=elements, **meta)
        self.assertEqual(configs, [({"XBLOCK": x}, w, 1) for x, w in expected])

    @parametrize(
        "types,expected",
        [
            (("*bf16", "*fp16", "i32"), _ALL),
            (("*fp32", "*fp32"), _ALL[:2]),
            (("*fp8e4nv",) * 4, _ALL),
            (("tensordesc<bf16[64, 64]>", "*fp16"), _ALL),
            (("nvTmaDesc", "*"), _ALL[:2]),
            (("*fp32",) * 4, _F),
        ],
    )
    def test_candidates_fit_the_bytes_per_block(self, types, expected):
        # At most 128 KiB of the kernel's tensors per block: 4 bytes per element fill
        # XBLOCK 32,768; descriptors count their dtype, or 4 bytes
        meta = {"grid_cap": 44, "grid_max_xblock": 65536}
        configs = self._configs(self._X, types=types, **meta)
        self.assertEqual(configs, [({"XBLOCK": x}, w, 1) for x, w in expected])

    def test_candidates_short_of_min_elem_per_thread_are_dropped(self):
        # The byte budget leaves 1,024 and 256 elements, short of min_elem_per_thread
        # on 4 warps, which fp8 conversions need: get_configs' configs apply
        for numel, grid_cap, count, elements, dynamic in (
            (1 << 24, 44, 20, 16, False),
            (1 << 24, 44, 20, 16, True),
            (1 << 20, 132, 65, 4, False),
        ):
            kwargs = {"types": ("*fp32",) * count, "elements": elements}
            stock = self._configs({"x": numel}, **kwargs)
            meta = {"grid_cap": grid_cap, "grid_dynamic_xnumel": dynamic}
            self.assertEqual(self._configs({"x": numel}, **kwargs, **meta), stock)

    def test_stock_configs_are_kept(self):
        # Outside capped_grid_device_types, for reductions and without the hook
        for device_type in ("hip", "xpu"):
            device = self._cuda._replace(type=device_type)
            capped = self._configs(self._X, device, grid_cap=44)
            self.assertEqual(capped, self._configs(self._X, device))
        self.assertEqual(self._configs(self._R, grid_cap=66), self._configs(self._R))

        class _Bare(CodegenConfigHeuristics):
            def get_configs(self, *args, **kwargs):
                return PointwiseHeuristic().get_configs(*args, **kwargs)

        lookup = "torch._inductor.heuristics.registry.get_codegen_heuristic"
        with patch(lookup, return_value=_Bare()):
            capped = self._configs(self._X, grid_cap=44)
            self.assertEqual(capped, self._configs(self._X))

    def test_meta_is_checked_before_the_device_hook(self):
        y_hints, r_y_hints = {"y": 1024, "x": 1024}, {"x": 128, "y": 8, "r0_": 32768}
        fail = {"side_effect": RuntimeError}
        with (
            patch.object(PointwiseHeuristic, "get_capped_grid_configs", **fail),
            patch(f"{self._TH}._reduction_configs", **fail),
        ):
            self.assertRaises(RuntimeError, self._configs, self._X, grid_cap=44)
            self.assertRaises(RuntimeError, self._configs, self._R, grid_cap=66)
            for size_hints, meta, message in (
                (self._X, {"grid_cap": 44, "grid_type": "Grid1D"}, "needs grid_type"),
                (self._X, {"grid_type": "CappedGrid1D"}, "grid_cap, .*got None"),
                (self._X, {"grid_cap": 0}, "at least 1, got 0"),
                (self._R, {"grid_cap": True}, "an int .*got True"),
                (y_hints, {"grid_cap": 44}, "along x only"),
                (r_y_hints, {"grid_cap": 44}, "along x only"),
                (self._R, {"grid_cap": 66, "no_x_dim": True}, "got no_x_dim"),
                (self._X, {"grid_cap": 44, "grid_max_xblock": 3000}, "power of 2"),
                (self._R, {"grid_cap": 66, "grid_max_xblock": 4096.0}, "power of 2"),
            ):
                with self.subTest(size_hints=size_hints, **meta):
                    with self.assertRaisesRegex(AssertionError, message):
                        self._configs(size_hints, **meta)

    def test_cached_autotune_checks_capped_meta(self):
        # Other heuristic types, combo kernels (no size hints), and cooperative and
        # mix-order (RSPLIT_SIZE) reductions are rejected
        def autotune(heuristic_type, size_hints, triton_meta, meta):
            return cached_autotune(
                size_hints,
                [Config({"XBLOCK": 64, "R0_BLOCK": 64}, num_warps=4, num_stages=1)],
                triton_meta={"device": self._cuda, **triton_meta},
                heuristic_type=heuristic_type,
                inductor_meta={"grid_type": "CappedGrid1D", "grid_cap": 44, **meta},
            )

        rows = {"x": 64, "r0_": 64}
        for args in (
            (HeuristicType.PERSISTENT_REDUCTION, rows, {}, {}),
            (HeuristicType.TEMPLATE, None, {}, {}),
            (HeuristicType.POINTWISE, None, {}, {}),
            (HeuristicType.REDUCTION, rows, {"launch_cooperative_grid": True}, {}),
            (HeuristicType.REDUCTION, rows, {}, {"RSPLIT_SIZE": 8}),
        ):
            with self.subTest(args=args):
                with self.assertRaisesRegex(AssertionError, "only for pointwise"):
                    autotune(*args)
        autotune(HeuristicType.POINTWISE, rows, {}, {})
        autotune(HeuristicType.REDUCTION, rows, {}, {})
        with self.assertRaisesRegex(AssertionError, "needs grid_type CappedGrid1D"):
            autotune(HeuristicType.POINTWISE, rows, {}, {"grid_type": "Grid1D"})

    def test_every_config_of_a_capped_kernel_is_checked(self):
        # Also get_configs' and the TMA filter's: XBLOCK 65,536 divides neither the
        # default max block nor TRITON_MAX_BLOCK["X"], which uncapped kernels keep
        wide = Config({"XBLOCK": 65536}, num_warps=8, num_stages=1)
        wide_r = Config({"XBLOCK": 65536, "R0_BLOCK": 8}, num_warps=8, num_stages=1)
        rows = {"x": 1 << 20, "r0_": 32768}
        with (
            patch.object(PointwiseHeuristic, "capped_grid_device_types", ()),
            patch.object(PointwiseHeuristic, "get_configs", return_value=[wide]),
            patch(f"{self._TH}._reduction_configs", return_value=[wide_r]),
        ):
            for size_hints in (self._X, rows):
                self.assertEqual(len(self._configs(size_hints)), 1)
                with self.assertRaisesRegex(AssertionError, "must divide"):
                    self._configs(size_hints, grid_cap=44)
        tma = {"tma_min_block_sizes": {"XBLOCK": 65536}}
        for size_hints in (self._X, rows):
            with self.assertRaisesRegex(AssertionError, "must divide"):
                self._configs(size_hints, grid_cap=44, **tma)
        # A device hook's configs
        with patch.object(PointwiseHeuristic, "get_capped_grid_configs") as hook:
            for xblock, numel, message in (
                (3072, 1 << 24, "must divide"),
                (0, 1 << 24, "at least 1"),
                (2, 1, "one x element needs XBLOCK 1"),
            ):
                hook.return_value = [Config({"XBLOCK": xblock}, num_warps=1)]
                with self.assertRaisesRegex(AssertionError, message):
                    self._configs({"x": numel}, grid_cap=44)

    def test_capped_reduction_blocks_keep_the_stock_limits(self):
        # XBLOCK divides TRITON_MAX_BLOCK["X"] whatever grid_max_xblock (a pointwise
        # kernel's max block grows with it); R0_BLOCK within TRITON_MAX_BLOCK; a block
        # within TRITON_MAX_TENSOR_NUMEL elements
        x_max, r0_max = TRITON_MAX_BLOCK["X"], TRITON_MAX_BLOCK["R0_"]
        edge = {"XBLOCK": x_max, "R0_BLOCK": TRITON_MAX_TENSOR_NUMEL // x_max}
        capped = {"grid_type": "CappedGrid1D", "grid_cap": 66}
        wide = {**capped, "grid_max_xblock": 65536}
        for blocks, message in (
            ({"XBLOCK": 2 * x_max, "R0_BLOCK": 1}, "must divide"),
            ({"XBLOCK": 1, "R0_BLOCK": 2 * r0_max}, "'R0_BLOCK' too large"),
            ({**edge, "R0_BLOCK": 2 * edge["R0_BLOCK"]}, "TRITON_MAX_TENSOR_NUMEL"),
        ):
            with self.assertRaisesRegex(AssertionError, message):
                check_capped_grid_config(Config(blocks, num_warps=8), wide, self._R)
        check_capped_grid_config(Config(edge, num_warps=8), wide, self._R)
        check_capped_grid_config(Config({"XBLOCK": 65536}, num_warps=8), wide, self._X)

    @parametrize(
        "size_hints,xblock,larger_is_faster",
        [
            # Above TRITON_MAX_BLOCK["X"], coordinate descent only halves XBLOCK
            (_X, 32768, True),
            (_X, 16384, False),
            # Below it, XBLOCK rises up to it, the max block of a looped reduction
            (_X, 1024, True),
            ({"x": 1 << 20, "r0_": 64}, 64, True),
        ],
    )
    def test_coordinate_descent_keeps_capped_configs_valid(
        self, size_hints, xblock, larger_is_faster
    ):
        # Every config it times passes the capped check (warp_size and
        # max_threads_per_block spare it a device query)
        meta = {"grid_type": "CappedGrid1D", "grid_cap": 44, "grid_max_xblock": 65536}
        meta.update(warp_size=32, max_threads_per_block=1024)
        blocks = {"XBLOCK": xblock}
        if "r0_" in size_hints:
            blocks["R0_BLOCK"] = 8
        tried = []

        def bench(cfg):
            tried.append(cfg)
            block = cfg.kwargs["XBLOCK"]
            return 1 / block if larger_is_faster else block

        tuner = CoordescTuner(size_hints=size_hints, inductor_meta=meta)
        tuner.autotune(bench, Config(blocks, num_warps=8))
        for cfg in tried:
            check_capped_grid_config(cfg, meta, size_hints)
        top = max(xblock, TRITON_MAX_BLOCK["X"])
        self.assertEqual(max(cfg.kwargs["XBLOCK"] for cfg in tried), top)

    def test_lookup_table_pins_of_capped_kernels_are_checked(self):
        # A failing pin is ignored with one warning per pin and reason
        _warn_ignored_pin.cache_clear()
        fn = types.SimpleNamespace(
            src="def triton_poi_fused_0(): pass", __name__="triton_poi_fused_0"
        )
        capped = {"grid_type": "CappedGrid1D", "grid_cap": 44}

        def pinned(blocks, inductor_meta, size_hints=None):
            size_hints = size_hints or self._X
            fn_hash = generate_lookup_hash_from_source_code(str(size_hints), fn.src)
            pin = {**blocks, "num_warps": 8, "num_stages": 1}
            with config.patch(autotune_lookup_table={fn_hash: pin}):
                return lookup_autotune_config(size_hints, fn, inductor_meta)

        # XBLOCK 65,536 does not divide the default max block
        with self.assertLogs(self._TH, level="WARNING") as logs:
            self.assertIsNone(pinned({"XBLOCK": 65536}, capped))
        self.assertIn("'num_stages': 1} of triton_poi_fused_0", logs.output[0])
        with self.assertNoLogs(self._TH, level="WARNING"):
            self.assertIsNone(pinned({"XBLOCK": 65536}, capped))
        self.assertEqual(pinned({"XBLOCK": 65536}, {}).kwargs, {"XBLOCK": 65536})
        self.assertEqual(pinned({"XBLOCK": 16384}, capped).kwargs, {"XBLOCK": 16384})
        rows = {"x": 1 << 20, "r0_": 32768}
        for blocks, size_hints in (
            ({"XBLOCK": 16384.0}, None),
            ({"XBLOCK": 1024, "YBLOCK": 2}, None),
            ({"XBLOCK": 1024}, rows),
            ({"XBLOCK": TRITON_MAX_BLOCK["X"], "R0_BLOCK": 512}, rows),
        ):
            with self.assertLogs(self._TH, level="WARNING"):
                self.assertIsNone(pinned(blocks, capped, size_hints))
        # A meta error raises, as in pointwise() and reduction()
        with self.assertRaisesRegex(AssertionError, "power of 2"):
            pinned({"XBLOCK": 16384}, {**capped, "grid_max_xblock": 3000})

    @parametrize(
        "meta,r0_blocks",
        [
            ({}, [4096]),
            ({"grid_type": "CappedGrid1D", "grid_cap": 66}, []),
            ({"grid_type": "CappedGrid1D", "grid_cap": 1000}, [4096]),
            ({"grid_cap": 66}, [4096]),
        ],
    )
    def test_rblock_scaling_counts_the_capped_programs(self, meta, r0_blocks):
        # 64 registers per thread on 16 warps: 2 programs per SM, 264 on 132 SMs, so
        # a smaller R0_BLOCK cannot raise the occupancy of 66 programs
        autotuner = CachingAutotuner.__new__(CachingAutotuner)
        autotuner.device_props = self._cuda._replace(multi_processor_count=132)
        autotuner.size_hints = {"x": 4096, "r0_": 65536}
        autotuner.inductor_meta = meta
        result = MagicMock()
        blocks = {"XBLOCK": 1, "R0_BLOCK": 8192}
        result.config = Config(blocks, num_warps=16, num_stages=1)
        result.kernel.n_regs = 64
        autotuner.compile_results = [result]
        autotuner._ensure_kernel_loaded = lambda: None
        candidates = autotuner._iter_rblock_scale_candidates()
        self.assertEqual([c.kwargs["R0_BLOCK"] for c in candidates], r0_blocks)


@skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
class TestCappedGridLaunch(TestCase):
    @skipUnless(GPU_TYPE == "cuda" and not torch.version.hip, "candidates for CUDA")
    def test_pointwise_candidates_run_through_the_python_launcher(self):
        # Each candidate for a bf16-to-fp16 cast, alone in its autotuner, casts every
        # element on at most grid_cap programs; each program stores the count it saw
        from torch._dynamo.device_interface import get_interface_for_device

        @triton.jit
        def triton_(in_ptr0, out_ptr0, programs_ptr0, xnumel, XBLOCK: tl.constexpr):
            for x_block in tl.range(
                tl.program_id(0), (xnumel - 1) // XBLOCK + 1, tl.num_programs(0)
            ):
                xindex = x_block * XBLOCK + tl.arange(0, XBLOCK)[:]
                xmask = xindex < xnumel
                tmp0 = tl.load(in_ptr0 + xindex, xmask).to(tl.float32)
                tl.store(out_ptr0 + xindex, (tmp0 * 2.0).to(tl.float16), xmask)
            tl.store(programs_ptr0 + tl.program_id(0), tl.num_programs(0))

        device = DeviceProperties.create(torch.device(GPU_TYPE))
        meta = {"grid_type": "CappedGrid1D", "grid_cap": 4, "grid_max_xblock": 65536}
        # The cast's own pointers, 4 bytes per element, fill the budget at 32,768
        signature = {"in_ptr0": "*bf16", "out_ptr0": "*fp16"}
        triton_meta = {"device": device, "signature": signature, "constants": {}}
        configs = pointwise(
            {"x": 1 << 24}, triton_meta, inductor_meta=meta, return_configs=True
        )
        self.assertEqual(max(cfg.kwargs["XBLOCK"] for cfg in configs), 32768)
        signature |= {"programs_ptr0": "*i32", "xnumel": "i32", "XBLOCK": "constexpr"}
        wrapper = AttrsDescriptorWrapper(divisible_by_16=(0, 1, 2), equal_to_1=())
        triton_meta["configs"] = [wrapper]
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        for cfg in configs:
            autotuner = CachingAutotuner(
                fn=triton_,
                triton_meta={**triton_meta, "constants": {}},
                configs=[cfg],
                save_cache_hook=False,
                mutated_arg_names=[],
                reset_to_zero_arg_names=[],
                optimize_mem=True,
                heuristic_type=HeuristicType.POINTWISE,
                inductor_meta=meta,
            )
            xblock = cfg.kwargs["XBLOCK"]
            # bf16 values from 1 to 2, which convert to fp16 exactly when doubled
            for xnumel in (4 * 65536 + 5, xblock + 3):
                x = (torch.rand(xnumel, device=GPU_TYPE) + 1).to(torch.bfloat16)
                out = torch.zeros(xnumel, device=GPU_TYPE, dtype=torch.float16)
                # one slot per block, so a broken cap shows as extra programs
                blocks = triton.cdiv(xnumel, xblock)
                seen = torch.zeros(blocks, device=GPU_TYPE, dtype=torch.int32)
                autotuner.run(x, out, seen, xnumel, stream=stream)
                device_interface.synchronize()
                self.assertEqual(out, (x.float() * 2).half(), atol=0, rtol=0)
                programs = min(blocks, 4)
                expected = [programs] * programs + [0] * (blocks - programs)
                self.assertEqual(seen.tolist(), expected)


class TestGrid2DWithYZOverflowZeroYnumel(TestCase):
    """Regression test for https://github.com/pytorch/pytorch/issues/178530"""

    def test_grid2d_yz_overflow_zero_ynumel_python(self):
        from torch._inductor.runtime.triton_heuristics import Grid2DWithYZOverflow

        grid = Grid2DWithYZOverflow(inductor_meta={}, mode="python")
        grid.generate({"XBLOCK": 128, "YBLOCK": 128})
        # ynumel=0 must not raise ZeroDivisionError
        x, y, z = grid.eval_slow(
            {"xnumel": 256, "ynumel": 0, "XBLOCK": 128, "YBLOCK": 128}
        )
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_grid2d_yz_overflow_zero_ynumel_cpp(self):
        from torch._inductor.runtime.triton_heuristics import Grid2DWithYZOverflow

        grid = Grid2DWithYZOverflow(inductor_meta={}, mode="cpp")
        grid.generate({"XBLOCK": 128, "YBLOCK": 128})
        # cpp mode: the generated expression should contain a zero-guard
        self.assertIn("== 0", str(grid.y_grid))

    def test_grid2d_yz_overflow_nonzero_ynumel_unchanged(self):
        from torch._inductor.runtime.triton_heuristics import Grid2DWithYZOverflow

        grid = Grid2DWithYZOverflow(inductor_meta={}, mode="python")
        grid.generate({"XBLOCK": 128, "YBLOCK": 128})
        # Normal case: ynumel > 0 still works correctly
        x, y, z = grid.eval_slow(
            {"xnumel": 256, "ynumel": 256, "XBLOCK": 128, "YBLOCK": 128}
        )
        self.assertEqual(x, 2)
        self.assertEqual(y, 2)
        self.assertEqual(z, 1)

    def test_grid2d_yz_overflow_large_ynumel(self):
        from torch._inductor.runtime.triton_heuristics import Grid2DWithYZOverflow

        grid = Grid2DWithYZOverflow(inductor_meta={}, mode="python")
        grid.generate({"XBLOCK": 128, "YBLOCK": 128})
        # Large ynumel that requires overflow splitting across y and z
        x, y, z = grid.eval_slow(
            {"xnumel": 128, "ynumel": 128 * 131070, "XBLOCK": 128, "YBLOCK": 128}
        )
        self.assertEqual(x, 1)
        # y * z must cover all y blocks
        self.assertGreaterEqual(y * z, 131070)


class TestFastLauncherDeviceSupport(TestCase):
    @staticmethod
    def _make_autotuner(device_type):
        def triton_():
            pass

        device = DeviceProperties(
            type=device_type,
            index=0,
            multi_processor_count=1,
            cc=0,
            max_threads_per_block=1024,
            warp_size=32,
        )
        triton_meta = {
            "signature": {},
            "device": device,
            "constants": {},
            "configs": [],
        }
        return CachingAutotuner(
            fn=triton_,
            triton_meta=triton_meta,
            configs=[triton_config({"x": 1}, 1, warp_size=device.warp_size_or_default)],
            save_cache_hook=False,
            mutated_arg_names=[],
            reset_to_zero_arg_names=[],
            optimize_mem=False,
            heuristic_type=HeuristicType.POINTWISE,
            inductor_meta={"use_fast_triton_launcher": True},
        )

    @staticmethod
    def _make_static_launcher():
        class Kernel:
            function = 1
            num_warps = 4
            shared = 0
            arg_tys = ""
            has_global_scratch = False
            has_profile_scratch = False

            def run(self):
                pass

        def launcher_template():
            pass

        kernel = Kernel()
        launcher = types.FunctionType(
            launcher_template.__code__, {"runner": kernel.run}, "launcher"
        )
        launcher._is_static = True
        return launcher

    def test_fast_launcher_not_built_for_xpu(self):
        autotuner = self._make_autotuner("xpu")
        launcher = self._make_static_launcher()

        with patch("torch._C._FastCudaLauncher", create=True) as fast_launcher:
            self.assertIsNone(autotuner._build_fast_launcher(launcher))
            fast_launcher.assert_not_called()


class TestDynamicScaleRblockCacheInteraction(TestCase):
    """Tests for _dynamic_scale_rblock + autotune cache interaction.

    _dynamic_scale_rblock can add configs that aren't in the original
    config list. When such a config wins autotuning and gets cached,
    the warm-start path must still be able to load it from the cache
    and skip re-autotuning.
    """

    @staticmethod
    def _make_compile_result(cfg):
        from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult

        result = MagicMock(spec=StaticTritonCompileResult)
        result.config = cfg
        return result

    @staticmethod
    def _make_autotuner_with_results(configs, compile_results):
        args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        args["configs"] = configs
        autotuner = CachingAutotuner(**args)
        autotuner.compile_results = compile_results
        return autotuner

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_dynamic_scale_rblock_skipped_on_cache_hit(self):
        """
        _dynamic_scale_rblock must be a no-op when the autotune cache
        already determined the best config (cache_state == 'hit').
        """
        cfg = triton_config({"x": 16}, 64)
        result = self._make_compile_result(cfg)
        autotuner = self._make_autotuner_with_results([cfg], [result])
        autotuner.autotune_cache_info = {"autotune_cache_state": "hit"}

        n_before = len(autotuner.compile_results)
        autotuner._dynamic_scale_rblock()
        n_after = len(autotuner.compile_results)

        self.assertEqual(n_before, n_after)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_dynamic_scale_rblock_runs_on_cache_miss(self):
        """
        _dynamic_scale_rblock should not be skipped when the autotune
        cache missed — this is a cold-start scenario.
        """
        cfg = triton_config({"x": 16}, 64)
        result = self._make_compile_result(cfg)
        autotuner = self._make_autotuner_with_results([cfg], [result])
        autotuner.autotune_cache_info = {"autotune_cache_state": "miss"}

        autotuner._dynamic_scale_rblock()

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_dynamic_scale_rblock_runs_when_cache_info_none(self):
        """
        _dynamic_scale_rblock should not be skipped when
        autotune_cache_info is None (e.g. loaded from old pickle).
        """
        cfg = triton_config({"x": 16}, 64)
        result = self._make_compile_result(cfg)
        autotuner = self._make_autotuner_with_results([cfg], [result])
        autotuner.autotune_cache_info = None

        autotuner._dynamic_scale_rblock()

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_load_cached_autotuning_reconstructs_unknown_config(self):
        """
        When the cached best config is not in the original configs list
        (e.g. it was added by _dynamic_scale_rblock), _load_cached_autotuning
        must reconstruct it from the cached data instead of returning None.
        """
        from torch._inductor.runtime.autotune_cache import _load_cached_autotuning
        from torch._inductor.runtime.triton_heuristics import hash_configs

        original_configs = [
            triton_config({"x": 1}, 16, num_stages=1),
            triton_config({"x": 8}, 4, num_stages=1),
        ]
        configs_hash = hash_configs(original_configs)

        best_config_data = {
            "XBLOCK": 1,
            "R0_BLOCK": 1024,
            "num_warps": 8,
            "num_stages": 1,
            "configs_hash": configs_hash,
            "found_by_coordesc": False,
            "time_taken_ms": 42,
        }

        result = _load_cached_autotuning(
            best_config_data, configs_hash, original_configs, {}
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.kwargs["XBLOCK"], 1)
        self.assertEqual(result.kwargs["R0_BLOCK"], 1024)
        self.assertEqual(result.num_warps, 8)
        self.assertEqual(result.num_stages, 1)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_load_cached_autotuning_still_matches_known_config(self):
        """
        When the cached best config IS in the original configs list,
        _load_cached_autotuning must return that same config object.
        """
        from torch._inductor.runtime.autotune_cache import _load_cached_autotuning
        from torch._inductor.runtime.triton_heuristics import hash_configs

        cfg_a = triton_config({"x": 1}, 16, num_stages=1)
        cfg_b = triton_config({"x": 8}, 4, num_stages=1)
        original_configs = [cfg_a, cfg_b]
        configs_hash = hash_configs(original_configs)

        best_config_data = {
            **cfg_b.kwargs,
            "num_warps": cfg_b.num_warps,
            "num_stages": cfg_b.num_stages,
            "configs_hash": configs_hash,
            "found_by_coordesc": False,
            "time_taken_ms": 42,
        }

        result = _load_cached_autotuning(
            best_config_data, configs_hash, original_configs, {}
        )

        self.assertIsNotNone(result)
        self.assertIs(result, cfg_b)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_autotune_cache_round_trips_maxnreg(self):
        """
        Two configs that differ only in maxnreg must survive save + load as
        themselves: the cached winner must come back capped, not as the
        uncapped config with the same warps/stages.
        """
        from torch._inductor.runtime.autotune_cache import (
            _load_cached_autotuning,
            AutotuneCache,
            AutotuneCacheBundler,
        )
        from torch._inductor.runtime.triton_heuristics import hash_configs

        uncapped = triton.Config({"x": 1}, num_warps=4, num_stages=2)
        capped = triton.Config({"x": 1}, num_warps=4, num_stages=2, maxnreg=64)
        original_configs = [uncapped, capped]
        configs_hash = hash_configs(original_configs)

        cache = AutotuneCache(configs_hash)
        local = MagicMock()
        cache.local_cache = (local, "key")
        with patch.object(AutotuneCacheBundler, "put"):
            cache.save(capped, time_taken_ns=1_000_000)
        saved = dict(local.put.call_args[0][1])
        self.assertEqual(saved.get("maxnreg"), 64)

        result = _load_cached_autotuning(saved, configs_hash, original_configs, {})
        self.assertIs(result, capped)
        self.assertNotIn("maxnreg", result.kwargs)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_load_cached_autotuning_rejects_hash_mismatch(self):
        """
        When configs_hash doesn't match, _load_cached_autotuning must
        return None regardless of whether the config matches.
        """
        from torch._inductor.runtime.autotune_cache import _load_cached_autotuning

        original_configs = [triton_config({"x": 1}, 16, num_stages=1)]

        best_config_data = {
            "XBLOCK": 1,
            "num_warps": 1,
            "num_stages": 1,
            "configs_hash": "wrong_hash",
            "found_by_coordesc": False,
        }

        result = _load_cached_autotuning(
            best_config_data, "correct_hash", original_configs, {}
        )

        self.assertIsNone(result)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_recheck_autotune_cache_compiles_dynamic_config(self):
        """
        When the cached best config was added by _dynamic_scale_rblock
        and isn't in compile_results, recheck_autotune_cache must compile
        it (not silently drop it, leaving all original configs active).
        """
        from triton import Config

        cfg_a = Config({"XBLOCK": 1, "R0_BLOCK": 2048}, num_warps=16, num_stages=1)
        cfg_b = Config({"XBLOCK": 8, "R0_BLOCK": 512}, num_warps=4, num_stages=1)
        result_a = self._make_compile_result(cfg_a)
        result_b = self._make_compile_result(cfg_b)

        autotuner = self._make_autotuner_with_results(
            [cfg_a, cfg_b], [result_a, result_b]
        )

        dynamic_cfg = Config(
            {"XBLOCK": 1, "R0_BLOCK": 1024}, num_warps=16, num_stages=1
        )

        mock_precompile = MagicMock(return_value=self._make_compile_result(dynamic_cfg))
        autotuner._precompile_config = mock_precompile

        reload_fn = MagicMock()
        reload_fn.return_value.fn = MagicMock()

        with patch(
            "torch._inductor.runtime.triton_heuristics.check_autotune_cache",
            return_value=([dynamic_cfg], None, {"autotune_cache_state": "hit"}),
        ):
            autotuner.recheck_autotune_cache(reload_kernel_from_src=reload_fn)

        self.assertEqual(len(autotuner.compile_results), 1)
        mock_precompile.assert_called_once_with(dynamic_cfg)


@skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
class TestCheckLauncherCallArgs(TestCase):
    """Unit tests for CachingAutotuner._check_launcher_call_args.

    These exercise the arg-mismatch error path without Triton compilation,
    but the shared cos-kernel fixture still builds a CachingAutotuner (which
    resolves DeviceProperties for GPU_TYPE), so they require a GPU.
    """

    def _make_autotuner(self):
        """Return a CachingAutotuner configured with the cos kernel fixture."""
        args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        return CachingAutotuner(**args)

    def _make_launcher(self, n_positional: int):
        """Return a minimal callable that looks like a generated launcher.

        The only attribute _check_launcher_call_args inspects is
        ``_expected_positional_count``, so we don't need real Triton machinery.
        """

        def launcher(*args, stream=None):
            pass

        launcher._expected_positional_count = n_positional
        return launcher

    def test_correct_arg_count_passes_silently(self):
        """No exception when args exactly matches _expected_positional_count."""
        autotuner = self._make_autotuner()
        launcher = self._make_launcher(n_positional=3)
        # Pass exactly 3 positional args - should be a no-op.
        autotuner._check_launcher_call_args(launcher, (1, 2, 3))

    def test_too_many_args_raises_type_error(self):
        """TypeError with a helpful message when too many positional args are passed."""
        autotuner = self._make_autotuner()
        launcher = self._make_launcher(n_positional=3)
        # Simulate the 'stream' being passed positionally (4 args, expected 3).
        with self.assertRaises(TypeError) as cm:
            autotuner._check_launcher_call_args(launcher, (1, 2, 3, 4))
        self.assertIn("stream", str(cm.exception).lower())
        self.assertIn("keyword", str(cm.exception).lower())

    def test_error_message_includes_kernel_name(self):
        """The TypeError message should include the kernel name from inductor_meta."""
        args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        args["inductor_meta"] = {
            "kernel_name": "my_custom_kernel",
            "grid_type": "Grid1D",
        }
        autotuner = CachingAutotuner(**args)
        launcher = self._make_launcher(n_positional=2)
        with self.assertRaises(TypeError) as cm:
            autotuner._check_launcher_call_args(launcher, (1, 2, 3))
        self.assertIn("my_custom_kernel", str(cm.exception))

    def test_launcher_without_expected_count_is_skipped(self):
        """Launchers without the generated metadata are skipped gracefully."""

        def raw_launcher(*args, stream=None):
            pass

        # No _expected_positional_count attribute set.
        autotuner = self._make_autotuner()
        # Should not raise, even with many args.
        autotuner._check_launcher_call_args(raw_launcher, (1, 2, 3, 4, 5))


class TestWarpSizeUnification(TestCase):
    """Tests for the unified warp_size threading through config helpers."""

    def test_warp_size_or_default(self):
        none_props = DeviceProperties(
            type="cuda", index=0, multi_processor_count=80, cc=80, warp_size=None
        )
        with self.assertRaisesRegex(
            RuntimeError, "cuda device properties must report warp_size"
        ):
            none_props.warp_size_or_default

        cpu_props = DeviceProperties(
            type="cpu", index=0, multi_processor_count=80, cc=80, warp_size=None
        )
        self.assertEqual(cpu_props.warp_size_or_default, 32)

        w32 = DeviceProperties(
            type="cuda", index=0, multi_processor_count=80, cc=80, warp_size=32
        )
        self.assertEqual(w32.warp_size_or_default, 32)

        w64 = DeviceProperties(
            type="hip", index=0, multi_processor_count=80, cc=80, warp_size=64
        )
        self.assertEqual(w64.warp_size_or_default, 64)

    def test_num_warps_halves_on_wave64(self):
        # wave64 (AMD CDNA/gfx9) halves the range so total threads match wave32.
        self.assertEqual(_num_warps(8, max_num_warps=8, warp_size=64), 4)

    def test_num_warps_does_not_halve_on_wave32(self):
        # wave32 (NVIDIA and AMD RDNA) must not halve.
        self.assertEqual(_num_warps(8, max_num_warps=8, warp_size=32), 8)
        # Default warp_size is 32; confirm the default path matches wave32.
        self.assertEqual(_num_warps(8, max_num_warps=8), 8)

    def test_check_max_grid_x_respects_warp_size_hip(self):
        # _check_max_grid_x only uses warp_size on the HIP path, where the
        # bound is on total threads (num_blocks * num_warps * warp_size).
        # Force the HIP branch so the assertion exercises that path even on
        # a non-HIP build.
        size_hints = {"x": 2**32}
        with patch.object(torch.version, "hip", "6.0.0"):
            x32, _ = _check_max_grid_x(size_hints, x=1, num_warps=8, warp_size=32)
            x64, _ = _check_max_grid_x(size_hints, x=1, num_warps=8, warp_size=64)
        # Doubling warp_size halves the per-block thread budget, so x must
        # double to keep total threads under 2**31 - 1.
        self.assertEqual(x64, 2 * x32)
        self.assertLessEqual(x32, TRITON_MAX_BLOCK["X"])
        self.assertLessEqual(x64, TRITON_MAX_BLOCK["X"])

    def test_check_max_grid_x_ignores_warp_size_on_nvidia(self):
        # NVIDIA bounds num_blocks directly, so warp_size must not change x.
        size_hints = {"x": 2**32}
        with patch.object(torch.version, "hip", None):
            x32, _ = _check_max_grid_x(size_hints, x=1, num_warps=8, warp_size=32)
            x64, _ = _check_max_grid_x(size_hints, x=1, num_warps=8, warp_size=64)
        self.assertEqual(x32, x64)

    def test_triton_config_uses_warp_size_for_min_elem(self):
        # min_elem_per_thread * warp_size * num_warps drives the floor for
        # block_size; doubling warp_size should at least double the XBLOCK.
        size_hints = {"x": 4096}
        cfg32 = triton_config(
            size_hints, 128, num_warps=1, min_elem_per_thread=4, warp_size=32
        )
        cfg64 = triton_config(
            size_hints, 128, num_warps=1, min_elem_per_thread=4, warp_size=64
        )
        self.assertEqual(cfg64.kwargs["XBLOCK"], 2 * cfg32.kwargs["XBLOCK"])


class TestMakeLaunchersMemory(TestCase):
    def test_failed_config_exception_not_retained(self):
        """Regression (D107597017 / PR #184285): CachingAutotuner._make_launchers
        must not retain the failed-config build exception past its return. That
        exception's traceback pins its frame chain -- and via those frames' f_back
        the benchmarking callers up to do_bench, whose 256MB Triton L2-flush buffer
        then leaks one-per-autotuned-kernel until cyclic GC (which never runs under
        gc.disable(), as APS training does). Pure reference-cycle check, no GPU: a
        sentinel standing in for the L2 buffer, held only by a frame that calls
        _make_launchers, must be freed by refcount once _make_launchers returns.
        """
        import contextlib
        import gc
        import types
        import weakref

        class _Result:
            config = types.SimpleNamespace(num_stages=1, kwargs={})

        results = [_Result(), _Result()]

        def fake_make_launcher(result):
            # First config builds; the last fails, returning a live exception whose
            # traceback references this call stack -- as the real _make_launcher does
            # for OutOfResources / OutOfMemoryError.
            if result is results[0]:
                return object(), None
            try:
                raise RuntimeError("out of resource (test)")
            except RuntimeError as e:
                return None, e

        fake_self = types.SimpleNamespace(
            launchers=[],
            compile_results=results,
            triton_meta={"device": 0},
            device_props=types.SimpleNamespace(type="cuda", index=0),
            inductor_meta={},
            get_device_interface=lambda: None,
            _make_launcher=fake_make_launcher,
        )

        class _Sentinel:
            pass

        def run():
            l2_buffer = _Sentinel()  # stands in for do_bench's L2-flush buffer
            ref = weakref.ref(l2_buffer)
            with patch(
                "torch._dynamo.device_interface.DeviceGuard",
                lambda *args, **kwargs: contextlib.nullcontext(),
            ):
                CachingAutotuner._make_launchers(fake_self)
            return ref

        gc.disable()
        try:
            ref = run()
            self.assertIsNone(
                ref(),
                "_make_launchers retained the failed-config exception; its traceback "
                "pins caller frames (do_bench's L2-flush buffer) until cyclic GC.",
            )
        finally:
            gc.enable()

        self.assertEqual(len(fake_self.launchers), 1)


if __name__ == "__main__":
    if IS_LINUX:
        run_tests()
