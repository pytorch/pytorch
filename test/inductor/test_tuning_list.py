# Owner(s): ["module: inductor"]

import sys
import unittest
from unittest import mock

import torch
from torch._inductor import config as inductor_config, tuning_list
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.runtime.hints import DeviceProperties, ReductionHint
from torch._inductor.runtime.triton_heuristics import (
    CachingAutotuner,
    persistent_reduction,
    pointwise,
    reduction,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.tuning_list import InvalidTuningList
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu_and_triton


try:
    import triton  # @manual  # noqa: F401
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904


# label -> entry point, size hints, reduction hint. Shapes are chosen so the heuristic's
# own output is small and stable enough to assert exact counts on.
KERNELS = {
    "pointwise": (pointwise, {"x": 1048576}, None),
    "reduction": (reduction, {"x": 8192, "r0_": 512}, ReductionHint.INNER),
    "wide_reduction": (reduction, {"x": 8192, "r0_": 8192}, ReductionHint.INNER),
    "persistent_reduction": (
        persistent_reduction,
        {"x": 8192, "r0_": 128},
        ReductionHint.OUTER,
    ),
    "inner_persistent_reduction": (
        persistent_reduction,
        {"x": 8192, "r0_": 128},
        ReductionHint.INNER,
    ),
    "tiled_reduction": (
        reduction,
        {"x": 512, "y": 128, "r0_": 256},
        ReductionHint.INNER,
    ),
    "two_axis_reduction": (
        reduction,
        {"x": 8192, "r0_": 64, "r1_": 64},
        ReductionHint.INNER,
    ),
    "two_axis_persistent_reduction": (
        persistent_reduction,
        {"x": 8192, "r0_": 16, "r1_": 16},
        ReductionHint.INNER,
    ),
}

# A valid entry for each kernel.
BLOCKS = {
    "pointwise": {"XBLOCK": 64},
    "reduction": {"XBLOCK": 8, "R0_BLOCK": 128},
    "persistent_reduction": {"XBLOCK": 16},
    "tiled_reduction": {"XBLOCK": 8, "YBLOCK": 8, "R0_BLOCK": 64},
}

# A second candidate for the kernels that start with one. Its blocks differ from both the
# entry above and the heuristic's own candidate: a config that builds to one already in
# the list is folded into it.
SECOND = {
    "reduction": {"XBLOCK": 16, "R0_BLOCK": 64},
    "persistent_reduction": {"XBLOCK": 8},
}


def fake_device(**overrides):
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
    )._replace(**overrides)


@instantiate_parametrized_tests
class TestTuningList(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def candidates(self, label, table=None, **inductor_meta):
        """The candidates the compiler would benchmark, looked up as codegen does."""
        entry_point, size_hints, reduction_hint = KERNELS[label]
        device = fake_device()
        with inductor_config.patch({"tuning_list.table": table or {}}):
            entry = tuning_list.lookup(entry_point.__name__, size_hints, device)
        if entry is not None:
            inductor_meta["tuning_list"] = entry
        kwargs = {
            "size_hints": size_hints,
            "triton_meta": {"device": device, "signature": {}},
            "inductor_meta": inductor_meta,
            "return_configs": True,
        }
        if reduction_hint is not None:
            kwargs["reduction_hint"] = reduction_hint
        return entry_point(**kwargs)

    def key(self, label):
        """The table key this kernel looks under: its category, then its arity."""
        entry_point, size_hints, _ = KERNELS[label]
        return f"{entry_point.__name__}/{len(size_hints)}"

    def table(self, label, *configs, **entry):
        return {self.key(label): {"configs": list(configs) or [BLOCKS[label]], **entry}}

    @parametrize("label", ["reduction", "persistent_reduction"])
    def test_reduction_kernels_get_one_candidate_by_default(self, label):
        self.assertEqual(len(self.candidates(label)), 1)

    @parametrize("label", ["reduction", "persistent_reduction"])
    def test_supplied_configs_reach_single_candidate_kernels(self, label):
        # A splice inside the heuristic would miss these: its single-candidate return
        # comes first.
        table = self.table(label, BLOCKS[label], SECOND[label])
        self.assertEqual(len(self.candidates(label)), 1)
        self.assertEqual(len(self.candidates(label, table)), 3)

    @requires_gpu_and_triton
    @parametrize("compile_threads", [1, 2])
    def test_supplied_configs_reach_the_compiled_kernel(self, compile_threads):
        # Compiled once without the table first: kernels are cached on their source, and
        # the pool's workers are separate processes.
        def total(t):
            return t.sum(dim=1)

        benchmarked = []
        benchmark_all_configs = CachingAutotuner.benchmark_all_configs

        def record(autotuner, *args, **kwargs):
            benchmarked.extend(r.config.kwargs for r in autotuner.compile_results)
            return benchmark_all_configs(autotuner, *args, **kwargs)

        x = torch.randn(1024, 8192, device=GPU_TYPE)
        table = self.table("reduction", BLOCKS["reduction"], SECOND["reduction"])
        with inductor_config.patch(compile_threads=compile_threads):
            if compile_threads > 1:
                self.assertTrue(AsyncCompile.wait_process_pool_ready())
            torch.compile(total)(x)
            torch._dynamo.reset()
            with (
                inductor_config.patch({"tuning_list.table": table}),
                mock.patch.object(CachingAutotuner, "benchmark_all_configs", record),
            ):
                torch.compile(total)(x)
        self.assertIn(BLOCKS["reduction"], benchmarked)
        self.assertIn(SECOND["reduction"], benchmarked)

    @parametrize("max_autotune", [False, True])
    def test_nothing_is_gated_on_max_autotune(self, max_autotune):
        # Counted as a gain over the same flag's baseline, since the flag changes that too.
        meta = {"max_autotune": max_autotune}
        gained = len(
            self.candidates("reduction", self.table("reduction"), **meta)
        ) - len(self.candidates("reduction", **meta))
        self.assertEqual(gained, 1)

    def test_append_keeps_the_generated_candidates(self):
        generated = self.candidates("pointwise")
        appended = self.candidates("pointwise", self.table("pointwise"))
        self.assertEqual(len(appended), len(generated) + 1)
        self.assertEqual(
            [c.kwargs for c in appended[: len(generated)]],
            [c.kwargs for c in generated],
        )
        self.assertEqual(appended[-1].kwargs["XBLOCK"], 64)

    def test_replace_drops_the_generated_candidates(self):
        replaced = self.candidates("pointwise", self.table("pointwise", mode="replace"))
        self.assertEqual([c.kwargs["XBLOCK"] for c in replaced], [64])

    # Not pointwise: its heuristic sets num_elements_per_warp per candidate, so an entry
    # has to name that too to match.
    @parametrize(
        "label",
        [
            "reduction",
            "persistent_reduction",
            "inner_persistent_reduction",
            "tiled_reduction",
        ],
    )
    def test_naming_the_heuristics_own_blocks_reproduces_its_configs(self, label):
        stock = self.candidates(label)
        blocks = [
            {k: v for k, v in c.kwargs.items() if k.endswith("BLOCK")} for c in stock
        ]
        supplied = self.candidates(label, self.table(label, *blocks, mode="replace"))
        self.assertEqual(
            [(c.kwargs, c.num_warps, c.num_stages) for c in supplied],
            [(c.kwargs, c.num_warps, c.num_stages) for c in stock],
        )

    def test_many_loads_halve_the_warp_cap_as_the_heuristic_does(self):
        table = self.table("wide_reduction", {"XBLOCK": 1, "R0_BLOCK": 4096})
        warps = [
            self.candidates("wide_reduction", table, num_load=loads)[-1].num_warps
            for loads in (1, 10)
        ]
        self.assertEqual(warps, [16, 8])

    @parametrize("key", ["pointwise/2", "reduction/1", "persistent_reduction/2"])
    def test_entry_applies_only_to_its_category_and_arity(self, key):
        table = {key: {"configs": [BLOCKS["reduction"]]}}
        self.assertEqual(
            len(self.candidates("reduction", table)),
            len(self.candidates("reduction")),
        )

    def test_device_specific_key_wins_over_the_generic_one(self):
        table = {
            "reduction/2": {"configs": [BLOCKS["reduction"]]},
            "gfx942+reduction/2": {"configs": [BLOCKS["reduction"]]},
        }
        with inductor_config.patch({"tuning_list.table": table}):
            entry = tuning_list.lookup(
                "reduction", {"x": 8192, "r0_": 512}, fake_device(cc="gfx942")
            )
        self.assertEqual(entry["key"], "gfx942+reduction/2")

    def test_generic_key_matches_when_the_device_has_no_entry(self):
        table = {"reduction/2": {"configs": [BLOCKS["reduction"]]}}
        with inductor_config.patch({"tuning_list.table": table}):
            entry = tuning_list.lookup(
                "reduction", {"x": 8192, "r0_": 512}, fake_device(cc="gfx942")
            )
        self.assertEqual(entry["key"], "reduction/2")

    def test_numeric_device_capability_is_not_a_device_key(self):
        table = {
            "reduction/2": {"configs": [BLOCKS["reduction"]]},
            "80+reduction/2": {"configs": [SECOND["reduction"]]},
        }
        with inductor_config.patch({"tuning_list.table": table}):
            entry = tuning_list.lookup(
                "reduction", {"x": 8192, "r0_": 512}, fake_device()
            )
        self.assertEqual(entry["key"], "reduction/2")

    def test_persistent_candidates_carry_no_reduction_block(self):
        table = self.table("persistent_reduction")
        supplied = self.candidates("persistent_reduction", table)[-1]
        self.assertNotIn("R0_BLOCK", supplied.kwargs)

    def test_persistent_reduction_rejects_a_reduction_block(self):
        table = self.table("persistent_reduction", {"XBLOCK": 16, "R0_BLOCK": 64})
        with self.assertRaisesRegex(InvalidTuningList, "spans the whole reduction"):
            self.candidates("persistent_reduction", table)

    def test_tiled_reduction_rejects_num_warps(self):
        # Accepting it would drop every config in the entry instead.
        table = self.table(
            "tiled_reduction", {**BLOCKS["tiled_reduction"], "num_warps": 4}
        )
        with self.assertRaisesRegex(InvalidTuningList, "derives it from the tile"):
            self.candidates("tiled_reduction", table)

    def test_untiled_reduction_accepts_num_warps(self):
        table = self.table("reduction", {**BLOCKS["reduction"], "num_warps": 8})
        self.assertEqual(self.candidates("reduction", table)[-1].num_warps, 8)

    @parametrize(
        "entry,expected",
        [
            ([{"XBLOCK": 8}], "must be a dict carrying 'configs'"),
            ({"mode": "append"}, "must be a dict carrying 'configs'"),
            ({"configs": [], "mode": "merge"}, "expected one of"),
        ],
    )
    def test_malformed_entry_is_fatal(self, entry, expected):
        with self.assertRaisesRegex(InvalidTuningList, expected):
            self.candidates("reduction", {"reduction/2": entry})

    @parametrize(
        "config,expected",
        [
            ({"XBLOCK": 8, "R0_BLOCK": 128, "nonsense": 1}, "does not accept"),
            ({"XBLOCK": 8}, "does not name R0_BLOCK"),
            ({"XBLOCK": 0, "R0_BLOCK": 128}, "expected a positive int"),
            ({"XBLOCK": True, "R0_BLOCK": 128}, "expected a positive int"),
            ("not a dict", "expected a dict"),
        ],
    )
    def test_malformed_config_is_fatal(self, config, expected):
        with self.assertRaisesRegex(InvalidTuningList, expected):
            self.candidates("reduction", self.table("reduction", config))

    def test_config_that_cannot_be_built_is_dropped_not_fatal(self):
        table = self.table("pointwise", {"XBLOCK": 3})
        self.assertEqual(
            len(self.candidates("pointwise", table)),
            len(self.candidates("pointwise")),
        )

    def test_block_larger_than_the_extent_is_clamped(self):
        table = self.table("reduction", {"XBLOCK": 8, "R0_BLOCK": 4096})
        self.assertEqual(
            self.candidates("reduction", table)[-1].kwargs["R0_BLOCK"], 512
        )

    def test_strict_reduction_keeps_its_planned_block(self):
        # Planning the block the heuristic would have chosen anyway is what leaves the
        # raise free to fire on the supplied ones.
        planned = self.candidates("reduction")[0].kwargs["R0_BLOCK"]
        table = self.table("reduction", BLOCKS["reduction"], SECOND["reduction"])
        kept = self.candidates("reduction", table, strict_reduction_rblock=planned)
        self.assertEqual(len(kept), 1)

    @parametrize(
        "label,blocks",
        [
            ("two_axis_reduction", BLOCKS["tiled_reduction"]),
            ("two_axis_persistent_reduction", {"XBLOCK": 16, "YBLOCK": 8}),
        ],
    )
    def test_kernel_tiled_over_two_reduction_axes_is_left_alone(self, label, blocks):
        # Its key is that of a kernel tiled on y, whose entries name blocks it lacks.
        table = self.table(label, blocks)
        self.assertEqual(
            [c.kwargs for c in self.candidates(label, table)],
            [c.kwargs for c in self.candidates(label)],
        )


if __name__ == "__main__":
    run_tests()
