# Owner(s): ["module: inductor"]

import sys
import unittest
from unittest import mock

import numpy as np

import torch
from torch._inductor import config as inductor_config, tuning_list
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.choices import (
    InductorChoices,
    register_inductor_choices,
    unregister_inductor_choices,
)
from torch._inductor.heuristics.triton_codegen.reduction import ReductionHeuristic
from torch._inductor.runtime.hints import DeviceProperties, ReductionHint
from torch._inductor.runtime.triton_heuristics import (
    CachingAutotuner,
    cooperative_reduction,
    persistent_reduction,
    pointwise,
    reduction,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.tuning_list import InvalidTuningList
from torch._inductor.virtualized import V
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
    "tiled_pointwise": (pointwise, {"y": 1024, "x": 1024}, None),
    "pointwise_3d": (pointwise, {"z": 64, "y": 64, "x": 256}, None),
    "reduction": (reduction, {"x": 8192, "r0_": 512}, ReductionHint.INNER),
    "outer_reduction": (reduction, {"x": 8192, "r0_": 512}, ReductionHint.OUTER),
    "tiny_reduction": (reduction, {"x": 8192, "r0_": 512}, ReductionHint.OUTER_TINY),
    "default_reduction": (reduction, {"x": 8192, "r0_": 512}, ReductionHint.DEFAULT),
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
    "tiled_persistent_reduction": (
        persistent_reduction,
        {"x": 512, "y": 64, "r0_": 128},
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
    "small_persistent_reduction": (
        persistent_reduction,
        {"x": 2, "r0_": 128},
        ReductionHint.OUTER,
    ),
    "cooperative_reduction": (
        cooperative_reduction,
        {"x": 8, "r0_": 65536},
        ReductionHint.INNER,
    ),
}

# fake_device overrides. The registry picks heuristics by device type and falls back to the
# default ones where none is registered, as for "hip" off ROCm builds.
DEVICES = {
    "cuda": {},
    "hip": {"type": "hip", "cc": "gfx942", "major": 9, "warp_size": 64},
}

# A valid entry for each kernel.
BLOCKS = {
    "pointwise": {"XBLOCK": 64},
    "tiled_pointwise": {"XBLOCK": 64, "YBLOCK": 32},
    "pointwise_3d": {"XBLOCK": 32, "YBLOCK": 8, "ZBLOCK": 4},
    "reduction": {"XBLOCK": 8, "R0_BLOCK": 128},
    "persistent_reduction": {"XBLOCK": 16},
    "tiled_persistent_reduction": {"XBLOCK": 16, "YBLOCK": 8},
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

    def lookup(self, label, table, triton_meta=None, inductor_meta=None):
        """The entry codegen would carry in this kernel's source, or None."""
        entry_point, size_hints, _ = KERNELS[label]
        heuristic = entry_point.__name__
        triton_meta = {"device": fake_device(), **(triton_meta or {})}
        inductor_meta = inductor_meta or {}
        with inductor_config.patch({"tuning_list.table": table}):
            supplied = V.choices.triton_kernel_configs(
                heuristic, size_hints, triton_meta, inductor_meta
            )
        return tuning_list.check(
            supplied, heuristic, size_hints, triton_meta, inductor_meta
        )

    def candidates(self, label, table=None, device_type="cuda", **inductor_meta):
        """The candidates the compiler would benchmark, looked up as codegen does."""
        entry_point, size_hints, reduction_hint = KERNELS[label]
        device = fake_device(**DEVICES[device_type])
        if entry := self.lookup(label, table or {}, {"device": device}, inductor_meta):
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
        """The table key this kernel looks under: its category, then its number of axes."""
        entry_point, size_hints, _ = KERNELS[label]
        return f"{entry_point.__name__}/{len(size_hints)}"

    def table(self, label, *configs, **entry):
        return {self.key(label): {"configs": list(configs) or [BLOCKS[label]], **entry}}

    @parametrize("device_type", list(DEVICES))
    @parametrize("label", ["reduction", "persistent_reduction"])
    def test_reduction_kernels_get_one_candidate_by_default(self, label, device_type):
        self.assertEqual(len(self.candidates(label, device_type=device_type)), 1)

    @parametrize("device_type", list(DEVICES))
    @parametrize("label", ["reduction", "persistent_reduction"])
    def test_supplied_configs_reach_single_candidate_kernels(self, label, device_type):
        table = self.table(label, BLOCKS[label], SECOND[label])
        self.assertEqual(len(self.candidates(label, device_type=device_type)), 1)
        self.assertEqual(len(self.candidates(label, table, device_type=device_type)), 3)

    @requires_gpu_and_triton
    @parametrize("source", ["table", "choices"])
    @parametrize("compile_threads", [1, 2])
    def test_supplied_configs_reach_the_compiled_kernel(self, compile_threads, source):
        # Compiled once without them first: kernels are cached on their source, and the
        # pool's workers are separate processes.
        def total(t):
            return t.sum(dim=1)

        entry = {"configs": [BLOCKS["reduction"], SECOND["reduction"]]}

        class Choices(InductorChoices):
            def triton_kernel_configs(
                self, heuristic, size_hints, triton_meta, inductor_meta
            ):
                return entry if heuristic == "reduction" else None

            def uuid(self):
                return "test_tuning_list"

        benchmarked = []
        benchmark_all_configs = CachingAutotuner.benchmark_all_configs

        def record(autotuner, *args, **kwargs):
            benchmarked.extend(r.config.kwargs for r in autotuner.compile_results)
            return benchmark_all_configs(autotuner, *args, **kwargs)

        x = torch.randn(1024, 8192, device=GPU_TYPE)
        table = {"reduction/2": entry} if source == "table" else {}
        with inductor_config.patch(compile_threads=compile_threads):
            if compile_threads > 1:
                self.assertTrue(AsyncCompile.wait_process_pool_ready())
            torch.compile(total)(x)
            torch._dynamo.reset()
            if source == "choices":
                # Registered rather than set, so its uuid() keys the FX graph cache.
                register_inductor_choices("test_tuning_list", Choices)
                self.addCleanup(unregister_inductor_choices, "test_tuning_list")
            with (
                inductor_config.patch({"tuning_list.table": table}),
                mock.patch.object(CachingAutotuner, "benchmark_all_configs", record),
            ):
                actual = torch.compile(total)(x)
        self.assertIn(BLOCKS["reduction"], benchmarked)
        self.assertIn(SECOND["reduction"], benchmarked)
        self.assertEqual(actual, total(x), atol=1e-3, rtol=1e-3)

    @parametrize("max_autotune", [False, True])
    def test_nothing_is_gated_on_max_autotune(self, max_autotune):
        # Counted as a gain over the same flag's baseline, since the flag changes that too.
        meta = {"max_autotune": max_autotune}
        gained = len(
            self.candidates("reduction", self.table("reduction"), **meta)
        ) - len(self.candidates("reduction", **meta))
        self.assertEqual(gained, 1)

    @parametrize("label", ["pointwise", "tiled_pointwise"])
    def test_append_keeps_the_generated_candidates(self, label):
        generated = self.candidates(label)
        appended = self.candidates(label, self.table(label))
        self.assertEqual(len(appended), len(generated) + 1)
        self.assertEqual(
            [c.kwargs for c in appended[: len(generated)]],
            [c.kwargs for c in generated],
        )
        self.assertEqual(appended[-1].kwargs, BLOCKS[label])

    @parametrize(
        "label",
        [
            "pointwise",
            "tiled_pointwise",
            "pointwise_3d",
            "reduction",
            "tiled_reduction",
            "persistent_reduction",
            "tiled_persistent_reduction",
        ],
    )
    def test_replace_drops_the_generated_candidates(self, label):
        replaced = self.candidates(label, self.table(label, mode="replace"))
        self.assertEqual([c.kwargs for c in replaced], [BLOCKS[label]])

    # Each hint returns from a different place in the heuristic.
    @parametrize(
        "label",
        ["reduction", "outer_reduction", "tiny_reduction", "default_reduction"],
    )
    def test_every_reduction_hint_appends_the_supplied_configs(self, label):
        generated = self.candidates(label)
        appended = self.candidates(label, self.table(label, BLOCKS["reduction"]))
        self.assertEqual(
            [c.kwargs for c in appended],
            [c.kwargs for c in generated] + [BLOCKS["reduction"]],
        )

    def test_device_post_processing_sees_appended_configs_but_not_a_replacement(self):
        # The DEFAULT hint ends in _finalize_configs, where a device heuristic adds and
        # filters candidates; ROCm's adds two.
        seen = []

        def finalize(heuristic, configs, make_config, size_hints, inductor_meta):
            seen.extend(c.kwargs for c in configs)
            return configs

        appended = self.table("default_reduction", BLOCKS["reduction"])
        replaced = self.table("default_reduction", BLOCKS["reduction"], mode="replace")
        with mock.patch.object(ReductionHeuristic, "_finalize_configs", finalize):
            self.candidates("default_reduction", appended)
            self.assertIn(BLOCKS["reduction"], seen)
            seen.clear()
            only = self.candidates("default_reduction", replaced)
        self.assertEqual(seen, [])
        self.assertEqual([c.kwargs for c in only], [BLOCKS["reduction"]])

    def test_choices_handler_supplies_the_entry_in_place_of_the_table(self):
        class Choices(InductorChoices):
            def triton_kernel_configs(
                self, heuristic, size_hints, triton_meta, inductor_meta
            ):
                return {"configs": [SECOND["reduction"]]}

        with V.set_choices_handler(Choices()):
            supplied = self.candidates("reduction", self.table("reduction"))
            # The kernels under Not Covered stay so whoever supplies the entry.
            self.assertIsNone(self.lookup("cooperative_reduction", {}))
            self.assertIsNone(
                self.lookup("tiled_reduction", {}, {"native_matmul": True})
            )
        self.assertEqual([c.kwargs for c in supplied[1:]], [SECOND["reduction"]])

    def test_choices_handler_entry_is_checked_like_a_table_entry(self):
        class Choices(InductorChoices):
            def triton_kernel_configs(
                self, heuristic, size_hints, triton_meta, inductor_meta
            ):
                return {"configs": [{"XBLOCK": 8}]}

        with (
            V.set_choices_handler(Choices()),
            self.assertRaisesRegex(InvalidTuningList, "does not name R0_BLOCK"),
        ):
            self.candidates("reduction")

    # Not pointwise: its heuristic sets num_elements_per_warp per candidate, so an entry
    # has to name that too to match. The ROCm heuristic does the same with waves_per_eu.
    @parametrize("device_type", list(DEVICES))
    @parametrize(
        "label",
        [
            "reduction",
            "persistent_reduction",
            "inner_persistent_reduction",
            "tiled_reduction",
        ],
    )
    def test_naming_the_heuristics_own_blocks_reproduces_its_configs(
        self, label, device_type
    ):
        stock = self.candidates(label, device_type=device_type)
        named = [
            {
                k: v
                for k, v in c.kwargs.items()
                if k.endswith("BLOCK") or k == "waves_per_eu"
            }
            for c in stock
        ]
        table = self.table(label, *named, mode="replace")
        supplied = self.candidates(label, table, device_type=device_type)
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
    def test_entry_applies_only_to_its_category_and_number_of_axes(self, key):
        table = {key: {"configs": [BLOCKS["reduction"]]}}
        self.assertEqual(
            len(self.candidates("reduction", table)),
            len(self.candidates("reduction")),
        )

    def test_device_specific_key_wins_over_the_generic_one(self):
        table = {
            "reduction/2": {"configs": [BLOCKS["reduction"]]},
            "gfx942+reduction/2": {"configs": [SECOND["reduction"]]},
        }
        entry = self.lookup("reduction", table, {"device": fake_device(cc="gfx942")})
        self.assertEqual(entry["configs"], [SECOND["reduction"]])

    def test_generic_key_matches_when_the_device_has_no_entry(self):
        table = {
            "reduction/2": {"configs": [BLOCKS["reduction"]]},
            "gfx90a+reduction/2": {"configs": [SECOND["reduction"]]},
        }
        entry = self.lookup("reduction", table, {"device": fake_device(cc="gfx942")})
        self.assertEqual(entry["configs"], [BLOCKS["reduction"]])

    def test_cuda_compute_capability_is_a_device_key(self):
        table = {
            "reduction/2": {"configs": [BLOCKS["reduction"]]},
            "80+reduction/2": {"configs": [SECOND["reduction"]]},
        }
        entry = self.lookup("reduction", table)
        self.assertEqual(entry["configs"], [SECOND["reduction"]])

    @parametrize(
        "key", ["reductions/2", "reduction", "reduction/two", "gfx942+reduction"]
    )
    def test_malformed_key_is_fatal(self, key):
        # Checked on every lookup, so a typo fails even where no kernel would match it.
        table = {key: {"configs": [BLOCKS["reduction"]]}}
        with self.assertRaisesRegex(InvalidTuningList, "is not '"):
            self.candidates("pointwise", table)

    def test_native_matmul_kernel_is_left_alone(self):
        # Its configs come from a matmul-specific constructor with its own limits.
        table = self.table("tiled_reduction")
        self.assertIsNone(
            self.lookup("tiled_reduction", table, {"native_matmul": True})
        )

    @parametrize(
        "label,triton_meta,inductor_meta,reason",
        [
            ("cooperative_reduction", {}, {}, "only pointwise"),
            ("tiled_reduction", {"native_matmul": True}, {}, "native matmul"),
            ("reduction", {}, {"strict_reduction_rblock": 512}, "is planned"),
            ("two_axis_reduction", {}, {}, "two reduction axes"),
        ],
    )
    def test_skipping_a_matching_entry_is_logged(
        self, label, triton_meta, inductor_meta, reason
    ):
        # A cooperative reduction is looked up under the reduction it splits.
        table = {
            "reduction/2": {"configs": [BLOCKS["reduction"]]},
            "reduction/3": {"configs": [BLOCKS["tiled_reduction"]]},
        }
        with self.assertLogs("torch._inductor.tuning_list.table", "DEBUG") as logged:
            self.assertIsNone(self.lookup(label, table, triton_meta, inductor_meta))
        self.assertIn(reason, logged.output[-1])

    def test_numpy_ints_are_stored_as_plain_ints(self):
        # The entry is written into the kernel source, which does not import numpy.
        config = {"XBLOCK": np.int64(8), "R0_BLOCK": np.int64(128)}
        entry = self.lookup("reduction", self.table("reduction", config))
        self.assertEqual(
            {type(v) for one in entry["configs"] for v in one.values()}, {int}
        )

    def test_batch_invariant_persistent_reduction_ignores_the_batch(self):
        # batch_invariant implies deterministic, which ships one candidate; replace makes
        # it the supplied one.
        table = self.table("persistent_reduction", mode="replace")
        meta = {"batch_invariant": True, "deterministic": True}
        small, large = (
            [(c.kwargs, c.num_warps) for c in self.candidates(label, table, **meta)]
            for label in ("small_persistent_reduction", "persistent_reduction")
        )
        self.assertEqual(small, large)

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
            ({"XBLOCK": 0, "R0_BLOCK": 128}, "expected a power of 2"),
            ({"XBLOCK": 3, "R0_BLOCK": 128}, "expected a power of 2"),
            ({"XBLOCK": 8, "R0_BLOCK": 96}, "expected a power of 2"),
            ({**BLOCKS["reduction"], "num_warps": 6}, "expected a power of 2"),
            ({**BLOCKS["reduction"], "num_stages": -1}, "expected a non-negative int"),
            ({"XBLOCK": True, "R0_BLOCK": 128}, "expected an int"),
            ({"XBLOCK": 8.0, "R0_BLOCK": 128}, "expected an int"),
            ("not a dict", "expected a dict"),
        ],
    )
    def test_malformed_config_is_fatal(self, config, expected):
        with self.assertRaisesRegex(InvalidTuningList, expected):
            self.candidates("reduction", self.table("reduction", config))

    def test_config_that_cannot_be_built_is_dropped_not_fatal(self):
        # Above every backend's XBLOCK limit, and below this kernel's extent, so clamping
        # leaves it there.
        table = self.table("pointwise", {"XBLOCK": 16384})
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
