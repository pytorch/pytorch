# Owner(s): ["module: cuda"]

"""Checks common_cuda.GATE_ARCH_RANGES against the predicates it describes.

The table is read statically by CI tooling to decide which GPU a gated test
needs. A hand-maintained copy of predicate logic drifts silently, so this
re-derives each range by evaluating the real predicate across a sweep of
simulated compute capabilities and fails if the two disagree.
"""

import ast
import unittest
from unittest import mock

import torch
import torch.testing._internal.common_cuda as cc
from torch.testing._internal.common_utils import run_tests, TestCase


# Wide enough to bracket every declared range, and to include the capabilities
# CI actually runs on: sm75 g4dn, sm80 a100, sm86 a10g, sm89 L4, sm90 H100,
# sm100 B200.
SWEEP = [
    (5, 0),
    (5, 3),
    (6, 0),
    (7, 0),
    (7, 5),
    (8, 0),
    (8, 6),
    (8, 9),
    (9, 0),
    (10, 0),
    (10, 1),
    (11, 0),
    (12, 0),
    (13, 0),
]

# Predicates whose value comes from a named evaluate_* helper we can call
# repeatedly. LazyVal caches on first force and discards its callback, so the
# helper is the only re-evaluatable form.
EVALUATORS = {
    "PLATFORM_SUPPORTS_BF16_ATOMICS": "evaluate_platform_supports_bf16_atomics",
    "PLATFORM_SUPPORTS_HALF_ATOMICS": "evaluate_platform_supports_half_atomics",
    "PLATFORM_SUPPORTS_BF16": "evaluate_platform_supports_bf16",
    "PLATFORM_SUPPORTS_FLASH_ATTENTION": "evaluate_platform_supports_flash_attention",
    "PLATFORM_SUPPORTS_MEM_EFF_ATTENTION": "evaluate_platform_supports_efficient_attention",
    "PLATFORM_SUPPORTS_CUDNN_ATTENTION": "evaluate_platform_supports_cudnn_attention",
    "PLATFORM_SUPPORTS_FP8": "evaluate_platform_supports_fp8",
    "PLATFORM_SUPPORTS_FP8_GROUPED_GEMM": "evaluate_platform_supports_fp8_grouped_gemm",
    "PLATFORM_SUPPORTS_MX_GEMM": "evaluate_platform_supports_mx_gemm",
    "PLATFORM_SUPPORTS_MXFP8_GROUPED_GEMM": "evaluate_platform_supports_mxfp8_grouped_gemm",
    "PLATFORM_SUPPORTS_FP8_SPARSE": "evaluate_platform_supports_fp8_sparse",
}

# Predicates defined as a disjunction of others; their range is the union, so the
# declared floor must be the lowest component floor.
COMPOSITES = {
    "has_triton_tma": [
        "has_triton_tensor_descriptor_host_tma",
        "has_triton_experimental_host_tma",
    ],
    "PLATFORM_SUPPORTS_FUSED_ATTENTION": [
        "PLATFORM_SUPPORTS_FLASH_ATTENTION",
        "PLATFORM_SUPPORTS_MEM_EFF_ATTENTION",
        "PLATFORM_SUPPORTS_CUDNN_ATTENTION",
    ],
}

TRITON_PROBES = [
    "has_triton_tma_device",
    "has_triton_stable_tma_api",
    "has_triton_tensor_descriptor_host_tma",
    "has_triton_experimental_host_tma",
    "has_datacenter_blackwell_tma_device",
]


def in_range(cap, arch_range):
    lo, hi = arch_range
    return cap >= lo and (hi is None or cap < hi)


class TestArchRanges(TestCase):
    def _sweep(self, call):
        """{capability: bool} for `call` evaluated at each simulated capability."""
        observed = {}
        for cap in SWEEP:
            with (
                mock.patch.object(torch.cuda, "is_available", lambda: True),
                mock.patch.object(
                    torch.cuda, "get_device_capability", lambda *a, **k: cap
                ),
            ):
                # The evaluate_* helpers read these module globals rather than
                # calling get_device_capability directly.
                flags = {
                    name: (
                        cap >= rng[0] and (rng[1] is None or cap < rng[1])
                        if name.startswith(("SM", "IS_SM"))
                        else None
                    )
                    for name, rng in cc.GATE_ARCH_RANGES.items()
                }
                patches = {k: v for k, v in flags.items() if v is not None}
                with mock.patch.multiple(cc, **patches):
                    try:
                        observed[cap] = bool(call())
                    except Exception:
                        observed[cap] = None
        return observed

    def _check(self, name, observed):
        declared = cc.GATE_ARCH_RANGES[name]
        usable = {c: v for c, v in observed.items() if v is not None}
        if not any(usable.values()):
            raise unittest.SkipTest(
                f"{name} is False at every capability on this build, so its range "
                f"cannot be observed here; it depends on build or library support "
                f"beyond compute capability"
            )
        mismatched = [
            (cap, val) for cap, val in usable.items() if val != in_range(cap, declared)
        ]
        self.assertEqual(
            mismatched,
            [],
            f"GATE_ARCH_RANGES[{name!r}] = {declared} disagrees with the predicate at "
            f"{[c for c, _ in mismatched]}; observed True at "
            f"{sorted(c for c, v in usable.items() if v)}",
        )

    def test_platform_supports_ranges_match_predicates(self):
        for name, evaluator in EVALUATORS.items():
            fn = getattr(cc, evaluator, None)
            if fn is None:
                self.fail(f"{evaluator} no longer exists; update EVALUATORS")
            with self.subTest(predicate=name):
                self._check(name, self._sweep(fn))

    def test_triton_probe_ranges_match_predicates(self):
        from torch.utils import _triton

        for name in TRITON_PROBES:
            fn = getattr(_triton, name)
            with self.subTest(predicate=name):

                def call(fn=fn):
                    fn.cache_clear()
                    return fn()

                self._check(name, self._sweep(call))

    def test_composite_ranges_are_the_union_of_their_parts(self):
        for name, parts in COMPOSITES.items():
            declared = cc.GATE_ARCH_RANGES[name]
            floors = [cc.GATE_ARCH_RANGES[p][0] for p in parts]
            self.assertEqual(
                declared[0],
                min(floors),
                f"{name} is a disjunction of {parts}; its floor should be {min(floors)}",
            )
            unbounded = any(cc.GATE_ARCH_RANGES[p][1] is None for p in parts)
            self.assertEqual(
                declared[1] is None,
                unbounded,
                f"{name}: an unbounded component makes the union unbounded",
            )

    def test_no_arch_gate_predicate_is_unclassified(self):
        # A new SM*/IS_SM*/PLATFORM_SUPPORTS_* predicate must be given a range or
        # explicitly declared not capability-gated. Without this, adding one and
        # gating a test on it leaves that test's GPU requirement invisible.
        import re

        import torch.testing._internal.common_distributed as cd

        shaped = set()
        for module in (cc, cd):
            tree = ast.parse(open(module.__file__, encoding="utf-8").read())
            for node in tree.body:
                targets = (
                    [node.target]
                    if isinstance(node, ast.AnnAssign)
                    else getattr(node, "targets", [])
                )
                for t in targets:
                    name = getattr(t, "id", None)
                    if name and re.match(
                        r"^(SM\d+OrLater|IS_SM\w+|PLATFORM_SUPPORTS_\w+)$", name
                    ):
                        shaped.add(name)
        self.assertEqual(
            shaped - set(cc.GATE_ARCH_RANGES) - set(cc.NOT_ARCH_GATED),
            set(),
            "add these to GATE_ARCH_RANGES, or to NOT_ARCH_GATED with a reason",
        )

    def test_every_gate_predicate_is_covered(self):
        # A new arch-gate predicate must be added here as well as to the table,
        # or its declared range would never be verified.
        known = (
            set(EVALUATORS)
            | set(TRITON_PROBES)
            | set(COMPOSITES)
            | {n for n in cc.GATE_ARCH_RANGES if n.startswith(("SM", "IS_SM"))}
        )
        self.assertEqual(
            set(cc.GATE_ARCH_RANGES) - known,
            set(),
            "predicates in GATE_ARCH_RANGES with no verification path",
        )


if __name__ == "__main__":
    run_tests()
