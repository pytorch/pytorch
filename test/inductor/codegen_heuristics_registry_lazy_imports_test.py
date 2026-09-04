# Owner(s): ["module: inductor"]

import importlib
import unittest

from torch._inductor.heuristics.registry import get_codegen_heuristic


class CodegenHeuristicsRegistryLazyImportsTest(unittest.TestCase):
    def test_codegen_heuristics_are_registered_on_demand(self) -> None:
        # Cache the parent package without realizing its lazy child imports.
        importlib.import_module("torch._inductor.heuristics.triton_codegen")

        expected_types = {
            "pointwise": (
                "torch._inductor.heuristics.triton_codegen.pointwise",
                "PointwiseHeuristic",
            ),
            "reduction": (
                "torch._inductor.heuristics.triton_codegen.reduction",
                "ReductionHeuristic",
            ),
        }

        for name, expected_type in expected_types.items():
            with self.subTest(name=name):
                heuristic = get_codegen_heuristic(name, "cpu")
                heuristic_type = type(heuristic)
                self.assertEqual(
                    (heuristic_type.__module__, heuristic_type.__name__), expected_type
                )
