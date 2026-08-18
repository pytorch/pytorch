# Owner(s): ["module: dsl-native-ops"]

from __future__ import annotations

import sys

from torch._native.ops.sum.inner_tree_plan import (
    compute_inner_tree_params,
    InnerTreeParams,
    vec_size,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestInnerTreePlan(TestCase):
    @parametrize("itemsize, expected", [(2, 8), (4, 4), (8, 2)])
    def test_vec_size(self, itemsize: int, expected: int) -> None:
        self.assertEqual(vec_size(itemsize), expected)

    @parametrize(
        "inputs_per_output, num_outputs, vector_size, expected",
        [
            (512, 32, 4, InnerTreeParams(4, 512, 1, 1, 2, 1)),
            (8192, 32, 4, InnerTreeParams(16, 8192, 1, 2, 1, 4)),
            (8193, 32, 4, InnerTreeParams(8, 8192, 2, 3, 1, 8)),
            (32769, 32, 4, InnerTreeParams(16, 8192, 5, 2, 1, 4)),
        ],
    )
    def test_compute_inner_tree_params(
        self,
        inputs_per_output: int,
        num_outputs: int,
        vector_size: int,
        expected: InnerTreeParams,
    ) -> None:
        self.assertEqual(
            compute_inner_tree_params(inputs_per_output, num_outputs, vector_size),
            expected,
        )

    def test_import_does_not_load_cutlass(self) -> None:
        self.assertNotIn("cutlass", sys.modules)


instantiate_parametrized_tests(TestInnerTreePlan)


if __name__ == "__main__":
    run_tests()
