# Owner(s): ["module: dynamo"]

import numpy

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_array_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_array_equal


@instantiate_parametrized_tests
class TestAdvancedIndexing(TestCase):
    """Test advanced indexing for NumPy compatibility (separated advanced indices)."""

    def _test_cases(self, test_cases, description=""):
        """Unified test runner for multiple test cases."""
        for case in test_cases:
            shape, index = case["shape"], case["index"]
            name = case.get("name", f"{description}: {index}")

            with self.subTest(name=name):
                # Create test arrays
                torch_arr = np.arange(numpy.prod(shape)).reshape(shape)
                numpy_arr = numpy.arange(numpy.prod(shape)).reshape(shape)

                # Test getitem
                tr, nr = torch_arr[index], numpy_arr[index]
                self.assertEqual(tr.shape, nr.shape, f"{name} getitem shape mismatch")
                assert_array_equal(
                    tr.tensor.numpy() if hasattr(tr, "tensor") else tr, nr
                )

                # Test setitem
                tc, nc = torch_arr.copy(), numpy_arr.copy()
                value = np.ones(tr.shape) * 999
                val_np = value.tensor.numpy() if hasattr(value, "tensor") else value
                tc[index], nc[index] = value, val_np
                assert_array_equal(
                    tc.tensor.numpy() if hasattr(tc, "tensor") else tc,
                    nc,
                    f"{name} setitem mismatch",
                )

    def test_basic_patterns(self):
        """Test fundamental indexing patterns across dimensions."""
        cases = [
            # 1D cases
            *[
                {"shape": (10,), "index": idx}
                for idx in [([0, 2, 4],), ([1, 3, 5, 7],), ([0, 0, 1, 1],)]
            ],
            # 2D cases
            *[
                {"shape": (4, 6), "index": idx}
                for idx in [
                    (slice(None), [0]),
                    ([0], slice(None)),
                    (slice(1, 3), [0, 2]),
                    ([0, 2], slice(1, 4)),
                    ([0, 2], 1),
                    (1, [0, 2, 4]),
                ]
            ],
            # 3D cases
            *[
                {"shape": (3, 4, 5), "index": idx}
                for idx in [
                    ([0, 2], slice(None), slice(None)),
                    (slice(None), [0, 2], slice(None)),
                    (slice(None), slice(None), [0, 2, 4]),
                    ([0, 1], slice(None), 2),
                    ([0, 2], 1, slice(None)),
                    (1, [1, 3], slice(None)),
                    (slice(None), [0, 2], 3),
                    ([0, 1], slice(None), [0, 2]),
                    ([1, 2], slice(1, 3), [1, 3]),
                    (slice(None), [0], slice(None)),
                ]
            ],
            # 4D cases
            *[
                {"shape": (2, 3, 4, 5), "index": idx}
                for idx in [
                    (slice(None), [0], 0, slice(None)),
                    (slice(None), [0], slice(None), 0),
                    (slice(None), [0, 1], 0, slice(None)),
                    (slice(None), [0, 1], slice(None), 0),
                    ([0, 1], slice(None), [0, 2], slice(None)),
                    (slice(None), [0, 2], slice(None), [0, 3]),
                    ([0], slice(None), slice(None), [1, 3]),
                    (slice(None), slice(None), [0, 2], [1, 4]),
                ]
            ],
        ]
        self._test_cases(cases, "Basic patterns")

    def test_separated_indices(self):
        """Test multiple separated advanced indices and special patterns."""
        cases = [
            # Multiple separated on different shapes
            *[
                {"shape": (3, 4, 5), "index": idx}
                for idx in [
                    ([0, 1], slice(None), [1, 2]),
                    ([0, 2], slice(1, 3), [0, 1]),
                    ([0, 1], 1, [1, 2]),
                ]
            ],
            # Higher dimensional separations
            *[
                {"shape": (2, 3, 4, 5, 2), "index": idx}
                for idx in [
                    ([0], slice(None), [1], slice(None), [0]),
                    ([0, 1], 0, [1], slice(None), [0, 1]),
                ]
            ],
            # Complex patterns
            {
                "shape": (3, 4, 5, 6, 7, 8),
                "index": (
                    [1, 2],
                    slice(None),
                    [2, 4],
                    slice(None),
                    [5, 2],
                    slice(None),
                ),
            },
            {
                "shape": (3, 4, 5, 6, 7, 8, 9, 10),
                "index": (
                    [1, 2],
                    slice(None),
                    slice(None),
                    [2, 4],
                    slice(None),
                    slice(None),
                    [5, 2],
                    slice(None),
                ),
            },
            # Adjacent vs separated
            {"shape": (3, 4, 5), "index": ([0, 1], [1, 2]), "name": "Adjacent indices"},
            {
                "shape": (3, 4, 5),
                "index": ([0, 1], slice(None), [1, 2]),
                "name": "Separated indices",
            },
            # Edge cases with negative indices
            *[
                {"shape": (3, 4, 5), "index": idx}
                for idx in [
                    ([0], slice(None), [1]),
                    ([-1], slice(None), [-1]),
                    ([0, -1], slice(None), [1, -1]),
                ]
            ],
        ]
        self._test_cases(cases, "Separated indices")

    def test_current_logic_issues(self):
        """Test cases that expose issues with current transpose logic."""
        cases = [
            {
                "shape": (2, 3, 4, 5),
                "index": (0, slice(None), [1], slice(None)),
                "name": "False negative: [0, :, [1], :]",
            },
            {
                "shape": (2, 3, 4, 5),
                "index": (0, [1], slice(None), 0),
                "name": "False negative: [0, [1], :, 0]",
            },
        ]
        self._test_cases(cases, "Current logic issues")

    def test_comprehensive_edge_cases(self):
        """Test comprehensive edge cases for advanced indexing separation detection."""
        cases = [
            {
                "shape": (5, 6, 7, 8, 9),
                "index": (slice(None), slice(None), [1, 2], slice(None), slice(None)),
                "name": "Single advanced index at position 2 with slices before and after",
            },
            {
                "shape": (5, 6, 7, 8, 9),
                "index": (slice(None), [1, 2], slice(None), 3, slice(None)),
                "name": "Single advanced index with slice before and scalar after slice",
            },
            {
                "shape": (5, 6, 7, 8, 9),
                "index": (
                    slice(None),
                    slice(None),
                    slice(None),
                    [1, 2, 3],
                    slice(None),
                ),
                "name": "Single advanced index at position 3 with slices",
            },
            {
                "shape": (5, 6, 7, 8, 9),
                "index": ([1, 2], slice(None), slice(None), slice(None), slice(None)),
                "name": "Single advanced index at position 0",
            },
            {
                "shape": (5, 6, 7, 8, 9),
                "index": (slice(None), slice(None), slice(None), slice(None), [1, 2]),
                "name": "Single advanced index at last position",
            },
            {
                "shape": (5, 6, 7, 8, 9),
                "index": (2, [1, 2], slice(None), slice(None), slice(None)),
                "name": "Single advanced index with scalar before",
            },
            {
                "shape": (5, 6, 7, 8, 9),
                "index": (slice(None), [1, 2], 3, slice(None), slice(None)),
                "name": "Single advanced index with slice before and scalar immediately after",
            },
            {
                "shape": (5, 6, 7, 8, 9),
                "index": (2, 3, [1, 2], 4, slice(None)),
                "name": "Single advanced index with scalars before and after",
            },
        ]
        self._test_cases(cases, "Edge cases")

    def test_high_dimensional_broadcast_cases(self):
        """Test broadcast-like indexing patterns with 5D inputs creating 8-9D outputs."""

        # Helper to create indices
        def make_indices():
            return {
                "ind_1": [[1, 2], [3, 4], [5, 6]],  # Shape (3, 2)
                "ind_2": [[7, 8], [9, 10], [11, 12]],  # Shape (3, 2)
            }

        base_shape = (10, 20, 30, 40, 50)
        patterns = [
            (
                "[:, ind_1, ind_2]",
                (slice(None), "ind_1", "ind_2"),
                "Adjacent advanced indices",
            ),
            (
                "[:, ind_1, :, ind_2]",
                (slice(None), "ind_1", slice(None), "ind_2"),
                "Separated advanced indices",
            ),
            (
                "[ind_1, :, :, ind_2, :]",
                ("ind_1", slice(None), slice(None), "ind_2", slice(None)),
                "Multiple separations",
            ),
            (
                "[ind_1, ind_2, :, :, :]",
                ("ind_1", "ind_2", slice(None), slice(None), slice(None)),
                "Adjacent at start",
            ),
        ]

        cases = []
        for pattern_name, pattern_tuple, description in patterns:
            indices = make_indices()
            # Convert pattern tuple to actual index tuple
            index_tuple = tuple(
                indices[item] if isinstance(item, str) else item
                for item in pattern_tuple
            )
            cases.append(
                {
                    "shape": base_shape,
                    "index": index_tuple,
                    "name": f"{description}: {pattern_name}",
                }
            )

        # Higher dimensional indices
        hd_indices = {
            "ind_1": [[[1, 2]], [[3, 4]]],  # Shape (2, 1, 2)
            "ind_2": [[[5, 6]], [[7, 8]]],  # Shape (2, 1, 2)
        }
        cases.append(
            {
                "shape": base_shape,
                "index": (
                    slice(None),
                    hd_indices["ind_1"],
                    slice(None),
                    hd_indices["ind_2"],
                ),
                "name": "Higher dimensional indices creating 8D output",
            }
        )

        # Even higher dimensional
        vhd_indices = {
            "ind_1": [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]],  # Shape (2, 2, 1, 2)
            "ind_2": [
                [[[9, 10]], [[11, 12]]],
                [[[13, 14]], [[15, 16]]],
            ],  # Shape (2, 2, 1, 2)
        }
        cases.append(
            {
                "shape": base_shape,
                "index": (
                    slice(None),
                    vhd_indices["ind_1"],
                    slice(None),
                    vhd_indices["ind_2"],
                ),
                "name": "Even higher dimensional indices creating 9D output",
            }
        )

        self._test_cases(cases, "High dimensional broadcast")

    def test_numpy_docs_examples(self):
        """Test cases adapted from NumPy documentation."""
        cases = [
            # Example 1: Basic case with single advanced index + slice
            {
                "shape": (5, 7),
                "index": ([0, 2, 4], slice(1, 3)),
                "name": "NumPy docs: y[array([0, 2, 4]), 1:3]",
            },
            # Example 2: Equivalent operation
            {
                "shape": (4, 3),
                "index": ([0, 1, 2], [0, 1, 0]),
                "name": "NumPy docs: x[[0, 1, 2], [0, 1, 0]]",
            },
            # Example 3: Simple broadcasting case
            {
                "shape": (4, 3),
                "index": ([[0], [3]], [0, 2]),
                "name": "NumPy docs: broadcasting case",
            },
            # Example 4: More complex patterns from NumPy docs
            {
                "shape": (5, 7),
                "index": ([0, 2, 4], 1),
                "name": "NumPy docs: mixed advanced and scalar",
            },
            {
                "shape": (3, 4),
                "index": ([0, 2], slice(None)),
                "name": "NumPy docs: advanced + slice",
            },
        ]

        self._test_cases(cases, "NumPy docs examples")

    def test_numpy_specification_examples(self):
        """Test specific examples from NumPy documentation to ensure spec compliance."""
        base_shape = (10, 20, 30, 40, 50)
        indices = {
            "ind_1": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],  # (3, 4)
            "ind_2": [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],  # (3, 4)
        }

        cases = [
            {
                "shape": base_shape,
                "index": (slice(None), indices["ind_1"], indices["ind_2"]),
                "name": "NumPy docs: x[:, ind_1, ind_2] with (3,4) broadcast - adjacent indices",
            },
            {
                "shape": base_shape,
                "index": (slice(None), indices["ind_1"], slice(None), indices["ind_2"]),
                "name": "NumPy docs: x[:, ind_1, :, ind_2] separated case - broadcast dims move to front",
            },
        ]
        self._test_cases(cases, "NumPy specification")

    def test_special_cases(self):
        """Test special edge cases and corner scenarios."""
        cases = [
            # Broadcasting patterns
            {
                "shape": (4, 6),
                "index": ([[0, 1], [0, 1]], [[0, 1], [2, 3]]),
                "name": "2D broadcasting",
            },
            {
                "shape": (3, 4, 5),
                "index": ([0, 1, 2], [0, 1, 0]),
                "name": "Element selection",
            },
            # Mixed patterns
            {
                "shape": (3, 4, 5),
                "index": ([0, 2], slice(1, 3), [1, 3]),
                "name": "Mixed advanced and slice",
            },
            {
                "shape": (4, 5, 6),
                "index": (slice(None), [1, 3], slice(2, 5)),
                "name": "Slice-advanced-slice pattern",
            },
            # Corner cases
            {
                "shape": (2, 3, 4),
                "index": ([0], [1], [2]),
                "name": "Single element selection",
            },
            {
                "shape": (5, 5, 5),
                "index": ([0, 1], slice(None), [2, 3]),
                "name": "Symmetric shape separation",
            },
        ]
        self._test_cases(cases, "Special cases")

    def test_numpy_state_machine_edge_cases(self):
        """Test edge cases that specifically verify NumPy's state machine logic."""
        cases = [
            # Integer indices mixed with advanced indices - NumPy treats these together
            {
                "shape": (3, 4, 5, 6),
                "index": (0, [1, 2], slice(None), 3),
                "name": "Integer-fancy-slice-integer pattern",
            },
            {
                "shape": (4, 5, 6),
                "index": ([0, 1], 2, [3, 4]),
                "name": "Fancy-integer-fancy pattern",
            },
            # Ellipsis handling
            {
                "shape": (3, 4, 5, 6),
                "index": (..., [1, 2], slice(None)),
                "name": "Ellipsis before advanced index",
            },
            {
                "shape": (3, 4, 5, 6),
                "index": ([0, 1], ..., 2),
                "name": "Advanced index with ellipsis",
            },
            # Newaxis (None) handling
            {
                "shape": (3, 4, 5),
                "index": (None, [1, 2], slice(None)),
                "name": "Newaxis before advanced index",
            },
            {
                "shape": (3, 4, 5),
                "index": ([1, 2], None, slice(None)),
                "name": "Advanced index with newaxis",
            },
            # Multiple advanced indices with complex separations
            {
                "shape": (3, 4, 5, 6, 7),
                "index": ([0, 1], slice(None), [2, 3], slice(1, 3), [4, 5]),
                "name": "Multiple separated advanced indices",
            },
            # Edge case: all indices are advanced
            {
                "shape": (3, 4, 5),
                "index": ([0, 1], [2, 3], [1, 4]),
                "name": "All advanced indices",
            },
            # Zero-dimensional tensor cases
            {
                "shape": (2, 3, 4),
                "index": (torch.tensor(1), [1, 2]),
                "name": "Zero-dim tensor with list",
            },
            {
                "shape": (2, 3, 4),
                "index": (torch.tensor(1), slice(None), [1, 2]),
                "name": "Zero-dim tensor with slice and list",
            },
            {
                "shape": (2, 3, 4),
                "index": (torch.tensor(0), torch.tensor(1), [1, 2]),
                "name": "Multiple zero-dim tensors",
            },
            {
                "shape": (3, 4),
                "index": (torch.tensor(True),),
                "name": "Zero-dim bool tensor",
            },
        ]

        # Convert torch tensor indices to regular equivalents for numpy comparison
        numpy_cases = []
        for case in cases:
            if any(
                isinstance(idx, torch.Tensor) and idx.ndim == 0
                for idx in case["index"]
                if isinstance(idx, torch.Tensor)
            ):
                numpy_index = tuple(
                    idx.item()
                    if isinstance(idx, torch.Tensor) and idx.ndim == 0
                    else idx
                    for idx in case["index"]
                )
                numpy_cases.append(
                    {"shape": case["shape"], "index": numpy_index, "name": case["name"]}
                )
            else:
                numpy_cases.append(case)

        self._test_cases(numpy_cases, "NumPy state machine edge cases")


if __name__ == "__main__":
    run_tests()
