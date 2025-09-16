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

    def _generate_standard_indices(self, shape):
        """Generate standard index patterns for given shape."""
        if len(shape) == 1:
            return [([0, 2, 4],), ([1, 3, 5, 7],), ([0, 0, 1, 1],)]
        elif len(shape) == 2:
            return [
                (slice(None), [0]),
                ([0], slice(None)),
                (slice(1, 3), [0, 2]),
                ([0, 2], slice(1, 4)),
                ([0, 2], 1),
                (1, [0, 2, 4]),
            ]
        elif len(shape) == 3:
            return [
                ([0, 2], slice(None), slice(None)),
                (slice(None), [0, 2], slice(None)),
                (slice(None), slice(None), [0, 2, 4]),
                ([0, 1], slice(None), 2),
                ([0, 2], 1, slice(None)),
                (1, [1, 3], slice(None)),
                (slice(None), [0, 2], 3),
                ([0, 1], slice(None), [0, 2]),
                ([1, 2], slice(1, 3), [1, 3]),
                ([0, 1], 1, [1, 2]),
            ]
        elif len(shape) == 4:
            return [
                (slice(None), [0], 0, slice(None)),
                (slice(None), [0, 1], slice(None), 0),
                ([0, 1], slice(None), [0, 2], slice(None)),
                (slice(None), [0, 2], slice(None), [0, 3]),
                ([0], slice(None), slice(None), [1, 3]),
                (slice(None), slice(None), [0, 2], [1, 4]),
            ]
        return []

    def _generate_broadcast_indices(self, shape):
        """Generate broadcast-style index patterns."""
        if len(shape) >= 2:
            # Only generate patterns that fit within the shape bounds
            patterns = []
            if shape[0] >= 2 and shape[1] >= 4:
                patterns.append(([[0, 1], [0, 1]], [[0, 1], [2, 3]]))  # 2D broadcasting
            if shape[0] >= 3 and shape[1] >= 2:
                patterns.append(([0, 1, 2], [0, 1, 0]))  # Element selection
            return patterns
        return []

    def test_comprehensive_indexing(self):
        """Test comprehensive indexing patterns across multiple dimensions."""
        test_shapes = [(10,), (4, 6), (3, 4, 5), (2, 3, 4, 5), (5, 6, 7, 8, 9)]

        all_cases = []

        for shape in test_shapes:
            # Standard patterns
            for idx in self._generate_standard_indices(shape):
                all_cases.append({"shape": shape, "index": idx})

            # Broadcast patterns (for 2D+)
            if len(shape) >= 2:
                for idx in self._generate_broadcast_indices(shape):
                    all_cases.append({"shape": shape, "index": idx})

            # Separated indices patterns (for 3D+)
            if len(shape) >= 3:
                all_cases.extend(
                    [
                        {
                            "shape": shape,
                            "index": ([0, 1], slice(None), [1, 2]),
                            "name": "Separated indices",
                        },
                        {
                            "shape": shape,
                            "index": ([0, 1], [1, 2]),
                            "name": "Adjacent indices",
                        },
                    ]
                )

            # Edge cases with negative indices
            if len(shape) >= 3:
                all_cases.extend(
                    [
                        {"shape": shape, "index": ([0], slice(None), [1])},
                        {"shape": shape, "index": ([-1], slice(None), [-1])},
                        {"shape": shape, "index": ([0, -1], slice(None), [1, -1])},
                    ]
                )

        self._test_cases(all_cases, "Comprehensive indexing")

    def test_advanced_separation_patterns(self):
        """Test advanced separation patterns and edge cases."""
        cases = [
            # Complex multi-dimensional separations
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
                "name": "6D multiple separations",
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
                "name": "8D multiple separations",
            },
            # Current logic issues
            {
                "shape": (2, 3, 4, 5),
                "index": (0, slice(None), [1], slice(None)),
                "name": "Logic test: [0, :, [1], :]",
            },
            {
                "shape": (2, 3, 4, 5),
                "index": (0, [1], slice(None), 0),
                "name": "Logic test: [0, [1], :, 0]",
            },
            # High-dimensional edge cases
            {
                "shape": (5, 6, 7, 8, 9),
                "index": (slice(None), slice(None), [1, 2], slice(None), slice(None)),
                "name": "Single advanced index at middle position",
            },
            {
                "shape": (5, 6, 7, 8, 9),
                "index": ([1, 2], slice(None), slice(None), slice(None), slice(None)),
                "name": "Single advanced index at start",
            },
            {
                "shape": (5, 6, 7, 8, 9),
                "index": (slice(None), slice(None), slice(None), slice(None), [1, 2]),
                "name": "Single advanced index at end",
            },
            {
                "shape": (5, 6, 7, 8, 9),
                "index": (2, [1, 2], slice(None), slice(None), slice(None)),
                "name": "Advanced index with scalar before",
            },
            {
                "shape": (5, 6, 7, 8, 9),
                "index": (2, 3, [1, 2], 4, slice(None)),
                "name": "Advanced index with scalars around",
            },
        ]
        self._test_cases(cases, "Advanced separation patterns")

    def test_broadcast_and_numpy_compatibility(self):
        """Test broadcasting patterns and NumPy documentation examples."""
        base_shape = (10, 20, 30, 40, 50)

        # Standard broadcast indices
        indices_2d = {
            "ind_1": [[1, 2], [3, 4], [5, 6]],
            "ind_2": [[7, 8], [9, 10], [11, 12]],
        }
        indices_3d = {"ind_1": [[[1, 2]], [[3, 4]]], "ind_2": [[[5, 6]], [[7, 8]]]}
        indices_4d = {
            "ind_1": [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]],
            "ind_2": [[[[9, 10]], [[11, 12]]], [[[13, 14]], [[15, 16]]]],
        }

        cases = [
            # 2D broadcast patterns
            {
                "shape": base_shape,
                "index": (slice(None), indices_2d["ind_1"], indices_2d["ind_2"]),
                "name": "Adjacent 2D broadcast indices",
            },
            {
                "shape": base_shape,
                "index": (
                    slice(None),
                    indices_2d["ind_1"],
                    slice(None),
                    indices_2d["ind_2"],
                ),
                "name": "Separated 2D broadcast indices",
            },
            # 3D broadcast patterns
            {
                "shape": base_shape,
                "index": (
                    slice(None),
                    indices_3d["ind_1"],
                    slice(None),
                    indices_3d["ind_2"],
                ),
                "name": "3D broadcast creating 8D output",
            },
            # 4D broadcast patterns
            {
                "shape": base_shape,
                "index": (
                    slice(None),
                    indices_4d["ind_1"],
                    slice(None),
                    indices_4d["ind_2"],
                ),
                "name": "4D broadcast creating 9D output",
            },
            # NumPy documentation examples
            {
                "shape": (5, 7),
                "index": ([0, 2, 4], slice(1, 3)),
                "name": "NumPy docs: mixed advanced and slice",
            },
            {
                "shape": (4, 3),
                "index": ([0, 1, 2], [0, 1, 0]),
                "name": "NumPy docs: element selection",
            },
            {
                "shape": (4, 3),
                "index": ([[0], [3]], [0, 2]),
                "name": "NumPy docs: broadcasting case",
            },
        ]
        self._test_cases(cases, "Broadcast and NumPy compatibility")

    def test_special_index_types(self):
        """Test special index types including tensors, ellipsis, and newaxis."""
        cases = [
            # Ellipsis handling
            {
                "shape": (3, 4, 5, 6),
                "index": (..., [1, 2], slice(None)),
                "name": "Ellipsis with advanced index",
            },
            {
                "shape": (3, 4, 5, 6),
                "index": ([0, 1], ..., 2),
                "name": "Advanced index with ellipsis",
            },
            # Newaxis handling
            {
                "shape": (3, 4, 5),
                "index": (None, [1, 2], slice(None)),
                "name": "Newaxis with advanced index",
            },
            {
                "shape": (3, 4, 5),
                "index": ([1, 2], None, slice(None)),
                "name": "Advanced index with newaxis",
            },
            # Complex mixing
            {
                "shape": (3, 4, 5, 6),
                "index": (0, [1, 2], slice(None), 3),
                "name": "Integer-advanced-slice-integer",
            },
            {
                "shape": (4, 5, 6),
                "index": ([0, 1], 2, [3, 4]),
                "name": "Advanced-integer-advanced",
            },
            {
                "shape": (3, 4, 5, 6, 7),
                "index": ([0, 1], slice(None), [2, 3], slice(1, 3), [4, 5]),
                "name": "Multiple separated advanced indices",
            },
            {
                "shape": (3, 4, 5),
                "index": ([0, 1], [2, 3], [1, 4]),
                "name": "All advanced indices",
            },
            # Boolean indexing cases (issue #158134)
            # Tests for boolean values that trigger advanced indexing
            {
                "shape": (3, 4),
                "index": (True,),
                "name": "Boolean True indexing",
            },
            {
                "shape": (3, 4),
                "index": (False,),
                "name": "Boolean False indexing",
            },
            {
                "shape": (2, 3, 4),
                "index": (True, slice(None)),
                "name": "Boolean True with slice",
            },
            # Tuple indexing cases
            {
                "shape": (3, 4, 5),
                "index": ((0, 1), slice(None)),
                "name": "Tuple indexing with slice",
            },
            {
                "shape": (4, 5),
                "index": ((0, 1, 2),),
                "name": "Tuple indexing only",
            },
            {
                "shape": (3, 4, 5),
                "index": ((0, 1), (2, 3)),
                "name": "Multiple tuple indexing",
            },
        ]

        # Handle torch tensor cases separately
        torch_cases = [
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
            {
                "shape": (2, 3, 4),
                "index": (torch.tensor(False),),
                "name": "Zero-dim False tensor",
            },
        ]

        # Convert torch tensor indices for numpy comparison
        numpy_torch_cases = []
        for case in torch_cases:
            numpy_index = tuple(
                idx.item() if isinstance(idx, torch.Tensor) and idx.ndim == 0 else idx
                for idx in case["index"]
            )
            numpy_torch_cases.append(
                {"shape": case["shape"], "index": numpy_index, "name": case["name"]}
            )

        self._test_cases(cases + numpy_torch_cases, "Special index types")

    def test_ellipsis(self):
        """Tests containing ellipsis."""
        cases = [
            # Ellipsis + Basic indexing
            {
                "shape": (3, 4, 5),
                "index": (slice(None), 0, ..., slice(None)),
                "name": "empty ellipsis without advanced indexing",
            },
            {
                "shape": (3, 4, 5),
                "index": (slice(None), ..., 0),
                "name": "non-empty ellipsis without advanced indexing",
            },
            # Ellipsis + Advanced indexing without separation
            {
                "shape": (3, 4, 5),
                "index": (slice(None), ..., slice(None), (0, 1)),
                "name": "empty ellipsis without separation",
            },
            {
                "shape": (3, 4, 5),
                "index": (slice(None), ..., (0, 1)),
                "name": "non-empty ellipsis without separation",
            },
            # Ellipsis + Advanced indexing with separation
            {
                "shape": (3, 4, 5),
                "index": (slice(None), (0, 1), ..., (0, 1)),
                "name": "empty ellipsis separation",
            },
            {
                "shape": (1, 3, 4, 5),
                "index": (slice(None), (0, 1), ..., (0, 1)),
                "name": "non-empty ellipsis separation",
            },
            {
                "shape": (4, 3, 5),
                "index": (slice(None), ((0,), (1,)), ..., (0, 1)),
                "name": "empty ellipsis separation with 2-depth int sequence",
            },
            {
                "shape": (4, 3, 5, 6),
                "index": (slice(None), ((0,), (1,)), ..., (0, 1), slice(None)),
                "name": "empty ellipsis separation with 2-depth int sequence and end slice",
            },
            {
                "shape": (4, 3, 5, 6),
                "index": (slice(None), ((0,), (1,)), ..., (0, 1), (((0, 1), (1, 2)),)),
                "name": "empty ellipsis separation with 2 and 3-depth int sequence",
            },
            # Ellipsis + Boolean masks in advanced indexing with separation
            {
                "shape": (3, 4, 5),
                "index": (slice(None), True, True, True, ..., 0, 0),
                "name": "empty ellipsis separation with 0-dim boolean masks",
            },
            {
                "shape": (4, 3, 5),
                "index": (slice(None), (True, True, False), ..., (0, 1)),
                "name": "empty ellipsis separation with 1-dim boolean masks",
            },
            # TODO(manuelcandales) Fix issue #71673 and enable this case
            # {
            #     "shape": (1, 2, 2, 4, 5),
            #     "index": (slice(None), ((True, False), (True, True)), (0, 1, 2), ..., (0,)),
            #     "name": "empty ellipsis separation with 2-dim boolean masks",
            # },
        ]
        self._test_cases(cases, "Ellipsis and advanced indexing separation")


if __name__ == "__main__":
    run_tests()
