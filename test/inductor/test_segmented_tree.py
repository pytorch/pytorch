# Owner(s): ["module: inductor"]

from hypothesis import given, strategies as st

from torch._inductor.codegen.segmented_tree import SegmentedTree
from torch._inductor.test_case import run_tests, TestCase


# Helper functions for operations
def max_op(a, b):
    return max(a, b)


def add_op(a, b):
    return a + b


# Naive implementations for reference
def naive_range_max(arr, start, end):
    return max(arr[start : end + 1])


def naive_range_update(arr, start, end, value):
    for i in range(start, end + 1):
        arr[i] += value


# Strategies for hypothesis testing
positive_integers = st.lists(
    st.integers(min_value=1, max_value=100), min_size=1, max_size=50
)


def valid_range_indices(array_length):
    return st.tuples(
        st.integers(min_value=0, max_value=array_length - 1),
        st.integers(min_value=0, max_value=array_length - 1),
    ).map(lambda x: (min(x), max(x)))


update_values = st.integers(min_value=1, max_value=50)


class TestSegmentedTree(TestCase):
    # Basic construction and initialization tests
    def test_basic_construction(self):
        values = [1, 3, 5, 7, 9]
        tree = SegmentedTree(values, add_op, max_op, 0)
        if tree.summarize_range(0, 4) != 9:
            raise AssertionError(f"Expected 9, got {tree.summarize_range(0, 4)}")

    def test_empty_array(self):
        with self.assertRaises(ValueError):
            SegmentedTree([], add_op, max_op, 0)

    # Property-based tests
    @given(values=positive_integers)
    def test_max_query_matches_naive(self, values):
        tree = SegmentedTree(values, add_op, max_op, 0)

        for start in range(len(values)):
            for end in range(start, len(values)):
                expected = naive_range_max(values, start, end)
                actual = tree.summarize_range(start, end)
                if actual != expected:
                    raise AssertionError(
                        f"Range [{start}:{end}] expected {expected}, got {actual}"
                    )

    @given(
        values=positive_integers, range_indices=st.data(), update_value=update_values
    )
    def test_range_update(self, values, range_indices, update_value):
        # Create a copy for naive implementation
        naive_values = values.copy()

        # Create segment tree
        tree = SegmentedTree(values, add_op, max_op, 0)

        # Get valid range indices
        start, end = range_indices.draw(valid_range_indices(len(values)))

        # Apply updates
        tree.update_range(start, end, update_value)
        naive_range_update(naive_values, start, end, update_value)

        # Verify all possible ranges
        for i in range(len(values)):
            for j in range(i, len(values)):
                expected = naive_range_max(naive_values, i, j)
                actual = tree.summarize_range(i, j)
                if actual != expected:
                    raise AssertionError(
                        f"After update, range [{i}:{j}] expected {expected}, got {actual}"
                    )

    @given(values=positive_integers, range_data=st.data())
    def test_multiple_operations(self, values, range_data):
        # Create a copy for naive implementation
        naive_values = values.copy()
        tree = SegmentedTree(values, add_op, max_op, 0)

        # Perform multiple operations
        num_operations = 5
        for _ in range(num_operations):
            # Randomly choose between query and update
            operation_type = range_data.draw(st.sampled_from(["query", "update"]))
            start, end = range_data.draw(valid_range_indices(len(values)))

            if operation_type == "query":
                expected = naive_range_max(naive_values, start, end)
                actual = tree.summarize_range(start, end)
                if actual != expected:
                    raise AssertionError(
                        f"Range query [{start}:{end}] expected {expected}, got {actual}"
                    )
            else:  # update
                update_value = range_data.draw(update_values)
                tree.update_range(start, end, update_value)
                naive_range_update(naive_values, start, end, update_value)

    def test_single_element_ranges(self):
        values = [1, 3, 5, 7, 9]
        tree = SegmentedTree(values, add_op, max_op, 0)

        for i in range(len(values)):
            if tree.summarize_range(i, i) != values[i]:
                raise AssertionError(f"Single element range at index {i} failed")

    def test_full_array_range(self):
        values = [1, 3, 5, 7, 9]
        tree = SegmentedTree(values, add_op, max_op, 0)

        # Test querying the entire array
        if tree.summarize_range(0, len(values) - 1) != max(values):
            raise AssertionError(
                f"Expected {max(values)}, got {tree.summarize_range(0, len(values) - 1)}"
            )

        # Update the entire array and test again
        update_value = 10
        tree.update_range(0, len(values) - 1, update_value)
        expected = max([v + update_value for v in values])
        if tree.summarize_range(0, len(values) - 1) != expected:
            raise AssertionError(
                f"Expected {expected}, got {tree.summarize_range(0, len(values) - 1)}"
            )

    def test_boundary_conditions(self):
        values = [1, 3, 5, 7, 9]
        tree = SegmentedTree(values, add_op, max_op, 0)

        # Test first element
        if tree.summarize_range(0, 0) != values[0]:
            raise AssertionError(
                f"Expected {values[0]}, got {tree.summarize_range(0, 0)}"
            )

        # Test last element
        if tree.summarize_range(len(values) - 1, len(values) - 1) != values[-1]:
            raise AssertionError(
                f"Expected {values[-1]}, got {tree.summarize_range(len(values) - 1, len(values) - 1)}"
            )

        # Test first two elements
        if tree.summarize_range(0, 1) != max(values[0:2]):
            raise AssertionError(
                f"Expected {max(values[0:2])}, got {tree.summarize_range(0, 1)}"
            )

        # Test last two elements
        if tree.summarize_range(len(values) - 2, len(values) - 1) != max(values[-2:]):
            raise AssertionError(
                f"Expected {max(values[-2:])}, got {tree.summarize_range(len(values) - 2, len(values) - 1)}"
            )

    def test_invalid_ranges(self):
        values = [1, 3, 5, 7, 9]
        tree = SegmentedTree(values, add_op, max_op, 0)

        # Test start > end
        with self.assertRaises(ValueError):
            tree.summarize_range(3, 2)

        with self.assertRaises(ValueError):
            tree.update_range(4, 2, 10)

    def test_out_of_bounds(self):
        values = [1, 3, 5, 7, 9]
        tree = SegmentedTree(values, add_op, max_op, 0)

        # Test negative indices
        with self.assertRaises(ValueError):
            tree.summarize_range(-1, 3)

        with self.assertRaises(ValueError):
            tree.summarize_range(0, -1)

        # Test indices >= n
        with self.assertRaises(ValueError):
            tree.summarize_range(0, len(values))

        with self.assertRaises(ValueError):
            tree.summarize_range(len(values), len(values) + 1)

        # Test update with out of bounds indices
        with self.assertRaises(ValueError):
            tree.update_range(-1, 3, 10)

        with self.assertRaises(ValueError):
            tree.update_range(0, len(values), 10)

    def test_overlapping_updates(self):
        values = [1, 3, 5, 7, 9]
        naive_values = values.copy()
        tree = SegmentedTree(values, add_op, max_op, 0)

        # Apply overlapping updates
        tree.update_range(0, 2, 5)  # Update [0, 1, 2]
        naive_range_update(naive_values, 0, 2, 5)

        tree.update_range(1, 3, 3)  # Update [1, 2, 3]
        naive_range_update(naive_values, 1, 3, 3)

        # Verify all possible ranges
        for i in range(len(values)):
            for j in range(i, len(values)):
                expected = naive_range_max(naive_values, i, j)
                actual = tree.summarize_range(i, j)
                if actual != expected:
                    raise AssertionError(
                        f"After overlapping updates, range [{i}:{j}] expected {expected}, got {actual}"
                    )

    def test_sequential_updates_and_queries(self):
        values = [2, 4, 6, 8, 10, 12, 14]
        naive_values = values.copy()
        tree = SegmentedTree(values, add_op, max_op, 0)

        # Sequence of operations
        operations = [
            ("update", 1, 3, 5),  # Update range [1, 2, 3] with +5
            ("query", 0, 4),  # Query range [0, 1, 2, 3, 4]
            ("update", 2, 5, 3),  # Update range [2, 3, 4, 5] with +3
            ("query", 1, 3),  # Query range [1, 2, 3]
            ("update", 0, 6, 2),  # Update entire array with +2
            ("query", 0, 6),  # Query entire array
            ("query", 3, 5),  # Query range [3, 4, 5]
        ]

        for op in operations:
            if op[0] == "update":
                _, start, end, value = op
                tree.update_range(start, end, value)
                naive_range_update(naive_values, start, end, value)

                # Verify tree state after update
                for i in range(len(values)):
                    for j in range(i, len(values)):
                        expected = naive_range_max(naive_values, i, j)
                        actual = tree.summarize_range(i, j)
                        if actual != expected:
                            raise AssertionError(
                                f"After update ({start}, {end}, {value}), query [{i}:{j}] expected {expected}, got {actual}"
                            )
            else:  # query
                _, start, end = op
                expected = naive_range_max(naive_values, start, end)
                if tree.summarize_range(start, end) != expected:
                    raise AssertionError(
                        f"Query [{start}:{end}] expected {expected}, got {tree.summarize_range(start, end)}"
                    )


if __name__ == "__main__":
    run_tests()
