import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten, TreeSpec, LeafSpec
from torch.utils._pytree import _broadcast_to_and_flatten

class TestPytree(TestCase):
    def test_treespec_equality(self):
        self.assertTrue(LeafSpec() == LeafSpec())
        self.assertTrue(TreeSpec(list, None, []) == TreeSpec(list, None, []))
        self.assertTrue(TreeSpec(list, None, [LeafSpec()]) == TreeSpec(list, None, [LeafSpec()]))
        self.assertFalse(TreeSpec(tuple, None, []) == TreeSpec(list, None, []))
        self.assertTrue(TreeSpec(tuple, None, []) != TreeSpec(list, None, []))

    def test_flatten_unflatten_leaf(self):
        def run_test_with_leaf(leaf):
            values, treespec = tree_flatten(leaf)
            self.assertEqual(values, [leaf])
            self.assertEqual(treespec, LeafSpec())

            unflattened = tree_unflatten(values, treespec)
            self.assertEqual(unflattened, leaf)

        run_test_with_leaf(1)
        run_test_with_leaf(1.)
        run_test_with_leaf(None)
        run_test_with_leaf(bool)
        run_test_with_leaf(torch.randn(3, 3))

    def test_flatten_unflatten_list(self):
        def run_test(lst):
            expected_spec = TreeSpec(list, None, [LeafSpec() for _ in lst])
            values, treespec = tree_flatten(lst)
            self.assertTrue(isinstance(values, list))
            self.assertEqual(values, lst)
            self.assertEqual(treespec, expected_spec)

            unflattened = tree_unflatten(values, treespec)
            self.assertEqual(unflattened, lst)
            self.assertTrue(isinstance(unflattened, list))

        run_test([])
        run_test([1., 2])
        run_test([torch.tensor([1., 2]), 2, 10, 9, 11])

    def test_flatten_unflatten_tuple(self):
        def run_test(tup):
            expected_spec = TreeSpec(tuple, None, [LeafSpec() for _ in tup])
            values, treespec = tree_flatten(tup)
            self.assertTrue(isinstance(values, list))
            self.assertEqual(values, list(tup))
            self.assertEqual(treespec, expected_spec)

            unflattened = tree_unflatten(values, treespec)
            self.assertEqual(unflattened, tup)
            self.assertTrue(isinstance(unflattened, tuple))

        run_test(())
        run_test((1.,))
        run_test((1., 2))
        run_test((torch.tensor([1., 2]), 2, 10, 9, 11))

    def test_flatten_unflatten_dict(self):
        def run_test(tup):
            expected_spec = TreeSpec(dict, list(tup.keys()),
                                     [LeafSpec() for _ in tup.values()])
            values, treespec = tree_flatten(tup)
            self.assertTrue(isinstance(values, list))
            self.assertEqual(values, list(tup.values()))
            self.assertEqual(treespec, expected_spec)

            unflattened = tree_unflatten(values, treespec)
            self.assertEqual(unflattened, tup)
            self.assertTrue(isinstance(unflattened, dict))

        run_test({})
        run_test({'a': 1})
        run_test({'abcdefg': torch.randn(2, 3)})
        run_test({1: torch.randn(2, 3)})
        run_test({'a': 1, 'b': 2, 'c': torch.randn(2, 3)})

    def test_flatten_unflatten_nested(self):
        def run_test(pytree):
            values, treespec = tree_flatten(pytree)
            self.assertTrue(isinstance(values, list))
            self.assertEqual(len(values), treespec.num_leaves)

            # NB: python basic data structures (dict list tuple) all have
            # contents equality defined on them, so the following works for them.
            unflattened = tree_unflatten(values, treespec)
            self.assertEqual(unflattened, pytree)

        cases = [
            [()],
            ([],),
            {'a': ()},
            {'a': 0, 'b': [{'c': 1}]},
            {'a': 0, 'b': [1, {'c': 2}, torch.randn(3)], 'c': (torch.randn(2, 3), 1)},
        ]


    def test_treemap(self):
        def run_test(pytree):
            def f(x):
                return x * 3
            sm1 = sum(map(tree_flatten(pytree)[0], f))
            sm2 = tree_flatten(tree_map(f, pytree))[0]
            self.assertEqual(sm1, sm2)

            def invf(x):
                return x // 3

            self.assertEqual(tree_flatten(tree_flatten(pytree, f), invf), pytree)

            cases = [
                [()],
                ([],),
                {'a': ()},
                {'a': 1, 'b': [{'c': 2}]},
                {'a': 0, 'b': [2, {'c': 3}, 4], 'c': (5, 6)},
            ]
            for case in cases:
                run_test(case)


    def test_treespec_repr(self):
        # Check that it looks sane
        pytree = (0, [0, 0, 0])
        _, spec = tree_flatten(pytree)
        self.assertEqual(
            repr(spec), 'TreeSpec(tuple, None, [*, TreeSpec(list, None, [*, *, *])])')

    def test_broadcast_to_and_flatten(self):
        cases = [
            (1, (), []),

            # Same (flat) structures
            ((1,), (0,), [1]),
            ([1], [0], [1]),
            ((1, 2, 3), (0, 0, 0), [1, 2, 3]),
            ({'a': 1, 'b': 2}, {'a': 0, 'b': 0}, [1, 2]),

            # Mismatched (flat) structures
            ([1], (0,), None),
            ([1], (0,), None),
            ((1,), [0], None),
            ((1, 2, 3), (0, 0), None),
            ({'a': 1, 'b': 2}, {'a': 0}, None),
            ({'a': 1, 'b': 2}, {'a': 0, 'c': 0}, None),
            ({'a': 1, 'b': 2}, {'a': 0, 'b': 0, 'c': 0}, None),

            # Same (nested) structures
            ((1, [2, 3]), (0, [0, 0]), [1, 2, 3]),
            ((1, [(2, 3), 4]), (0, [(0, 0), 0]), [1, 2, 3, 4]),

            # Mismatched (nested) structures
            ((1, [2, 3]), (0, (0, 0)), None),
            ((1, [2, 3]), (0, [0, 0, 0]), None),

            # Broadcasting single value
            (1, (0, 0, 0), [1, 1, 1]),
            (1, [0, 0, 0], [1, 1, 1]),
            (1, {'a': 0, 'b': 0}, [1, 1]),
            (1, (0, [0, [0]], 0), [1, 1, 1, 1]),
            (1, (0, [0, [0, [], [[[0]]]]], 0), [1, 1, 1, 1, 1]),

            # Broadcast multiple things
            ((1, 2), ([0, 0, 0], [0, 0]), [1, 1, 1, 2, 2]),
            ((1, 2), ([0, [0, 0], 0], [0, 0]), [1, 1, 1, 1, 2, 2]),
            (([1, 2, 3], 4), ([0, [0, 0], 0], [0, 0]), [1, 2, 2, 3, 4, 4]),
        ]
        for pytree, to_pytree, expected in cases:
            _, to_spec = tree_flatten(to_pytree)
            result = _broadcast_to_and_flatten(pytree, to_spec)
            self.assertEqual(result, expected, msg=str([pytree, to_spec, expected]))


if __name__ == '__main__':
    run_tests()
