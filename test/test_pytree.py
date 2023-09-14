# Owner(s): ["module: pytree"]

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._pytree import (
    tree_flatten,
    tree_map,
    tree_unflatten,
    TreeSpec,
    LeafSpec,
    treespec_dumps,
    treespec_loads,
    _register_pytree_node,
)
import unittest
from torch.utils._pytree import _broadcast_to_and_flatten, tree_map_only, tree_all
from torch.utils._pytree import tree_any, tree_all_only, tree_any_only
from collections import namedtuple, OrderedDict
from torch.testing._internal.common_utils import parametrize, subtest, instantiate_parametrized_tests, TEST_WITH_TORCHDYNAMO

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

    def test_flatten_unflatten_odict(self):
        def run_test(odict):
            expected_spec = TreeSpec(
                OrderedDict,
                list(odict.keys()),
                [LeafSpec() for _ in odict.values()])
            values, treespec = tree_flatten(odict)
            self.assertTrue(isinstance(values, list))
            self.assertEqual(values, list(odict.values()))
            self.assertEqual(treespec, expected_spec)

            unflattened = tree_unflatten(values, treespec)
            self.assertEqual(unflattened, odict)
            self.assertTrue(isinstance(unflattened, OrderedDict))

        od = OrderedDict()
        run_test(od)

        od['b'] = 1
        od['a'] = torch.tensor(3.14)
        run_test(od)

    def test_flatten_unflatten_namedtuple(self):
        Point = namedtuple('Point', ['x', 'y'])

        def run_test(tup):
            expected_spec = TreeSpec(namedtuple, Point, [LeafSpec() for _ in tup])
            values, treespec = tree_flatten(tup)
            self.assertTrue(isinstance(values, list))
            self.assertEqual(values, list(tup))
            self.assertEqual(treespec, expected_spec)

            unflattened = tree_unflatten(values, treespec)
            self.assertEqual(unflattened, tup)
            self.assertTrue(isinstance(unflattened, Point))

        run_test(Point(1., 2))
        run_test(Point(torch.tensor(1.), 2))

    @parametrize("op", [
        subtest(torch.max, name='max'),
        subtest(torch.min, name='min'),
    ])
    def test_flatten_unflatten_return_type(self, op):
        x = torch.randn(3, 3)
        expected = op(x, dim=0)

        values, spec = tree_flatten(expected)
        # Check that values is actually List[Tensor] and not (ReturnType(...),)
        for value in values:
            self.assertTrue(isinstance(value, torch.Tensor))
        result = tree_unflatten(values, spec)

        self.assertEqual(type(result), type(expected))
        self.assertEqual(result, expected)

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


    def test_tree_only(self):
        self.assertEqual(tree_map_only(int, lambda x: x + 2, [0, "a"]), [2, "a"])


    def test_tree_all_any(self):
        self.assertTrue(tree_all(lambda x: x % 2, [1, 3]))
        self.assertFalse(tree_all(lambda x: x % 2, [0, 1]))
        self.assertTrue(tree_any(lambda x: x % 2, [0, 1]))
        self.assertFalse(tree_any(lambda x: x % 2, [0, 2]))
        self.assertTrue(tree_all_only(int, lambda x: x % 2, [1, 3, "a"]))
        self.assertFalse(tree_all_only(int, lambda x: x % 2, [0, 1, "a"]))
        self.assertTrue(tree_any_only(int, lambda x: x % 2, [0, 1, "a"]))
        self.assertFalse(tree_any_only(int, lambda x: x % 2, [0, 2, "a"]))


    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "Dynamo test in test_treespec_repr_dynamo.")
    def test_treespec_repr(self):
        # Check that it looks sane
        pytree = (0, [0, 0, [0]])
        _, spec = tree_flatten(pytree)
        self.assertEqual(repr(spec), ("TreeSpec(tuple, None, [*,\n"
                                      "  TreeSpec(list, None, [*,\n"
                                      "    *,\n"
                                      "    TreeSpec(list, None, [*])])])"))

    @unittest.skipIf(not TEST_WITH_TORCHDYNAMO, "Eager test in test_treespec_repr.")
    def test_treespec_repr_dynamo(self):
        # Check that it looks sane
        pytree = (0, [0, 0, [0]])
        _, spec = tree_flatten(pytree)
        self.assertExpectedInline(repr(spec),
                                  """\
TreeSpec(TupleVariable, None, [*,
  TreeSpec(ListVariable, None, [*,
    *,
    TreeSpec(ListVariable, None, [*])])])""")

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

    @parametrize("spec", [
        TreeSpec(list, None, []),
        TreeSpec(tuple, None, []),
        TreeSpec(dict, [], []),
        TreeSpec(list, None, [LeafSpec()]),
        TreeSpec(list, None, [LeafSpec(), LeafSpec()]),
        TreeSpec(tuple, None, [LeafSpec(), LeafSpec(), LeafSpec()]),
        TreeSpec(dict, ['a', 'b', 'c'], [LeafSpec(), LeafSpec(), LeafSpec()]),
        TreeSpec(OrderedDict, ['a', 'b', 'c'], [
            TreeSpec(
                tuple,
                None,
                [LeafSpec(), LeafSpec()]
            ),
            LeafSpec(),
            TreeSpec(
                dict,
                ['a', 'b', 'c'],
                [LeafSpec(), LeafSpec(), LeafSpec()]
            ),
        ]),
        TreeSpec(list, None, [
            TreeSpec(tuple, None, [
                LeafSpec(),
                LeafSpec(),
                TreeSpec(list, None, [
                    LeafSpec(),
                    LeafSpec(),
                ]),
            ]),
        ]),
    ],)
    def test_pytree_serialize(self, spec):
        serialized_spec = treespec_dumps(spec)
        self.assertTrue(isinstance(serialized_spec, str))
        self.assertTrue(spec == treespec_loads(serialized_spec))

    def test_pytree_serialize_namedtuple(self):
        Point = namedtuple("Point", ["x", "y"])
        spec = TreeSpec(namedtuple, Point, [LeafSpec(), LeafSpec()])

        roundtrip_spec = treespec_loads(treespec_dumps(spec))
        # The context in the namedtuple is different now because we recreated
        # the namedtuple type.
        self.assertEqual(spec.context._fields, roundtrip_spec.context._fields)

    def test_pytree_custom_type_serialize(self):
        class DummyType:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        _register_pytree_node(
            DummyType,
            lambda dummy: ([dummy.x, dummy.y], None),
            lambda xs, _: Dummy(*xs),
            to_dumpable_context=lambda context: "moo",
            from_dumpable_context=lambda dumpable_context: None,
        )
        spec = TreeSpec(DummyType, None, [LeafSpec(), LeafSpec()])
        serialized_spec = treespec_dumps(spec, 1)
        self.assertTrue("moo" in serialized_spec)
        roundtrip_spec = treespec_loads(serialized_spec)
        self.assertEqual(roundtrip_spec, spec)

    def test_pytree_serialize_register_bad(self):
        class DummyType:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        with self.assertRaisesRegex(ValueError, "Both to_dumpable_context and from_dumpable_context"):
            _register_pytree_node(
                DummyType,
                lambda dummy: ([dummy.x, dummy.y], None),
                lambda xs, _: Dummy(*xs),
                to_dumpable_context=lambda context: "moo",
            )

    def test_pytree_context_serialize_bad(self):
        class DummyType:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        _register_pytree_node(
            DummyType,
            lambda dummy: ([dummy.x, dummy.y], None),
            lambda xs, _: Dummy(*xs),
            to_dumpable_context=lambda context: DummyType,
            from_dumpable_context=lambda dumpable_context: None,
        )

        spec = TreeSpec(DummyType, None, [LeafSpec(), LeafSpec()])

        with self.assertRaisesRegex(TypeError, "Object of type type is not JSON serializable"):
            treespec_dumps(spec)

    def test_pytree_serialize_bad_input(self):
        with self.assertRaises(AttributeError):
            treespec_dumps("random_blurb")

    def test_pytree_serialize_bad_protocol(self):
        import json

        Point = namedtuple("Point", ["x", "y"])
        spec = TreeSpec(namedtuple, Point, [LeafSpec(), LeafSpec()])

        with self.assertRaisesRegex(ValueError, "Unknown protocol"):
            treespec_dumps(spec, -1)

        serialized_spec = treespec_dumps(spec)
        protocol, data = json.loads(serialized_spec)
        bad_protocol_serialized_spec = json.dumps((-1, data))

        with self.assertRaisesRegex(ValueError, "Unknown protocol"):
            treespec_loads(bad_protocol_serialized_spec)

    def test_saved_serialized(self):
        complicated_spec = TreeSpec(OrderedDict, [1, 2, 3], [
            TreeSpec(
                tuple,
                None,
                [LeafSpec(), LeafSpec()]
            ),
            LeafSpec(),
            TreeSpec(
                dict,
                [4, 5, 6],
                [LeafSpec(), LeafSpec(), LeafSpec()]
            ),
        ])

        serialized_spec = treespec_dumps(complicated_spec)
        saved_spec = (
            '[1, {"type": "collections.OrderedDict", "context": "[1, 2, 3]", '
            '"children_spec": [{"type": "builtins.tuple", "context": "null", '
            '"children_spec": [{"type": null, "context": null, '
            '"children_spec": []}, {"type": null, "context": null, '
            '"children_spec": []}]}, {"type": null, "context": null, '
            '"children_spec": []}, {"type": "builtins.dict", "context": '
            '"[4, 5, 6]", "children_spec": [{"type": null, "context": null, '
            '"children_spec": []}, {"type": null, "context": null, "children_spec": '
            '[]}, {"type": null, "context": null, "children_spec": []}]}]}]'
        )
        self.assertEqual(serialized_spec, saved_spec)
        self.assertEqual(complicated_spec, treespec_loads(saved_spec))


instantiate_parametrized_tests(TestPytree)

if __name__ == '__main__':
    run_tests()
