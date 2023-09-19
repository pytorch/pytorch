# Owner(s): ["module: pytree"]

import pickle
import unittest
from collections import namedtuple, OrderedDict

import torch
import torch.utils._pytree as py_pytree
import torch.utils.pytree as cxx_pytree
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


GlobalPoint = namedtuple("GlobalPoint", ["x", "y"])


class GlobalDummyType:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class TestPytree(TestCase):
    def test_treespec_equality(self):
        self.assertTrue(
            py_pytree.LeafSpec() == py_pytree.LeafSpec(),
        )
        self.assertTrue(
            py_pytree.TreeSpec(list, None, []) == py_pytree.TreeSpec(list, None, []),
        )
        self.assertTrue(
            py_pytree.TreeSpec(list, None, [py_pytree.LeafSpec()])
            == py_pytree.TreeSpec(list, None, [py_pytree.LeafSpec()]),
        )
        self.assertFalse(
            py_pytree.TreeSpec(tuple, None, []) == py_pytree.TreeSpec(list, None, []),
        )
        self.assertTrue(
            py_pytree.TreeSpec(tuple, None, []) != py_pytree.TreeSpec(list, None, []),
        )

    def test_flatten_unflatten_leaf(self):
        def run_test_with_leaf(leaf):
            values, treespec = py_pytree.tree_flatten(leaf)
            self.assertEqual(values, [leaf])
            self.assertEqual(treespec, py_pytree.LeafSpec())

            unflattened = py_pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, leaf)

        run_test_with_leaf(1)
        run_test_with_leaf(1.0)
        run_test_with_leaf(None)
        run_test_with_leaf(bool)
        run_test_with_leaf(torch.randn(3, 3))

    def test_flatten_unflatten_list(self):
        def run_test(lst):
            expected_spec = py_pytree.TreeSpec(
                list, None, [py_pytree.LeafSpec() for _ in lst]
            )
            values, treespec = py_pytree.tree_flatten(lst)
            self.assertTrue(isinstance(values, list))
            self.assertEqual(values, lst)
            self.assertEqual(treespec, expected_spec)

            unflattened = py_pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, lst)
            self.assertTrue(isinstance(unflattened, list))

        run_test([])
        run_test([1.0, 2])
        run_test([torch.tensor([1.0, 2]), 2, 10, 9, 11])

    def test_flatten_unflatten_tuple(self):
        def run_test(tup):
            expected_spec = py_pytree.TreeSpec(
                tuple, None, [py_pytree.LeafSpec() for _ in tup]
            )
            values, treespec = py_pytree.tree_flatten(tup)
            self.assertTrue(isinstance(values, list))
            self.assertEqual(values, list(tup))
            self.assertEqual(treespec, expected_spec)

            unflattened = py_pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, tup)
            self.assertTrue(isinstance(unflattened, tuple))

        run_test(())
        run_test((1.0,))
        run_test((1.0, 2))
        run_test((torch.tensor([1.0, 2]), 2, 10, 9, 11))

    def test_flatten_unflatten_odict(self):
        def run_test(odict):
            expected_spec = py_pytree.TreeSpec(
                OrderedDict,
                list(odict.keys()),
                [py_pytree.LeafSpec() for _ in odict.values()],
            )
            values, treespec = py_pytree.tree_flatten(odict)
            self.assertTrue(isinstance(values, list))
            self.assertEqual(values, list(odict.values()))
            self.assertEqual(treespec, expected_spec)

            unflattened = py_pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, odict)
            self.assertTrue(isinstance(unflattened, OrderedDict))

        od = OrderedDict()
        run_test(od)

        od["b"] = 1
        od["a"] = torch.tensor(3.14)
        run_test(od)

    def test_flatten_unflatten_namedtuple(self):
        Point = namedtuple("Point", ["x", "y"])

        def run_test(tup):
            expected_spec = py_pytree.TreeSpec(
                namedtuple, Point, [py_pytree.LeafSpec() for _ in tup]
            )
            values, treespec = py_pytree.tree_flatten(tup)
            self.assertTrue(isinstance(values, list))
            self.assertEqual(values, list(tup))
            self.assertEqual(treespec, expected_spec)

            unflattened = py_pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, tup)
            self.assertTrue(isinstance(unflattened, Point))

        run_test(Point(1.0, 2))
        run_test(Point(torch.tensor(1.0), 2))

    @parametrize(
        "op",
        [
            subtest(torch.max, name="max"),
            subtest(torch.min, name="min"),
        ],
    )
    def test_flatten_unflatten_return_type(self, op):
        x = torch.randn(3, 3)
        expected = op(x, dim=0)

        values, spec = py_pytree.tree_flatten(expected)
        # Check that values is actually List[Tensor] and not (ReturnType(...),)
        for value in values:
            self.assertTrue(isinstance(value, torch.Tensor))
        result = py_pytree.tree_unflatten(values, spec)

        self.assertEqual(type(result), type(expected))
        self.assertEqual(result, expected)

    def test_flatten_unflatten_dict(self):
        def run_test(dct):
            expected_spec = py_pytree.TreeSpec(
                dict, list(dct.keys()), [py_pytree.LeafSpec() for _ in dct.values()]
            )
            values, treespec = py_pytree.tree_flatten(dct)
            self.assertTrue(isinstance(values, list))
            self.assertEqual(values, list(dct.values()))
            self.assertEqual(treespec, expected_spec)

            unflattened = py_pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, dct)
            self.assertTrue(isinstance(unflattened, dict))

        run_test({})
        run_test({"a": 1})
        run_test({"abcdefg": torch.randn(2, 3)})
        run_test({1: torch.randn(2, 3)})
        run_test({"a": 1, "b": 2, "c": torch.randn(2, 3)})

    def test_flatten_unflatten_nested(self):
        def run_test(pytree):
            values, treespec = py_pytree.tree_flatten(pytree)
            self.assertTrue(isinstance(values, list))
            self.assertEqual(len(values), treespec.num_leaves)

            # NB: python basic data structures (dict list tuple) all have
            # contents equality defined on them, so the following works for them.
            unflattened = py_pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, pytree)

        cases = [
            [()],
            ([],),
            {"a": ()},
            {"a": 0, "b": [{"c": 1}]},
            {"a": 0, "b": [1, {"c": 2}, torch.randn(3)], "c": (torch.randn(2, 3), 1)},
        ]
        for case in cases:
            run_test(case)

    def test_treemap(self):
        def run_test(pytree):
            def f(x):
                return x * 3

            sm1 = sum(map(f, py_pytree.tree_flatten(pytree)[0]))
            sm2 = sum(py_pytree.tree_flatten(py_pytree.tree_map(f, pytree))[0])
            self.assertEqual(sm1, sm2)

            def invf(x):
                return x // 3

            self.assertEqual(
                py_pytree.tree_map(invf, py_pytree.tree_map(f, pytree)),
                pytree,
            )

        cases = [
            [()],
            ([],),
            {"a": ()},
            {"a": 1, "b": [{"c": 2}]},
            {"a": 0, "b": [2, {"c": 3}, 4], "c": (5, 6)},
        ]
        for case in cases:
            run_test(case)

    def test_tree_only(self):
        self.assertEqual(
            py_pytree.tree_map_only(int, lambda x: x + 2, [0, "a"]), [2, "a"]
        )

    def test_tree_all_any(self):
        self.assertTrue(py_pytree.tree_all(lambda x: x % 2, [1, 3]))
        self.assertFalse(py_pytree.tree_all(lambda x: x % 2, [0, 1]))
        self.assertTrue(py_pytree.tree_any(lambda x: x % 2, [0, 1]))
        self.assertFalse(py_pytree.tree_any(lambda x: x % 2, [0, 2]))
        self.assertTrue(py_pytree.tree_all_only(int, lambda x: x % 2, [1, 3, "a"]))
        self.assertFalse(py_pytree.tree_all_only(int, lambda x: x % 2, [0, 1, "a"]))
        self.assertTrue(py_pytree.tree_any_only(int, lambda x: x % 2, [0, 1, "a"]))
        self.assertFalse(py_pytree.tree_any_only(int, lambda x: x % 2, [0, 2, "a"]))

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "Dynamo test in test_treespec_repr_dynamo.")
    def test_treespec_repr(self):
        # Check that it looks sane
        pytree = (0, [0, 0, [0]])
        _, spec = py_pytree.tree_flatten(pytree)
        self.assertEqual(
            repr(spec),
            (
                "TreeSpec(tuple, None, [*,\n"
                "  TreeSpec(list, None, [*,\n"
                "    *,\n"
                "    TreeSpec(list, None, [*])])])"
            ),
        )

    @unittest.skipIf(not TEST_WITH_TORCHDYNAMO, "Eager test in test_treespec_repr.")
    def test_treespec_repr_dynamo(self):
        # Check that it looks sane
        pytree = (0, [0, 0, [0]])
        _, spec = py_pytree.tree_flatten(pytree)
        self.assertExpectedInline(
            repr(spec),
            """\
TreeSpec(TupleVariable, None, [*,
  TreeSpec(ListVariable, None, [*,
    *,
    TreeSpec(ListVariable, None, [*])])])""",
        )

    def test_broadcast_to_and_flatten(self):
        cases = [
            (1, (), []),
            # Same (flat) structures
            ((1,), (0,), [1]),
            ([1], [0], [1]),
            ((1, 2, 3), (0, 0, 0), [1, 2, 3]),
            ({"a": 1, "b": 2}, {"a": 0, "b": 0}, [1, 2]),
            # Mismatched (flat) structures
            ([1], (0,), None),
            ([1], (0,), None),
            ((1,), [0], None),
            ((1, 2, 3), (0, 0), None),
            ({"a": 1, "b": 2}, {"a": 0}, None),
            ({"a": 1, "b": 2}, {"a": 0, "c": 0}, None),
            ({"a": 1, "b": 2}, {"a": 0, "b": 0, "c": 0}, None),
            # Same (nested) structures
            ((1, [2, 3]), (0, [0, 0]), [1, 2, 3]),
            ((1, [(2, 3), 4]), (0, [(0, 0), 0]), [1, 2, 3, 4]),
            # Mismatched (nested) structures
            ((1, [2, 3]), (0, (0, 0)), None),
            ((1, [2, 3]), (0, [0, 0, 0]), None),
            # Broadcasting single value
            (1, (0, 0, 0), [1, 1, 1]),
            (1, [0, 0, 0], [1, 1, 1]),
            (1, {"a": 0, "b": 0}, [1, 1]),
            (1, (0, [0, [0]], 0), [1, 1, 1, 1]),
            (1, (0, [0, [0, [], [[[0]]]]], 0), [1, 1, 1, 1, 1]),
            # Broadcast multiple things
            ((1, 2), ([0, 0, 0], [0, 0]), [1, 1, 1, 2, 2]),
            ((1, 2), ([0, [0, 0], 0], [0, 0]), [1, 1, 1, 1, 2, 2]),
            (([1, 2, 3], 4), ([0, [0, 0], 0], [0, 0]), [1, 2, 2, 3, 4, 4]),
        ]
        for pytree, to_pytree, expected in cases:
            _, to_spec = py_pytree.tree_flatten(to_pytree)
            result = py_pytree._broadcast_to_and_flatten(pytree, to_spec)
            self.assertEqual(result, expected, msg=str([pytree, to_spec, expected]))

    @parametrize(
        "spec",
        [
            py_pytree.TreeSpec(list, None, []),
            py_pytree.TreeSpec(tuple, None, []),
            py_pytree.TreeSpec(dict, [], []),
            py_pytree.TreeSpec(list, None, [py_pytree.LeafSpec()]),
            py_pytree.TreeSpec(
                list, None, [py_pytree.LeafSpec(), py_pytree.LeafSpec()]
            ),
            py_pytree.TreeSpec(
                tuple,
                None,
                [py_pytree.LeafSpec(), py_pytree.LeafSpec(), py_pytree.LeafSpec()],
            ),
            py_pytree.TreeSpec(
                dict,
                ["a", "b", "c"],
                [py_pytree.LeafSpec(), py_pytree.LeafSpec(), py_pytree.LeafSpec()],
            ),
            py_pytree.TreeSpec(
                OrderedDict,
                ["a", "b", "c"],
                [
                    py_pytree.TreeSpec(
                        tuple, None, [py_pytree.LeafSpec(), py_pytree.LeafSpec()]
                    ),
                    py_pytree.LeafSpec(),
                    py_pytree.TreeSpec(
                        dict,
                        ["a", "b", "c"],
                        [
                            py_pytree.LeafSpec(),
                            py_pytree.LeafSpec(),
                            py_pytree.LeafSpec(),
                        ],
                    ),
                ],
            ),
            py_pytree.TreeSpec(
                list,
                None,
                [
                    py_pytree.TreeSpec(
                        tuple,
                        None,
                        [
                            py_pytree.LeafSpec(),
                            py_pytree.LeafSpec(),
                            py_pytree.TreeSpec(
                                list,
                                None,
                                [
                                    py_pytree.LeafSpec(),
                                    py_pytree.LeafSpec(),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
    def test_pytree_serialize(self, spec):
        serialized_spec = py_pytree.treespec_dumps(spec)
        self.assertTrue(isinstance(serialized_spec, str))
        self.assertTrue(spec == py_pytree.treespec_loads(serialized_spec))

    def test_pytree_serialize_namedtuple(self):
        Point = namedtuple("Point", ["x", "y"])
        spec = py_pytree.TreeSpec(
            namedtuple, Point, [py_pytree.LeafSpec(), py_pytree.LeafSpec()]
        )

        roundtrip_spec = py_pytree.treespec_loads(py_pytree.treespec_dumps(spec))
        # The context in the namedtuple is different now because we recreated
        # the namedtuple type.
        self.assertEqual(spec.context._fields, roundtrip_spec.context._fields)

    def test_pytree_custom_type_serialize(self):
        class DummyType:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        py_pytree._register_pytree_node(
            DummyType,
            lambda dummy: ([dummy.x, dummy.y], None),
            lambda xs, _: DummyType(*xs),
            to_dumpable_context=lambda context: "moo",
            from_dumpable_context=lambda dumpable_context: None,
        )
        spec = py_pytree.TreeSpec(
            DummyType, None, [py_pytree.LeafSpec(), py_pytree.LeafSpec()]
        )
        serialized_spec = py_pytree.treespec_dumps(spec, 1)
        self.assertTrue("moo" in serialized_spec)
        roundtrip_spec = py_pytree.treespec_loads(serialized_spec)
        self.assertEqual(roundtrip_spec, spec)

    def test_pytree_serialize_register_bad(self):
        class DummyType:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        with self.assertRaisesRegex(
            ValueError, "Both to_dumpable_context and from_dumpable_context"
        ):
            py_pytree._register_pytree_node(
                DummyType,
                lambda dummy: ([dummy.x, dummy.y], None),
                lambda xs, _: DummyType(*xs),
                to_dumpable_context=lambda context: "moo",
            )

    def test_pytree_context_serialize_bad(self):
        class DummyType:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        py_pytree._register_pytree_node(
            DummyType,
            lambda dummy: ([dummy.x, dummy.y], None),
            lambda xs, _: DummyType(*xs),
            to_dumpable_context=lambda context: DummyType,
            from_dumpable_context=lambda dumpable_context: None,
        )

        spec = py_pytree.TreeSpec(
            DummyType, None, [py_pytree.LeafSpec(), py_pytree.LeafSpec()]
        )

        with self.assertRaisesRegex(
            TypeError, "Object of type type is not JSON serializable"
        ):
            py_pytree.treespec_dumps(spec)

    def test_pytree_serialize_bad_input(self):
        with self.assertRaises(AttributeError):
            py_pytree.treespec_dumps("random_blurb")

    def test_pytree_serialize_bad_protocol(self):
        import json

        Point = namedtuple("Point", ["x", "y"])
        spec = py_pytree.TreeSpec(
            namedtuple, Point, [py_pytree.LeafSpec(), py_pytree.LeafSpec()]
        )

        with self.assertRaisesRegex(ValueError, "Unknown protocol"):
            py_pytree.treespec_dumps(spec, -1)

        serialized_spec = py_pytree.treespec_dumps(spec)
        protocol, data = json.loads(serialized_spec)
        bad_protocol_serialized_spec = json.dumps((-1, data))

        with self.assertRaisesRegex(ValueError, "Unknown protocol"):
            py_pytree.treespec_loads(bad_protocol_serialized_spec)

    def test_saved_serialized(self):
        complicated_spec = py_pytree.TreeSpec(
            OrderedDict,
            [1, 2, 3],
            [
                py_pytree.TreeSpec(
                    tuple, None, [py_pytree.LeafSpec(), py_pytree.LeafSpec()]
                ),
                py_pytree.LeafSpec(),
                py_pytree.TreeSpec(
                    dict,
                    [4, 5, 6],
                    [
                        py_pytree.LeafSpec(),
                        py_pytree.LeafSpec(),
                        py_pytree.LeafSpec(),
                    ],
                ),
            ],
        )

        serialized_spec = py_pytree.treespec_dumps(complicated_spec)
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
        self.assertEqual(complicated_spec, py_pytree.treespec_loads(saved_spec))


class TestCxxPytree(TestCase):
    def test_treespec_equality(self):
        self.assertTrue(cxx_pytree.LeafSpec() == cxx_pytree.LeafSpec())

    def test_flatten_unflatten_leaf(self):
        def run_test_with_leaf(leaf):
            values, treespec = cxx_pytree.tree_flatten(leaf)
            self.assertEqual(values, [leaf])
            self.assertEqual(treespec, cxx_pytree.LeafSpec())

            unflattened = cxx_pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, leaf)

        run_test_with_leaf(1)
        run_test_with_leaf(1.0)
        run_test_with_leaf(None)
        run_test_with_leaf(bool)
        run_test_with_leaf(torch.randn(3, 3))

    def test_flatten_unflatten_list(self):
        def run_test(lst):
            expected_spec = cxx_pytree.tree_structure([0] * len(lst))
            values, treespec = cxx_pytree.tree_flatten(lst)
            self.assertTrue(isinstance(values, list))
            self.assertEqual(values, lst)
            self.assertEqual(treespec, expected_spec)

            unflattened = cxx_pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, lst)
            self.assertTrue(isinstance(unflattened, list))

        run_test([])
        run_test([1.0, 2])
        run_test([torch.tensor([1.0, 2]), 2, 10, 9, 11])

    def test_flatten_unflatten_tuple(self):
        def run_test(tup):
            expected_spec = cxx_pytree.tree_structure((0,) * len(tup))
            values, treespec = cxx_pytree.tree_flatten(tup)
            self.assertTrue(isinstance(values, list))
            self.assertEqual(values, list(tup))
            self.assertEqual(treespec, expected_spec)

            unflattened = cxx_pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, tup)
            self.assertTrue(isinstance(unflattened, tuple))

        run_test(())
        run_test((1.0,))
        run_test((1.0, 2))
        run_test((torch.tensor([1.0, 2]), 2, 10, 9, 11))

    def test_flatten_unflatten_odict(self):
        def run_test(odict):
            expected_spec = cxx_pytree.tree_structure(OrderedDict.fromkeys(odict, 0))
            values, treespec = cxx_pytree.tree_flatten(odict)
            self.assertTrue(isinstance(values, list))
            self.assertEqual(values, list(odict.values()))
            self.assertEqual(treespec, expected_spec)

            unflattened = cxx_pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, odict)
            self.assertTrue(isinstance(unflattened, OrderedDict))

        od = OrderedDict()
        run_test(od)

        od["b"] = 1
        od["a"] = torch.tensor(3.14)
        run_test(od)

    def test_flatten_unflatten_namedtuple(self):
        Point = namedtuple("Point", ["x", "y"])

        def run_test(tup):
            expected_spec = cxx_pytree.tree_structure(Point(0, 1))
            values, treespec = cxx_pytree.tree_flatten(tup)
            self.assertTrue(isinstance(values, list))
            self.assertEqual(values, list(tup))
            self.assertEqual(treespec, expected_spec)

            unflattened = cxx_pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, tup)
            self.assertTrue(isinstance(unflattened, Point))

        run_test(Point(1.0, 2))
        run_test(Point(torch.tensor(1.0), 2))

    @parametrize(
        "op",
        [
            subtest(torch.max, name="max"),
            subtest(torch.min, name="min"),
        ],
    )
    def test_flatten_unflatten_return_type(self, op):
        x = torch.randn(3, 3)
        expected = op(x, dim=0)

        values, spec = cxx_pytree.tree_flatten(expected)
        # Check that values is actually List[Tensor] and not (ReturnType(...),)
        for value in values:
            self.assertTrue(isinstance(value, torch.Tensor))
        result = cxx_pytree.tree_unflatten(values, spec)

        self.assertEqual(type(result), type(expected))
        self.assertEqual(result, expected)

    def test_flatten_unflatten_dict(self):
        def run_test(dct):
            expected_spec = cxx_pytree.tree_structure(dict.fromkeys(dct, 0))
            values, treespec = cxx_pytree.tree_flatten(dct)
            self.assertTrue(isinstance(values, list))
            self.assertEqual(values, list(dct.values()))
            self.assertEqual(treespec, expected_spec)

            unflattened = cxx_pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, dct)
            self.assertTrue(isinstance(unflattened, dict))

        run_test({})
        run_test({"a": 1})
        run_test({"abcdefg": torch.randn(2, 3)})
        run_test({1: torch.randn(2, 3)})
        run_test({"a": 1, "b": 2, "c": torch.randn(2, 3)})

    def test_flatten_unflatten_nested(self):
        def run_test(pytree):
            values, treespec = cxx_pytree.tree_flatten(pytree)
            self.assertTrue(isinstance(values, list))
            self.assertEqual(len(values), treespec.num_leaves)

            # NB: python basic data structures (dict list tuple) all have
            # contents equality defined on them, so the following works for them.
            unflattened = cxx_pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, pytree)

        cases = [
            [()],
            ([],),
            {"a": ()},
            {"a": 0, "b": [{"c": 1}]},
            {"a": 0, "b": [1, {"c": 2}, torch.randn(3)], "c": (torch.randn(2, 3), 1)},
        ]
        for case in cases:
            run_test(case)

    def test_treemap(self):
        def run_test(pytree):
            def f(x):
                return x * 3

            sm1 = sum(map(f, cxx_pytree.tree_flatten(pytree)[0]))
            sm2 = sum(cxx_pytree.tree_flatten(cxx_pytree.tree_map(f, pytree))[0])
            self.assertEqual(sm1, sm2)

            def invf(x):
                return x // 3

            self.assertEqual(
                cxx_pytree.tree_map(invf, cxx_pytree.tree_map(f, pytree)),
                pytree,
            )

        cases = [
            [()],
            ([],),
            {"a": ()},
            {"a": 1, "b": [{"c": 2}]},
            {"a": 0, "b": [2, {"c": 3}, 4], "c": (5, 6)},
        ]
        for case in cases:
            run_test(case)

    def test_tree_only(self):
        self.assertEqual(
            cxx_pytree.tree_map_only(int, lambda x: x + 2, [0, "a"]), [2, "a"]
        )

    def test_tree_all_any(self):
        self.assertTrue(cxx_pytree.tree_all(lambda x: x % 2, [1, 3]))
        self.assertFalse(cxx_pytree.tree_all(lambda x: x % 2, [0, 1]))
        self.assertTrue(cxx_pytree.tree_any(lambda x: x % 2, [0, 1]))
        self.assertFalse(cxx_pytree.tree_any(lambda x: x % 2, [0, 2]))
        self.assertTrue(cxx_pytree.tree_all_only(int, lambda x: x % 2, [1, 3, "a"]))
        self.assertFalse(cxx_pytree.tree_all_only(int, lambda x: x % 2, [0, 1, "a"]))
        self.assertTrue(cxx_pytree.tree_any_only(int, lambda x: x % 2, [0, 1, "a"]))
        self.assertFalse(cxx_pytree.tree_any_only(int, lambda x: x % 2, [0, 2, "a"]))

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "Dynamo test in test_treespec_repr_dynamo.")
    def test_treespec_repr(self):
        # Check that it looks sane
        pytree = (0, [0, 0, [0]])
        _, spec = cxx_pytree.tree_flatten(pytree)
        self.assertEqual(
            repr(spec),
            ("PyTreeSpec((*, [*, *, [*]]), NoneIsLeaf)"),
        )

    @unittest.skipIf(not TEST_WITH_TORCHDYNAMO, "Eager test in test_treespec_repr.")
    def test_treespec_repr_dynamo(self):
        # Check that it looks sane
        pytree = (0, [0, 0, [0]])
        _, spec = cxx_pytree.tree_flatten(pytree)
        self.assertExpectedInline(
            repr(spec),
            "PyTreeSpec((*, [*, *, [*]]), NoneIsLeaf)",
        )

    def test_broadcast_to_and_flatten(self):
        cases = [
            (1, (), []),
            # Same (flat) structures
            ((1,), (0,), [1]),
            ([1], [0], [1]),
            ((1, 2, 3), (0, 0, 0), [1, 2, 3]),
            ({"a": 1, "b": 2}, {"a": 0, "b": 0}, [1, 2]),
            # Mismatched (flat) structures
            ([1], (0,), None),
            ([1], (0,), None),
            ((1,), [0], None),
            ((1, 2, 3), (0, 0), None),
            ({"a": 1, "b": 2}, {"a": 0}, None),
            ({"a": 1, "b": 2}, {"a": 0, "c": 0}, None),
            ({"a": 1, "b": 2}, {"a": 0, "b": 0, "c": 0}, None),
            # Same (nested) structures
            ((1, [2, 3]), (0, [0, 0]), [1, 2, 3]),
            ((1, [(2, 3), 4]), (0, [(0, 0), 0]), [1, 2, 3, 4]),
            # Mismatched (nested) structures
            ((1, [2, 3]), (0, (0, 0)), None),
            ((1, [2, 3]), (0, [0, 0, 0]), None),
            # Broadcasting single value
            (1, (0, 0, 0), [1, 1, 1]),
            (1, [0, 0, 0], [1, 1, 1]),
            (1, {"a": 0, "b": 0}, [1, 1]),
            (1, (0, [0, [0]], 0), [1, 1, 1, 1]),
            (1, (0, [0, [0, [], [[[0]]]]], 0), [1, 1, 1, 1, 1]),
            # Broadcast multiple things
            ((1, 2), ([0, 0, 0], [0, 0]), [1, 1, 1, 2, 2]),
            ((1, 2), ([0, [0, 0], 0], [0, 0]), [1, 1, 1, 1, 2, 2]),
            (([1, 2, 3], 4), ([0, [0, 0], 0], [0, 0]), [1, 2, 2, 3, 4, 4]),
        ]
        for pytree, to_pytree, expected in cases:
            _, to_spec = cxx_pytree.tree_flatten(to_pytree)
            result = cxx_pytree._broadcast_to_and_flatten(pytree, to_spec)
            self.assertEqual(result, expected, msg=str([pytree, to_spec, expected]))

    @parametrize(
        "spec",
        [
            cxx_pytree.tree_structure([]),
            cxx_pytree.tree_structure(()),
            cxx_pytree.tree_structure({}),
            cxx_pytree.tree_structure([0]),
            cxx_pytree.tree_structure([0, 1]),
            cxx_pytree.tree_structure((0, 1, 2)),
            cxx_pytree.tree_structure({"a": 0, "b": 1, "c": 2}),
            cxx_pytree.tree_structure(
                OrderedDict([("a", (0, 1)), ("b", 2), ("c", {"a": 3, "b": 4, "c": 5})])
            ),
            cxx_pytree.tree_structure([(0, 1, [2, 3])]),
        ],
    )
    def test_pytree_serialize(self, spec):
        serialized_spec = cxx_pytree.treespec_dumps(spec)
        self.assertTrue(isinstance(serialized_spec, bytes))
        self.assertTrue(spec == cxx_pytree.treespec_loads(serialized_spec))

    def test_pytree_serialize_namedtuple(self):
        spec = cxx_pytree.tree_structure(GlobalPoint(0, 1))

        roundtrip_spec = cxx_pytree.treespec_loads(cxx_pytree.treespec_dumps(spec))
        self.assertEqual(roundtrip_spec, spec)

        LocalPoint = namedtuple("LocalPoint", ["x", "y"])
        spec = cxx_pytree.tree_structure(LocalPoint(0, 1))

        with self.assertRaises(pickle.PicklingError):
            cxx_pytree.treespec_dumps(spec)

    def test_pytree_custom_type_serialize(self):
        cxx_pytree.register_pytree_node(
            GlobalDummyType,
            lambda dummy: ([dummy.x, dummy.y], None),
            lambda xs, _: GlobalDummyType(*xs),
        )
        spec = cxx_pytree.tree_structure(GlobalDummyType(0, 1))
        serialized_spec = cxx_pytree.treespec_dumps(spec)
        roundtrip_spec = cxx_pytree.treespec_loads(serialized_spec)
        self.assertEqual(roundtrip_spec, spec)

        class LocalDummyType:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        cxx_pytree.register_pytree_node(
            LocalDummyType,
            lambda dummy: ([dummy.x, dummy.y], None),
            lambda xs, _: LocalDummyType(*xs),
        )
        spec = cxx_pytree.tree_structure(LocalDummyType(0, 1))
        with self.assertRaises(AttributeError):
            serialized_spec = cxx_pytree.treespec_dumps(spec)

    def test_pytree_serialize_bad_input(self):
        with self.assertRaises(TypeError):
            cxx_pytree.treespec_dumps("random_blurb")


instantiate_parametrized_tests(TestPytree)
instantiate_parametrized_tests(TestCxxPytree)

if __name__ == "__main__":
    run_tests()
