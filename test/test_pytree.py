# Owner(s): ["module: pytree"]

import enum
import inspect
import os
import re
import subprocess
import sys
import time
import unittest
from collections import defaultdict, deque, namedtuple, OrderedDict, UserDict
from dataclasses import dataclass, field
from enum import auto
from typing import Any, NamedTuple, Optional

import torch
import torch.utils._pytree as python_pytree
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


pytree_modules = {
    "python": python_pytree,
}
if not IS_FBCODE:
    import torch.utils._cxx_pytree as cxx_pytree

    pytree_modules["cxx"] = cxx_pytree
else:
    # optree is not yet enabled in fbcode, so just re-test the python implementation
    cxx_pytree = python_pytree


parametrize_pytree_module = parametrize(
    "pytree",
    [subtest(module, name=name) for name, module in pytree_modules.items()],
)


GlobalPoint = namedtuple("GlobalPoint", ["x", "y"])


class GlobalDummyType:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if not isinstance(other, GlobalDummyType):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


cxx_pytree.register_pytree_node(
    GlobalDummyType,
    lambda dummy: ([dummy.x, dummy.y], None),
    lambda xs, _: GlobalDummyType(*xs),
    serialized_type_name="GlobalDummyType",
)


class TestEnum(enum.Enum):
    A = auto()


class TestGenericPytree(TestCase):
    def test_aligned_public_apis(self):
        public_apis = python_pytree.__all__

        self.assertEqual(public_apis, cxx_pytree.__all__)

        for name in public_apis:
            cxx_api = getattr(cxx_pytree, name)
            python_api = getattr(python_pytree, name)

            self.assertEqual(inspect.isclass(cxx_api), inspect.isclass(python_api))
            self.assertEqual(
                inspect.isfunction(cxx_api),
                inspect.isfunction(python_api),
            )
            if inspect.isfunction(cxx_api):
                cxx_signature = inspect.signature(cxx_api)
                python_signature = inspect.signature(python_api)

                # Check the parameter names are the same.
                cxx_param_names = list(cxx_signature.parameters)
                python_param_names = list(python_signature.parameters)
                self.assertEqual(cxx_param_names, python_param_names)

                # Check the positional parameters are the same.
                cxx_positional_param_names = [
                    n
                    for n, p in cxx_signature.parameters.items()
                    if (
                        p.kind
                        in {
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        }
                    )
                ]
                python_positional_param_names = [
                    n
                    for n, p in python_signature.parameters.items()
                    if (
                        p.kind
                        in {
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        }
                    )
                ]
                self.assertEqual(
                    cxx_positional_param_names,
                    python_positional_param_names,
                )

                for python_name, python_param in python_signature.parameters.items():
                    self.assertIn(python_name, cxx_signature.parameters)
                    cxx_param = cxx_signature.parameters[python_name]

                    # Check parameter kinds and default values are the same.
                    self.assertEqual(cxx_param.kind, python_param.kind)
                    self.assertEqual(cxx_param.default, python_param.default)

                    # Check parameter annotations are the same.
                    if "TreeSpec" in str(cxx_param.annotation):
                        self.assertIn("TreeSpec", str(python_param.annotation))
                        self.assertEqual(
                            re.sub(
                                r"(?:\b)([\w\.]*)TreeSpec(?:\b)",
                                "TreeSpec",
                                str(cxx_param.annotation),
                            ),
                            re.sub(
                                r"(?:\b)([\w\.]*)TreeSpec(?:\b)",
                                "TreeSpec",
                                str(python_param.annotation),
                            ),
                            msg=(
                                f"C++ parameter {cxx_param} "
                                f"does not match Python parameter {python_param} "
                                f"for API `{name}`"
                            ),
                        )
                    else:
                        self.assertEqual(
                            cxx_param.annotation,
                            python_param.annotation,
                            msg=(
                                f"C++ parameter {cxx_param} "
                                f"does not match Python parameter {python_param} "
                                f"for API `{name}`"
                            ),
                        )

    @parametrize_pytree_module
    def test_register_pytree_node(self, pytree):
        class MyDict(UserDict):
            pass

        d = MyDict(a=1, b=2, c=3)

        # Custom types are leaf nodes by default
        values, spec = pytree.tree_flatten(d)
        self.assertEqual(values, [d])
        self.assertIs(values[0], d)
        self.assertEqual(d, pytree.tree_unflatten(values, spec))
        self.assertTrue(spec.is_leaf())

        # Register MyDict as a pytree node
        pytree.register_pytree_node(
            MyDict,
            lambda d: (list(d.values()), list(d.keys())),
            lambda values, keys: MyDict(zip(keys, values)),
        )

        values, spec = pytree.tree_flatten(d)
        self.assertEqual(values, [1, 2, 3])
        self.assertEqual(d, pytree.tree_unflatten(values, spec))

        # Do not allow registering the same type twice
        with self.assertRaisesRegex(ValueError, "already registered"):
            pytree.register_pytree_node(
                MyDict,
                lambda d: (list(d.values()), list(d.keys())),
                lambda values, keys: MyDict(zip(keys, values)),
            )

    @parametrize_pytree_module
    def test_flatten_unflatten_leaf(self, pytree):
        def run_test_with_leaf(leaf):
            values, treespec = pytree.tree_flatten(leaf)
            self.assertEqual(values, [leaf])
            self.assertEqual(treespec, pytree.treespec_leaf())

            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, leaf)

        run_test_with_leaf(1)
        run_test_with_leaf(1.0)
        run_test_with_leaf(None)
        run_test_with_leaf(bool)
        run_test_with_leaf(torch.randn(3, 3))

    @parametrize(
        "pytree,gen_expected_fn",
        [
            subtest(
                (
                    python_pytree,
                    lambda tup: python_pytree.TreeSpec(
                        tuple, None, [python_pytree.treespec_leaf() for _ in tup]
                    ),
                ),
                name="python",
            ),
            subtest(
                (cxx_pytree, lambda tup: cxx_pytree.tree_structure((0,) * len(tup))),
                name="cxx",
            ),
        ],
    )
    def test_flatten_unflatten_tuple(self, pytree, gen_expected_fn):
        def run_test(tup):
            expected_spec = gen_expected_fn(tup)
            values, treespec = pytree.tree_flatten(tup)
            self.assertIsInstance(values, list)
            self.assertEqual(values, list(tup))
            self.assertEqual(treespec, expected_spec)

            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, tup)
            self.assertIsInstance(unflattened, tuple)

        run_test(())
        run_test((1.0,))
        run_test((1.0, 2))
        run_test((torch.tensor([1.0, 2]), 2, 10, 9, 11))

    @parametrize(
        "pytree,gen_expected_fn",
        [
            subtest(
                (
                    python_pytree,
                    lambda lst: python_pytree.TreeSpec(
                        list, None, [python_pytree.treespec_leaf() for _ in lst]
                    ),
                ),
                name="python",
            ),
            subtest(
                (cxx_pytree, lambda lst: cxx_pytree.tree_structure([0] * len(lst))),
                name="cxx",
            ),
        ],
    )
    def test_flatten_unflatten_list(self, pytree, gen_expected_fn):
        def run_test(lst):
            expected_spec = gen_expected_fn(lst)
            values, treespec = pytree.tree_flatten(lst)
            self.assertIsInstance(values, list)
            self.assertEqual(values, lst)
            self.assertEqual(treespec, expected_spec)

            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, lst)
            self.assertIsInstance(unflattened, list)

        run_test([])
        run_test([1.0, 2])
        run_test([torch.tensor([1.0, 2]), 2, 10, 9, 11])

    @parametrize(
        "pytree,gen_expected_fn",
        [
            subtest(
                (
                    python_pytree,
                    lambda dct: python_pytree.TreeSpec(
                        dict,
                        list(dct.keys()),
                        [python_pytree.treespec_leaf() for _ in dct.values()],
                    ),
                ),
                name="python",
            ),
            subtest(
                (
                    cxx_pytree,
                    lambda dct: cxx_pytree.tree_structure(dict.fromkeys(dct, 0)),
                ),
                name="cxx",
            ),
        ],
    )
    def test_flatten_unflatten_dict(self, pytree, gen_expected_fn):
        def run_test(dct):
            expected_spec = gen_expected_fn(dct)
            values, treespec = pytree.tree_flatten(dct)
            self.assertIsInstance(values, list)
            self.assertEqual(values, list(dct.values()))
            self.assertEqual(treespec, expected_spec)

            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, dct)
            self.assertIsInstance(unflattened, dict)

        run_test({})
        run_test({"a": 1})
        run_test({"abcdefg": torch.randn(2, 3)})
        run_test({1: torch.randn(2, 3)})
        run_test({"a": 1, "b": 2, "c": torch.randn(2, 3)})

    @parametrize(
        "pytree,gen_expected_fn",
        [
            subtest(
                (
                    python_pytree,
                    lambda odict: python_pytree.TreeSpec(
                        OrderedDict,
                        list(odict.keys()),
                        [python_pytree.treespec_leaf() for _ in odict.values()],
                    ),
                ),
                name="python",
            ),
            subtest(
                (
                    cxx_pytree,
                    lambda odict: cxx_pytree.tree_structure(
                        OrderedDict.fromkeys(odict, 0)
                    ),
                ),
                name="cxx",
            ),
        ],
    )
    def test_flatten_unflatten_ordereddict(self, pytree, gen_expected_fn):
        def run_test(odict):
            expected_spec = gen_expected_fn(odict)
            values, treespec = pytree.tree_flatten(odict)
            self.assertIsInstance(values, list)
            self.assertEqual(values, list(odict.values()))
            self.assertEqual(treespec, expected_spec)

            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, odict)
            self.assertIsInstance(unflattened, OrderedDict)

        od = OrderedDict()
        run_test(od)

        od["b"] = 1
        od["a"] = torch.tensor(3.14)
        run_test(od)

    @parametrize(
        "pytree,gen_expected_fn",
        [
            subtest(
                (
                    python_pytree,
                    lambda ddct: python_pytree.TreeSpec(
                        defaultdict,
                        [ddct.default_factory, list(ddct.keys())],
                        [python_pytree.treespec_leaf() for _ in ddct.values()],
                    ),
                ),
                name="python",
            ),
            subtest(
                (
                    cxx_pytree,
                    lambda ddct: cxx_pytree.tree_structure(
                        defaultdict(ddct.default_factory, dict.fromkeys(ddct, 0))
                    ),
                ),
                name="cxx",
            ),
        ],
    )
    def test_flatten_unflatten_defaultdict(self, pytree, gen_expected_fn):
        def run_test(ddct):
            expected_spec = gen_expected_fn(ddct)
            values, treespec = pytree.tree_flatten(ddct)
            self.assertIsInstance(values, list)
            self.assertEqual(values, list(ddct.values()))
            self.assertEqual(treespec, expected_spec)

            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, ddct)
            self.assertEqual(unflattened.default_factory, ddct.default_factory)
            self.assertIsInstance(unflattened, defaultdict)

        run_test(defaultdict(list, {}))
        run_test(defaultdict(int, {"a": 1}))
        run_test(defaultdict(int, {"abcdefg": torch.randn(2, 3)}))
        run_test(defaultdict(int, {1: torch.randn(2, 3)}))
        run_test(defaultdict(int, {"a": 1, "b": 2, "c": torch.randn(2, 3)}))

    @parametrize(
        "pytree,gen_expected_fn",
        [
            subtest(
                (
                    python_pytree,
                    lambda deq: python_pytree.TreeSpec(
                        deque, deq.maxlen, [python_pytree.treespec_leaf() for _ in deq]
                    ),
                ),
                name="python",
            ),
            subtest(
                (
                    cxx_pytree,
                    lambda deq: cxx_pytree.tree_structure(
                        deque(deq, maxlen=deq.maxlen)
                    ),
                ),
                name="cxx",
            ),
        ],
    )
    def test_flatten_unflatten_deque(self, pytree, gen_expected_fn):
        def run_test(deq):
            expected_spec = gen_expected_fn(deq)
            values, treespec = pytree.tree_flatten(deq)
            self.assertIsInstance(values, list)
            self.assertEqual(values, list(deq))
            self.assertEqual(treespec, expected_spec)

            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, deq)
            self.assertEqual(unflattened.maxlen, deq.maxlen)
            self.assertIsInstance(unflattened, deque)

        run_test(deque([]))
        run_test(deque([1.0, 2]))
        run_test(deque([torch.tensor([1.0, 2]), 2, 10, 9, 11], maxlen=8))

    @parametrize_pytree_module
    def test_flatten_unflatten_namedtuple(self, pytree):
        Point = namedtuple("Point", ["x", "y"])

        def run_test(tup):
            if pytree is python_pytree:
                expected_spec = python_pytree.TreeSpec(
                    namedtuple, Point, [python_pytree.treespec_leaf() for _ in tup]
                )
            else:
                expected_spec = cxx_pytree.tree_structure(Point(0, 1))
            values, treespec = pytree.tree_flatten(tup)
            self.assertIsInstance(values, list)
            self.assertEqual(values, list(tup))
            self.assertEqual(treespec, expected_spec)

            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, tup)
            self.assertIsInstance(unflattened, Point)

        run_test(Point(1.0, 2))
        run_test(Point(torch.tensor(1.0), 2))

    @parametrize(
        "op",
        [
            subtest(torch.max, name="max"),
            subtest(torch.min, name="min"),
        ],
    )
    @parametrize_pytree_module
    def test_flatten_unflatten_return_types(self, pytree, op):
        x = torch.randn(3, 3)
        expected = op(x, dim=0)

        values, spec = pytree.tree_flatten(expected)
        # Check that values is actually List[Tensor] and not (ReturnType(...),)
        for value in values:
            self.assertIsInstance(value, torch.Tensor)
        result = pytree.tree_unflatten(values, spec)

        self.assertEqual(type(result), type(expected))
        self.assertEqual(result, expected)

    @parametrize_pytree_module
    def test_flatten_unflatten_nested(self, pytree):
        def run_test(tree):
            values, treespec = pytree.tree_flatten(tree)
            self.assertIsInstance(values, list)
            self.assertEqual(len(values), treespec.num_leaves)

            # NB: python basic data structures (dict list tuple) all have
            # contents equality defined on them, so the following works for them.
            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, tree)

        cases = [
            [()],
            ([],),
            {"a": ()},
            {"a": 0, "b": [{"c": 1}]},
            {"a": 0, "b": [1, {"c": 2}, torch.randn(3)], "c": (torch.randn(2, 3), 1)},
        ]
        for case in cases:
            run_test(case)

    @parametrize_pytree_module
    def test_flatten_with_is_leaf(self, pytree):
        def run_test(tree, one_level_leaves):
            values, treespec = pytree.tree_flatten(
                tree, is_leaf=lambda x: x is not tree
            )
            self.assertIsInstance(values, list)
            self.assertEqual(len(values), treespec.num_nodes - 1)
            self.assertEqual(len(values), treespec.num_leaves)
            self.assertEqual(len(values), treespec.num_children)
            self.assertEqual(values, one_level_leaves)

            self.assertEqual(
                treespec,
                pytree.tree_structure(
                    pytree.tree_unflatten([0] * treespec.num_leaves, treespec)
                ),
            )

            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, tree)

        cases = [
            ([()], [()]),
            (([],), [[]]),
            ({"a": ()}, [()]),
            ({"a": 0, "b": [{"c": 1}]}, [0, [{"c": 1}]]),
            (
                {
                    "a": 0,
                    "b": [1, {"c": 2}, torch.ones(3)],
                    "c": (torch.zeros(2, 3), 1),
                },
                [0, [1, {"c": 2}, torch.ones(3)], (torch.zeros(2, 3), 1)],
            ),
        ]
        for case in cases:
            run_test(*case)

    @parametrize_pytree_module
    def test_tree_map(self, pytree):
        def run_test(tree):
            def f(x):
                return x * 3

            sm1 = sum(map(f, pytree.tree_leaves(tree)))
            sm2 = sum(pytree.tree_leaves(pytree.tree_map(f, tree)))
            self.assertEqual(sm1, sm2)

            def invf(x):
                return x // 3

            self.assertEqual(
                pytree.tree_map(invf, pytree.tree_map(f, tree)),
                tree,
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

    @parametrize_pytree_module
    def test_tree_map_multi_inputs(self, pytree):
        def run_test(tree):
            def f(x, y, z):
                return x, [y, (z, 0)]

            tree_x = tree
            tree_y = pytree.tree_map(lambda x: (x + 1,), tree)
            tree_z = pytree.tree_map(lambda x: {"a": x * 2, "b": 2}, tree)

            self.assertEqual(
                pytree.tree_map(f, tree_x, tree_y, tree_z),
                pytree.tree_map(lambda x: f(x, (x + 1,), {"a": x * 2, "b": 2}), tree),
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

    @parametrize_pytree_module
    def test_tree_map_dict_order(self, pytree):
        d = {"b": 2, "a": 1, "c": 3}
        od = OrderedDict([("b", 2), ("a", 1), ("c", 3)])
        dd = defaultdict(int, {"b": 2, "a": 1, "c": 3})
        for tree in (d, od, dd):
            result = pytree.tree_map(lambda x: x, tree)
            self.assertEqual(
                list(result.keys()),
                list(tree.keys()),
                msg=f"Dictionary keys order changed in tree_map: {tree!r} vs. {result!r}",
            )
            self.assertEqual(
                list(result.values()),
                list(tree.values()),
                msg=f"Dictionary keys order changed in tree_map: {tree!r} vs. {result!r}",
            )

    @parametrize_pytree_module
    def test_tree_map_only(self, pytree):
        self.assertEqual(pytree.tree_map_only(int, lambda x: x + 2, [0, "a"]), [2, "a"])

    @parametrize_pytree_module
    def test_tree_map_only_predicate_fn(self, pytree):
        self.assertEqual(
            pytree.tree_map_only(lambda x: x == 0, lambda x: x + 2, [0, 1]), [2, 1]
        )

    @parametrize_pytree_module
    def test_tree_all_any(self, pytree):
        self.assertTrue(pytree.tree_all(lambda x: x % 2, [1, 3]))
        self.assertFalse(pytree.tree_all(lambda x: x % 2, [0, 1]))
        self.assertTrue(pytree.tree_any(lambda x: x % 2, [0, 1]))
        self.assertFalse(pytree.tree_any(lambda x: x % 2, [0, 2]))
        self.assertTrue(pytree.tree_all_only(int, lambda x: x % 2, [1, 3, "a"]))
        self.assertFalse(pytree.tree_all_only(int, lambda x: x % 2, [0, 1, "a"]))
        self.assertTrue(pytree.tree_any_only(int, lambda x: x % 2, [0, 1, "a"]))
        self.assertFalse(pytree.tree_any_only(int, lambda x: x % 2, [0, 2, "a"]))

    @parametrize_pytree_module
    def test_broadcast_to_and_flatten(self, pytree):
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
        for tree, to_tree, expected in cases:
            _, to_spec = pytree.tree_flatten(to_tree)
            result = pytree._broadcast_to_and_flatten(tree, to_spec)
            self.assertEqual(result, expected, msg=str([tree, to_spec, expected]))

    @parametrize_pytree_module
    def test_pytree_serialize_bad_input(self, pytree):
        with self.assertRaises(TypeError):
            pytree.treespec_dumps("random_blurb")

    @parametrize_pytree_module
    def test_is_namedtuple(self, pytree):
        DirectNamedTuple1 = namedtuple("DirectNamedTuple1", ["x", "y"])

        class DirectNamedTuple2(NamedTuple):
            x: int
            y: int

        class IndirectNamedTuple1(DirectNamedTuple1):
            pass

        class IndirectNamedTuple2(DirectNamedTuple2):
            pass

        self.assertTrue(pytree.is_namedtuple(DirectNamedTuple1(0, 1)))
        self.assertTrue(pytree.is_namedtuple(DirectNamedTuple2(0, 1)))
        self.assertTrue(pytree.is_namedtuple(IndirectNamedTuple1(0, 1)))
        self.assertTrue(pytree.is_namedtuple(IndirectNamedTuple2(0, 1)))
        self.assertFalse(pytree.is_namedtuple(time.gmtime()))
        self.assertFalse(pytree.is_namedtuple((0, 1)))
        self.assertFalse(pytree.is_namedtuple([0, 1]))
        self.assertFalse(pytree.is_namedtuple({0: 1, 1: 2}))
        self.assertFalse(pytree.is_namedtuple({0, 1}))
        self.assertFalse(pytree.is_namedtuple(1))

        self.assertTrue(pytree.is_namedtuple(DirectNamedTuple1))
        self.assertTrue(pytree.is_namedtuple(DirectNamedTuple2))
        self.assertTrue(pytree.is_namedtuple(IndirectNamedTuple1))
        self.assertTrue(pytree.is_namedtuple(IndirectNamedTuple2))
        self.assertFalse(pytree.is_namedtuple(time.struct_time))
        self.assertFalse(pytree.is_namedtuple(tuple))
        self.assertFalse(pytree.is_namedtuple(list))

        self.assertTrue(pytree.is_namedtuple_class(DirectNamedTuple1))
        self.assertTrue(pytree.is_namedtuple_class(DirectNamedTuple2))
        self.assertTrue(pytree.is_namedtuple_class(IndirectNamedTuple1))
        self.assertTrue(pytree.is_namedtuple_class(IndirectNamedTuple2))
        self.assertFalse(pytree.is_namedtuple_class(time.struct_time))
        self.assertFalse(pytree.is_namedtuple_class(tuple))
        self.assertFalse(pytree.is_namedtuple_class(list))

    @parametrize_pytree_module
    def test_is_structseq(self, pytree):
        class FakeStructSeq(tuple):
            n_fields = 2
            n_sequence_fields = 2
            n_unnamed_fields = 0

            __slots__ = ()
            __match_args__ = ("x", "y")

            def __new__(cls, sequence):
                return super().__new__(cls, sequence)

            @property
            def x(self):
                return self[0]

            @property
            def y(self):
                return self[1]

        DirectNamedTuple1 = namedtuple("DirectNamedTuple1", ["x", "y"])

        class DirectNamedTuple2(NamedTuple):
            x: int
            y: int

        self.assertFalse(pytree.is_structseq(FakeStructSeq((0, 1))))
        self.assertTrue(pytree.is_structseq(time.gmtime()))
        self.assertFalse(pytree.is_structseq(DirectNamedTuple1(0, 1)))
        self.assertFalse(pytree.is_structseq(DirectNamedTuple2(0, 1)))
        self.assertFalse(pytree.is_structseq((0, 1)))
        self.assertFalse(pytree.is_structseq([0, 1]))
        self.assertFalse(pytree.is_structseq({0: 1, 1: 2}))
        self.assertFalse(pytree.is_structseq({0, 1}))
        self.assertFalse(pytree.is_structseq(1))

        self.assertFalse(pytree.is_structseq(FakeStructSeq))
        self.assertTrue(pytree.is_structseq(time.struct_time))
        self.assertFalse(pytree.is_structseq(DirectNamedTuple1))
        self.assertFalse(pytree.is_structseq(DirectNamedTuple2))
        self.assertFalse(pytree.is_structseq(tuple))
        self.assertFalse(pytree.is_structseq(list))

        self.assertFalse(pytree.is_structseq_class(FakeStructSeq))
        self.assertTrue(
            pytree.is_structseq_class(time.struct_time),
        )
        self.assertFalse(pytree.is_structseq_class(DirectNamedTuple1))
        self.assertFalse(pytree.is_structseq_class(DirectNamedTuple2))
        self.assertFalse(pytree.is_structseq_class(tuple))
        self.assertFalse(pytree.is_structseq_class(list))

        # torch.return_types.* are all PyStructSequence types
        for cls in vars(torch.return_types).values():
            if isinstance(cls, type) and issubclass(cls, tuple):
                self.assertTrue(pytree.is_structseq(cls))
                self.assertTrue(pytree.is_structseq_class(cls))
                self.assertFalse(pytree.is_namedtuple(cls))
                self.assertFalse(pytree.is_namedtuple_class(cls))

                inst = cls(range(cls.n_sequence_fields))
                self.assertTrue(pytree.is_structseq(inst))
                self.assertTrue(pytree.is_structseq(type(inst)))
                self.assertFalse(pytree.is_structseq_class(inst))
                self.assertTrue(pytree.is_structseq_class(type(inst)))
                self.assertFalse(pytree.is_namedtuple(inst))
                self.assertFalse(pytree.is_namedtuple_class(inst))
            else:
                self.assertFalse(pytree.is_structseq(cls))
                self.assertFalse(pytree.is_structseq_class(cls))
                self.assertFalse(pytree.is_namedtuple(cls))
                self.assertFalse(pytree.is_namedtuple_class(cls))

    @parametrize_pytree_module
    def test_enum_treespec_roundtrip(self, pytree):
        data = {TestEnum.A: 5}
        spec = pytree.tree_structure(data)

        serialized = pytree.treespec_dumps(spec)
        deserialized_spec = pytree.treespec_loads(serialized)
        self.assertEqual(spec, deserialized_spec)


class TestPythonPytree(TestCase):
    def test_deprecated_register_pytree_node(self):
        class DummyType:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        with self.assertWarnsRegex(
            FutureWarning, "torch.utils._pytree._register_pytree_node"
        ):
            python_pytree._register_pytree_node(
                DummyType,
                lambda dummy: ([dummy.x, dummy.y], None),
                lambda xs, _: DummyType(*xs),
            )

        with self.assertWarnsRegex(UserWarning, "already registered"):
            python_pytree._register_pytree_node(
                DummyType,
                lambda dummy: ([dummy.x, dummy.y], None),
                lambda xs, _: DummyType(*xs),
            )

    def test_import_pytree_doesnt_import_optree(self):
        # importing torch.utils._pytree shouldn't import optree.
        # only importing torch.utils._cxx_pytree should.
        script = """
import sys
import torch
import torch.utils._pytree
assert "torch.utils._pytree" in sys.modules
if "torch.utils._cxx_pytree" in sys.modules:
    raise RuntimeError("importing torch.utils._pytree should not import torch.utils._cxx_pytree")
if "optree" in sys.modules:
    raise RuntimeError("importing torch.utils._pytree should not import optree")
"""
        try:
            subprocess.check_output(
                [sys.executable, "-c", script],
                stderr=subprocess.STDOUT,
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            self.fail(
                msg=(
                    "Subprocess exception while attempting to run test: "
                    + e.output.decode("utf-8")
                )
            )

    def test_treespec_equality(self):
        self.assertEqual(
            python_pytree.treespec_leaf(),
            python_pytree.treespec_leaf(),
        )
        self.assertEqual(
            python_pytree.TreeSpec(list, None, []),
            python_pytree.TreeSpec(list, None, []),
        )
        self.assertEqual(
            python_pytree.TreeSpec(list, None, [python_pytree.treespec_leaf()]),
            python_pytree.TreeSpec(list, None, [python_pytree.treespec_leaf()]),
        )
        self.assertFalse(
            python_pytree.TreeSpec(tuple, None, [])
            == python_pytree.TreeSpec(list, None, []),
        )
        self.assertTrue(
            python_pytree.TreeSpec(tuple, None, [])
            != python_pytree.TreeSpec(list, None, []),
        )

    def test_treespec_repr(self):
        # Check that it looks sane
        tree = (0, [0, 0, [0]])
        spec = python_pytree.tree_structure(tree)
        self.assertEqual(
            repr(spec),
            (
                "TreeSpec(tuple, None, [*,\n"
                "  TreeSpec(list, None, [*,\n"
                "    *,\n"
                "    TreeSpec(list, None, [*])])])"
            ),
        )

    @parametrize(
        "spec",
        [
            # python_pytree.tree_structure([])
            python_pytree.TreeSpec(list, None, []),
            # python_pytree.tree_structure(())
            python_pytree.TreeSpec(tuple, None, []),
            # python_pytree.tree_structure({})
            python_pytree.TreeSpec(dict, [], []),
            # python_pytree.tree_structure([0])
            python_pytree.TreeSpec(list, None, [python_pytree.treespec_leaf()]),
            # python_pytree.tree_structure([0, 1])
            python_pytree.TreeSpec(
                list,
                None,
                [python_pytree.treespec_leaf(), python_pytree.treespec_leaf()],
            ),
            # python_pytree.tree_structure((0, 1, 2))
            python_pytree.TreeSpec(
                tuple,
                None,
                [
                    python_pytree.treespec_leaf(),
                    python_pytree.treespec_leaf(),
                    python_pytree.treespec_leaf(),
                ],
            ),
            # python_pytree.tree_structure({"a": 0, "b": 1, "c": 2})
            python_pytree.TreeSpec(
                dict,
                ["a", "b", "c"],
                [
                    python_pytree.treespec_leaf(),
                    python_pytree.treespec_leaf(),
                    python_pytree.treespec_leaf(),
                ],
            ),
            # python_pytree.tree_structure(OrderedDict([("a", (0, 1)), ("b", 2), ("c", {"a": 3, "b": 4, "c": 5})])
            python_pytree.TreeSpec(
                OrderedDict,
                ["a", "b", "c"],
                [
                    python_pytree.TreeSpec(
                        tuple,
                        None,
                        [python_pytree.treespec_leaf(), python_pytree.treespec_leaf()],
                    ),
                    python_pytree.treespec_leaf(),
                    python_pytree.TreeSpec(
                        dict,
                        ["a", "b", "c"],
                        [
                            python_pytree.treespec_leaf(),
                            python_pytree.treespec_leaf(),
                            python_pytree.treespec_leaf(),
                        ],
                    ),
                ],
            ),
            # python_pytree.tree_structure([(0, 1, [2, 3])])
            python_pytree.TreeSpec(
                list,
                None,
                [
                    python_pytree.TreeSpec(
                        tuple,
                        None,
                        [
                            python_pytree.treespec_leaf(),
                            python_pytree.treespec_leaf(),
                            python_pytree.TreeSpec(
                                list,
                                None,
                                [
                                    python_pytree.treespec_leaf(),
                                    python_pytree.treespec_leaf(),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            # python_pytree.tree_structure(defaultdict(list, {"a": [0, 1], "b": [1, 2], "c": {}}))
            python_pytree.TreeSpec(
                defaultdict,
                [list, ["a", "b", "c"]],
                [
                    python_pytree.TreeSpec(
                        list,
                        None,
                        [python_pytree.treespec_leaf(), python_pytree.treespec_leaf()],
                    ),
                    python_pytree.TreeSpec(
                        list,
                        None,
                        [python_pytree.treespec_leaf(), python_pytree.treespec_leaf()],
                    ),
                    python_pytree.TreeSpec(dict, [], []),
                ],
            ),
        ],
    )
    def test_pytree_serialize(self, spec):
        # Ensure that the spec is valid
        self.assertEqual(
            spec,
            python_pytree.tree_structure(
                python_pytree.tree_unflatten([0] * spec.num_leaves, spec)
            ),
        )

        serialized_spec = python_pytree.treespec_dumps(spec)
        self.assertIsInstance(serialized_spec, str)
        self.assertEqual(spec, python_pytree.treespec_loads(serialized_spec))

    def test_pytree_serialize_defaultdict_enum(self):
        spec = python_pytree.TreeSpec(
            defaultdict,
            [list, [TestEnum.A]],
            [
                python_pytree.TreeSpec(
                    list,
                    None,
                    [
                        python_pytree.treespec_leaf(),
                    ],
                ),
            ],
        )
        serialized_spec = python_pytree.treespec_dumps(spec)
        self.assertIsInstance(serialized_spec, str)

    def test_pytree_serialize_enum(self):
        spec = python_pytree.TreeSpec(dict, TestEnum.A, [python_pytree.treespec_leaf()])

        serialized_spec = python_pytree.treespec_dumps(spec)
        self.assertIsInstance(serialized_spec, str)

    def test_pytree_serialize_namedtuple(self):
        Point1 = namedtuple("Point1", ["x", "y"])
        python_pytree._register_namedtuple(
            Point1,
            serialized_type_name="test_pytree.test_pytree_serialize_namedtuple.Point1",
        )

        spec = python_pytree.tree_structure(Point1(1, 2))
        self.assertIs(spec.type, namedtuple)
        roundtrip_spec = python_pytree.treespec_loads(
            python_pytree.treespec_dumps(spec)
        )
        self.assertEqual(spec, roundtrip_spec)

        class Point2(NamedTuple):
            x: int
            y: int

        python_pytree._register_namedtuple(
            Point2,
            serialized_type_name="test_pytree.test_pytree_serialize_namedtuple.Point2",
        )

        spec = python_pytree.tree_structure(Point2(1, 2))
        self.assertIs(spec.type, namedtuple)
        roundtrip_spec = python_pytree.treespec_loads(
            python_pytree.treespec_dumps(spec)
        )
        self.assertEqual(spec, roundtrip_spec)

        class Point3(Point2):
            pass

        python_pytree._register_namedtuple(
            Point3,
            serialized_type_name="test_pytree.test_pytree_serialize_namedtuple.Point3",
        )

        spec = python_pytree.tree_structure(Point3(1, 2))
        self.assertIs(spec.type, namedtuple)
        roundtrip_spec = python_pytree.treespec_loads(
            python_pytree.treespec_dumps(spec)
        )
        self.assertEqual(spec, roundtrip_spec)

    def test_pytree_serialize_namedtuple_bad(self):
        DummyType = namedtuple("DummyType", ["x", "y"])

        spec = python_pytree.tree_structure(DummyType(1, 2))

        with self.assertRaisesRegex(
            NotImplementedError, "Please register using `_register_namedtuple`"
        ):
            python_pytree.treespec_dumps(spec)

    def test_pytree_custom_type_serialize_bad(self):
        class DummyType:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        python_pytree.register_pytree_node(
            DummyType,
            lambda dummy: ([dummy.x, dummy.y], None),
            lambda xs, _: DummyType(*xs),
        )

        spec = python_pytree.tree_structure(DummyType(1, 2))
        with self.assertRaisesRegex(
            NotImplementedError, "No registered serialization name"
        ):
            python_pytree.treespec_dumps(spec)

    def test_pytree_custom_type_serialize(self):
        class DummyType:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        python_pytree.register_pytree_node(
            DummyType,
            lambda dummy: ([dummy.x, dummy.y], None),
            lambda xs, _: DummyType(*xs),
            serialized_type_name="test_pytree_custom_type_serialize.DummyType",
            to_dumpable_context=lambda context: "moo",
            from_dumpable_context=lambda dumpable_context: None,
        )
        spec = python_pytree.tree_structure(DummyType(1, 2))
        serialized_spec = python_pytree.treespec_dumps(spec, 1)
        self.assertIn("moo", serialized_spec)
        roundtrip_spec = python_pytree.treespec_loads(serialized_spec)
        self.assertEqual(roundtrip_spec, spec)

    def test_pytree_serialize_register_bad(self):
        class DummyType:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        with self.assertRaisesRegex(
            ValueError, "Both to_dumpable_context and from_dumpable_context"
        ):
            python_pytree.register_pytree_node(
                DummyType,
                lambda dummy: ([dummy.x, dummy.y], None),
                lambda xs, _: DummyType(*xs),
                serialized_type_name="test_pytree_serialize_register_bad.DummyType",
                to_dumpable_context=lambda context: "moo",
            )

    def test_pytree_context_serialize_bad(self):
        class DummyType:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        python_pytree.register_pytree_node(
            DummyType,
            lambda dummy: ([dummy.x, dummy.y], None),
            lambda xs, _: DummyType(*xs),
            serialized_type_name="test_pytree_serialize_serialize_bad.DummyType",
            to_dumpable_context=lambda context: DummyType,
            from_dumpable_context=lambda dumpable_context: None,
        )

        spec = python_pytree.tree_structure(DummyType(1, 2))

        with self.assertRaisesRegex(
            TypeError, "Object of type type is not JSON serializable"
        ):
            python_pytree.treespec_dumps(spec)

    def test_pytree_serialize_bad_protocol(self):
        import json

        Point = namedtuple("Point", ["x", "y"])
        spec = python_pytree.tree_structure(Point(1, 2))
        python_pytree._register_namedtuple(
            Point,
            serialized_type_name="test_pytree.test_pytree_serialize_bad_protocol.Point",
        )

        with self.assertRaisesRegex(ValueError, "Unknown protocol"):
            python_pytree.treespec_dumps(spec, -1)

        serialized_spec = python_pytree.treespec_dumps(spec)
        _, data = json.loads(serialized_spec)
        bad_protocol_serialized_spec = json.dumps((-1, data))

        with self.assertRaisesRegex(ValueError, "Unknown protocol"):
            python_pytree.treespec_loads(bad_protocol_serialized_spec)

    def test_saved_serialized(self):
        # python_pytree.tree_structure(OrderedDict([(1, (0, 1)), (2, 2), (3, {4: 3, 5: 4, 6: 5})]))
        complicated_spec = python_pytree.TreeSpec(
            OrderedDict,
            [1, 2, 3],
            [
                python_pytree.TreeSpec(
                    tuple,
                    None,
                    [python_pytree.treespec_leaf(), python_pytree.treespec_leaf()],
                ),
                python_pytree.treespec_leaf(),
                python_pytree.TreeSpec(
                    dict,
                    [4, 5, 6],
                    [
                        python_pytree.treespec_leaf(),
                        python_pytree.treespec_leaf(),
                        python_pytree.treespec_leaf(),
                    ],
                ),
            ],
        )
        # Ensure that the spec is valid
        self.assertEqual(
            complicated_spec,
            python_pytree.tree_structure(
                python_pytree.tree_unflatten(
                    [0] * complicated_spec.num_leaves, complicated_spec
                )
            ),
        )

        serialized_spec = python_pytree.treespec_dumps(complicated_spec)
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
        self.assertEqual(complicated_spec, python_pytree.treespec_loads(saved_spec))

    def test_tree_map_with_path(self):
        tree = [{i: i for i in range(10)}]
        all_zeros = python_pytree.tree_map_with_path(
            lambda kp, val: val - kp[1].key + kp[0].idx, tree
        )
        self.assertEqual(all_zeros, [dict.fromkeys(range(10), 0)])

    def test_dataclass(self):
        @dataclass
        class Data:
            a: torch.Tensor
            b: str = "moo"
            c: Optional[str] = None
            d: str = field(init=False, default="")

        python_pytree.register_dataclass(Data)
        old_data = Data(torch.tensor(3), "b", "c")
        old_data.d = "d"
        new_data = python_pytree.tree_map(lambda x: x, old_data)
        self.assertEqual(new_data.a, torch.tensor(3))
        self.assertEqual(new_data.b, "b")
        self.assertEqual(new_data.c, "c")
        self.assertEqual(new_data.d, "")
        python_pytree._deregister_pytree_node(Data)

        with self.assertRaisesRegex(ValueError, "Missing fields"):
            python_pytree.register_dataclass(Data, field_names=["a", "b"])

        with self.assertRaisesRegex(ValueError, "Unexpected fields"):
            python_pytree.register_dataclass(Data, field_names=["a", "b", "e"])

        with self.assertRaisesRegex(ValueError, "Unexpected fields"):
            python_pytree.register_dataclass(Data, field_names=["a", "b", "c", "d"])

        python_pytree.register_dataclass(
            Data, field_names=["a"], drop_field_names=["b", "c"]
        )
        old_data = Data(torch.tensor(3), "b", "c")
        new_data = python_pytree.tree_map(lambda x: x, old_data)
        self.assertEqual(new_data.a, torch.tensor(3))
        self.assertEqual(new_data.b, "moo")
        self.assertEqual(new_data.c, None)
        python_pytree._deregister_pytree_node(Data)

    def test_register_dataclass_class(self):
        class CustomClass:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        with self.assertRaisesRegex(ValueError, "field_names must be specified"):
            python_pytree.register_dataclass(CustomClass)

        python_pytree.register_dataclass(CustomClass, field_names=["x", "y"])
        try:
            c = CustomClass(torch.tensor(0), torch.tensor(1))
            mapped = python_pytree.tree_map(lambda x: x + 1, c)
            self.assertEqual(mapped.x, torch.tensor(1))
            self.assertEqual(mapped.y, torch.tensor(2))
        finally:
            python_pytree._deregister_pytree_node(CustomClass)

    def test_constant(self):
        # Either use `frozen=True` or `unsafe_hash=True` so we have a
        # non-default `__hash__`.
        @dataclass(unsafe_hash=True)
        class Config:
            norm: str

        python_pytree.register_constant(Config)

        config = Config("l1")
        elements, spec = python_pytree.tree_flatten(config)
        self.assertEqual(elements, [])
        self.assertEqual(spec.context.value, config)

    def test_constant_default_eq_error(self):
        class Config:
            def __init__(self, norm: str):
                self.norm = norm

        try:
            python_pytree.register_constant(Config)
            self.assertFalse(True)  # must raise error before this
        except TypeError as e:
            msg = "register_constant(cls) expects `cls` to have a non-default `__eq__` implementation."
            self.assertIn(msg, str(e))

    def test_constant_default_hash_error(self):
        class Config:
            def __init__(self, norm: str):
                self.norm = norm

            def __eq__(self, other):
                return self.norm == other.norm

        try:
            python_pytree.register_constant(Config)
            self.assertFalse(True)  # must raise error before this
        except TypeError as e:
            msg = "register_constant(cls) expects `cls` to have a non-default `__hash__` implementation."
            self.assertIn(msg, str(e))

    def test_tree_map_with_path_multiple_trees(self):
        @dataclass
        class ACustomPytree:
            x: Any
            y: Any
            z: Any

        tree1 = [ACustomPytree(x=12, y={"cin": [1, 4, 10], "bar": 18}, z="leaf"), 5]
        tree2 = [ACustomPytree(x=2, y={"cin": [2, 2, 2], "bar": 2}, z="leaf"), 2]

        python_pytree.register_pytree_node(
            ACustomPytree,
            flatten_fn=lambda f: ([f.x, f.y], f.z),
            unflatten_fn=lambda xy, z: ACustomPytree(xy[0], xy[1], z),
            flatten_with_keys_fn=lambda f: ((("x", f.x), ("y", f.y)), f.z),
        )
        from_two_trees = python_pytree.tree_map_with_path(
            lambda kp, a, b: a + b, tree1, tree2
        )
        from_one_tree = python_pytree.tree_map(lambda a: a + 2, tree1)
        self.assertEqual(from_two_trees, from_one_tree)

    def test_tree_flatten_with_path_is_leaf(self):
        leaf_dict = {"foo": [(3)]}
        tree = (["hello", [1, 2], leaf_dict],)
        key_leaves, _ = python_pytree.tree_flatten_with_path(
            tree, is_leaf=lambda x: isinstance(x, dict)
        )
        self.assertTrue(key_leaves[-1][1] is leaf_dict)

    def test_tree_flatten_with_path_roundtrip(self):
        class ANamedTuple(NamedTuple):
            x: torch.Tensor
            y: int
            z: str

        @dataclass
        class ACustomPytree:
            x: Any
            y: Any
            z: Any

        python_pytree.register_pytree_node(
            ACustomPytree,
            flatten_fn=lambda f: ([f.x, f.y], f.z),
            unflatten_fn=lambda xy, z: ACustomPytree(xy[0], xy[1], z),
            flatten_with_keys_fn=lambda f: ((("x", f.x), ("y", f.y)), f.z),
        )

        SOME_PYTREES = [
            (None,),
            ["hello", [1, 2], {"foo": [(3)]}],
            [ANamedTuple(x=torch.rand(2, 3), y=1, z="foo")],
            [ACustomPytree(x=12, y={"cin": [1, 4, 10], "bar": 18}, z="leaf"), 5],
        ]
        for tree in SOME_PYTREES:
            key_leaves, spec = python_pytree.tree_flatten_with_path(tree)
            actual = python_pytree.tree_unflatten(
                [leaf for _, leaf in key_leaves], spec
            )
            self.assertEqual(actual, tree)

    def test_tree_leaves_with_path(self):
        class ANamedTuple(NamedTuple):
            x: torch.Tensor
            y: int
            z: str

        @dataclass
        class ACustomPytree:
            x: Any
            y: Any
            z: Any

        python_pytree.register_pytree_node(
            ACustomPytree,
            flatten_fn=lambda f: ([f.x, f.y], f.z),
            unflatten_fn=lambda xy, z: ACustomPytree(xy[0], xy[1], z),
            flatten_with_keys_fn=lambda f: ((("x", f.x), ("y", f.y)), f.z),
        )

        SOME_PYTREES = [
            (None,),
            ["hello", [1, 2], {"foo": [(3)]}],
            [ANamedTuple(x=torch.rand(2, 3), y=1, z="foo")],
            [ACustomPytree(x=12, y={"cin": [1, 4, 10], "bar": 18}, z="leaf"), 5],
        ]
        for tree in SOME_PYTREES:
            flat_out, _ = python_pytree.tree_flatten_with_path(tree)
            leaves_out = python_pytree.tree_leaves_with_path(tree)
            self.assertEqual(flat_out, leaves_out)

    def test_key_str(self):
        class ANamedTuple(NamedTuple):
            x: str
            y: int

        tree = (["hello", [1, 2], {"foo": [(3)], "bar": [ANamedTuple(x="baz", y=10)]}],)
        flat, _ = python_pytree.tree_flatten_with_path(tree)
        paths = [f"{python_pytree.keystr(kp)}: {val}" for kp, val in flat]
        self.assertEqual(
            paths,
            [
                "[0][0]: hello",
                "[0][1][0]: 1",
                "[0][1][1]: 2",
                "[0][2]['foo'][0]: 3",
                "[0][2]['bar'][0].x: baz",
                "[0][2]['bar'][0].y: 10",
            ],
        )

    def test_flatten_flatten_with_key_consistency(self):
        """Check that flatten and flatten_with_key produces consistent leaves/context."""
        reg = python_pytree.SUPPORTED_NODES

        EXAMPLE_TREE = {
            list: [1, 2, 3],
            tuple: (1, 2, 3),
            dict: {"foo": 1, "bar": 2},
            namedtuple: namedtuple("ANamedTuple", ["x", "y"])(1, 2),
            OrderedDict: OrderedDict([("foo", 1), ("bar", 2)]),
            defaultdict: defaultdict(int, {"foo": 1, "bar": 2}),
            deque: deque([1, 2, 3]),
            torch.Size: torch.Size([1, 2, 3]),
            immutable_dict: immutable_dict({"foo": 1, "bar": 2}),
            immutable_list: immutable_list([1, 2, 3]),
        }

        for typ in reg:
            example = EXAMPLE_TREE.get(typ)
            if example is None:
                continue
            flat_with_path, spec1 = python_pytree.tree_flatten_with_path(example)
            flat, spec2 = python_pytree.tree_flatten(example)

            self.assertEqual(flat, [x[1] for x in flat_with_path])
            self.assertEqual(spec1, spec2)

    def test_key_access(self):
        class ANamedTuple(NamedTuple):
            x: str
            y: int

        tree = (["hello", [1, 2], {"foo": [(3)], "bar": [ANamedTuple(x="baz", y=10)]}],)
        flat, _ = python_pytree.tree_flatten_with_path(tree)
        for kp, val in flat:
            self.assertEqual(python_pytree.key_get(tree, kp), val)


class TestCxxPytree(TestCase):
    def setUp(self):
        if IS_FBCODE:
            raise unittest.SkipTest("C++ pytree tests are not supported in fbcode")

    def assertEqual(self, x, y, *args, **kwargs):
        x_typename, y_typename = type(x).__name__, type(y).__name__
        if not ("treespec" in x_typename.lower() or "treespec" in y_typename.lower()):
            super().assertEqual(x, y, *args, **kwargs)

        # The Dynamo polyfill returns a polyfilled Python class for C++ PyTreeSpec instead of the
        # C++ class. So we compare the type names and reprs instead because the types themselves
        # won't be equal.
        super().assertEqual(x_typename, y_typename, *args, **kwargs)
        if not TEST_WITH_TORCHDYNAMO or type(x) is type(y):
            super().assertEqual(x, y, *args, **kwargs)
        else:
            super().assertEqual(
                x.unflatten(range(x.num_leaves)),
                y.unflatten(range(y.num_leaves)),
                *args,
                **kwargs,
            )

    def test_treespec_equality(self):
        self.assertEqual(cxx_pytree.treespec_leaf(), cxx_pytree.treespec_leaf())

    def test_treespec_repr(self):
        # Check that it looks sane
        tree = (0, [0, 0, [0]])
        spec = cxx_pytree.tree_structure(tree)
        self.assertEqual(
            repr(spec), "PyTreeSpec((*, [*, *, [*]]), NoneIsLeaf, namespace='torch')"
        )

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
            cxx_pytree.tree_structure(
                defaultdict(list, {"a": [0, 1], "b": [1, 2], "c": {}})
            ),
        ],
    )
    def test_pytree_serialize(self, spec):
        self.assertEqual(
            spec,
            cxx_pytree.tree_structure(
                cxx_pytree.tree_unflatten([0] * spec.num_leaves, spec)
            ),
        )

        serialized_spec = cxx_pytree.treespec_dumps(spec)
        self.assertIsInstance(serialized_spec, str)

        roundtrip_spec = cxx_pytree.treespec_loads(serialized_spec)
        self.assertEqual(roundtrip_spec, spec)

    def test_pytree_serialize_namedtuple(self):
        python_pytree._register_namedtuple(
            GlobalPoint,
            serialized_type_name="test_pytree.test_pytree_serialize_namedtuple.GlobalPoint",
        )
        spec = cxx_pytree.tree_structure(GlobalPoint(0, 1))

        roundtrip_spec = cxx_pytree.treespec_loads(cxx_pytree.treespec_dumps(spec))
        self.assertEqual(roundtrip_spec.type._fields, spec.type._fields)

        LocalPoint = namedtuple("LocalPoint", ["x", "y"])
        python_pytree._register_namedtuple(
            LocalPoint,
            serialized_type_name="test_pytree.test_pytree_serialize_namedtuple.LocalPoint",
        )
        spec = cxx_pytree.tree_structure(LocalPoint(0, 1))

        roundtrip_spec = cxx_pytree.treespec_loads(cxx_pytree.treespec_dumps(spec))
        self.assertEqual(roundtrip_spec.type._fields, spec.type._fields)

    def test_pytree_custom_type_serialize(self):
        spec = cxx_pytree.tree_structure(GlobalDummyType(0, 1))
        serialized_spec = cxx_pytree.treespec_dumps(spec)
        roundtrip_spec = cxx_pytree.treespec_loads(serialized_spec)
        self.assertEqual(roundtrip_spec, spec)

        class LocalDummyType:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __eq__(self, other):
                if not isinstance(other, LocalDummyType):
                    return NotImplemented
                return self.x == other.x and self.y == other.y

            def __hash__(self):
                return hash((self.x, self.y))

        cxx_pytree.register_pytree_node(
            LocalDummyType,
            lambda dummy: ([dummy.x, dummy.y], None),
            lambda xs, _: LocalDummyType(*xs),
            serialized_type_name="LocalDummyType",
        )
        spec = cxx_pytree.tree_structure(LocalDummyType(0, 1))
        serialized_spec = cxx_pytree.treespec_dumps(spec)
        roundtrip_spec = cxx_pytree.treespec_loads(serialized_spec)
        self.assertEqual(roundtrip_spec, spec)


instantiate_parametrized_tests(TestGenericPytree)
instantiate_parametrized_tests(TestPythonPytree)
instantiate_parametrized_tests(TestCxxPytree)


if __name__ == "__main__":
    run_tests()
