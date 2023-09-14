# Owner(s): ["module: dynamo"]
import functools

import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._C._dynamo import guards

GuardManager = guards.GuardManager
ItemGuardAccessor = guards.ItemGuardAccessor
AttrGuardAccessor = guards.AttrGuardAccessor
PythonLambdaGuardAccessor = guards.PythonLambdaGuardAccessor


def equals_match(x, expected):
    return x == expected


def equals_match_failure_fn(x, expected):
    return f"expected {expected} found {x}"


def ge_match(x, expected):
    return x >= expected


def ge_match_failure_fn(x, expected):
    return f"expected >= {expected} found {x}"


def less_match(x, expected):
    return x < expected


def less_match_failure_fn(x, expected):
    return f"expected < {expected} found {x}"


class GuardManagerTests(torch._dynamo.test_case.TestCase):
    def test_python_lambda_leaf_guard(self):
        const_guard = guards.PythonLambdaGuard(
            functools.partial(equals_match, expected=5),
            functools.partial(equals_match_failure_fn, expected=5),
        )
        self.assertTrue(const_guard(5))
        self.assertFalse(const_guard(4))
        self.assertFalse(const_guard("foo"))

    def test_guard_manager_leaf_guard(self):
        guard_manager = GuardManager()
        guard_manager.add_lambda_guard(
            lambda x: isinstance(x, int),
            lambda x: f"Expected int but got {type(x)}",
        )
        guard_manager.add_lambda_guard(
            functools.partial(ge_match, expected=5),
            functools.partial(ge_match_failure_fn, expected=5),
        )
        guard_manager.add_lambda_guard(
            functools.partial(less_match, expected=10),
            functools.partial(less_match_failure_fn, expected=10),
        )
        self.assertEqual(len(guard_manager.get_leaf_guards()), 3)
        self.assertEqual(len(guard_manager.get_accessors()), 0)
        self.assertTrue(guard_manager.check(6))
        self.assertFalse(guard_manager.check(4))
        self.assertFalse(guard_manager.check("foo"))

    def test_attr_guard_manager(self):
        class Foo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        foo = Foo(1, 2)
        guard_manager = GuardManager()
        guard_manager.add_lambda_guard(
            lambda x: isinstance(x, Foo),
            lambda x: f"Expected Foo but got {type(x)}",
        )
        guard_manager.x.add_lambda_guard(
            functools.partial(equals_match, expected=foo.x),
            functools.partial(equals_match_failure_fn, expected=foo.x),
        )
        guard_manager.y.add_lambda_guard(
            functools.partial(equals_match, expected=foo.y),
            functools.partial(equals_match_failure_fn, expected=foo.y),
        )
        self.assertEqual(len(guard_manager.get_leaf_guards()), 1)
        # 2 child managers, one for x and one for y
        self.assertEqual(len(guard_manager.get_accessors()), 2)
        self.assertTrue(isinstance(guard_manager.get_accessors()[0], AttrGuardAccessor))
        self.assertTrue(isinstance(guard_manager.get_accessors()[1], AttrGuardAccessor))
        # Check leaf guards on child managers
        self.assertEqual(len(guard_manager.x.get_leaf_guards()), 1)
        self.assertEqual(len(guard_manager.y.get_leaf_guards()), 1)

        self.assertTrue(guard_manager.check(foo))
        self.assertFalse(guard_manager.check(Foo(3, 4)))
        self.assertFalse(guard_manager.check("foo"))

    def test_item_guard_manager(self):
        class Foo:
            def __init__(self, x, y):
                self._x = x
                self._y = y

            def __getitem__(self, name):
                if name == "x":
                    return self._x
                elif name == "y":
                    return self._y
                else:
                    raise KeyError(f"{name} not in {self}")

        foo = Foo(1, 2)
        guard_manager = GuardManager()
        guard_manager.add_lambda_guard(
            lambda x: isinstance(x, Foo),
            lambda x: f"Expected Foo but got {type(x)}",
        )
        guard_manager["x"].add_lambda_guard(
            functools.partial(equals_match, expected=foo["x"]),
            functools.partial(equals_match_failure_fn, expected=foo["x"]),
        )
        guard_manager["y"].add_lambda_guard(
            functools.partial(equals_match, expected=foo["y"]),
            functools.partial(equals_match_failure_fn, expected=foo["y"]),
        )
        self.assertEqual(len(guard_manager.get_leaf_guards()), 1)
        # 2 child managers, one for x and one for y
        self.assertEqual(len(guard_manager.get_accessors()), 2)
        self.assertTrue(isinstance(guard_manager.get_accessors()[0], ItemGuardAccessor))
        self.assertTrue(isinstance(guard_manager.get_accessors()[1], ItemGuardAccessor))
        # Check leaf guards on child managers
        self.assertEqual(len(guard_manager["x"].get_leaf_guards()), 1)
        self.assertEqual(len(guard_manager["y"].get_leaf_guards()), 1)

        self.assertTrue(guard_manager.check(foo))
        self.assertFalse(guard_manager.check(Foo(3, 4)))
        self.assertFalse(guard_manager.check("foo"))

    def test_dict_guard_manager(self):
        foo = {
            "x": 1,
            "y": 2,
        }
        guard_manager = GuardManager()
        guard_manager.add_lambda_guard(
            lambda x: isinstance(x, dict),
            lambda x: f"Expected dict but got {type(x)}",
        )
        guard_manager["x"].add_lambda_guard(
            functools.partial(equals_match, expected=foo["x"]),
            functools.partial(equals_match_failure_fn, expected=foo["x"]),
        )
        guard_manager["y"].add_lambda_guard(
            functools.partial(equals_match, expected=foo["y"]),
            functools.partial(equals_match_failure_fn, expected=foo["y"]),
        )
        self.assertEqual(len(guard_manager.get_leaf_guards()), 1)
        # 2 child managers, one for x and one for y
        self.assertEqual(len(guard_manager.get_accessors()), 2)
        self.assertTrue(isinstance(guard_manager.get_accessors()[0], ItemGuardAccessor))
        self.assertTrue(isinstance(guard_manager.get_accessors()[1], ItemGuardAccessor))
        # Check leaf guards on child managers
        self.assertEqual(len(guard_manager["x"].get_leaf_guards()), 1)
        self.assertEqual(len(guard_manager["y"].get_leaf_guards()), 1)

        self.assertTrue(guard_manager.check(foo))
        self.assertFalse(guard_manager.check({"x": 3, "y": 4}))
        self.assertFalse(guard_manager.check("foo"))

    def test_lambda_accessor(self):
        def accessor_fn(x):
            return type(x)

        guard_manager = GuardManager()
        guard_manager.lambda_accessor(accessor_fn).add_lambda_guard(
            lambda x: x is int, lambda x: "type mismatch"
        )
        guard_manager.lambda_accessor(accessor_fn).add_lambda_guard(
            lambda x: x is not str, lambda x: "type mismatch"
        )
        self.assertEqual(len(guard_manager.get_leaf_guards()), 0)
        self.assertEqual(len(guard_manager.get_accessors()), 1)
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[0], PythonLambdaGuardAccessor)
        )
        self.assertEqual(
            len(guard_manager.lambda_accessor(accessor_fn).get_leaf_guards()), 2
        )

        self.assertTrue(guard_manager.check(5))
        self.assertFalse(guard_manager.check("foo"))

    def test_reshuffling_and_reason(self):
        class Pair:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        f_locals = {
            "foo": 5,
            "bar": Pair(1, 2),
        }
        guard_manager = GuardManager()

        guard_manager["foo"].add_lambda_guard(
            lambda x: isinstance(x, int),
            lambda x: f"Expected int but got {type(x)}",
        )
        # Just add same guard to test if guard reshuffling happens on failure
        for _ in range(5):
            guard_manager["foo"].add_lambda_guard(
                functools.partial(equals_match, expected=5),
                functools.partial(equals_match_failure_fn, expected=5),
            )

        guard_manager["bar"].add_lambda_guard(
            lambda x: isinstance(x, Pair),
            lambda x: f"Expected Pair but got {type(x)}",
        )
        guard_manager["bar"].x.add_lambda_guard(
            lambda x: isinstance(x, int),
            lambda x: f"Expected int but got {type(x)}",
        )
        guard_manager["bar"].x.add_lambda_guard(
            functools.partial(equals_match, expected=1),
            functools.partial(equals_match_failure_fn, expected=1),
        )
        guard_manager["bar"].y.add_lambda_guard(
            lambda x: isinstance(x, int),
            lambda x: f"Expected int but got {type(x)}",
        )
        guard_manager["bar"].y.add_lambda_guard(
            functools.partial(equals_match, expected=2),
            functools.partial(equals_match_failure_fn, expected=2),
        )

        # Check structure
        self.assertEqual(len(guard_manager.get_leaf_guards()), 0)
        self.assertEqual(len(guard_manager.get_accessors()), 2)
        self.assertTrue(isinstance(guard_manager.get_accessors()[0], ItemGuardAccessor))
        self.assertTrue(isinstance(guard_manager.get_accessors()[1], ItemGuardAccessor))
        self.assertEqual(len(guard_manager["foo"].get_leaf_guards()), 6)
        self.assertEqual(len(guard_manager["bar"].get_leaf_guards()), 1)
        self.assertEqual(len(guard_manager["bar"].get_accessors()), 2)
        self.assertTrue(
            isinstance(guard_manager["bar"].get_accessors()[0], AttrGuardAccessor)
        )
        self.assertTrue(
            isinstance(guard_manager["bar"].get_accessors()[1], AttrGuardAccessor)
        )
        self.assertEqual(len(guard_manager["bar"].x.get_leaf_guards()), 2)
        self.assertEqual(len(guard_manager["bar"].y.get_leaf_guards()), 2)

        # Check happy case
        self.assertTrue(guard_manager.check(f_locals))

        # Check with debug info to test reshuffling of guards
        class PairImpostor:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        # changing bar type, this means that child managers will shuffle after the first failure.
        f_locals_perturbed = {
            "foo": 5,
            "bar": PairImpostor(1, 2),
        }
        result, first_debug_info = guard_manager.check_with_debug_info(
            f_locals_perturbed
        )
        self.assertFalse(result)
        self.assertTrue("Expected Pair but got" in first_debug_info.failure_reason)
        result, second_debug_info = guard_manager.check_with_debug_info(
            f_locals_perturbed
        )
        self.assertFalse(result)
        self.assertTrue(
            first_debug_info.num_guards_executed > second_debug_info.num_guards_executed
        )
