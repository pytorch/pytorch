# Owner(s): ["module: dynamo"]
import enum
import functools
from collections import OrderedDict

import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._C._dynamo import guards

RootGuardManager = guards.RootGuardManager
GetAttrGuardAccessor = guards.GetAttrGuardAccessor
GetItemGuardAccessor = guards.GetItemGuardAccessor
GetDictItemGuardAccessor = guards.GetDictItemGuardAccessor
TypeGuardAccessor = guards.TypeGuardAccessor
NoTensorAliasingGuard = guards.NoTensorAliasingGuard
install_no_tensor_aliasing_guard = guards.install_no_tensor_aliasing_guard


global_pair = {
    "x": torch.randn(4),
    "y": 1,
}


def equals_match(x, expected):
    return x == expected


def equals_match_failure_fn(expected):
    return f"x == {expected}"


def ge_match(x, expected):
    return x >= expected


def ge_match_failure_fn(expected):
    return f"expected >= {expected}"


def less_match(x, expected):
    return x < expected


def less_match_failure_fn(expected):
    return f"expected < {expected}"


def id_type(x):
    return id(type(x))


class GuardManagerTests(torch._dynamo.test_case.TestCase):
    def test_python_lambda_leaf_guard(self):
        const_guard = guards.PythonLambdaGuard(
            functools.partial(equals_match, expected=5),
            equals_match_failure_fn(5),
        )
        self.assertTrue(const_guard(5))
        self.assertFalse(const_guard(4))
        self.assertFalse(const_guard("foo"))

    def test_type_guard(self):
        foo = 4
        guard = guards.TYPE_MATCH(id_type(foo), "type(x) == int")

        self.assertTrue(guard(5))
        self.assertTrue(guard(4))
        self.assertFalse(guard("foo"))

        foo = {"a": 1}
        guard = guards.TYPE_MATCH(id_type(foo), "type(x) == dict")
        self.assertTrue(guard(foo))
        self.assertTrue(guard({}))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

        class Foo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        foo = Foo(1, 2)

        guard = guards.TYPE_MATCH(id_type(foo), "type(x) == Foo")
        self.assertTrue(guard(foo))
        self.assertFalse(guard({}))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

    def test_id_guard(self):
        foo = 4
        guard = guards.ID_MATCH(id(foo), "id(x) == id(foo)")

        self.assertTrue(guard(foo))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

        foo = {"a": 1}
        guard = guards.ID_MATCH(id(foo), "id(x) == id(foo)")
        self.assertTrue(guard(foo))
        self.assertFalse(guard({"a": 1}))
        self.assertFalse(guard({}))
        self.assertFalse(guard(5))

    def test_equals_guard(self):
        foo = 4
        guard = guards.EQUALS_MATCH(foo, "x == 4")

        self.assertTrue(guard(4))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

        # dict
        foo = {"a": 1}
        guard = guards.EQUALS_MATCH(foo, "x == foo")
        self.assertTrue(guard(foo))
        self.assertTrue(guard({"a": 1}))
        self.assertFalse(guard({}))
        self.assertFalse(guard({"a": 1, "b": 2}))
        self.assertFalse(guard(5))

        # tuple
        foo = (1, 2, 3)
        guard = guards.EQUALS_MATCH(foo, "x == foo")
        self.assertTrue(guard(foo))
        self.assertTrue(guard((1, 2, 3)))
        self.assertFalse(guard((1, 2, 3, 4)))
        self.assertFalse(guard({}))

        # list
        foo = [1, 2, 3]
        guard = guards.EQUALS_MATCH(foo, "x == foo")
        self.assertTrue(guard(foo))
        self.assertTrue(guard([1, 2, 3]))
        self.assertFalse(guard([1, 2, 3, 4]))

        # type
        foo = int
        guard = guards.EQUALS_MATCH(foo, "x == foo")
        self.assertTrue(guard(foo))
        self.assertTrue(guard(int))
        self.assertFalse(guard(float))

        # TODO(janimesh) - Add more tests for other types

    def test_guard_manager_leaf_guard(self):
        guard_manager = RootGuardManager()
        guard_manager.add_type_match_guard(id_type(5), "type(x) == int")
        guard_manager.add_lambda_guard(
            functools.partial(ge_match, expected=5),
            ge_match_failure_fn(expected=5),
        )
        guard_manager.add_lambda_guard(
            functools.partial(less_match, expected=10),
            less_match_failure_fn(expected=10),
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
        guard_manager = RootGuardManager()
        guard_manager.add_type_match_guard(id_type(foo), "type(x) == Foo")
        guard_manager.x.add_lambda_guard(
            functools.partial(equals_match, expected=foo.x),
            equals_match_failure_fn(foo.x),
        )
        guard_manager.y.add_lambda_guard(
            functools.partial(equals_match, expected=foo.y),
            equals_match_failure_fn(foo.y),
        )
        self.assertEqual(len(guard_manager.get_leaf_guards()), 1)
        # 2 child managers, one for x and one for y
        self.assertEqual(len(guard_manager.get_accessors()), 2)
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[0], GetAttrGuardAccessor)
        )
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[1], GetAttrGuardAccessor)
        )
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
        guard_manager = RootGuardManager()
        guard_manager.add_type_match_guard(id_type(foo), "type(x) == Foo")
        guard_manager["x"].add_lambda_guard(
            functools.partial(equals_match, expected=foo["x"]),
            equals_match_failure_fn(foo["x"]),
        )
        guard_manager["y"].add_lambda_guard(
            functools.partial(equals_match, expected=foo["y"]),
            equals_match_failure_fn(foo["y"]),
        )
        self.assertEqual(len(guard_manager.get_leaf_guards()), 1)
        # 2 child managers, one for x and one for y
        self.assertEqual(len(guard_manager.get_accessors()), 2)
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[0], GetItemGuardAccessor)
        )
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[1], GetItemGuardAccessor)
        )
        # Check leaf guards on child managers
        self.assertEqual(len(guard_manager["x"].get_leaf_guards()), 1)
        self.assertEqual(len(guard_manager["y"].get_leaf_guards()), 1)

        self.assertTrue(guard_manager.check(foo))
        self.assertFalse(guard_manager.check(Foo(3, 4)))
        self.assertFalse(guard_manager.check("foo"))

    def test_item_int_guard_manager(self):
        foo = (1, 2, 3)

        guard_manager = RootGuardManager()
        guard_manager.add_type_match_guard(id_type(foo), "type(x) == tuple")
        guard_manager[0].add_lambda_guard(
            lambda x: x == 1,
            "Expected int",
        )

        self.assertTrue(guard_manager.check(foo))

    def test_item_slice_guard_manager(self):
        foo = [1, 2, 3]

        guard_manager = RootGuardManager()
        guard_manager.add_lambda_guard(
            lambda x: isinstance(x, list),
            "Expected tuple",
        )
        guard_manager[slice(2)].add_lambda_guard(
            lambda x: x[0] == 1,
            "Expected int",
        )

        self.assertTrue(guard_manager.check(foo))
        foo[0] = 5
        self.assertFalse(guard_manager.check(foo))

    def test_item_enum_guard_manager(self):
        class MyEnum(enum.Enum):
            FOO = 10
            BAR = 20

        foo = {
            MyEnum.FOO: 1,
            MyEnum.BAR: 2,
        }

        guard_manager = RootGuardManager()
        guard_manager.add_lambda_guard(
            lambda x: isinstance(x, dict),
            "Expected dict",
        )
        guard_manager[MyEnum.FOO].add_lambda_guard(
            lambda x: x == 1,
            "Expected int",
        )
        guard_manager[MyEnum.BAR].add_lambda_guard(
            lambda x: x == 2,
            "Expected int",
        )
        self.assertTrue(guard_manager.check(foo))
        self.assertFalse(guard_manager.check("foo"))

    def test_dict_guard_manager(self):
        foo = {
            "x": 1,
            "y": 2,
        }
        guard_manager = RootGuardManager()
        guard_manager.add_lambda_guard(
            lambda x: isinstance(x, dict),
            "type(x) is dict",
        )
        guard_manager.dict_get_item_manager("x").add_lambda_guard(
            functools.partial(equals_match, expected=foo["x"]),
            equals_match_failure_fn(foo["x"]),
        )
        guard_manager.dict_get_item_manager("y").add_lambda_guard(
            functools.partial(equals_match, expected=foo["y"]),
            equals_match_failure_fn(foo["y"]),
        )
        self.assertEqual(len(guard_manager.get_leaf_guards()), 1)
        # 2 child managers, one for x and one for y
        self.assertEqual(len(guard_manager.get_accessors()), 2)
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[0], GetDictItemGuardAccessor)
        )
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[1], GetDictItemGuardAccessor)
        )
        # Check leaf guards on child managers
        self.assertEqual(
            len(guard_manager.dict_get_item_manager("x").get_leaf_guards()), 1
        )
        self.assertEqual(
            len(guard_manager.dict_get_item_manager("y").get_leaf_guards()), 1
        )

        self.assertTrue(guard_manager.check(foo))
        self.assertFalse(guard_manager.check({"x": 3, "y": 4}))
        self.assertFalse(guard_manager.check("foo"))

    def test_ordered_dict_guard_manager(self):
        foo = OrderedDict([("x", 1), ("y", 2)])
        guard_manager = RootGuardManager()
        guard_manager.add_lambda_guard(
            lambda x: isinstance(x, OrderedDict),
            "Expected OrderedDict",
        )
        guard_manager.dict_get_item_manager("x").add_lambda_guard(
            lambda x: x == 1,
            "Expected int",
        )
        guard_manager.dict_get_item_manager("y").add_lambda_guard(
            lambda x: x == 2,
            "Expected int",
        )
        # self.assertEqual(len(guard_manager.get_leaf_guards()), 1)
        self.assertTrue(guard_manager.check(foo))

    def test_tensor_aliasing_guard(self):
        guard_manager = RootGuardManager()
        f_locals = {
            "x": torch.randn(3, 4),
            "y": torch.randn(3, 4),
        }

        x_guard_mgr = guard_manager.dict_get_item_manager("x")
        y_guard_mgr = guard_manager.dict_get_item_manager("y")
        install_no_tensor_aliasing_guard(x_guard_mgr, y_guard_mgr, "x is not y")

        # Check structure
        x_guards = x_guard_mgr.get_leaf_guards()
        y_guards = y_guard_mgr.get_leaf_guards()
        self.assertEqual(len(x_guards), 1)
        self.assertEqual(len(y_guards), 1)
        self.assertTrue(isinstance(x_guards[0], NoTensorAliasingGuard))
        self.assertTrue(isinstance(y_guards[0], NoTensorAliasingGuard))
        # Check that the two guards are the same object
        self.assertTrue(x_guards[0] is y_guards[0])

        self.assertEqual(len(x_guard_mgr.get_leaf_guards()), 1)
        self.assertEqual(len(y_guard_mgr.get_leaf_guards()), 1)
        self.assertTrue(guard_manager.check(f_locals))

        a = torch.randn(3, 4)
        f_locals_aliased = {
            "x": a,
            "y": a,
        }
        self.assertFalse(guard_manager.check(f_locals_aliased))

        a = torch.randn(3, 4)
        f_locals_not_aliased_same_value = {
            "x": torch.zeros(4),
            "y": torch.zeros(4),
        }
        self.assertTrue(guard_manager.check(f_locals_not_aliased_same_value))

    def test_tensor_aliasing_guard_reset(self):
        # Check that guard state is reset on failure
        guard_manager = RootGuardManager()

        a = torch.randn(3, 4)
        b = torch.randn(3, 4)
        f_locals = {
            "x": a,
            "y": 4,
            "z": b,
        }

        x_guard_mgr = guard_manager.dict_get_item_manager("x")
        y_guard_mgr = guard_manager.dict_get_item_manager("y")
        z_guard_mgr = guard_manager.dict_get_item_manager("z")

        install_no_tensor_aliasing_guard(x_guard_mgr, z_guard_mgr, "x is not y")
        y_guard_mgr.add_lambda_guard(
            lambda x: x == 4,
            "Expected int",
        )

        # first use check_verbose as it does not shuffle the guards on failures.
        # The order of accessors is x, y and z . Let the guard fail on y. This
        # would call the tensor aliasing guard for x.
        f_locals_to_fail = {
            "x": a,
            "y": 5,
            "z": a,
        }
        self.assertFalse(guard_manager.check_verbose(f_locals_to_fail).result)
        # Now if we did not reset the guard on x, it would be expecting a tensor
        # not aliased to a. Lets send an input that is supposed to eval to True
        # but with "x" : a
        f_locals = {
            "x": a,
            "y": 4,
            "z": b,
        }
        self.assertTrue(guard_manager.check_verbose(f_locals).result)

        # Lets check the same behavior using check function.
        self.assertFalse(guard_manager.check(f_locals_to_fail))
        f_locals = {
            "x": b,
            "y": 4,
            "z": a,
        }
        self.assertTrue(guard_manager.check(f_locals))

    def test_reshuffling_and_reason(self):
        class Pair:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        f_locals = {
            "foo": 5,
            "bar": Pair(1, 2),
        }
        guard_manager = RootGuardManager()

        guard_manager.dict_get_item_manager("foo").add_lambda_guard(
            lambda x: isinstance(x, int),
            "Expected int",
        )
        # Just add same guard to test if guard reshuffling happens on failure
        for _ in range(5):
            guard_manager.dict_get_item_manager("foo").add_lambda_guard(
                functools.partial(equals_match, expected=5),
                equals_match_failure_fn(5),
            )

        guard_manager.dict_get_item_manager("bar").add_lambda_guard(
            lambda x: isinstance(x, Pair),
            "Expected Pair",
        )
        guard_manager.dict_get_item_manager("bar").x.add_lambda_guard(
            lambda x: isinstance(x, int),
            "Expected int",
        )
        guard_manager.dict_get_item_manager("bar").x.add_lambda_guard(
            functools.partial(equals_match, expected=1),
            equals_match_failure_fn(1),
        )
        guard_manager.dict_get_item_manager("bar").y.add_lambda_guard(
            lambda x: isinstance(x, int),
            "Expected int",
        )
        guard_manager.dict_get_item_manager("bar").y.add_lambda_guard(
            functools.partial(equals_match, expected=2),
            equals_match_failure_fn(2),
        )

        # Check structure
        self.assertEqual(len(guard_manager.get_leaf_guards()), 0)
        self.assertEqual(len(guard_manager.get_accessors()), 2)
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[0], GetDictItemGuardAccessor)
        )
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[1], GetDictItemGuardAccessor)
        )
        self.assertEqual(
            len(guard_manager.dict_get_item_manager("foo").get_leaf_guards()), 6
        )
        self.assertEqual(
            len(guard_manager.dict_get_item_manager("bar").get_leaf_guards()), 1
        )
        self.assertEqual(
            len(guard_manager.dict_get_item_manager("bar").get_accessors()), 2
        )
        self.assertTrue(
            isinstance(
                guard_manager.dict_get_item_manager("bar").get_accessors()[0],
                GetAttrGuardAccessor,
            )
        )
        self.assertTrue(
            isinstance(
                guard_manager.dict_get_item_manager("bar").get_accessors()[1],
                GetAttrGuardAccessor,
            )
        )
        self.assertEqual(
            len(guard_manager.dict_get_item_manager("bar").x.get_leaf_guards()), 2
        )
        self.assertEqual(
            len(guard_manager.dict_get_item_manager("bar").y.get_leaf_guards()), 2
        )

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
        first_debug_info = guard_manager.check_verbose(f_locals_perturbed)
        self.assertFalse(first_debug_info.result)
        self.assertTrue("Expected Pair" in first_debug_info.failed_guard)

        guard_manager.check(f_locals_perturbed)

        second_debug_info = guard_manager.check_verbose(f_locals_perturbed)
        self.assertFalse(second_debug_info.result)
        self.assertTrue(
            first_debug_info.num_guards_executed > second_debug_info.num_guards_executed
        )

    def test_globals(self):
        guard_manager = RootGuardManager()
        gpair_mgr = guard_manager.globals_dict_manager(globals()).dict_get_item_manager(
            "global_pair"
        )

        gpair_mgr.add_lambda_guard(
            lambda x: isinstance(x, dict)
            and isinstance(x["x"], torch.Tensor)
            and isinstance(x["y"], int),
            "global guard fail",
        )

        global global_pair
        self.assertTrue(guard_manager.check(global_pair))
        global_pair["y"] = "foo"
        self.assertFalse(guard_manager.check(global_pair))

    def test_type(self):
        guard_manager = RootGuardManager()

        class A:
            a = 4

        class B(A):
            def mul(self, x):
                super().mul(x)

        foo = B()
        f_locals = {"foo": foo}

        # len(type(foo).__mro__) == 2
        foo_mgr = guard_manager.dict_get_item_manager("foo")
        type_manager = foo_mgr.type_manager()
        self.assertTrue(isinstance(foo_mgr.get_accessors()[0], TypeGuardAccessor))
        mro_manager = type_manager.__mro__
        self.assertTrue(
            isinstance(type_manager.get_accessors()[0], GetAttrGuardAccessor)
        )
        mro_manager.add_length_check_guard(
            3,
            "Expected length 2",
        )

        # type(foo).__mro__[0].a = 4
        item_manager = mro_manager[1]
        self.assertTrue(
            isinstance(mro_manager.get_accessors()[0], GetItemGuardAccessor)
        )
        attr_manager = item_manager.a
        self.assertTrue(
            isinstance(item_manager.get_accessors()[0], GetAttrGuardAccessor)
        )
        attr_manager.add_lambda_guard(
            lambda x: x == 4,
            "Expected value 4",
        )

        self.assertTrue(guard_manager.check(f_locals))

    # DefaultsSource
    def test_fn_defaults(self):
        def foo(a, b=1, *, c=2):
            return a + b + c

        guard_manager = RootGuardManager()
        guard_manager.add_lambda_guard(lambda x: callable(x), "Expected callable")

        # Guard on b = 1
        guard_manager.__defaults__[0].add_lambda_guard(
            lambda x: x == 1,
            "Expected int",
        )

        # Guard on c = 2
        guard_manager.__kwdefaults__.dict_get_item_manager("c").add_lambda_guard(
            lambda x: x == 2,
            "Expected int",
        )

        self.assertTrue(guard_manager.check(foo))
        foo.__defaults__ = (4,)
        self.assertFalse(guard_manager.check(foo))
        foo.__defaults__ = (1,)
        self.assertTrue(guard_manager.check(foo))
        foo.__kwdefaults__["c"] = 1
        self.assertFalse(guard_manager.check(foo))

    def test_iter(self):
        a = (1, 1, 3, 4, 5, 6)
        foo = iter(a)

        guard_manager = RootGuardManager()
        guard_manager.add_lambda_guard(
            lambda x: isinstance(x, type(iter(tuple()))),
            "Expected iterator",
        )

        def tuple_iterator_getitem(it, index):
            # Not straightforward to write the __reduce__ equivalent in C++
            _, (obj,), start = it.__reduce__()
            if len(obj) <= start + index:
                return None
            return obj[start + index]

        accessor = functools.partial(tuple_iterator_getitem, index=2)
        foo_mgr = guard_manager.lambda_manager(accessor)
        foo_mgr.add_lambda_guard(
            lambda x: x == 3,
            "Expected value 3",
        )

        # Check that we can use the same accessor
        foo_mgr = guard_manager.lambda_manager(accessor)
        foo_mgr.add_lambda_guard(
            lambda x: x - 1 == 2,
            "Expected value 3",
        )
        self.assertEqual(len(guard_manager.get_accessors()), 1)

        accessor = functools.partial(tuple_iterator_getitem, index=1)
        foo_mgr = guard_manager.lambda_manager(accessor)
        foo_mgr.add_lambda_guard(
            lambda x: x == 1,
            "Expected value 3",
        )

        self.assertEqual(len(guard_manager.get_accessors()), 2)

        self.assertTrue(guard_manager.check(foo))
        # check that iterator hasn't moved
        self.assertTrue(guard_manager.check(foo))
        self.assertTrue(guard_manager.check(iter((1, 1, 3))))
        self.assertFalse(guard_manager.check(iter((1, 1, 1, 1))))
        self.assertFalse(guard_manager.check(iter((1,))))
        self.assertFalse(guard_manager.check("foo"))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
