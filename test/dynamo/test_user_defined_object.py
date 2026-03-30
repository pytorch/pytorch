# Owner(s): ["module: dynamo"]

import dataclasses
import types
import unittest

import torch
import torch._dynamo.testing as dynamo_testing
from torch._dynamo.test_case import run_tests, TestCase


class SlotsOnly:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


class SlotsAndDict:
    __slots__ = ("x", "__dict__")

    def __init__(self, x):
        self.x = x


@dataclasses.dataclass(frozen=True, slots=True)
class FrozenSlots:
    x: int
    y: int


class SlotsAndSetattr:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value * 2)


class SlotsAndDictAndSetattr:
    __slots__ = ("x", "__dict__")

    def __init__(self, x):
        self.x = x

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value * 2)


class SlotsBase:
    __slots__ = ("x",)

    def __init__(self):
        self.x = 0


class SlotsDerived(SlotsBase):
    __slots__ = ("y",)

    def __init__(self):
        super().__init__()
        self.y = 0


class Plain:
    pass


class SlotsChildOfPlain(Plain):
    __slots__ = ("z",)

    def __init__(self):
        self.z = 0


class Slots:
    __slots__ = ("x",)


class SlotsShadowed(SlotsBase):
    x = 42  # class attribute shadows parent's slot descriptor


class SlotsAndProperty:
    __slots__ = ("_x",)

    def __init__(self, x):
        self._x = x

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value * 2


class TestSlotsAttrAssignment(TestCase):
    """Tests for attribute assignment on objects with __slots__."""

    def test_valid_slot_assignment(self):
        # Case 1: assign to a declared slot — should succeed
        def fn(t):
            obj = SlotsOnly(1, 2)
            obj.x = 99
            return t + obj.x

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_invalid_slot_assignment_raises(self):
        # Case 2: assign to an undeclared attr on a slotted object (no __dict__)
        # should raise AttributeError in eager; compiled raises an exception too
        def fn(t):
            obj = SlotsOnly(1, 2)
            obj.z = 99
            return t + obj.x

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.ones(1)
        self.assertRaises(AttributeError, fn, t)
        self.assertRaises(Exception, compiled_fn, t)

    def test_slots_with_dict_allows_arbitrary_attrs(self):
        # Case 3: __slots__ includes __dict__ — arbitrary attr assignment should work
        def fn(t):
            obj = SlotsAndDict(1)
            obj.extra = 42
            return t + obj.x + obj.extra

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_frozen_dataclass_with_slots_construction(self):
        # Case 4: frozen dataclass with slots uses object.__setattr__ in __init__
        # to bypass the frozen __setattr__. Dynamo must allow this for slot descriptors.
        def fn(t):
            obj = FrozenSlots(3, 4)
            return t + obj.x + obj.y

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_custom_setattr_with_slots(self):
        # Case 5: __slots__ + custom __setattr__ — the custom __setattr__ is traced
        def fn(t):
            obj = SlotsAndSetattr(1)
            obj.x = 10
            return t + obj.x

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_with_dict_valid_slot_assignment(self):
        # Case 6: __slots__ + __dict__: assigning to a declared slot still works
        def fn(t):
            obj = SlotsAndDict(1)
            obj.x = 99
            return t + obj.x

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_with_dict_undeclared_attr_goes_to_dict(self):
        # Case 7: __slots__ + __dict__: assigning to an undeclared attr goes to
        # __dict__ instead of raising AttributeError
        def fn(t):
            obj = SlotsAndDict(1)
            obj.z = 42
            return t + obj.z

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_custom_setattr_with_slots_and_dict(self):
        # Case 8: __slots__ + __dict__ + custom __setattr__ — custom __setattr__
        # is traced for both slot and non-slot attrs
        def fn(t):
            obj = SlotsAndDictAndSetattr(1)
            obj.x = 10
            obj.extra = 3
            return t + obj.x + obj.extra

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_inheritance_parent_and_child_slots(self):
        # Subclass adds its own slot on top of parent's slot — both accessible
        def fn(t):
            obj = SlotsDerived()
            obj.x = 1
            obj.y = 2
            return t + obj.x + obj.y

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_child_inherits_dict_from_no_slots_parent(self):
        # Subclass with __slots__ inheriting from a parent without __slots__
        # gets __dict__ from the parent, so arbitrary attrs are allowed
        def fn(t):
            obj = SlotsChildOfPlain()
            obj.z = 1
            obj.extra = 42
            return t + obj.z + obj.extra

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_object_setattr_bypasses_custom_setattr(self):
        # object.__setattr__ skips the custom __setattr__ and writes directly to slot
        def fn(t):
            obj = SlotsAndSetattr(1)
            object.__setattr__(obj, "x", 5)
            return t + obj.x

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_getattr_default_on_unset_slot(self):
        # getattr with a default on an unset slot returns the default
        def fn(t):
            obj = Slots()
            val = getattr(obj, "x", 99)
            return t + val

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slot_read_after_delete_raises(self):
        # Reading a slot after deletion raises AttributeError in both eager and compiled
        def fn(t):
            obj = Slots()
            obj.x = 1
            del obj.x
            return t + obj.x

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.ones(1)
        self.assertRaises(AttributeError, fn, t)
        self.assertRaises(Exception, compiled_fn, t)

    def test_slot_shadowed_by_class_attribute(self):
        # Class attribute in subclass shadows parent slot descriptor:
        # reads return the class attribute, writes raise AttributeError
        def fn(t):
            obj = SlotsShadowed()
            return t + obj.x  # returns class attr 42

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slot_assignment_with_object_as_argument(self):
        # Slotted object passed as argument (not created inside fn)
        def fn(t, obj):
            obj.x = 10
            return t + obj.x

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.ones(1)
        obj = Slots()
        self.assertEqual(fn(t.clone(), obj), compiled_fn(t.clone(), obj))

    def test_slot_mutation_materialized_on_argument(self):
        # Slot mutation on an object passed as argument must be visible after
        # the compiled function returns (side effect materialization)
        def fn(t, obj):
            obj.x = 10
            return t.sin()

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        obj = Slots()
        compiled_fn(torch.ones(1), obj)
        self.assertEqual(obj.x, 10)

    def test_slot_delete_materialized(self):
        # del on a slot inside a compiled fn must be visible after the call returns
        def fn(t, obj):
            del obj.x
            return t.sin()

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        obj = Slots()
        obj.x = 1
        compiled_fn(torch.ones(1), obj)
        self.assertFalse(hasattr(obj, "x"))

    def test_hasattr_on_slotted_object(self):
        # hasattr inside compiled code reflects actual slot state
        def fn(t):
            obj = Slots()
            before = hasattr(obj, "x")  # False — slot not set
            obj.x = 5
            after = hasattr(obj, "x")  # True — slot is now set
            return t + before + after

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_with_property_setter(self):
        # property setter is called instead of writing directly to the slot
        def fn(t):
            obj = SlotsAndProperty(1)
            obj.x = 5  # calls setter: _x = 5 * 2 = 10
            return t + obj.x  # calls getter: returns _x = 10

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slot_assignment_no_recompile_same_type(self):
        # Calling compiled fn repeatedly with the same slotted object type
        # must not trigger recompilation
        cnts = dynamo_testing.CompileCounter()

        def fn(t, obj):
            obj.x = 10
            return t + obj.x

        compiled_fn = torch.compile(fn, backend=cnts)
        t = torch.ones(1)
        compiled_fn(t, Slots())
        compiled_fn(t, Slots())
        compiled_fn(t, Slots())
        self.assertEqual(cnts.frame_count, 1)

    def test_slot_assignment_recompiles_on_type_change(self):
        # Compiled fn sees slot assigned to int first, then float — guards recompile
        cnts = dynamo_testing.CompileCounter()

        def fn(t, a, obj):
            obj.x = a
            return t + obj.x

        compiled_fn = torch.compile(fn, backend=cnts)
        t = torch.ones(1)

        compiled_fn(t, 1, Slots())
        compiled_fn(t, 1, Slots())
        self.assertEqual(cnts.frame_count, 1)  # same type, no recompile

        x = t.clone()
        res = compiled_fn(x, 1.0, Slots())
        self.assertEqual(cnts.frame_count, 2)  # float instead of int — recompile
        self.assertEqual(res, fn(x, 1.0, Slots()))


class WithGetattribute:
    # __slots__ = ("x", "_side_effects")

    def __init__(self, x):
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "_side_effects", set())

    def __getattribute__(self, name):
        effects = object.__getattribute__(self, "_side_effects")
        effects.add(name)
        return object.__getattribute__(self, name)


class TestSlotsFromCPython(TestCase):
    """Slot tests extracted from CPython's test_descr.py::test_slots."""

    def setUp(self):
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    def test_slots_empty(self):
        class C:
            __slots__ = []

        def fn(t):
            x = C()
            self.assertFalse(hasattr(x, "__dict__"))
            self.assertFalse(hasattr(x, "foo"))
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_single(self):
        class C:
            __slots__ = ["a"]

        def fn(t):
            x = C()
            self.assertFalse(hasattr(x, "__dict__"))
            self.assertFalse(hasattr(x, "a"))
            x.a = 1
            self.assertEqual(x.a, 1)
            x.a = None
            self.assertEqual(x.a, None)
            del x.a
            self.assertFalse(hasattr(x, "a"))
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_multiple(self):
        class C:
            __slots__ = ["a", "b", "c"]

        def fn(t):
            x = C()
            self.assertFalse(hasattr(x, "__dict__"))
            self.assertFalse(hasattr(x, "a"))
            self.assertFalse(hasattr(x, "b"))
            self.assertFalse(hasattr(x, "c"))
            x.a = 1
            x.b = 2
            x.c = 3
            self.assertEqual(x.a, 1)
            self.assertEqual(x.b, 2)
            self.assertEqual(x.c, 3)
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_name_mangling(self):
        class C:
            __slots__ = ["__a"]

            def __init__(self, value):
                self.__a = value

            def get(self):
                return self.__a

        def fn(t):
            x = C(5)
            self.assertFalse(hasattr(x, "__dict__"))
            self.assertFalse(hasattr(x, "__a"))
            self.assertEqual(x.get(), 5)
            try:
                x.__a = 6
            except AttributeError:
                pass
            else:
                self.fail("Double underscored names not mangled")
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_string_not_expanded(self):
        # A single string is not expanded as a sequence
        class C:
            __slots__ = "abc"  # noqa: PLC0205

        def fn(t):
            c = C()
            c.abc = 5
            self.assertEqual(c.abc, 5)
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_tuple(self):
        slots = ("foo", "bar")

        class C:
            __slots__ = slots

        def fn(t):
            x = C()
            x.foo = 5
            self.assertEqual(x.foo, 5)
            self.assertIs(type(slots[0]), str)
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_get_unset_raises(self):
        class X:
            __slots__ = "a"  # noqa: PLC0205

        def fn(t):
            with self.assertRaises(AttributeError):
                X().a
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_str_subclass(self):
        # gh-98783: string subclass in __slots__
        class SubStr(str):  # noqa: SLOT000
            pass

        class X:
            __slots__ = (SubStr("x"),)

        def fn(t):
            X().x = 1
            with self.assertRaises(AttributeError):
                X().a
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_special_dict(self):
        # __dict__ in __slots__ enables arbitrary attr assignment
        class D:
            __slots__ = ["__dict__"]

        def fn(t):
            a = D()
            self.assertTrue(hasattr(a, "__dict__"))
            self.assertFalse(hasattr(a, "__weakref__"))
            a.foo = 42
            self.assertEqual(a.__dict__, {"foo": 42})
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_special_weakref(self):
        # __weakref__ in __slots__ — no __dict__, arbitrary attr raises
        class W:
            __slots__ = ["__weakref__"]

        def fn(t):
            a = W()
            self.assertTrue(hasattr(a, "__weakref__"))
            self.assertFalse(hasattr(a, "__dict__"))
            with self.assertRaises(AttributeError):
                a.foo = 42
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_special_inherit_dict_weakref(self):
        # Inheriting from both __dict__ and __weakref__ slot classes
        class D:
            __slots__ = ["__dict__"]

        class W:
            __slots__ = ["__weakref__"]

        class C1(W, D):
            __slots__ = []

        def fn(t):
            a = C1()
            self.assertTrue(hasattr(a, "__dict__"))
            self.assertTrue(hasattr(a, "__weakref__"))
            a.foo = 42
            self.assertEqual(a.__dict__, {"foo": 42})
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    @unittest.expectedFailure
    def test_slots_special2_classcell(self):
        # Testing __classcell__ in __slots__
        class Meta(type):
            def __new__(metacls, name, bases, namespace, attr):
                self.assertIn(attr, namespace)
                return super().__new__(metacls, name, bases, namespace)

        class C1:
            def __init__(self):
                self.b = 42

        class C2(C1, metaclass=Meta, attr="__classcell__"):
            __slots__ = ["__classcell__"]

            def __init__(self):
                super().__init__()

        def fn(t):
            self.assertIsInstance(
                C2.__dict__["__classcell__"], types.MemberDescriptorType
            )
            c = C2()
            self.assertEqual(c.b, 42)
            self.assertFalse(hasattr(c, "__classcell__"))
            c.__classcell__ = 42
            self.assertEqual(c.__classcell__, 42)
            with self.assertRaises(TypeError):

                class C3:
                    __classcell__ = 42
                    __slots__ = ["__classcell__"]

            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_multiple_inheritance(self):
        # SF bug 575229: multiple inheritance w/ slots dumps core
        class A:
            __slots__ = ()

        class B:
            pass

        class C(A, B):
            __slots__ = ()

        def fn(t):
            self.assertTrue(hasattr(C, "__dict__"))
            self.assertTrue(hasattr(C, "__weakref__"))
            C().x = 2
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)


class TestUserDefinedClassDict(TestCase):
    def test_class_dict_read(self):
        class MyClass:
            x = 3

        def fn(t):
            t = t + MyClass.__dict__["x"]
            t = t + MyClass.__dict__.get("x", 0)
            t = t + MyClass.__dict__.get("z", 99)
            t = t + (1 if "x" in MyClass.__dict__ else 0)
            t = t + (1 if "z" in MyClass.__dict__ else 0)
            return t

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_class_dict_via_arg(self):
        class MyClass:
            x = 7

        def fn(t, cls):
            return t + cls.__dict__.get("x", 0)

        cnt = dynamo_testing.CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        result = compiled(torch.tensor([0.0]), MyClass)
        self.assertEqual(result, torch.tensor([7.0]))

    def test_class_dict_mutation_recompiles(self):
        # Mutating a class attribute between calls should trigger recompilation,
        # and the compiled function should see the updated value.
        class MyClass:
            x = 1

        def fn(t):
            return t + MyClass.__dict__["x"]

        cnt = dynamo_testing.CompileCounter()
        compiled = torch.compile(fn, backend=cnt)

        result1 = compiled(torch.tensor([0.0]))
        self.assertEqual(result1, torch.tensor([1.0]))
        self.assertEqual(cnt.frame_count, 1)

        MyClass.x = 10
        result2 = compiled(torch.tensor([0.0]))
        self.assertEqual(result2, torch.tensor([10.0]))
        # Should have recompiled due to guard failure
        self.assertEqual(cnt.frame_count, 2)

    def test_class_dict_add_key_recompiles(self):
        # Adding a new attribute to the class should trigger recompilation
        # when the compiled code checks for key presence.
        class MyClass:
            x = 1

        def fn(t):
            return t + (1 if "y" in MyClass.__dict__ else 0)

        cnt = dynamo_testing.CompileCounter()
        compiled = torch.compile(fn, backend=cnt)

        result1 = compiled(torch.tensor([0.0]))
        self.assertEqual(result1, torch.tensor([0.0]))
        self.assertEqual(cnt.frame_count, 1)

        MyClass.y = 99
        result2 = compiled(torch.tensor([0.0]))
        self.assertEqual(result2, torch.tensor([1.0]))
        # Should have recompiled
        self.assertEqual(cnt.frame_count, 2)

    def test_class_dict_delete_key_recompiles(self):
        # Deleting a class attribute should trigger recompilation.
        class MyClass:
            x = 5
            y = 10

        def fn(t):
            return t + MyClass.__dict__.get("y", 0)

        cnt = dynamo_testing.CompileCounter()
        compiled = torch.compile(fn, backend=cnt)

        result1 = compiled(torch.tensor([0.0]))
        self.assertEqual(result1, torch.tensor([10.0]))
        self.assertEqual(cnt.frame_count, 1)

        del MyClass.y
        result2 = compiled(torch.tensor([0.0]))
        self.assertEqual(result2, torch.tensor([0.0]))
        # Should have recompiled
        self.assertEqual(cnt.frame_count, 2)


if __name__ == "__main__":
    run_tests()
