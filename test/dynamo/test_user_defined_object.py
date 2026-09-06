# Owner(s): ["module: dynamo"]

import copy
import dataclasses
import sys
import types
import unittest

import torch
import torch._dynamo.testing as dynamo_testing
from torch._dynamo.exc import Unsupported
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import make_dynamo_test


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

    def test_direct_dict_write_does_not_shadow_data_descriptor(self):
        class Foo:
            @property
            def x(self):
                return 10

        def fn(t, obj):
            obj.__dict__["x"] = 99
            return t + obj.x + obj.__dict__["x"]

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.ones(1)
        self.assertEqual(fn(t, Foo()), compiled_fn(t, Foo()))

    def test_readonly_property_assignment_raises(self):
        class Foo:
            @property
            def x(self):
                return 10

        def fn(obj):
            obj.x = 99
            return obj.x

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertRaises(AttributeError, fn, Foo())
        self.assertRaisesRegex(Exception, "has no setter", compiled_fn, Foo())

    def test_delattr_instance_dict_exposes_non_data_descriptor(self):
        class Descriptor:
            def __get__(self, obj, owner):
                return 5

        class Foo:
            x = Descriptor()

            def __init__(self):
                self.x = 7

        def fn(t, obj):
            del obj.x
            return t + obj.x

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.ones(1)
        compiled_obj = Foo()
        self.assertEqual(fn(t, Foo()), compiled_fn(t, compiled_obj))
        self.assertNotIn("x", compiled_obj.__dict__)

    def test_property_deleter(self):
        class Foo:
            def __init__(self):
                self.deleted = False
                self._x = 4

            @property
            def x(self):
                return self._x

            @x.deleter
            def x(self):
                self.deleted = True
                self._x = 0

        def fn(t, obj):
            del obj.x
            return t + obj.x + obj.deleted

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.ones(1)
        compiled_obj = Foo()
        self.assertEqual(fn(t, Foo()), compiled_fn(t, compiled_obj))
        self.assertTrue(compiled_obj.deleted)
        self.assertEqual(compiled_obj._x, 0)

    def test_property_without_deleter_raises(self):
        class Foo:
            @property
            def x(self):
                return 10

        def fn(obj):
            del obj.x
            return obj.x

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertRaises(AttributeError, fn, Foo())
        self.assertRaisesRegex(Exception, "has no deleter", compiled_fn, Foo())

    def test_slot_and_dict_mutation_same_object(self):
        class Foo:
            __slots__ = ("x", "__dict__")

        def fn(t, obj):
            obj.x = 2
            obj.__dict__["y"] = 3
            return t + obj.x + obj.__dict__["y"]

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.ones(1)
        compiled_obj = Foo()
        self.assertEqual(fn(t, Foo()), compiled_fn(t, compiled_obj))
        self.assertEqual(compiled_obj.x, 2)
        self.assertEqual(compiled_obj.__dict__["y"], 3)

    def test_dunder_dict_assignment_updates_attribute_lookup(self):
        class Foo:
            __slots__ = ("__dict__",)

        def fn(t):
            obj = Foo()
            obj.__dict__ = {"y": 2}
            return t + obj.y + obj.__dict__["y"]

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_custom_descriptor_shadows_base_slot(self):
        class Descriptor:
            def __get__(self, obj, owner):
                if obj is None:
                    return self
                return obj.y * 2

            def __set__(self, obj, value):
                obj.y = value + 1

        class Base:
            __slots__ = ("x", "__dict__")

        class Foo(Base):
            x = Descriptor()

        def fn(t, obj):
            obj.x = 4
            return t + obj.x + obj.y

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.ones(1)
        compiled_obj = Foo()
        self.assertEqual(fn(t, Foo()), compiled_fn(t, compiled_obj))
        self.assertEqual(compiled_obj.y, 5)

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


class TestClassSetattr(TestCase):
    def test_setattr_class_attribute(self):
        class MyModule:
            x = 10

        def fn():
            MyModule.x = 20
            return MyModule.x

        opt_fn = torch.compile(fn, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        result = opt_fn()
        self.assertEqual(result, 20)

        MyModule.x = 10


# ---------------------------------------------------------------------------
# __setitem__ on user-defined classes / metaclasses
# ---------------------------------------------------------------------------


# Metaclasses kept at module level — Dynamo's traced LOAD_BUILD_CLASS does
# not currently propagate the `metaclass=` kwarg.


class _SetitemMetaBasic(type):
    def __setitem__(cls, key, value):
        cls._store[key] = value

    def __getitem__(cls, key):
        return cls._store[key]


class _ClassWithBasicMeta(metaclass=_SetitemMetaBasic):
    _store: dict = {}


class _SetitemMetaPerClass(type):
    def __setitem__(cls, key, value):
        cls.entries[key] = value

    def __getitem__(cls, key):
        return cls.entries[key]


class _PerClassEntries(metaclass=_SetitemMetaPerClass):
    entries: dict = {}


class _SetitemMetaValidating(type):
    def __setitem__(cls, key, value):
        if not isinstance(key, str):
            raise TypeError("class registry expects string keys")
        cls.registry[key] = value


class _ValidatingClass(metaclass=_SetitemMetaValidating):
    registry: dict = {}


class _SetitemDelitemMeta(type):
    def __setitem__(cls, key, value):
        cls._store[key] = value

    def __getitem__(cls, key):
        return cls._store[key]

    def __delitem__(cls, key):
        del cls._store[key]


class _DelClassMeta(metaclass=_SetitemDelitemMeta):
    _store: dict = {}


class TestUserDefinedSetitem(TestCase):
    """__setitem__ on user-defined classes (UDOV) and metaclasses (UDCV).

    enable_trace_load_build_class lets us define helper classes inside the
    test body — keeps the helper next to the assertion that exercises it.
    """

    def setUp(self):
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        self._b_prev = torch._dynamo.config.enable_trace_load_build_class
        torch._dynamo.config.enable_trace_unittest = True
        torch._dynamo.config.enable_trace_load_build_class = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev
        torch._dynamo.config.enable_trace_load_build_class = self._b_prev

    # -- instance __setitem__ --

    @make_dynamo_test
    def test_validating_ok(self):
        class V:
            def __init__(self):
                self.data = {}

            def __getitem__(self, key):
                return self.data[key]

            def __setitem__(self, key, value):
                if not isinstance(key, str):
                    raise TypeError("only string keys allowed")
                if value < 0:
                    raise ValueError("negative values forbidden")
                self.data[key] = value

        obj = V()
        obj["a"] = 5
        self.assertEqual(obj["a"], 5)
        with self.assertRaises(TypeError):
            obj[1] = 5
        with self.assertRaises(ValueError):
            obj["a"] = -1

    @make_dynamo_test
    def test_transforming_value(self):
        class T:
            def __init__(self):
                self.data = {}

            def __getitem__(self, key):
                return self.data[key]

            def __setitem__(self, key, value):
                self.data[key] = value * 2

        obj = T()
        obj["a"] = 5
        self.assertEqual(obj["a"], 10)

    @make_dynamo_test
    def test_inherited_method(self):
        class Base:
            def __init__(self):
                self.data = {}

            def __getitem__(self, key):
                return self.data[key]

            def __setitem__(self, key, value):
                self.data[key] = value

        class Derived(Base):
            pass

        obj = Derived()
        obj["a"] = 5
        self.assertEqual(obj["a"], 5)

    @make_dynamo_test
    def test_overriding_method(self):
        class Base:
            def __init__(self):
                self.data = {}

            def __getitem__(self, key):
                return self.data[key]

            def __setitem__(self, key, value):
                self.data[key] = value

        class Override(Base):
            def __setitem__(self, key, value):
                self.data[key] = value + 100

        obj = Override()
        obj["a"] = 5
        self.assertEqual(obj["a"], 105)

    @make_dynamo_test
    def test_return_value_ignored(self):
        class R:
            def __init__(self):
                self.data = {}

            def __setitem__(self, key, value):
                self.data[key] = value
                return "ignored"

            def __getitem__(self, key):
                return self.data[key]

        obj = R()
        obj["a"] = 5
        self.assertEqual(obj["a"], 5)

    @make_dynamo_test
    def test_side_effects_in_method(self):
        class S:
            def __init__(self):
                self.data = {}
                self.last_key = None
                self.last_value = None
                self.call_count = 0

            def __getitem__(self, key):
                return self.data[key]

            def __setitem__(self, key, value):
                self.last_key = key
                self.last_value = value
                self.call_count += 1
                self.data[key] = value

        obj = S()
        obj["a"] = 1
        obj["b"] = 2
        self.assertEqual(obj.call_count, 2)
        self.assertEqual(obj.last_key, "b")
        self.assertEqual(obj.last_value, 2)
        self.assertEqual(obj["a"], 1)
        self.assertEqual(obj["b"], 2)

    @make_dynamo_test
    def test_explicit_method_call(self):
        class B:
            def __init__(self):
                self.data = {}

            def __getitem__(self, key):
                return self.data[key]

            def __setitem__(self, key, value):
                self.data[key] = value

        obj = B()
        obj.__setitem__("a", 5)
        self.assertEqual(obj["a"], 5)

    @make_dynamo_test
    def test_multiple_keys(self):
        class B:
            def __init__(self):
                self.data = {}

            def __getitem__(self, key):
                return self.data[key]

            def __setitem__(self, key, value):
                self.data[key] = value

        obj = B()
        for i in range(5):
            obj[i] = i * 10
        for i in range(5):
            self.assertEqual(obj[i], i * 10)

    @make_dynamo_test
    def test_no_setitem_raises_typeerror(self):
        class N:
            def __init__(self, data):
                self.data = data

            def __getitem__(self, key):
                return self.data[key]

        obj = N([1, 2, 3])
        with self.assertRaises(TypeError):
            obj[0] = 100

    # -- metaclass __setitem__: Cls[k] = v --

    @make_dynamo_test
    def test_metaclass_basic(self):
        _ClassWithBasicMeta["a"] = 1
        self.assertEqual(_ClassWithBasicMeta["a"], 1)

    @make_dynamo_test
    def test_metaclass_multiple(self):
        _PerClassEntries["x"] = 10
        _PerClassEntries["y"] = 20
        self.assertEqual(_PerClassEntries["x"], 10)
        self.assertEqual(_PerClassEntries["y"], 20)

    @make_dynamo_test
    def test_metaclass_validating(self):
        _ValidatingClass["k"] = 99
        self.assertEqual(_ValidatingClass.registry["k"], 99)
        with self.assertRaises(TypeError):
            _ValidatingClass[123] = 99

    # -- __delitem__ on user-defined classes --

    @make_dynamo_test
    def test_delitem_basic(self):
        class D:
            def __init__(self):
                self.data = {"a": 1, "b": 2}

            def __getitem__(self, key):
                return self.data[key]

            def __delitem__(self, key):
                del self.data[key]

        obj = D()
        del obj["a"]
        self.assertNotIn("a", obj.data)
        self.assertEqual(obj.data, {"b": 2})

    @make_dynamo_test
    def test_delitem_tracks(self):
        class D:
            def __init__(self):
                self.data = {"a": 1, "b": 2, "c": 3}
                self.deleted = []

            def __delitem__(self, key):
                self.deleted.append(key)
                del self.data[key]

        obj = D()
        del obj["a"]
        del obj["c"]
        self.assertEqual(obj.deleted, ["a", "c"])
        self.assertEqual(obj.data, {"b": 2})

    @make_dynamo_test
    def test_delitem_validating(self):
        class D:
            def __init__(self):
                self.data = {1: "a"}

            def __delitem__(self, key):
                if not isinstance(key, int):
                    raise TypeError("only int keys")
                del self.data[key]

        obj = D()
        del obj[1]
        self.assertEqual(obj.data, {})

        obj2 = D()
        with self.assertRaises(TypeError):
            del obj2["nope"]

    @make_dynamo_test
    def test_delitem_explicit_method_call(self):
        class D:
            def __init__(self):
                self.data = {"a": 1}

            def __delitem__(self, key):
                del self.data[key]

        obj = D()
        obj.__delitem__("a")
        self.assertEqual(obj.data, {})

    @make_dynamo_test
    def test_delitem_no_method_typeerror(self):
        class D:
            def __init__(self):
                self.data = [1, 2, 3]

            def __getitem__(self, key):
                return self.data[key]

        obj = D()
        with self.assertRaises(TypeError):
            del obj[0]

    # -- metaclass __delitem__: del Cls[k] --

    @make_dynamo_test
    def test_metaclass_delitem(self):
        _DelClassMeta["x"] = 1
        _DelClassMeta["y"] = 2
        del _DelClassMeta["x"]
        self.assertNotIn("x", _DelClassMeta._store)
        self.assertEqual(_DelClassMeta["y"], 2)


class TestObjectConstruction(TestCase):
    @make_dynamo_test
    def test_object_call_identity(self):
        a = object()
        b = object()
        self.assertEqual(a is a, True)
        self.assertEqual(a is b, False)
        self.assertEqual(type(a) is object, True)

    @make_dynamo_test
    def test_object_call_as_sentinel(self):
        sentinel = object()
        self.assertEqual(sentinel == 1, False)
        self.assertEqual(sentinel == sentinel, True)

    def test_object_call_escapes_graph_breaks(self):
        # A bare object() that escapes the compiled region is opaque and
        # sourceless, so reconstruction graph-breaks (runs in eager) rather
        # than failing; the returned value is a real object instance.
        cnt = dynamo_testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            return x + 1, object()

        _, s = fn(torch.randn(3))
        self.assertIs(type(s), object)
        self.assertEqual(cnt.frame_count, 0)


class _Namespace(types.SimpleNamespace):
    pass


class _OverridingNamespace(types.SimpleNamespace):
    def __repr__(self) -> str:
        return "overridden"

    def __eq__(self, other: object) -> bool:
        return True


class _InitNamespace(types.SimpleNamespace):
    def __init__(self, a: int = 1) -> None:
        super().__init__(a=a, b=a * 2)


class _RequiredArgNamespace(types.SimpleNamespace):
    def __init__(self, a: int) -> None:
        super().__init__(a=a)


@torch._dynamo.config.patch(enable_trace_unittest=True)
class TestSimpleNamespace(TestCase):
    """types.SimpleNamespace, ported from CPython's SimpleNamespaceTests."""

    @make_dynamo_test
    def test_constructor(self):
        def check(ns, expected):
            self.assertEqual(len(ns.__dict__), len(expected))
            self.assertEqual(vars(ns), expected)
            self.assertEqual(list(vars(ns).items()), list(expected.items()))

        check(types.SimpleNamespace(), {})
        check(types.SimpleNamespace(x=1, y=2), {"x": 1, "y": 2})

    @make_dynamo_test
    def test_constructor_positional(self):
        # namespace_init grew its optional positional argument in 3.13.
        if sys.version_info < (3, 13):
            with self.assertRaises(TypeError):
                types.SimpleNamespace({"x": 1})
            return

        def check(ns, expected):
            self.assertEqual(len(ns.__dict__), len(expected))
            self.assertEqual(vars(ns), expected)
            self.assertEqual(list(vars(ns).items()), list(expected.items()))

        check(
            types.SimpleNamespace({"x": 1, "y": 2}, x=4, z=3), {"x": 4, "y": 2, "z": 3}
        )
        check(types.SimpleNamespace([["x", 1], ["y", 2]]), {"x": 1, "y": 2})
        check(types.SimpleNamespace([], x=4), {"x": 4})

    @make_dynamo_test
    def test_constructor_errors(self):
        with self.assertRaises(TypeError):
            types.SimpleNamespace([], [])  # too many positional arguments
        with self.assertRaises(TypeError):
            types.SimpleNamespace(1)  # not a mapping or iterable
        with self.assertRaises(TypeError):
            types.SimpleNamespace([1])  # non-iterable element
        # Below 3.13 no positional argument is accepted at all, so the pair
        # check never runs and the argument count TypeError comes out instead.
        pair_error = ValueError if sys.version_info >= (3, 13) else TypeError
        with self.assertRaises(pair_error):
            types.SimpleNamespace([["x"]])  # not a pair
        with self.assertRaises(TypeError):
            types.SimpleNamespace({1: 2})  # non-string key
        with self.assertRaises(TypeError):
            types.SimpleNamespace(**{1: 2})

    @make_dynamo_test
    def test_attrget(self):
        ns = types.SimpleNamespace(x=1, y=2)
        self.assertEqual(ns.x, 1)
        self.assertEqual(ns.y, 2)
        with self.assertRaises(AttributeError):
            ns.z

    @make_dynamo_test
    def test_attrset(self):
        ns = types.SimpleNamespace(x=1)
        ns.y = "ham"
        ns.z = None
        self.assertEqual(ns.__dict__, dict(x=1, y="ham", z=None))

    @make_dynamo_test
    def test_attrdel(self):
        ns = types.SimpleNamespace(x=1, y=2, w=3)
        with self.assertRaises(AttributeError):
            del ns.spam
        del ns.y
        self.assertEqual(vars(ns), dict(x=1, w=3))
        ns.y = "spam"
        self.assertEqual(vars(ns), dict(x=1, w=3, y="spam"))

    @make_dynamo_test
    def test_repr(self):
        ns = types.SimpleNamespace(x=1, y=2, w=3)
        self.assertEqual(repr(ns), "namespace(x=1, y=2, w=3)")
        self.assertEqual(repr(types.SimpleNamespace()), "namespace()")
        self.assertEqual(repr(types.SimpleNamespace(x="spam")), "namespace(x='spam')")

    @make_dynamo_test
    def test_repr_subclass(self):
        self.assertEqual(repr(_Namespace(a=1)), "_Namespace(a=1)")

    @make_dynamo_test
    def test_recursive_repr(self):
        ns = types.SimpleNamespace(c="cookie")
        ns.spam = ns
        self.assertEqual(repr(ns), "namespace(c='cookie', spam=namespace(...))")
        sub = _Namespace()
        sub.spam = sub
        self.assertEqual(repr(sub), "_Namespace(spam=_Namespace(...))")

    @make_dynamo_test
    def test_equal(self):
        ns = types.SimpleNamespace()
        ns.x = 1
        self.assertEqual(types.SimpleNamespace(), types.SimpleNamespace())
        self.assertEqual(types.SimpleNamespace(x=1), ns)
        self.assertFalse(ns == types.SimpleNamespace())
        # __eq__ returns NotImplemented against a non-namespace
        self.assertFalse(types.SimpleNamespace() == 1)
        self.assertTrue(types.SimpleNamespace() != 1)

    @make_dynamo_test
    def test_ordering_unsupported(self):
        with self.assertRaises(TypeError):
            types.SimpleNamespace(x=1) < types.SimpleNamespace(x=2)  # noqa: B015

    @make_dynamo_test
    def test_as_dict(self):
        ns = types.SimpleNamespace(spam="spamspamspam")
        with self.assertRaises(TypeError):
            len(ns)
        with self.assertRaises(TypeError):
            iter(ns)
        with self.assertRaises(TypeError):
            "spam" in ns  # noqa: B015
        with self.assertRaises(TypeError):
            ns["spam"]

    @make_dynamo_test
    def test_nested(self):
        ns1 = types.SimpleNamespace(a=1, b=2)
        ns2 = types.SimpleNamespace(x=ns1)
        self.assertEqual(vars(ns1), dict(a=1, b=2))
        self.assertEqual(ns2.x.a, 1)
        self.assertEqual(ns2.x, ns1)

    @make_dynamo_test
    def test_recursive(self):
        ns1 = types.SimpleNamespace(c="cookie")
        ns2 = types.SimpleNamespace()
        ns3 = types.SimpleNamespace(x=1)
        ns1.spam = ns1
        ns2.spam = ns3
        ns3.spam = ns2
        self.assertEqual(ns1.spam, ns1)
        self.assertEqual(ns1.spam.spam, ns1)
        self.assertEqual(ns2.spam.spam, ns2)

    @make_dynamo_test
    def test_subclass(self):
        spam = _Namespace(ham=8, eggs=9)
        self.assertIs(type(spam), _Namespace)
        self.assertEqual(vars(spam), {"ham": 8, "eggs": 9})

    @make_dynamo_test
    def test_subclass_overrides_c_slots(self):
        # A Python __repr__/__eq__ on the subclass replaces the C slot.
        self.assertEqual(repr(_OverridingNamespace(a=1)), "overridden")
        self.assertTrue(_OverridingNamespace(a=1) == 1)

    @unittest.skipIf(sys.version_info < (3, 13), "copy.replace added in 3.13")
    @make_dynamo_test
    def test_replace(self):
        ns = types.SimpleNamespace(x=11, y=22)
        ns2 = copy.replace(ns)
        self.assertEqual(ns2, ns)
        self.assertIsNot(ns2, ns)
        self.assertIs(type(ns2), types.SimpleNamespace)
        ns2.x = 3
        self.assertEqual(ns.x, 11)
        self.assertEqual(vars(copy.replace(ns, x=1)), {"x": 1, "y": 22})
        self.assertEqual(vars(copy.replace(ns, x=1, y=2)), {"x": 1, "y": 2})

    @unittest.skipIf(sys.version_info < (3, 13), "copy.replace added in 3.13")
    @make_dynamo_test
    def test_replace_subclass(self):
        spam2 = copy.replace(_Namespace(ham=8, eggs=9), ham=5)
        self.assertIs(type(spam2), _Namespace)
        self.assertEqual(vars(spam2), {"ham": 5, "eggs": 9})

    @make_dynamo_test
    def test_subclass_init_via_super(self):
        # namespace_init is a tp_init slot wrapper, reached through super().
        self.assertEqual(vars(_InitNamespace(2)), {"a": 2, "b": 4})

    @unittest.skipIf(sys.version_info < (3, 13), "copy.replace added in 3.13")
    @make_dynamo_test
    def test_replace_subclass_with_init(self):
        got = copy.replace(_InitNamespace(1), a=5)
        self.assertIs(type(got), _InitNamespace)
        self.assertEqual(vars(got), {"a": 5, "b": 2})

    @unittest.skipIf(sys.version_info < (3, 13), "copy.replace added in 3.13")
    @make_dynamo_test
    def test_replace_keeps_constructor_only_attrs(self):
        # namespace_replace calls type(self)(), so the copy starts from what the
        # constructor produced; PyDict_Update only overlays what self carries.
        ns = _InitNamespace(1)
        del ns.b
        self.assertEqual(vars(copy.replace(ns, a=5)), {"a": 5, "b": 2})

    @unittest.skipIf(sys.version_info < (3, 13), "copy.replace added in 3.13")
    @make_dynamo_test
    def test_replace_subclass_init_requires_arg(self):
        # type(self)() is what raises here, so skipping __init__ would swallow it.
        with self.assertRaises(TypeError):
            copy.replace(_RequiredArgNamespace(1), a=5)

    @make_dynamo_test
    def test_constructor_arity_message(self):
        if sys.version_info < (3, 13):
            with self.assertRaisesRegex(TypeError, "no positional arguments expected"):
                types.SimpleNamespace({}, {})
            return
        # PyArg_UnpackTuple gets its funcname from _PyType_Name(Py_TYPE(ns)),
        # so a subclass reports its own name.
        too_many = "expected at most 1 argument, got 2"
        with self.assertRaisesRegex(TypeError, f"SimpleNamespace {too_many}"):
            types.SimpleNamespace({}, {})
        with self.assertRaisesRegex(TypeError, f"_Namespace {too_many}"):
            _Namespace({}, {})

    @unittest.skipIf(sys.version_info < (3, 13), "positional argument added in 3.13")
    def test_str_subclass_key_graph_breaks(self):
        # PyUnicode_Check takes a str subclass, but Dynamo models the instance
        # dict with exact-str names, so this graph-breaks instead of tracing.
        class MyStr(str):
            __slots__ = ()

        def fn(x):
            return len(vars(types.SimpleNamespace({MyStr("a"): 1}))), x + 1

        x = torch.randn(3)
        with self.assertRaisesRegex(Unsupported, "str subclass key in SimpleNamespace"):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(torch.compile(fn, backend="eager")(x)[0], fn(x)[0])

    @unittest.skipIf(sys.version_info < (3, 13), "positional argument added in 3.13")
    def test_non_constant_key_graph_breaks(self):
        # Formatting a tensor produces a StringFormatVariable, which is str-typed
        # but has no constant value to use as an attribute name.
        def fn(x):
            return len(vars(types.SimpleNamespace({f"a{x}": 1}))), x + 1

        with self.assertRaisesRegex(Unsupported, "non-constant key in SimpleNamespace"):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3))

    def test_holds_tensors_in_one_graph(self):
        def fn(x):
            ns = types.SimpleNamespace(a=x + 1)
            ns.b = ns.a * 2
            del ns.a
            return ns

        cnt = dynamo_testing.CompileCounter()
        x = torch.randn(3)
        got = torch.compile(fn, backend=cnt, fullgraph=True)(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertIs(type(got), types.SimpleNamespace)
        self.assertEqual(vars(got), vars(fn(x)))

    def test_argument_from_outside(self):
        # A namespace built before the compiled region reaches Dynamo through
        # VariableBuilder rather than SideEffects, so it needs the same model.
        def fn(ns, x):
            before = repr(ns)
            ns.total = ns.scale * x
            return before, ns.total, ns == types.SimpleNamespace(name="cfg", scale=2)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(3)
        ns_eager = types.SimpleNamespace(name="cfg", scale=2)
        ns_compiled = types.SimpleNamespace(name="cfg", scale=2)
        self.assertEqual(fn(ns_eager, x), opt_fn(ns_compiled, x))
        # The write to ns.total has to replay onto the caller's object.
        self.assertEqual(vars(ns_eager), vars(ns_compiled))

    @unittest.skipIf(sys.version_info < (3, 13), "copy.replace added in 3.13")
    def test_replace_from_outside(self):
        def fn(ns, x):
            ns.total = ns.scale * x
            return ns == copy.replace(ns, total=ns.total)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(3)
        ns_eager = types.SimpleNamespace(name="cfg", scale=2)
        ns_compiled = types.SimpleNamespace(name="cfg", scale=2)
        self.assertEqual(fn(ns_eager, x), opt_fn(ns_compiled, x))
        self.assertEqual(vars(ns_eager), vars(ns_compiled))


if __name__ == "__main__":
    run_tests()
