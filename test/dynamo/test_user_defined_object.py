# Owner(s): ["module: dynamo"]

import dataclasses
import types
import unittest

import torch
import torch._dynamo.testing as dynamo_testing
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


class TestUserDefinedObjectConstruction(TestCase):
    def test_instantiate_class_with_custom_getattribute(self):
        class Foo:
            def __init__(self, a):
                self.a = a

            def __getattribute__(self, name):
                return super().__getattribute__(name)

        def fn(t):
            _ = Foo(3)
            return t.sin()

        x = torch.randn(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_instantiate_class_with_custom_getattribute_and_attr_read(self):
        class Foo:
            def __init__(self, a):
                self.a = a

            def __getattribute__(self, name):
                return super().__getattribute__(name)

        def fn(t):
            f = Foo(3)
            return t.sin() + f.a

        x = torch.randn(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_instantiate_class_ignores_instance_init_attr(self):
        class Foo:
            def __new__(cls, *args):
                obj = object.__new__(cls)

                def wrong_init(*args, **kwargs):
                    raise AssertionError("should call class __init__")

                obj.__init__ = wrong_init
                return obj

            def __init__(self, a):
                self.a = a

        def fn(t):
            f = Foo(3)
            return t.sin() + f.a

        x = torch.randn(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_explicit_class_object_init_with_extra_arg_not_noop(self):
        class Foo:
            pass

        def fn(t):
            f = Foo()
            Foo.__init__(f, 1)
            return t.sin()

        x = torch.randn(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            opt_fn(x)

    def test_explicit_bound_object_init_with_extra_arg_not_noop(self):
        class Foo:
            pass

        def fn(t):
            f = Foo()
            f.__init__(1)
            return t.sin()

        x = torch.randn(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            opt_fn(x)

    def test_explicit_object_init_with_extra_arg_not_noop(self):
        class Foo:
            pass

        def fn(t):
            f = Foo()
            object.__init__(f, 1)
            return t.sin()

        x = torch.randn(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            opt_fn(x)

    def test_super_object_init_with_extra_arg_not_noop(self):
        class Foo:
            def __init__(self):
                super().__init__(1)

        def fn(t):
            Foo()
            return t.sin()

        x = torch.randn(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            opt_fn(x)

    def test_instantiate_class_with_staticmethod_init(self):
        class Foo:
            def __new__(cls, *args):
                return object.__new__(cls)

            def init(a):
                pass

            __init__ = staticmethod(init)

        def fn(t):
            Foo(3)
            return t.sin()

        x = torch.randn(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_instantiate_class_with_classmethod_init(self):
        class Foo:
            def __new__(cls, *args):
                return object.__new__(cls)

            def init(cls, a):
                pass

            __init__ = classmethod(init)

        def fn(t):
            Foo(3)
            return t.sin()

        x = torch.randn(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_instantiate_class_with_custom_getattribute_reads_init_state(self):
        class Foo:
            def __init__(self, a):
                object.__setattr__(self, "ready", True)
                self.a = a

            def __getattribute__(self, name):
                if super().__getattribute__("ready"):
                    return super().__getattribute__(name)
                raise RuntimeError("not ready")

        def fn(t):
            f = Foo(3)
            return t.sin() + f.a

        x = torch.randn(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_instantiate_class_with_custom_getattribute_normal_setattr(self):
        class Foo:
            def __init__(self, a):
                self.ready = True
                self.a = a

            def __getattribute__(self, name):
                try:
                    ready = super().__getattribute__("ready")
                except AttributeError:
                    raise RuntimeError("not ready") from None
                if ready:
                    return super().__getattribute__(name)
                raise RuntimeError("not ready")

        def fn(t):
            f = Foo(3)
            return t.sin() + f.a

        x = torch.randn(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_instantiate_slotted_class_with_custom_getattribute(self):
        class Foo:
            __slots__ = ("ready", "a")

            def __init__(self, a):
                object.__setattr__(self, "ready", True)
                self.a = a

            def __getattribute__(self, name):
                if super().__getattribute__("ready"):
                    return super().__getattribute__(name)
                raise RuntimeError("not ready")

        def fn(t):
            f = Foo(3)
            return t.sin() + f.a

        x = torch.randn(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))


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

        opt_fn = torch.compile(fn, fullgraph=True)
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


if __name__ == "__main__":
    run_tests()
