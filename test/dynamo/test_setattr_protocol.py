# Owner(s): ["module: dynamo"]
"""Tests for tp_setattro: setattr()/delattr(), STORE_ATTR/DELETE_ATTR and the
__setattr__/__delattr__ slots."""

import collections
import functools
import unittest

import torch
from torch._dynamo.exc import Unsupported
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


class _Plain:
    def __init__(self, x):
        self.x = x


class _Slotted:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x


class _CustomSetattr:
    def __init__(self):
        object.__setattr__(self, "log", [])

    def __setattr__(self, name, value):
        self.log.append(f"set {name}")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        self.log.append(f"del {name}")
        object.__delattr__(self, name)


class _WithProperty:
    def __init__(self):
        self._v = 0

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        self._v = value + 1

    @v.deleter
    def v(self):
        self._v = -1


class _ReadOnlyProperty:
    @property
    def v(self):
        return 1


class _Cached:
    def __init__(self, x):
        self.x = x

    @functools.cached_property
    def y(self):
        return self.x + 1


class _Descriptor:
    def __set__(self, obj, value):
        obj.__dict__["k"] = value * 3

    def __delete__(self, obj):
        obj.__dict__["k"] = -1


class _WithDescriptor:
    f = _Descriptor()

    def __init__(self):
        self.__dict__["k"] = 0


class _SetOnly:
    def __set__(self, obj, value):
        obj.__dict__["k"] = value


class _WithSetOnly:
    f = _SetOnly()


_Point = collections.namedtuple("_Point", ["a", "b"])


class TpSetattroTests(TestCase):
    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    def _check(self, fn, *args):
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(*args), compiled(*args))

    def test_store_attr(self):
        def fn(x):
            obj = _Plain(x)
            obj.y = x + 1
            return obj.x, obj.y

        self._check(fn, torch.ones(3))

    def test_delete_attr(self):
        def fn(x):
            obj = _Plain(x)
            obj.y = 2
            del obj.y
            return hasattr(obj, "y"), obj.x

        self._check(fn, torch.ones(3))

    @parametrize("dunder", (False, True))
    def test_setattr_and_delattr(self, dunder):
        def fn(x):
            obj = _Plain(x)
            if dunder:
                obj.__setattr__("y", x + 1)
            else:
                obj.y = x + 1
            got = obj.y
            if dunder:
                obj.__delattr__("y")
            else:
                delattr(obj, "y")
            return got, hasattr(obj, "y")

        self._check(fn, torch.ones(3))

    def test_object_setattr_unbound(self):
        def fn(x):
            obj = _Plain(x)
            object.__setattr__(obj, "y", x + 1)
            return obj.y

        self._check(fn, torch.ones(3))

    def test_custom_setattr_is_traced(self):
        # A type that overrides tp_setattro traces its own __setattr__ rather
        # than the generic one.
        def fn(x):
            obj = _CustomSetattr()
            obj.a = x
            obj.b = 2
            del obj.b
            return obj.log, obj.a

        self._check(fn, torch.ones(3))

    def test_property_setter_and_deleter(self):
        def fn(x):
            obj = _WithProperty()
            obj.v = 41
            first = obj.v
            del obj.v
            return first, obj.v

        self._check(fn, torch.ones(3))

    def test_property_without_setter_or_deleter(self):
        def fn(x):
            obj = _ReadOnlyProperty()
            out = []
            try:
                obj.v = 3
            except AttributeError as e:
                out.append(str(e))
            try:
                del obj.v
            except AttributeError as e:
                out.append(str(e))
            return out

        self._check(fn, torch.ones(3))

    def test_property_setter_swap_recompiles(self):
        # The setter is reached through the descriptor's source, so it carries a
        # guard: swapping the property after tracing must recompile rather than
        # silently reuse the old fset.
        class Box:
            def __init__(self):
                self._v = 0

            @property
            def v(self):
                return self._v

            @v.setter
            def v(self, value):
                self._v = value + 1

        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(box, x):
            box.v = 10
            return box._v + x

        x = torch.zeros(1)
        self.assertEqual(fn(Box(), x), torch.full((1,), 11.0))

        def new_setter(self, value):
            self._v = value + 100

        Box.v = property(Box.v.fget, new_setter)
        self.assertEqual(fn(Box(), x), torch.full((1,), 110.0))
        self.assertEqual(cnt.frame_count, 2)

    def test_cached_property_write(self):
        # functools.cached_property has no __set__, so the write must fall
        # through to the instance dict and shadow the cached value.
        def fn(x):
            obj = _Cached(2)
            first = obj.y
            obj.y = 100
            return first, obj.y

        self._check(fn, torch.ones(3))

    def test_slots(self):
        def fn(x):
            obj = _Slotted(x)
            obj.x = x + 1
            return obj.x

        self._check(fn, torch.ones(3))

    def test_slotted_object_has_no_dict(self):
        def fn(x):
            obj = _Slotted(x)
            with self.assertRaises(AttributeError):
                obj.y = 1
            return x.sin()

        self._check(fn, torch.ones(3))

    def test_no_instance_dict(self):
        def fn(x):
            d = collections.deque([x])
            with self.assertRaises(AttributeError):
                d.attr = 1
            return x.sin()

        self._check(fn, torch.ones(3))

    def test_readonly_getset(self):
        def fn(x):
            d = collections.deque([x], maxlen=2)
            with self.assertRaises(AttributeError):
                d.maxlen = 10
            return x.sin()

        self._check(fn, torch.ones(3))

    def test_readonly_descriptor(self):
        # namedtuple fields are _tuplegetter, a data descriptor that refuses
        # writes.  Dynamo's message for it does not match CPython's, so only the
        # exception type is compared.
        def fn(x):
            p = _Point(x, 2)
            try:
                p.a = 3
            except AttributeError as e:
                return type(e).__name__
            return "no error"

        self._check(fn, torch.ones(3))

    def test_python_descriptor(self):
        # A descriptor class written in Python: tp_descr_set is
        # slot_tp_descr_set, which dispatches back to __set__/__delete__.
        def fn(x):
            obj = _WithDescriptor()
            obj.f = 5
            first = obj.__dict__["k"]
            del obj.f
            return first, obj.__dict__["k"]

        self._check(fn, torch.ones(3))

    def test_python_descriptor_input(self):
        # Same, but the object comes in as an input so the descriptor is sourced
        # by walking the MRO rather than wrapped sourcelessly.
        def fn(x, obj):
            obj.f = 5
            return obj.__dict__["k"]

        self._check(fn, torch.ones(3), _WithDescriptor())

    def test_python_descriptor_without_delete(self):
        # Both halves share tp_descr_set, so a delete reaches slot_tp_descr_set
        # and fails looking up __delete__: a bare AttributeError('__delete__').
        def fn(x):
            obj = _WithSetOnly()
            try:
                del obj.f
            except AttributeError as e:
                return type(e).__name__, str(e)
            return "no error", ""

        self._check(fn, torch.ones(3))

    def test_python_descriptor_dunder_set(self):
        def fn(x):
            obj = _WithDescriptor()
            _WithDescriptor.__dict__["f"].__set__(obj, 7)
            return obj.__dict__["k"]

        self._check(fn, torch.ones(3))

    def test_readonly_member(self):
        # slice.start is a READONLY PyMemberDef, so PyMember_SetOne raises
        # "readonly attribute".  MemberDescriptorVariable.tp_descr_set_impl
        # ignores the readonly flag and reports the getset message instead.
        def fn(x):
            s = slice(1, 2)
            try:
                s.start = 5
            except AttributeError as e:
                return str(e)
            return "no error"

        self._check(fn, torch.ones(3))

    def test_readonly_member_untracked_object(self):
        # Same readonly member write, but on a VT that does not support
        # attribute mutation: the unconditional store_attr trips an internal
        # assertion instead of raising AttributeError.
        def fn(x):
            def g(a, b=1):
                return a

            try:
                g.__code__.co_argcount = 7
            except AttributeError as e:
                return str(e)
            return "no error"

        self._check(fn, torch.ones(3))

    def test_non_str_name(self):
        def fn(x):
            obj = _Plain(x)
            try:
                setattr(obj, 1, 2)
            except TypeError as e:
                return str(e)
            return "no error"

        self._check(fn, torch.ones(3))

    def test_exception_attributes(self):
        def fn(x):
            e = ValueError("boom")
            e.args = ("bang",)
            e.attr = x + 1
            cause = KeyError("k")
            e.__cause__ = cause
            return e.args, e.attr, e.__cause__ is cause, e.__suppress_context__

        self._check(fn, torch.ones(3))

    def test_exception_attributes_input(self):
        # BaseException.args is a getset with a real C setter.  An exception
        # coming in as an input is not an ExceptionVariable, so the write falls
        # to GetSetDescriptorVariable, which only knows about setters declared
        # in the VT's own tp_getset and wrongly raises AttributeError.
        def fn(x, e):
            e.args = ("bang",)
            return e.args

        self._check(fn, torch.ones(3), ValueError("boom"))

    def test_getset_descriptor_dunder_set(self):
        def fn(x, e):
            type(e).args.__set__(e, ("bang",))
            return e.args

        self._check(fn, torch.ones(3), ValueError("boom"))

    def test_class_assignment_is_unsupported(self):
        # object.__class__ has a C setter Dynamo does not model: the write must
        # graph break rather than raise AttributeError.
        class A:
            pass

        class B:
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            o = A()
            o.__class__ = B
            return type(o) is B

        with self.assertRaisesRegex(Unsupported, "__class__ assignment"):
            fn(torch.ones(3))

    def test_tensor_requires_grad(self):
        # requires_grad is a getset with a real C setter that TensorVariable
        # does not model.  Whatever Dynamo does with the write, it must not
        # report it as a write to a read-only attribute.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            x.requires_grad = True
            return x.sin()

        try:
            fn(torch.zeros(3))
        except Unsupported as e:
            self.assertNotIn("is not writable", str(e))

    def test_function_attributes(self):
        def fn(x):
            def g():
                pass

            g.attr = x + 1
            g.__annotations__ = {"a": int}
            got = (g.attr, g.__annotations__)
            del g.attr
            return got, hasattr(g, "attr")

        self._check(fn, torch.ones(3))

    @unittest.expectedFailure
    def test_defaultdict_default_factory(self):
        def fn(x):
            d = collections.defaultdict(list)
            d.default_factory = set
            return d.default_factory, d["a"]

        self._check(fn, torch.ones(3))

    def test_tensor_grad(self):
        def fn(x):
            y = x + 1
            y.grad = x
            return y.grad, y._grad

        self._check(fn, torch.ones(3))

    def test_tensor_attribute(self):
        def fn(x):
            y = x + 1
            y.attr = 3
            return y.attr

        self._check(fn, torch.ones(3))

    def test_tensor_unsupported_getset(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = x + 1
            y.real = x
            return y

        with self.assertRaisesRegex(Unsupported, "Failed to set tensor attribute"):
            fn(torch.ones(3))

    def test_module_attribute(self):
        class Mod(torch.nn.Module):
            def forward(self, x):
                self.attr = x + 1
                return self.attr

        mod = Mod()
        compiled = torch.compile(mod, backend="eager", fullgraph=True)
        x = torch.ones(3)
        self.assertEqual(mod(x), compiled(x))


instantiate_parametrized_tests(TpSetattroTests)


if __name__ == "__main__":
    run_tests()
