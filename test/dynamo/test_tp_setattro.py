# Owner(s): ["module: dynamo"]
"""Tests for tp_setattro: setattr()/delattr(), STORE_ATTR/DELETE_ATTR and the
__setattr__/__delattr__ slots."""

import collections
import functools
import sys
import unittest

import torch
from torch._dynamo.exc import Unsupported
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    xfailIf,
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


class _SlotlessWithMethod:
    __slots__ = ()

    def f(self):
        pass


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


class _SlottedUnset:
    __slots__ = ("a",)


class _WithMethodDefaults:
    def m(self, a=1, *, b=2):
        return "old"


class _Meta(type):
    def __setattr__(cls, name, value):
        type.__setattr__(cls, name, ("meta", value))


class _WithMeta(metaclass=_Meta):
    pass


_Point = collections.namedtuple("_Point", ["a", "b"])


def _slot_target(a, b=1, *, c=2):
    return a + b + c


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
                setattr(obj, "y", x + 1)  # noqa: B010
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
                out.append(type(e).__name__)
            try:
                del obj.v
            except AttributeError as e:
                out.append(type(e).__name__)
            return out

        self._check(fn, torch.ones(3))

    def test_no_dict_message_depends_on_type_lookup(self):
        # CPython picks the message from whether _PyType_Lookup found anything:
        # a name resolving to a class attribute is "read-only", an unknown name
        # is "no attribute ... and no __dict__".
        def fn(x):
            obj = _SlotlessWithMethod()
            out = []
            for name in ("f", "missing"):
                try:
                    setattr(obj, name, x)
                except AttributeError as e:
                    out.append(type(e).__name__)
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

    # 3.11's cached_property.__get__ enters an RLock, which Dynamo cannot trace.
    @xfailIf(sys.version_info < (3, 12))
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
                return type(e).__name__
            return "no error"

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
                return type(e).__name__
            return "no error"

        self._check(fn, torch.ones(3))

    @unittest.expectedFailure
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
                return type(e).__name__
            return "no error"

        self._check(fn, torch.ones(3))

    def test_non_str_name(self):
        def fn(x):
            obj = _Plain(x)
            try:
                setattr(obj, 1, 2)
            except TypeError as e:
                return type(e).__name__
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

    # Blocked on the getattr side, not on setattr: `type(e)` wraps as
    # BuiltinVariable (is_builtin_callable wins over the exception-class branch
    # in trace_rules.lookup_callable), and BuiltinVariable.tp_getattro_impl
    # returns a blanket GetAttrVariable, so `type(e).args` never resolves to a
    # GetSetDescriptorVariable and tp_descr_set_impl is never reached.
    @unittest.expectedFailure
    def test_getset_descriptor_dunder_set(self):
        # BaseException.args is a getset with a real C setter, reached here
        # through the descriptor rather than STORE_ATTR.
        def fn(x):
            e = ValueError("boom")
            type(e).args.__set__(e, ("bang",))
            return e.args

        self._check(fn, torch.ones(3))

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

        with self.assertRaises(Unsupported):
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

    @unittest.expectedFailure
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

    def test_nested_function_defaults_write(self):
        # gb_type "Write to unmodeled getset/member attribute".
        def fn(x):
            def g(a, b=1):
                return a + b

            g.__defaults__ = (5,)
            return g(1)

        self._check(fn, torch.ones(3))

    def test_nested_function_kwdefaults_write(self):
        def fn(x):
            def g(a, *, b=1):
                return a + b

            g.__kwdefaults__ = {"b": 5}
            return g(1), g.__kwdefaults__

        self._check(fn, torch.ones(3))

    @parametrize("slot", ("__defaults__", "__kwdefaults__", "__annotations__"))
    def test_sourced_function_slot_write_replays(self, slot):
        # The write must reach the real function after the graph, not just the
        # in-graph read.
        value = {"__defaults__": (10,), "__kwdefaults__": {"c": 20}}.get(
            slot, {"a": str}
        )

        def fn(x):
            setattr(_slot_target, slot, value)
            return x + 1, getattr(_slot_target, slot)

        def reset():
            _slot_target.__defaults__ = (1,)
            _slot_target.__kwdefaults__ = {"c": 2}
            _slot_target.__annotations__ = {"a": int}

        reset()
        expected = fn(torch.ones(3))
        expected_real = getattr(_slot_target, slot)

        reset()
        got = torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(3))
        self.assertEqual(expected, got)
        self.assertEqual(expected_real, getattr(_slot_target, slot))

    def test_sourced_function_defaults_write_affects_call(self):
        # Binding must use the pending write, not the stale slot.
        def fn(x):
            _slot_target.__defaults__ = (10,)
            _slot_target.__kwdefaults__ = {"c": 20}
            return x + 1, _slot_target(0)

        _slot_target.__defaults__, _slot_target.__kwdefaults__ = (1,), {"c": 2}
        expected = fn(torch.ones(3))

        _slot_target.__defaults__, _slot_target.__kwdefaults__ = (1,), {"c": 2}
        self.assertEqual(
            expected, torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(3))
        )

    def test_sourced_function_slot_write_bad_type(self):
        def fn(x):
            out = []
            try:
                _slot_target.__defaults__ = 1
            except TypeError as e:
                out.append(type(e).__name__)
            try:
                _slot_target.__kwdefaults__ = 1
            except TypeError as e:
                out.append(type(e).__name__)
            return out

        self._check(fn, torch.ones(3))

    @unittest.expectedFailure
    def test_partial_setstate(self):
        # functools.partial.__setstate__ is not traced at all: the call graph
        # breaks before the namespace dict in the 4th slot is ever written.
        def fn(x):
            p = functools.partial(pow, 2)
            p.__setstate__((pow, (3,), {}, {"attr": 1}))
            return p.attr

        self._check(fn, torch.ones(3))

    # Failures found by adversarial probing of the tp_setattro protocol.  Each
    # is a compiled-vs-eager divergence; the comment names the responsible path.

    @unittest.expectedFailure
    def test_partial_input_attribute_write_replays(self):
        # object_generic_setattr_str registers a sourced non-UDO object with
        # track_attribute_mutation_new, so the write is never replayed.
        def fn(x, p):
            p.attr = 5
            return x + 1, dict(p.__dict__)

        p1, p2 = functools.partial(pow, 2), functools.partial(pow, 2)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(torch.ones(3), p1), compiled(torch.ones(3), p2))
        self.assertEqual(p1.__dict__, p2.__dict__)

    @unittest.expectedFailure
    def test_tensor_data_different_shape(self):
        # TensorVariable._set_data dropped the shape check that used to graph
        # break when requires_grad is set, so the fake tensor keeps the old shape.
        def fn(t, u):
            t.data = u
            return t.shape[0], (t * 2).shape[0]

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(torch.ones(2, requires_grad=True), torch.zeros(3))
        got = compiled(torch.ones(2, requires_grad=True), torch.zeros(3))
        self.assertEqual(expected, got)

    @unittest.expectedFailure
    def test_tensor_data_non_tensor(self):
        # _set_data reads value.dtype before checking the value is a tensor.
        def fn(x):
            y = x + 1
            try:
                y.data = 3
            except TypeError as e:
                return type(e).__name__
            return "no error"

        self._check(fn, torch.ones(3))

    @unittest.expectedFailure
    def test_sourced_function_defaults_delete(self):
        # `del f.__defaults__` stores a DeletedVariable that is never replayed;
        # CPython sets the slot to None.
        def fn(x):
            del _slot_target.__defaults__
            return x + 1

        _slot_target.__defaults__ = (1,)
        torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(3))
        try:
            self.assertIsNone(_slot_target.__defaults__)
        finally:
            _slot_target.__defaults__ = (1,)

    def test_sourced_function_defaults_set_none(self):
        # _set_defaults rejects ConstantVariable(None) with the tuple TypeError.
        def fn(x):
            _slot_target.__defaults__ = None
            return x + 1, _slot_target.__defaults__

        _slot_target.__defaults__ = (1,)
        try:
            self._check(fn, torch.ones(3))
        finally:
            _slot_target.__defaults__ = (1,)

    def test_object_setattr_unbound_on_input(self):
        # BuiltinVariable.call_method hands object_generic_setattr an unrealized
        # LazyVariableTracker, which reports "sourced but untracked".
        def fn(x, obj):
            object.__setattr__(obj, "y", 2)
            return obj.y + x

        self._check(fn, torch.ones(3), _Plain(1))

    @unittest.expectedFailure
    def test_object_delattr_unbound_on_input(self):
        def fn(x, obj):
            object.__delattr__(obj, "x")
            return hasattr(obj, "x"), x + 1

        self._check(fn, torch.ones(3), _Plain(1))

    def test_class_name_write(self):
        # type.__name__ lives on the metaclass; get_source_by_walking_mro walks
        # the class MRO and raises "not found in MRO".
        class K:
            pass

        def fn(x):
            K.__name__ = "KK"
            return K.__name__, x + 1

        self._check(fn, torch.ones(3))

    @unittest.expectedFailure
    def test_class_assignment_compatible_layout(self):
        # __class__ hits the readonly_setter in UserDefinedObjectVariable.tp_getset
        # and raises AttributeError('readonly attribute') instead of graph
        # breaking, so a user except clause observes the wrong exception.
        class A:
            pass

        class B(A):
            pass

        def fn(x, obj):
            try:
                obj.__class__ = B
                return type(obj).__name__
            except AttributeError as e:
                return type(e).__name__

        self._check(fn, torch.ones(3), A())

    def test_bound_method_doc_write(self):
        # method.__doc__ is a readonly getset; UserMethodVariable.tp_members
        # models it as writable and hits store_attr on an untracked VT.
        def fn(x):
            obj = _WithMethodDefaults()
            try:
                obj.m.__doc__ = "x"
            except AttributeError as e:
                return type(e).__name__
            return "no error"

        self._check(fn, torch.ones(3))

    @unittest.expectedFailure
    def test_readonly_member_raises_inside_region(self):
        # MemberDescriptorVariable.tp_descr_set_impl has no model for
        # __globals__ and records the write for replay, so the AttributeError
        # fires after the region instead of inside the user's try block.
        def fn(x, f):
            try:
                f.__globals__ = {}
            except AttributeError:
                return "caught", x + 1
            return "not raised", x + 1

        self._check(fn, torch.ones(3), _slot_target)

    @unittest.expectedFailure
    def test_immutable_type_setattr(self):
        # object_generic_setattr_str has no immutable-type check; eager raises
        # "cannot set 'x' attribute of immutable type 'int'".
        def fn(x):
            try:
                int.x = 1
            except TypeError as e:
                return type(e).__name__
            return "no error"

        self._check(fn, torch.ones(3))

    @unittest.expectedFailure
    def test_object_setattr_on_class(self):
        # eager: "can't apply this __setattr__ to type object".
        class K:
            pass

        def fn(x):
            try:
                object.__setattr__(K, "y", 1)
            except TypeError as e:
                return type(e).__name__
            return "no error"

        self._check(fn, torch.ones(3))

    @unittest.expectedFailure
    def test_type_setattr_unbound(self):
        # _wrap_setattr counts the explicit class argument and reports
        # "expected 2 arguments, got 3".
        class K:
            pass

        def fn(x):
            type.__setattr__(K, "y", 7)
            got = K.y
            type.__delattr__(K, "y")
            return got, hasattr(K, "y"), x + 1

        self._check(fn, torch.ones(3))

    def test_no_dict_message_builtin_object(self):
        # CPython: "'int' object has no attribute 'x' and no __dict__ for
        # setting new attributes"; Dynamo reports the read-only message.
        def fn(x):
            try:
                (1).x = 2
            except AttributeError as e:
                return type(e).__name__
            return "no error"

        self._check(fn, torch.ones(3))

    def test_readonly_getset_message(self):
        # DequeVariable.call_method used to produce the exact C message
        # "attribute 'maxlen' of 'collections.deque' objects is not writable".
        def fn(x):
            d = collections.deque([x], maxlen=2)
            try:
                d.maxlen = 10
            except AttributeError as e:
                return type(e).__name__
            return "no error"

        self._check(fn, torch.ones(3))

    @unittest.expectedFailure
    def test_delete_unset_slot(self):
        # MemberDescriptorVariable.tp_descr_set_impl stores DeletedVariable
        # without checking that the slot holds a value.
        def fn(x):
            obj = _SlottedUnset()
            try:
                del obj.a
            except AttributeError as e:
                return type(e).__name__
            return "no error"

        self._check(fn, torch.ones(3))

    @unittest.expectedFailure
    def test_module_delete_missing_attribute(self):
        # nn.Module.__delattr__ ends in super().__delattr__, which still takes
        # the legacy object.__delattr__ branch in SuperVariable with no
        # existence check.
        def fn(x, mod):
            try:
                del mod.zzz
            except AttributeError as e:
                return type(e).__name__, x + 1
            return "no error", x + 1

        self._check(fn, torch.ones(3), torch.nn.Linear(1, 1))

    @unittest.expectedFailure
    def test_setattr_added_to_class_recompiles(self):
        # UserDefinedObjectVariable.tp_setattro_impl checks
        # type(value).__setattr__ is object.__setattr__ without a guard.
        class R:
            def __init__(self):
                self.x = 0

        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(obj, x):
            obj.x = 5
            return obj.x + x

        x = torch.ones(1)
        self.assertEqual(fn(R(), x), torch.full((1,), 6.0))

        R.__setattr__ = lambda self, k, v: object.__setattr__(self, k, v * 100)
        obj = R()
        self.assertEqual(fn(obj, x), torch.full((1,), 501.0))
        self.assertEqual(obj.x, 500)
        self.assertEqual(cnt.frame_count, 2)

    @unittest.expectedFailure
    def test_metaclass_setattr(self):
        # UserDefinedClassVariable.tp_setattro_impl goes straight to
        # object_generic_setattr and never consults type(cls).__setattr__.
        def fn(x):
            _WithMeta.attr = 1
            return _WithMeta.attr, x + 1

        try:
            self._check(fn, torch.ones(3))
        finally:
            type.__delattr__(_WithMeta, "attr")

    @unittest.expectedFailure
    def test_tensor_delete_missing_attribute(self):
        # object_generic_setattr_str asks get_dict_vt for the tensor, which
        # TensorVariable does not support.
        def fn(x):
            y = x + 1
            try:
                del y.meta
            except AttributeError as e:
                return type(e).__name__
            return "no error"

        self._check(fn, torch.ones(3))

    @unittest.expectedFailure
    def test_tensor_grad_delete_then_read(self):
        # _set_grad stores DeletedVariable, which the following read cannot
        # resolve to None.
        def fn(x):
            y = x + 1
            y.grad = x
            del y.grad
            return y.grad

        self._check(fn, torch.ones(3))

    @unittest.expectedFailure
    def test_function_name_delete(self):
        # _set_name only rejects non-None non-str; eager raises
        # "__name__ must be set to a string object".
        def fn(x):
            def g():
                pass

            try:
                del g.__name__
            except TypeError as e:
                return type(e).__name__
            return "no error"

        self._check(fn, torch.ones(3))

    @unittest.expectedFailure
    def test_dict_order_after_attr_write_on_input(self):
        # Pending writes are listed before the existing keys of a sourced
        # instance dict.
        def fn(x, obj):
            obj.y = 5
            return list(obj.__dict__), x + 1

        self._check(fn, torch.ones(3), _Plain(1))

    @unittest.expectedFailure
    def test_exception_cause_class(self):
        # eager: "exception cause must be None or derive from BaseException".
        def fn(x):
            e = ValueError("boom")
            try:
                e.__cause__ = KeyError
            except TypeError as err:
                return str(err)
            return "no error"

        self._check(fn, torch.ones(3))


instantiate_parametrized_tests(TpSetattroTests)


if __name__ == "__main__":
    run_tests()
