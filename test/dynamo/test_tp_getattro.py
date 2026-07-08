# Owner(s): ["module: dynamo"]
"""Tests for getattro_impl: unified attribute access protocol in Dynamo."""

import torch
import torch._dynamo.test_case
import torch._dynamo.testing


class TpGetattroTests(torch._dynamo.test_case.TestCase):
    # --- getattr() builtin ---

    def test_getattr_constant(self):
        def fn():
            return (42).__class__

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertIs(result, int)

    def test_getattr_with_default_exists(self):
        def fn():
            return getattr("hello", "__class__", None)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertIs(result, str)

    def test_getattr_with_default_missing(self):
        def fn():
            return getattr("hello", "nonexistent", 42)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 42)

    def test_getattr_with_none_default(self):
        def fn():
            return getattr("hello", "nonexistent", None)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertIsNone(result)

    # --- hasattr() builtin ---

    def test_hasattr_true(self):
        def fn(x):
            if hasattr(x, "shape"):
                return x + 1
            return x

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, x + 1)

    def test_hasattr_false(self):
        def fn():
            return hasattr(42, "nonexistent")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertFalse(result)

    def test_hasattr_constant_true(self):
        def fn():
            return hasattr("hello", "upper")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertTrue(result)

    def test_hasattr_false_then_access(self):
        """hasattr returning False must not leak exception state."""

        def fn(x):
            _ = hasattr(42, "nonexistent")
            return x.shape[0]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(5))
        self.assertEqual(result, 5)

    def test_hasattr_sequence(self):
        """Multiple hasattr calls must each restore exception state."""

        def fn():
            a = hasattr(42, "__add__")
            b = hasattr(42, "nonexistent")
            c = hasattr("hi", "upper")
            d = hasattr("hi", "nonexistent")
            return (a, b, c, d)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, (True, False, True, False))

    def test_hasattr_false_in_except(self):
        """hasattr inside an except block must preserve the active exception."""
        import sys

        def fn(x):
            try:
                raise ValueError("test")
            except ValueError:
                has = hasattr(42, "nonexistent")
                exc_type = sys.exc_info()[0]
                if not has and exc_type is ValueError:
                    return x + 1
            return x

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, x + 1)

    def test_hasattr_user_function_true(self):
        def bar():
            pass

        def fn():
            return hasattr(bar, "__name__")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertTrue(result)

    def test_hasattr_user_function_false(self):
        def bar():
            pass

        def fn():
            return hasattr(bar, "nonexistent")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertFalse(result)

    def test_hasattr_skip_function_true(self):
        def fn():
            return hasattr(print, "__name__")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertTrue(result)

    def test_hasattr_skip_function_false(self):
        def fn():
            return hasattr(print, "nonexistent")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertFalse(result)

    def test_hasattr_python_module_true(self):
        import math

        def fn():
            return hasattr(math, "sqrt")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertTrue(result)

    def test_hasattr_python_module_false(self):
        import math

        def fn():
            return hasattr(math, "nonexistent")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertFalse(result)

    # --- Tensor attributes ---

    def test_tensor_shape(self):
        def fn(x):
            return x.shape[0]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(5, 3))
        self.assertEqual(result, 5)

    def test_tensor_dtype(self):
        def fn(x):
            return x.dtype

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3))
        self.assertEqual(result, torch.float32)

    def test_tensor_device(self):
        def fn(x):
            return x.device

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, x.device)

    def test_tensor_grad_alias(self):
        cnt = torch._dynamo.testing.CompileCounter()

        def fn(x):
            return x._grad

        x = torch.randn(3, requires_grad=True)
        x.grad = torch.ones(3)
        result = torch.compile(fn, backend=cnt)(x)
        self.assertEqual(result, x.grad)

    # --- User-defined objects ---

    def test_udov_instance_attr(self):
        class MyObj:
            def __init__(self):
                self.val = 42

        def fn(obj):
            return obj.val

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 42)

    def test_udov_class_attr(self):
        class MyObj:
            class_val = 99

        def fn(obj):
            return obj.class_val

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 99)

    def test_udov_property(self):
        class MyObj:
            @property
            def val(self):
                return 42

        def fn(obj):
            return obj.val

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 42)

    def test_udov_getattr_fallback(self):
        class MyObj:
            def __getattr__(self, name):
                if name == "dynamic":
                    return 123
                raise AttributeError(name)

        def fn(obj):
            return obj.dynamic

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 123)

    def test_udov_getattribute_override(self):
        class MyObj:
            def __getattribute__(self, name):
                if name == "special":
                    return 999
                return super().__getattribute__(name)

        def fn(obj):
            return obj.special

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 999)

    # --- User-defined classes (type_getattro) ---

    def test_class_attr(self):
        class MyClass:
            x = 42

        def fn():
            return MyClass.x

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 42)

    def test_class_bases(self):
        class A:
            pass

        class B(A):
            pass

        def fn():
            return B.__bases__

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, (A,))

    def test_class_base(self):
        class A:
            pass

        class B(A):
            pass

        def fn():
            return B.__base__

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertIs(result, A)

    def test_builtin_type_bases(self):
        """__bases__ on a BuiltinVariable type returns a proper TupleVariable."""

        def fn():
            return ArithmeticError.__bases__

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, ArithmeticError.__bases__)

    def test_builtin_type_bases_len(self):
        """len() on __bases__ of a builtin type works (regression for test_c_classes)."""

        def fn():
            return len(ArithmeticError.__bases__)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, len(ArithmeticError.__bases__))

    def test_builtin_type_bases_reversed(self):
        """reversed() over __bases__ works (the original test_c_classes failure path)."""

        def fn():
            return list(reversed(ArithmeticError.__bases__))

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, list(reversed(ArithmeticError.__bases__)))

    def test_builtin_type_bases_slicing(self):
        """Slicing __bases__ works (used by functools._c3_mro)."""

        def fn():
            bases = OSError.__bases__
            return bases[:1], bases[1:]

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        expected = OSError.__bases__[:1], OSError.__bases__[1:]
        self.assertEqual(result, expected)

    # --- Module attributes ---

    def test_nn_module_forward(self):
        m = torch.nn.Linear(3, 4)
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x, m):
            return m(x)

        result = fn(torch.randn(3), m)
        self.assertEqual(result.shape, torch.Size([4]))
        self.assertEqual(cnt.frame_count, 1)

    # --- Dunder method dispatch ---

    def test_dunder_getattribute(self):
        class MyObj:
            def __init__(self):
                self.val = 42

        def fn(obj):
            return obj.__getattribute__("val")

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 42)

    def test_dunder_getattribute_skips_getattr(self):
        """obj.__getattribute__("nonexistent") raises AttributeError even
        when __getattr__ is defined.  This matches CPython: the bytecode
        path (LOAD_ATTR + CALL) resolves __getattribute__ as a bound
        WrapperDescriptor that calls object.__getattribute__ directly,
        which does not invoke __getattr__.
        """

        class MyObj:
            def __getattr__(self, name):
                return 42

        def fn(obj):
            return obj.__getattribute__("nonexistent")

        with self.assertRaises(AttributeError):
            torch.compile(fn, backend="eager")(MyObj())

    # --- Sparse tensor blocking ---

    def test_sparse_tensor_attr_access_graph_breaks(self):
        cnt = torch._dynamo.testing.CompileCounter()

        def fn(x):
            _ = x.shape
            return x

        x = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [2, 3]]),
            torch.tensor([4.0, 5.0]),
            size=(4, 4),
        )
        result = torch.compile(fn, backend=cnt)(x)
        self.assertEqual(result.to_dense(), x.to_dense())
        # Sparse tensor attribute access triggers graph break
        self.assertEqual(cnt.frame_count, 0)

    # --- TorchInGraphFunctionVariable ---

    def test_torch_in_graph_function_getattro(self):
        def fn(x):
            return torch.sin(x)

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, torch.sin(x))

    # --- Descriptor protocol (tp_descr_get through getattro_impl) ---

    def test_data_descriptor_priority_over_instance_dict(self):
        """Data descriptors (property) take precedence over instance __dict__."""

        class MyObj:
            @property
            def x(self):
                return 99

        obj = MyObj()
        obj.__dict__["x"] = 1

        def fn(obj):
            return obj.x

        result = torch.compile(fn, backend="eager")(obj)
        self.assertEqual(result, 99)

    def test_non_data_descriptor_shadowed_by_instance_dict(self):
        """Instance __dict__ takes precedence over non-data descriptors."""

        class Desc:
            def __get__(self, obj, objtype=None):
                return 99

        class MyObj:
            x = Desc()

        obj = MyObj()
        obj.__dict__["x"] = 1

        def fn(obj):
            return obj.x

        result = torch.compile(fn, backend="eager")(obj)
        self.assertEqual(result, 1)

    def test_staticmethod_descriptor(self):
        class MyObj:
            @staticmethod
            def greet():
                return 42

        def fn(obj):
            return obj.greet()

        result = torch.compile(fn, backend="eager", fullgraph=True)(MyObj())
        self.assertEqual(result, 42)

    def test_classmethod_descriptor(self):
        class MyObj:
            val = 10

            @classmethod
            def get_val(cls):
                return cls.val

        def fn(obj):
            return obj.get_val()

        result = torch.compile(fn, backend="eager", fullgraph=True)(MyObj())
        self.assertEqual(result, 10)

    def test_classmethod_descriptor_on_class(self):
        class MyObj:
            val = 10

            @classmethod
            def get_val(cls):
                return cls.val

        def fn():
            return MyObj.get_val()

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 10)

    def test_property_setter(self):
        class MyObj:
            def __init__(self):
                self._x = 0

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, val):
                self._x = val * 2

        def fn(obj):
            obj.x = 5
            return obj.x

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 10)

    def test_slots_member_descriptor(self):
        class MyObj:
            __slots__ = ("x", "y")

            def __init__(self):
                self.x = 1
                self.y = 2

        def fn(obj):
            return obj.x + obj.y

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 3)

    def test_namedtuple_field_access(self):
        from collections import namedtuple

        Point = namedtuple("Point", ["x", "y"])

        def fn():
            p = Point(3, 4)
            return p.x + p.y

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 7)

    def test_wrapper_descriptor_binding(self):
        """list.__add__ is a wrapper_descriptor; [1].__add__ binds it."""

        def fn():
            x = [1, 2]
            y = [3, 4]
            return x.__add__(y)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, [1, 2, 3, 4])

    def test_method_descriptor_binding(self):
        """dict.keys is a method_descriptor; {}.keys() binds and calls it."""

        def fn():
            d = {"a": 1, "b": 2}
            return list(d.keys())

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(sorted(result), ["a", "b"])

    def test_classmethod_descriptor_dict_fromkeys(self):
        """dict.fromkeys is a classmethod_descriptor."""

        def fn():
            return dict.fromkeys(["a", "b"], 0)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, {"a": 0, "b": 0})

    # --- Consistency ---

    def test_getattr_matches_dot_access(self):
        class MyObj:
            x = 42

        def fn(obj):
            return obj.x == obj.x

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertTrue(result)

    # --- generic_getattr dispatch ---

    def test_generic_getattr_side_effects(self):
        class MyObj:
            def __init__(self):
                self.x = 1

        def fn(obj):
            obj.x = 42
            return obj.x

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 42)

    # --- UnspecializedNNModule pending mutation ---

    def test_unspecialized_nn_module_pending_mutation_graph_breaks(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 4)

            def forward(self, x):
                self.extra_val = 1
                _params = list(self.parameters())
                return self.linear(x)

        m = MyModule()
        cnt = torch._dynamo.testing.CompileCounter()
        result = torch.compile(m, backend=cnt)(torch.randn(3))
        self.assertEqual(result.shape, torch.Size([4]))
        self.assertEqual(cnt.frame_count, 1)

    # --- object_generic_getattr on converted VTs ---

    def test_constant_method_via_generic_getattr(self):
        """ConstantVariable now resolves methods through the descriptor protocol
        via object_generic_getattr, instead of falling back to GetAttrVariable.
        """

        def fn():
            return "hello".upper()

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, "HELLO")

    def test_constant_class_attr_via_generic_getattr(self):
        """(42).__class__ resolves through getset_descriptor on object."""

        def fn():
            x = 42
            return x.__class__

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertIs(result, int)

    def test_range_method_via_generic_getattr(self):
        """RangeVariable now resolves methods through the descriptor protocol."""

        def fn():
            r = range(10)
            return r.count(5)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 1)

    def test_range_index_via_generic_getattr(self):
        def fn():
            r = range(10)
            return r.index(7)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 7)

    # --- object_generic_getattr edge cases ---

    def test_constant_nonexistent_attr_raises(self):
        """Step 7: mro_lookup returns NO_SUCH_SUBOBJ -> AttributeError."""

        def fn():
            x = 42
            return x.nonexistent

        with self.assertRaises(AttributeError):
            torch.compile(fn, backend="eager")()

    def test_range_start_stop_step(self):
        """RangeVariable.getattro_impl fast path for start/stop/step."""

        def fn():
            r = range(2, 10, 3)
            return r.start, r.stop, r.step

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, (2, 10, 3))

    def test_class_flags(self):
        class A:
            pass

        def fn():
            return A.__flags__

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, A.__flags__)

    # --- Dunder __getattr__ explicit call ---

    def test_dunder_getattr_explicit_call(self):
        class MyObj:
            def __getattr__(self, name):
                if name == "dynamic":
                    return 123
                raise AttributeError(name)

        def fn(obj):
            return obj.__getattr__("dynamic")

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 123)

    # --- BoundBuiltinMethodVariable slots ---

    def test_bound_builtin_method_hash(self):
        """hash() on a bound builtin method produced by object_generic_getattr."""

        def fn():
            s = "hello"
            h = hash(s.upper)
            return isinstance(h, int)

        result = torch.compile(fn, backend="eager")()
        self.assertTrue(result)

    def test_bound_builtin_method_identity_comparison(self):
        """Bound builtin methods use identity comparison."""

        def fn():
            s = "hello"
            m1 = s.upper
            m2 = s.upper
            return m1 is not m2

        result = torch.compile(fn, backend="eager")()
        self.assertTrue(result)

    def test_bmv_load_then_call(self):
        """Load a method into a variable, then call it through CMV."""

        def fn():
            r = range(10)
            m = r.count
            return m(5)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 1)

    def test_bmv_defers_graph_break_to_call_time(self):
        """CallMethodVariable defers graph breaks from LOAD_ATTR to CALL.

        When a method exists on the type (MRO walk finds it) but the VT's
        call_method doesn't handle it, CMV is returned at load time and
        the graph break happens at call time, not at attribute access time.
        """

        # Loading the method succeeds (CMV returned, no graph break).
        @torch.compile(backend="eager", fullgraph=True)
        def fn_load(x):
            r = range(10)
            r.__reduce__
            return x + 1

        x = torch.randn(3)
        self.assertEqual(fn_load(x), x + 1)

        torch._dynamo.reset()

        # Calling it graph-breaks (call_method doesn't handle __reduce__).
        def fn_call(x):
            r = range(10)
            r.__reduce__()
            return x + 1

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn_call, backend="eager", fullgraph=True)(x)

    # --- ConstantVariable: CallMethodVariable (format/join have call_method) ---

    def test_str_format_via_bound_method(self):
        def fn():
            return "hello {}".format("world")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, "hello world")

    def test_str_join_via_bound_method(self):
        def fn():
            return ", ".join(["a", "b", "c"])

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, "a, b, c")

    # --- ConstantVariable: other constant types ---

    def test_float_method_via_generic_getattr(self):
        def fn():
            return (3.14).is_integer()

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertFalse(result)

    def test_int_method_via_generic_getattr(self):
        def fn():
            return (255).bit_length()

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 8)

    def test_complex_real_imag(self):
        def fn():
            c = 3 + 4j
            return c.real, c.imag

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, (3.0, 4.0))

    def test_bytes_method_via_generic_getattr(self):
        def fn():
            return b"hello".decode("utf-8")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, "hello")

    def test_dict_method_as_standalone_attr(self):
        # Regression test: accessing d.keys as a standalone value (not
        # immediately called) must not create an AttrSource chain through
        # the DictGuardManager, which doesn't support getattr_manager.
        def fn(d):
            m = d.keys
            return m.__name__

        result = torch.compile(fn, backend="eager", fullgraph=True)({"a": 1})
        self.assertEqual(result, "keys")

    def test_pybind11_class_attr_constant_fold(self):
        """LOAD_ATTR on a constant class with a pybind11 metaclass (which
        overrides __getattribute__) should constant-fold rather than
        graph-break."""

        def fn(x):
            return x + torch.nn.attention.SDPBackend.MATH.value

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_constant_fold_fallback_recompiles_on_change(self):
        """When _load_attr constant-folds because the metaclass overrides
        __getattribute__, the guard must trigger recompilation if the
        attribute value changes."""

        class Meta(type):
            def __getattribute__(cls, name):
                return type.__getattribute__(cls, name)

        class MyClass(metaclass=Meta):
            value = 10

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            return x + MyClass.value

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 10)
        self.assertEqual(cnt.frame_count, 1)

        MyClass.value = 20
        self.assertEqual(fn(x), x + 20)
        self.assertEqual(cnt.frame_count, 2)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
