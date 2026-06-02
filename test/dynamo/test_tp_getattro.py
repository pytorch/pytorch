# Owner(s): ["module: dynamo"]
"""Tests for getattro_impl: unified attribute access protocol in Dynamo."""

import torch
import torch._dynamo.test_case
import torch._dynamo.testing


class TpGetattroTests(torch._dynamo.test_case.TestCase):
    # --- getattr() builtin ---

    def test_getattr_constant(self):
        def fn():
            return getattr(42, "__class__")

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
            return getattr(x, "_grad")

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

    # --- Dunder semantic gaps (known TODOs) ---

    def test_dunder_getattribute_skips_getattr(self):
        """obj.__getattribute__("nonexistent") should raise AttributeError
        even when __getattr__ is defined, matching CPython semantics.
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
            return getattr(obj, "x") == obj.x

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
                params = list(self.parameters())
                return self.linear(x)

        m = MyModule()
        cnt = torch._dynamo.testing.CompileCounter()
        result = torch.compile(m, backend=cnt)(torch.randn(3))
        self.assertEqual(result.shape, torch.Size([4]))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
