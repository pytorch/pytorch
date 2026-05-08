# Owner(s): ["module: dynamo"]
"""Tests for mp_subscript_impl: unified __getitem__ dispatch via vt_getitem in Dynamo.

Tests exercise the vt_getitem → mp_subscript_impl path via operator.getitem(),
and the call_method("__getitem__") → mp_subscript_impl path via obj.__getitem__().

See TODO(follow-up) comments on each mp_subscript_impl override for remaining
CPython behavioral gaps.
"""

import collections
import operator
import types
import unittest

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON, HAS_GPU


class GetItemTests(torch._dynamo.test_case.TestCase):
    def _compile(self, fn, *args):
        return torch.compile(fn, backend="eager", fullgraph=True)(*args)

    # --- BaseListVariable (ListVariable) ---

    def test_list_int_index(self):
        def fn(x):
            items = [x, x + 1, x + 2]
            return (
                operator.getitem(items, 0),
                operator.getitem(items, 1),
                operator.getitem(items, 2),
            )

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_list_getitem_compiled_directly(self):
        compiled = torch.compile(operator.getitem, backend="eager", fullgraph=True)
        items = [10, 20, 30]
        self.assertEqual(compiled(items, 0), 10)
        self.assertEqual(compiled(items, 2), 30)

    def test_list_slice(self):
        def fn(x):
            items = [x, x + 1, x + 2]
            full = operator.getitem(items, slice(None))
            partial = operator.getitem(items, slice(0, 2))
            single = operator.getitem(items, slice(1, 2))
            return full, partial, single

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_list_negative_index(self):
        def fn(x):
            items = [x, x + 1, x + 2]
            return operator.getitem(items, -1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_list_bool_index(self):
        def fn(x):
            items = [x, x + 1, x + 2]
            return (
                operator.getitem(items, False),
                operator.getitem(items, True),
            )

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_list_invalid_index_type(self):
        def fn(x):
            items = [x, x + 1, x + 2]
            return operator.getitem(items, "a")

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    def test_list_index_via_index_dunder(self):
        """Custom __index__ object used as list index — _PyIndex_Check + nb_index_impl."""

        class Idx:
            def __index__(self):
                return 2

        def fn(x):
            items = [x, x + 1, x + 2]
            return operator.getitem(items, Idx())

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- BaseListVariable (TupleVariable) ---

    def test_tuple_int_index(self):
        def fn(x):
            items = (x, x + 1, x + 2)
            return operator.getitem(items, 0)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tuple_negative_index(self):
        def fn(x):
            items = (x, x + 1, x + 2)
            return operator.getitem(items, -1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tuple_slice(self):
        def fn(x):
            items = (x, x + 1, x + 2)
            full = operator.getitem(items, slice(None))
            partial = operator.getitem(items, slice(0, 2))
            single = operator.getitem(items, slice(1, 2))
            return full, partial, single

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tuple_bool_index(self):
        def fn(x):
            items = (x, x + 1, x + 2)
            return operator.getitem(items, False)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tuple_invalid_index_type(self):
        def fn(x):
            items = (x, x + 1, x + 2)
            return operator.getitem(items, "a")

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    def test_tuple_index_via_index_dunder(self):
        class Idx:
            def __index__(self):
                return 2

        def fn(x):
            items = (x, x + 1, x + 2)
            return operator.getitem(items, Idx())

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- RangeVariable ---

    def test_range_int_index(self):
        def fn(x):
            r = range(0, 10, 2)
            return x + operator.getitem(r, 3)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_range_negative_index(self):
        def fn(x):
            r = range(0, 10, 2)
            return x + operator.getitem(r, -1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_range_slice(self):
        def fn(x):
            r = range(0, 10, 2)
            result = operator.getitem(r, slice(1, 3))
            return x + result[0]

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_range_bool_index(self):
        def fn(x):
            r = range(0, 10, 2)
            return x + operator.getitem(r, True)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_range_index_via_index_dunder(self):
        class Idx:
            def __index__(self):
                return 2

        def fn(x):
            r = range(0, 10, 2)
            return x + operator.getitem(r, Idx())

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_range_invalid_index_type(self):
        def fn(x):
            r = range(0, 10, 2)
            return x + operator.getitem(r, "a")

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    # --- SizeVariable ---

    def test_size_int_index(self):
        def fn(x):
            s = x.size()
            return x + operator.getitem(s, 0)

        x = torch.randn(4, 8)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_size_negative_index(self):
        def fn(x):
            s = x.size()
            return x + operator.getitem(s, -1)

        x = torch.randn(4, 8)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_size_slice(self):
        def fn(x):
            s = x.size()
            result = operator.getitem(s, slice(0, 1))
            return x + result[0]

        x = torch.randn(4, 8)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_size_bool_index(self):
        def fn(x):
            s = x.size()
            return x + operator.getitem(s, False)

        x = torch.randn(4, 8)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_size_index_via_index_dunder(self):
        class Idx:
            def __index__(self):
                return 1

        def fn(x):
            s = x.size()
            return x + operator.getitem(s, Idx())

        x = torch.randn(4, 8)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_size_invalid_index_type(self):
        def fn(x):
            s = x.size()
            return x + operator.getitem(s, "a")

        x = torch.randn(4, 8)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    # --- ConstDictVariable ---

    def test_dict_str_key(self):
        def fn(x):
            d = {"a": x, "b": x + 1}
            return operator.getitem(d, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_dict_int_key(self):
        def fn(x):
            d = {0: x, 1: x + 1}
            return operator.getitem(d, 1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_dict_missing_key(self):
        def fn(x):
            d = {"a": x}
            return operator.getitem(d, "missing")

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    # --- DefaultDictVariable ---

    def test_defaultdict_existing_key(self):
        def fn(x):
            d = collections.defaultdict(lambda: x + 99)
            d["a"] = x
            return operator.getitem(d, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_defaultdict_missing_key(self):
        def fn(x):
            d = collections.defaultdict(list)
            operator.getitem(d, "new")
            return x

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- TensorVariable ---

    def test_tensor_int_index(self):
        def fn(x):
            return operator.getitem(x, 0)

        x = torch.randn(4, 4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tensor_slice(self):
        def fn(x):
            return operator.getitem(x, slice(0, 2))

        x = torch.randn(4, 4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tensor_tuple_index(self):
        def fn(x):
            return operator.getitem(x, (0, slice(1, 3)))

        x = torch.randn(4, 4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tensor_getitem_torch_function_mode(self):
        """TorchFunctionMode intercepts tensor __getitem__ and can modify behavior."""

        class AddOneMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                if func is torch.Tensor.__getitem__:
                    return result + 1
                return result

        def fn(x):
            with AddOneMode():
                return operator.getitem(x, 0)

        x = torch.randn(4, 4)
        expected = fn(x)
        compiled = torch.compile(fn, backend="eager")(x)
        self.assertEqual(expected, compiled)

    def test_tensor_getitem_torch_function_subclass(self):
        """Tensor subclass with __torch_function__ intercepts __getitem__."""

        class ScaledTensor(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                result = super().__torch_function__(func, types, args, kwargs or {})
                if func is torch.Tensor.__getitem__:
                    return result * 2
                return result

        def fn(x):
            return operator.getitem(x, 0)

        x = ScaledTensor(torch.randn(4, 4))
        expected = fn(x)
        compiled = torch.compile(fn, backend="eager")(x)
        self.assertEqual(expected, compiled)

    # --- NamedTupleVariable (via UserDefinedTupleVariable) ---

    def test_namedtuple_int_index(self):
        def fn(x):
            result = torch.topk(x, 2)
            return operator.getitem(result, 1)

        x = torch.randn(10)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_namedtuple_values_index(self):
        def fn(x):
            result = torch.topk(x, 2)
            return operator.getitem(result, 0)

        x = torch.randn(10)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- TypingVariable ---

    def test_typing_subscript(self):
        def fn(x):
            t = list[int]  # noqa: F841
            return x + 1

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- MappingProxyVariable ---

    def test_mappingproxy_getitem(self):
        def fn(x):
            d = {"a": 1, "b": 2}
            proxy = types.MappingProxyType(d)
            return x + operator.getitem(proxy, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- NNModuleVariable (ModuleList) ---

    def test_nn_module_list_int_index(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(4, 4) for _ in range(3)]
                )

            def forward(self, x):
                return operator.getitem(self.layers, 1)(x)

        model = Model()
        x = torch.randn(4)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        self.assertEqual(model(x), compiled(x))

    # --- NNModuleVariable (ModuleDict) ---

    def test_nn_module_dict_str_key(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleDict({"fc": torch.nn.Linear(4, 4)})

            def forward(self, x):
                return operator.getitem(self.layers, "fc")(x)

        model = Model()
        x = torch.randn(4)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        self.assertEqual(model(x), compiled(x))

    # --- NNModuleVariable (Sequential) ---

    def test_nn_sequential_int_index(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seq = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )

            def forward(self, x):
                return operator.getitem(self.seq, 0)(x)

        model = Model()
        x = torch.randn(4)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        self.assertEqual(model(x), compiled(x))

    # --- UserDefinedObjectVariable ---

    def test_user_defined_object_getitem(self):
        class Container:
            def __init__(self, items):
                self.items = items

            def __getitem__(self, key):
                return self.items[key]

        def fn(x):
            c = Container([x, x + 1])
            return operator.getitem(c, 0)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_object_without_getitem(self):
        class NoGetItem:
            pass

        def fn(x):
            obj = NoGetItem()
            return operator.getitem(obj, 0)

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    # --- UserDefinedListVariable ---

    def test_user_defined_list_getitem(self):
        class MyList(list):
            pass

        def fn(x):
            items = MyList([x, x + 1, x + 2])
            return operator.getitem(items, 1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- UserDefinedTupleVariable ---

    def test_user_defined_tuple_getitem(self):
        class MyTuple(tuple):  # noqa: SLOT001
            pass

        def fn(x):
            items = MyTuple((x, x + 1, x + 2))
            return operator.getitem(items, 1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- UserDefinedDictVariable ---

    def test_user_defined_dict_getitem(self):
        class MyDict(dict):
            pass

        def fn(x):
            d = MyDict(a=x, b=x + 1)
            return operator.getitem(d, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_user_defined_dict_missing(self):
        class MyDict(dict):
            def __missing__(self, key):
                return 42

        def fn(x):
            d = MyDict(a=1)
            return x + operator.getitem(d, "b")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_user_defined_dict_custom_missing(self):
        class DefaultDict(dict):
            def __missing__(self, key):
                self[key] = len(self)
                return self[key]

        def fn(x):
            d = DefaultDict()
            d["a"] = 1
            val = operator.getitem(d, "b")
            return x + val

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_collections_counter_getitem(self):
        def fn(x):
            c = collections.Counter({"a": 1, "b": 2})
            return x + operator.getitem(c, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- UserDefined* with overridden __getitem__ ---

    def test_user_defined_dict_overridden_getitem(self):
        """Dict subclass with custom __getitem__ should NOT delegate to _base_vt."""

        class MyDict(dict):
            def __getitem__(self, key):
                return super().__getitem__(key) + 100

        def fn(x):
            d = MyDict(a=1, b=2)
            return x + operator.getitem(d, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_user_defined_list_overridden_getitem(self):
        """List subclass with custom __getitem__ should NOT delegate to _base_vt."""

        class MyList(list):
            def __getitem__(self, key):
                return super().__getitem__(key) * 2

        def fn(x):
            items = MyList([x, x + 1, x + 2])
            return operator.getitem(items, 1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_counter_missing_key(self):
        """Counter.__missing__ returns 0 for missing keys."""

        def fn(x):
            c = collections.Counter({"a": 1, "b": 2})
            return x + operator.getitem(c, "missing")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- TorchScriptObjectVariable ---

    def test_opaque_object_getitem(self):
        from torch._library.opaque_object import (
            MemberType,
            OpaqueBase,
            register_opaque_type,
        )

        class OpaqueScaler(OpaqueBase):
            def __init__(self, scale):
                self.scale = scale

            def apply(self, x):
                return x * self.scale

        class OpaqueContainer(OpaqueBase):
            def __init__(self, items):
                self.items = items

            def __getitem__(self, idx):
                return self.items[idx]

        register_opaque_type(
            OpaqueScaler,
            typ="reference",
            members={
                "scale": MemberType.USE_REAL,
                "apply": MemberType.INLINED,
            },
        )
        register_opaque_type(
            OpaqueContainer,
            typ="reference",
            members={
                "items": MemberType.USE_REAL,
                "__getitem__": MemberType.INLINED,
            },
        )

        def fn(x, c):
            scaler = operator.getitem(c, 0)
            return scaler.apply(x)

        x = torch.randn(4)
        c = OpaqueContainer([OpaqueScaler(2.0), OpaqueScaler(3.0)])
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, c), compiled(x, c))

    # --- TritonKernelVariable ---

    @unittest.skipUnless(HAS_GPU and HAS_CUDA_AND_TRITON, "requires gpu and triton")
    def test_triton_kernel_getitem_grid(self):
        from torch.testing._internal.triton_utils import add_kernel

        def fn(x, y):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements // 256,)
            bound = operator.getitem(add_kernel, grid)
            bound(x, y, output, n_elements, BLOCK_SIZE=256)
            return output

        x = torch.randn(256, device="cuda")
        y = torch.randn(256, device="cuda")
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, y), compiled(x, y))

    # ===================================================================
    # CPython behavioral gaps — expectedFailure until implemented
    # ===================================================================

    # GAP 1: deque has only sq_item (int index), no mp_subscript.
    # CPython: deque[slice] → TypeError "sequence index must be integer, not 'slice'"
    # Dynamo: DequeVariable inherits BaseListVariable.mp_subscript_impl which accepts slices.
    # TODO: DequeVariable should override mp_subscript_impl to reject slices, matching
    # CPython's deque which only has sq_item (Modules/_collectionsmodule.c:1888).
    @unittest.expectedFailure
    def test_deque_slice_should_reject(self):
        """deque does not support slicing in CPython — only sq_item (int index)."""

        def fn(x):
            d = collections.deque([x, x + 1, x + 2])
            return operator.getitem(d, slice(0, 2))

        x = torch.randn(4)
        with self.assertRaises(TypeError):
            self._compile(fn, x)

    # TODO: deque int index works but through the wrong dispatch path.
    # CPython: PyObject_GetItem Branch 2 → _PyIndex_Check(key) → PyNumber_AsSsize_t → sq_item.
    # Dynamo: inherited BaseListVariable.mp_subscript_impl (Branch 1 path).
    # Result is correct, dispatch path diverges. Fix when sq_item branch is implemented.
    def test_deque_int_index(self):
        def fn(x):
            d = collections.deque([x, x + 1, x + 2])
            return operator.getitem(d, 1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # GAP 2: dict_subscript calls _PyObject_HashFast → TypeError for unhashable keys.
    # TODO: ConstDictVariable.mp_subscript_impl should check tp_hash and raise TypeError
    # for unhashable keys, matching CPython's dict_subscript (Objects/dictobject.c:3680).
    @unittest.expectedFailure
    def test_dict_unhashable_key(self):
        """dict[unhashable] should raise TypeError, not KeyError or silent failure."""

        def fn(x):
            d = {0: x, 1: x + 1}
            return operator.getitem(d, [0])

        x = torch.randn(4)
        with self.assertRaises(TypeError):
            self._compile(fn, x)

    # TODO: str/bytes subscript works via constant fold fallback (base mp_subscript_impl
    # raises Unsupported → _make_handler → operator.getitem("hello", 0) evaluates at
    # Python level), not via a proper mp_subscript_impl override mirroring CPython's
    # unicode_subscript / bytes_subscript. Should add dedicated overrides on
    # ConstantVariable to match CPython's dispatch path.
    # CPython: Objects/unicodeobject.c:13809 (unicode_subscript)
    # CPython: Objects/bytesobject.c (bytes_subscript)

    def test_str_subscript(self):
        def fn(x):
            s = "hello"
            c = operator.getitem(s, 0)
            return x + len(c)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_bytes_subscript(self):
        def fn(x):
            b = b"hello"
            val = operator.getitem(b, 0)
            return x + val

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # ===================================================================
    # Explicit __getitem__ dunder call path tests
    # Exercises: obj.__getitem__(key) → LOAD_ATTR + CALL, which may
    # route through call_method → mp_subscript_impl rather than
    # vt_getitem → mp_subscript_impl.
    # ===================================================================

    def test_list_dunder_getitem(self):
        def fn(x):
            items = [x, x + 1, x + 2]
            return items.__getitem__(1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_dict_dunder_getitem(self):
        def fn(x):
            d = {"a": x, "b": x + 1}
            return d.__getitem__("a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_user_defined_dict_missing_dunder_getitem(self):
        """__missing__ fallback must work via __getitem__ method call, not just operator.getitem."""

        class MyDict(dict):
            def __missing__(self, key):
                return 42

        def fn(x):
            d = MyDict(a=1)
            return x + d.__getitem__("b")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
