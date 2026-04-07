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

    def test_tensor_negative_index(self):
        def fn(x):
            return operator.getitem(x, -1)

        x = torch.randn(4, 4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tensor_bool_index(self):
        def fn(x):
            return operator.getitem(x, True)

        x = torch.randn(4, 4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tensor_ellipsis_index(self):
        def fn(x):
            return operator.getitem(x, ...)

        x = torch.randn(4, 4)
        self.assertEqual(fn(x), self._compile(fn, x))

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
            operator.getitem(list, int)
            return x + 1

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_typing_subscript_dict(self):
        def fn(x):
            operator.getitem(dict, (str, int))
            return x + 1

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_typing_subscript_non_constant(self):
        """Subscripting a typing construct with a traced (non-constant) value should graph break."""

        def fn(x):
            return operator.getitem(list, x)

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

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

    def test_nn_module_list_negative_index(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(4, 4) for _ in range(3)]
                )

            def forward(self, x):
                return operator.getitem(self.layers, -1)(x)

        model = Model()
        x = torch.randn(4)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        self.assertEqual(model(x), compiled(x))

    def test_nn_sequential_slice(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seq = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(4, 4),
                )

            def forward(self, x):
                sub = operator.getitem(self.seq, slice(0, 2))
                return sub(x)

        model = Model()
        x = torch.randn(4)
        # Slice on nn.Module triggers convert_to_unspecialized → RestartAnalysis,
        # so don't use fullgraph=True.
        cnt = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(model, backend=cnt)
        self.assertEqual(model(x), compiled(x))
        self.assertGreaterEqual(cnt.frame_count, 1)

    def test_nn_module_dict_missing_key(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleDict({"fc": torch.nn.Linear(4, 4)})

            def forward(self, x):
                return operator.getitem(self.layers, "nonexistent")(x)

        model = Model()
        x = torch.randn(4)
        # ModuleDict["nonexistent"] raises KeyError at trace time, which
        # surfaces as Unsupported (observed exception, no handler in compiled fn).
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(model, backend="eager", fullgraph=True)(x)

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

    # --- GetAttrVariable (__dict__ access) ---

    def test_getattr_dict_getitem(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                layer = operator.getitem(self.__dict__["_modules"], "linear")
                return layer(x)

        model = Model()
        x = torch.randn(4)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        self.assertEqual(model(x), compiled(x))

    def test_getattr_dict_missing_key(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return operator.getitem(self.__dict__["_modules"], "nonexistent")(x)

        model = Model()
        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(model, backend="eager", fullgraph=True)(x)

    def test_getattr_dict_int_key(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return operator.getitem(self.__dict__, 0)

        model = Model()
        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(model, backend="eager", fullgraph=True)(x)

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

    # --- DequeVariable (sq_item path) ---
    # CPython's deque only has sq_item (Modules/_collectionsmodule.c:1888), not
    # mp_subscript. vt_getitem dispatches to sq_item_impl for deque.

    def test_deque_int_index(self):
        def fn(x):
            d = collections.deque([x, x + 1, x + 2])
            return operator.getitem(d, 1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_deque_negative_index(self):
        def fn(x):
            d = collections.deque([x, x + 1, x + 2])
            return operator.getitem(d, -1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_deque_bool_index(self):
        def fn(x):
            d = collections.deque([x, x + 1, x + 2])
            return operator.getitem(d, True)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_deque_index_via_index_dunder(self):
        class Idx:
            def __index__(self):
                return 2

        def fn(x):
            d = collections.deque([x, x + 1, x + 2])
            return operator.getitem(d, Idx())

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_deque_slice_should_reject(self):
        """deque does not support slicing in CPython — only sq_item (int index)."""

        def fn(x):
            d = collections.deque([x, x + 1, x + 2])
            return operator.getitem(d, slice(0, 2))

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    def test_deque_invalid_index_type(self):
        def fn(x):
            d = collections.deque([x, x + 1, x + 2])
            return operator.getitem(d, "a")

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    # --- Unhashable dict key (tp_hash check) ---
    # CPython: dict_subscript calls _PyObject_HashFast → TypeError for unhashable keys.
    # Dynamo: _HashableTracker checks tp_hash via C-level slot detection. Types with
    # tp_hash = PyObject_HashNotImplemented (list, set, dict) get a graph break with
    # a clear "unhashable dict key" message.

    def test_dict_unhashable_key(self):
        """dict[unhashable_list] should raise TypeError."""

        def fn(x):
            d = {0: x, 1: x + 1}
            return operator.getitem(d, [0])

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    def test_dict_unhashable_key_set(self):
        """dict[unhashable_set] should raise TypeError."""

        def fn(x):
            d = {0: x, 1: x + 1}
            return operator.getitem(d, {1, 2})

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    def test_dict_unhashable_key_dict(self):
        """dict[unhashable_dict] should raise TypeError."""

        def fn(x):
            d = {0: x, 1: x + 1}
            return operator.getitem(d, {"a": 1})

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    def test_user_defined_dict_unhashable_key(self):
        """dict subclass[unhashable_list] should raise TypeError."""

        class MyDict(dict):
            pass

        def fn(x):
            d = MyDict({0: x, 1: x + 1})
            return operator.getitem(d, [0])

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    # --- ConstantVariable (str) ---
    # CPython: unicode_subscript (Objects/unicodeobject.c:13809)

    def test_str_subscript(self):
        def fn(x):
            s = "hello"
            c = operator.getitem(s, 0)
            return x + len(c)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_str_slice(self):
        def fn(x):
            s = "hello"
            return x + len(operator.getitem(s, slice(1, 3)))

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_str_negative_index(self):
        def fn(x):
            s = "hello"
            return x + len(operator.getitem(s, -1))

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_str_invalid_key_type(self):
        """CPython: TypeError('string indices must be integers or slices, not float')."""

        def fn(x):
            return operator.getitem("hello", 1.5)

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    # --- ConstantVariable (bytes) ---
    # CPython: bytes_subscript (Objects/bytesobject.c)

    def test_bytes_subscript(self):
        def fn(x):
            b = b"hello"
            val = operator.getitem(b, 0)
            return x + val

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_bytes_slice(self):
        def fn(x):
            b = b"hello"
            return x + len(operator.getitem(b, slice(1, 3)))

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_bytes_negative_index(self):
        def fn(x):
            b = b"hello"
            return x + operator.getitem(b, -1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_bytes_invalid_key_type(self):
        """CPython: TypeError('bytes indices must be integers or slices, not float')."""

        def fn(x):
            return operator.getitem(b"hello", 1.5)

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

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

    # --- getitem_const self-validation tests ---
    # These verify that getitem_const validates keys internally (matching CPython
    # where each *_subscript function validates its own keys), so direct callers
    # that bypass mp_subscript_impl still get proper type checking.

    def test_list_getitem_const_rejects_float_key(self):
        """List getitem_const should reject float keys even without mp_subscript_impl."""

        def fn(x):
            items = [x, x + 1]
            return items[1.5]

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, torch.randn(4))

    def test_tuple_getitem_const_rejects_float_key(self):
        def fn(x):
            items = (x, x + 1)
            return items[1.5]

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, torch.randn(4))

    def test_range_getitem_const_rejects_float_key(self):
        def fn(x):
            r = range(10)
            return x + r[1.5]

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, torch.randn(4))

    def test_str_getitem_const_rejects_float_key(self):
        def fn(x):
            s = "hello"
            return x + len(s[1.5])

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, torch.randn(4))

    def test_bytes_getitem_const_rejects_float_key(self):
        def fn(x):
            b = b"hello"
            return x + b[1.5]

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, torch.randn(4))

    def test_list_getitem_const_accepts_index_dunder(self):
        """getitem_const should accept __index__-able keys via nb_index coercion."""

        class MyIndex:
            def __index__(self):
                return 1

        def fn(x):
            items = [x, x + 1, x + 2]
            return items[MyIndex()]

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
