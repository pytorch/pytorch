# Owner(s): ["module: dynamo"]
"""Tests for vt_getitem: CPython PyObject_GetItem dispatch in Dynamo.

Tests are organized by dispatch branch:
  - Branch 1 (mp_subscript): list, tuple, range, size, dict, defaultdict, tensor, etc.
  - Branch 2 (sq_item via vt_sequence_getitem): deque (natural), reversed str/bytes
  - Branch 3 (__class_getitem__): type subscript
  - Explicit __getitem__ dunder calls

See TODO(follow-up) comments on each mp_subscript_impl override for remaining
CPython behavioral gaps.
"""

import collections
import operator
import types
import typing
import unittest

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.constant import ConstantVariable
from torch._dynamo.variables.lists import BaseListVariable, DequeVariable, RangeVariable
from torch._library.opaque_object import MemberType, OpaqueBase, register_opaque_type
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
            t = list[int]  # noqa: F841
            return x + 1

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_typing_subscript_dict(self):
        def fn(x):
            operator.getitem(typing.Dict, (str, int))  # noqa: UP006
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

    def test_getattr_dict_subscript(self):
        """obj.__dict__["key"] → GetAttrVariable → DunderDictVariable."""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.__dict__["_modules"]["linear"](x)

        model = Model()
        x = torch.randn(4)
        compiled = torch.compile(model, backend="eager")
        self.assertEqual(model(x), compiled(x))

    # --- TorchScriptObjectVariable ---

    def test_opaque_object_getitem(self):
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

    # ===================================================================
    # Branch 1 (mp_subscript): str/bytes
    # ConstantVariable.mp_subscript_impl for str/bytes.
    # CPython: Objects/unicodeobject.c:13809 (unicode_subscript)
    # CPython: Objects/bytesobject.c (bytes_subscript)
    # ===================================================================

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

    # ===================================================================
    # Branch 3: __class_getitem__ (type subscript)
    # Exercises: MyClass[int] → type.__getitem__ → __class_getitem__
    # In Python 3.10+, type.__getitem__ sets mp_subscript on type objects,
    # so this goes through Branch 1 of vt_getitem.
    # ===================================================================

    def test_class_getitem_builtin(self):
        """list[int] → GenericAlias via type.__getitem__."""

        def fn(x):
            alias = list[int]
            return x + len(alias.__args__)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_class_getitem_custom(self):
        """Custom class with __class_getitem__."""

        class MyGeneric:
            def __class_getitem__(cls, item):
                return item

        def fn(x):
            result = MyGeneric[int]
            return x + (1 if result is int else 0)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_not_subscriptable(self):
        """Non-subscriptable object should raise TypeError."""

        def fn(x):
            return operator.getitem(42, 0)

        x = torch.randn(4)
        with self.assertRaises(TypeError):
            torch.compile(fn, backend="eager")(x)

    # ===================================================================
    # Branch 2: sq_item via vt_sequence_getitem
    #
    # CPython's PyObject_GetItem branch 2: types with sq_item but no
    # mp_subscript go through _PyIndex_Check → PySequence_GetItem → sq_item.
    #
    # Sub-sections:
    #   (a) Natural dispatch — deque (only sq_item, no mp_subscript)
    #   (b) reversed() → sq_item — str/bytes lack __reversed__, so
    #       reversed() falls back to vt_sequence_getitem naturally
    # ===================================================================

    # --- (a) Natural dispatch: deque ---
    # CPython's deque only has sq_item (Modules/_collectionsmodule.c:1888),
    # not mp_subscript. vt_getitem dispatches to sq_item_impl directly.

    def test_deque_int_index(self):
        def fn(x):
            d = collections.deque([x, x + 1, x + 2])
            return operator.getitem(d, 1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_deque_negative_index(self):
        """vt_sequence_getitem wraps negative indices via sq_length before sq_item."""

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
        """deque[0:2] → TypeError — sq_item does not accept slices."""

        def fn(x):
            d = collections.deque([x, x + 1, x + 2])
            return operator.getitem(d, slice(0, 2))

        x = torch.randn(4)
        with self.assertRaises(TypeError):
            torch.compile(fn, backend="eager")(x)

    def test_deque_invalid_index_type(self):
        """deque['a'] → TypeError, matching CPython's branch 2 _PyIndex_Check."""

        def fn(x):
            d = collections.deque([x, x + 1, x + 2])
            return operator.getitem(d, "a")

        x = torch.randn(4)
        with self.assertRaises(TypeError):
            torch.compile(fn, backend="eager")(x)

    def test_deque_out_of_range(self):
        """deque[100] → IndexError('deque index out of range')."""

        def fn(x):
            d = collections.deque([x, x + 1, x + 2])
            return operator.getitem(d, 100)

        x = torch.randn(4)
        with self.assertRaises(IndexError):
            torch.compile(fn, backend="eager")(x)

    def test_deque_reversed(self):
        """reversed(deque) uses __reversed__ (deque has it since 3.8)."""

        def fn(x):
            d = collections.deque([x, x + 1, x + 2])
            items = list(reversed(d))
            return items[0]

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- (b) iter/list/tuple/reversed on deque ---
    # deque has __iter__, so these go through the normal iteration path
    # (unpack_var_sequence), not the sq_item fallback in builtin.py.

    def test_iter_deque(self):
        def fn(x):
            d = collections.deque([x, x + 1, x + 2])
            return list(iter(d))

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_list_deque(self):
        def fn(x):
            d = collections.deque([x, x + 1, x + 2])
            return list(d)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tuple_deque(self):
        def fn(x):
            d = collections.deque([x, x + 1, x + 2])
            return tuple(d)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- (d) generic_getiter sequence protocol fallback ---
    # User-defined class with __getitem__ + __len__ but no __iter__:
    # CPython gives it both mp_subscript and sq_item via slot wrappers.
    # generic_getiter (#178462) detects no tp_iter, falls back to
    # pysequence_check → sequence_iterator polyfill → __getitem__.

    def test_iter_seqonly(self):
        """iter(SeqOnly) → generic_getiter → sequence_iterator → __getitem__."""

        class SeqOnly:
            def __init__(self, items):
                self.items = items

            def __getitem__(self, i):
                return self.items[i]

            def __len__(self):
                return len(self.items)

        def fn(x):
            s = SeqOnly([1, 2, 3])
            total = 0
            for item in s:
                total += item
            return x + total

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_list_seqonly(self):
        """list(SeqOnly) → generic_getiter → sequence_iterator → __getitem__."""

        class SeqOnly:
            def __init__(self, items):
                self.items = items

            def __getitem__(self, i):
                return self.items[i]

            def __len__(self):
                return len(self.items)

        def fn(x):
            s = SeqOnly([1, 2, 3])
            return x + sum(list(s))

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tuple_seqonly(self):
        """tuple(SeqOnly) → generic_getiter → sequence_iterator → __getitem__."""

        class SeqOnly:
            def __init__(self, items):
                self.items = items

            def __getitem__(self, i):
                return self.items[i]

            def __len__(self):
                return len(self.items)

        def fn(x):
            s = SeqOnly([1, 2, 3])
            return x + sum(tuple(s))

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- (e) Sequence protocol fallback (in operator) ---
    # A user-defined class with __getitem__ + __len__ but no __iter__
    # exercises the sequence protocol in CPython and the __getitem__
    # fallback in Dynamo's CONTAINS_OP polyfill.

    def test_contains_getitem_fallback(self):
        """'in' operator works on types with __getitem__ but no __iter__."""

        class SeqOnly:
            def __init__(self, items):
                self.items = items

            def __getitem__(self, i):
                return self.items[i]

            def __len__(self):
                return len(self.items)

        def fn(x):
            s = SeqOnly([1, 2, 3])
            return x + (10 if 2 in s else 0)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_contains_getitem_fallback_missing(self):
        """'in' returns False when item is not found via __getitem__ iteration."""

        class SeqOnly:
            def __init__(self, items):
                self.items = items

            def __getitem__(self, i):
                return self.items[i]

            def __len__(self):
                return len(self.items)

        def fn(x):
            s = SeqOnly([1, 2, 3])
            return x + (10 if 99 in s else 0)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # ===================================================================
    # sq_item_impl override checks
    # These types have mp_subscript so Branch 1 always wins at runtime.
    # Verify the defensive sq_item_impl overrides exist and aren't the
    # base class fallback (which calls unimplemented).
    # ===================================================================

    def test_sq_item_impl_overrides(self):
        """All sequence types override sq_item_impl from the base class."""
        base = VariableTracker.sq_item_impl
        for cls in (BaseListVariable, RangeVariable, ConstantVariable, DequeVariable):
            self.assertIsNot(
                cls.sq_item_impl, base, f"{cls.__name__} must override sq_item_impl"
            )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
