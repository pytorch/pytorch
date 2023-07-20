# Owner(s): ["module: dynamo"]
# flake8: noqa
import collections
import functools
import inspect
import itertools
import operator
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple
from unittest.mock import patch

import torch

import torch._dynamo.test_case
import torch._dynamo.testing
from torch import sub
from torch._dynamo.testing import expectedFailureDynamic, requires_numpy_pytorch_interop
from torch._dynamo.utils import same
from torch.nn import functional as F
from torch.testing._internal.common_utils import (
    disable_translation_validation_if_dynamic_shapes,
)

d = torch.ones(10, 10)
e = torch.nn.Linear(10, 10)
flag = True


clip01 = functools.partial(torch.clip, min=0.0, max=1.0)


def constant3(a, b):
    return a - b + (1.0 + 2)


def func_with_default(a, b, some_default_arg=True):
    if some_default_arg:
        return a - b


def make_test(fn):
    nargs = len(inspect.signature(fn).parameters)

    def test_fn(self):
        return torch._dynamo.testing.standard_test(self, fn=fn, nargs=nargs)

    return test_fn


@torch.jit.script_if_tracing
def inline_script_if_tracing(x):
    return x + 1.2


@torch.jit.ignore
def inline_ignore(x):
    return x + 3.4


@torch.jit.unused
def inline_unused(x):
    return x + 5.6


class FunctionTests(torch._dynamo.test_case.TestCase):
    @make_test
    def test_inline_jit_annotations(x):
        x = inline_script_if_tracing(x)
        x = inline_ignore(x)
        x = inline_unused(x)
        return

    @make_test
    def test_add(a, b):
        return a + b

    @make_test
    def test_add_(a, b):
        a_copy = torch.tensor(a)
        return a_copy.add_(b, alpha=5.0)

    @make_test
    def test_addcdiv(a, b, c):
        # dynamo decomposes this to avoid a graph break when
        # the value kwarg is populated
        return torch.addcdiv(a, b, c, value=5.0)

    @make_test
    def test_addcdiv_(a, b, c):
        a_copy = torch.tensor(a)
        return a_copy.addcdiv_(b, c, value=5.0)

    @make_test
    def test_is_not_null(a, b):
        if a is not None and b is not None:
            return a + b

    @make_test
    def test_functools_partial(a, b):
        return clip01(a + b)

    @make_test
    def test_itertools_product(a, b):
        v = a
        for x, i in itertools.product([a, b], [1, 2]):
            v = v + x * i
        return v

    @make_test
    def test_itertools_chain(a, b):
        v = a
        for x in itertools.chain([a, b], [1, 2]):
            v = v + x
        return v

    @make_test
    def test_itertools_combinations(a, b):
        combs = []
        for size in itertools.combinations((1, 2, 3, 4), 2):
            combs.append(torch.ones(size))
        return combs

    @make_test
    def test_constant1(a, b, c):
        return a - b * c + 1.0

    @make_test
    def test_constant2(a, b, c):
        return a - b * c + 1

    @make_test
    def test_constant3(a):
        b = 1
        c = 2
        d = 3
        return b + c - d + a

    @make_test
    def test_constant4(a, b):
        c = 2
        d = 3
        if c > d:
            return a - b
        return b - a

    @make_test
    def test_finfo(a, b):
        if torch.iinfo(torch.int32).bits == 32:
            return torch.finfo(a.dtype).min * b

    @make_test
    def test_globalfn(a, b):
        return sub(a, b)

    @make_test
    def test_viatorch(a, b):
        return torch.sub(a, b)

    @make_test
    def test_viamethod(a, b):
        return a.sub(b)

    @make_test
    def test_indirect1(a, b):
        t = a.sub
        return t(b)

    @make_test
    def test_indirect2(a, b):
        t = a.sub
        args = (b,)
        return t(*args)

    @make_test
    def test_indirect3(a, b):
        t = a.sub
        args = (b,)
        kwargs = {}
        return t(*args, **kwargs)

    @make_test
    def test_methodcall1(a, b, c):
        return constant3(a, b) * c

    @make_test
    def test_methodcall2(a, b):
        return constant3(a=b, b=a) + 1

    @make_test
    def test_methodcall3(a, b):
        return constant3(a, b=1.0) + b

    @make_test
    def test_device_constant(a):
        return a + torch.ones(1, device=torch.device("cpu"))

    @make_test
    def test_tuple1(a, b):
        args = (a, b)
        return sub(*args)

    @make_test
    def test_tuple2(a, b):
        args = [a, b]
        return sub(*args)

    @make_test
    def test_is_in_onnx_export(x, y):
        if torch.onnx.is_in_onnx_export():
            return x - 1
        else:
            return y + 1

    @make_test
    def test_is_fx_tracing(x, y):
        if torch.fx._symbolic_trace.is_fx_tracing():
            return x - 1
        else:
            return y + 1

    @make_test
    def test_listarg1(a, b):
        return torch.cat([a, b])

    @make_test
    def test_listarg2(a, b):
        return torch.cat((a, b), dim=0)

    @make_test
    def test_listarg3(a, b):
        kwargs = {"tensors": (a, b), "dim": 0}
        return torch.cat(**kwargs)

    @make_test
    def test_listarg4(a, b):
        return torch.cat(tensors=[a, b], dim=0)

    @make_test
    def test_listarg5(a, b):
        args = [(a, b)]
        kwargs = {"dim": 0}
        return torch.cat(*args, **kwargs)

    @make_test
    def test_deque(a, b):
        d = collections.deque([a, b])
        d.append(a + 1)
        d.extend([a, b])
        d.insert(0, "foo")
        tmp = d.pop()

        another_deque = collections.deque([tmp])
        d.extendleft(another_deque)
        another_deque.clear()
        d.extend(another_deque)

        d[2] = "setitem"
        d = d.copy()
        d.append(d.popleft())

        empty = collections.deque()
        d.extend(empty)

        # dynamo same() util doesn't support deque so just return a list
        return list(d)

    @make_test
    def test_slice1(a):
        return a[5]

    @make_test
    def test_slice2(a):
        return a[:5]

    @make_test
    def test_slice3(a):
        return a[5:]

    @make_test
    def test_slice4(a):
        return a[2:5]

    @make_test
    def test_slice5(a):
        return a[::2]

    @make_test
    def test_slice6(a):
        return torch.unsqueeze(a, 0)[:, 2:]

    @make_test
    def test_range1(a):
        return torch.tensor(range(a.size(0)))

    @make_test
    def test_range2(x, y):
        r = x + y
        for i in range(x.size(0) + 2):
            r = r / y
        return r

    @make_test
    def test_unpack1(a):
        a, b = a[:5], a[5:]
        return a - b

    @make_test
    def test_unpack2(a):
        packed = [a[:5], a[5:]]
        a, b = packed
        return a - b

    @make_test
    def test_unpack3(a):
        packed = (a[:5], a[5:])
        a, b = packed
        return a - b

    @make_test
    def test_fn_with_self_set(a, b):
        # avg_pool2d is an odd one with __self__ set
        return F.avg_pool2d(
            torch.unsqueeze(a, 0) * torch.unsqueeze(b, 1), kernel_size=2, padding=1
        )

    @make_test
    def test_return_tuple1(a, b):
        return (a - b, b - a, a, b)

    @make_test
    def test_globalvar(a, b):
        return a - b + d

    @make_test
    def test_globalmodule(x):
        return e(x)

    @make_test
    def test_inline_with_default(a, b, c):
        return func_with_default(a, b) * c

    @make_test
    def test_inner_function(x):
        def fn(x):
            return torch.add(x, x)

        return fn(x)

    @make_test
    def test_transpose_for_scores(x):
        new_x_shape = x.size()[:-1] + (2, 5)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1)

    @make_test
    def test_return_tuple2(x):
        return (torch.add(x, x), x)

    @make_test
    def test_load_global_bool(x):
        if flag:
            return torch.add(x, x)
        else:
            return x

    @make_test
    def test_len_tensor(x):
        z = len(x)
        return torch.add(x, z)

    @make_test
    def test_len_constant_list(x):
        z = len([1, 2, 3])
        return torch.add(x, z)

    @make_test
    def test_len_constant_dict(x):
        z = len({"foo": "bar"})
        return torch.add(x, z)

    @make_test
    def test_dict_copy(x):
        z = dict({"foo": x + 1})
        return z

    @make_test
    def test_callable_lambda(x):
        if callable(lambda x: True):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_callable_torch(x):
        if callable(torch.abs):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_callable_builtin(x):
        if callable(sum):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_len_constant_misc_iterables(x):
        a = len((1, 2, 3))
        b = len("test str")
        c = a + b
        return torch.add(x, c)

    @make_test
    def test_dict_kwargs(x):
        z = dict(text_embed=x + 1, other=x + 2)
        return z

    @make_test
    def test_ordered_dict_kwargs(x):
        z = collections.OrderedDict(sample=torch.ones(10))
        return z

    @make_test
    def test_float(x):
        y = float(1.2)
        y += float("1.2")
        return torch.add(x, y)

    @make_test
    def test_is_floating_point(x):
        y = x + 1
        return torch.is_floating_point(y), torch.is_floating_point(input=y)

    @make_test
    def test_dtype(x):
        if x.dtype == torch.float32:
            return x + 1

    @make_test
    def test_get_default_dtype(x):
        if x.dtype == torch.get_default_dtype():
            return x + 1
        else:
            return x - 1

    @make_test
    def test_get_autocast_gpu_dtype(x):
        dtype = torch.get_autocast_gpu_dtype()
        return x.type(dtype)

    @make_test
    def test_get_calculate_correct_fan(x):
        fan_in = torch.nn.init._calculate_correct_fan(x, "fan_in")
        return x + fan_in

    @make_test
    def test_is_complex(x):
        if torch.is_complex(x):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_get_privateuse1_name(x):
        if torch._C._get_privateuse1_backend_name() == "privateuseone":
            return x + 1
        else:
            return x - 1

    @make_test
    def test_device(x):
        if not x.is_cuda:
            return x + 1

    @make_test
    def test_tensor_type(a, b):
        m = a.to(torch.float16)
        return b.type(m.type())

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @make_test
    def test_tensor_type2(a, b):
        m = a.to("cuda")
        return m + b.type(m.type())

    @make_test
    def test_tensor_type3(a, b):
        m = a.type(torch.HalfTensor)
        return b.type(m.type())

    @make_test
    def test_tensor_type4(a, b):
        m = a.type("torch.HalfTensor")
        return b.type(m.type())

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @make_test
    def test_tensor_type5(a, b):
        m = a.type(torch.cuda.HalfTensor)
        return b.type(m.type())

    @make_test
    def test_ndim(x):
        if x.ndim == 2 and x.ndimension() == 2 and x.dim() == 2:
            return x + 1

    @make_test
    def test_T(x):
        return torch.ones_like(x.T)

    @make_test
    def test_mT(x):
        return torch.ones_like(x.mT)

    @make_test
    def test_is_sparse(x):
        if not x.is_sparse:
            return x + 1

    @make_test
    def test_shape1(x):
        if x.shape[0] == 10:
            return x + 1

    @make_test
    def test_shape2(x):
        if x.size(1) == 10:
            return x + 1

    @make_test
    def test_del(a, b):
        c = a + 1
        d = c + 2
        del c, a
        return b + d

    @make_test
    def test_chunks1(x):
        chunk_size = 5
        assert x.shape[0] % chunk_size == 0
        assert x.shape[0] // chunk_size == 2
        return x[:chunk_size] - x[chunk_size:]

    @make_test
    def test_import1(x, y):
        import torch
        from torch import sub

        return sub(torch.add(x, y), y)

    @make_test
    def test_return_dict(x, y):
        z = [x + y, y, False]
        return {"x": x, "z": z, "a": x, "b": z, "c": x}

    @make_test
    def test_return_dict2(x, y):
        tmp = {"x": x}
        tmp["z"] = [x + y, y]
        tmp["y"] = y
        tmp["z"].append(False)
        return tmp

    @make_test
    def test_funcdef_closure(x, y):
        x = x + y + 1.0

        def inner(z):
            nonlocal x, y
            y = x + z + 20.0
            x = y + z + 10.0

        inner(2.0)
        inner(3.0)

        return x, y

    @make_test
    def test_module_constant(x, y):
        r = x + y
        for i in range(torch._dynamo.testing.three):
            r = r / y
        return r

    @make_test
    def test_inline_softmax(x, y):
        # This is common in sme huggingface models
        return torch.nn.Softmax(dim=-1)(x + y * 2)

    @make_test
    def test_dtype_compare(a, b):
        if a.dtype == torch.float16:
            return a + 10
        if a.dtype == torch.float32:
            return a - b * 32

    @make_test
    def test_build_list_unpack(a, b):
        it1 = (x + 1 for x in (a, b))
        it2 = (x - 1 for x in (a, b))
        return torch.cat([*it1, *it2], dim=-1)

    @make_test
    def test_tensor_len(a, b):
        return a + b + len(a) + b.__len__()

    @make_test
    def test_pop(a, b):
        ll = [a, b]
        ll.append(a + 1)
        ll.extend(
            [
                b + 2,
                a + b,
            ]
        )
        ll.pop(-1)
        ll.pop(0)
        ll.pop()
        v1, v2 = ll
        return v1 - v2

    @make_test
    def test_list_convert(a, b):
        ll = [a + 2, b]
        ll = tuple(ll)
        tmp = b + 3
        ll = list(ll)
        v1, v2 = ll
        return v1 - v2 + tmp

    @make_test
    def test_list_add(a, b):
        l1 = (a, b)
        l2 = ()  # being a LOAD_CONST in the bytecode
        l3 = l1 + l2
        return l3[0] + l3[1]

    @make_test
    def test_startswith(a, b):
        x = a + b
        if "foobar".startswith("foo") and "test" in constant3.__module__:
            x = x + 1
        return x

    @make_test
    def test_dict_ops(a, b):
        tmp = {"a": a + 1, "b": b + 2}
        v = tmp.pop("b") + tmp.get("a") + tmp.get("missing", 3) + tmp.pop("missing", 4)
        tmp.update({"d": 3})
        tmp["c"] = v + tmp["d"]
        if "c" in tmp and "missing" not in tmp:
            return tmp["c"] - tmp["a"] + len(tmp)

    def test_dict_param_keys(self):
        a_param = torch.nn.Parameter(torch.ones([4, 4]))

        def fn(a):
            tmp = {"a": a, a_param: 3}
            return tmp["a"] + tmp[a_param]

        test = make_test(fn)
        test(self)

    def _test_default_dict_helper(self, factory):
        dd = collections.defaultdict(factory)
        param = torch.nn.Parameter(torch.ones([2, 2]))

        def fn(x):
            dd["a"] = x + 1
            dd[param] = 123
            dd["c"] = x * 2
            return dd["b"], dd

        x = torch.randn(10, 10)
        ref = fn(x)
        opt_fn = torch._dynamo.optimize_assert("eager")(fn)
        res = opt_fn(x)

        self.assertTrue(same(ref[0], res[0]))
        self.assertTrue(same(ref[1]["a"], res[1]["a"]))
        self.assertTrue(same(ref[1]["c"], res[1]["c"]))
        self.assertTrue(same(ref[1][param], res[1][param]))

    def test_default_dict(self):
        self._test_default_dict_helper(dict)

    def test_default_dict_lambda(self):
        self._test_default_dict_helper(lambda: dict())

    def test_default_dict_closure(self):
        def factory():
            return dict()

        self._test_default_dict_helper(factory)

    def test_default_dict_constr(self):
        param = torch.nn.Parameter(torch.ones([2, 2]))

        def fn(x):
            dd = collections.defaultdict(lambda: dict())
            dd["a"] = x + 1
            dd[param] = 123
            dd["c"] = x * 2
            return dd["b"], dd

        x = torch.randn(10, 10)
        ref = fn(x)
        opt_fn = torch._dynamo.optimize_assert("eager")(fn)
        res = opt_fn(x)

        self.assertTrue(same(ref[0], res[0]))
        self.assertTrue(same(ref[1]["a"], res[1]["a"]))
        self.assertTrue(same(ref[1]["c"], res[1]["c"]))
        self.assertTrue(same(ref[1][param], res[1][param]))

    @make_test
    def test_call_dict1(x):
        d1 = dict()
        d1["x"] = x + 1
        d2 = collections.OrderedDict()
        d2["x"] = x + 2
        return d1["x"] + d2["x"] + 1

    @make_test
    def test_call_dict2(x):
        d1 = dict()
        d1["x"] = x
        d2 = collections.OrderedDict(d1)
        if isinstance(d2, collections.OrderedDict):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_call_dict3(x):
        my_list = [("a", x), ("b", x + 1), ("c", x + 2)]
        d1 = dict(my_list)
        d1["a"] = x + 10
        d2 = collections.OrderedDict(my_list)
        d2["c"] = x + 20
        return d1["a"] + d2["c"] + 1

    @make_test
    def test_call_dict4(x):
        my_list = (("a", x), ("b", x + 1), ("c", x + 2))
        d1 = dict(my_list)
        d1["a"] = x + 10
        d2 = collections.OrderedDict(my_list)
        d2["c"] = x + 20
        return d1["a"] + d2["c"] + 1

    @make_test
    def test_call_dict5(x):
        my_list = iter([("a", x), ("b", x + 1), ("c", x + 2)])
        d1 = dict(my_list)
        d1["a"] = x + 10
        d2 = collections.OrderedDict(my_list)
        d2["c"] = x + 20
        return d1["a"] + d2["c"] + 1

    @make_test
    def test_min_max(a, b):
        c = a + b
        a = a.sum()
        b = b.sum()
        a = min(max(a, 0), 1)
        b = max(0, min(1, b))
        return max(a, b) - min(a, b) + c

    @make_test
    def test_map_sum(a, b, c, d):
        return sum(map(lambda x: x + 1, [a, b, c, d]))

    @make_test
    def test_reduce(a, b, c, d):
        return functools.reduce(operator.add, [a, b, c, d])

    @make_test
    def test_tuple_contains(a, b):
        v1 = "a"
        v2 = "b"
        v3 = "c"
        vals1 = (v1, v2, v3)
        vals2 = ("d", "e", "f")
        if "a" in vals1 and "b" not in vals2:
            return a + b
        return a - b

    @make_test
    def test_tuple_iadd(a, b):
        output = (a, b)
        output += (a + b, a - b)
        return output

    @make_test
    def test_unpack_ex1(x):
        output = (x, x + 1, x + 2, x + 3)
        a, b, *cd = output
        return a - b / cd[0]

    @make_test
    def test_unpack_ex2(x):
        output = (x, x + 1, x + 2, x + 3)
        *ab, c, d = output
        return c - d / ab[0]

    @make_test
    def test_unpack_ex3(x):
        output = (x, x + 1, x + 2, x + 3)
        a, *bc, d = output
        return a - d / bc[0]

    @make_test
    def test_const_tuple_add1(x):
        output = (x, x + 1, x + 2, x + 3)
        output = () + output + ()
        return output[2] + output[3]

    @make_test
    def test_const_tuple_add2(x):
        output = (x, x + 1, x + 2, x + 3)
        output = (None,) + output + (None,)
        return output[2] + output[3]

    @make_test
    def test_list_truth(a, b):
        tmp = [1, 2, 3]
        if tmp:
            return a + b
        else:
            return a - b

    @make_test
    def test_list_reversed(a, b):
        tmp = [a + 1, a + 2, a + 3]
        return a + b + next(iter(reversed(tmp)))

    @make_test
    def test_list_sorted1(x):
        tmp = [1, 10, 3, 0]
        return x + 1, sorted(tmp), sorted(tmp, reverse=True)

    @make_test
    def test_list_sorted2(x):
        y = [
            ("john", "A", 8),
            ("jane", "B", 5),
            ("dave", "B", 10),
        ]
        return (
            x + 1,
            sorted(y),
            sorted(y, key=lambda student: student[2]),
            sorted(y, key=lambda student: student[2], reverse=True),
        )

    @make_test
    def test_tuple_sorted(x):
        tmp = (1, 10, 3, 0)
        return x + 1, sorted(tmp), sorted(tmp, reverse=True)

    @make_test
    def test_dict_sorted(x):
        tmp = {1: "D", 10: "B", 3: "E", 0: "F"}
        return x + 1, sorted(tmp), sorted(tmp, reverse=True)

    @make_test
    def test_list_clear(a, b):
        tmp = [a + 1, a + 2]
        tmp.clear()
        tmp.append(a + b)
        return tmp

    @make_test
    def test_not_list(a):
        return not [a + 1]

    @make_test
    def test_islice_chain(a, b):
        tmp1 = [a + 1, a + 2]
        tmp2 = [a + 3, a + 4]
        a, b = list(itertools.islice(itertools.chain(tmp1, tmp2), 1, 3))
        c = next(itertools.islice(tmp1, 1, None))
        return a - b / c

    @make_test
    def test_namedtuple(a, b):
        mytuple = collections.namedtuple("mytuple", ["x", "y", "xy"])
        tmp = mytuple(a, b, a + b)
        return mytuple(tmp.x, tmp[1], tmp.xy + b)

    @make_test
    def test_namedtuple_defaults(a, b):
        mytuple = collections.namedtuple(
            "mytuple", ["x", "y", "xy"], defaults=(None, 1, None)
        )
        tmp = mytuple(a, xy=b)
        return mytuple(tmp.x, tmp[1], tmp.xy + b)

    class MyNamedTuple(NamedTuple):
        first: torch.Tensor
        second: torch.Tensor

        def add(self) -> torch.Tensor:
            return self.first + self.second

        @staticmethod
        def static_method() -> int:
            return 1

        @classmethod
        def class_method(cls) -> str:
            return cls.__name__

    @make_test
    def test_namedtuple_user_methods(a, b):
        mytuple = FunctionTests.MyNamedTuple(a, b)
        return mytuple.add(), mytuple.static_method(), mytuple.class_method()

    @make_test
    def test_is_quantized(a, b):
        if not a.is_quantized:
            return a + b

    @make_test
    def test_fstrings1(a, b):
        x = 1.229
        tmp = f"{x:.2f} bar"
        if tmp.startswith("1.23"):
            return a + b

    # https://github.com/pytorch/pytorch/issues/103602
    @expectedFailureDynamic
    @make_test
    def test_fstrings2(x):
        tmp = f"{x.shape[0]} bar"
        if tmp.startswith("10"):
            return x + 1

    @make_test
    def test_fstrings3(x):
        tmp = f"{x.__class__.__name__} foo"
        if tmp.startswith("Tensor"):
            return x + 1

    @make_test
    def test_tensor_new_with_size(x):
        y = torch.rand(5, 8)
        z = x.new(y.size())
        assert z.size() == y.size()

    @make_test
    def test_tensor_new_with_shape(x):
        y = torch.rand(5, 8)
        z = x.new(y.shape)
        assert z.size() == y.size()

    @make_test
    def test_jit_annotate(x):
        y = torch.jit.annotate(Any, x + 1)
        return y + 2

    @expectedFailureDynamic
    @make_test
    def test_is_contiguous_memory_format(tensor):
        if torch.jit.is_scripting():
            return None
        elif tensor.is_contiguous(memory_format=torch.contiguous_format):
            return tensor + 1

    @make_test
    def test_list_slice_assignment(x):
        m = [1, 2, 3, 4]
        m[1:] = [6] * (len(m) - 1)
        return x + 1

    @make_test
    def test_distributed_is_available(x):
        if torch.distributed.is_available():
            return x + 1
        else:
            return x - 1

    @unittest.skipIf(
        not torch.distributed.is_available(), "requires distributed package"
    )
    @make_test
    def test_distributed_is_initialized(x):
        if torch.distributed.is_initialized():
            return x + 1
        else:
            return x - 1

    @disable_translation_validation_if_dynamic_shapes
    @make_test
    def test_torch_distributions_functions(x):
        normal = torch.distributions.Normal(x, torch.tensor(1))
        independent = torch.distributions.Independent(normal, 1)
        return independent.log_prob(x)

    @make_test
    def test_context_wrapping_nested_functions_no_closure(x):
        @torch.no_grad()
        def augment(x: torch.Tensor) -> torch.Tensor:
            return (x + 1) * 2

        return augment(x)

    # # This is to test the new syntax for pattern matching
    # # ("match ... case ...") added on python 3.10.
    # # Uncomment these test cases if you run on 3.10+
    # @make_test
    # def test_match_sequence(a):
    #     point = (5, 8)
    #     match point:
    #         case (0, 0):
    #             return a
    #         case (0, y):
    #             return a - y
    #         case (x, 0):
    #             return a + x
    #         case (x, y):
    #             return a + x - y

    # @make_test
    # def test_match_mapping_and_match_keys(x):
    #     param = {"a": 0.5}
    #     match param:
    #         case {"a": param}:
    #             return x * param
    #         case {"b": param}:
    #             return x / param

    @requires_numpy_pytorch_interop
    @make_test
    def test_numpy_meshgrid(x, y):
        import numpy as np

        r1, r2 = np.meshgrid(x.numpy(), y.numpy())
        return torch.from_numpy(r1), torch.from_numpy(r2)

    @requires_numpy_pytorch_interop
    @make_test
    def test_torch_from_numpy(x):
        a = x.numpy()
        b = torch.from_numpy(a)
        if b.size(0) == 1:
            return torch.tensor(True)
        else:
            return torch.tensor(False)

    @requires_numpy_pytorch_interop
    @make_test
    def test_numpy_size(x):
        a = x.numpy()
        return a.size

    @requires_numpy_pytorch_interop
    @make_test
    def test_numpy_attributes(x):
        a = x.numpy()
        return (
            a.itemsize,
            a.strides,
            a.shape,
            a.ndim,
            a.size,
            torch.from_numpy(a.T),
            torch.from_numpy(a.real),
            torch.from_numpy(a.imag),
        )

    @requires_numpy_pytorch_interop
    @make_test
    def test_mean_sum_np(x: torch.Tensor):
        import numpy as np

        x_mean = np.mean(x.numpy(), 1)
        x_sum = np.sum(x_mean)
        x_sum_array = np.asarray(x_sum)
        return torch.from_numpy(x_sum_array)

    @requires_numpy_pytorch_interop
    @make_test
    def test_return_numpy_ndarray(x):
        a = x.numpy()
        return a.T

    @requires_numpy_pytorch_interop
    @make_test
    def test_return_multiple_numpy_ndarray(x):
        a = x.numpy()
        return a.T, a.imag, a.real

    @requires_numpy_pytorch_interop
    @make_test
    def test_ndarray_method(x):
        a = x.numpy()
        return a.copy()

    @requires_numpy_pytorch_interop
    @make_test
    def test_ndarray_transpose(x):
        a = x.numpy()
        return a.transpose(0, 1)

    @requires_numpy_pytorch_interop
    @make_test
    def test_ndarray_reshape(x):
        a = x.numpy()
        return a.reshape([1, a.size])

    @requires_numpy_pytorch_interop
    @make_test
    def test_ndarray_methods_returning_scalar(x):
        a = x.numpy()
        return a.max(axis=0), a.all(axis=0)

    @requires_numpy_pytorch_interop
    @make_test
    def test_ndarray_builtin_functions(x):
        a = x.numpy()
        return a + a, a - a

    @requires_numpy_pytorch_interop
    @make_test
    def test_numpy_dtype_argument_to_function(x):
        import numpy as np

        return np.ones_like(x, dtype=np.float64)

    @requires_numpy_pytorch_interop
    @make_test
    def test_numpy_linalg(x):
        import numpy as np

        return np.linalg.norm(x.numpy(), axis=0)

    @requires_numpy_pytorch_interop
    @make_test
    def test_numpy_fft(x):
        import numpy as np

        return np.fft.fftshift(x.numpy())

    @requires_numpy_pytorch_interop
    @make_test
    def test_numpy_random():
        import numpy as np

        x = np.random.randn(2, 2)
        return x - x


def global_func_with_default_tensor_args(
    x=torch.zeros((2, 2)), *, kw_x=torch.zeros((1, 2))
):
    x.add_(1)
    kw_x.add_(1)
    return x, kw_x


class ModuleWithDefaultTensorArgsMethod(torch.nn.Module):
    def forward(self, x=torch.zeros((2, 2)), *, kw_x=torch.zeros((1, 2))):
        x.add_(1)
        kw_x.add_(1)
        return x, kw_x


class WrapperModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = ModuleWithDefaultTensorArgsMethod()

    def forward(self):
        return self.m()


class DefaultsTests(torch._dynamo.test_case.TestCase):
    def test_func_default_tensor_args(self):
        """
        Tests that we indeed reference (and mutate) "the one" default tensor arg
        stored on the globally allocated function object, both from the orig and
        compiled function
        """

        def func():
            return global_func_with_default_tensor_args()

        cnts = torch._dynamo.testing.CompileCounter()
        compiled_func = torch.compile(func, backend=cnts)
        for i in range(4):
            if i % 2 == 0:
                x, kw_x = func()
            else:
                x, kw_x = compiled_func()
            # the inner func mutates += 1 each call
            self.assertTrue(same(x, torch.ones_like(x) + i))
            self.assertTrue(same(kw_x, torch.ones_like(kw_x) + i))
        # Calling compiled_func twice does not recompile
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

        # But with a change to the guarded default tensor, we do recompile
        with patch.object(
            global_func_with_default_tensor_args,
            "__defaults__",
            (torch.ones((3, 4, 5)),),
        ):
            x, kw_x = compiled_func()
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

        with patch.object(
            global_func_with_default_tensor_args,
            "__kwdefaults__",
            {"kw_x": torch.ones((3, 4, 5))},
        ):
            x, kw_x = compiled_func()
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 6)

    def test_meth_default_tensor_args(self):
        """
        Tests that we indeed reference (and mutate) "the one" default tensor arg
        stored on the globally allocated function object, both from the orig and
        compiled function
        """
        mod = WrapperModule()
        cnts = torch._dynamo.testing.CompileCounter()
        compiled_mod = torch.compile(mod, backend=cnts)
        for i in range(4):
            if i % 2 == 0:
                x, kw_x = mod()
            else:
                x, kw_x = compiled_mod()
            # the inner func mutates += 1 each call
            self.assertTrue(same(x, torch.ones_like(x) + i))
            self.assertTrue(same(kw_x, torch.ones_like(kw_x) + i))
        # Calling compiled_func twice does not recompile
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

        # But with a change to the guarded default tensor, we do recompile
        with patch.object(
            ModuleWithDefaultTensorArgsMethod.forward,
            "__defaults__",
            (torch.ones((3, 4, 5)),),
        ):
            x, kw_x = compiled_mod()
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

        with patch.object(
            ModuleWithDefaultTensorArgsMethod.forward,
            "__kwdefaults__",
            {"kw_x": torch.ones((3, 4, 5))},
        ):
            x, kw_x = compiled_mod()
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 6)

    def test_func_default_torch_args(self):
        """
        Tests other types of torch types as function default (size, dtype, device)
        """

        def func_with_default_torch_args(
            dt=torch.float16, ds=torch.Size((1, 2, 3)), dd=torch.device("cpu")
        ):
            return torch.ones(ds, dtype=dt, device=dd)

        def func():
            return func_with_default_torch_args()

        cnts = torch._dynamo.testing.CompileCounter()
        compiled_func = torch.compile(func, backend=cnts)
        out = func()
        compiled_out = compiled_func()
        self.assertEqual(out.dtype, compiled_out.dtype)
        self.assertEqual(out.device, compiled_out.device)
        self.assertEqual(out.size(), compiled_out.size())
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

    def test_dataclass_factory(self):
        @dataclass
        class Output:
            scalar: int = 2
            named_tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
            lists: List[torch.Tensor] = field(default_factory=list)

            def scale(self):
                return self.scalar * 2

        def fn(x):
            # Check default dict assignment
            a = Output(1)
            # Check that dataclass methods can be inlined
            scaled_value = a.scale()

            # Check that normal assignment works
            b = Output(5, named_tensors={"x": x})

            # Check default int assignment
            c = Output()

            # Check that the default members are properly initialized
            if isinstance(a.named_tensors, dict):
                x = torch.sin(x)

            # Change dataclass
            c.scalar = 6
            c.named_tensors["x"] = x

            # Return dataclaass as well to check reconstruction
            return c, torch.cos(x) * scaled_value + b.named_tensors["x"] + c.scalar

        cnts = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.randn(4)
        eager_dataclass, out = fn(x)
        compiled_dataclass, compiled_out = compiled_fn(x)
        self.assertEqual(eager_dataclass.scalar, compiled_dataclass.scalar)
        self.assertEqual(
            eager_dataclass.named_tensors["x"], compiled_dataclass.named_tensors["x"]
        )
        self.assertTrue(same(out, compiled_out))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 5)

    def test_dataclass_nested(self):
        @dataclass
        class Base:
            outer_a: int
            outer_b: int

        @dataclass
        class Derived(Base):
            inner_a: Any = field(default_factory=list)

        def fn(x):
            l = Derived(1, 2)
            return l.outer_a * x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        res = fn(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
