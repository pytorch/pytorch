# Owner(s): ["module: dynamo"]
# flake8: noqa: E731, C405, F811, C418, C417
import collections
import functools
import inspect
import itertools
import math
import operator
import random
import sys
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple
from unittest.mock import patch

import numpy as np

import torch

import torch._dynamo.test_case
import torch._dynamo.testing
from torch import sub
from torch._dynamo.testing import (
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
    expectedFailureDynamic,
    normalize_gm,
)
from torch._dynamo.utils import ifdynstaticdefault, same
from torch._dynamo.variables import ConstantVariable
from torch._dynamo.variables.lists import RangeVariable

from torch.nn import functional as F
from torch.testing._internal.common_utils import (
    disable_translation_validation_if_dynamic_shapes,
    instantiate_parametrized_tests,
    parametrize,
)

# Defines all the kernels for tests
from torch.testing._internal.triton_utils import *  # noqa: F403

d = torch.ones(10, 10)
e = torch.nn.Linear(10, 10)
flag = True


class CustomDictSubclass(collections.OrderedDict):
    pass


clip01 = functools.partial(torch.clip, min=0.0, max=1.0)


def constant3(a, b):
    return a - b + (1.0 + 2)


_variable = 0


def update_global(x):
    global _variable
    _variable += 1
    # Check that updated global variable value is picked up
    return x * _variable


def func_with_default(a, b, some_default_arg=True):
    if some_default_arg:
        return a - b


def make_test(fn=None, expected_frame_count=1):
    if fn is None:
        return lambda fn: make_test(fn, expected_frame_count=expected_frame_count)

    nargs = len(inspect.signature(fn).parameters)

    def test_fn(self):
        return torch._dynamo.testing.standard_test(
            self,
            fn=fn,
            nargs=nargs,
            expected_frame_count=expected_frame_count,
        )

    return test_fn


class MyCls:
    a = 1


@torch.jit.script_if_tracing
def inline_script_if_tracing(x):
    return x + 1.2


@torch.jit.ignore
def inline_ignore(x):
    return x + 3.4


@torch.jit.unused
def inline_unused(x):
    return x + 5.6


@functools.lru_cache
def inline_lru_cache_fn_with_default_args(x, y, _=None):
    return torch.sin(x * y)


@torch.jit.script_if_tracing
def inline_script_if_tracing_fn_with_default_args(x, y, c=1.2):
    return torch.cos(x * y) + c


class FunctionTests(torch._dynamo.test_case.TestCase):
    @make_test
    def test_inline_jit_annotations(x):
        x = inline_script_if_tracing(x)
        x = inline_ignore(x)
        x = inline_unused(x)
        return

    @make_test
    def test_inline_script_if_tracing_fn_with_default_args(a, b):
        return inline_script_if_tracing_fn_with_default_args(a, b)

    @make_test
    def test_inline_lru_cache_fn_with_default_args(a, b):
        return inline_lru_cache_fn_with_default_args(a, 2, b)

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
    def test_itertools_chain_from_iterable(a, b):
        v = a
        for x in itertools.chain.from_iterable([[a, b], [1, 2]]):
            v = v + x
        return v

    @make_test
    def test_obj_eq(a, b):
        v = a + b
        if MyCls() == None:  # noqa: E711
            return -1
        if MyCls() != None:  # noqa: E711
            v = v.sin()
        if MyCls() == MyCls():
            return -2
        if MyCls() != MyCls():
            return v + 1
        return -3

    @make_test
    def test_cls_eq(a, b):
        v = a + b
        if MyCls == None:  # noqa: E711
            return -1
        if MyCls != None:  # noqa: E711
            v = v.sin()
        if MyCls != MyCls:
            return -2
        if MyCls == MyCls:
            return v + 1
        return -3

    @make_test
    def test_obj_is(a, b):
        v = a + b
        if MyCls() is None:  # noqa: E711
            return -1
        if MyCls() is not None:  # noqa: E711
            v = v.sin()
        if MyCls() is MyCls():
            return -2
        if MyCls() is not MyCls():
            return v + 1
        return -3

    @make_test
    def test_cls_is(a, b):
        v = a + b
        if MyCls is None:  # noqa: E711
            return -1
        if MyCls is not None:  # noqa: E711
            v = v.sin()
        if MyCls is not MyCls:
            return -2
        if MyCls is MyCls:
            return v + 1
        return -3

    @make_test
    def test_itertools_combinations(a, b):
        combs = []
        for size in itertools.combinations((1, 2, 3, 4), 2):
            combs.append(torch.ones(size))
        return combs

    @make_test
    def test_np_iinfo(a):
        max_dim = np.iinfo(np.int16).max
        return a + max_dim

    @make_test
    def test_np_finfo(a):
        min_dim = np.finfo(np.float32).min
        return a + min_dim

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
    def test_cls_hasattr(self, x):
        if hasattr(MyCls, "a"):
            x = x + 1
        if hasattr(MyCls, "b"):
            x = x + 2
        return x

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
    def test_dict_keys(x):
        d = {3: x}
        keys = d.keys()
        d[4] = x + 1
        d2 = {3: 2, 4: "aa"}
        return 3 in keys, 4 in keys, 5 in keys, d2.keys() == keys

    @make_test
    def test_dict_values(x):
        d = {3: x}
        values = d.values()
        d[3] = x + 1
        d[4] = x + 2
        return len(values)

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

    def test_callable_class(self):
        class CallableClass:
            def __call__():
                pass

        class NotCallableClass:
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn1(x, arg):
            if callable(arg):
                return x
            return x + 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn2(x, arg):
            if callable(arg):
                return x * 2
            return x + 1

        input = torch.randn(4)

        for f in [fn1, fn2]:
            self.assertEqual(f(input, NotCallableClass()), input + 1)
            self.assertEqual(
                f(input, CallableClass()), input if f is fn1 else input * 2
            )

            # passing tensor and scalars
            self.assertEqual(f(input, 1), input + 1)
            self.assertEqual(f(input, 1.1), input + 1)
            self.assertEqual(f(input, True), input + 1)
            self.assertEqual(f(input, input), input + 1)

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
    def test_custom_dict_kwargs(x):
        z = CustomDictSubclass(sample=torch.ones(10))
        return z

    @make_test
    def test_float(x):
        y = float(1.2)  # noqa: UP018
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
    def test_is_any_autocast_enabled(x):
        if torch._C._is_any_autocast_enabled():
            return x + 1
        else:
            return x - 1

    @make_test
    def test_list_compare_polyfill(x):
        for a, b, c in [
            [(1, 2, 3), (1, 2, 3), 7.77],
            [(1, 4, 3), (1, 2, 3), 3.33],
            [(1, 2), (1, 2, 3), 5.55],
            [(1, 2, 3), (1, 2), 11.11],
            [(1, -1, 3), (1, 2, 3), 13.33],
        ]:
            if a != b:
                x += 1 * c
            if a == b:
                x += 2 * c
            if a < b:
                x += 4 * c
            if a > b:
                x += 8 * c
            if a <= b:
                x += 16 * c
            if a >= b:
                x += 32 * c
        return x

    @make_test
    def test_promote_types(x):
        if x.dtype == torch.promote_types(torch.int32, torch.float32):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_cublas_allow_tf32(x):
        if torch.backends.cuda.matmul.allow_tf32:
            return x.sin() + 1

        return x.cos() - 1

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
    def test_tensor_is_complex(x):
        if x.is_complex():
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

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @make_test
    def test_get_device_properties_tensor_device(a):
        x = a.to("cuda")
        prop = torch.cuda.get_device_properties(x.device)
        if prop.major == 8:
            return x + prop.multi_processor_count
        return x + prop.max_threads_per_multi_processor

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
    def test_tensor_element_size(a):
        if a.element_size() > 1:
            return (a + a.element_size(), a - a.element_size())
        return (a - a.element_size(), a + a.element_size())

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
    def test_list_index_with_constant_tensor(a, b):
        l1 = [a, b, a + 1, b + 1]
        return l1[torch.as_tensor(2)]

    @make_test
    def test_startswith(a, b):
        x = a + b
        if "foobar".startswith("foo") and "test" in constant3.__module__:
            x = x + 1
        return x

    @make_test
    def test_dict_ops(a, b):
        tmp = {"a": a + 1, "b": b + 2}
        assert tmp.get("zzz") is None
        v = tmp.pop("b") + tmp.get("a") + tmp.get("missing", 3) + tmp.pop("missing", 4)
        tmp.update({"d": 3})
        tmp["c"] = v + tmp["d"]
        if "c" in tmp and "missing" not in tmp:
            return tmp["c"] - tmp["a"] + len(tmp)

    @make_test
    def test_inline_jit__unwrap_optional(x):
        if torch.jit._unwrap_optional(x) is None:
            return torch.ones(2, 2)
        return x.sin()

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
            dd.update({"b": x * 3})
            dd.update([["d", x - 2], ("e", x + 2)])
            dd.update(zip("ab", [x + 3, x + 4]))
            return dd["b"], dd

        x = torch.randn(10, 10)
        ref = fn(x)
        opt_fn = torch._dynamo.optimize_assert("eager")(fn)
        res = opt_fn(x)

        self.assertTrue(same(ref[0], res[0]))
        self.assertTrue(same(ref[1]["a"], res[1]["a"]))
        self.assertTrue(same(ref[1]["b"], res[1]["b"]))
        self.assertTrue(same(ref[1]["c"], res[1]["c"]))
        self.assertTrue(same(ref[1]["d"], res[1]["d"]))
        self.assertTrue(same(ref[1]["e"], res[1]["e"]))
        self.assertTrue(same(ref[1][param], res[1][param]))

    def test_dict_tuple_lazy_guard(self):
        @torch.compile(backend="eager")
        def fn(x, y):
            return torch.sin(x) * y[1]

        fn(torch.randn(3), {1: 1, 2: 2})
        # Changing the value of other key should not causing recompilation
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(3), {1: 1, 2: 3})

        fn(torch.randn(3), (1, 2, 3))
        # Changing the value of index 0, 2 (not 1) should not cause recompilation
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(3), (11, 2, 13))

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
    def test_dict_fromkeys(x, y):
        lst = ["a", "b"]
        d = dict.fromkeys(lst)
        d1 = dict.fromkeys(d, x + 1)
        d2 = collections.defaultdict.fromkeys(iter(d1), x - 2)
        d3 = collections.OrderedDict.fromkeys(tuple(lst), value=y)
        return d1["a"] * d2["b"] + d2["a"] + d1["b"] + d3["a"] + d3["b"] + 1

    @make_test
    def test_dict_copy(x):
        my_list = [("a", x), ("b", x + 1), ("c", x + 2)]
        d1 = dict(my_list)
        d1["a"] = x + 10
        d2 = d1.copy()
        d2["a"] = x - 5
        d2["b"] = x + 3
        d3 = collections.OrderedDict(my_list)
        d3["c"] = x + 20
        d4 = d3.copy()
        d4["c"] = x - 10
        return d1["a"] * d2["a"] + d2["b"] + d3["c"] * d4["c"] + 1

    @make_test
    def test_dict_update(x, y, z):
        d = {"a": x, "b": y}
        d.update({"a": y - 1})
        d.update([("b", z + 1), ["c", z]])
        d.update(zip("ab", [z + 3, y + 2]))

        od = collections.OrderedDict(a=x * 3, b=y + 2)
        od.update({"a": y + 5})
        od.update([["b", z + 6], ("c", z - 7)])
        od.update(zip("ab", [z - 3, x + 2]))
        return d["a"] * od["a"] + od["c"] + d["b"] + od["b"] * d["c"]

    @make_test
    def test_min_max(a, b):
        c = a + b
        a = a.sum()
        b = b.sum()
        a = min(max(a, 0), 1)
        b = max(0, min(1, b))
        return max(a, b) - min(a, b) + c

    @make_test
    def test_symbool_to_int(x):
        # this is roughly the pattern found in einops.unpack()
        if sum(s == -1 for s in x.size()) == 0:
            return x + 1
        else:
            return x - 1

    @make_test
    def test_map_sum(a, b, c, d):
        return sum(map(lambda x: x + 1, [a, b, c, d]))

    @make_test
    def test_sum(a, b, c, d):
        return sum([a, b, c, d])

    @make_test
    def test_sum_with_start_arg(a, b, c, d):
        return sum([b, c, d], a)

    @make_test
    def test_sum_with_start_kwarg(a, b, c, d):
        return sum([b, c, d], start=a)

    @make_test(expected_frame_count=0)
    def test_sum_shortcut():
        return sum([0, 1.0, 2, 3.0])

    @make_test(expected_frame_count=0)
    def test_sum_shortcut_with_start_arg():
        return sum([0, 1.0, 2, 3.0], -10)

    @make_test(expected_frame_count=0)
    def test_sum_shortcut_with_start_kwarg():
        return sum([0, 1.0, 2, 3.0], start=-10)

    @make_test
    def test_reduce(a, b, c, d):
        return functools.reduce(operator.add, [a, b, c, d])

    @make_test
    def test_reduce_with_initial(a, b, c, d):
        return functools.reduce(operator.add, [b, c, d], a)

    @make_test(expected_frame_count=0)
    def test_reduce_with_single(x):
        return functools.reduce(lambda a, b: (a, b), [x])

    @make_test(expected_frame_count=0)
    def test_reduce_with_single_with_initial(x, y):
        return functools.reduce(lambda a, b: (a, b), [y], x)

    @make_test(expected_frame_count=0)
    def test_reduce_with_none_initial(x):
        return functools.reduce(lambda a, b: (a, b), [x], None)

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

    @unittest.skipIf(
        sys.version_info < (3, 9),
        "SET_UPDATE was added at Python 3.9",
    )
    @make_test
    def test_set_update_bytecode(x):
        # This produces bytecode SET_UPDATE since python 3.9
        var = {"apple", "banana", "cherry"}
        if isinstance(var, set):
            return x + 1
        else:
            return x - 1

    @unittest.skipIf(
        sys.version_info < (3, 9),
        "SET_UPDATE was added at Python 3.9",
    )
    @make_test
    def test_set_update_list_with_duplicated_items(x):
        list1 = ["apple", "banana", "apple"]
        list2 = ["orange", "banana"]
        if len({*list1, *list2}) == 3:
            return x + 1
        else:
            return x - 1

    @make_test
    def test_set_contains(a, b):
        vals = set(["a", "b", "c"])
        if "a" in vals:
            x = a + b
        else:
            x = a - b
        if "d" in vals:
            y = a + b
        else:
            y = a - b
        return x, y

    def test_set_isdisjoint(self):
        x = {"apple", "banana", "cherry"}
        y = {"google", "microsoft", "apple"}

        def fn(a):
            if x.isdisjoint(y):
                return a + 1
            else:
                return a - 1

        test = make_test(fn)
        test(self)

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
    def test_namedtuple_hasattr(a, b):
        mytuple = FunctionTests.MyNamedTuple(a, b)

        def isinstance_namedtuple(obj) -> bool:
            return (
                isinstance(obj, tuple)
                and hasattr(obj, "_asdict")
                and hasattr(obj, "_fields")
            )

        if isinstance_namedtuple(mytuple):
            return a + b
        else:
            return a - b

    @make_test
    def test_torch_size_hasattr(x):
        if hasattr(x.shape, "_fields"):
            return x + 1
        else:
            return x - 1

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

    @make_test
    def test_is_contiguous_memory_format(tensor):
        if torch.jit.is_scripting():
            return None
        elif tensor.is_contiguous(memory_format=torch.contiguous_format):
            return tensor + 1

    def test_is_contiguous_frame_counts(self):
        data = [
            torch.rand(10),
            torch.rand(2, 3, 32, 32),
            torch.rand(2, 3, 32, 32).contiguous(memory_format=torch.channels_last),
            torch.rand(10)[::2],
            torch.rand(12),
            torch.rand(2, 3, 24, 24).contiguous(memory_format=torch.channels_last),
            torch.rand(50)[::2],
            torch.rand(2, 3, 32, 32)[:, :, 2:-2, 3:-3],
        ]
        # dynamo should recompile for all inputs in static shapes mode
        expected_frame_counts_static = [1, 2, 3, 4, 5, 6, 7, 8]
        # dynamo should recompile for items 0, 1, 2, 6 in dynamic shapes mode
        expected_frame_counts_dynamic = [1, 2, 3, 4, 4, 4, 4, 5]
        expected_frame_counts = ifdynstaticdefault(
            expected_frame_counts_static, expected_frame_counts_dynamic
        )
        dynamic = ifdynstaticdefault(False, True)

        def func(x):
            if x.is_contiguous():
                return x + 1
            elif x.is_contiguous(memory_format=torch.channels_last):
                return x + 2
            else:
                return x + 3

        cnt = torch._dynamo.testing.CompileCounter()
        cfunc = torch._dynamo.optimize_assert(cnt, dynamic=dynamic)(func)

        assert cnt.frame_count == 0
        for i, x in enumerate(data):
            expected = func(x)
            output = cfunc(x)
            self.assertTrue(same(output, expected))
            assert cnt.frame_count == expected_frame_counts[i]

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

    def test_math_radians(self):
        def func(x, a):
            return x + math.radians(a)

        cnt = torch._dynamo.testing.CompileCounter()
        cfunc = torch._dynamo.optimize_assert(cnt)(func)

        assert cnt.frame_count == 0
        x = torch.rand(10)
        expected = func(x, 12)
        output = cfunc(x, 12)
        self.assertTrue(same(output, expected))
        assert cnt.frame_count == 1

    @make_test
    def test_numpy_meshgrid(x, y):
        r1, r2 = np.meshgrid(x.numpy(), y.numpy())
        return torch.from_numpy(r1), torch.from_numpy(r2)

    @make_test
    def test_torch_from_numpy(x):
        a = x.numpy()
        b = torch.from_numpy(a)
        if b.size(0) == 1:
            return torch.tensor(True)
        else:
            return torch.tensor(False)

    @make_test
    def test_numpy_size(x):
        a = x.numpy()
        return a.size

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

    @make_test
    def test_mean_sum_np(x: torch.Tensor):
        x_mean = np.mean(x.numpy(), 1)
        x_sum = np.sum(x_mean)
        x_sum_array = np.asarray(x_sum)
        return torch.from_numpy(x_sum_array)

    @make_test
    def test_return_numpy_ndarray(x):
        a = x.numpy()
        return a.T

    @make_test
    def test_return_multiple_numpy_ndarray(x):
        a = x.numpy()
        return a.T, a.imag, a.real

    @make_test
    def test_ndarray_method(x):
        a = x.numpy()
        return a.copy()

    @make_test
    def test_ndarray_transpose(x):
        a = x.numpy()
        return a.transpose(0, 1)

    @make_test
    def test_ndarray_reshape(x):
        a = x.numpy()
        return a.reshape([1, a.size])

    @make_test
    def test_ndarray_methods_returning_scalar(x):
        a = x.numpy()
        return a.max(axis=0), a.all(axis=0)

    @make_test
    def test_ndarray_builtin_functions(x):
        a = x.numpy()
        return a + a, a - a

    @make_test
    def test_numpy_dtype_argument_to_function(x):
        return np.ones_like(x, dtype=np.float64)

    @make_test
    def test_numpy_dtype_call_in_function(x):
        dt = np.dtype("float")
        return np.full_like(x, 2.4, dtype=dt)

    @make_test
    def test_numpy_linalg(x):
        return np.linalg.norm(x.numpy(), axis=0)

    @make_test
    def test_numpy_fft(x):
        return np.fft.fftshift(x.numpy())

    @make_test
    def test_numpy_random():
        x = np.random.randn(2, 2)
        return x - x

    @make_test
    def test_partials_torch_op_kwarg(x):
        par_mul = functools.partial(torch.mul, other=torch.ones(10, 10))
        return par_mul(x)

    @make_test
    def test_partials_torch_op_arg(x):
        par_mul = functools.partial(torch.mul, torch.ones(10, 10))
        return par_mul(x)

    @make_test
    def test_partials_udf_arg(x):
        par_mul = functools.partial(udf_mul, torch.ones(10, 10))
        return par_mul(x)

    @make_test
    def test_list_add_then_mutate(x):
        my_list = [1, x]
        y = x / 4.0
        my_list = my_list + [x / 2.0, 4]
        my_list.append(y)
        return sum(my_list)

    @make_test
    def test_list_expand_lhs(x):
        return sum(4 * [x])

    @make_test
    def test_in_not_in(x):
        mylist = [1, 2, 3, 4, 5, x]
        myotherlist = [1, 2, 3, 4, 5]
        assert 3 in mylist
        assert 6 not in myotherlist
        return sum(mylist)

    @make_test
    def test_partials_udf_kwarg(x):
        par_mul = functools.partial(udf_mul, y=torch.ones(10, 10))
        return par_mul(x)

    @make_test
    def test_partials_udf_kwarg_module(x, y):
        par_mod = functools.partial(udf_module, mod=SmallNN())
        return par_mod(x=x, y=y)

    @make_test
    def test_partials_udf_kwarg_method(x, y):
        par_mod = functools.partial(udf_module, mod=SmallNN().forward)
        return par_mod(x=x, y=y)

    @make_test
    def test_partials_lambda(x):
        multiply = lambda x, y: x * y
        triple = functools.partial(multiply, y=3)
        return triple(x)

    @unittest.skipUnless(torch.distributed.is_available(), "requires torch.distributed")
    @make_test
    def test_flat_param_same_storage_size(x, y):
        import torch.distributed.fsdp._flat_param as flat_param

        if flat_param._same_storage_size(x, 100):
            x = x + 1
        else:
            x = x - 1
        if flat_param._same_storage_size(y, 123):
            y = y + 1
        else:
            y = y - 1
        return x, y

    @parametrize(
        "attr",
        (
            # True
            "__subclasshook__",
            "__lt__",
            "__hash__",
            "__ge__",
            "__le__",
            "__gt__",
            "__dict__",
            "__getattribute__",
            "__setattr__",
            "__doc__",
            "__repr__",
            "__dir__",
            "__init__",
            "__new__",
            "__class__",
            "__eq__",
            "__delattr__",
            "__reduce__",
            "__module__",
            "__format__",
            "__str__",
            "__sizeof__",
            "__ne__",
            "__call__",
            "__reduce_ex__",
            "__init_subclass__",
            "args",
            "keywords",
            "func",
            # False
            "__code__",
            "__kwdefaults__",
            "__defaults__",
            "__name__",
            "__annotations__",
            "__get__",
            "__builtins__",
            "__qualname__",
            "__globals__",
            "__closure__",
        ),
    )
    def test_partials_hasattr(self, attr):
        def fn(t):
            f = lambda x, y: torch.sin(x) + torch.cos(y)
            p = functools.partial(f, y=t)
            if hasattr(p, attr):
                return p(t)
            else:
                return torch.zeros_like(t)

        t = torch.randn(3, 4)
        counter = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fullgraph=True, backend=counter)(fn)
        self.assertEqual(opt_fn(t), fn(t))
        self.assertGreater(counter.frame_count, 0)

    @unittest.expectedFailure
    def test_partials_hasattr_set_attr(self):
        def fn(t):
            f = lambda x, y: torch.sin(x) + torch.cos(y)
            p = functools.partial(f, y=t)
            p.__name__ = "test"
            if hasattr(p, "__name__"):
                return p(t)
            else:
                return torch.zeros_like(t)

        t = torch.randn(3, 4)
        counter = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fullgraph=True, backend=counter)(fn)
        self.assertEqual(opt_fn(t), fn(t))

    def test_pow_int(self):
        def fn(a, b):
            return torch.pow(a, b)

        x = torch.ones(2, 2)
        opt_fn = torch.compile(fullgraph=True, backend="eager", dynamic=True)(fn)
        self.assertEqual(opt_fn(x, 2), fn(x, 2))

    def test_tensor_size_indexed_by_symint(self):
        def fn(x, y):
            index = x.shape[-1]
            return x + y.shape[index]

        x = torch.rand(10, 2)
        y = torch.rand(10, 8, 6)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_partials_as_input_partials_lambda(self):
        def fn(f0, f1, x):
            return f0(x) * f1(x)

        multiply = lambda x, y: x * y
        lambda0 = functools.partial(multiply, y=3)
        lambda1 = functools.partial(multiply, y=2)

        cnts = torch._dynamo.testing.CompileCounter()
        torch._dynamo.optimize(cnts, nopython=True)(fn)(
            lambda0, lambda1, torch.randn(2, 2)
        )
        self.assertEqual(cnts.frame_count, 1)

    def test_partials_as_input_partials_mod(self):
        def fn(f0, f1, x):
            return f0(x) * f1(x)

        lambda0 = functools.partial(SmallNN(), y=torch.randn(2, 2))
        lambda1 = functools.partial(SmallNN(), y=torch.randn(2, 2))

        cnts = torch._dynamo.testing.CompileCounter()
        x = torch.randn(2, 2)
        dynamo_result = torch._dynamo.optimize(cnts, nopython=True)(fn)(
            lambda0, lambda1, x
        )
        self.assertEqual(cnts.frame_count, 1)

        eager_result = fn(lambda0, lambda1, x)
        self.assertEqual(eager_result, dynamo_result)

    def test_partials_as_input_UDF(self):
        def fn(f0, f1, x):
            return f0(x) * f1(x)

        lambda0 = functools.partial(udf_mul, y=torch.randn(2, 2))
        lambda1 = functools.partial(udf_mul, y=torch.randn(2, 2))

        cnts = torch._dynamo.testing.CompileCounter()
        x = torch.randn(2, 2)
        dynamo_result = torch._dynamo.optimize(cnts, nopython=True)(fn)(
            lambda0, lambda1, x
        )
        self.assertEqual(cnts.frame_count, 1)

        eager_result = fn(lambda0, lambda1, x)
        self.assertEqual(eager_result, dynamo_result)

    def test_partials_graph_break_reconstruct(self):
        def fn(udf_mul_0, udf_mul_1, x):
            lambda0 = functools.partial(udf_mul_0, y=x)
            lambda1 = functools.partial(udf_mul_1, y=x)

            print("break")
            return torch.mul(lambda0(x), lambda1(x))

        backend = EagerAndRecordGraphs()
        cnts = CompileCounterWithBackend(backend)
        x = torch.randn(2, 2)
        dynamo_result = torch._dynamo.optimize(cnts)(fn)(udf_mul, udf_mul, x)

        eager_result = fn(udf_mul, udf_mul, x)
        gm = backend.graphs[0]
        self.assertEqual(eager_result, dynamo_result)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_lambda0_keywords_y_: "f32[2, 2]"):
        l_lambda0_keywords_y_ = L_lambda0_keywords_y_

        mul: "f32[2, 2]" = l_lambda0_keywords_y_ * l_lambda0_keywords_y_
        mul_1: "f32[2, 2]" = l_lambda0_keywords_y_ * l_lambda0_keywords_y_;  l_lambda0_keywords_y_ = None

        mul_2: "f32[2, 2]" = torch.mul(mul, mul_1);  mul = mul_1 = None
        return (mul_2,)
""",
            )
        else:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s0: "Sym(s0)", L_lambda0_keywords_y_: "f32[s0, s0]"):
        l_lambda0_keywords_y_ = L_lambda0_keywords_y_

        mul: "f32[s0, s0]" = l_lambda0_keywords_y_ * l_lambda0_keywords_y_
        mul_1: "f32[s0, s0]" = l_lambda0_keywords_y_ * l_lambda0_keywords_y_;  l_lambda0_keywords_y_ = None

        mul_2: "f32[s0, s0]" = torch.mul(mul, mul_1);  mul = mul_1 = None
        return (mul_2,)
""",
            )

    def test_partials_graph_break_reconstruct_mix(self):
        def fn(udf_mul_0, udf_add_1, x):
            lambda0 = functools.partial(udf_mul_0, y=x)
            lambda1 = functools.partial(udf_add_1, x)

            print("break")
            return torch.mul(lambda0(x), lambda1(x))

        backend = EagerAndRecordGraphs()
        cnts = CompileCounterWithBackend(backend)
        x = torch.randn(2, 2)
        dynamo_result = torch._dynamo.optimize(cnts)(fn)(udf_mul, udf_add, x)

        eager_result = fn(udf_mul, udf_add, x)
        gm = backend.graphs[0]
        self.assertEqual(eager_result, dynamo_result)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_lambda0_keywords_y_: "f32[2, 2]"):
        l_lambda0_keywords_y_ = L_lambda0_keywords_y_

        mul: "f32[2, 2]" = l_lambda0_keywords_y_ * l_lambda0_keywords_y_

        add: "f32[2, 2]" = l_lambda0_keywords_y_ + l_lambda0_keywords_y_;  l_lambda0_keywords_y_ = None

        mul_1: "f32[2, 2]" = torch.mul(mul, add);  mul = add = None
        return (mul_1,)
""",
            )
        else:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s0: "Sym(s0)", L_lambda0_keywords_y_: "f32[s0, s0]"):
        l_lambda0_keywords_y_ = L_lambda0_keywords_y_

        mul: "f32[s0, s0]" = l_lambda0_keywords_y_ * l_lambda0_keywords_y_

        add: "f32[s0, s0]" = l_lambda0_keywords_y_ + l_lambda0_keywords_y_;  l_lambda0_keywords_y_ = None

        mul_1: "f32[s0, s0]" = torch.mul(mul, add);  mul = add = None
        return (mul_1,)
""",
            )

    def test_partials_graph_break_reconstruct_mix_no_source(self):
        def fn(udf_mul_0, x):
            udf_add_1 = lambda x, y: x + y

            lambda0 = functools.partial(udf_mul_0, y=x)
            lambda1 = functools.partial(udf_add_1, x)

            print("break")
            return torch.mul(lambda0(x), lambda1(x))

        backend = EagerAndRecordGraphs()
        cnts = CompileCounterWithBackend(backend)
        x = torch.randn(2, 2)
        dynamo_result = torch._dynamo.optimize(cnts)(fn)(udf_mul, x)

        eager_result = fn(udf_mul, x)
        gm = backend.graphs[0]
        self.assertEqual(eager_result, dynamo_result)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_lambda0_keywords_y_: "f32[2, 2]"):
        l_lambda0_keywords_y_ = L_lambda0_keywords_y_

        mul: "f32[2, 2]" = l_lambda0_keywords_y_ * l_lambda0_keywords_y_

        add: "f32[2, 2]" = l_lambda0_keywords_y_ + l_lambda0_keywords_y_;  l_lambda0_keywords_y_ = None

        mul_1: "f32[2, 2]" = torch.mul(mul, add);  mul = add = None
        return (mul_1,)
""",
            )
        else:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s0: "Sym(s0)", L_lambda0_keywords_y_: "f32[s0, s0]"):
        l_lambda0_keywords_y_ = L_lambda0_keywords_y_

        mul: "f32[s0, s0]" = l_lambda0_keywords_y_ * l_lambda0_keywords_y_

        add: "f32[s0, s0]" = l_lambda0_keywords_y_ + l_lambda0_keywords_y_;  l_lambda0_keywords_y_ = None

        mul_1: "f32[s0, s0]" = torch.mul(mul, add);  mul = add = None
        return (mul_1,)
""",
            )

    def test_partials_graph_break_reconstruct_args_and_kwargs(self):
        def fn(udf_mul_0, x):
            lambda0 = functools.partial(udf_mul_0, x, 4, z=x)
            lambda1 = functools.partial(udf_mul_0, 4, z=x)

            return torch.mul(lambda0(), lambda1(5))

        backend = EagerAndRecordGraphs()
        cnts = CompileCounterWithBackend(backend)
        x = torch.randn(2, 2)
        dynamo_result = torch._dynamo.optimize(cnts)(fn)(udf_mul2, x)

        eager_result = fn(udf_mul2, x)
        gm = backend.graphs[0]
        self.assertEqual(eager_result, dynamo_result)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 2]"):
        l_x_ = L_x_

        mul: "f32[2, 2]" = l_x_ * 4
        mul_1: "f32[2, 2]" = mul * l_x_;  mul = None
        mul_2: "f32[2, 2]" = 20 * l_x_;  l_x_ = None

        mul_3: "f32[2, 2]" = torch.mul(mul_1, mul_2);  mul_1 = mul_2 = None
        return (mul_3,)
""",
            )
        else:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s0: "Sym(s0)", L_x_: "f32[s0, s0]"):
        l_x_ = L_x_

        mul: "f32[s0, s0]" = l_x_ * 4
        mul_1: "f32[s0, s0]" = mul * l_x_;  mul = None
        mul_2: "f32[s0, s0]" = 20 * l_x_;  l_x_ = None

        mul_3: "f32[s0, s0]" = torch.mul(mul_1, mul_2);  mul_1 = mul_2 = None
        return (mul_3,)
""",
            )

    def test_partials_recompilation(self):
        def fn(f0, f1, x):
            return f0(x) * f1(x)

        lambda0 = functools.partial(udf_mul, y=torch.randn(2, 2))
        lambda1 = functools.partial(udf_mul, y=torch.randn(2, 2))

        cnts = torch._dynamo.testing.CompileCounter()

        x = torch.randn(2, 2)
        fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        dynamo_result = fn(lambda0, lambda1, x)
        self.assertEqual(cnts.frame_count, 1)

        fn(lambda1, lambda0, x)
        self.assertEqual(
            cnts.frame_count, 1
        )  # No recompile! Tensor and udf_mul guarded

        lambda2 = functools.partial(udf_mul, y=torch.randn(3, 3))
        x = torch.randn(3, 3)
        fn(lambda2, lambda2, x)
        self.assertEqual(cnts.frame_count, 2)  # Recompile! Tensor size changed

        multiply = lambda x, y: x * y
        lambda3 = functools.partial(multiply, y=torch.randn(3, 3))
        x = torch.randn(3, 3)
        fn(lambda3, lambda3, x)

        self.assertEqual(cnts.frame_count, 3)  # Recompile! func id changed

        def fn2(f0, f1, args):
            return f0(*args) * f1(*args)

        cnts = torch._dynamo.testing.CompileCounter()

        x = torch.randn(2, 2)
        fn2 = torch._dynamo.optimize(cnts, nopython=True)(fn2)
        dynamo_result = fn2(lambda0, lambda1, [x])
        self.assertEqual(cnts.frame_count, 1)  # start over

        lambda4 = functools.partial(multiply, y=3, x=torch.randn(3, 3))
        fn2(lambda4, lambda4, [])

        self.assertEqual(cnts.frame_count, 2)  # Recompile! Different kwarg keys

        lambda5 = functools.partial(multiply, 1)
        x = torch.randn(3, 3)
        fn2(lambda5, lambda5, [x])

        self.assertEqual(cnts.frame_count, 3)  # Recompile! Different arg keys

        lambda6 = lambda x: x + x
        fn2(lambda6, lambda6, [x])
        self.assertEqual(
            cnts.frame_count, 4
        )  # Recompile! input is no longer a functools partial

    def test_manual_seed(self):
        @torch.compile
        def foo():
            torch.manual_seed(3)
            return torch.randint(0, 5, (5,))

        self.assertEqual(foo(), foo())
        self.assertEqual(foo(), foo())

    def test_partial_across_graph_break_uninvoked(self):
        from functools import partial

        def bar(x, **kwargs):
            return x + x

        @torch.compile(backend="eager", dynamic=True)
        def foo(x, i):
            def inner():
                print("this is a graph_break")
                return op(x)

            op = partial(bar, dim=10)
            x = inner()
            op = partial(bar, other=10)
            return inner() + x

        foo(torch.rand(1), 10)

    def test_no_recompile_inner_function(self):
        def forward(inp):
            def g(y):
                return inp + y

            print("graph break")
            return g(torch.rand([1]))

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(forward)

        input = torch.rand([2])
        _ = opt_fn(input)
        _ = opt_fn(input)
        _ = opt_fn(input)
        # Should not have recompiled
        self.assertEqual(cnts.frame_count, 1)

    def test_no_recompile_inner_lambda(self):
        def forward(inp):
            g = lambda y: inp + y
            print("graph break")
            return g(torch.rand([1]))

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(forward)

        input = torch.rand([2])
        _ = opt_fn(input)
        _ = opt_fn(input)
        _ = opt_fn(input)
        # Should not have recompiled
        self.assertEqual(cnts.frame_count, 1)

    def test_complex_closure(self):
        @torch.compile
        def forward(y):
            def a():
                def x(z):
                    return y + z

                return x

            return a()

        input1 = torch.rand([2])
        input2 = torch.rand([2])
        res = forward(input1)(input2)
        self.assertTrue(same(res, input1 + input2))

    def test_non_inlined_closure(self):
        @torch.compile()
        def program(x, y):
            one = lambda x, y: x + y

            def inner():
                # Force no inlining
                torch._dynamo.graph_break()
                return one(x, y)

            res = inner()
            one = lambda x, y: x - y
            res += inner()
            return res

        input1 = torch.randn(1)
        input2 = torch.randn(1)

        self.assertTrue(same(program(input1, input2), input1 + input1))

    @parametrize("int_or_float", ("int", "float"))
    def test_np_constant_collections_as_input(self, int_or_float):
        info_func = getattr(np, f"{int_or_float[0]}info")
        dt_string_arg = f"{int_or_float}16"
        np_dt_attr = getattr(np, dt_string_arg)

        dt_args = [dt_string_arg, np_dt_attr]
        arg_variants_iter = itertools.chain(
            dt_args, map(np.dtype, dt_args), map(info_func, dt_args)
        )

        def func(a, b, info_or_dt):
            return a + info_func(info_or_dt).max

        opt_fn = torch.compile(func)

        a = torch.randn(2)
        b = torch.randn(2)
        eager_result = func(a, b, dt_args[0])

        for arg in arg_variants_iter:
            opt_result = opt_fn(a, b, arg)
            self.assertTrue(same(opt_result, eager_result))

    @parametrize(
        "typ, info_func",
        [
            (int, np.iinfo),
            (float, np.finfo),
        ],
        name_fn=lambda t, _: t.__name__,
    )
    def test_np_constant_collections_guards(self, typ, info_func):
        def func_info(a, info):
            return a + info.max

        def func_dtype(a, dt):
            return a + info_func(dt).max

        dt_args = [
            np.dtype(typ),
            np.ones((1,), dtype=typ).dtype,
            np.dtype(np.dtype(typ).name),
            np.dtype(typ.__name__),
        ]
        cnts_1 = torch._dynamo.testing.CompileCounter()
        opt_fn_dtype = torch._dynamo.optimize(cnts_1)(func_dtype)
        a = torch.zeros(3, dtype=typ)
        for arg in dt_args:
            r = opt_fn_dtype(a, arg)
        # each should produce an identical arg
        self.assertEqual(cnts_1.frame_count, 1)

        cnts_2 = torch._dynamo.testing.CompileCounter()
        opt_fn_info = torch._dynamo.optimize(cnts_2)(func_info)
        info_args = [info_func(dt) for dt in dt_args]
        for arg in info_args:
            r = opt_fn_info(a, arg)

        # each should produce an identical arg
        self.assertEqual(cnts_2.frame_count, 1)

        if typ is float:
            dt_extra = np.dtype(np.float16)
        else:
            dt_extra = np.dtype(np.int16)
        info_extra = info_func(dt_extra)

        eager_result_dtype = func_dtype(a, dt_extra)
        compile_result_dtype = opt_fn_dtype(a, dt_extra)
        self.assertEqual(cnts_1.frame_count, 2)
        self.assertEqual(eager_result_dtype, compile_result_dtype)

        eager_result_info = func_info(a, info_extra)
        compile_result_info = opt_fn_info(a, info_extra)
        self.assertEqual(cnts_2.frame_count, 2)
        self.assertEqual(eager_result_info, compile_result_info)

    def test_compare_constant_and_tensor(self):
        for op in [
            operator.lt,
            operator.le,
            operator.gt,
            operator.ge,
            operator.ne,
            operator.eq,
            operator.is_,
            operator.is_not,
        ]:
            with self.subTest(op=op):

                def fn(x):
                    return op(-10, x)

                opt_fn = torch.compile(fullgraph=True)(fn)

                x = torch.randn(10)
                self.assertEqual(opt_fn(x), fn(x))

    def test_pos(self):
        def fn(x, y):
            return operator.pos(x) * +y

        opt_fn = torch.compile(fullgraph=True, dynamic=True)(fn)

        def test(x, y):
            self.assertEqual(opt_fn(x, y), fn(x, y))

        test(torch.ones(4), 1)
        test(1, torch.ones(4))
        test(-1, -1)
        test(-1.1, 1.1)
        test(True, False)
        test(torch.ones(4, dtype=torch.float32), 1.1)

    def test_index(self):
        def fn(x, t):
            v = operator.index(x)
            torch.mul(t, v)

        def test(a, b):
            self.assertEqual(opt_fn(a, b), fn(a, b))

        for dynamic in [True, False]:
            torch._dynamo.reset()
            opt_fn = torch._dynamo.optimize(dynamic=dynamic)(fn)
            t = torch.ones(1)
            test(10, t)
            test(-100, t)
            test(10, t)
            test(False, t)
            test(True, t)

    def test_truth(self):
        def fn(x, y):
            return operator.truth(x) and bool(y)

        opt_fn = torch.compile(fullgraph=True, dynamic=False)(fn)

        def test(x, y):
            self.assertEqual(opt_fn(x, y), fn(x, y))

        test(1, 100)
        test(-1.1, True)
        test(-1.1, 1.1)
        test(True, False)
        test(torch.ones(1), 1)
        test(torch.zeros(1), 1)
        test(torch.ones(1), torch.ones(1))

    def test_unary_fold_op(self):
        for op in (operator.abs, abs, operator.neg, operator.pos, operator.truth):
            with self.subTest(op=op):

                def fn():
                    a = range(-10, 10)
                    return list(map(op, a))

                opt_fn = torch._dynamo.optimize(nopython=True)(fn)
                self.assertEqual(opt_fn(), fn())

    def test_unary_fold_op_seq(self):
        for op in (operator.length_hint,):
            with self.subTest(op=op):

                def fn():
                    a = [tuple(range(-10, i)) for i in range(10)]
                    return tuple(map(op, a))

                opt_fn = torch._dynamo.optimize(nopython=True)(fn)
                self.assertEqual(opt_fn(), fn())

    def gen_random_range_args(self):
        args_count = random.randint(1, 3)
        args = [random.randint(-10, 10) for _ in range(args_count)]
        if args_count == 3 and args[2] == 0:
            args[2] = 1
        return args

    def test_range_length(self):
        def test(*args, expected=None):
            r = range(*args)
            range_variable = RangeVariable([ConstantVariable.create(v) for v in args])

            self.assertEqual(len(r), range_variable.range_length())

            if expected is not None:
                self.assertEqual(len(r), expected)

        test(1, 1, 1, expected=0)
        test(1, 0, expected=0)
        test(-10, expected=0)

        test(4, expected=4)
        test(10, expected=10)

        # step >1
        test(1, 10, 2, expected=5)

        # negative step
        test(10, 1, -1, expected=9)
        test(10, 1, -3)

        # Fuzz testing
        for i in range(100):
            args = self.gen_random_range_args()
            print("testing :", args)
            test(*args)

    def test_indexed_range(self):
        def test(range, index, expected=None):
            range_variable = RangeVariable(
                [
                    ConstantVariable.create(v)
                    for v in [range.start, range.stop, range.step]
                ]
            )

            self.assertEqual(
                range[index],
                range_variable.apply_index(index).as_python_constant(),
            )

            if expected is not None:
                self.assertEqual(range[index], expected)

        test(range(10), 1, expected=1)
        test(range(10, 20, 2), 1, expected=12)

        # Fuzz testing
        for i in range(100):
            range_args = self.gen_random_range_args()
            r = range(*range_args)

            if len(r) == 0:
                continue

            index = random.randint(0, len(r) - 1)

            print("testing:", r, index)
            test(r, index)

    def test_sliced_range(self):
        def test(range, slice, expected=None):
            range_variable = RangeVariable(
                [
                    ConstantVariable.create(v)
                    for v in [range.start, range.stop, range.step]
                ]
            )

            self.assertEqual(
                range[slice],
                range_variable.apply_slice(slice).as_python_constant(),
            )

            if expected is not None:
                self.assertEqual(
                    range[slice],
                    expected,
                )

        test(range(10), slice(1, 10, 2), expected=range(1, 10, 2))
        test(range(10), slice(None, 10, None), expected=range(0, 10))
        test(range(10), slice(-1, 7, None), expected=range(9, 7))
        test(range(10), slice(-1, 7, 2), expected=range(9, 7, 2))
        test(range(1, 10, 2), slice(3, 7, 2), expected=range(7, 11, 4))
        test(range(1, 10, 2), slice(-3, 7, 2), expected=range(5, 11, 4))
        test(range(-1, -5, -3), slice(5, None, -3), expected=range(-4, 2, 9))

        def rand_slice():
            def flip_coin():
                # 1 out of 10
                return random.randint(1, 10) == 5

            def r_item(allow_zero=True):
                i = random.randint(-10, 10)
                if not allow_zero and i == 0:
                    i = 1
                if flip_coin():
                    i = None
                return i

            arg_count = random.randint(1, 3)

            if arg_count == 1:
                return slice(r_item())
            elif arg_count == 2:
                return slice(r_item(), r_item())
            else:
                return slice(r_item(), r_item(), r_item(False))

        # Fuzz testing
        for i in range(100):
            range_args = self.gen_random_range_args()
            r = range(*range_args)
            # generate random slice
            s = rand_slice()

            print("testing:", r, s)
            test(r, s)

    def test_range_with_slice_index(self):
        def fn(x):
            acc = 1
            for k in range(2)[1::2]:
                acc *= acc * k
            return x * acc

        opt_fn = torch.compile(fullgraph=True)(fn)
        x = torch.ones(1)
        self.assertEqual(opt_fn(x), fn(x))

    def test_range_with_index(self):
        def fn(x):
            acc = 1
            acc *= acc * range(10, 20, 2)[2]
            return x * acc

        opt_fn = torch.compile(fullgraph=True)(fn)
        x = torch.ones(1)
        self.assertEqual(opt_fn(x), fn(x))

    def test_rand_inlined(self):
        @torch.compile(backend="eager", dynamic=True)
        def fn():
            idx_size = [10]
            idx_size[random.randint(0, 0)] = random.randint(1, 8)
            t = tuple(idx_size)
            src_size = [random.randint(1, 5) + s for s in idx_size]
            idx = torch.empty(t)

        fn()

    def test_rand_tensor_partial(self):
        from collections import namedtuple
        from functools import partial

        SdpaShape = namedtuple(
            "Sdpa_Shape", ["batch", "num_heads", "seq_len", "head_dim"]
        )

        @torch.compile(backend="eager")
        def func():
            make_tensor = partial(
                torch.rand, device="cpu", dtype=torch.float16, requires_grad=True
            )

            bsz, num_heads, seq_len_q, seq_len_kv, head_dim = (16, 16, 128, 128, 16)
            make_q_tensor = partial(
                make_tensor, SdpaShape(bsz, num_heads, seq_len_q, head_dim)
            )
            make_kv_tensor = partial(
                make_tensor, SdpaShape(bsz, num_heads, seq_len_kv, head_dim)
            )
            t1 = make_q_tensor()
            t2 = make_kv_tensor()
            t3 = t1 + t2

        func()

    def test_to(self):
        @torch.compile(backend="eager")
        def fn():
            t = torch.ones(2)
            y = t.to("meta")

        fn()

    def test_elipsis(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(a, ind, val):
            a[ind] = val
            return a

        arr = np.zeros(4)
        self.assertEqual(fn(arr, np.s_[...], np.ones(4)), np.ones(4))

        arr = np.array([[1, 1], [2, 2]])
        self.assertEqual(
            fn(arr, np.s_[0, ...], np.zeros(2)), np.array([[0, 0], [2, 2]])
        )

        arr = np.array([[1, 1], [2, 2]])
        self.assertEqual(
            fn(arr, np.s_[1, ...], np.zeros(2)), np.array([[1, 1], [0, 0]])
        )

        arr = np.array([[1, 1], [2, 2]])
        self.assertEqual(
            fn(arr, np.s_[..., 0], np.array([3, 3])), np.array([[3, 1], [3, 2]])
        )

        arr = np.array([[1, 1], [2, 2]])
        self.assertEqual(
            fn(arr, np.s_[..., 1], np.array([3, 3])), np.array([[1, 3], [2, 3]])
        )


def udf_mul(x, y):
    return x * y


def udf_mul2(x, y, z):
    return x * y * z


def udf_add(x, y):
    return x + y


class SmallNN(torch.nn.Module):
    def forward(self, x, y):
        combined = torch.cat((x, y), dim=1)
        out = torch.nn.ReLU()(combined)
        out = torch.nn.ReLU()(out)
        return out


def udf_module(mod, x, y):
    return mod(x, y)


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

    def test_listlike_of_tensors_contains_constant(self):
        for listlike in [set, list]:

            def fn(x):
                x.add_(1)
                s = listlike([x])
                res = 1 in s
                return res

            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            x = torch.randn(1)
            ref = opt_fn(x)
            res = fn(x)
            self.assertEqual(ref, res)

    def test_cast_tensor_single_elem(self):
        with torch._dynamo.config.patch({"capture_scalar_outputs": True}):
            for t, val in [
                (float, 1.0),
                (float, 1),
                (float, True),
                (int, 1),
                (int, False),
                # (int, 1.0), # fails due to a >= 0 comparison in sym_int
            ]:  # , bool, complex]: no casting for sym_bool, no sym_complex

                def fn(x):
                    x = x + 1
                    return t(x)

                opt_fn = torch.compile(
                    fn, backend="eager", fullgraph=True, dynamic=False
                )
                x = torch.tensor([val])
                res = fn(x)
                ref = opt_fn(x)
                self.assertEqual(ref, res)

                # Cannot handle non single-elem
                with self.assertRaises(ValueError):
                    fn(torch.tensor([val] * 2))
                with self.assertRaises(torch._dynamo.exc.TorchRuntimeError):
                    opt_fn(torch.tensor([val] * 2))

    def test_set_construction(self):
        def fn(x):
            y = x.add_(1)
            s = set({x})
            s.add(y)
            return len(s)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        res = fn(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)

    def test_is_tensor_tensor(self):
        def fn(x, y):
            if x is y:
                return x * 2
            else:
                return x + y

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        x = torch.zeros(2)
        y = torch.ones(2)

        self.assertEqual(fn(x, y), fn_opt(x, y))
        self.assertEqual(fn(x, x), fn_opt(x, x))

    def test_is_not_tensor_tensor(self):
        def fn(x, y):
            if x is not y:
                return x * 2
            else:
                return x + y

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        x = torch.zeros(2)
        y = torch.ones(2)

        self.assertEqual(fn(x, y), fn_opt(x, y))
        self.assertEqual(fn(x, x), fn_opt(x, x))

    def test_is_mutated_tensor_tensor(self):
        def fn(x):
            y = x.add_(1)
            return x is y

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        z = torch.ones(4)

        self.assertEqual(fn(z), fn_opt(z))

    def test_is_mutated_tensor_tensor_across_graph_break(self):
        def fn(x):
            y = x.add_(1)
            cond = x is y
            x.add_(1)
            # The real tensor values are recovered when graph breaking.
            # Hence we recover the invariant.
            torch._dynamo.graph_break()
            x.add_(1)
            return x is y, cond

        fn_opt = torch.compile(backend="eager", dynamic=True)(fn)

        z = torch.ones(4)

        self.assertEqual(fn(z), fn_opt(z))

    def test_is_mutated_tensor_tensor(self):
        def fn(x):
            y = x.add_(1)
            return y is x

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        z = torch.ones(4, 1)

        self.assertEqual(fn(z), fn_opt(z))

    def test_is_init_in_compile_mutated_tensor_tensor(self):
        def fn(x):
            z = x.clone()
            y = z.add_(1)
            return y is z

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        z = torch.ones(4, 1)

        self.assertEqual(fn(z), fn_opt(z))

    def test_is_init_in_compile_vmapped_mutated_tensor_tensor(self):
        def fn(z):
            x = z.clone()
            y = torch.vmap(torch.Tensor.acos_)(x)
            _ = y is z
            return y is x

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        z = torch.ones(4, 1)

        self.assertEqual(fn(z), fn_opt(z))

    def test_is_vmapped_mutated_tensor_tensor(self):
        def fn(x):
            y = torch.vmap(torch.Tensor.acos_)(x)
            return y is x

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        z = torch.ones(4, 1)

        self.assertEqual(fn(z), fn_opt(z))

    def test_is_init_in_compile_vmapped_mutated_tensor_tensor_multi_arg(self):
        def fn(y, z):
            a = y.clone()
            b = z.clone()

            def g(a, b):
                return a.acos_(), b.acos_()

            c, d = torch.vmap(g)(a, b)
            return a is c is b is d

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        y = torch.ones(4, 2)
        z = torch.ones(4, 10)

        self.assertEqual(fn(y, z), fn_opt(y, z))
        self.assertEqual(fn(y, y), fn_opt(y, y))

    def test_in_set_would_fail_broadcast(self):
        param = torch.zeros(5)
        param2 = torch.zeros(5, 10)

        tensor_list = set()
        tensor_list.add(param2)
        assert param not in tensor_list

        def fn(param, param2):
            param.add_(1)
            tensor_list = set([param2])
            return param in tensor_list

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        self.assertEqual(opt_fn(param, param2), fn(param, param2))
        self.assertEqual(cnts.frame_count, 1)
        # Test aliased
        self.assertEqual(opt_fn(param, param), fn(param, param))
        self.assertEqual(cnts.frame_count, 2)  # Recompiles

    def test_in_set_inplace(self):
        param = torch.zeros(5)
        param2 = torch.zeros(5, 10)

        tensor_list = set()
        tensor_list.add(param2)
        assert param not in tensor_list

        def fn(param, param2):
            y = param.add_(1)  # Tensor method
            z = torch.Tensor.add_(y, 1)  # torch function
            tensor_list = set([param2])
            return y in tensor_list and z in tensor_list

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        self.assertEqual(opt_fn(param, param2), fn(param, param2))
        self.assertEqual(cnts.frame_count, 1)
        # Test aliased
        self.assertEqual(opt_fn(param, param), fn(param, param))
        self.assertEqual(cnts.frame_count, 2)  # Recompiles

    def test_reconstructed_name(self):
        lst = []

        @torch._dynamo.disable
        def disallowed(g):
            lst.append(g.__name__)

        def f():
            def g():
                return ()

            disallowed(g)

        f_opt = torch._dynamo
        opt_f = torch._dynamo.optimize(backend="eager")(f)
        opt_f()
        f()
        self.assertEqual(len(lst), 2)
        self.assertEqual(lst[0], lst[1])

    @unittest.skipIf(
        sys.version_info < (3, 10),
        "zip strict kwargs not implemented for Python < 3.10",
    )
    def test_zip_strict(self):
        def fn(x, ys, zs):
            x = x.clone()
            for y, z in zip(ys, zs, strict=True):
                x += y * z
            return x

        opt_fn = torch._dynamo.optimize(backend="eager")(fn)
        nopython_fn = torch._dynamo.optimize(backend="eager", nopython=True)(fn)

        x = torch.ones(3)
        ys = [1.0, 2.0, 3.0]
        zs = [2.0, 5.0, 8.0]

        self.assertEqual(opt_fn(x, ys, zs), fn(x, ys, zs))

        # If nopython, should raise UserError
        with self.assertRaisesRegex(torch._dynamo.exc.UserError, "zip()"):
            nopython_fn(x, ys[:1], zs)

        # Should cause fallback if allow graph break
        with self.assertRaisesRegex(ValueError, "zip()"):
            opt_fn(x, ys[:1], zs)


instantiate_parametrized_tests(FunctionTests)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
