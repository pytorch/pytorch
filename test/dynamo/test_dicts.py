# Owner(s): ["module: dynamo"]

# ruff: noqa: TRY002
# flake8: noqa

import dataclasses
import gc
import itertools
import types
import unittest
import weakref
from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Optional, Tuple

import torch
import torch._dynamo.config
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._functorch.config
import torch.nn
import torch.utils.checkpoint
from torch._dynamo.testing import same
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase


class SimpleDict(dict):
    pass


class DictTests(torch._dynamo.test_case.TestCase):
    def test_dict_subclass_instantiation(self):
        def fn(x):
            sd = SimpleDict(x=5)
            return sd["x"] * x

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_subclass_local_mutation(self):
        def fn(x):
            sd = SimpleDict(x=5)
            z = sd["x"] * x
            sd["x"] = 10
            return z * sd["x"]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_subclass_local_with_non_dict_method(self):
        # Checks that add_1 method is inlined
        class MethodDict(dict):
            def add_1(self, x):
                return x + 1

        def fn(x):
            sd = MethodDict(x=5)
            z = sd["x"] * x
            sd["x"] = 10
            return sd.add_1(z * sd["x"])

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_contains(self):
        sd = dict()
        sd[2] = 5
        sd[4] = 10

        def fn(x):
            if 1 in sd:
                x = x * 2
            else:
                x = x * 3
            return x

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

        # Ensure a recompilation
        sd[1] = 15
        self.assertEqual(fn(x), opt_fn(x))

        # Ensure not recompilation because the traced program remains same here.
        sd[2] = 10
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            self.assertEqual(fn(x), opt_fn(x))

    def test_dict_subclass_methods_fallback_readonly(self):
        sd = SimpleDict()
        sd[2] = 5
        sd[4] = 10
        # check that regular attr accesses work well
        sd.attr = 4

        def fn(x):
            for value in sd.values():
                x = x * value
            for key in sd.keys():
                x = x * key
            for k, v in sd.items():
                x = x * k
                x = x * v
            # for k in sd:
            #     x = x * k

            if 1 in sd:
                x = x * 2
            else:
                x = x * 3

            x = x * sd.get(2, 0)
            x = x * sd.get(3, 4)
            x = len(sd) * x
            x = x * sd.attr
            return x

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

        # Ensure a recompilation
        sd[6] = 15
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_subclass_instantiation_return(self):
        def fn(x):
            sd = SimpleDict(x=5 * x)
            sd["y"] = 10
            return sd

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(type(ref), type(res))
        self.assertEqual(ref["x"], res["x"])
        self.assertEqual(ref["y"], res["y"])

    def test_dict_subclass_methods_fallback_mutation(self):
        def fn(sd, x):
            for value in sd.values():
                x = x * value
            sd[6] = 14
            for key in sd.keys():
                x = x * key
            for k, v in sd.items():
                x = x * k
                x = x * v
            # for k in sd:
            #     x = x * k

            if 1 in sd:
                x = x * 2
            else:
                x = x * 3

            x = x * sd.get(2, 0)
            x = x * sd.get(3, 4)
            x = len(sd) * x
            return x

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        sd1 = SimpleDict()
        sd1[2] = 5
        sd1[4] = 10

        sd2 = SimpleDict()
        sd2[2] = 5
        sd2[4] = 10
        self.assertTrue(sd1 == sd2)

        self.assertEqual(fn(sd1, x), opt_fn(sd2, x))
        self.assertTrue(sd1 == sd2)

    def test_dict_subclass_setitem(self):
        class SetItemDict(dict):
            def __setitem__(self, key, value):
                super().__setitem__(key, value + 1)

        def fn(x):
            sd = SetItemDict(x=5 * x)
            sd["y"] = 10
            return sd

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(type(ref), type(res))
        self.assertEqual(ref["x"], res["x"])
        self.assertEqual(ref["y"], res["y"])

    def test_custom_iter_dict(self):
        class ReversedDict(dict):
            def __iter__(self):
                return reversed(list(self.keys()))

        d = {
            "foo": 1,
            "bar": 2,
        }

        d = ReversedDict(d)

        @torch.compile(backend="eager")
        def fn(x, d):
            # Forces side effects attribute reapplication logic
            d.sample = 1
            d["baz"] = 4
            return x * d["foo"] * d["bar"]

        fn(torch.randn(4), d)
        # This is intentional because the dict is mutated, so we will have a recompilation.
        fn(torch.randn(4), d)
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(4), d)

    def test_custom_keys_iter_dict(self):
        class ReversedDict(dict):
            def keys(self):
                return ["bar", "foo"]

        d = {
            "foo": 1,
            "bar": 2,
        }

        d = ReversedDict(d)

        @torch.compile(backend="eager")
        def fn(x, d):
            return x * d["foo"] * d["bar"]

        fn(torch.randn(4), d)
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(4), d)

    def test_dict_guard_on_keys_order(self):
        d = {
            2: 4,
            3: 5,
        }

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, d):
            for key, value in d.items():
                x = x * key + value
            return x

        opt_fn = torch.compile(fn, backend=cnts)
        opt_fn(torch.randn(4), d)
        opt_fn(torch.randn(4), d)
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # move 2 to the end
        d[2] = d.pop(2)

        x = torch.randn(4)
        res = opt_fn(x, d)
        # Check recompilation
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(res, fn(x, d))

    def test_dict_guard_on_keys_order2(self):
        d = {
            2: 4,
            3: 5,
        }

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, d):
            for key in d:
                value = d[key]
                x = x * key + value
            return x

        opt_fn = torch.compile(fn, backend=cnts)
        opt_fn(torch.randn(4), d)
        opt_fn(torch.randn(4), d)
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # move 2 to the end
        d[2] = d.pop(2)

        x = torch.randn(4)
        res = opt_fn(x, d)
        # Check recompilation
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(res, fn(x, d))

    def test_ordered_dict_reordered_keys(self):
        d = OrderedDict()
        d[2] = 4
        d[3] = 5
        d.move_to_end(2)

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, d):
            y = 0
            for idx, (key, value) in enumerate(d.items()):
                if idx == 0:
                    y += torch.sin(x * value)
                else:
                    y += torch.cos(x * value)
            return y

        opt_fn = torch.compile(fn, backend=cnts)
        x = torch.randn(4)
        self.assertEqual(opt_fn(x, d), fn(x, d))

    def test_ordered_dict_subclass_reordered_keys(self):
        class ODSubclass(OrderedDict):
            def keys(self):
                return super().keys()

        d = ODSubclass()
        d[2] = 4
        d[3] = 5
        d.move_to_end(2)

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, d):
            y = 0
            for idx, (key, value) in enumerate(d.items()):
                if idx == 0:
                    y += torch.sin(x * value)
                else:
                    y += torch.cos(x * value)
            return y

        opt_fn = torch.compile(fn, backend=cnts)
        x = torch.randn(4)
        self.assertEqual(opt_fn(x, d), fn(x, d))

    def test_lazy_key_guarding(self):
        d = {"a": 2, "b": 3, "c": 5}

        def fn(x):
            return x * d["a"]

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

        # Since key c was not used, it should not lead to a recompilation
        d.pop("c")
        d["d"] = 10

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)

    def test_lazy_key_non_const_guarding(self):
        d = {
            list: 2,
            dict: 3,
            OrderedDict: 5,
            namedtuple: 7,
        }

        def fn(x):
            return x * d[list]

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

        # Since key c was not used, it should not lead to a recompilation
        d.pop(dict)
        d[defaultdict] = 10

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)

    def test_dict_mutation_side_effect(self):
        def fn(d):
            d["c"] = d["a"] + d.pop("b")
            return d

        args1 = {"a": torch.randn(10), "b": torch.randn(10)}
        args2 = dict(args1)
        assert fn(args1) is args1
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertIs(opt_fn(args2), args2)
        self.assertTrue(same(args1, args2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

    def test_dict_copy_alias(self):
        @torch.compile(backend="eager", fullgraph=True)
        def run(x, d0):
            d1 = d0.copy()
            d1[0] = 1
            return x + 1, d1

        d0 = {}
        res, d1 = run(torch.zeros(1), d0)
        self.assertTrue(same(res, torch.ones(1)))
        self.assertEqual(d0, {})
        self.assertEqual(d1, {0: 1})

    def test_dict_subclass_get_method(self):
        class dotdict(dict):
            """dot.notation access to dictionary attributes"""

            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        config = dotdict({"a": 1, "b": 2})

        def fn(x):
            x2 = x * 2
            x3 = x * config.get("a", 3)
            return x3

        x = torch.randn(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_order_keys(self):
        def fn(d):
            c = 0
            for v in d.values():
                c += v
            return c

        args1 = {}
        args1["a"] = torch.rand(10)
        args1["b"] = torch.rand(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(fn(args1), opt_fn(args1))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

        # A different order of keys recompiles
        args2 = {}
        args2["b"] = args1["b"]
        args2["a"] = args1["a"]
        self.assertEqual(fn(args2), opt_fn(args2))
        self.assertEqual(cnts.frame_count, 2)
        # Extra calls don't recompile
        self.assertEqual(cnts.frame_count, 2)

    def test_dict_namedtuple(self):
        def fn(d):
            if namedtuple in d:
                return d[3] * 2
            else:
                return d[3] * 3

        args1 = {namedtuple: None, 3: torch.randn(3)}
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(fn(args1), opt_fn(args1))
        self.assertEqual(cnts.frame_count, 1)
        # Test a failing namedtuple guard
        args2 = {2: None, 3: torch.randn(3)}
        self.assertEqual(fn(args2), opt_fn(args2))
        self.assertEqual(cnts.frame_count, 2)

    def test_dict_order_keys_tensors(self):
        def fn(d, x):
            return d[x] + 3

        args1 = {}
        x = torch.randn(10)
        y = torch.randn(10)
        z = torch.randn(10)
        args1[x] = y
        args1[3] = z

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(fn(args1, x), opt_fn(args1, x))
        self.assertEqual(cnts.frame_count, 1)

        # Calling again doesn't recompile (same id and key order)
        opt_fn(args1, x)
        self.assertEqual(cnts.frame_count, 1)
        args2 = {}
        args2[3] = z
        args2[x] = y

        # Different order recompiles
        self.assertEqual(fn(args2, x), opt_fn(args2, x))
        self.assertEqual(cnts.frame_count, 2)

    def test_dict_order_keys_modules(self):
        def fn(d, x):
            return d[x](torch.ones(2, 2))

        args1 = {}
        x = torch.nn.Linear(2, 2)
        y = torch.nn.Linear(2, 2)
        z = torch.nn.Linear(2, 2)
        args1[x] = y
        args1[3] = z

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(fn(args1, x), opt_fn(args1, x))
        self.assertEqual(cnts.frame_count, 1)

        # Calling again doesn't recompile (same id and key order)
        opt_fn(args1, x)
        self.assertEqual(cnts.frame_count, 1)
        args2 = {}
        args2[3] = z
        args2[x] = y

        # Different order recompiles
        self.assertEqual(fn(args2, x), opt_fn(args2, x))
        self.assertEqual(cnts.frame_count, 2)

    def test_contains_dunder_dict(self):
        class UserDefined:
            def __init__(self) -> None:
                self.a = 3
                self.b = 5

            def run(self, x):
                if "a" in self.__dict__:
                    x = x * self.a
                if "b" in self.__dict__:
                    x = x * self.b
                self.c = 7
                if "c" in self.__dict__:
                    x = x * self.c
                return x * self.__dict__.get("a") * self.__dict__.get("z", 2)

        obj = UserDefined()

        def fn(x):
            return obj.run(x)

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_contains_module_dunder_dict(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = 1
                self.bar = 2
                self.baz = 3

            def forward(self, x):
                if "foo" in self.__dict__:
                    return x * self.bar
                return x * self.baz

        mod = MyModule()
        x = torch.randn(10)
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        self.assertEqual(mod(x), opt_mod(x))

    def test_update_dunder_dict(self):
        class UserDefined:
            def run(self, x):
                self.__dict__["a"] = 10
                return x * self.a + self.__dict__["a"]

        obj1 = UserDefined()
        obj2 = UserDefined()

        def fn(x, obj):
            return obj.run(x)

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x, obj1)
        res = opt_fn(x, obj2)
        self.assertEqual(ref, res)
        # Make sure only `a` is updated.
        self.assertEqual(obj1.__dict__, obj2.__dict__)

    def test_update_module_dunder_dict(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                self.__dict__["a"] = 10
                return x * self.a + self.__dict__["a"]

        mod = MyModule()
        x = torch.randn(10)
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        self.assertEqual(mod(x), opt_mod(x))

    def test_dict_reconstruct_keeps_original_order(self):
        def fn():
            modules = OrderedDict([("act", torch.nn.ReLU())])
            module_dict = torch.nn.ModuleDict(modules)

            next_modules = {"fc4": torch.nn.Linear(5, 6), "act3": torch.nn.Sigmoid()}
            modules.update(next_modules.items())
            module_dict.update(next_modules)
            return modules, module_dict

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        modules, module_dict = opt_fn()

        self.assertEqual(len(module_dict), len(modules))
        for k1, m2 in zip(modules, module_dict.children()):
            self.assertTrue(modules[k1] is m2)

    def test_dict_subclass_initialization_in_graph(self):
        for super_class in (
            OrderedDict,
            dict,
        ):

            class CustomDict(super_class):
                def __new__(self, *args, **kwargs):
                    return super().__new__(self, *args, **kwargs)

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)

            def fn(x):
                c = CustomDict()
                c["key"] = x
                assert "key" in c
                return c["key"] + 1

            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

            x = torch.rand(4)
            self.assertEqual(fn(x), opt_fn(x))

    def test_dict_list_values(self):
        def inner_fn(args):
            return [x[1].shape for x in args]

        @torch.compile(backend="eager")
        def fn(tensors):
            return inner_fn(zip(itertools.count(), tensors["args"]))

        fn({"args": [torch.ones(5, 5), torch.ones(5, 6), torch.ones(5, 7)]})
        fn({"args": [torch.ones(5, 5)]})

    def test_dict_iter(self):
        class MyMod(torch.nn.Module):
            def forward(self, x):
                z = {"my": 1, "const": 2, "dict": 3, "variable": 4}
                tot = 0
                for key in z:
                    tot += z[key]

                return tot

        x = torch.tensor([0])
        model = MyMod()
        opt_model = torch.compile(model, backend="eager", fullgraph=True)
        y = opt_model(x)

        self.assertEqual(y, 10)

    def test_dict_subclass_contains(self):
        # pattern from huggingface
        class ClassInstantier(OrderedDict):
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def f(x, d):
            if "key1" in d:
                x = x + 2
            if "key2" in d:
                x = x + 4
            x = x + 8
            return x

        result = f(torch.ones(8), ClassInstantier({"key1": torch.ones(8)}))
        self.assertTrue(same(result, torch.full([8], 11.0)))

        result = f(torch.ones(8), ClassInstantier({"key2": torch.ones(8)}))
        self.assertTrue(same(result, torch.full([8], 13.0)))

    def test_dict_tag_guard(self):
        class Foo:
            def __init__(self) -> None:
                self.scalar = 10

        def fn(d, x):
            return d["a"] * d["b"] * d["c"].scalar * x

        foo = Foo()

        d = {"a": 2, "b": 3, "c": foo}

        opt_fn = torch.compile(fn, backend="eager")
        inp = torch.randn(3, 3)
        self.assertEqual(fn(d, inp), opt_fn(d, inp))

        d["a"] = 4
        self.assertEqual(fn(d, inp), opt_fn(d, inp))

        # Check that recompilation happens
        foo.scalar = 12
        self.assertEqual(fn(d, inp), opt_fn(d, inp))

    def test_empty_dict_recompilation(self):
        def fn(d, x):
            if d:
                return torch.cos(x)
            return torch.sin(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn({}, x), opt_fn({}, x))
        self.assertEqual(fn({"a": 1}, x), opt_fn({"a": 1}, x))

    def test_udf_dict_reconstruction(self):
        class MyDict(dict):
            pass

        def fn(x, klass):
            x = x * 2
            sc_dict = dict.__new__(klass)
            sc_dict["x"] = x
            if isinstance(sc_dict, MyDict):
                sc_dict.attr = 3
            return sc_dict

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x, MyDict)
        res = opt_fn(x, MyDict)
        self.assertEqual(ref, res)
        self.assertTrue(isinstance(res, MyDict))
        self.assertEqual(ref.attr, res.attr)

        ref = fn(x, dict)
        res = opt_fn(x, dict)
        self.assertEqual(ref, res)
        self.assertTrue(isinstance(res, dict))

    def test_weakref_dict(self):
        states = weakref.WeakKeyDictionary()

        mod1 = torch.nn.Module()
        mod2 = torch.nn.Module()

        states[mod1] = 2
        states[mod2] = 3

        def fn(x):
            if mod1 in states:
                x = torch.sin(x)
            if mod2 in states:
                x = torch.cos(x)
            return x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_fn_id(self):
        def fn(x, f):
            d = {id(f): 3}
            return x * d[id(f)]

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)

        def nothing():
            pass

        f = nothing
        self.assertEqual(fn(x, f), opt_fn(x, f))

    def test_mapping_proxy_for_local(self):
        def fn(x):
            d = {"a": 2, "b": 3, "c": 5 * x}
            mp = types.MappingProxyType(d)
            y = torch.sin(x * mp["a"])
            for k, v in mp.items():
                y += torch.cos(x * v)
            return mp

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertTrue(type(res) is types.MappingProxyType)

    def test_mapping_proxy_for_nonlocal(self):
        d = {"a": 2, "b": 3, "c": 5}

        def fn(x):
            mp = types.MappingProxyType(d)
            y = torch.sin(x * mp["a"])
            for k, v in mp.items():
                y += torch.cos(x * v)
            d["d"] = 4
            return mp

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertTrue(type(res) is types.MappingProxyType)

        # check update to d is reflected in res
        d["e"] = 5
        self.assertEqual(d["e"], res["e"])

    def test_move_to_end(self):
        def fn(x):
            d = OrderedDict({"a": torch.cos(x), "b": 3, "c": 5})
            d.move_to_end("a")
            return d

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(["b", "c", "a"], list(opt_fn(x).keys()))
        self.assertEqual(fn(x), opt_fn(x))

    def test_overridden_get_item(self):
        class MyDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.calls = 0

            def __getitem__(self, key):
                self.calls += 1
                return super().__getitem__(key) + 1

        def fn(x, d):
            d["d"] = 4
            return x * d["a"] + d["b"] + d["c"] + d["d"]

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        d1 = MyDict({"a": 2, "b": 3, "c": 5})
        ref = fn(x, d1)

        d2 = MyDict({"a": 2, "b": 3, "c": 5})
        res = opt_fn(x, d2)
        self.assertEqual(ref, res)
        self.assertEqual(d1.calls, d2.calls)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
