# Owner(s): ["module: dynamo"]
"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_export_persist_assert)
"""
import functools
import inspect
import math
import operator
import unittest
from enum import Enum
from typing import Dict, List, Sequence
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from functorch.experimental.control_flow import cond
from torch._dynamo import config
from torch._export import dynamic_dim
from torch._export.constraints import constrain_as_size, constrain_as_value
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ConstraintViolationError
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import skipIfRocm


class ExportTests(torch._dynamo.test_case.TestCase):
    # TODO(voz): Refactor to a shared test function.
    # The tests in this file are a little redundant,
    # They all take a func, run it with eager, then export it, then compare
    def test_export(self):
        def pre_attention_state_ops(input, mems, state):
            lc_key = state[0]
            lc_val = state[1]
            bar = []
            for i in range(0, 4):
                bar2 = []
                for j in range(0, 3):
                    bar2.append(
                        lc_key + lc_val + torch.tensor([0.1, 0.25, 0.4, 0.5, 0.1])
                    )
                bar.append(bar2)

            return bar

        def func():
            mems = torch.tensor([[[1.8364, 0.2724, -1.4917, -0.4367, 0.8640]]])
            state = [
                torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
                torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
            ]
            i = torch.tensor(
                [
                    [0.0313, -0.1487, -0.3846, -0.5321],
                    [-1.7073, 1.3331, -0.0890, -1.4935],
                    [-0.8314, -0.1862, -0.5935, 1.5232],
                ]
            )
            return pre_attention_state_ops(i, mems, state)

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func()

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)
        out_graph = exported[0]

        dynamo_result = out_graph()
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_mismatched_out(self):
        def func(x):
            y = x + 1
            return ([x, x], (y, y))

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(torch.tensor([[[1.3737, 0.1]]]))

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, torch.tensor([[[1.3737, 0.1]]]))
        out_graph = exported[0]

        dynamo_result = out_graph(torch.tensor([[[1.3737, 0.1]]]))

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_shape_control_flow_1(self):
        def func(x):
            if x.shape[0] > 10:
                return x.cos()
            return x.sin()

        opt_func = torch._dynamo.optimize("eager")(func)
        real_result = opt_func(torch.ones(6, 4))

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, torch.ones(6, 4))
        out_graph, out_guards = exported

        dynamo_result = out_graph(torch.ones(6, 4))

        from torch._guards import GuardSource

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))
        hit = False
        for guard in out_guards:
            if guard.source == GuardSource.SHAPE_ENV:
                hit = True
                self.assertTrue("L['x'].size()[0] <= 10" in guard.code_list)

        self.assertTrue(hit)

    def test_export_control_flow_with_getattr(self):
        class Animal(Enum):
            COW = "moo"

        class MyModule(torch.nn.Module):
            def __init__(self, a):
                super().__init__()
                self.a = a

            def forward(self, x):
                if self.a == Animal.COW.value:
                    return x * x
                else:
                    raise ValueError("bad")

        module = MyModule("moo")
        input = (torch.ones(4, 3),)
        resA = module(*input)
        graph, _ = torch._dynamo.export(module, *input)
        resB = graph(*input)
        self.assertTrue(torch._dynamo.utils.same(resA, resB))

    def test_export_graph_bypass(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
        ]

        def func(x):
            first = x[2]
            second = x[2]
            return first * second

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, inp)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_list_unpack(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
        ]

        def func(x):
            first = x[2]
            second = x[2]
            return x[0], first * second, x[1], x[2]

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, inp)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_with_shallow_list_copy_wo_side_effects(self):
        def f(x):
            y = x.copy()
            return y[0] + y[1]

        inp = [torch.tensor([1.3, 3.77, 0.1]), torch.tensor([8.7, 6.23, 9.9])]
        gm, _ = torch._dynamo.export(f, inp, aten_graph=True, tracing_mode="symbolic")
        self.assertTrue(torch._dynamo.utils.same(gm(inp), f(inp)))

    def test_export_with_shallow_list_copy_with_side_effects(self):
        def f(x):
            y = x.copy()
            x[0] = x[1]
            y.append(torch.tensor([[100]]))
            return x[0] + x[1], y[0] + y[1], y[2]

        inp = [torch.tensor([1.3, 3.77, 0.1]), torch.tensor([8.7, 6.23, 9.9])]
        gm, _ = torch._dynamo.export(f, inp, aten_graph=True, tracing_mode="symbolic")
        res = gm(inp)
        ref = f(inp)
        self.assertTrue(torch._dynamo.utils.same(res, ref))
        self.assertEqual(res[0], res[1])

    def test_export_mismatched_out_2(self):
        def func(x):
            y = x + 1
            return ([x, x], (y, y))

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(torch.tensor([[[1.3737, 0.1]]]))

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, torch.tensor([[[1.3737, 0.1]]]))
        out_graph = exported[0]

        dynamo_result = out_graph(torch.tensor([[[1.3737, 0.1]]]))

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_graph_with_list(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
            torch.tensor([0.4, 0.4]),
        ]

        def func(x):
            first = x[2]
            second = x[2]
            return first * second, x

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, inp)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_graph_with_complex_reorder(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
            torch.tensor([0.4, 0.4]),
        ]

        def func(x):
            first = x[0]
            second = x[1]
            third = x[2]
            return third, first, second, first * second, first * third

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, inp)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dupes(self):
        inp = torch.tensor([0.1, 0.1])

        def func(x):
            y = x + 1
            return y, y

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, inp)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dupes_2(self):
        inp = torch.tensor([0.1, 0.1])

        def func(x):
            y = x + 1
            return y, y

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, inp)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dupes_and_bypass(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.4, 0.4])
        inps = [inp, inp2]

        def func(x, z):
            y = x + 1
            return y, y, z

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, *inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dupes_and_bypass_with_non_tensor_arg(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        def func(x, z, k):
            y = x + k
            return y, y, z

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, *inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dupes_and_bypass_reorder_with_non_tensor_arg(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        def func(x, z, k):
            y = x + k
            return z, y, y

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, *inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    @config.patch(capture_scalar_outputs=True)
    def test_dupes_and_bypass_with_non_tensor_output(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        def func(x, z, k):
            y = x + k
            return y[0].item(), y, z

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, *inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_zeroes_in_and_out_different_shape_on_test(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            return [[a], [b, c], [a + b], [[c + c]]]

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(*inps_rand)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, *inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps_rand)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    @config.patch(capture_scalar_outputs=True)
    def test_zeroes_in_new_shape_scalar_out(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            return a[0].item() + b[0].item() + c[0].item()

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(*inps_rand)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, *inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps_rand)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    @config.patch(capture_scalar_outputs=True)
    def test_zeroes_in_new_shape_scalar_out_permute(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            return b[0].item() + c[0].item() + a[0].item() + a[0].item()

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(*inps_rand)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, *inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps_rand)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    @config.patch(capture_scalar_outputs=True)
    def test_zeroes_in_new_shape_scalar_out_permute_dupe_and_bypass(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            return a, b[0].item() + c[0].item() + a[0].item() + a[0].item(), a

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(*inps_rand)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, *inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps_rand)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_func_return(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            x = a + b + c

            def func2(y):
                return x * y

            return func2(x)

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(*inps_rand)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, *inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps_rand)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dict_return(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            x = a + b + c
            return {"a": x}

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(*inps_rand)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, *inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps_rand)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_with_aten_graph(self):
        def pre_attention_state_ops(input, mems, state):
            lc_key = state[0]
            lc_val = state[1]
            bar = []
            for i in range(0, 4):
                bar2 = []
                for j in range(0, 3):
                    bar2.append(
                        lc_key + lc_val + torch.tensor([0.1, 0.25, 0.4, 0.5, 0.1])
                    )
                bar.append(bar2)

            return bar

        def func():
            mems = torch.tensor([[[1.8364, 0.2724, -1.4917, -0.4367, 0.8640]]])
            state = [
                torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
                torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
            ]
            i = torch.tensor(
                [
                    [0.0313, -0.1487, -0.3846, -0.5321],
                    [-1.7073, 1.3331, -0.0890, -1.4935],
                    [-0.8314, -0.1862, -0.5935, 1.5232],
                ]
            )
            return pre_attention_state_ops(i, mems, state)

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func()

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)
        out_graph = exported[0]

        dynamo_result = out_graph()
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_mismatched_out_with_aten_graph(self):
        def func(x):
            y = x + 1
            return ([x, x], (y, y))

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(torch.tensor([[[1.3737, 0.1]]]))

        torch._dynamo.reset()

        exported = torch._dynamo.export(
            func, torch.tensor([[[1.3737, 0.1]]]), aten_graph=True
        )
        out_graph = exported[0]

        dynamo_result = out_graph(torch.tensor([[[1.3737, 0.1]]]))

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_graph_bypass_with_aten_graph(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
        ]

        def func(x):
            first = x[2]
            second = x[2]
            return first * second

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, inp, aten_graph=True)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_list_unpack_with_aten_graph(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
        ]

        def func(x):
            first = x[2]
            second = x[2]
            return x[0], first * second, x[1], x[2]

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, inp, aten_graph=True)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_mismatched_out_2_with_aten_graph(self):
        def func(x):
            y = x + 1
            return ([x, x], (y, y))

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(torch.tensor([[[1.3737, 0.1]]]))

        torch._dynamo.reset()

        exported = torch._dynamo.export(
            func, torch.tensor([[[1.3737, 0.1]]]), aten_graph=True
        )
        out_graph = exported[0]

        dynamo_result = out_graph(torch.tensor([[[1.3737, 0.1]]]))

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_graph_with_list_with_aten_graph(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
            torch.tensor([0.4, 0.4]),
        ]

        def func(x):
            first = x[2]
            second = x[2]
            return first * second, x

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, inp, aten_graph=True)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_graph_with_complex_reorder_with_aten_graph(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
            torch.tensor([0.4, 0.4]),
        ]

        def func(x):
            first = x[0]
            second = x[1]
            third = x[2]
            return third, first, second, first * second, first * third

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, inp, aten_graph=True)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dupes_with_aten_graph(self):
        inp = torch.tensor([0.1, 0.1])

        def func(x):
            y = x + 1
            return y, y

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, inp, aten_graph=True)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dupes_2_with_aten_graph(self):
        inp = torch.tensor([0.1, 0.1])

        def func(x):
            y = x + 1
            return y, y

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, inp, aten_graph=True)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dupes_and_bypass_with_aten_graph(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.4, 0.4])
        inps = [inp, inp2]

        def func(x, z):
            y = x + 1
            return y, y, z

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, *inps, aten_graph=True)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dupes_and_bypass_with_non_tensor_arg_with_aten_graph(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        def func(x, z, k):
            y = x + k
            return y, y, z

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, *inps, aten_graph=True)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dupes_and_bypass_reorder_with_non_tensor_arg_with_aten_graph(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        def func(x, z, k):
            y = x + k
            return z, y, y

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, *inps, aten_graph=True)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    @config.patch(capture_scalar_outputs=True)
    def test_dupes_and_bypass_with_non_tensor_output_with_aten_graph(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        def func(x, z, k):
            y = x + k
            return y[0].item(), y, z

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, *inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_zeroes_in_and_out_different_shape_on_test_with_aten_graph(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            return [[a], [b, c], [a + b], [[c + c]]]

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(*inps_rand)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, *inps, aten_graph=True)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps_rand)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_func_return_with_aten_graph(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            x = a + b + c

            def func2(y):
                return x * y

            return func2(x)

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(*inps_rand)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, *inps, aten_graph=True)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps_rand)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dict_return_with_aten_graph(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            x = a + b + c
            return {"a": x}

        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(*inps_rand)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, *inps, aten_graph=True)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps_rand)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_with_stack_trace(self):
        inp = torch.randn(4, 4)

        class MyBlock(torch.nn.Module):
            def forward(self, x):
                x = torch.nn.functional.linear(x, torch.randn(4, 4))
                return torch.cos(x).relu() + 1

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block = MyBlock()

            def forward(self, x):
                out = self.block(x)
                return out

        exported = torch._dynamo.export(MyModule(), inp, aten_graph=False)
        out_graph = exported[0]

        for node in out_graph.graph.nodes:
            if node.op not in {"placeholder", "output"}:
                self.assertTrue(node.stack_trace is not None)
                self.assertTrue(node.meta["nn_module_stack"] is not None)
                self.assertTrue(node.meta["source_fn"] is not None)

        torch._dynamo.reset()

        exported = torch._dynamo.export(MyModule(), inp, aten_graph=True)
        out_graph = exported[0]
        for node in out_graph.graph.nodes:
            if node.op == "call_function":
                self.assertTrue(node.stack_trace is not None)
                self.assertTrue(node.meta["nn_module_stack"] is not None)
                self.assertTrue(node.meta["source_fn"] is not None)
                self.assertTrue(node.meta["val"] is not None)
                self.assertTrue(node.meta["original_aten"] is not None)

    def test_export_preserves_nn_module_stack_for_get_attr(self):
        inp = torch.randn(4, 4)

        class MyBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1, 1))
                self.register_buffer("buffer", torch.ones(1, 1))

            def forward(self, x):
                x = torch.nn.functional.linear(x, torch.randn(4, 4))
                return torch.cos(x).relu() + self.weight + self.buffer

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block = MyBlock()

            def forward(self, x):
                out = self.block(x)
                return out

        m = MyModule()
        exported = torch._dynamo.export(m, inp, aten_graph=False)
        out_graph = exported[0]

        attr_access_count = 0
        for node in out_graph.graph.nodes:
            if node.op == "get_attr":
                attr_access_count += 1
                self.assertTrue(node.meta["nn_module_stack"] is not None)
        self.assertEqual(attr_access_count, 2)

        torch._dynamo.reset()

        exported = torch._dynamo.export(m, inp, aten_graph=True)
        out_graph = exported[0]

        attr_access_count = 0
        for node in out_graph.graph.nodes:
            if node.op == "get_attr":
                attr_access_count += 1
                self.assertTrue(node.meta["nn_module_stack"] is not None)
        self.assertEqual(attr_access_count, 2)

    def test_export_compare_optimize_with_make_fx(self):
        inp = torch.tensor([0.1, 0.1])
        linear = torch.nn.Linear(2, 2)

        def func(x):
            x = x + 1
            y = x.t()
            y = y.relu()
            y = linear(y)
            return y

        exported = torch._dynamo.export(func, inp, aten_graph=True)
        out_graph = exported[0]
        export_result = out_graph(inp)

        torch._dynamo.reset()

        def compiler(gm, sample_inputs):
            def fw(*args):
                aten_gm = make_fx(gm)(*args)
                return aten_gm(*args)

            return fw

        opt_func = torch._dynamo.optimize(compiler, nopython=True, dynamic=True)(func)
        make_fx_result_through_backend = opt_func(inp)

        fx_g = make_fx(func)(inp)
        make_fx_result_through_direct = fx_g(inp)

        self.assertTrue(
            torch._dynamo.utils.same(make_fx_result_through_backend, export_result)
        )
        self.assertTrue(
            torch._dynamo.utils.same(make_fx_result_through_direct, export_result)
        )

    def test_export_with_constant_method_on_module(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 2))
                self.linear = torch.nn.Linear(2, 2)

            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                return torch.nonzero(x)

            def forward(self, x):
                y = torch.sin(x)
                x = self.linear(x)
                y = self.helper_fn(x)
                return y

        module = MyModule()
        real_result = module(torch.tensor([[1.0, 0], [0, 0]]))
        module = MyModule()
        graph, _ = torch._dynamo.export(module, torch.tensor([[0.0, 0], [0, 0]]))
        result = graph(torch.tensor([[1.0, 0.0], [0, 0]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        result = graph(torch.tensor([[1, 0], [0.25, 0.25]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_method_on_module_invoke_twice(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 2))
                self.linear = torch.nn.Linear(2, 2)

            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                return torch.nonzero(x)

            def forward(self, x):
                y = torch.sin(x)
                x = self.linear(x)
                y = self.helper_fn(x) + self.helper_fn(x)
                return y

        module = MyModule()
        real_result = module(torch.tensor([[1.0, 0], [0, 0]]))
        module = MyModule()
        graph, _ = torch._dynamo.export(module, torch.tensor([[0.0, 0], [0, 0]]))
        result = graph(torch.tensor([[1.0, 0.0], [0, 0]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        result = graph(torch.tensor([[1, 0], [0.25, 0.25]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_free_function(self):
        @torch._dynamo.assume_constant_result
        def helper_fn(x):
            return torch.nonzero(x)

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 2))
                self.linear = torch.nn.Linear(2, 2)

            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                return torch.nonzero(x)

            def forward(self, x):
                y = torch.sin(x)
                x = self.linear(x)
                y = helper_fn(x) + self.helper_fn(x)
                return y

        module = MyModule()
        real_result = module(torch.tensor([[1.0, 0], [0, 0]]))
        module = MyModule()
        graph, _ = torch._dynamo.export(module, torch.tensor([[0.0, 0], [0, 0]]))
        result = graph(torch.tensor([[1.0, 0.0], [0, 0]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        result = graph(torch.tensor([[1, 0], [0.25, 0.25]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_free_function_and_class_method(self):
        @torch._dynamo.assume_constant_result
        def helper_fn(x):
            return torch.nonzero(x)

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 2))
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                y = torch.sin(x)
                x = self.linear(x)
                y = helper_fn(x)
                return y

        module = MyModule()
        real_result = module(torch.tensor([[1.0, 0], [0, 0]]))
        module = MyModule()
        graph, _ = torch._dynamo.export(module, torch.tensor([[0.0, 0], [0, 0]]))
        result = graph(torch.tensor([[1.0, 0.0], [0, 0]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        result = graph(torch.tensor([[1, 0], [0.25, 0.25]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_free_function_and_class_method_multiarg(self):
        @torch._dynamo.assume_constant_result
        def helper_fn(x):
            return torch.nonzero(x)

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 2))
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x, z):
                y = torch.sin(x)
                x = self.linear(x)
                y = helper_fn(x) + helper_fn(z)
                return y

        module = MyModule()
        real_result = module(
            torch.tensor([[1.0, 0], [0, 0]]), torch.tensor([[1.0, 0], [0, 0]])
        )
        module = MyModule()
        graph, _ = torch._dynamo.export(
            module, torch.tensor([[0.0, 0], [0, 0]]), torch.tensor([[1.0, 0], [0, 0]])
        )
        result = graph(
            torch.tensor([[1.0, 0.0], [0, 0]]), torch.tensor([[1.0, 0.0], [0, 0]])
        )
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        result = graph(
            torch.tensor([[1, 0], [0.25, 0.25]]), torch.tensor([[1, 0], [0.25, 0.25]])
        )
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_free_function_and_class_method_multiarg_diff(self):
        @torch._dynamo.assume_constant_result
        def helper_fn(x):
            return torch.nonzero(x)

        class MyModule(torch.nn.Module):
            def forward(self, x, z):
                y = helper_fn(x) + helper_fn(z)
                return y

        module = MyModule()
        real_result = module(
            torch.tensor([[1.0, 0], [0, 0]]), torch.tensor([[1.0, 0], [0, 0]])
        )
        module = MyModule()
        graph, _ = torch._dynamo.export(
            module, torch.tensor([[0.0, 0], [0, 0]]), torch.tensor([[0.0, 0], [0.5, 0]])
        )
        result = graph(
            torch.tensor([[1.0, 0.0], [0, 0]]), torch.tensor([[0.0, 1.0], [0, 0]])
        )
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        result = graph(
            torch.tensor([[1, 0], [0.25, 0.25]]),
            torch.tensor([[0.33, 0.33], [0.25, 0.25]]),
        )
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_tuple_nonzero(self):
        class MyModule(torch.nn.Module):
            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                return (torch.nonzero(x), torch.nonzero(x))

            def forward(self, x):
                y = torch.tensor([0.5])
                elements = self.helper_fn(x)
                all_y = []
                for element in elements:
                    for item in element:
                        all_y.append(y * item)
                return all_y

        module = MyModule()
        real_result = module(torch.tensor([1.0, 1.0]))
        graph, guards = torch._dynamo.export(module, torch.tensor([1.0, 1.0]))

        # Tensor input can be almost anything here, and the result will capture what we
        # made constant at compile time.
        result = graph(torch.tensor([[[1.0, 0], [0, 0]], [[1.0, 0], [0, 0]]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_list_nonzero(self):
        class MyModule(torch.nn.Module):
            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                return [torch.nonzero(x), torch.nonzero(x)]

            def forward(self, x):
                y = torch.tensor([0.5])
                elements = self.helper_fn(x)
                all_y = []
                for element in elements:
                    for item in element:
                        all_y.append(y * item)
                return all_y

        module = MyModule()
        real_result = module(torch.tensor([1.0, 1.0]))
        graph, guards = torch._dynamo.export(module, torch.tensor([1.0, 1.0]))

        # Tensor input can be almost anything here, and the result will capture what we
        # made constant at compile time.
        result = graph(torch.tensor([[[1.0, 0], [0, 0]], [[1.0, 0], [0, 0]]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_list_nonzero_free_function(self):
        @torch._dynamo.assume_constant_result
        def helper_fn(x):
            return [torch.nonzero(x), torch.nonzero(x)]

        class MyModule(torch.nn.Module):
            def forward(self, x):
                y = torch.tensor([0.5])
                elements = helper_fn(x)
                all_y = []
                for element in elements:
                    for item in element:
                        all_y.append(y * item)
                return all_y

        module = MyModule()
        real_result = module(torch.tensor([1.0, 1.0]))
        graph, guards = torch._dynamo.export(module, torch.tensor([1.0, 1.0]))

        # Tensor input can be almost anything here, and the result will capture what we
        # made constant at compile time.
        result = graph(torch.tensor([[[1.0, 0], [0, 0]], [[1.0, 0], [0, 0]]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_dict_values(self):
        class MyModule(torch.nn.Module):
            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                return {"x": x, "x^2": x * x}

            def forward(self, x):
                y = torch.tensor([0.5])
                elements = self.helper_fn(x)
                y = y * elements["x"]
                y = y * elements["x^2"]
                return y

        module = MyModule()
        real_result = module(torch.tensor([2.0, 2.0]))
        graph, guards = torch._dynamo.export(module, torch.tensor([2.0, 2.0]))

        # Tensor input can be almost anything here, and the result will capture what we
        # made constant at compile time.
        result = graph(torch.tensor([[[1.0, 0], [0, 0]], [[1.0, 0], [0, 0]]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_none_control_flow(self):
        class MyModule(torch.nn.Module):
            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                if x.item() < 0:
                    return None
                else:
                    return x

            def forward(self, x):
                y = torch.tensor([0.5])
                x = self.helper_fn(x)
                if x is None:
                    return y
                return y * x

        module = MyModule()
        real_result = module(torch.tensor([-1]))

        # X is negative, so .item() < 0, which means we return y
        self.assertEqual(real_result, torch.tensor([0.5]))

        graph, guards = torch._dynamo.export(module, torch.tensor([-1]))
        result = graph(torch.tensor([2]))
        # X is positive, but we compiled helper_fn to return None, so it will still return y
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_not_none_control_flow(self):
        class MyModule(torch.nn.Module):
            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                if x.item() < 0:
                    return None
                else:
                    return x

            def forward(self, x):
                y = torch.tensor([0.5])
                x = self.helper_fn(x)
                if x is None:
                    return y
                return y * x

        module = MyModule()
        real_result = module(torch.tensor([2]))

        # X is positive, so .item() > 0, which means we return y * x
        self.assertEqual(real_result, torch.tensor([1.0]))

        graph, guards = torch._dynamo.export(module, torch.tensor([2]))
        result = graph(torch.tensor([-0.5]))
        # X is negative, but we compiled helper_fn to return x, so it will still return y * x
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_none_control_flow_free_func(self):
        @torch._dynamo.assume_constant_result
        def helper_fn(x):
            if x.item() < 0:
                return None
            else:
                return x

        class MyModule(torch.nn.Module):
            def forward(self, x):
                y = torch.tensor([0.5])
                x = helper_fn(x)
                if x is None:
                    return y
                return y * x

        module = MyModule()
        real_result = module(torch.tensor([-1]))

        # X is negative, so .item() < 0, which means we return y
        self.assertEqual(real_result, torch.tensor([0.5]))

        graph, guards = torch._dynamo.export(module, torch.tensor([-1]))
        result = graph(torch.tensor([2]))
        # X is positive, but we compiled helper_fn to return None, so it will still return y
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_not_none_control_flow_pos(self):
        class MyModule(torch.nn.Module):
            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                if x.item() < 0:
                    return None
                else:
                    return x

            def forward(self, x):
                y = torch.tensor([0.5])
                x = self.helper_fn(x)
                if x is None:
                    return y
                return y * x

        module = MyModule()
        real_result = module(torch.tensor([2]))

        # X is positive, so .item() > 0, which means we return y * x
        self.assertEqual(real_result, torch.tensor([1.0]))

        graph, guards = torch._dynamo.export(module, torch.tensor([2]))
        result = graph(torch.tensor([-0.5]))
        # X is negative, but we compiled helper_fn to return x, so it will still return y * x
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_not_none_control_flow_free_func(self):
        @torch._dynamo.assume_constant_result
        def helper_fn(x):
            if x.item() < 0:
                return None
            else:
                return x

        class MyModule(torch.nn.Module):
            def forward(self, x):
                y = torch.tensor([0.5])
                x = helper_fn(x)
                if x is None:
                    return y
                return y * x

        module = MyModule()
        real_result = module(torch.tensor([2]))

        # X is positive, so .item() > 0, which means we return y * x
        self.assertEqual(real_result, torch.tensor([1.0]))

        graph, guards = torch._dynamo.export(module, torch.tensor([2]))
        result = graph(torch.tensor([-0.5]))
        # X is negative, but we compiled helper_fn to return x, so it will still return y * x
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_not_return_const(self):
        class MyModule(torch.nn.Module):
            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                return self.val

            def forward(self, x):
                y = torch.tensor([0.5])
                x = self.helper_fn(x)
                if x == "A":
                    return y
                return -1

        module = MyModule()
        module.val = "A"
        resA = module(torch.tensor([2]))
        graph, guards = torch._dynamo.export(module, torch.tensor([2]))
        module.val = "B"
        resB = graph(torch.tensor([2]))
        self.assertTrue(torch._dynamo.utils.same(resA, resB))

    def test_export_decomp(self):
        def f(x):
            return x.t() + x.t()

        def nop(x):
            return x.cos()

        graph, _ = torch._dynamo.export(
            f,
            (torch.randn(5)),
            aten_graph=True,
            decomposition_table={torch.ops.aten.t.default: nop},
        )
        self.assertEqual(
            len([n for n in graph.graph.nodes if n.target == torch.ops.aten.t.default]),
            0,
        )

        graph, _ = torch._dynamo.export(
            f, (torch.randn(5)), aten_graph=True, decomposition_table=None
        )
        self.assertEqual(
            len([n for n in graph.graph.nodes if n.target == torch.ops.aten.t.default]),
            2,
        )

    def test_export_decomp_asserts_bad_args(self):
        def f(x):
            return x.t() + x.t()

        def nop(x):
            return x.cos()

        with self.assertRaises(AssertionError):
            graph, _ = torch._dynamo.export(
                f,
                (torch.randn(5)),
                aten_graph=False,
                decomposition_table={torch.ops.aten.t.default: nop},
            )

    @config.patch(capture_scalar_outputs=True)
    def test_export_with_module_layer(self):
        from functorch.experimental.control_flow import cond

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, pred, x):
                def true_fn(val):
                    return self.linear(val) * torch.tensor(2)

                def false_fn(val):
                    return self.linear(val) * torch.tensor(-1)

                return cond(pred, true_fn, false_fn, [x])

        mod = Module()
        x = torch.randn([3, 3])
        pred = torch.tensor(x[0][0].item() < 0)
        real_result = mod.forward(pred, x)

        torch._dynamo.reset()

        exported = torch._dynamo.export(mod.forward, pred, x)
        out_graph = exported[0]

        dynamo_result = out_graph(pred, x)
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

        # New X, just to show we did not specialize
        x = x * -1
        pred = torch.tensor(x[0][0].item() < 0)
        real_result_2 = mod.forward(pred, x)
        dynamo_result_2 = out_graph(pred, x)
        self.assertTrue(torch._dynamo.utils.same(real_result_2, dynamo_result_2))

    @config.patch(capture_scalar_outputs=True)
    def test_export_with_cond_branches_calling_methods(self):
        from functorch.experimental.control_flow import cond

        class Module(torch.nn.Module):
            # ok
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def t(self, val):
                return val + 1

            def f(self, val):
                return val - 1

            def true_fn(self, val):
                return self.linear(val) + self.t(val)

            def false_fn(self, val):
                return self.linear(val) - self.f(val)

            def forward(self, pred, x):
                return cond(pred, self.true_fn, self.false_fn, [x])

        mod = Module()
        x = torch.randn([3, 3])
        pred = torch.tensor(x[0][0].item() < 0)
        real_result = mod.forward(pred, x)
        out_graph, _ = torch._dynamo.export(mod.forward, pred, x)
        dynamo_result = out_graph(pred, x)
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    @config.patch(capture_scalar_outputs=True)
    def test_export_with_cond_closure(self):
        from functorch.experimental.control_flow import cond

        class ModuleAccidentallyPassingError(torch.nn.Module):
            # error
            def __init__(self):
                super().__init__()

            def forward(self, pred, x):
                def true_fn(val):
                    return x * 2

                def false_fn(val):
                    return x - 2

                return cond(pred, true_fn, false_fn, [x])

        class ModuleAccidentallyPassingFixed(torch.nn.Module):
            # error
            def __init__(self):
                super().__init__()

            def forward(self, pred, x):
                def true_fn(x):
                    return x * 2

                def false_fn(x):
                    return x - 2

                return cond(pred, true_fn, false_fn, [x])

        class ModuleNoAccidentError(torch.nn.Module):
            # error
            def __init__(self):
                super().__init__()

            def forward(self, pred, x):
                def true_fn(val):
                    return x * 2

                def false_fn(val):
                    return x - 2

                return cond(pred, true_fn, false_fn, [x + 1])

        class ModuleNoAccidentFixed(torch.nn.Module):
            # error
            def __init__(self):
                super().__init__()

            def forward(self, pred, x):
                def true_fn(x):
                    return x * 2

                def false_fn(x):
                    return x - 2

                return cond(pred, true_fn, false_fn, [x + 1])

        class ModuleClosureReproError(torch.nn.Module):
            # error
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, pred, x):
                y = x + x

                def true_fn(val):
                    return self.linear(val) * (x + y)

                def false_fn(val):
                    return val * (y - x)

                return cond(pred, true_fn, false_fn, [x])

        class ModuleClosureReproFixed(torch.nn.Module):
            # error
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, pred, x):
                y = x + x

                def true_fn(x, y):
                    return self.linear(x) * (x + y)

                def false_fn(x, y):
                    return x * (y - x)

                return cond(pred, true_fn, false_fn, [x, y])

        for Module in [
            ModuleAccidentallyPassingError,
            ModuleNoAccidentError,
            ModuleClosureReproError,
        ]:
            mod = Module()
            x = torch.randn([3, 3])
            pred = torch.tensor(x[0][0].item() < 0)
            with self.assertRaisesRegex(
                torch._dynamo.exc.UserError,
                "Cannot create subgraph for nested function.*because it closes over variables",
            ):
                torch._dynamo.export(mod.forward, pred, x)

        for Module in [
            ModuleAccidentallyPassingFixed,
            ModuleNoAccidentFixed,
            ModuleClosureReproFixed,
        ]:
            mod = Module()
            x = torch.randn([3, 3])
            pred = torch.tensor(x[0][0].item() < 0)
            real_result = mod.forward(pred, x)
            out_graph, _ = torch._dynamo.export(mod.forward, pred, x)
            dynamo_result = out_graph(pred, x)
            self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_with_cond_with_closed_function(self):
        def hello(x):
            return x + 1

        def hi(x):
            return x + 2

        def foo(pred, x):
            def true_fn(x):
                return hello(x)

            def false_fn(x):
                return hi(x)

            return cond(pred, true_fn, false_fn, [x])

        x = torch.randn(5)
        pred = x[0] > 0
        real_result = foo(pred, x)
        out_graph, _ = torch._dynamo.export(foo, pred, x)
        dynamo_result = out_graph(pred, x)
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_with_cond_dynamic_shape_pred(self):
        from functorch.experimental.control_flow import cond

        class Module(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return x + x

                def false_fn(x):
                    return x[:2]

                return cond(x.shape[0] <= 2, true_fn, false_fn, [x])

        mod = Module()
        x = torch.randn(2, 2)
        out_graph, _ = torch._dynamo.export(mod, x)
        test_x = torch.randn(3, 2)
        self.assertEqual(out_graph(test_x), mod(test_x))

    def test_export_with_map_cond(self):
        from functorch.experimental.control_flow import cond, map

        class Module(torch.nn.Module):
            def inner(self, x, pred):
                def true_fn(x):
                    return x + x

                def false_fn(x):
                    return x * x

                return cond(pred, true_fn, false_fn, [x])

            def forward(self, pred, xs):
                def body(x, pred):
                    return self.inner(x, pred)

                return map(body, xs, pred)

        mod = Module()
        x = torch.randn(3, 2, 1)
        pred_x = torch.tensor(True)

        y = torch.randn(4, 3, 2)
        pred_y = torch.tensor(False)
        real_result = mod(pred_y, y)

        out_graph, _ = torch._dynamo.export(mod, pred_x, x)
        self.assertEqual(real_result, out_graph(pred_y, y))

    def test_export_with_map_zero_sized_tensor(self):
        from functorch.experimental.control_flow import map

        class Module(torch.nn.Module):
            def forward(self, xs):
                def body(x):
                    return x + 1

                return map(body, xs)

        mod = Module()
        xs = torch.randn(0, 2)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "zero-sized tensor",
        ):
            out_graph, _ = torch._dynamo.export(mod, xs)

    def test_export_meta_val(self):
        def f(x, y, z):
            return x * y + z

        gm, _ = torch._dynamo.export(
            f,
            torch.ones(3, 2),
            torch.zeros(3, 2),
            torch.ones(3, 2),
            aten_graph=True,
        )
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                self.assertIn("val", node.meta)

    def test_input_container_type(self):
        def f(x: torch.Tensor, y: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
            return {"a": x.sum() + sum(y).sum()}

        inp = (torch.randn(6, 5), [torch.randn(6, 5), torch.randn(6, 5)])

        gm, _ = torch._dynamo.export(f, *inp, aten_graph=True)

        self.assertEqual(gm(*inp), f(*inp))

    @config.patch(assume_static_by_default=False)
    def test_export_symbolic_shape(self):
        def f(x: torch.Tensor) -> torch.Tensor:
            return torch.empty(x.shape[0] * 2)

        inp = (torch.randn(6, 5),)
        gm, _ = torch._dynamo.export(f, *inp, aten_graph=True)

        has_sym_size = False
        for node in gm.graph.nodes:
            if node.target is torch.ops.aten.sym_size:
                has_sym_size = True

        self.assertTrue(has_sym_size)

    @config.patch(assume_static_by_default=False)
    def test_dynamic_slicing(self):
        def f(x):
            return x[: x.shape[0] - 2, x.shape[1] - 1 :: 2]

        gm_aten_mode, _ = torch._dynamo.export(f, torch.randn(4, 5), aten_graph=True)

        inp = torch.randn(6, 7)
        self.assertEqual(gm_aten_mode(inp).shape, f(inp).shape)

        count = 0
        # aten graph should flatten getitem calls to actual
        # slice kernel call.
        for node in gm_aten_mode.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.slice.Tensor
            ):
                count += 1

        self.assertEqual(count, 2)

        gm_torch_mode, _ = torch._dynamo.export(f, torch.randn(4, 5), aten_graph=False)

        # In torch mode, the graph should contain 3 getitem methods
        # one for x.shape[0]-2 and one for x.shape[1]-1 and one for slice
        # this is because Tensor class has its' own getitem method
        # which gets translated to aten.Slice later.
        count = 0
        for node in gm_torch_mode.graph.nodes:
            if node.op == "call_function" and node.target == operator.getitem:
                count += 1

        self.assertEqual(count, 3)
        self.assertEqual(gm_torch_mode(inp).shape, f(inp).shape)

    def test_dynamic_slicing_invalid(self):
        def g(x, y):
            return x[y : x.shape[0]]

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Dynamic slicing on data-dependent value is not supported",
        ):
            torch._dynamo.export(
                g,
                torch.randn(4, 5),
                torch.tensor(2),
                aten_graph=True,
            )

    @skipIfRocm
    @config.patch(capture_scalar_outputs=True)
    def test_dynamic_slicing_simple(self):
        def f(x):
            return x[slice(None, None, None)]

        gm, _ = torch._dynamo.export(f, torch.randn(4, 5), aten_graph=True)

        inp = torch.randn(6, 7)
        self.assertEqual(gm(inp), f(inp))

    # pre_autograd seems to violate new fake tensor invariants
    @unittest.expectedFailure
    def test_pre_autograd_simple(self):
        def f(x):
            y = torch.ones_like(x)
            return torch.matmul(x, y)

        gm, _ = torch._dynamo.export(
            f,
            torch.randn(5, 5),
            aten_graph=True,
            pre_autograd=True,
            tracing_mode="fake",
        )

        inp = torch.randn(6, 6)
        self.assertEqual(gm(inp), f(inp))
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x):
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    ones_like_default = torch.ops.aten.ones_like.default(arg0, pin_memory = False)
    matmul_default = torch.ops.aten.matmul.default(arg0, ones_like_default);  arg0 = ones_like_default = None
    return pytree.tree_unflatten([matmul_default], self._out_spec)""",
        )

    @skipIfRocm
    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_export_cond_in_aten_symbolic(self):
        class ConditionOp(torch.nn.Module):
            def true_fn(self, x, y):
                return x * y

            def false_fn(self, x, y):
                return x + y

            def forward(self, pred, x, y):
                return cond(pred, self.true_fn, self.false_fn, [x, y])

        model = ConditionOp()
        inp = (
            torch.tensor(False),
            torch.randn(4, 4),
            torch.randn(4, 4),
        )
        gm, _ = torch._dynamo.export(model, *inp, aten_graph=True)

        gm.print_readable()

        self.assertEqual(gm(*inp), model(*inp))

    def test_export_with_kwargs(self):
        def fn_with_kwargs(pos0, tuple0, *myargs, mykw0=None, **mykwargs):
            out = pos0
            for arg in tuple0:
                out *= arg
            for arg in myargs:
                out *= arg
            out *= mykw0
            out *= mykwargs["input0"] * mykwargs["input1"]
            return out

        mykwargs = {"input0": torch.randn(4), "input1": torch.randn(4)}
        tuple0 = (torch.randn(4), torch.randn(4))
        mykw0 = torch.randn(4)
        pos0 = torch.randn(4)
        myargs = [torch.randn(4), torch.randn(4)]

        expected_argument_names = [
            "pos0",
            "tuple0",
            "myargs_0",
            "myargs_1",
            "mykw0",
            "input0",
            "input1",
        ]
        self._test_export_preserving_original_signature(
            fn_with_kwargs,
            expected_argument_names,
            pos0,
            tuple0,
            *myargs,
            mykw0=mykw0,
            **mykwargs,
        )

    def test_export_with_kwargs_and_empty_args(self):
        def fn_with_kwargs(mykw0=None, **mykwargs):
            out = mykw0
            out *= mykwargs["input0"] * mykwargs["input1"]
            return out

        mykwargs = {"input0": torch.randn(4), "input1": torch.randn(4)}
        mykw0 = torch.randn(4)

        expected_argument_names = ["mykw0"] + list(mykwargs.keys())
        self._test_export_preserving_original_signature(
            fn_with_kwargs, expected_argument_names, mykw0, **mykwargs
        )

    def test_export_with_args_and_empty_kwargs(self):
        def fn_with_kwargs(pos0, tuple0, *myargs):
            out = pos0
            for arg in tuple0:
                out *= arg
            for arg in myargs:
                out *= arg
            return out

        tuple0 = (torch.randn(4), torch.randn(4))
        pos0 = torch.randn(4)
        myargs = [torch.randn(4), torch.randn(4)]

        expected_argument_names = ["pos0", "tuple0", "myargs_0", "myargs_1"]
        self._test_export_preserving_original_signature(
            fn_with_kwargs, expected_argument_names, pos0, tuple0, *myargs
        )

    @common_utils.parametrize(
        "default_value",
        [
            common_utils.subtest(None, name="None"),
            common_utils.subtest(42.0, name="float"),
            common_utils.subtest(
                # FIXME: AssertionError: Dynamo input and output is a strict subset of traced input/output
                torch.randn(4),
                name="tensor",
                decorators=[unittest.expectedFailure],
            ),
            common_utils.subtest(
                # FIXME: AssertionError: Dynamo input and output is a strict subset of traced input/output
                (torch.randn(4),),
                name="tuple",
                decorators=[unittest.expectedFailure],
            ),
        ],
    )
    def test_export_with_args_with_default(self, default_value):
        def fn(pos0, pos1_default=default_value):
            out = pos0
            if pos1_default is None:
                pos1_default = torch.randn(4)
            if isinstance(pos1_default, tuple):
                pos1_default = pos1_default[0]
            out *= pos1_default
            return out

        pos0 = torch.randn(4)
        expected_argument_names = ["pos0"]
        self._test_export_preserving_original_signature(
            fn, expected_argument_names, pos0
        )

    @common_utils.parametrize(
        "default_value",
        [
            common_utils.subtest(None, name="None"),
            common_utils.subtest(42.0, name="float"),
            common_utils.subtest(
                # FIXME: AssertionError: Dynamo input and output is a strict subset of traced input/output
                torch.randn(4),
                name="tensor",
                decorators=[unittest.expectedFailure],
            ),
            common_utils.subtest(
                # FIXME: AssertionError: Dynamo input and output is a strict subset of traced input/output
                (torch.randn(4),),
                name="tuple",
                decorators=[unittest.expectedFailure],
            ),
        ],
    )
    def test_export_with_kwargs_with_default(self, default_value):
        def fn(pos0, *, kw0, kw1_default=default_value, **kwargs):
            out = pos0
            out += kw0
            if kw1_default is None:
                kw1_default = torch.randn(4)
            elif isinstance(kw1_default, tuple):
                kw1_default = kw1_default[0]
            out += kw1_default
            out += kwargs["kw2"]
            return out

        pos0 = torch.randn(4)
        kw0 = torch.randn(4)
        kw2 = torch.randn(4)

        args = (pos0,)
        kwargs = {"kw0": kw0, "kw2": kw2}
        expected_argument_names = ["pos0", "kw0", "kw2"]
        self._test_export_preserving_original_signature(
            fn, expected_argument_names, *args, **kwargs
        )

    def test_export_with_wrapped_fn(self):
        # To ensure dynamo.export is robust to wrapped functions
        # when it cannot use `inspect` to retrieve original signature
        # info.
        def _fn(pos0, pos1=1.0, *args, kw0, kw1=2.0, **kwargs):
            out = pos0
            out += pos1
            out += kw0
            out += kw1
            for arg in args:
                out += arg
            for kwarg in kwargs.values():
                out += kwarg
            return out

        def wrapped_fn(*args, **kwargs):
            return _fn(*args, **kwargs)

        pos0 = torch.randn(4)
        kw0 = torch.randn(4)
        args = (pos0, torch.randn(4), torch.randn(4))
        kwargs = {"kw0": kw0, "kw2": torch.randn(4)}
        expected_argument_names = [f"args_{i}" for i in range(len(args))] + list(
            kwargs.keys()
        )

        self._test_export_preserving_original_signature(
            wrapped_fn, expected_argument_names, *args, **kwargs
        )

    def test_export_with_functools_wrapped_method(self):
        def test_decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

            @test_decorator
            def method_to_test(self, pos0, pos1=1.0, *args, kw0, kw1=2.0, **kwargs):
                out = pos0
                out += pos1
                out += kw0
                out += kw1
                for arg in args:
                    out += arg
                for kwarg in kwargs.values():
                    out += kwarg
                return out

        pos0 = torch.randn(4)
        pos1 = torch.randn(4)
        unnamed_pos = torch.randn(4)
        kw0 = torch.randn(4)
        args = (pos0, pos1, unnamed_pos)
        kwargs = {"kw0": kw0, "kw2": torch.randn(4), "unnamed_kw": torch.randn(4)}
        expected_argument_names = [
            "pos0",
            "pos1",
            "args_0",  # 3rd unnamed positional argument
        ] + list(kwargs.keys())
        m = MyModule()

        self._test_export_preserving_original_signature(
            m.method_to_test, expected_argument_names, *args, **kwargs
        )

    def test_export_with_functools_wrapped_fn(self):
        def test_decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        @test_decorator
        def _fn(pos0, pos1=1.0, *args, kw0, kw1=2.0, **kwargs):
            out = pos0
            out += pos1
            out += kw0
            out += kw1
            for arg in args:
                out += arg
            for kwarg in kwargs.values():
                out += kwarg
            return out

        def wrapped_fn(*args, **kwargs):
            return _fn(*args, **kwargs)

        pos0 = torch.randn(4)
        kw0 = torch.randn(4)
        args = (pos0, torch.randn(4), torch.randn(4))
        kwargs = {"kw0": kw0, "kw2": torch.randn(4)}
        expected_argument_names = [f"args_{i}" for i in range(len(args))] + list(
            kwargs.keys()
        )

        self._test_export_preserving_original_signature(
            wrapped_fn, expected_argument_names, *args, **kwargs
        )

    def _test_export_preserving_original_signature(
        self, fn, expected_argument_names: Sequence[str], *args, **kwargs
    ):
        torch._dynamo.reset()
        exported = torch._dynamo.export(
            fn,
            *args,
            **kwargs,
            aten_graph=False,
        )

        out_graph = exported[0]
        dynamo_result = out_graph(*args, **kwargs)
        real_result = fn(*args, **kwargs)
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

        # Check that the exported graph preserves same argument names.
        self.assertEqual(
            inspect.getfullargspec(out_graph.forward).args[1:], expected_argument_names
        )

    def test_export_meta(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.ones(2, 3))

            def forward(self, x):
                return self.p + x

        with torch.device("meta"):
            m = MyModule()

        inp = torch.ones(2, 3, device="meta")
        exported = torch._dynamo.export(m, inp)
        out_graph = exported[0]
        dynamo_result = out_graph(inp)
        self.assertEqual(dynamo_result, m(inp))

    def test_export_raise_guard_full_constraint(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            if x.shape[0] == 3:
                return x.sin()
            return x.cos()

        torch._dynamo.export(my_dyn_fn, y)

        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.export(my_dyn_fn, y, constraints=[dynamic_dim(y, 0)])

    def test_export_module_specify_constraints_signature(self):
        y = torch.randn([3, 3, 3])

        class Mod(torch.nn.Module):
            def forward(self, x):
                if x.shape[0] == 3:
                    return x.sin()
                return x.cos()

        mod = Mod()
        torch._dynamo.export(mod, y)

        with self.assertRaisesRegex(
            ConstraintViolationError, "def specify_constraints\\(x\\):"
        ):
            torch._dynamo.export(mod, y, constraints=[dynamic_dim(y, 0)])

    def test_export_raise_guard_partial_constraint(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            if x.shape[0] > 3:
                return x.sin()
            return x.cos()

        torch._dynamo.export(my_dyn_fn, y)

        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.export(my_dyn_fn, y, constraints=[dynamic_dim(y, 0)])

    def test_export_raise_on_relationship(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(a, b, c):
            if a.shape[0] == b.shape[1] == c.shape[2]:
                return a.sin()

            return a.cos()

        torch._dynamo.export(my_dyn_fn, y, y, y)
        constraints = [dynamic_dim(y, 0)]
        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.export(my_dyn_fn, y, y, y, constraints=constraints)
        constraints += [
            dynamic_dim(y, 1) == dynamic_dim(y, 0),
            dynamic_dim(y, 2) == dynamic_dim(y, 0),
        ]
        torch._dynamo.export(my_dyn_fn, y, y, y, constraints=constraints)

    def test_export_no_raise(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(a, b, c):
            if a.shape[1] == 3:
                return a.cos()
            return a * b * c

        torch._dynamo.export(my_dyn_fn, y, y, y)
        torch._dynamo.export(my_dyn_fn, y, y, y, constraints=[dynamic_dim(y, 0)])

    @skipIfRocm
    def test_export_multi_dynamic_dim_unsafe_relationship(self):
        x = torch.randn([3, 3, 3])
        y = torch.randn([2, 2, 2])
        z = torch.randn([3, 3, 3])

        def my_dyn_fn(a, b, c):
            if a.shape[0] == c.shape[0]:
                return a.cos()
            return a * c, b

        torch._dynamo.export(my_dyn_fn, x, y, z)
        constraints = [dynamic_dim(x, 0), dynamic_dim(y, 0), dynamic_dim(z, 0)]
        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.export(my_dyn_fn, x, y, z, constraints=constraints)
        constraints.append(dynamic_dim(z, 0) == dynamic_dim(x, 0))
        torch._dynamo.export(my_dyn_fn, x, y, z, constraints=constraints)

    @config.patch(
        dynamic_shapes=True,
        capture_dynamic_output_shape_ops=True,
        specialize_int=True,
        capture_scalar_outputs=True,
    )
    def test_export_preserve_constraints_as_metadata_scalar(self):
        def f(x, y):
            b = x.item()
            constrain_as_size(b, min=2, max=5)
            return torch.empty((b, y.shape[0]))

        x = torch.tensor([3])
        y = torch.randn([8, 8, 6])
        example_inputs = [x, y]
        constraints = [dynamic_dim(y, 0) >= 6, dynamic_dim(y, 0) <= 10]
        gm, _ = torch._dynamo.export(
            f,
            *example_inputs,
            constraints=constraints,
            aten_graph=True,
            tracing_mode="symbolic",
        )

        self.assertEqual(
            gm.meta["input_shape_constraints"],
            [c.serializable_spec for c in constraints],
        )
        preserved = False
        for _, vr in gm.meta["inline_constraints"].items():
            # Should have the constraint with min=2, max=5
            if vr.lower == 2 and vr.upper == 5:
                preserved = True
        self.assertTrue(preserved)

    @torch._dynamo.config.patch(
        dynamic_shapes=True,
        capture_dynamic_output_shape_ops=True,
        specialize_int=True,
        capture_scalar_outputs=True,
    )
    def test_export_preserve_constraints_as_metadata_tensor(self):
        def f(x):
            b = x.nonzero()
            constrain_as_value(b.shape[0], min=2, max=5)
            return b

        y = torch.tensor([8, 8, 6])
        constraints = []
        gm, _ = torch._dynamo.export(
            f,
            y,
            constraints=constraints,
            aten_graph=True,
            tracing_mode="symbolic",
        )

        preserved = False
        for _, vr in gm.meta["inline_constraints"].items():
            # Should have the constraint with min=2, max=5
            if vr.lower == 2 and vr.upper == 5:
                preserved = True
        self.assertTrue(preserved)

    @config.patch(
        dynamic_shapes=True,
        capture_dynamic_output_shape_ops=True,
        specialize_int=True,
        capture_scalar_outputs=True,
    )
    def test_exported_graph_serialization(self):
        import io

        def f(x, y):
            b = x.item()
            constrain_as_size(b, min=2, max=5)
            return torch.empty((b, y.shape[0]))

        x = torch.tensor([3])
        y = torch.randn([8, 8, 6])
        example_inputs = [x, y]
        constraints = [dynamic_dim(y, 0) >= 6, dynamic_dim(y, 0) <= 10]
        gm, _ = torch._dynamo.export(
            f,
            *example_inputs,
            constraints=constraints,
            aten_graph=True,
            tracing_mode="symbolic",
        )

        # Ensure the exported graph module with metadata is serializable,
        # metadata won't be saved in the serialized module
        buffer = io.BytesIO()
        torch.save(gm, buffer)

    def test_export_dynamic_dim_not_1(self):
        x = torch.randn([1, 1, 1])

        def my_dyn_fn(a):
            if a.shape[0] != 1:
                return a.cos()
            return a * a

        torch._dynamo.export(my_dyn_fn, x)
        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.export(my_dyn_fn, x, constraints=[dynamic_dim(x, 0)])

    def test_symbool(self):
        def f(x):
            a = torch.scalar_tensor(x.shape[0] > 4)
            return x.sin().sum() + a.sum()

        gm, _ = torch._dynamo.export(f, torch.ones(6, 4), aten_graph=True)
        self.assertEqual(gm(torch.ones(3, 4)), f(torch.ones(3, 4)))

    def test_export_multi_dynamic_dim_constraint(self):
        x = torch.randn([3, 3, 3])
        y = torch.randn([2, 2, 2])
        z = torch.randn([3, 3, 3])

        def my_dyn_fn(a, b, c):
            if a.shape[0] == c.shape[0]:
                return a.cos()
            return a * c, b

        torch._dynamo.export(my_dyn_fn, x, y, z)
        constraints = [dynamic_dim(x, 0), dynamic_dim(x, 1), dynamic_dim(x, 2)]
        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.export(my_dyn_fn, x, y, z, constraints=constraints)
        constraints.append(dynamic_dim(z, 0) == dynamic_dim(x, 0))
        torch._dynamo.export(my_dyn_fn, x, y, z, constraints=constraints)

    def test_export_dynamic_dim_raise_on_compound_range_constraint(self):
        x = torch.ones(6, 4, 4)
        with self.assertRaisesRegex(TypeError, "Cannot determine truth value"):
            4 < dynamic_dim(x, 0) <= 6  # noqa: B015

    def test_export_dynamic_dim_range_constraint(self):
        x = torch.ones(6, 4, 4)
        constraints = [
            4 < dynamic_dim(x, 0),
            dynamic_dim(x, 0) <= 6,
        ]

        def foo(x):
            if x.shape[0] > 3:  # ok
                return x.sin()
            return x.cos()

        torch._dynamo.export(
            foo,
            x,
            constraints=constraints,
            aten_graph=True,
        )

        def bar(x):
            if x.shape[0] > 5:  # error
                return x.sin()
            return x.cos()

        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.export(
                bar,
                x,
                constraints=constraints,
                aten_graph=True,
            )

    def test_list_contains(self):
        def func(x):
            assert x.size(-1) in [4, 5, 6], "bad"
            return x + x

        inps = (torch.randn(1, 5),)
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, *inps, aten_graph=True)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_list_not_contains(self):
        def func(x):
            assert x.size(0) not in [4, 5, 6], "bad1"
            assert "monkey" not in ["cow", "pig"], "bad2"
            return x + x

        inps = (torch.randn(1, 5),)
        opt_func = torch._dynamo.optimize("eager", nopython=True, dynamic=True)(func)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, *inps, aten_graph=True)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_identity(self):
        inp = torch.tensor([0.1, 0.1])

        def func(x):
            return x

        torch._dynamo.reset()
        exported, _ = torch._dynamo.export(func, inp)
        dynamo_result = exported(inp)
        self.assertTrue(torch._dynamo.utils.same(inp, dynamo_result))

    def test_export_specialized_int(self):
        class Foo(torch.nn.Module):
            def __init__(
                self,
                input_dim,
            ):
                super().__init__()
                self.torch_module = torch.nn.LayerNorm(
                    input_dim, eps=1e-5, elementwise_affine=True
                )
                self.int_val = 100

            def forward(self, input):
                return input.cos() * self.int_val * self.torch_module.eps

        mod = Foo(128)
        inp = torch.randn(3, 128)

        # In export, int & float in forward should always be specialized
        gm, _ = torch._dynamo.export(mod, inp, aten_graph=True)
        count = 0
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                count += 1
        self.assertEqual(count, 1)

    def test_export_with_nonzero_static(self):
        class BasicModule(torch.nn.Module):
            def __init__(self, static_size):
                super().__init__()
                self.static_size = static_size

            def forward(self, x):
                return torch.nonzero_static(x, size=self.static_size)

        input_tensors = torch.tensor([6, 8]), torch.zeros(2, 3)
        static_sizes = 3, 4
        for input_tensor, static_size in zip(input_tensors, static_sizes):
            m = BasicModule(static_size)
            gm, _ = torch._dynamo.export(m, input_tensor, aten_graph=True)
            res = gm(input_tensor)
            self.assertEqual(res.size(0), static_size)
            self.assertTrue(
                torch._dynamo.utils.same(
                    res, torch.nonzero_static(input_tensor, size=static_size)
                )
            )

    def test_export_pass_arg_by_name(self):
        class BasicModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.my_lin = torch.nn.Linear(3, 4, bias=True)

            def forward(self, x):
                return self.my_lin(x)

        mod, input_tensor = BasicModule(), torch.randn(2, 3)
        gm, guard = torch._dynamo.export(mod, input_tensor, aten_graph=True)
        ref = mod(x=input_tensor)
        res = gm(x=input_tensor)
        self.assertTrue(torch._dynamo.utils.same(ref, res))

    def test_export_pass_arg_by_name_star_args(self):
        class BasicModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.my_lin = torch.nn.Linear(3, 4, bias=True)

            def forward(self, *args):
                return self.my_lin(args[0]) * self.my_lin(args[1])

        mod, input_tensor, input_tensor2 = (
            BasicModule(),
            torch.randn(2, 3),
            torch.randn(2, 3),
        )
        gm, guard = torch._dynamo.export(
            mod, input_tensor, input_tensor2, aten_graph=True
        )
        ref = mod(input_tensor, input_tensor2)
        res = gm(input_tensor, input_tensor2)
        self.assertTrue(torch._dynamo.utils.same(ref, res))

    def test_export_mark_dynamic_conflict_dynamic_dim(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            if x.shape[0] > 3:
                return x.sin()
            return x.cos()

        torch._dynamo.mark_dynamic(y, 0)
        with self.assertRaisesRegex(
            RuntimeError,
            "Constraints violated",
        ):
            torch._dynamo.export(my_dyn_fn, y, constraints=[dynamic_dim(y, 0)])

    def test_export_dynamic_dim_cleanup(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            return x.cos()

        constraints = [dynamic_dim(y, 0)]
        torch._dynamo.export(my_dyn_fn, y, constraints=constraints)

    @config.patch(capture_dynamic_output_shape_ops=True)
    def test_export_dynamic_control_flow_error(self):
        def f(x):
            if x.nonzero() > 3:
                return x.cos()
            return x.sin()

        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Dynamic control flow is not supported at the moment",
        ):
            gm, _ = torch._dynamo.export(f, torch.randn(5, 6), aten_graph=True)

    @config.patch(assume_static_by_default=False)
    def test_export_persist_assert(self):
        def f(x):
            assert x.shape[0] > 4, "Shape must be more than 4"
            return x.cos() + x.sin()

        gm, guard = torch._dynamo.export(
            f, torch.randn(5, 4, 6), aten_graph=True, tracing_mode="symbolic"
        )

        def has_aten_op(gm, op):
            for node in gm.graph.nodes:
                if node.target == op:
                    return True
            return False

        self.assertTrue(has_aten_op(gm, torch.ops.aten._assert_async.msg))

        gm.graph.eliminate_dead_code()
        gm.recompile()
        self.assertTrue(has_aten_op(gm, torch.ops.aten._assert_async.msg))

        with self.assertRaisesRegex(RuntimeError, "Shape must be more than 4"):
            gm(torch.randn(3, 4, 5))

    def test_access_class_method_from_user_class(self):
        class A:
            @classmethod
            def func(cls):
                return torch.Tensor([4, 5])

        def f(x):
            a = A()
            return x.sum() + type(a).func().sum()

        with self.assertRaisesRegex(torch._dynamo.exc.UserError, "Can't call type()"):
            gm, _ = torch._dynamo.export(f, torch.ones(6, 4), aten_graph=True)

        def f_correct(x):
            a = A()
            return x.sum() + a.__class__.func().sum()

        gm, _ = torch._dynamo.export(f_correct, torch.ones(6, 4), aten_graph=True)

        self.assertEqual(f_correct(torch.ones(6, 4)), gm(torch.ones(6, 4)))

    @config.patch(dynamic_shapes=True)
    def test_functionalize(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer1", torch.ones(6, 2))

            def forward(self, x):
                x.add_(2)
                return x.sum() + self.buffer1.sum()

        example_inputs = (torch.ones(1, 2, 3),)
        gm, _ = torch._dynamo.export(
            Foo(),
            *example_inputs,
            aten_graph=True,
            tracing_mode="symbolic",
            functionalize=True,
        )

        count = 0
        for node in gm.graph.nodes:
            if node.target == torch.ops.aten.add_.Tensor:
                count += 1
        self.assertEqual(count, 0)
        test_inp = (torch.ones(1, 2, 3),)
        test_inp_v2 = (torch.ones(1, 2, 3),)
        self.assertEqual(gm(*test_inp), Foo()(*test_inp_v2))

    @config.patch(dynamic_shapes=True)
    def test_not_functionalize(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer1", torch.ones(6, 2))

            def forward(self, x):
                x.add_(2)
                return x.sum() + self.buffer1.sum()

        example_inputs = (torch.ones(1, 2, 3),)
        gm, _ = torch._dynamo.export(
            Foo(),
            *example_inputs,
            aten_graph=True,
            tracing_mode="symbolic",
            functionalize=False,
        )
        count = 0
        for node in gm.graph.nodes:
            if node.target == torch.ops.aten.add_.Tensor:
                count += 1
        self.assertEqual(count, 1)
        test_inp = (torch.ones(1, 2, 3),)
        test_inp_v2 = (torch.ones(1, 2, 3),)
        self.assertEqual(gm(*test_inp), Foo()(*test_inp_v2))

    @config.patch(dynamic_shapes=True, assume_static_by_default=False)
    def test_functionalize_cond(self):
        def foo(x):
            def true_true_fn(x):
                return x.sum() + 6

            def true_false_fn(x):
                return x.sum() + 9

            def true_fn(x):
                return cond(x.shape[0] > 6, true_true_fn, true_false_fn, [x])

            def false_fn(x):
                return x.sum() - 1

            return cond(x.shape[0] > 5, true_fn, false_fn, [x])

        example_inputs = (torch.ones(5, 2, 3),)
        gm, _ = torch._dynamo.export(
            foo,
            *example_inputs,
            aten_graph=True,
            tracing_mode="symbolic",
            functionalize=True,
        )
        self.assertEqual(gm(torch.ones(7, 2, 3)), foo(torch.ones(7, 2, 3)))

    @config.patch(dynamic_shapes=True)
    def test_functionalize_simple(self):
        def foo(x):
            def true_fn(x):
                return x.sum() + 1

            def false_fn(x):
                return x.sum() - 1

            return cond(x.shape[0] > 5, true_fn, false_fn, [x])

        example_inputs = (torch.ones(5, 2, 3),)
        gm, _ = torch._dynamo.export(
            foo,
            *example_inputs,
            aten_graph=True,
            tracing_mode="symbolic",
            functionalize=True,
        )
        self.assertEqual(gm.true_graph_0(torch.ones(6, 4)), torch.ones(6, 4).sum() + 1)
        self.assertEqual(gm.false_graph_0(torch.ones(6, 4)), torch.ones(6, 4).sum() - 1)

    @config.patch(dynamic_shapes=True)
    def test_round_dynamic_shapes(self):
        def f(x):
            return x[: round(x.shape[0] / 2)]

        def f_correct(x):
            return x[: math.floor(x.shape[0] / 2)]

        with self.assertRaisesRegex(torch._dynamo.exc.UserError, "Calling round()"):
            gm, _ = torch._dynamo.export(f, torch.ones(6, 4), aten_graph=True)

        gm, _ = torch._dynamo.export(f_correct, torch.ones(6, 4), aten_graph=True)

        self.assertEqual(f_correct(torch.ones(6, 4)), gm(torch.ones(6, 4)))

    def test_cond_supported_pred_types(self):
        def true_fn(x):
            return x.cos()

        def false_fn(x):
            return x.sin()

        def f_pred_traced_as_symnode_var(x):
            return cond(x.shape[0] > 2, true_fn, false_fn, [x])

        def f_pred_traced_as_tensor_var(x):
            return cond(x.all(), true_fn, false_fn, [x])

        def f_pred_complex_expression_traced_as_symnode_var(x):
            return cond(
                x.dim() > 1 and x.shape[1] > 5 and x.shape[1] <= 10,
                true_fn,
                false_fn,
                [x],
            )

        example_inputs = (torch.rand(5, 8),)
        for f in [
            f_pred_traced_as_symnode_var,
            f_pred_traced_as_tensor_var,
            f_pred_complex_expression_traced_as_symnode_var,
        ]:
            gm, _ = torch._dynamo.export(f, *example_inputs, aten_graph=True)
            self.assertEqual(gm(*example_inputs), f(*example_inputs))

    def test_mixed_real_and_fake_inputs(self):
        class _TestPattern(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.bn = torch.nn.BatchNorm2d(1)

            def forward(self, input):
                running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
                scale_factor = self.bn.weight / running_std
                weight_shape = [1] * len(self.conv.weight.shape)
                weight_shape[0] = -1
                bias_shape = [1] * len(self.conv.weight.shape)
                bias_shape[1] = -1
                scaled_weight = self.conv.weight * scale_factor.reshape(weight_shape)
                zero_bias = torch.zeros_like(self.conv.bias, dtype=input.dtype)
                conv = self.conv._conv_forward(input, scaled_weight, zero_bias)
                conv_orig = conv / scale_factor.reshape(bias_shape)
                conv_orig = conv_orig + self.conv.bias.reshape(bias_shape)
                conv = self.bn(conv_orig)
                return conv

        example_inputs = (torch.randn(1, 1, 3, 3),)
        torch._dynamo.export(
            _TestPattern(),
            *example_inputs,
            aten_graph=True,
        )

    @config.patch(
        capture_dynamic_output_shape_ops=True,
        capture_scalar_outputs=True,
        assume_static_by_default=False,
    )
    def test_sym_contains(self):
        def f(x, y):
            return x.size(0) in y

        gm, _ = torch._dynamo.export(f, torch.ones(2), torch.ones(3), aten_graph=True)

        true_inp = (torch.Tensor([6, 4, 5]), torch.ones(6, 4).add_(5))
        false_inp = (torch.Tensor([6, 4, 5]), torch.ones(6, 4).add_(2))
        self.assertEqual(gm(*true_inp), f(*true_inp))
        self.assertEqual(gm(*false_inp), f(*false_inp))

    def test_cond_raise_user_error_on_missing_args(self):
        def true_fn(x):
            return x.cos()

        def false_fn(x):
            return x.sin()

        def f(x):
            return cond(x.shape[0] > 10, true_fn, false_fn)

        example_inputs = (torch.rand(5),)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected 4 arguments",
        ):
            torch._dynamo.export(f, *example_inputs, aten_graph=True)

    def test_cond_raise_user_error_on_unsupported_pred(self):
        def f_unsupported_pred(x):
            pred = torch.nn.Module()
            return cond(pred, lambda x: x.sin(), lambda x: x.cos(), [x])

        example_inputs = (torch.rand(5),)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected pred to be bool/int or a tensor",
        ):
            torch._dynamo.export(
                f_unsupported_pred,
                *example_inputs,
                aten_graph=True,
            )

    def test_cond_raise_user_error_on_non_list_operands(self):
        def f_non_list_operands(x):
            return cond(torch.tensor(True), lambda x: x.sin(), lambda x: x.cos(), x)

        example_inputs = (torch.rand(5),)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected a list but got",
        ):
            torch._dynamo.export(
                f_non_list_operands,
                *example_inputs,
                aten_graph=True,
            )

    def test_cond_raise_user_error_on_non_tensor_operands(self):
        def f_non_tensor_operands(x):
            a: float = 3.14
            return cond(
                torch.tensor(1234), lambda x, a: x.sin(), lambda x, a: x.cos(), [x, a]
            )

        example_inputs = (torch.rand(5),)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected a list of tensors",
        ):
            torch._dynamo.export(
                f_non_tensor_operands,
                *example_inputs,
                aten_graph=True,
            )

    def test_cond_raise_user_error_on_branch_args_mismatch(self):
        def true_fn(x, y):
            return x.sin()

        def false_fn(x):
            return x.cos()

        def f_branch_args_mismatch(x, y):
            return cond(torch.tensor([[[[100]]]]), true_fn, false_fn, [x, y])

        example_inputs = (torch.rand(5), torch.rand(2))
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "too many positional arguments",
        ):
            torch._dynamo.export(
                f_branch_args_mismatch,
                *example_inputs,
                aten_graph=True,
            )

    def test_cond_raise_user_error_on_branch_return_non_tensor(self):
        def f_branch_return_non_tensor(x):
            return cond(x.shape[0] <= 5, lambda x: 3.14, lambda x: 3.14, [x])

        example_inputs = (torch.rand(5),)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected branch out type to be a single tensor",
        ):
            torch._dynamo.export(
                f_branch_return_non_tensor,
                *example_inputs,
                aten_graph=True,
            )

    def test_cond_raise_user_error_on_branch_return_multiple_tenors(self):
        def f_branch_return_multiple_tensors(x, y):
            return cond(y, lambda x: (x, x), lambda x: (x, x), [x])

        example_inputs = (torch.randn(4), torch.randn(2))
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected branch out type to be a single tensor",
        ):
            torch._dynamo.export(
                f_branch_return_multiple_tensors,
                *example_inputs,
                aten_graph=True,
            )

    def test_multiple_outputs_op_with_evaluator(self):
        class TopKModel(torch.nn.Module):
            def forward(self, x):
                values, _ = torch.topk(x, 3)
                return torch.sum(values)

        x = torch.arange(1.0, 6.0, requires_grad=True)
        torch._dynamo.export(TopKModel(), x)

    def test_cond_raise_user_error_on_mismatch_return_length(self):
        def true_fn(x):
            return x

        def false_fn(x):
            return (x, x)

        def f_mismatch_return_length(x):
            return cond(torch.tensor(100), true_fn, false_fn, [x])

        example_inputs = (torch.rand(5),)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected branch out type to be a single tensor",
        ):
            torch._dynamo.export(
                f_mismatch_return_length,
                *example_inputs,
                aten_graph=True,
            )

    def test_cond_raise_user_error_on_mismatch_return_tensor_meta(self):
        def true_fn(x):
            return torch.tensor([[3], [2]])

        def false_fn(x):
            return torch.tensor([3.14])

        def f_return_tensor_mismatch(x):
            return cond(x.shape[0] < 3, true_fn, false_fn, [x])

        example_inputs = (torch.rand(5),)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected each tensor to have same metadata but got",
        ):
            torch._dynamo.export(
                f_return_tensor_mismatch,
                *example_inputs,
                aten_graph=True,
            )

    def test_byte_tensor_does_not_crash(self):
        # See https://github.com/pytorch/pytorch/issues/100455
        def func(text):
            tensor = torch.ByteTensor(list(bytes(text, "utf8")))
            return tensor + tensor

        text = "".join(chr(a % 90 + 40) for a in range(111))
        opt_func = torch._dynamo.optimize("eager", dynamic=True)(func)
        for i in [99, 100]:
            input = text[:i]
            opt_func(input)

    def test_export_defaults_ok(self):
        class DynamicSliceExportMod(torch.nn.Module):
            def forward(self, x):
                results = []
                for i in range(4):
                    results.append(x[: x.size(0) - i, i : x.size(2), i:3])
                return tuple(results)

        gm, _ = torch._dynamo.export(
            DynamicSliceExportMod(),
            torch.randn(5, 5, 5),
            aten_graph=True,
        )

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x):
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    slice_tensor = torch.ops.aten.slice.Tensor(arg0, 2, 0, 3)
    sym_size = torch.ops.aten.sym_size(arg0, 0)
    sub = sym_size - 1
    slice_tensor_1 = torch.ops.aten.slice.Tensor(arg0, 0, 0, sub);  sub = None
    sym_size_1 = torch.ops.aten.sym_size(arg0, 2)
    slice_tensor_2 = torch.ops.aten.slice.Tensor(slice_tensor_1, 1, 1, sym_size_1);  slice_tensor_1 = None
    slice_tensor_3 = torch.ops.aten.slice.Tensor(slice_tensor_2, 2, 1, 3);  slice_tensor_2 = None
    sub_1 = sym_size - 2
    slice_tensor_4 = torch.ops.aten.slice.Tensor(arg0, 0, 0, sub_1);  sub_1 = None
    slice_tensor_5 = torch.ops.aten.slice.Tensor(slice_tensor_4, 1, 2, sym_size_1);  slice_tensor_4 = None
    slice_tensor_6 = torch.ops.aten.slice.Tensor(slice_tensor_5, 2, 2, 3);  slice_tensor_5 = None
    sub_2 = sym_size - 3;  sym_size = None
    slice_tensor_7 = torch.ops.aten.slice.Tensor(arg0, 0, 0, sub_2);  arg0 = sub_2 = None
    slice_tensor_8 = torch.ops.aten.slice.Tensor(slice_tensor_7, 1, 3, sym_size_1);  slice_tensor_7 = sym_size_1 = None
    slice_tensor_9 = torch.ops.aten.slice.Tensor(slice_tensor_8, 2, 3, 3);  slice_tensor_8 = None
    return pytree.tree_unflatten([slice_tensor, slice_tensor_3, slice_tensor_6, slice_tensor_9], self._out_spec)""",
        )


common_utils.instantiate_parametrized_tests(ExportTests)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
