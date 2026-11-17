# Owner(s): ["module: dynamo"]

import collections
import copy
import dis
import functools
import gc
import importlib
import io
import math
import os
import pdb
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest
import unittest.mock
import warnings
import weakref

import torch
from torch.overrides import TorchFunctionMode
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    slowTest,
    TEST_NUMPY,
    TEST_WITH_CROSSREF,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)
from torch.utils._device import DeviceContext
from torch.utils._python_dispatch import TorchDispatchMode


# These tests look at which kernels the dispatcher runs; the crossref and
# dynamo test configurations interpose on exactly that.
UNDISTURBED = not (TEST_WITH_CROSSREF or TEST_WITH_TORCHDYNAMO)

# The Rust runtime is only in builds made with USE_CFG_COMPILE=1, and only
# understands the bytecode of CPython 3.12 and 3.14.
try:
    cfg_rs = importlib.import_module("torch._cfg_rs")
except ImportError:
    cfg_rs = None
# The tests are about the VM: nothing is left to the interpreter for being
# faster there, unless a test asks for it.
os.environ["TORCH_REWRITE_DEMOTE"] = "0"

HAS_RUNTIME = (
    UNDISTURBED and cfg_rs is not None and sys.version_info[:2] in ((3, 12), (3, 14))
)
canonical = torch.rewrite.canonical
# The passes over the whole function, which the tests of one pass over a block keep.
WHOLE = canonical[:3]


def without(*names):
    """The standard passes without some of them."""
    return [p for p in canonical if p not in names]


def with_graphs(backend, **options):
    """The standard passes behind a graph pass."""
    return [torch.rewrite.graphs(backend, **options), *canonical]


def narrowing():
    """The standard passes with narrow_dims in front of fold."""
    passes = list(canonical)
    passes.insert(passes.index("fold"), "narrow_dims")
    return passes


def call_outcome(fn, *args, **kwargs):
    """What a call returns, or the type and the message of what it raises."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return type(e), str(e)


# How pdb can listen: with a trace function, and from 3.14 on, where
# pdb.set_trace() and breakpoint() make it do so, through sys.monitoring.
PDB_BACKENDS = ("settrace", "monitoring")[: 1 + (sys.version_info >= (3, 14))]


def block_stats(fn, key):
    return [b[key] for b in fn.stats()["blocks"]]


@unittest.skipUnless(UNDISTURBED, "the test configuration changes dispatch")
class TestCallSite(TestCase):
    """Call sites built from the capture of one Python-level operation."""

    hw_classification = HardwareClassification.GENERIC

    def site(self, kind, run, *operands, **options):
        result, site, reason = torch._C._cfg_make_site(kind, run, operands, **options)
        self.assertIsNotNone(site, reason)
        return result, site

    def rejects(self, reason, kind, run, *operands, **options):
        site, why = torch._C._cfg_make_site(kind, run, operands, **options)[1:]
        self.assertEqual((site, why), (None, reason))

    def test_binary_operator(self):
        x, y = torch.randn(3), torch.randn(3)
        result, site = self.site("binary", lambda: x + y, x, y)
        self.assertEqual(result, torch.add(x, y))
        self.assertEqual(site.info()["op"], "aten::add.Tensor")
        self.assertEqual(site.info()["args"], ["self=$0", "other=$1", "alpha=1"])

    def test_reflected_operator(self):
        x, y = torch.randn(3), torch.randn(3)
        # What Python falls back to for x - y when x does not know how.
        result, site = self.site("binary", lambda: y.__rsub__(x), x, y)
        self.assertEqual(result, x - y)
        self.assertEqual(site.info()["op"], "aten::rsub.Tensor")
        self.assertEqual(site.info()["args"], ["self=$1", "other=$0", "alpha=1"])

    def test_unary_operator(self):
        x = torch.randn(3)
        info = self.site("unary", lambda: -x, x)[1].info()
        self.assertEqual((info["op"], info["args"]), ("aten::neg", ["self=$0"]))

    def test_rejects_what_is_not_one_dispatcher_call(self):
        x, y, bad = torch.randn(3), torch.randn(3), torch.randn(4)

        def nested():
            return torch._C._cfg_dispatch_trace(lambda: x + y)[0]

        def raises():
            try:
                return x + bad
            except RuntimeError:
                return None

        self.rejects("no dispatcher call", "binary", lambda: None, x, y)
        self.rejects("not a single dispatcher call", "binary", lambda: x + y + y, x, y)
        self.rejects("not a single dispatcher call", "binary", nested, x, y)
        self.rejects("the kernel raised", "binary", raises, x, bad)
        two_results = "operator signature is not supported"
        self.rejects(two_results, "unary", lambda: x.max(0), x)

    def test_rejects_operands_that_do_not_line_up(self):
        x, y = torch.randn(3), torch.randn(3)
        self.rejects("not a binary operator", "binary", lambda: -x, x)
        self.rejects("not a unary operator", "unary", lambda: x + y, x, y)
        self.rejects("unknown operation kind", "ternary", lambda: x + y, x, y)
        other_tensor = "tensor operand is not the kernel argument"
        self.rejects(other_tensor, "unary", lambda: -x, y)
        extra_tensor = "a tensor argument does not come from an operand"
        self.rejects(extra_tensor, "unary", lambda: x + y, x)
        index = torch.tensor([0])
        indexed = torch.ops.aten.index.Tensor
        self.rejects(extra_tensor, "unary", lambda: indexed(x, [index]), x)
        not_a_tensor = "unsupported operand type tuple"
        self.rejects(not_a_tensor, "unary", lambda: torch.stack((x, y)), (x, y))
        self.rejects("the result is not the kernel output", "unary", lambda: [-x], x)

    def test_resolution_chain(self):
        x, y = torch.randn(2, 3), torch.randn(2, 3)
        chain = self.site("binary", lambda: x + y, x, y)[1].info()["chain"]
        self.assertEqual(chain, ["AutogradCPU", "CPU"])
        chain = self.site("binary", lambda: x.__iadd__(y), x, y)[1].info()["chain"]
        self.assertEqual(chain, ["AutogradCPU", "ADInplaceOrView", "CPU"])
        # A kernel that calls other operators ends the chain.
        chain = self.site("unary", lambda: x.T, x)[1].info()["chain"]
        self.assertEqual(chain, ["AutogradCPU"])

    def test_rejects_a_torch_function_mode(self):
        x, y = torch.randn(3), torch.randn(3)
        with torch.overrides.BaseTorchFunctionMode():
            self.rejects(
                "a torch function mode is active", "binary", lambda: x + y, x, y
            )

    def check(self, kind, run, *operands, **options):
        result, site = self.site(kind, run, *operands, **options)
        self.assertEqual(site.call(*operands), result)
        return site

    def test_call_matches_eager(self):
        x, y = torch.randn(3), torch.randn(3)
        a, b = torch.randn(5), torch.randn(5)
        site = self.check("binary", lambda: x * y, x, y)
        self.assertEqual(site.call(a, b), a * b)
        site = self.check("unary", lambda: -x, x)
        self.assertEqual(site.call(a), -a)
        site = self.check(
            "call", lambda: torch.add(x, y), torch.add, x, y, callee=torch.add
        )
        self.assertEqual(site.call(torch.add, a, b), a + b)
        site = self.check("method", lambda: x.mul(y), x, y, callee=x.mul)
        self.assertEqual(site.call(a, b), a * b)

    def test_common_signatures_are_called_unboxed(self):
        x, y = torch.randn(3), torch.randn(3)
        result, site = self.site("binary", lambda: x + y, x, y)
        self.assertEqual(site.info()["calling_convention"], "unboxed")
        self.assertEqual(site.call(x, y), result)
        # clamp takes two optional scalars, which is not one of the signatures.
        clamp = torch.clamp
        result, site = self.site(
            "call", lambda: clamp(x, 0.0, 1.0), clamp, x, 0.0, 1.0, callee=clamp
        )
        self.assertEqual(site.info()["calling_convention"], "boxed")
        self.assertEqual(site.call(clamp, x, 0.0, 1.0), result)

    def test_info_of_a_call(self):
        x, y = torch.randn(3), torch.randn(3)
        site = self.check(
            "call", lambda: torch.add(x, y), torch.add, x, y, callee=torch.add
        )
        info = site.info()
        self.assertEqual(info["key"], "AutogradCPU")
        self.assertEqual((info["entry_level"], info["skips_autograd"]), (0, False))
        self.assertEqual(info["args"], ["self=$1", "other=$2", "alpha=1"])
        # The callee is an operand, but not an argument of the kernel.
        self.assertEqual(info["operand_args"], [-1, 0, 1])

    def test_number_operands(self):
        x, a = torch.randn(3), torch.randn(5)
        site = self.check("binary", lambda: x * 2, x, 2)
        self.assertEqual(site.call(a, 3), a * 3)
        self.assertIs(site.call(a, 2.5), NotImplemented)
        site = self.check("binary", lambda: x * 2.5, x, 2.5)
        self.assertEqual(site.call(a, 0.5), a * 0.5)
        site = self.check("binary", lambda: 2 - x, 2, x)
        self.assertEqual(site.info()["op"], "aten::rsub.Scalar")
        self.assertEqual(site.info()["args"], ["self=$1", "other=$0", "alpha=1"])
        self.assertEqual(site.call(3, a), 3 - a)
        self.rejects(
            "int operand does not fit in int64", "binary", lambda: x * 2, x, 2**70
        )

    def test_keyword_operands(self):
        x, y = torch.randn(3), torch.randn(3)
        a, b = torch.randn(5), torch.randn(5)
        site = self.check(
            "call",
            lambda: torch.add(x, y, alpha=2),
            *(torch.add, x, y, 2),
            callee=torch.add,
            kwnames=("alpha",),
        )
        self.assertEqual(site.info()["args"], ["self=$1", "other=$2", "alpha=$3"])
        self.assertEqual(site.call(torch.add, a, b, 3), torch.add(a, b, alpha=3))
        neg = torch.neg
        site = self.check(
            "call", lambda: neg(input=x), neg, x, callee=neg, kwnames=("input",)
        )
        self.assertEqual(site.info()["args"], ["self=$1"])
        self.rejects(
            "keyword 'axis' is not a schema argument",
            "call",
            lambda: torch.sum(x, axis=0),
            *(torch.sum, x, 0),
            callee=torch.sum,
            kwnames=("axis",),
        )

    def test_int_sequence_operands(self):
        m = torch.randn(2, 3)
        site = self.check(
            "method", lambda: m.reshape((3, 2)), m, (3, 2), callee=m.reshape
        )
        self.assertEqual(site.call(m, (6, 1)), m.reshape((6, 1)))
        self.assertIs(site.call(m, [6, 1]), NotImplemented)
        # m.view(3, 2): the argument parser packs the positional ints.
        site = self.check("method", lambda: m.view(3, 2), m, 3, 2, callee=m.view)
        self.assertEqual(site.info()["args"], ["self=$0", "size=[$1, $2]"])
        self.assertEqual(site.call(m, 6, 1), m.view(6, 1))

    def test_operands_baked_into_the_site(self):
        x, a = torch.randn(3), torch.randn(5)
        # A constant of the compiled function is not looked at again.
        site = self.check("binary", lambda: x * 2, x, 2, constants=(False, True))
        self.assertEqual(site.info()["args"], ["self=$0", "other=<tensor>"])
        self.assertEqual(site.call(a, None), a * 2)
        clamp = torch.clamp
        site = self.check(
            "call", lambda: clamp(x, None, 0.5), clamp, x, None, 0.5, callee=clamp
        )
        self.assertEqual(site.info()["args"], ["self=$1", "min=None", "max=$3"])
        self.assertEqual(site.call(clamp, a, None, 0.1), clamp(a, None, 0.1))
        self.assertIs(site.call(clamp, a, 0.0, 0.1), NotImplemented)
        # A dtype is guarded by identity.
        site = self.check(
            "method", lambda: x.to(torch.float64), x, torch.float64, callee=x.to
        )
        self.assertEqual(site.call(a, torch.float64), a.double())
        self.assertIs(site.call(a, torch.float32), NotImplemented)

    def test_non_tensor_results(self):
        x, a = torch.randn(3), torch.randn(5)
        equal = torch.equal
        site = self.check("call", lambda: equal(x, x), equal, x, x, callee=equal)
        self.assertIs(site.call(equal, a, a), True)
        self.assertIs(site.call(equal, a, -a), False)
        one = x[0]
        site = self.check("method", lambda: one.item(), one, callee=one.item)
        self.assertEqual(site.call(a[1]), a[1].item())
        site = self.check("unary", lambda: bool(one), one)
        self.assertIs(site.call(torch.zeros(())), False)

    def test_guard_misses(self):
        x, y = torch.randn(3), torch.randn(3)
        site = self.check("binary", lambda: x * y, x, y)
        self.assertIs(site.call(x, 2), NotImplemented)
        self.assertIs(site.call(x.to_sparse(), y), NotImplemented)
        with torch.overrides.BaseTorchFunctionMode():
            self.assertIs(site.call(x, y), NotImplemented)
        autograd = torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradCPU)
        with torch._C._ExcludeDispatchKeyGuard(autograd):
            self.assertIs(site.call(x, y), NotImplemented)
        with torch.no_grad():
            self.assertEqual(site.call(x, y), x * y)
        site = self.check(
            "call", lambda: torch.add(x, y), torch.add, x, y, callee=torch.add
        )
        self.assertIs(site.call(torch.sub, x, y), NotImplemented)
        with self.assertRaisesRegex(ValueError, "wrong number of operands"):
            site.call(x, y)

    def test_kernel_error_raises(self):
        x, y = torch.randn(3), torch.randn(3)
        site = self.check("binary", lambda: x + y, x, y)
        with self.assertRaisesRegex(RuntimeError, "must match"):
            site.call(x, torch.randn(4))
        self.assertEqual(site.call(x, y), x + y)

    def test_site_called_from_inside_another_site(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        x, y = torch.ones(3), torch.ones(3)
        state = {"inner": None, "nest": False}

        class CallsASite(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if state["inner"] is None:
                    state["inner"] = torch._C._cfg_make_site(
                        "binary", lambda: x * y, (x, y)
                    )[1]
                elif state["nest"]:
                    state["nest"] = False
                    state["inner"].call(x, y)
                return func(*args, **(kwargs or {}))

        with CallsASite():
            _ = x + y
            outer = self.site("binary", lambda: x + y, x, y)[1]
            state["nest"] = True
            self.assertEqual(outer.call(x, y), torch.full((3,), 2.0))
        self.assertFalse(state["nest"])

    def test_miss_on_a_sequence_element_leaves_no_error(self):
        m = torch.randn(2, 3)
        site = self.site(
            "method", lambda: m.reshape((3, 2)), m, (3, 2), callee=m.reshape
        )[1]
        self.assertEqual(site.call(m, (6, 1)), m.reshape((6, 1)))
        self.assertIs(site.call(m, (6, 1.0)), NotImplemented)

    def test_rejected_site_releases_its_references(self):
        import sys

        x = torch.randn(3)
        before = sys.getrefcount(torch.sum)
        for _ in range(100):
            self.rejects(
                "keyword 'axis' is not a schema argument",
                "call",
                lambda: torch.sum(x, axis=0),
                torch.sum,
                x,
                0,
                callee=torch.sum,
                kwnames=("axis",),
            )
        self.assertLess(sys.getrefcount(torch.sum) - before, 10)

    def test_autograd_is_kept_by_default(self):
        x, y = torch.randn(3), torch.randn(3)
        g = torch.randn(3, requires_grad=True)
        site = self.check("binary", lambda: x * y, x, y)
        site.call(g, y).sum().backward()
        self.assertEqual(g.grad, y)

    def test_skip_autograd(self):
        x, y = torch.randn(3), torch.randn(3)
        g = torch.randn(3, requires_grad=True)
        site = self.check("binary", lambda: x * y, x, y, skip_autograd=True)
        info = site.info()
        self.assertEqual(info["chain"], ["AutogradCPU", "CPU"])
        self.assertEqual((info["key"], info["entry_level"]), ("CPU", 1))
        self.assertTrue(info["skips_autograd"])
        self.assertIs(site.call(g, y), NotImplemented)
        with torch.no_grad():
            self.assertEqual(site.call(g, y), g * y)
        # A call that tracked a gradient when it was captured keeps autograd.
        site = self.check("binary", lambda: g * y, g, y, skip_autograd=True)
        info = site.info()
        self.assertEqual((info["key"], info["entry_level"]), ("AutogradCPU", 0))
        self.assertFalse(info["skips_autograd"])
        self.assertIsNotNone(site.call(g, y).grad_fn)

    def test_skip_autograd_of_a_composite(self):
        linear = torch.nn.functional.linear
        x, w = torch.randn(2, 3), torch.randn(4, 3)
        site = self.check(
            "call",
            lambda: linear(x, w),
            linear,
            x,
            w,
            callee=linear,
            skip_autograd=True,
        )
        info = site.info()
        # There is no kernel below to enter at: linear itself is the entry.
        self.assertEqual((info["chain"], info["entry_level"]), (["AutogradCPU"], 0))
        self.assertTrue(info["skips_autograd"])
        self.assertFalse(site.call(linear, x, w).requires_grad)
        self.assertIs(site.call(linear, x, w.clone().requires_grad_()), NotImplemented)

    def test_rejects_callees_that_are_not_torch_builtins(self):
        x, y = torch.randn(3), torch.randn(3)
        relu = torch.nn.functional.relu
        self.rejects(
            "callee is not a torch builtin",
            "call",
            lambda: relu(x),
            relu,
            x,
            callee=relu,
        )
        self.rejects(
            "callee is not a torch builtin method of the receiver",
            "method",
            lambda: x.mul(y),
            *(x, y),
            callee=y.mul,
        )

    def test_c_api_is_a_capsule(self):
        self.assertEqual(type(torch._C._cfg_c_api()).__name__, "PyCapsule")


@unittest.skipUnless(UNDISTURBED, "the test configuration changes dispatch")
class TestDispatchCapture(TestCase):
    """The dispatcher hook that torch.rewrite records kernels with."""

    hw_classification = HardwareClassification.GENERIC

    def trace(self, fn):
        return torch._C._cfg_dispatch_trace(fn)[1]

    def detail(self, fn):
        return torch._C._cfg_dispatch_trace(fn, detail=True)[1]

    def test_call_and_redispatch_chain(self):
        x, y = torch.randn(3), torch.randn(3)
        self.assertEqual(
            self.trace(lambda: torch.add(x, y)),
            [
                ("aten::add.Tensor", "AutogradCPU", 0, False),
                ("aten::add.Tensor", "CPU", 1, True),
            ],
        )

    def test_inplace_goes_through_adinplaceorview(self):
        x, y = torch.randn(3), torch.randn(3)
        records = self.trace(lambda: x.add_(y))
        self.assertEqual(
            [r[1] for r in records[:3]], ["AutogradCPU", "ADInplaceOrView", "CPU"]
        )
        self.assertEqual([r[3] for r in records[:3]], [False, True, True])

    def test_composite_kernel_makes_nested_calls(self):
        x, w = torch.randn(2, 3), torch.randn(4, 3)
        records = self.trace(lambda: torch.nn.functional.linear(x, w))
        self.assertEqual(records[0][0], "aten::linear")
        nested = [r for r in records if r[2] == 1 and not r[3]]
        self.assertIn("aten::matmul", [r[0] for r in nested])

    def test_result_is_returned(self):
        x = torch.ones(3)
        result, records = torch._C._cfg_dispatch_trace(lambda: x * 2)
        self.assertEqual(result, torch.full((3,), 2.0))
        self.assertEqual(records[0][0], "aten::mul.Tensor")

    def test_detail_links_kernels_to_their_callers(self):
        x, y = torch.randn(3), torch.randn(3)
        records = self.detail(lambda: x.add_(y))
        self.assertEqual(
            [(r["key"], r["parent"]) for r in records],
            [("AutogradCPU", -1), ("ADInplaceOrView", 0), ("CPU", 1)],
        )
        x, w = torch.randn(2, 3), torch.randn(4, 3)
        records = self.detail(lambda: torch.nn.functional.linear(x, w))
        self.assertEqual(records[0]["parent"], -1)
        for r in records[1:]:
            self.assertEqual(records[r["parent"]]["depth"], r["depth"] - 1)

    def test_detail_marks_kernels_that_raised(self):
        x, bad = torch.ones(3), torch.ones(4)

        def fn():
            with self.assertRaisesRegex(RuntimeError, "must match"):
                torch.add(x, bad)
            return torch.add(x, x)

        records = self.detail(fn)
        self.assertEqual([r["threw"] for r in records], [True, True, False, False])
        self.assertEqual(records[0]["outputs"], ())

    def test_detail_has_the_thread_local_key_sets(self):
        x, y = torch.randn(3), torch.randn(3)
        top, backend = self.detail(lambda: torch.add(x, y))
        excluded = str(torch._C._dispatch_tls_local_exclude_set())
        self.assertNotIn("AutogradOther", excluded)
        self.assertEqual(top["tls_excluded"], excluded)
        # The Autograd kernel excludes the autograd keys before it redispatches.
        self.assertIn("AutogradOther", backend["tls_excluded"])
        self.assertEqual(top["tls_included"], backend["tls_included"])

    def test_detail_keeps_inputs_and_outputs_of_top_level_calls(self):
        x, y = torch.randn(3), torch.randn(3)
        result, (top, backend) = torch._C._cfg_dispatch_trace(
            lambda: torch.add(x, y), detail=True
        )
        self.assertIs(top["inputs"][0], x)
        self.assertIs(top["inputs"][1], y)
        # alpha, a default the argument parser filled in
        self.assertEqual(top["inputs"][2:], (1,))
        self.assertEqual(len(top["outputs"]), 1)
        self.assertIs(top["outputs"][0], result)
        self.assertEqual((backend["inputs"], backend["outputs"]), (None, None))

    def test_exception_propagates(self):
        x = torch.ones(3)
        self.assertEqual(self.trace(lambda: 1 + 1), [])
        with self.assertRaisesRegex(RuntimeError, "must match"):
            self.trace(lambda: x + torch.ones(4))
        self.assertEqual(self.trace(lambda: None), [])

    def test_captures_nest(self):
        x = torch.ones(3)

        def outer():
            inner = self.trace(lambda: x + 1)
            self.assertEqual(len([r for r in inner if r[2] == 0]), 1)
            return x * 2

        records = self.trace(outer)
        self.assertEqual([r[0] for r in records if r[2] == 0], ["aten::mul.Tensor"])


G = 10


def py_arith(a, b):
    return (
        (a + b) * 2
        - a // 3
        + a % 4
        + b**2
        - (a & b)
        + (a | 1)
        + (a ^ b)
        + (a << 1)
        + a / 3
    )


def py_branch(a, b):
    if a > b:
        return a - b
    elif a == b:
        return 0
    return b - a


def py_loop(n, m):
    s = 0
    for i in range(n):
        if i % 2 == 0:
            s += i
        else:
            s -= m
    return s


def py_while(n, m):
    i = 0
    acc = []
    while i < n:
        acc.append(i * m)
        i += 1
        if i == 7:
            break
    else:
        acc.append(-1)
    return acc


def py_nested(n, m):
    out = []
    for i in range(n):
        for j in range(m):
            if j > i:
                continue
            out.append((i, j))
    return out


def py_containers(a, b):
    t = (a, b)
    lst = [a, b, *t]
    extra = {"c": 3}
    d = {"a": a, "b": b, **extra}
    return t, lst, d, lst[0], d["a"], lst[1:3], lst[::-1], a in lst, a is not None


def py_comprehensions(n, m):
    return (
        [i * m for i in range(n) if i % 3],
        {i: i * i for i in range(n)},
        {i % 3 for i in range(n)},
    )


def py_strings(a, b):
    return f"{a}:{b!r}:{a:>5}|", "-".join([str(a), str(b)]), str(a) + str(b)


def py_globals_and_builtins(a, b):
    return G + a + len([a, b]) + max(a, b) + math.floor(a / 2)


def py_unpack(a, b):
    x, (y, z) = a, (b, a + b)
    return x * 100 + y * 10 + z


def py_setitem(n, m):
    lst = [0] * n
    d = {}
    for i in range(n):
        lst[i] = i * m
        d[i] = lst[i] * 2
    lst[1:3] = [9, 9]
    del d[0]
    return lst, d


def py_ternary(a, b):
    return (
        ("neg" if a < 0 else "zero" if a == 0 else "pos"),
        (a and b),
        (a or b),
        (not a),
    )


PY_FUNCTIONS = [
    py_arith,
    py_branch,
    py_loop,
    py_while,
    py_nested,
    py_containers,
    py_comprehensions,
    py_strings,
    py_globals_and_builtins,
    py_unpack,
    py_setitem,
    py_ternary,
]


@unittest.skipUnless(HAS_RUNTIME, "torch.rewrite runtime not built")
class TestCfgFrontend(TestCase):
    """Bytecode -> SSA graph, and the generic tier that runs it."""

    hw_classification = HardwareClassification.GENERIC

    def test_decoder_matches_dis(self):
        def loop(x, n):
            total = 0
            for i in range(n):
                if i % 2:
                    continue
                total += x[i]
            return total

        # Enough constants and a long enough body for EXTENDED_ARG prefixes.
        namespace = {}
        body = "".join(
            f"    x = x + {i}.5 if x > {i} else x - {i}.5\n" for i in range(300)
        )
        exec(f"def wide(x):\n{body}    return x\n", namespace)

        for fn in (loop, namespace["wide"]):
            expected = [
                (
                    i.offset,
                    i.opname,
                    i.arg or 0,
                    i.argval if i.opcode in dis.hasjrel else None,
                )
                for i in dis.get_instructions(fn.__code__)
                if i.opname != "EXTENDED_ARG"
            ]
            self.assertEqual(cfg_rs.decode(fn.__code__), expected)
        self.assertTrue(
            any(arg > 255 for _, _, arg, _ in cfg_rs.decode(namespace["wide"].__code__))
        )

    @parametrize("fn", PY_FUNCTIONS, name_fn=lambda fn: fn.__name__)
    @parametrize(
        "args", [(7, 3), (0, 4), (10, 2)], name_fn=lambda args: f"{args[0]}_{args[1]}"
    )
    def test_matches_cpython(self, fn, args):
        def outcome(f):
            try:
                return f(*args)
            except Exception as e:
                return type(e), str(e)

        compiled = torch.rewrite(fn, fallback=False)
        for _ in range(2):
            self.assertEqual(outcome(compiled), outcome(fn))

    def test_whole_function_is_one_graph(self):
        def f(x, n):
            for _ in range(n):
                if x.sum() > 0:
                    x = x - 1
                else:
                    x = x + 2
            return x

        text = torch.rewrite(f, fallback=False).dump()
        self.assertIn("phi", text)
        self.assertIn("br_exhausted", text)
        self.assertIn("binary - ", text)
        self.assertIn("binary + ", text)

    def test_arguments(self):
        def f(a, b=2, *args, c=3, **kw):
            return a, b, args, c, sorted(kw.items())

        compiled = torch.rewrite(f, fallback=False)
        for args, kwargs in [
            ((1,), {}),
            ((1, 5, 6, 7), {}),
            ((1,), {"c": 4, "z": 1}),
            ((), {"a": 1, "b": 3}),
        ]:
            self.assertEqual(compiled(*args, **kwargs), f(*args, **kwargs))
        with self.assertRaisesRegex(TypeError, "multiple values"):
            compiled(1, a=2)
        with self.assertRaisesRegex(TypeError, "missing"):
            compiled()

    def test_closure_and_recursion(self):
        k = 3

        def fact(n):
            return k if n <= 1 else n * compiled(n - 1)

        compiled = torch.rewrite(fact, fallback=False)
        self.assertEqual(compiled(5), 360)

    def test_exceptions(self):
        def f(a):
            if a < 0:
                raise ValueError("negative")
            if a > 100:
                return undefined_name  # noqa: F821
            if a > 10:
                del a
                return a
            return math.sqrt(a)

        compiled = torch.rewrite(f, fallback=False)
        self.assertEqual(compiled(4), 2.0)
        with self.assertRaisesRegex(ValueError, "negative"):
            compiled(-1)
        with self.assertRaises(UnboundLocalError):
            compiled(11)
        with self.assertRaises(NameError):
            compiled(101)

        # The function has to contain an assert statement, so it is given as
        # source.
        namespace = {}
        exec("def checked(a):\n    assert a != 1, 'not one'\n    return a\n", namespace)
        compiled = torch.rewrite(namespace["checked"], fallback=False)
        self.assertEqual(compiled(2), 2)
        with self.assertRaisesRegex(AssertionError, "not one"):
            compiled(1)

    def test_traceback_names_the_line(self):
        def f(x, y):
            z = x + 1
            return z @ y

        compiled = torch.rewrite(f, fallback=False)
        for _ in range(3):
            try:
                compiled(torch.randn(2, 3), torch.randn(2, 3))
            except RuntimeError:
                tb = traceback.format_exc()
            else:
                self.fail("expected a shape error")
            self.assertIn(f"line {f.__code__.co_firstlineno + 2}, in f", tb)

    def test_unsupported(self):
        def f(x):
            yield x + 1

        with self.assertRaisesRegex(torch.rewrite.Unsupported, "generators"):
            torch.rewrite(f, fallback=False)
        self.assertTrue(torch.rewrite.is_available())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIs(torch.rewrite(f), f)
        self.assertIn(
            "left f as it is", str(caught[0].message).replace(f.__qualname__, "f")
        )

    def test_methods_and_modules(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.ones(3))

            def forward(self, x):
                return x * self.w + 1

        m = M()
        x = torch.randn(3)
        for compiled in (torch.rewrite(m), torch.rewrite(m.forward)):
            for _ in range(3):
                self.assertEqual(compiled(x), m(x))


instantiate_parametrized_tests(TestCfgFrontend)


@unittest.skipUnless(HAS_RUNTIME, "torch.rewrite runtime not built")
class TestCfgCompile(TestCase):
    """Profiling on real tensors and direct kernel calls."""

    hw_classification = HardwareClassification.ACCELERATOR

    def assertCompiledEqual(self, fn, argsets, **kwargs):
        compiled = torch.rewrite(fn, fallback=False, **kwargs)
        for args in argsets:
            self.assertEqual(compiled(*args), fn(*args))
        return compiled

    def test_kernels_are_called_directly(self, device):
        def f(x, y):
            z = x + y
            z = torch.mul(z, y)
            return z.relu()

        argsets = [
            (torch.randn(4, device=device), torch.randn(4, device=device))
            for _ in range(4)
        ]
        with torch.no_grad():
            compiled = self.assertCompiledEqual(f, argsets)
        stats = compiled.stats()
        self.assertEqual(stats["sites"], 3)
        self.assertEqual(stats["unlowered"], [])
        self.assertEqual(stats["site_misses"], 0)
        self.assertEqual(stats["site_hits"], 9)
        backend = "CPU" if device == "cpu" else "CUDA"
        for site in compiled.sites():
            (variant,) = site["variants"]
            # Recorded Autograd -> backend, entered at the backend kernel.
            self.assertEqual(variant["chain"], ["Autograd" + backend, backend])
            self.assertEqual(variant["key"], backend)
        self.assertIn("dispatch aten::add.Tensor", compiled.dump())

    def test_data_dependent_control_flow(self, device):
        def f(x, y):
            while x.sum() < 100:
                if x.max() > 10:
                    x = x * 3
                else:
                    x = x * 2 + y
            return x

        argsets = [
            (torch.full((3,), float(v), device=device), torch.ones(3, device=device))
            for v in (1, 20, 0.5, 200, 3)
        ]
        compiled = self.assertCompiledEqual(f, argsets)
        self.assertEqual(compiled.stats()["unlowered"], [])

    def test_shapes_are_dynamic(self, device):
        def f(x, y):
            if x.shape[0] > 2:
                x = x.sum(0, keepdim=True)
            return (x * y).reshape(-1)

        argsets = [
            (torch.randn(n, m, device=device), torch.randn(m, device=device))
            for n, m in [(1, 2), (5, 2), (3, 7), (1, 1), (9, 4)]
        ]
        compiled = self.assertCompiledEqual(f, argsets)
        # No block was ever sent back to the profiling tier.
        self.assertLessEqual(max(block_stats(compiled, "profile_runs")), 1)
        self.assertEqual(sum(block_stats(compiled, "guard_failures")), 0)

    def test_scalars(self, device):
        def f(x, a, b):
            y = x * 2 + a
            y = y / b - 1.5
            y = torch.clamp(y, -3, 3)
            return (2 - y) ** 2 + (y > a)

        argsets = [
            (torch.randn(4, device=device), a, b)
            for a, b in [(1, 2.0), (3, 0.5), (2.5, 4), (True, 3)]
        ]
        compiled = self.assertCompiledEqual(f, argsets)
        self.assertEqual(compiled.stats()["unlowered"], [])

    def test_autograd(self, device):
        def f(x, w):
            y = (x @ w).tanh() * 2
            if y.sum() > 0:
                y = y + 1
            return (y * y).sum()

        compiled = torch.rewrite(f, fallback=False)
        with torch.no_grad():
            for _ in range(2):
                compiled(
                    torch.randn(3, 3, device=device), torch.randn(3, 3, device=device)
                )
        for _ in range(3):
            x = torch.randn(3, 3, device=device)
            w = torch.randn(3, 3, device=device, requires_grad=True)
            w_ref = w.detach().clone().requires_grad_()
            out, ref = compiled(x, w), f(x, w_ref)
            self.assertTrue(out.requires_grad)
            out.backward()
            ref.backward()
            self.assertEqual(out, ref)
            self.assertEqual(w.grad, w_ref.grad)
        with torch.no_grad():
            self.assertFalse(compiled(x, w).requires_grad)

    def test_autograd_is_only_skipped_when_unused(self, device):
        def f(x, y):
            return x * y

        compiled = torch.rewrite(f, fallback=False)
        x = torch.randn(3, device=device)
        compiled(x, x)
        compiled(x, x.clone().requires_grad_())
        compiled(x, x)
        (site,) = compiled.sites()
        self.assertEqual([v["entry_level"] for v in site["variants"]], [1, 0])
        self.assertEqual((site["hits"], site["misses"]), (1, 1))

        strict = torch.rewrite(f, fallback=False, skip_autograd=False)
        strict(x, x)
        (site,) = strict.sites()
        self.assertEqual([v["entry_level"] for v in site["variants"]], [0])

    def test_inplace_and_views(self, device):
        def f(x, y):
            v = x.view(-1)
            v.add_(y.view(-1))
            x *= 2
            t = x.t()
            t.sub_(1)
            return x

        compiled = torch.rewrite(f, fallback=False)
        for _ in range(3):
            a = torch.randn(3, 3, device=device)
            b = torch.randn(3, 3, device=device)
            a_ref = a.clone()
            out = compiled(a, b)
            f(a_ref, b)
            self.assertIs(out, a)
            self.assertEqual(a, a_ref)
            self.assertEqual(a._version, a_ref._version)

        def leaf(x):
            return x.add_(1)

        compiled = torch.rewrite(leaf, fallback=False)
        x = torch.randn(3, device=device, requires_grad=True)
        for _ in range(3):
            with self.assertRaisesRegex(RuntimeError, "leaf Variable"):
                compiled(x)

    def test_sites_are_polymorphic(self, device):
        def f(x, y):
            return (x * y + 2).sum()

        compiled = torch.rewrite(f, fallback=False)
        makers = [
            lambda: torch.randn(4, device=device),
            lambda: torch.randn(4, device=device, dtype=torch.float64),
            lambda: torch.arange(4, device=device),
            lambda: torch.randn(2, 2, 2, device=device),
            lambda: torch.randn(4, device=device, requires_grad=True),
            lambda: torch.randn(4, device="cpu"),
        ]
        for make in makers:
            for _ in range(2):
                a = make()
                b = torch.ones_like(a).detach()
                out, ref = compiled(a, b), f(a, b)
                self.assertEqual(out, ref)
                self.assertEqual(out.dtype, ref.dtype)
                self.assertEqual(out.requires_grad, ref.requires_grad)
        # One version of the block serves all of it; the sites grew variants.
        self.assertEqual(block_stats(compiled, "versions"), [1])
        self.assertEqual(block_stats(compiled, "profile_runs"), [1])

    def test_dispatch_modes_still_see_the_operators(self, device):
        class Log(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.ops = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                self.ops.append(func.__name__)
                return func(*args, **(kwargs or {}))

        class Count(TorchFunctionMode):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls += 1
                return func(*args, **(kwargs or {}))

        def f(x, y):
            return (x + y).relu() * y

        compiled = torch.rewrite(f, fallback=False)
        x, y = torch.randn(3, device=device), torch.randn(3, device=device)
        for _ in range(3):
            compiled(x, y)
        with Log() as log:
            out = compiled(x, y)
        self.assertEqual(log.ops, ["add.Tensor", "relu.default", "mul.Tensor"])
        self.assertEqual(out, f(x, y))
        with Count() as count:
            compiled(x, y)
        self.assertEqual(count.calls, 3)
        self.assertEqual(torch.func.vmap(compiled)(x[None], y[None]), f(x, y)[None])

    def test_composite_runs_with_autograd_excluded(self, device):
        def f(x, w):
            return x @ w

        compiled = torch.rewrite(f, fallback=False)
        x, w = torch.randn(2, 3, device=device), torch.randn(3, 3, device=device)
        for _ in range(3):
            self.assertEqual(compiled(x, w), f(x, w))
        (site,) = compiled.sites()
        (variant,) = site["variants"]
        # matmul has no kernel of its own below Autograd: it stays the entry.
        self.assertEqual((variant["entry_level"], variant["skips_autograd"]), (0, True))
        w.requires_grad_()
        out = compiled(x, w)
        out.sum().backward()
        self.assertEqual(w.grad, x.sum(0)[:, None].expand(3, 3))

    def test_recursion_and_threads(self, device):
        def f(x, n):
            if n == 0:
                return x
            return compiled(x * 2, n - 1) + 1

        compiled = torch.rewrite(f, fallback=False)
        x = torch.ones(2, device=device)
        self.assertEqual(compiled(x, 3), torch.full((2,), 11.0))

        results = []

        def work():
            for _ in range(200):
                results.append(compiled(x, 2).sum().item())

        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(set(results), {12.0})
        self.assertEqual(len(results), 800)

    def test_intermediates_never_get_a_python_object(self, device):
        def f(x, y):
            z = x + y
            z = z * y
            return z.relu()

        compiled = torch.rewrite(f, fallback=False)
        x, y = torch.randn(4, device=device), torch.randn(4, device=device)
        compiled(x, y)
        before = compiled.stats()
        for _ in range(5):
            self.assertEqual(compiled(x, y), f(x, y))
        after = compiled.stats()
        # Three kernel results per call; only the one that is returned ever
        # becomes a Python object.
        self.assertEqual(after["unboxed_results"] - before["unboxed_results"], 15)
        self.assertEqual(after["boxed_later"] - before["boxed_later"], 5)

        boxed = torch.rewrite(f, fallback=False, unboxed=False)
        for _ in range(3):
            self.assertEqual(boxed(x, y), f(x, y))
        self.assertEqual(boxed.stats()["unboxed_results"], 0)

    def test_unboxed_tensors_reach_python_intact(self, device):
        seen = []

        def look(t):
            seen.append(t)
            return t.shape[0]

        def f(x, y, n):
            z = x + y
            alias = z
            z += 1  # in place, on a tensor without a Python object
            for _ in range(n):  # carried around the loop unboxed
                z = z * 2
            same = alias is z
            k = look(z)  # generic code gets a Python object
            t = z.t()
            t.add_(1)  # through a view
            return z, [z, t], same, k, z is seen[-1]

        compiled = torch.rewrite(f, fallback=False)
        for n in (0, 3, 0, 2):
            x = torch.randn(3, 2, device=device)
            y = torch.randn(3, 2, device=device)
            out, ref = compiled(x, y, n), f(x, y, n)
            self.assertEqual(out[0], ref[0])
            self.assertEqual(out[1], ref[1])
            self.assertEqual(out[2:], ref[2:])
            self.assertIs(out[0], out[1][0])
            self.assertIs(out[0], seen[-2])
        self.assertGreater(compiled.stats()["unboxed_results"], 0)

    def test_unboxed_tensors_are_released(self, device):
        def f(x, y, fail):
            a = x + y
            b = a * 2
            if fail:
                return b @ y.reshape(-1)[:1]
            return (a - b).sum()

        compiled = torch.rewrite(f, fallback=False)
        x = torch.randn(64, 64, device=device)
        for _ in range(3):
            compiled(x, x, False)
        if device != "cpu":
            torch.accelerator.synchronize()
            baseline = torch.accelerator.memory_allocated()
        for _ in range(10):
            compiled(x, x, False)
            with self.assertRaises(RuntimeError):
                compiled(x, x, True)
        if device != "cpu":
            torch.accelerator.synchronize()
            self.assertEqual(torch.accelerator.memory_allocated(), baseline)

    def test_first_call_is_the_trace(self, device):
        calls = []

        def f(x):
            calls.append(1)
            return x * 2

        compiled = torch.rewrite(f, fallback=False)
        x = torch.ones(2, device=device)
        self.assertEqual(compiled(x), x * 2)
        # The function ran once, on the real input: no separate tracing run.
        self.assertEqual(len(calls), 1)
        self.assertEqual(block_stats(compiled, "versions"), [1])


instantiate_device_type_tests(TestCfgCompile, globals())


@unittest.skipUnless(HAS_RUNTIME, "torch.rewrite runtime not built")
class TestCfgPasses(TestCase):
    """The pass pipeline, guards, and recompilation of single blocks."""

    hw_classification = HardwareClassification.GENERIC

    def test_function_attributes_are_read_through(self):
        @unittest.expectedFailure
        def f(x, k=2):
            return x * k

        f.note = "set by a decorator"
        compiled = torch.rewrite(f, fallback=False)
        self.assertIs(compiled.__code__, f.__code__)
        self.assertEqual(compiled.__defaults__, (2,))
        self.assertEqual(compiled.note, "set by a decorator")
        self.assertTrue(compiled.__unittest_expecting_failure__)
        self.assertEqual(compiled.__name__, "f")
        with self.assertRaisesRegex(
            AttributeError, "'function' object has no attribute 'nothing'"
        ):
            compiled.nothing

    def test_ndim(self):
        # Nothing is loaded between x and its ndim, unlike for x.dim(): the
        # pass that turns both into the rank must not look for what is.
        def rank(x):
            return x.ndim + 1

        compiled = torch.rewrite(rank, fallback=False)
        for shape in ((2, 3), (2, 3), (4,), (2, 3, 4)):
            self.assertEqual(compiled(torch.ones(shape)), len(shape) + 1)
        self.assertIn("tensor_dim", compiled.dump())

    def test_what_passes_takes(self):
        self.assertNotIn("narrow_dims", canonical)
        with self.assertRaisesRegex(ValueError, "unknown pass 'nope'"):
            torch.rewrite(lambda x: x, passes=["nope"])
        with self.assertRaisesRegex(TypeError, "not int"):
            torch.rewrite(lambda x: x, passes=[3])
        with self.assertRaisesRegex(ValueError, "comes after 'fold'"):
            torch.rewrite(lambda x: x, passes=["fold", torch.rewrite.inductor])
        two = [torch.rewrite.inductor, torch.rewrite.graphs("aot_eager")]
        with self.assertRaisesRegex(ValueError, "one of them"):
            torch.rewrite(lambda x: x, passes=two)

    def test_passes_are_applied_in_the_order_of_the_list(self):
        def f(x):
            return x * 2 + 1

        def run(place):
            seen = []

            def look(region):
                seen.append({i["op"] for i in region.instructions()})
                return False

            passes = ["resolve_dispatch"]
            passes.insert(place, look)
            compiled = torch.rewrite(f, passes=passes, fallback=False)
            for _ in range(3):
                self.assertEqual(compiled(torch.ones(2)), torch.full((2,), 3.0))
            return seen[0]

        # In front of resolve_dispatch a pass sees what the function wrote, and
        # behind it the kernel calls that those became.
        self.assertNotIn("dispatch", run(0))
        self.assertIn("dispatch", run(1))

    def test_a_pass_over_the_whole_function_is_asked_for_by_name(self):
        def run(passes):
            compiled = torch.rewrite(inl_shifted, passes=passes, fallback=False)
            for _ in range(3):
                self.assertEqual(
                    compiled(torch.ones(2), torch.ones(2)),
                    inl_shifted(torch.ones(2), torch.ones(2)),
                )
            return compiled.stats()["inlined"]

        self.assertEqual(run(["resolve_dispatch"]), [])
        self.assertEqual(run(["inline", "resolve_dispatch"]), ["inl_scale"])
        self.assertEqual(run([]), [])

    def test_generic_tier_only(self):
        def f(x, n):
            for _ in range(n):
                x = x * 2 + 1
            return x

        compiled = torch.rewrite(f, passes=[], fallback=False)
        for _ in range(3):
            self.assertEqual(compiled(torch.ones(2), 3), torch.full((2,), 15.0))
        self.assertEqual(sum(block_stats(compiled, "versions")), 0)
        self.assertEqual(sum(block_stats(compiled, "profile_runs")), 0)
        self.assertNotIn("specialized", compiled.dump())

    def test_without_passes_over_a_block_nothing_is_specialized(self):
        def f(x):
            return x + 1

        compiled = torch.rewrite(f, passes=WHOLE, fallback=False)
        for _ in range(3):
            self.assertEqual(compiled(torch.ones(2)), torch.full((2,), 2.0))
        self.assertEqual(block_stats(compiled, "versions"), [0])

    def test_narrow_dims_recompiles_only_the_guarded_block(self):
        def f(x, y):
            if x.shape[0] > 1:
                x = x.mean(0, keepdim=True)
            z = x + y
            while z.sum() < 10:
                z = z * 2 + 1
            return z

        compiled = torch.rewrite(f, passes=narrowing(), fallback=False)

        def run(n):
            x, y = torch.ones(n, 3), torch.ones(1, 3)
            self.assertEqual(compiled(x, y), f(x, y))

        run(4)
        first = block_stats(compiled, "profile_runs")
        self.assertEqual(first[0], 1)
        # The shape test of block 0 is gone: it jumps, under a range guard.
        text = compiled.dump()
        self.assertIn("if rank(%x.1) >= 1 and 2 <= size(%x.1, 0) <= inf", text)

        for n in (2, 8, 100):
            run(n)
        self.assertEqual(block_stats(compiled, "profile_runs"), first)

        run(1)
        after = block_stats(compiled, "profile_runs")
        self.assertEqual(after[0], 2)
        self.assertEqual(after[1:], first[1:])
        self.assertEqual(block_stats(compiled, "versions")[0], 2)
        self.assertIn("0 <= size(%x.1, 0) <= 1", compiled.dump())

        for n in (1, 5, 1, 9, 2):
            run(n)
        self.assertEqual(block_stats(compiled, "profile_runs"), after)

    def test_guard_on_a_value_computed_in_the_block(self):
        def f(x):
            y = x.t()
            if y.shape[0] > 2:
                return y.sum()
            return y * 2

        compiled = torch.rewrite(f, passes=narrowing(), fallback=False)
        for n in (3, 4, 1, 5, 2, 1, 7):
            x = torch.randn(2, n)
            self.assertEqual(compiled(x), f(x))
        self.assertIn("guard ", compiled.dump())
        self.assertGreaterEqual(block_stats(compiled, "guard_failures")[0], 1)

    def test_python_pass(self):
        pinned = []

        def pin_first_dim(region):
            changed = False
            for inst in region.instructions():
                if inst["op"] == "tensor_size" and region.speculate:
                    value = inst["uses"][0]
                    n = region.observed(value)["last"][0]
                    if region.guard_dim(value, 0, n, n):
                        pinned.append(n)
                        changed = True
            return changed

        def f(x):
            return x.sum() * x.shape[0]

        passes = list(canonical)
        passes.insert(passes.index("tensor_accessors") + 1, pin_first_dim)
        compiled = torch.rewrite(f, passes=passes, fallback=False)
        for n in (3, 3, 5, 3, 5):
            x = torch.ones(n, 2)
            self.assertEqual(compiled(x), f(x))
        self.assertEqual(pinned, [3, 5])
        text = compiled.dump()
        self.assertIn("3 <= size(%x.1, 0) <= 3", text)
        self.assertIn("5 <= size(%x.1, 0) <= 5", text)
        self.assertEqual(block_stats(compiled, "profile_runs"), [2])

    def test_pass_replaces_a_run_by_a_call(self):
        # The pass has the VM call `run` where the block had its
        # instructions; `run` gets the values the instructions took from
        # before them and gives the ones that are needed after them.
        made = []

        class Run:
            def __init__(self, replay):
                self.replay = replay
                self.inputs, self.outputs = replay.io(replay.start, replay.end)
                self.calls = 0

            def __call__(self, *args):
                self.calls += 1
                env = dict(zip(self.inputs, args))
                self.replay.run(self.replay.start, self.replay.end, env)
                return tuple(env[value] for value in self.outputs)

        def replace(region):
            insts = region.instructions()
            if any(inst["op"] == "call_object" for inst in insts):
                return False
            work = [i for i, inst in enumerate(insts) if inst["op"] == "binary"]
            if region.note is None:
                self.assertTrue(all(insts[i]["replayable"] for i in work))
                region.note = Run(region.replayer(work[0], work[-1] + 1))
                made.append(region.note)
            run = region.note
            return region.fuse(run.replay.start, run.replay.end, run)

        def f(x, y, k):
            a = x * y + k
            b = a - x
            return a, b * 2

        compiled = torch.rewrite(f, passes=[*WHOLE, replace], fallback=False)
        for k in range(6):
            x, y = torch.randn(3), torch.randn(3)
            self.assertEqual(compiled(x, y, k), f(x, y, k), atol=0, rtol=0)
        (run,) = made
        # The first call profiled the block; the others went through `run`.
        self.assertEqual(run.calls, 5)
        self.assertEqual(len(run.inputs), 3)
        # What is returned, and `b`, which the block keeps alive to its end
        # as the interpreter's frame would.
        self.assertEqual(len(run.outputs), 3)
        self.assertIn("call_object", compiled.dump())
        self.assertEqual(
            run.replay.name,
            "TestCfgPasses.test_pass_replaces_a_run_by_a_call.<locals>.f (block 0)",
        )
        # Four operators, the constant one of them takes, and the two
        # variables they bind.
        self.assertEqual(len(run.replay.text(run.replay.start, run.replay.end)), 7)

    def test_replayed_error_names_its_instruction(self):
        handles = []

        def grab(region):
            if not handles:
                handles.append(
                    (
                        region.replayer(0, len(region.instructions())),
                        region.instructions(),
                    )
                )
            return False

        def f(x, y):
            a = x + 1
            b = a @ y
            return b - 1

        compiled = torch.rewrite(f, passes=[*WHOLE, grab], fallback=False)
        compiled(torch.ones(2, 3), torch.ones(3, 2))
        ((replay, insts),) = handles
        binary = [i for i, inst in enumerate(insts) if inst["op"] == "binary"]
        lo, hi = 0, binary[-1] + 1
        inputs, outputs = replay.io(lo, hi)
        self.assertEqual(len(inputs), 2)
        env = dict(zip(inputs, (torch.ones(2, 3), torch.ones(3, 2))))
        replay.run(lo, hi, env)
        # The outputs are in the order they are defined: the result is last.
        self.assertEqual(env[outputs[-1]], f(torch.ones(2, 3), torch.ones(3, 2)))
        # The matrix product is the second of the three.
        env = dict(zip(inputs, (torch.ones(2, 3), torch.ones(4, 2))))
        with self.assertRaisesRegex(
            RuntimeError, "shapes cannot be multiplied"
        ) as caught:
            replay.run(lo, hi, env)
        self.assertEqual(caught.exception.cfg_index, binary[1])
        with self.assertRaisesRegex(IndexError, "outside the range"):
            replay.io(lo, hi + 100)

    def test_pass_takes_execution_elsewhere(self):
        # The pass puts a call in front of the block's instructions that can
        # do their work and say where to go on.
        calls = []

        def leaping(region):
            insts = region.instructions()
            if insts[0]["op"] == "leap":
                return False
            total = [inst for inst in insts if inst["op"] == "binary"][-1]["dst"]

            def hop(x, y):
                calls.append(1)
                if len(calls) % 2:
                    return None
                # Done up to the end of the block, with the sum in place.
                return (0, len(insts), {total: (x + y) * 2}, [])

            inputs = region.leap(0, hop)
            self.assertEqual(len(inputs), 2)
            return True

        def f(x, y):
            a = x + y
            return a * 2

        compiled = torch.rewrite(f, passes=[*WHOLE, leaping], fallback=False)
        for _ in range(7):
            x, y = torch.randn(3), torch.randn(3)
            self.assertEqual(compiled(x, y), f(x, y), atol=0, rtol=0)
        # The first call profiled the block; of the six others, three ran
        # the instructions and three did not.
        self.assertEqual(len(calls), 6)
        self.assertIn("leap", compiled.dump())

    def test_function_as_a_path_sees_it(self):
        handles = []

        def grab(region):
            if not handles:
                handles.append(region.replayer(0, len(region.instructions())))
            return False

        def f(xs, flag):
            total = 0
            for x in xs:
                total = total + x
            if flag:
                total = total * 2
            return total

        compiled = torch.rewrite(
            f, passes=["inline", "shortcuts", grab], fallback=False
        )
        self.assertEqual(compiled([1, 2, 3], True), 12)
        (replay,) = handles
        self.assertTrue(replay.leaps)
        self.assertIs(replay.globals, f.__globals__)
        # The first block ends in a jump to the head of the loop, which
        # leaves it when the iterator is exhausted.
        kind, value, targets = replay.terminator(0)
        self.assertEqual((kind, value, len(targets)), ("jump", None, 1))
        head = targets[0]
        kind, value, targets = replay.terminator(head)
        self.assertEqual((kind, len(targets)), ("br_null", 2))
        ops = [inst["op"] for inst in replay.instructions_of(head)]
        self.assertEqual(ops, ["iter_next"])
        self.assertTrue(replay.instructions_of(head)[0]["steppable"])
        self.assertFalse(replay.instructions_of(head)[0]["replayable"])
        # Into the head from the first block: the phi of `total` takes the
        # constant, and the loop variable has no value yet.
        moves, _ = replay.edge(0, head)
        self.assertIn(None, [source for _, source in moves])
        # Running the head on its own: the iterator gives its first item.
        env = dict.fromkeys(replay.live(head, 0), iter([7, 8]))
        replay.step(head, 0, 1, env)
        self.assertIn(7, env.values())
        # What is only kept alive is needed, but not read.
        last = len(replay.instructions_of(0))
        self.assertLessEqual(set(replay.live(0, last, True)), set(replay.live(0, last)))

    def test_settle_specializes_the_block_again(self):
        seen = []

        def note_taker(region):
            seen.append(region.note)
            if region.note is None:
                region.note = [region.replayer(0, len(region.instructions()))]
            return False

        def f(x):
            return x * 2

        compiled = torch.rewrite(f, passes=[note_taker, *canonical], fallback=False)
        for _ in range(3):
            self.assertEqual(compiled(torch.ones(2)), torch.full((2,), 2.0))
        (note,) = [n for n in seen if n is not None][:1]
        runs = len(seen)
        note[0].settle("again")
        # The pass ran on the block right away, and saw what was left for it.
        self.assertGreater(len(seen), runs)
        self.assertEqual(seen[-1], "again")
        self.assertEqual(compiled(torch.ones(2)), torch.full((2,), 2.0))

    def test_profiling_depth(self):
        depths = []

        def probe(x):
            depths.append(cfg_rs.profiling_depth())
            return x

        def inner(x):
            return probe(x) + 1

        def outer(x):
            return inner(x) * 2

        compiled = torch.rewrite(
            outer, fallback=False, passes=without("inline"), callees=True
        )
        for _ in range(3):
            compiled(torch.ones(2))
        # The first run of a block is its profiling run: on the first call
        # that is outer's, inner's and the one of probe itself.
        self.assertEqual(depths, [3, 0, 0])
        self.assertEqual(cfg_rs.profiling_depth(), 0)

    def test_region_is_only_valid_during_the_pass(self):
        kept = []

        def keep(region):
            kept.append(region)
            return False

        compiled = torch.rewrite(lambda x: x + 1, passes=[*WHOLE, keep], fallback=False)
        compiled(torch.ones(1))
        with self.assertRaisesRegex(RuntimeError, "only valid while its pass runs"):
            kept[0].instructions()

    def test_failing_pass_leaves_the_block_generic(self):
        def broken(region):
            raise KeyError("oops")

        def f(x):
            return x * 3

        compiled = torch.rewrite(f, passes=[*canonical, broken], fallback=False)
        for _ in range(3):
            self.assertEqual(compiled(torch.ones(2)), torch.full((2,), 3.0))
        stats = compiled.stats()
        self.assertEqual(block_stats(compiled, "versions"), [0])
        self.assertIn("pass failed", stats["blocks"][0]["rejected"])


SWAP_RMS = ["pow", "mean", "add", "rsqrt", "mul"]


def swap_rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w


def swap_twice(x, w):
    return swap_rms(x, w) + swap_rms(x + 1, w)


class SwapKernel:
    """Root-mean-square normalization, written another way."""

    def __init__(self, shift=0.0):
        self.calls = 0
        self.shift = shift
        self.took = None

    def __call__(self, *args):
        self.calls += 1
        self.took = [type(arg) for arg in args]
        (x,) = args
        return x / x.square().mean(-1, keepdim=True).add(1e-6).sqrt() + self.shift


@unittest.skipUnless(HAS_RUNTIME, "torch.rewrite runtime not built")
class TestCfgUserPasses(TestCase):
    """Passes of the user's: a kernel in place of operations, by a pass
    written in Python or in Rust."""

    hw_classification = HardwareClassification.GENERIC
    _library = None

    def run_swapped(self, fn, passes, calls=12):
        compiled = torch.rewrite(fn, passes=[*passes, *canonical], fallback=False)
        x, w = torch.randn(4, 8), torch.randn(8)
        for _ in range(calls):
            self.assertEqual(compiled(x, w), fn(x, w))
        return compiled, x, w

    def test_instructions_name_the_kernel_they_call(self):
        seen = []

        def look(region):
            seen.append([i["kernel"] for i in region.instructions() if i["kernel"]])
            return False

        compiled = torch.rewrite(swap_rms, passes=[look, *canonical], fallback=False)
        for _ in range(4):
            compiled(torch.randn(4, 8), torch.randn(8))
        self.assertEqual(
            max(seen, key=len),
            [
                "aten::pow.Tensor_Scalar",
                "aten::mean.dim",
                "aten::add.Tensor",
                "aten::rsqrt",
                "aten::mul.Tensor",
                "aten::mul.Tensor",
            ],
        )

    def test_swap_puts_a_kernel_in_place_of_a_run_of_operations(self):
        kernel = SwapKernel()
        swap = torch.rewrite.swap(SWAP_RMS, kernel, check=2)
        compiled, x, w = self.run_swapped(swap_rms, [swap])
        self.assertGreater(kernel.calls, 2)
        # It is handed what the run takes from the function: the tensor, and
        # neither torch.rsqrt nor the constants.
        self.assertEqual(kernel.took, [torch.Tensor])
        # What it gives is what the function goes on with, once it is no
        # longer checked.
        kernel.shift = 1.0
        self.assertEqual(compiled(x, w), swap_rms(x, w) + w)

    def test_swap_finds_every_place(self):
        kernel = SwapKernel()
        swap = torch.rewrite.swap(SWAP_RMS, kernel)
        compiled, x, w = self.run_swapped(swap_twice, [swap])
        before = kernel.calls
        compiled(x, w)
        self.assertEqual(kernel.calls - before, 2)

    def test_swap_checks_the_kernel_against_the_operations(self):
        swap = torch.rewrite.swap(SWAP_RMS, SwapKernel(shift=1e-2))
        with self.assertRaisesRegex(torch.rewrite.Mismatch, "differs by up to 0.01"):
            self.run_swapped(swap_rms, [swap])
        unchecked = torch.rewrite.swap(SWAP_RMS, SwapKernel(shift=1e-2), check=False)
        compiled = torch.rewrite(swap_rms, passes=[unchecked, *canonical])
        x, w = torch.randn(4, 8), torch.randn(8)
        for _ in range(6):
            out = compiled(x, w)
        self.assertEqual(out, swap_rms(x, w) + 1e-2 * w)

    def test_swap_leaves_other_operations_alone(self):
        kernel = SwapKernel()
        swap = torch.rewrite.swap(["pow", "mean", "sub"], kernel)
        self.run_swapped(swap_rms, [swap])
        self.assertEqual(kernel.calls, 0)

    def library(self):
        """The passes of test/rewrite_passes, built with cargo."""
        cls = type(self)
        if cls._library is None:
            cargo = shutil.which("cargo")
            if cargo is None:
                self.skipTest("cargo is not installed")
            here = os.path.dirname(os.path.abspath(__file__))
            manifest = os.path.join(here, "rewrite_passes", "Cargo.toml")
            target = tempfile.mkdtemp(prefix="rewrite_passes_")
            command = [cargo, "build", "--release", "--offline", "--locked"]
            command += ["--manifest-path", manifest, "--target-dir", target]
            subprocess.run(command, check=True, capture_output=True)
            (cls._library,) = (
                os.path.join(target, "release", name)
                for name in os.listdir(os.path.join(target, "release"))
                if name.startswith("librewrite_passes.") and not name.endswith(".d")
            )
        return torch.rewrite.load_passes(cls._library)

    def test_a_pass_written_in_rust_puts_a_kernel_in(self):
        passes = self.library()
        self.assertEqual(sorted(passes), ["fuse_rmsnorm", "strict"])
        kernel = SwapKernel()
        fuse = passes["fuse_rmsnorm"].bind(kernel)
        self.assertEqual(fuse.__name__, "fuse_rmsnorm")
        compiled, x, w = self.run_swapped(swap_rms, [fuse])
        self.assertGreater(kernel.calls, 4)
        self.assertEqual(kernel.took, [torch.Tensor])
        kernel.shift = 1.0
        self.assertEqual(compiled(x, w), swap_rms(x, w) + w)
        # Bound to a kernel that is wrong, it is found out as swap finds it.
        wrong = passes["fuse_rmsnorm"].bind(SwapKernel(shift=1e-2))
        with self.assertRaisesRegex(torch.rewrite.Mismatch, "differs by up to 0.01"):
            self.run_swapped(swap_rms, [wrong])

    def test_a_pass_written_in_rust_can_fail(self):
        passes = self.library()

        def rejected(compiled):
            return [b["rejected"] for b in compiled.stats()["blocks"]]

        # A pass that fails leaves its block as the function wrote it.
        compiled, _, _ = self.run_swapped(swap_rms, [passes["strict"]])
        self.assertIn(
            "a block that multiplies: aten::pow.Tensor_Scalar", str(rejected(compiled))
        )
        unbound, _, _ = self.run_swapped(swap_rms, [passes["fuse_rmsnorm"]])
        self.assertIn("needs a kernel", str(rejected(unbound)))

        def adds(x, w):
            return x + w

        compiled, _, _ = self.run_swapped(adds, [passes["strict"]])
        self.assertEqual(rejected(compiled), [None])
        self.assertEqual(block_stats(compiled, "versions"), [1])

    def test_a_library_that_is_none_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "cannot load"):
            torch.rewrite.load_passes("/nowhere/libpasses.so")
        with self.assertRaisesRegex(RuntimeError, "exports no passes"):
            torch.rewrite.load_passes(torch._C.__file__)


def inl_scale(x, k):
    return x * k


def inl_shifted(x, y):
    t = inl_scale(x, 2)
    if t.sum() > 0:
        return t + y
    return t - y


def inl_outer(x, y):
    a = inl_shifted(x, y)
    return a + inl_shifted(a, y)


def inl_fact(n):
    return 1 if n <= 1 else n * inl_fact(n - 1)


def inl_defaults(x, k=3):
    return x * k


def inl_uses_defaults(x):
    return inl_defaults(x) + inl_defaults(x, k=2)


def inl_raises(x):
    return x + inl_missing_name  # noqa: F821


def inl_calls_raises(x):
    y = x * 2
    return inl_raises(y)


def unr_loop(x, y):
    total = x
    for i in range(3):
        total = total * 2 + y
        if i == 1:
            total = total - 1
    return total, i


@unittest.skipUnless(HAS_RUNTIME, "torch.rewrite runtime not built")
class TestCfgGraphTransforms(TestCase):
    """Inlining and loop unrolling, which rewrite the graph before it runs."""

    hw_classification = HardwareClassification.GENERIC

    def test_inlines_calls_of_module_functions(self):
        compiled = torch.rewrite(inl_outer, fallback=False)
        x, y = torch.randn(3), torch.randn(3)
        for sign in (1, -1, 1):
            self.assertEqual(compiled(sign * x, y), inl_outer(sign * x, y))
        stats = compiled.stats()
        # Both calls of inl_shifted, and the call of inl_scale inside each.
        self.assertEqual(
            sorted(stats["inlined"]),
            ["inl_scale", "inl_scale", "inl_shifted", "inl_shifted"],
        )
        self.assertIn("is_function", compiled.dump())
        # The inlined bodies are specialized like the caller's own code.
        self.assertGreater(stats["sites"], 8)
        self.assertEqual(stats["site_misses"], 0)

    def test_inlined_call_is_guarded_on_the_callee(self):
        compiled = torch.rewrite(inl_outer, fallback=False)
        x, y = torch.randn(3), torch.randn(3)
        self.assertEqual(compiled(x, y), inl_outer(x, y))
        original = inl_scale
        try:
            globals()["inl_scale"] = lambda x, k: x * k + 1
            self.assertEqual(compiled(x, y), inl_outer(x, y))
        finally:
            globals()["inl_scale"] = original
        code = inl_scale.__code__
        try:
            inl_scale.__code__ = (lambda x, k: x * k - 5).__code__
            self.assertEqual(compiled(x, y), inl_outer(x, y))
        finally:
            inl_scale.__code__ = code
        self.assertEqual(compiled(x, y), inl_outer(x, y))

    def test_what_is_not_inlined(self):
        recursive = torch.rewrite(inl_fact, fallback=False)
        self.assertEqual(recursive(5), 120)
        self.assertEqual(recursive.stats()["inlined"], [])
        # Defaults and keywords stay calls.
        defaults = torch.rewrite(inl_uses_defaults, fallback=False)
        self.assertEqual(defaults(torch.ones(2)), torch.full((2,), 5.0))
        self.assertEqual(defaults.stats()["inlined"], [])
        off = torch.rewrite(inl_outer, fallback=False, passes=without("inline"))
        x, y = torch.randn(3), torch.randn(3)
        self.assertEqual(off(x, y), inl_outer(x, y))
        self.assertEqual(off.stats()["inlined"], [])

    def test_unrolls_constant_range_loops(self):
        compiled = torch.rewrite(unr_loop, fallback=False)
        x, y = torch.randn(3), torch.randn(3)
        for _ in range(3):
            total, i = compiled(x, y)
            expected, expected_i = unr_loop(x, y)
            self.assertEqual(total, expected)
            self.assertEqual(i, expected_i)
        self.assertEqual(compiled.stats()["unrolled_loops"], 1)
        off = torch.rewrite(unr_loop, fallback=False, passes=without("unroll"))
        self.assertEqual(off(x, y)[0], unr_loop(x, y)[0])
        self.assertEqual(off.stats()["unrolled_loops"], 0)

    def test_unrolled_loop_is_guarded_on_range(self):
        compiled = torch.rewrite(unr_loop, fallback=False)
        x, y = torch.randn(3), torch.randn(3)
        self.assertEqual(compiled(x, y)[1], 2)
        try:
            globals()["range"] = lambda k: (0, 1)
            total, i = compiled(x, y)
            self.assertEqual(total, unr_loop(x, y)[0])
            self.assertEqual(i, 1)
        finally:
            del globals()["range"]
        self.assertEqual(compiled(x, y)[1], 2)

    def test_unrolling_nested_and_empty_loops(self):
        def nested(x):
            for i in range(2):
                for j in range(3):
                    x = x + i * j
            else:
                x = x * 2
            return x

        def empty(x):
            for i in range(0):
                x = x + 1
            return x, i

        x = torch.randn(3)
        compiled = torch.rewrite(nested, fallback=False)
        self.assertEqual(compiled(x), nested(x))
        self.assertGreaterEqual(compiled.stats()["unrolled_loops"], 3)
        compiled = torch.rewrite(empty, fallback=False)
        self.assertEqual(compiled.stats()["unrolled_loops"], 1)
        with self.assertRaisesRegex(UnboundLocalError, "'i'"):
            compiled(x)

    def test_loops_that_are_not_unrolled(self):
        def breaks(x):
            for i in range(4):
                x = x + 1
                if i == 2:
                    break
            return x

        def long(x):
            for _ in range(100):
                x = x + 1
            return x

        def not_constant(x, n):
            for _ in range(n):
                x = x + 1
            return x

        x = torch.zeros(2)
        for fn, args in ((breaks, (x,)), (long, (x,)), (not_constant, (x, 3))):
            compiled = torch.rewrite(fn, fallback=False)
            self.assertEqual(compiled(*args), fn(*args))
            self.assertEqual(compiled.stats()["unrolled_loops"], 0)

    def test_error_in_inlined_function_has_both_frames(self):
        compiled = torch.rewrite(inl_calls_raises, fallback=False)
        self.assertEqual(compiled.stats()["inlined"], ["inl_raises"])
        try:
            compiled(torch.ones(2))
        except NameError:
            frames = traceback.extract_tb(sys.exc_info()[2])
        else:
            self.fail("expected NameError")
        self.assertEqual(
            [(f.name, f.line) for f in frames[-2:]],
            [
                ("inl_calls_raises", "return inl_raises(y)"),
                ("inl_raises", "return x + inl_missing_name  # noqa: F821"),
            ],
        )


class CalleeScale:
    def __init__(self, k):
        self.k = k

    def __call__(self, x):
        return x * self.k

    def twice(self, x, bias=0):
        return self(x) * 2 + bias


def callee_fact(n):
    return 1 if n <= 1 else n * callee_fact(n - 1)


def callee_raises(x):
    return x + callee_missing_name  # noqa: F821


@unittest.skipUnless(HAS_RUNTIME, "torch.rewrite runtime not built")
class TestCfgCallees(TestCase):
    """Python functions called from compiled code are compiled in turn."""

    hw_classification = HardwareClassification.GENERIC

    def callees(self, compiled):
        stats = compiled.stats()
        return stats["compiled_callees"], stats["interpreted_callees"]

    def test_functions_of_other_modules_are_compiled(self):
        def f(x):
            return torch.nn.functional.dropout(x, p=0.0, training=False) + 1

        compiled = torch.rewrite(f, fallback=False)
        x = torch.randn(3)
        for _ in range(3):
            self.assertEqual(compiled(x), f(x))
        # And the function behind an attribute of torch._VF, which C code calls.
        self.assertEqual(
            self.callees(compiled), (["VFModule.__getattr__", "dropout"], [])
        )

    def test_methods_and_callable_objects(self):
        def f(obj, x):
            bound = obj.twice
            return obj.twice(x) + bound(x, bias=1) + obj(x)

        compiled = torch.rewrite(f, fallback=False)
        obj, x = CalleeScale(3), torch.randn(3)
        for _ in range(3):
            self.assertEqual(compiled(obj, x), f(obj, x))
        self.assertEqual(
            self.callees(compiled), (["CalleeScale.__call__", "CalleeScale.twice"], [])
        )

    def test_star_args_calls(self):
        def target(a, b=2, *rest, scale=1):
            return (a + b + len(rest)) * scale

        def f(args, kwargs):
            return target(*args, **kwargs) + target(*args)

        compiled = torch.rewrite(f, fallback=False)
        for args, kwargs in (((1,), {}), ((1, 5, 7, 8), {"scale": 3})):
            self.assertEqual(compiled(args, kwargs), f(args, kwargs))
        self.assertEqual(len(self.callees(compiled)[0]), 1)

    def test_unsupported_callee_runs_in_the_interpreter(self):
        def generator(n):
            yield from range(n)

        def f(n):
            return sum(generator(n))

        compiled = torch.rewrite(f, fallback=False)
        self.assertEqual([compiled(4), compiled(5)], [6, 10])
        compiled_names, interpreted = self.callees(compiled)
        self.assertEqual(compiled_names, [])
        self.assertEqual(len(interpreted), 1)

    def test_recursion_and_modules(self):
        compiled = torch.rewrite(callee_fact, fallback=False, passes=without("inline"))
        self.assertEqual(compiled(6), 720)
        self.assertEqual(self.callees(compiled)[0], ["callee_fact"])

        def run(module, x):
            return module.forward(x) + module(x)

        linear = torch.nn.Linear(3, 2)
        compiled = torch.rewrite(run, fallback=False)
        x = torch.randn(4, 3)
        for _ in range(3):
            self.assertEqual(compiled(linear, x), run(linear, x))
        self.assertIn("Linear.forward", self.callees(compiled)[0])

    def test_changed_function_is_compiled_again(self):
        def target(x, k=2):
            return x * k

        def f(x):
            return target(x)

        compiled = torch.rewrite(f, fallback=False)
        self.assertEqual(compiled(3), 6)
        target.__defaults__ = (5,)
        self.assertEqual(compiled(3), 15)
        target.__code__ = (lambda x, k=2: x - k).__code__
        self.assertEqual(compiled(3), -2)

    def test_functions_made_per_call_share_one_compilation(self):
        def make():
            return lambda v: v + 1

        def f(n):
            total = 0
            for i in range(n):
                total += make()(i)
            return total

        compiled = torch.rewrite(f, fallback=False)
        self.assertEqual(compiled(40), f(40))
        # make, and the lambda once for the forty functions made from it.
        names, interpreted = self.callees(compiled)
        self.assertEqual(
            [name.rsplit(".", 1)[-1] for name in names], ["make", "<lambda>"]
        )
        self.assertEqual(interpreted, [])

    def test_functions_of_one_code_keep_their_closure_and_defaults(self):
        def make(k):
            def add(x, y=k * 10, *, z=k * 100):
                return x + y + z + k

            return add

        def f(g, x):
            return g(x)

        compiled = torch.rewrite(f, fallback=False)
        adders = [make(k) for k in range(12)]
        self.assertEqual([compiled(g, 1) for g in adders], [g(1) for g in adders])
        self.assertEqual(len(self.callees(compiled)[0]), 1)

    def test_what_a_closure_holds_dies_with_it(self):
        class Counted:
            alive = 0

            def __init__(self):
                Counted.alive += 1

            def __del__(self):
                Counted.alive -= 1

        def outer():
            held = Counted()

            def inner():
                return held

            return inner() is held

        def f(n):
            return [outer() for _ in range(n)], Counted.alive

        compiled = torch.rewrite(f, fallback=False)
        self.assertEqual(compiled(20), ([True] * 20, 0))
        self.assertEqual(
            [name.rsplit(".", 1)[-1] for name in self.callees(compiled)[0]],
            ["__del__", "__init__", "outer", "inner"],
        )

    def test_infinite_recursion(self):
        def forever():
            return forever()

        def guarded():
            try:
                return guarded()
            except ValueError:
                return -1

        for fn in (forever, guarded):
            compiled = torch.rewrite(fn, fallback=False)
            with self.assertRaisesRegex(
                RecursionError, "maximum recursion depth exceeded"
            ):
                compiled()

    def test_recursion_limit_means_the_same_depth(self):
        def count(n):
            try:
                return count(n + 1)
            except RecursionError:
                return n

        # A compiled call uses up the limit as an interpreted one does: both
        # get as deep, give or take the frames of the wrappers.
        compiled = torch.rewrite(count, fallback=False)
        self.assertLess(abs(compiled(0) - count(0)), 5)

    def test_recursion_deeper_than_the_native_stack(self):
        def depth(n):
            return 0 if n == 0 else 1 + depth(n - 1)

        compiled = torch.rewrite(depth, fallback=False)
        limit = sys.getrecursionlimit()
        sys.setrecursionlimit(1_000_000)
        try:
            self.assertEqual(compiled(500), 500)
            with self.assertRaisesRegex(
                RecursionError, "maximum recursion depth exceeded"
            ):
                compiled(200_000)
        finally:
            sys.setrecursionlimit(limit)
        self.assertEqual(compiled(100), 100)

    def test_bad_arguments_raise_what_the_interpreter_raises(self):
        def sig(a, b, /, c, d=4, *, e, f=6):
            return a, b, c, d, e, f

        def call(args, kwargs):
            return sig(*args, **kwargs)

        calls = (
            ((1, 2, 3), {"e": 5}),
            ((1, 2), {}),
            ((), {}),
            ((1, 2, 3, 4, 5), {"e": 1}),
            ((1, 2), {"c": 3, "e": 5, "g": 1}),
            ((1,), {"b": 2, "c": 3, "e": 5}),
            ((1, 2, 3), {}),
            ((1, 2, 3), {"e": 5, "c": 4}),
        )
        compiled = torch.rewrite(sig, fallback=False)
        compiled_caller = torch.rewrite(call, fallback=False)
        for args, kwargs in calls:
            expected = call_outcome(sig, *args, **kwargs)
            self.assertEqual(call_outcome(compiled, *args, **kwargs), expected)
            self.assertEqual(call_outcome(compiled_caller, args, kwargs), expected)
        message = "sig() missing 3 required positional arguments: 'a', 'b', and 'c'"
        self.assertTrue(call_outcome(compiled)[1].endswith(message))

    def test_bad_star_arguments_raise_what_the_interpreter_raises(self):
        def target(*args, **kwargs):
            return args, kwargs

        def star(x):
            return target(*x)

        def double_star(x):
            return target(**x)

        def both(a, k):
            return target(1, *a, k=2, **k)

        def in_a_list(x):
            return [1, *x]

        def in_a_dict(x):
            return {"a": 1, **x}

        cases = (
            (star, ([1, 2],), (5,), (None,)),
            (double_star, ({"a": 1},), ([1],), (None,)),
            (both, ([1], {"z": 3}), ([1], {"k": 5}), (3, {}), ([1], 4)),
            (in_a_list, ([2],), (5,)),
            (in_a_dict, ({"b": 2},), ([1],), (5,)),
        )
        for fn, *argsets in cases:
            compiled = torch.rewrite(fn, fallback=False)
            for args in argsets:
                self.assertEqual(call_outcome(compiled, *args), call_outcome(fn, *args))
        message = call_outcome(torch.rewrite(both, fallback=False), [1], {"k": 5})[1]
        self.assertTrue(
            message.endswith("target() got multiple values for keyword argument 'k'")
        )

    def test_callees_off(self):
        def f(obj, x):
            return obj.twice(x)

        compiled = torch.rewrite(f, fallback=False, callees=False)
        self.assertEqual(compiled(CalleeScale(2), 3), 12)
        self.assertEqual(self.callees(compiled), ([], []))

    def test_error_in_callee_has_both_frames(self):
        def f(x):
            return callee_raises(x)

        compiled = torch.rewrite(f, fallback=False, passes=without("inline"))
        try:
            compiled(torch.ones(2))
        except NameError:
            frames = traceback.extract_tb(sys.exc_info()[2])
        else:
            self.fail("expected NameError")
        self.assertEqual([frame.name for frame in frames[-2:]], ["f", "callee_raises"])


def hook_bottom(x):
    return x * 2 + 1


class HookPoint:
    def __init__(self, x):
        self.x = hook_bottom(x)

    @property
    def doubled(self):
        return hook_bottom(self.x)

    def __add__(self, other):
        return hook_bottom(self.x + other)

    def __getitem__(self, i):
        return hook_bottom(self.x[i])


def hook_constructs(x):
    return HookPoint(x).x


def hook_reads_a_property(x):
    return HookPoint(x).doubled


def hook_adds(x):
    return HookPoint(x) + 1


def hook_subscripts(x):
    return HookPoint(x)[0]


def hook_key(x):
    return -hook_bottom(x)


def hook_sorts(x):
    return sorted([x[1], x[0], x[2]], key=hook_key)


def hook_partial(x):
    return functools.partial(hook_bottom, x)()


def hook_maps(x):
    return list(map(hook_bottom, [x, x + 1]))


def hook_generator(xs):
    for x in xs:
        yield hook_bottom(x)


def hook_drives_a_generator(x):
    return list(hook_generator([x, x + 1]))


def hook_keywords(a, b=2, *rest, scale, shift=0, **more):
    return (a + b + sum(rest)) * scale + shift + sum(more.values())


def hook_binds_arguments(x):
    call = functools.partial(hook_keywords, x, scale=3, extra=x)
    return call(), call(5), call(5, 6, 7, shift=1, other=2)


def hook_cell(x):
    return lambda y: x + y


def hook_makes_closures(x):
    return [add(x) for add in map(hook_cell, [x, x + 1])]


class HookFragile:
    @property
    def broken(self):
        return hook_bottom(None)

    @property
    def endless(self):
        return self.endless


def hook_reads_broken(obj):
    return obj.broken


def hook_reads_endless(obj):
    return obj.endless


def hook_offered(*ignored):
    return cfg_rs.frames_offered()


def hook_asks(x):
    return hook_offered(), next(map(hook_offered, [x]))


def hook_late(x):
    return x + 1


def hook_slow_python(n, late):
    total = 0
    for i in range(n):
        total += i * i % 7
    if late:
        # A Python function that C code calls.
        total += next(map(hook_late, [0]))
    return total


def hook_calls_slow_python(n, x, late):
    return hook_slow_python(n, late), x + 1


@unittest.skipUnless(HAS_RUNTIME, "torch.rewrite runtime not built")
class TestCfgFrameHook(TestCase):
    """Python functions that C code calls under a compiled function, or that
    a function the interpreter runs calls, are compiled in turn."""

    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "fn,through",
        [
            (hook_constructs, "HookPoint.__init__"),
            (hook_reads_a_property, "HookPoint.doubled"),
            (hook_adds, "HookPoint.__add__"),
            (hook_subscripts, "HookPoint.__getitem__"),
            (hook_sorts, "hook_key"),
            (hook_partial, "hook_bottom"),
            (hook_maps, "hook_bottom"),
        ],
        name_fn=lambda fn, through: fn.__name__,
    )
    def test_what_c_code_calls_is_compiled(self, fn, through):
        compiled = torch.rewrite(fn, fallback=False)
        x = torch.arange(3.0)
        for _ in range(4):
            self.assertEqual(compiled(x), fn(x))
        self.assertIn(through, compiled.stats()["compiled_callees"])
        self.assertEqual(compiled.stats()["interpreted_callees"], [])

    def test_what_an_interpreted_function_calls_is_compiled(self):
        compiled = torch.rewrite(hook_drives_a_generator, fallback=False)
        x = torch.arange(3.0)
        for _ in range(4):
            self.assertEqual(compiled(x), hook_drives_a_generator(x))
        stats = compiled.stats()
        self.assertEqual(stats["interpreted_callees"], ["hook_generator"])
        self.assertIn("hook_bottom", stats["compiled_callees"])

    def test_arguments_are_bound_as_the_interpreter_binds_them(self):
        compiled = torch.rewrite(hook_binds_arguments, fallback=False)
        for _ in range(4):
            self.assertEqual(compiled(2), hook_binds_arguments(2))
        self.assertIn("hook_keywords", compiled.stats()["compiled_callees"])
        closures = torch.rewrite(hook_makes_closures, fallback=False)
        for _ in range(4):
            self.assertEqual(closures(2), hook_makes_closures(2))
        self.assertIn("hook_cell", closures.stats()["compiled_callees"])

    def test_an_exception_passes_through_the_frames(self):
        compiled = torch.rewrite(hook_reads_broken, fallback=False)

        def frames(fn):
            try:
                fn(HookFragile())
            except Exception as e:
                names = [f.name for f in traceback.extract_tb(e.__traceback__)]
                return type(e), names[:4]

        for _ in range(3):
            self.assertEqual(frames(compiled), frames(hook_reads_broken))
        self.assertIn("HookFragile.broken", compiled.stats()["compiled_callees"])
        endless = torch.rewrite(hook_reads_endless, fallback=False)
        self.assertEqual(frames(endless)[0], RecursionError)
        self.assertEqual(frames(compiled), frames(hook_reads_broken))

    def test_frames_are_offered_only_under_the_function(self):
        compiled = torch.rewrite(hook_asks, fallback=False)
        self.assertFalse(cfg_rs.frames_offered())
        # While a block is profiled, what its calls of C functions call is
        # the interpreter's.
        for _ in range(4):
            compiled(1)
        for _ in range(3):
            self.assertEqual(compiled(1), (True, True))
            self.assertFalse(cfg_rs.frames_offered())
        self.assertEqual(hook_asks(1), (False, False))
        alone = torch.rewrite(hook_asks, fallback=False, callees=False)
        for _ in range(6):
            self.assertEqual(alone(1), (False, False))

    def test_what_the_runtime_calls_is_not_offered(self):
        seen = []

        def look(region):
            seen.append(cfg_rs.frames_offered())
            return False

        def f(x):
            import colorsys  # noqa: F401

            return x * 2 + 1

        compiled = torch.rewrite(f, passes=[look, *canonical], fallback=False)
        with unittest.mock.patch("builtins.__import__", wraps=__import__) as imports:
            imports.side_effect = (
                lambda *a, **k: seen.append(cfg_rs.frames_offered())
                or unittest.mock.DEFAULT
            )
            for _ in range(3):
                compiled(torch.ones(2))
        self.assertGreater(len(seen), 3)
        self.assertEqual(set(seen), {False})

    def test_what_is_left_to_the_interpreter_is_left_with_what_it_calls(self):
        compiled = torch.rewrite(
            hook_calls_slow_python,
            fallback=False,
            demote=True,
            passes=without("inline"),
        )
        x = torch.ones(2)
        for _ in range(200):
            compiled(300, x, False)
        self.assertEqual(compiled.stats()["demoted_callees"], ["hook_slow_python"])
        # Now that the interpreter runs it, what C code calls under it is the
        # interpreter's too.
        for _ in range(4):
            self.assertEqual(compiled(300, x, True)[0], hook_slow_python(300, True))
        self.assertNotIn("hook_late", compiled.stats()["compiled_callees"])

    def test_the_compilers_are_left_to_the_interpreter(self):
        torch.manual_seed(0)
        model = RegionNet().train()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        passes = with_graphs("aot_eager")
        step = torch.rewrite(region_step, passes=passes, fallback=False)
        for _ in range(6):
            step(model, opt, torch.randn(4, 3, 6, 6), torch.randint(0, 5, (4,)))
        callees = step.stats()["compiled_callees"]
        # Autograd calls the backward of the graph, which is not the
        # function's code: neither its wrappers nor what made them.
        wrappers = [c for c in callees if "CompiledFunction" in c or "AOT" in c]
        self.assertEqual(wrappers, [])
        self.assertIn("SGD.step", callees)

    def test_a_function_dynamo_compiled_is_dynamo_s(self):
        graphs = []

        def backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        def g(x):
            return HookPoint(x).doubled

        dynamo = torch.compile(g, backend=backend)

        def f(x):
            return dynamo(x) + hook_reads_a_property(x)

        compiled = torch.rewrite(f, fallback=False)
        x = torch.arange(3.0)
        expected = g(x) + hook_reads_a_property(x)
        for _ in range(4):
            self.assertEqual(compiled(x), expected)
            self.assertFalse(cfg_rs.frames_offered())
        # Dynamo traced it, as it does when the interpreter makes the call.
        self.assertGreater(len(graphs), 0)
        callees = compiled.stats()["compiled_callees"]
        self.assertIn("HookPoint.doubled", callees)
        self.assertEqual([c for c in callees if c.endswith(".g")], [])


instantiate_parametrized_tests(TestCfgFrameHook)


@unittest.skipUnless(HAS_RUNTIME, "torch.rewrite runtime not built")
class TestCfgClosures(TestCase):
    """Functions that create functions: cells, nonlocal, defaults."""

    hw_classification = HardwareClassification.GENERIC

    def test_inner_function_shares_a_cell(self):
        def counter(start):
            count = start

            def bump(by=1, *, twice=False):
                nonlocal count
                count += by * (2 if twice else 1)
                return count

            return bump

        compiled = torch.rewrite(counter, fallback=False)
        made, expected = compiled(10), counter(10)
        self.assertEqual(
            [made(), made(5), made(twice=True)],
            [expected(), expected(5), expected(twice=True)],
        )
        self.assertEqual(made.__qualname__, expected.__qualname__)
        self.assertEqual(made.__defaults__, (1,))
        self.assertEqual(made.__kwdefaults__, {"twice": False})
        self.assertEqual(len(made.__closure__), 1)

    def test_argument_captured_by_an_inner_function(self):
        def f(x, y):
            def inner():
                return x + y

            x = x * 2
            return inner()

        compiled = torch.rewrite(f, fallback=False)
        x, y = torch.randn(3), torch.randn(3)
        self.assertEqual(compiled(x, y), f(x, y))

    def test_lambdas_with_defaults_and_annotations(self):
        def adders(ks):
            return [(lambda v, k=k: v + k) for k in ks]

        def annotated():
            def inner(a: int, b: "str" = "z") -> float:
                return 1.0

            return inner

        compiled = torch.rewrite(adders, fallback=False)
        self.assertEqual([add(10) for add in compiled([1, 2, 3])], [11, 12, 13])
        compiled = torch.rewrite(annotated, fallback=False)
        self.assertEqual(compiled().__annotations__, annotated().__annotations__)
        self.assertEqual(compiled().__defaults__, ("z",))

    def test_compiled_inner_function_writes_its_closure(self):
        def outer():
            total = 0

            def add(v):
                nonlocal total
                total += v
                return total

            return add

        add = outer()
        compiled = torch.rewrite(add, fallback=False)
        self.assertEqual([compiled(1), compiled(2), add(3)], [1, 3, 6])

    def test_empty_cells_raise_like_python(self):
        def local_cell():
            def inner():
                return value

            del inner
            return value  # noqa: F821
            value = 1

        def free_variable():
            def inner():
                return late

            return inner()
            late = 1

        for fn, error in ((local_cell, UnboundLocalError), (free_variable, NameError)):
            compiled = torch.rewrite(fn, fallback=False)
            with self.assertRaises(error) as expected:
                fn()
            with self.assertRaises(error) as got:
                compiled()
            self.assertEqual(str(got.exception), str(expected.exception))


def exc_double(x):
    if x < 0:
        raise ValueError("negative")
    return x * 2


@unittest.skipUnless(HAS_RUNTIME, "torch.rewrite runtime not built")
class TestCfgExceptions(TestCase):
    """try, except, finally and raise inside compiled functions."""

    hw_classification = HardwareClassification.GENERIC

    def outcome(self, fn, *args):
        args = [list(a) if isinstance(a, list) else a for a in args]
        try:
            result = ("returned", fn(*args))
        except Exception as e:
            result = ("raised", type(e).__name__, str(e), type(e.__cause__).__name__)
        return result, [a for a in args if isinstance(a, list)]

    def assertSameOutcome(self, fn, *argsets, **options):
        compiled = torch.rewrite(fn, fallback=False, **options)
        for args in argsets * 2:
            self.assertEqual(self.outcome(compiled, *args), self.outcome(fn, *args))
        return compiled

    def test_except_clause(self):
        def f(x):
            try:
                y = exc_double(x)
            except ValueError as e:
                y = len(str(e))
            return y

        compiled = self.assertSameOutcome(f, (3,), (-3,))
        self.assertIn("caught_exception", compiled.dump())

    def test_else_and_finally(self):
        def f(x, log):
            try:
                y = exc_double(x)
                log.append("body")
            except ValueError:
                y = -1
                log.append("except")
            else:
                log.append("else")
            finally:
                log.append("finally")
            return y

        self.assertSameOutcome(f, (3, []), (-3, []))

    def test_nested_handlers_and_return_through_finally(self):
        def f(x):
            try:
                try:
                    return exc_double(x)
                except KeyError:
                    return "inner"
                finally:
                    x = 100
            except ValueError as e:
                return ("outer", str(e), x)

        self.assertSameOutcome(f, (3,), (-3,))

    def test_raise_in_handlers(self):
        def bare(x):
            try:
                return exc_double(x)
            except ValueError:  # noqa: TRY203
                raise

        def chained(x):
            try:
                return exc_double(x)
            except ValueError as e:
                raise RuntimeError("wrapped") from e

        def no_active_exception():
            raise  # noqa: PLE0704

        self.assertSameOutcome(bare, (3,), (-3,))
        self.assertSameOutcome(chained, (3,), (-3,))
        self.assertSameOutcome(no_active_exception, ())

    def test_what_an_except_clause_matches(self):
        def several(x):
            try:
                return exc_double(x)
            except (KeyError, ValueError):
                return "caught"

        def other(x):
            try:
                return exc_double(x)
            except KeyError:
                return "caught"

        def not_a_class(x):
            try:
                return exc_double(x)
            except 3:  # noqa: B030
                return "caught"

        for fn in (several, other, not_a_class):
            self.assertSameOutcome(fn, (3,), (-3,))

    def test_try_in_a_loop(self):
        def f(xs):
            total = 0
            for x in xs:
                try:
                    total += exc_double(x)
                except ValueError:
                    total -= 1
                    continue
            return total

        self.assertSameOutcome(f, ([1, -2, 3, -4],))

    def test_handler_sees_the_locals_at_the_raise(self):
        def f(x):
            y = "before"
            try:
                y = "inside"
                exc_double(x)
                y = "after"
            except ValueError:
                return y
            return y

        def unbound(x):
            try:
                y = exc_double(x)
            except ValueError:
                pass
            return y

        self.assertSameOutcome(f, (3,), (-3,))
        self.assertSameOutcome(unbound, (3,), (-3,))

    def test_tensor_code_in_a_try_block_is_specialized(self):
        def f(x, y):
            try:
                z = x * 2 + y
                return z / y
            except RuntimeError:
                return x

        compiled = torch.rewrite(f, fallback=False)
        x, y = torch.randn(3), torch.randn(3)
        for _ in range(3):
            self.assertEqual(compiled(x, y), f(x, y))
        self.assertEqual(compiled(x, torch.randn(4)), x)
        self.assertGreaterEqual(compiled.stats()["sites"], 3)

    def test_traceback_through_a_handler(self):
        def f(x):
            try:
                return exc_double(x)
            except ValueError:  # noqa: TRY203
                raise

        compiled = torch.rewrite(
            f, fallback=False, passes=without("inline"), callees=False
        )
        try:
            compiled(-1)
        except ValueError:
            frames = traceback.extract_tb(sys.exc_info()[2])
        else:
            self.fail("expected ValueError")
        self.assertEqual(
            [(frame.name, frame.line) for frame in frames[-2:]],
            [
                ("f", "return exc_double(x)"),
                ("exc_double", 'raise ValueError("negative")'),
            ],
        )

    def test_handled_exception_is_restored(self):
        def f(x):
            try:
                exc_double(x)
            except ValueError:
                inside = sys.exc_info()[1]
            return type(inside).__name__, sys.exc_info()[1]

        self.assertSameOutcome(f, (-1,))


class ExcManager:
    def __init__(self, log, suppress=False):
        self.log, self.suppress = log, suppress

    def __enter__(self):
        self.log.append("enter")
        return 7

    def __exit__(self, kind, value, tb):
        self.log.append(("exit", kind.__name__ if kind else None))
        return self.suppress


@unittest.skipUnless(HAS_RUNTIME, "torch.rewrite runtime not built")
class TestCfgWith(TestCase):
    """with statements inside compiled functions."""

    hw_classification = HardwareClassification.GENERIC

    def test_with_calls_enter_and_exit(self):
        def f(x, log, suppress):
            y = 0
            with ExcManager(log, suppress) as m:
                y = m + exc_double(x)
            return y

        compiled = torch.rewrite(f, fallback=False)
        for x in (3, -3):
            for suppress in (False, True):
                outcomes = []
                for fn in (f, compiled):
                    log = []
                    try:
                        outcomes.append((fn(x, log, suppress), log))
                    except ValueError as e:
                        outcomes.append((str(e), log))
                self.assertEqual(outcomes[0], outcomes[1])

    def test_object_that_is_not_a_context_manager(self):
        class NoExit:
            def __enter__(self):
                return self

        class NoEnter:
            def __exit__(self, *args):
                return False

        def f(manager):
            with manager:
                return 1

        compiled = torch.rewrite(f, fallback=False)
        for manager in (NoExit(), NoEnter(), 3):
            self.assertEqual(call_outcome(compiled, manager), call_outcome(f, manager))
        # 3.14 names the class in full, with what it is defined in.
        missed = r"^'(.+\.)?NoExit' object does not support the context manager protocol \(missed __exit__ method\)$"
        kind, message = call_outcome(compiled, NoExit())
        self.assertIs(kind, TypeError)
        self.assertRegex(message, missed)

    def test_no_grad_block(self):
        def f(x):
            with torch.no_grad():
                y = x * 2
                z = y + 1
            return z, z.requires_grad, x * 2

        compiled = torch.rewrite(f, fallback=False)
        x = torch.ones(2, requires_grad=True)
        for _ in range(3):
            z, requires_grad, outside = compiled(x)
            self.assertEqual(z, torch.full((2,), 3.0))
            self.assertFalse(requires_grad)
            self.assertTrue(outside.requires_grad)
            self.assertTrue(torch.is_grad_enabled())

    def test_object_without_the_protocol(self):
        def f(x):
            with x:
                return 1

        compiled = torch.rewrite(f, fallback=False)
        with self.assertRaisesRegex(
            TypeError, "does not support the context manager protocol"
        ):
            compiled(3)

    def test_module_calls_are_compiled_all_the_way_down(self):
        def run(module, x):
            return module(x) + 1

        linear = torch.nn.Linear(3, 2)
        # Without the shortcut through Module.__call__, that is compiled too.
        compiled = torch.rewrite(run, fallback=False, passes=without("shortcuts"))
        x = torch.randn(4, 3)
        for _ in range(3):
            self.assertEqual(compiled(linear, x), run(linear, x))
        stats = compiled.stats()
        self.assertEqual(stats["interpreted_callees"], [])
        self.assertIn("Module._call_impl", stats["compiled_callees"])
        self.assertIn("Linear.forward", stats["compiled_callees"])


class BytecodeBase:
    def scale(self, x):
        return x * 2

    @property
    def name(self):
        return "base"


class BytecodeChild(BytecodeBase):
    def scale(self, x):
        return super().scale(x) + 1

    def explicit(self, x):
        return super(BytecodeChild, self).scale(x)  # noqa: UP008

    @property
    def name(self):
        return "child of " + super().name


class MatchPoint:
    __match_args__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


def match_describe(value):
    match value:
        case []:
            return "empty"
        case [only]:
            return "one", only
        case [first, *rest]:
            return "many", first, rest
        case {"kind": kind, **others}:
            return "mapping", kind, others
        case MatchPoint(0, y=height):
            return "on the axis", height
        case MatchPoint(x, y):
            return "point", x, y
        case int() | float() as number if number > 0:
            return "positive", number
        case str(text):
            return "text", text
        case complex(real=real, imag=imag):
            return "complex", real, imag
        case _:
            return "other"


def match_too_many(value):
    match value:
        case MatchPoint(a, b, c):
            return a, b, c


def match_same_attribute_twice(value):
    match value:
        case MatchPoint(a, x=b):
            return a, b


def match_not_a_class(value, pattern):
    match value:
        case pattern():
            return True


@unittest.skipUnless(HAS_RUNTIME, "torch.rewrite runtime not built")
class TestCfgMoreBytecodes(TestCase):
    """Imports inside functions, super() and star unpacking."""

    hw_classification = HardwareClassification.GENERIC

    def test_imports_inside_a_function(self):
        def f(x):
            import math
            import os.path
            from collections import deque, OrderedDict
            from os import path as p

            return (
                math.floor(x),
                p.basename("/a/b"),
                os.path.sep,
                OrderedDict.__name__,
                deque([x]).pop(),
            )

        def missing():
            from math import nothing_here  # noqa: F401

        compiled = torch.rewrite(f, fallback=False)
        self.assertEqual(compiled(2.5), f(2.5))
        with self.assertRaisesRegex(
            ImportError, "cannot import name 'nothing_here' from 'math'"
        ):
            torch.rewrite(missing, fallback=False)()

    def test_super(self):
        obj = BytecodeChild()
        for method in (BytecodeChild.scale, BytecodeChild.explicit):
            compiled = torch.rewrite(method, fallback=False)
            self.assertEqual(compiled(obj, 3), method(obj, 3))
        compiled = torch.rewrite(BytecodeChild.name.fget, fallback=False)
        self.assertEqual(compiled(obj), "child of base")

    def test_super_that_is_not_the_builtin(self):
        class Shadow:
            name = "something else"

        # A module that rebinds `super` gets what it asked for: its object is
        # called without arguments, as the source has it.
        method = BytecodeChild.name.fget
        compiled = torch.rewrite(method, fallback=False)
        with unittest.mock.patch(f"{__name__}.super", Shadow, create=True):
            self.assertEqual(compiled(BytecodeChild()), method(BytecodeChild()))
            self.assertEqual(compiled(BytecodeChild()), "child of something else")
        self.assertEqual(compiled(BytecodeChild()), "child of base")

    def test_star_unpacking(self):
        def f(seq):
            a, *b, c = seq
            *d, e = seq
            g, *h = seq
            return a, b, c, d, e, g, h

        compiled = torch.rewrite(f, fallback=False)
        for seq in ([1, 2, 3, 4], (1, 2), "abc", range(5)):
            self.assertEqual(compiled(seq), f(seq))
        for bad, error in (([1], ValueError), (5, TypeError)):
            with self.assertRaises(error) as expected:
                f(bad)
            with self.assertRaises(error) as got:
                compiled(bad)
            self.assertEqual(str(got.exception), str(expected.exception))

    def test_star_arguments_made_in_a_loop(self):
        def collect(*args, **kwargs):
            return args, kwargs

        def f(xs):
            return (
                collect(*xs, *[x * 2 for x in xs]),
                collect(1, *[x for x in xs if x], k=2),
                collect(*tuple(x + 1 for x in xs), **{str(x): x for x in xs}),
            )

        # The NULL next to `collect` is on the stack while the loops run.
        compiled = torch.rewrite(f, fallback=False)
        for xs in ([1, 2, 3], [], (0, 5)):
            self.assertEqual(compiled(xs), f(xs))

    def test_class_statement(self):
        def f(x):
            class Point:
                scale = 2

                def __init__(self, v):
                    self.v = v

                def scaled(self):
                    return self.v * Point.scale

            return Point(x).scaled(), Point.__qualname__

        compiled = torch.rewrite(f, fallback=False)
        self.assertEqual(compiled(3), f(3))
        self.assertEqual(compiled(torch.ones(2))[0], torch.full((2,), 2.0))
        methods = [n.rsplit(".", 1)[-1] for n in compiled.stats()["compiled_callees"]]
        self.assertEqual(methods, ["__init__", "scaled"])

    def test_del_attribute_global_and_closure_variable(self):
        def f(box):
            global bytecode_global
            box.a = 1
            del box.a
            bytecode_global = 5
            del bytecode_global
            cell = 3

            def inner():
                return cell  # noqa: F821

            before = inner()
            del cell
            return hasattr(box, "a"), before, call_outcome(inner)

        def missing_attribute(box):
            del box.missing

        def missing_global():
            global bytecode_global
            del bytecode_global

        def missing_closure_variable():
            cell = 1

            def inner():
                return cell  # noqa: F821

            for _ in range(2):
                del cell

        for fn, args in (
            (f, (BytecodeBase(),)),
            (missing_attribute, (BytecodeBase(),)),
            (missing_global, ()),
            (missing_closure_variable, ()),
        ):
            compiled = torch.rewrite(fn, fallback=False)
            self.assertEqual(call_outcome(compiled, *args), call_outcome(fn, *args))
        self.assertNotIn("bytecode_global", globals())

    def test_match_statement(self):
        compiled = torch.rewrite(match_describe, fallback=False)
        subjects = (
            [],
            [1],
            [1, 2, 3],
            (4, 5),
            "ab",
            {"kind": "k", "n": 1},
            {"other": 1},
            collections.defaultdict(list, kind=3),
            MatchPoint(0, 2),
            MatchPoint(1, 2),
            5,
            -5,
            2.5,
            True,
            1 + 2j,
            None,
            torch.ones(2),
        )
        for subject in subjects:
            self.assertEqual(compiled(subject), match_describe(subject))
        # Looking for the key did not create it.
        self.assertEqual(len(subjects[7]), 1)
        for fn, args in (
            (match_too_many, (MatchPoint(1, 2),)),
            (match_same_attribute_twice, (MatchPoint(1, 2),)),
            (match_not_a_class, (1, 2)),
        ):
            outcome = call_outcome(torch.rewrite(fn, fallback=False), *args)
            self.assertEqual(outcome, call_outcome(fn, *args))
            self.assertIs(outcome[0], TypeError)

    def test_jump_to_an_instruction_with_extended_arg(self):
        # A loop long enough that its header takes an EXTENDED_ARG prefix,
        # which is what the jump back to it then names.
        body = "".join(f"        x = x + {k}\n" for k in range(120))
        namespace = {}
        exec(f"def big(x, n):\n    for i in range(n):\n{body}    return x\n", namespace)
        big = namespace["big"]
        self.assertIn("EXTENDED_ARG", [ins.opname for ins in dis.get_instructions(big)])
        compiled = torch.rewrite(big, fallback=False)
        self.assertEqual(compiled(0, 3), big(0, 3))
        self.assertEqual(compiled(torch.zeros(2), 2), big(torch.zeros(2), 2))

    def test_optimizer_step_is_compiled(self):
        def step(opt):
            opt.zero_grad()
            opt.step()

        weight = torch.nn.Parameter(torch.ones(2))
        opt = torch.optim.SGD([weight], lr=0.5)
        compiled = torch.rewrite(step, fallback=False)
        for _ in range(2):
            weight.grad = torch.ones(2)
            opt.step()
            compiled(opt)
        self.assertEqual(weight, torch.zeros(2))
        self.assertIn("SGD.step", compiled.stats()["compiled_callees"])
        self.assertIn("Optimizer.zero_grad", compiled.stats()["compiled_callees"])


class FrameChild(BytecodeBase):
    def scale(self, x):
        parent = super()
        return parent.scale(x) + 1


def frame_caller():
    """Where the function that called the caller of this one is, as the stack has it."""
    frame = sys._getframe(1)
    names = []
    while frame is not None and len(names) < 3:
        names.append(frame.f_code.co_name)
        frame = frame.f_back
    caller = sys._getframe(1)
    return caller.f_lineno, names, caller.f_globals is globals()


def frame_inner():
    return frame_caller(), sys._getframe().f_code is frame_inner.__code__


def frame_outer():
    return frame_inner()


def frame_new_classes():
    pair = collections.namedtuple("pair", "a b")
    plain = type("plain", (), {})
    return pair.__module__, plain.__module__


def frame_warns():
    warnings.warn("from a compiled function", UserWarning, stacklevel=1)
    return 1


def frame_divides(x):
    y = x + 1
    return (y, 1 / (x - x))


def frame_calls_divides(x):
    return frame_divides(x)


def frame_raises_in_try():
    try:
        1 / 0
    finally:
        pass


def frame_raises_in_with():
    with ExcManager([], False):
        1 / 0


def frame_raises_in_except():
    try:
        1 / 0
    except ZeroDivisionError:
        raise KeyError("from the handler") from None


def frame_missing_global():
    return frame_surely_not_defined  # noqa: F821


def frame_unbound_local(flag):
    if flag:
        value = 1
    return value


def frame_locals(a, flag):
    if flag:
        late = a + 1
    return locals()


def frame_locals_with_cells(a):
    b = a * 2

    def inner():
        return a + b

    return {k: v for k, v in locals().items() if k != "inner"}, inner()


def frame_locals_of_a_closure(free):
    def uses_free(x):
        y = x + free
        return locals()

    return uses_free


def frame_names(a, b):
    c = a + b
    return dir(), c


def frame_vars(a, obj):
    b = a + 1
    return vars(), sorted(vars(obj)), "scale" in dir(obj), b


def frame_locals_in_a_comprehension(n):
    return [sorted(locals()) for i in range(n)]


def frame_locals_in_a_handler(a):
    try:
        b = a + 1
        raise ValueError("x")
    except ValueError:
        c = 2
        caught = sorted(locals())
    return caught, sorted(locals())


def frame_locals_after_del(a):
    b = a + 1
    del b
    return locals()


def frame_case_name(f, args):
    return f"{f.__name__}_{args[-1] if type(args[-1]) in (bool, int) else ''}"


def frame_variables():
    """The variables of the function that called this one, as its frame has them."""
    return dict(sys._getframe(1).f_locals)


def frame_binds(a, b=2, *rest, flag=False, **more):
    first = frame_variables()
    c = a + b
    second = frame_variables()
    c = None  # noqa: F841
    del a
    return first, second, frame_variables()


def frame_binds_in_a_loop(n):
    seen = []
    for i in range(n):
        square = i * i  # noqa: F841
        seen.append({k: v for k, v in frame_variables().items() if k != "seen"})
    return seen


def frame_binds_cells(a):
    b = a * 2

    def inner():
        return a + b

    return {k: v for k, v in frame_variables().items() if k != "inner"}, inner()


def frame_binds_in_a_handler(a):
    try:
        b = a + 1  # noqa: F841
        raise ValueError("x")
    except ValueError:
        c = 2  # noqa: F841
        caught = sorted(frame_variables())
    return caught, sorted(frame_variables())


def frame_binds_in_a_comprehension(n):
    return [sorted(frame_variables()) for i in range(n)], sorted(frame_variables())


def frame_evals(a):
    b = a + 1  # noqa: F841
    return eval("a + b"), eval("a + b", {"a": 10, "b": 20})


def frame_execs(a):
    scope = {}
    exec("c = a * 3", {"a": a}, scope)
    exec("d = a + 1")
    # From 3.13 on what exec binds is not in what locals() returns (PEP 667).
    return scope, locals().get("d")


def frame_reads_through_another_name(a, b):
    read = locals
    return sorted(read())


def frame_super(obj, x):
    return obj.scale(x)


def frame_fails(a):
    b = a + 1
    c = [b]
    del a
    return c[5]


def frame_calls_fails(a):
    d = a * 2  # noqa: F841
    return frame_fails(a)


def frame_tensors(x):
    y = x * 2
    z = y + 1
    return frame_variables(), z


def frame_tensor_loop(x, n):
    for _ in range(n):
        x = x * 2 + 1
    return x


def frame_reads_amid_tensor_work(x):
    y = x * 2 + 1
    y = y * 2 + 1
    y = y * 2 + 1
    frame = sys._getframe()
    here = (globals() is frame.f_globals, frame.f_code.co_name, frame.f_lineno)
    z = y * 2 + 1
    z = z * 2 + 1
    return z * 3, here


def frame_region(x):
    y = x * 2 + 1
    y = y * 2 + 1
    hidden = y * 3
    z = hidden * 2 + 1
    seen = dict(sys._getframe().f_locals)
    w = z * 2 + 1
    w = w * 2 + 1
    return z, seen, w * 3


def frame_path(x, flag):
    y = x * 2 + 1
    y = y * 2 + 1
    if flag:
        y = y * 3 + 1
    hidden = y * 3
    z = hidden * 2 + 1
    return z, frame_variables()


@unittest.skipUnless(HAS_RUNTIME, "torch.rewrite runtime not built")
class TestCfgFrames(TestCase):
    """A compiled function runs under a frame of the interpreter, which has its variables."""

    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "f,args",
        (
            (frame_binds, (1,)),
            (frame_binds, (1, 5, 6, 7)),
            (frame_binds_in_a_loop, (3,)),
            (frame_binds_cells, (3,)),
            (frame_binds_in_a_handler, (1,)),
            (frame_binds_in_a_comprehension, (2,)),
        ),
        name_fn=lambda f, args: f"{f.__name__}_{len(args)}",
    )
    def test_frame_has_the_variables(self, f, args):
        # What a callee finds in the frame of its caller.
        compiled = torch.rewrite(f, fallback=False)
        more = {"flag": True, "k": 1} if f is frame_binds else {}
        for _ in range(4):
            self.assertEqual(compiled(*args, **more), f(*args, **more))
        self.assertIn("store_local", compiled.dump())

    @parametrize(
        "f,args",
        (
            (frame_evals, (1,)),
            (frame_execs, (2,)),
            (frame_reads_through_another_name, (1, 2)),
            (frame_super, (FrameChild(), 3)),
        ),
        name_fn=lambda f, args: f.__name__,
    )
    def test_function_that_reads_its_frame_is_compiled(self, f, args):
        compiled = torch.rewrite(f, fallback=False)
        for _ in range(4):
            self.assertEqual(compiled(*args), f(*args))
        if f is frame_super:
            # `parent = super()` finds the class and the instance in the frame.
            self.assertIn("FrameChild.scale", compiled.stats()["compiled_callees"])

    @parametrize("inline", (False, True))
    def test_traceback_has_the_variables(self, inline):
        def variables(fn):
            try:
                fn(1)
            except IndexError as e:
                tb = e.__traceback__.tb_next
                return dict(tb.tb_frame.f_locals), dict(tb.tb_next.tb_frame.f_locals)

        compiled = torch.rewrite(
            frame_calls_fails,
            passes=canonical if inline else without("inline"),
            fallback=False,
        )
        for _ in range(3):
            self.assertEqual(variables(compiled), variables(frame_calls_fails))
        self.assertEqual(variables(compiled), ({"a": 1, "d": 2}, {"b": 2, "c": [2]}))
        self.assertEqual(compiled.stats()["inlined"], ["frame_fails"] if inline else [])

    def test_tensor_variable_is_bound_when_something_is_called(self):
        compiled = torch.rewrite(frame_tensors, fallback=False)
        x = torch.ones(3)
        for _ in range(8):
            seen, z = compiled(x)
            self.assertEqual(list(seen), ["x", "y", "z"])
            self.assertIs(seen["x"], x)
            self.assertIs(seen["z"], z)
            self.assertEqual(seen["y"], x * 2)
        # The kernels gave tensors without a Python object.
        self.assertGreater(compiled.stats()["unboxed_results"], 0)

    def test_binding_a_variable_does_not_give_a_tensor_its_object(self):
        compiled = torch.rewrite(frame_tensor_loop, fallback=False, demote=False)
        x = torch.ones(3)
        for _ in range(6):
            compiled(x, 40)
        before = compiled.stats()
        self.assertEqual(compiled(x, 40), frame_tensor_loop(x, 40))
        after = compiled.stats()
        self.assertGreaterEqual(
            after["unboxed_results"] - before["unboxed_results"], 80
        )
        self.assertLessEqual(after["boxed_later"] - before["boxed_later"], 2)

    @parametrize(
        "f,args",
        (
            (frame_locals, (1, True)),
            (frame_locals, (1, False)),
            (frame_locals_with_cells, (3,)),
            (frame_locals_of_a_closure(10), (1,)),
            (frame_names, (1, 2)),
            (frame_vars, (5, FrameChild())),
            (frame_locals_in_a_comprehension, (2,)),
            (frame_locals_in_a_handler, (1,)),
            (frame_locals_after_del, (1,)),
        ),
        name_fn=frame_case_name,
    )
    def test_locals_vars_and_dir_see_the_variables(self, f, args):
        # The variables are in registers, cells and frame slots; the builtins
        # would look for them in the frame.
        compiled = torch.rewrite(f, fallback=False)
        for _ in range(3):
            self.assertEqual(compiled(*args), f(*args))

    def test_locals_holds_tensors_of_specialized_code(self):
        def f(x, y):
            z = x + y
            w = z * 2
            return locals()

        compiled = torch.rewrite(f, fallback=False)
        x, y = torch.ones(2), torch.full((2,), 3.0)
        for _ in range(5):
            got = compiled(x, y)
        self.assertEqual(list(got), ["x", "y", "z", "w"])
        self.assertEqual(got, f(x, y))

    def test_global_called_locals_that_is_something_else_is_called(self):
        def f(a):
            return locals()

        compiled = torch.rewrite(f, fallback=False)
        self.assertEqual(compiled(1), {"a": 1})
        f.__globals__["locals"] = lambda: "not the builtin"
        try:
            self.assertEqual(compiled(1), "not the builtin")
        finally:
            del f.__globals__["locals"]
        self.assertEqual(compiled(1), {"a": 1})

    def test_attribute_named_like_a_builtin_that_reads_the_frame(self):
        # `model.eval()` is not `eval()`.
        def f(model, x):
            model.eval()
            return model(x)

        model, x = torch.nn.Linear(2, 2), torch.ones(2)
        compiled = torch.rewrite(f, fallback=False)
        self.assertEqual(compiled(model, x), f(model, x))

    @parametrize("inline", (False, True))
    def test_the_stack_shows_compiled_functions(self, inline):
        compiled = torch.rewrite(
            frame_outer,
            passes=canonical if inline else without("inline"),
            fallback=False,
        )
        self.assertEqual(compiled(), frame_outer())
        self.assertEqual(compiled()[0][1][:2], ["frame_inner", "frame_outer"])
        stats = compiled.stats()
        if inline:
            # An inlined function has a frame of its own while it runs.
            self.assertEqual(stats["inlined"], ["frame_inner", "frame_caller"])
            dump = compiled.dump(versions=False)
            self.assertEqual(dump.count("frame_enter <function frame_"), 2)
            self.assertEqual(dump.count("frame_leave"), 2)
        else:
            self.assertIn("frame_inner", stats["compiled_callees"])

    def test_new_classes_belong_to_the_module_of_the_compiled_function(self):
        compiled = torch.rewrite(frame_new_classes, fallback=False)
        self.assertEqual(compiled(), (__name__, __name__))

    def test_warning_names_the_line_of_the_compiled_function(self):
        def caught(fn):
            with warnings.catch_warnings(record=True) as log:
                warnings.simplefilter("always")
                fn()
            return [(w.filename, w.lineno, str(w.message)) for w in log]

        compiled = torch.rewrite(frame_warns, fallback=False)
        self.assertEqual(caught(compiled), caught(frame_warns))
        self.assertEqual(caught(compiled)[0][0], __file__)

    @parametrize("inline", (False, True))
    def test_traceback_has_frames_lines_and_columns(self, inline):
        def entries(fn):
            try:
                fn(3)
            except ZeroDivisionError as e:
                frames = traceback.extract_tb(e.__traceback__)[1:]
                last = e.__traceback__
                while last.tb_next is not None:
                    last = last.tb_next
                back = last.tb_frame.f_back.f_code.co_name
                return [(f.name, f.lineno, f.colno, f.end_colno) for f in frames], back

        compiled = torch.rewrite(
            frame_calls_divides,
            passes=canonical if inline else without("inline"),
            fallback=False,
        )
        self.assertEqual(entries(compiled), entries(frame_calls_divides))
        self.assertEqual(
            [entry[0] for entry in entries(compiled)[0]],
            ["frame_calls_divides", "frame_divides"],
        )
        self.assertEqual(
            compiled.stats()["inlined"], ["frame_divides"] if inline else []
        )
        # The frames of inlined functions are popped on the way out.
        self.assertTrue(
            sys._getframe().f_code.co_name.startswith("test_traceback_has_frames")
        )

    @parametrize(
        "fn",
        (frame_raises_in_try, frame_raises_in_with, frame_raises_in_except),
        name_fn=lambda fn: fn.__name__,
    )
    def test_frame_that_was_left_reports_the_line_the_interpreter_reports(self, fn):
        def where(f):
            try:
                f()
            except Exception as e:
                tb = e.__traceback__.tb_next
                frame = tb.tb_frame
                return frame.f_lineno, frame.f_lasti, tb.tb_lineno, tb.tb_lasti

        self.assertEqual(where(torch.rewrite(fn, fallback=False)), where(fn))

    def test_name_error_names_the_variable(self):
        def outcome(f, *args):
            try:
                f(*args)
            except NameError as e:
                return type(e), str(e), e.name

        for fn, args in ((frame_missing_global, ()), (frame_unbound_local, (False,))):
            compiled = torch.rewrite(fn, fallback=False)
            self.assertEqual(outcome(compiled, *args), outcome(fn, *args))
        self.assertEqual(outcome(frame_missing_global)[2], "frame_surely_not_defined")

    def test_traceback_keeps_what_the_locals_hold_alive(self):
        class Probe:
            pass

        probes = []

        def fails():
            probe = Probe()
            probes.append(weakref.ref(probe))
            raise ValueError("with a local")

        for fn in (fails, torch.rewrite(fails, fallback=False)):
            try:
                fn()
            except ValueError as e:
                # The traceback has the frame, and the frame the local.
                self.assertIsNotNone(probes[-1]())
                traceback.clear_frames(e.__traceback__)
                self.assertIsNone(probes[-1]())

    @parametrize("inline", (False, True))
    def test_calls_made_under_a_trace_function_run_in_the_interpreter(self, inline):
        compiled = torch.rewrite(
            frame_sets_a_trace,
            passes=canonical if inline else without("inline"),
            fallback=False,
        )
        for _ in range(4):
            self.assertEqual(compiled(3), frame_sets_a_trace(3))
        events = [event for event, _ in compiled(3)[3]]
        self.assertEqual(events, ["call", "line", "line", "return"])
        inlined = compiled.stats()["inlined"]
        self.assertEqual("frame_traced_callee" in inlined, inline)

    def test_callee_whose_globals_are_not_a_plain_dict_runs_in_the_interpreter(self):
        class Defaults(dict):
            def __missing__(self, key):
                return 7

        # The interpreter asks the mapping, which makes up what it lacks.
        adds = eval("lambda: made_up + 1", Defaults())

        def f(g):
            return g()

        with self.assertRaisesRegex(torch.rewrite.Unsupported, "not a plain dict"):
            torch.rewrite(adds, fallback=False)
        compiled = torch.rewrite(f, fallback=False)
        self.assertEqual(compiled(adds), 8)
        self.assertEqual(len(compiled.stats()["interpreted_callees"]), 1)

    def test_call_of_a_c_function_does_not_keep_its_tensors_alive(self):
        # The first calls of a C function are recorded, to see whether it is
        # one kernel: when it is a whole computation, its tensors come and go.
        def f(x, log):
            return FrameWork.apply(x, log)

        compiled = torch.rewrite(f, fallback=False)
        x, log = torch.ones(3), []
        for _ in range(4):
            self.assertEqual(compiled(x, log), x + 1)
        self.assertEqual(log, [[True] * 4] * 4)

    def test_a_function_that_enters_a_debugger_is_left_to_the_interpreter(self):
        stopped = []

        def enters(x):
            y = x + 1
            breakpoint()
            return y * 2

        def f(x):
            return enters(x) + 1

        with self.assertRaisesRegex(torch.rewrite.Unsupported, "stops a debugger"):
            torch.rewrite(enters, fallback=False)
        compiled = torch.rewrite(f, fallback=False)
        hook = lambda: stopped.append(sys._getframe(1).f_locals["y"])  # noqa: E731
        with unittest.mock.patch("sys.breakpointhook", hook):
            for _ in range(3):
                self.assertEqual(compiled(3), 9)
        # The debugger was in the frame of the function, which the
        # interpreter ran.
        self.assertEqual(stopped, [4, 4, 4])
        names = compiled.stats()["interpreted_callees"]
        self.assertEqual([name.rsplit(".", 1)[-1] for name in names], ["enters"])

    def test_a_debugger_is_told_when_a_frame_it_is_on_returns(self):
        events = []

        def tracer(frame, event, arg):
            events.append((frame.f_code.co_name, event))
            return tracer

        def enters():
            # What pdb.set_trace() does to the frames it is entered under.
            sys.settrace(tracer)
            sys._getframe(1).f_trace = tracer
            sys._getframe(2).f_trace = tracer

        def middle(x):
            enters()
            return x + 1

        def f(x):
            return middle(x) * 2

        def returns(fn):
            events.clear()
            try:
                self.assertEqual(fn(1), 4)
            finally:
                sys.settrace(None)
            return [e for e in events if e[1] == "return" and e[0] in ("middle", "f")]

        compiled = torch.rewrite(f, fallback=False, passes=without("inline"))
        for _ in range(3):
            self.assertEqual(returns(compiled), returns(f))
        self.assertEqual(returns(compiled), [("middle", "return"), ("f", "return")])
        self.assertEqual(compiled.stats()["interpreted_callees"], [])

    @parametrize("backend", PDB_BACKENDS)
    def test_pdb_stops_in_a_callee_and_on_the_way_out(self, backend):
        def leaf(x):
            return x + 1

        def stops(x):
            debugger.set_trace(sys._getframe())
            return leaf(x)

        def middle(x):
            return stops(x) * 2

        def f(x):
            return leaf(middle(x))

        def session(fn):
            nonlocal debugger
            commands = ["tbreak leaf", "continue", *["return"] * 4, "continue"]
            out = io.StringIO()
            debugger = pdb.Pdb(
                stdin=io.StringIO("\n".join(commands) + "\n"),
                stdout=out,
                nosigint=True,
                readrc=False,
                **({"backend": backend} if len(PDB_BACKENDS) > 1 else {}),
            )
            debugger.use_rawinput = False
            try:
                self.assertEqual(fn(1), 5)
            finally:
                sys.settrace(None)
                debugger.clear_all_breaks()
                if backend == "monitoring":
                    sys.monitoring.free_tool_id(sys.monitoring.DEBUGGER_ID)
            # "> file(line)function()->value" is where it stopped.
            stops = [line for line in out.getvalue().splitlines() if line[:1] == ">"]
            return [line.split(")", 1)[1] for line in stops]

        debugger = None
        compiled = torch.rewrite(f, fallback=False, passes=without("inline"))
        for _ in range(3):
            self.assertEqual(session(compiled), session(f))
        # Where it was entered, the breakpoint in a function that was called
        # afterwards, and the returns of that function and of the three
        # frames it was entered under, two of them rewritten.
        self.assertEqual(
            session(compiled),
            ["stops()", "leaf()", "leaf()->2", "stops()->2", "middle()->4", "f()->5"],
        )

    def test_callee_that_reads_its_frame_is_compiled(self):
        def fmt(name, value):
            return eval("'{}={}'.format(name, value)")

        def f(obj, x):
            return fmt("x", x), obj.scale(x)

        compiled = torch.rewrite(f, fallback=False)
        self.assertEqual(compiled(FrameChild(), 3), ("x=3", 7))
        self.assertEqual(compiled.stats()["interpreted_callees"], [])
        names = [
            name.rsplit(".", 1)[-1] for name in compiled.stats()["compiled_callees"]
        ]
        ours = [name for name in names if name in ("fmt", "scale")]
        self.assertEqual(sorted(ours), ["fmt", "scale", "scale"])


def frame_traced_callee(a):
    b = a + 1
    return b * 2


def frame_sets_a_trace(n):
    events = []

    def tracer(frame, event, arg):
        if frame.f_code is frame_traced_callee.__code__:
            events.append((event, frame.f_lineno - frame.f_code.co_firstlineno))
        return tracer

    before = frame_traced_callee(n)
    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        traced = frame_traced_callee(n)
    finally:
        sys.settrace(previous)
    return before, traced, frame_traced_callee(n), events


class FrameWork(torch.autograd.Function):
    """Computes with temporaries, in a function that is entered from C."""

    @staticmethod
    def forward(ctx, x, log):
        gone = []
        for _ in range(4):
            temporary = x * 2
            gone.append(weakref.ref(temporary))
            del temporary
        log.append([ref() is None for ref in gone])
        return x + 1

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def breaker_signals_itself(log):
    os.kill(os.getpid(), signal.SIGUSR1)
    log.append("after the call")


def breaker_spins(flag, limit):
    n = 0
    while not flag and n < limit:
        n += 1
    return n


@unittest.skipUnless(HAS_RUNTIME, "torch.rewrite runtime not built")
class TestCfgEvalBreaker(TestCase):
    """Signals, and the other threads, get their turn as they do under the interpreter."""

    hw_classification = HardwareClassification.GENERIC

    @unittest.skipUnless(hasattr(signal, "SIGUSR1"), "needs SIGUSR1")
    def test_signal_handler_runs_when_the_call_returns(self):
        def run(fn):
            log = []

            def handler(signum, frame):
                log.append(("handler", frame.f_code.co_name))

            previous = signal.signal(signal.SIGUSR1, handler)
            try:
                fn(log)
            finally:
                signal.signal(signal.SIGUSR1, previous)
            return log

        compiled = torch.rewrite(breaker_signals_itself, fallback=False)
        expected = [("handler", "breaker_signals_itself"), "after the call"]
        self.assertEqual(run(breaker_signals_itself), expected)
        self.assertEqual(run(compiled), expected)

    @unittest.skipUnless(hasattr(signal, "SIGUSR1"), "needs SIGUSR1")
    def test_exception_of_a_signal_handler_is_raised_by_the_call(self):
        def handler(signum, frame):
            raise KeyboardInterrupt("from the handler")

        compiled = torch.rewrite(breaker_signals_itself, fallback=False)
        previous = signal.signal(signal.SIGUSR1, handler)
        try:
            for fn in (breaker_signals_itself, compiled):
                log = []
                with self.assertRaisesRegex(KeyboardInterrupt, "from the handler"):
                    fn(log)
                self.assertEqual(log, [])
        finally:
            signal.signal(signal.SIGUSR1, previous)

    def test_other_threads_run_during_a_compiled_loop(self):
        flag = []

        def worker():
            time.sleep(0.05)
            flag.append(1)

        compiled = torch.rewrite(breaker_spins, fallback=False)
        self.assertEqual(compiled([], 10), 10)
        thread = threading.Thread(target=worker)
        thread.start()
        # The loop calls nothing: only the VM can let the other thread run.
        limit = 200_000_000
        count = compiled(flag, limit)
        thread.join()
        self.assertLess(count, limit)

    def test_garbage_is_collected_during_a_compiled_loop(self):
        def churn(n):
            for _ in range(n):
                cycle = []
                cycle.append(cycle)

        def collections(fn):
            before = gc.get_stats()[0]["collections"]
            fn(100_000)
            return gc.get_stats()[0]["collections"] - before

        # The lists are cycles, which only the collector frees, and the loop
        # calls nothing that would enter the interpreter: the collector runs
        # when the VM gives it the chance.
        compiled = torch.rewrite(churn, fallback=False)
        self.assertGreater(collections(churn), 20)
        self.assertGreater(collections(compiled), 20)


class LifetimeProbe:
    """Appends its name to a log when it is destroyed."""

    def __init__(self, name, log):
        self.name = name
        self.log = log

    def __del__(self):
        self.log.append(self.name)


def lifetime_unused_local(log):
    probe = LifetimeProbe("probe", log)  # noqa: F841
    log.append("body")
    return 1


def lifetime_rebound(log):
    probe = LifetimeProbe("first", log)
    log.append("between")
    probe = LifetimeProbe("second", log)  # noqa: F841
    log.append("after")
    return 2


def lifetime_raises(log):
    probe = LifetimeProbe("probe", log)  # noqa: F841
    log.append("body")
    raise ValueError("raised")


def lifetime_loop(log):
    for i in range(3):
        probe = LifetimeProbe(f"probe {i}", log)  # noqa: F841
        log.append(f"iteration {i}")
    log.append("after")
    return 3


def lifetime_argument(log, make=LifetimeProbe):
    def callee(probe):
        log.append("callee")
        return 4

    result = callee(make("argument", log))
    log.append("after the call")
    return result


def lifetime_buffer_export(log):
    data = bytearray(range(10))
    view = memoryview(data)  # noqa: F841
    try:
        data[1:-1] = range(11, 20)
    except BufferError:
        return "the export is still there"
    return "resized"


LIFETIME_FUNCTIONS = (
    lifetime_unused_local,
    lifetime_rebound,
    lifetime_raises,
    lifetime_loop,
    lifetime_argument,
    lifetime_buffer_export,
)


@unittest.skipUnless(HAS_RUNTIME, "torch.rewrite runtime not built")
class TestCfgLifetimes(TestCase):
    """A local variable keeps its object until it is rebound or the function is left."""

    hw_classification = HardwareClassification.GENERIC

    @parametrize("fn", LIFETIME_FUNCTIONS, name_fn=lambda fn: fn.__name__)
    @parametrize("unroll", (True, False))
    def test_objects_die_where_the_interpreter_lets_go_of_them(self, fn, unroll):
        def run(f):
            log = []
            try:
                result = f(log)
            except ValueError as e:
                result = str(e)
            return result, log

        compiled = torch.rewrite(
            fn, passes=canonical if unroll else without("unroll"), fallback=False
        )
        expected = run(fn)
        self.assertEqual(run(compiled), expected)
        self.assertEqual(run(compiled), expected)

    def test_hold_in_the_graph(self):
        compiled = torch.rewrite(lifetime_rebound, fallback=False)
        dump = compiled.dump(versions=False)
        # One for the rebinding, one for the return.
        self.assertEqual(
            [line.strip() for line in dump.splitlines() if "hold" in line],
            ["hold %probe.4", "hold %log.1, %probe.10"],
        )


instantiate_parametrized_tests(TestCfgFrames)
instantiate_parametrized_tests(TestCfgLifetimes)


class ShortcutNet(torch.nn.Module):
    kind = "net"

    def __init__(self):
        super().__init__()
        self.inner = torch.nn.Linear(3, 2)
        self.scale = torch.nn.Parameter(torch.full((2,), 2.0))
        self.register_buffer("shift", torch.ones(2))
        self.plain = 5

    @property
    def doubled(self):
        return self.plain * 2

    def forward(self, x):
        return self.inner(x) * self.scale + self.shift


def shortcut_call(module, x):
    return module(x)


def shortcut_attributes(module):
    return (
        module.inner,
        module.scale,
        module.shift,
        module.plain,
        module.kind,
        module.doubled,
        module.forward.__func__,
    )


def shortcut_missing(module):
    return module.no_such_thing


@unittest.skipUnless(HAS_RUNTIME, "torch.rewrite runtime not built")
class TestCfgShortcuts(TestCase):
    """Module.__call__ and Module.__getattr__ are skipped when a few checks show what they would do."""

    hw_classification = HardwareClassification.GENERIC

    def test_nn_module_is_what_the_shortcuts_were_written_for(self):
        # If this fails, nn.Module.__call__ or __getattr__ changed: read the
        # change and bring torch/rewrite/_shortcuts.py in line with it.
        self.assertTrue(torch.rewrite._shortcuts.unchanged())

    def test_module_call_goes_straight_to_forward(self):
        net, x = ShortcutNet(), torch.randn(4, 3)
        compiled = torch.rewrite(shortcut_call, fallback=False)
        for _ in range(3):
            self.assertEqual(compiled(net, x), net(x))
        callees = compiled.stats()["compiled_callees"]
        self.assertEqual(callees, ["Linear.forward", "ShortcutNet.forward"])
        self.assertEqual(compiled.stats()["interpreted_callees"], [])

    def test_module_with_hooks_is_called_the_long_way(self):
        net, x = ShortcutNet(), torch.randn(4, 3)
        compiled = torch.rewrite(shortcut_call, fallback=False)
        plain = net(x)
        self.assertEqual(compiled(net, x), plain)
        handle = net.register_forward_hook(lambda module, args, output: output + 100)
        self.assertEqual(compiled(net, x), plain + 100)
        self.assertIn("Module._call_impl", compiled.stats()["compiled_callees"])
        handle.remove()
        self.assertEqual(compiled(net, x), plain)
        # A hook on every module, and one on the submodule only.
        handle = torch.nn.modules.module.register_module_forward_hook(
            lambda module, args, output: output * 0
        )
        try:
            self.assertEqual(compiled(net, x), net(x))
            self.assertEqual(compiled(net, x), torch.zeros(4, 2))
        finally:
            handle.remove()
        handle = net.inner.register_forward_pre_hook(
            lambda module, args: (args[0] * 0,)
        )
        self.assertEqual(compiled(net, x), net(x))
        handle.remove()
        self.assertEqual(compiled(net, x), plain)

    def test_forward_replaced_on_the_instance(self):
        net, x = ShortcutNet(), torch.randn(4, 3)
        compiled = torch.rewrite(shortcut_call, fallback=False)
        self.assertEqual(compiled(net, x), net(x))
        net.forward = lambda x: x.sum()
        self.assertEqual(compiled(net, x), x.sum())
        del net.forward
        self.assertEqual(compiled(net, x), net(x))

    def getattr_calls(self, fn, *args):
        """How often the interpreter entered Module.__getattr__ during a call."""
        code = torch.nn.Module.__getattr__.__code__
        calls = []

        def profile(frame, event, arg):
            if event == "call" and frame.f_code is code:
                calls.append(frame)

        sys.setprofile(profile)
        try:
            fn(*args)
        finally:
            sys.setprofile(None)
        return len(calls)

    def test_parameters_buffers_and_submodules_are_found(self):
        net = ShortcutNet()
        compiled = torch.rewrite(shortcut_attributes, fallback=False)
        for _ in range(2):
            for got, expected in zip(compiled(net), shortcut_attributes(net)):
                self.assertIs(got, expected)
        self.assertEqual(self.getattr_calls(compiled, net), 0)
        plain = torch.rewrite(
            shortcut_attributes, fallback=False, passes=without("shortcuts")
        )
        # Without the shortcut the function is called, and is compiled as
        # everything is that is called.
        self.assertEqual(self.getattr_calls(plain, net), 0)
        self.assertIn("Module.__getattr__", plain.stats()["compiled_callees"])
        self.assertNotIn("Module.__getattr__", compiled.stats()["compiled_callees"])
        net.scale = torch.nn.Parameter(torch.zeros(2))
        net.plain = 7
        self.assertIs(compiled(net)[1], net.scale)
        self.assertEqual(compiled(net)[3:6], (7, "net", 14))
        missing = torch.rewrite(shortcut_missing, fallback=False)
        self.assertEqual(
            call_outcome(missing, net), call_outcome(shortcut_missing, net)
        )
        self.assertEqual(call_outcome(missing, net)[0], AttributeError)

    def test_property_that_raises_is_read_once(self):
        class Guarded(torch.nn.Module):
            reads = 0

            @property
            def fragile(self):
                type(self).reads += 1
                raise AttributeError("not now")

        def read(module):
            return module.fragile

        # The normal lookup runs the property; when it raises AttributeError,
        # Module.__getattr__ is asked, without the property being run again.
        module = Guarded()
        compiled = torch.rewrite(read, fallback=False)
        expected = call_outcome(read, module)
        self.assertEqual(Guarded.reads, 1)
        self.assertEqual(call_outcome(compiled, module), expected)
        self.assertEqual(Guarded.reads, 2)

    def test_without_shortcuts(self):
        net, x = ShortcutNet(), torch.randn(4, 3)
        compiled = torch.rewrite(
            shortcut_call, fallback=False, passes=without("shortcuts")
        )
        self.assertEqual(compiled(net, x), net(x))
        self.assertIn("Module._call_impl", compiled.stats()["compiled_callees"])


def demote_python(n):
    total = 0
    for i in range(n):
        total += i * i % 7
    return total


def demote_caller(n, x):
    return demote_python(n), x + 1


def demote_unsettled(n):
    total = 0
    for i in range(n):
        total += i * i % 7
    cfg_rs.unsettled()
    return total


def demote_unsettled_caller(n, x):
    return demote_unsettled(n), x + 1


def demote_tensor_caller(n, x):
    for _ in range(100):
        x = x + 1
    return demote_python(n), x


def demote_chain_1(n):
    return demote_python(n)


def demote_chain_2(n):
    return demote_chain_1(n)


def demote_chain_3(n):
    return demote_chain_2(n)


def demote_chain_4(n):
    return demote_chain_3(n)


@unittest.skipUnless(HAS_RUNTIME, "torch.rewrite runtime not built")
class TestCfgDemotion(TestCase):
    """What the interpreter runs faster is left to it."""

    hw_classification = HardwareClassification.GENERIC

    def test_python_callee_is_left_to_the_interpreter(self):
        compiled = torch.rewrite(
            demote_caller, fallback=False, demote=True, passes=without("inline")
        )
        x = torch.ones(2)
        for _ in range(200):
            total, y = compiled(300, x)
            self.assertEqual(total, demote_python(300))
            self.assertEqual(y, x + 1)
        stats = compiled.stats()
        # A loop of integer arithmetic is several times slower in the VM.
        self.assertEqual(stats["demoted_callees"], ["demote_python"])
        self.assertEqual(stats["compiled_callees"], ["demote_python"])

    def test_a_call_that_is_not_settled_does_not_count_against_a_function(self):
        # The same loop as demote_python, which is left to the interpreter;
        # this one says of every call that it is not what it will be.
        compiled = torch.rewrite(
            demote_unsettled_caller,
            fallback=False,
            demote=True,
            passes=without("inline"),
        )
        x = torch.ones(2)
        for _ in range(200):
            total, _ = compiled(300, x)
            self.assertEqual(total, demote_python(300))
        self.assertEqual(compiled.stats()["demoted_callees"], [])

    def test_a_region_that_is_deciding_keeps_measurements_open(self):
        calls = []
        model = RegionNet().eval()
        passes = with_graphs("aot_eager")
        compiled = torch.rewrite(region_infer, passes=passes, fallback=False)
        x = torch.randn(2, 3, 6, 6)
        said = unittest.mock.patch.object(cfg_rs, "unsettled", lambda: calls.append(1))
        with torch.no_grad(), said:
            for _ in range(3):
                compiled(model, x)
            early = len(calls)
            for _ in range(20):
                compiled(model, x)
            settled = len(calls)
            for _ in range(5):
                compiled(model, x)
        # Said while the region ran its instructions one by one, and no
        # longer once it had its graph.
        self.assertGreater(early, 0)
        self.assertEqual(len(calls), settled)
        states = [r["state"] for r in torch.rewrite.regions(compiled)]
        self.assertEqual(states.count("compiled"), 1)

    def test_python_function_is_left_to_the_interpreter(self):
        compiled = torch.rewrite(demote_python, fallback=False, demote=True)
        self.assertFalse(compiled.stats()["demoted"])
        for _ in range(200):
            self.assertEqual(compiled(300), demote_python(300))
        self.assertTrue(compiled.stats()["demoted"])

    def test_measurement_is_reported(self):
        compiled = torch.rewrite(demote_python, fallback=False, demote=True)
        self.assertTrue(compiled.stats()["measuring"])
        for _ in range(200):
            compiled(300)
        stats = compiled.stats()
        self.assertEqual((stats["measuring"], stats["demoted"]), (False, True))
        unmeasured = torch.rewrite(demote_python, fallback=False, demote=False)
        self.assertFalse(unmeasured.stats()["measuring"])

    def test_chain_of_calls_is_measured_at_once(self):
        # Five functions deep. Were each measured from scratch once the one
        # it calls is settled, the outermost would still be at it.
        compiled = torch.rewrite(
            demote_chain_4, fallback=False, demote=True, passes=without("inline")
        )
        for _ in range(150):
            self.assertEqual(compiled(300), demote_python(300))
        stats = compiled.stats()
        self.assertFalse(stats["measuring"])
        # The loop is with the interpreter: on its own, or along with
        # callers of it, which cost the same either way once it is.
        self.assertTrue(stats["demoted"] or stats["demoted_callees"])

    def test_measuring_covers_what_the_function_calls(self):
        # The tensor operations make the function faster compiled whatever
        # its callee is run as, so it is settled first. Its calls are only
        # as fast as they will be once the callee, a good part of each of
        # them, is with the interpreter.
        compiled = torch.rewrite(
            demote_tensor_caller, fallback=False, demote=True, passes=without("inline")
        )
        x = torch.ones(2)
        for _ in range(600):
            compiled(300, x)
            if not compiled.stats()["measuring"]:
                break
        stats = compiled.stats()
        self.assertFalse(stats["measuring"])
        self.assertEqual(stats["demoted_callees"], ["demote_python"])

    @slowTest
    def test_slow_function_is_measured_with_fewer_calls(self):
        # Over a second per call: three calls to warm up and two of each
        # kind, not eight and ten.
        def slow(x):
            time.sleep(1.3)
            return x + 1

        compiled = torch.rewrite(slow, fallback=False, demote=True)
        calls = 0
        while compiled.stats()["measuring"] and calls < 16:
            self.assertEqual(compiled(torch.ones(2)), torch.full((2,), 2.0))
            calls += 1
        self.assertFalse(compiled.stats()["measuring"])
        self.assertGreaterEqual(calls, 7)

    def test_measurement_starts_over_when_compiled_code_changes(self):
        compiled = torch.rewrite(demote_python, fallback=False, demote=True)
        for _ in range(200):
            # What a pass says when it has replaced part of a block.
            cfg_rs.plan_changed()
            compiled(300)
        stats = compiled.stats()
        self.assertEqual((stats["measuring"], stats["demoted"]), (True, False))
        for _ in range(200):
            compiled(300)
        stats = compiled.stats()
        self.assertEqual((stats["measuring"], stats["demoted"]), (False, True))

    def test_nothing_is_demoted_when_turned_off(self):
        compiled = torch.rewrite(
            demote_caller, fallback=False, demote=False, passes=without("inline")
        )
        for _ in range(200):
            compiled(300, torch.ones(2))
        stats = compiled.stats()
        self.assertEqual((stats["demoted"], stats["demoted_callees"]), (False, []))

    def test_error_in_a_callee_that_is_being_measured(self):
        def fails(x):
            if x > 150:
                raise ValueError("late")
            return x

        def caller(x):
            return fails(x)

        compiled = torch.rewrite(
            caller, fallback=False, demote=True, passes=without("inline")
        )
        for x in range(200):
            self.assertEqual(call_outcome(compiled, x), call_outcome(caller, x))


def e2e_halve(x):
    return x * 0.5


def e2e_program(x, y, log):
    if x.shape[0] > 1:
        x = x.mean(0, keepdim=True)
    total = e2e_halve(x)
    for i in range(3):
        total = total * 2 + y + i
    while total.sum() < 10:
        total = total + 1
    log.append(float(total.sum()))
    return total


class E2ENet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.scale = torch.nn.Parameter(torch.ones(8))
        self.fc2 = torch.nn.Linear(8, 1)
        self.register_buffer("calls", torch.zeros(()))

    def forward(self, x):
        self.calls += 1
        return self.fc2(torch.relu(self.fc1(x)) * self.scale)


def e2e_step(net, opt, x, y):
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(net(x), y)
    loss.backward()
    opt.step()
    return loss


@unittest.skipUnless(HAS_RUNTIME, "torch.rewrite runtime not built")
class TestCfgEndToEnd(TestCase):
    """Whole programs, compared bit for bit with the interpreter."""

    hw_classification = HardwareClassification.GENERIC

    def test_tensor_program(self):
        # An inlined call, an unrolled loop, a branch on a shape that is
        # narrowed under a guard, a loop on tensor data and a Python side
        # effect, in one graph.
        compiled = torch.rewrite(e2e_program, passes=narrowing(), fallback=False)
        log, expected_log = [], []
        for batch in (4, 1, 4, 8, 1):
            x, y = torch.randn(batch, 3), torch.randn(3)
            result, expected = compiled(x, y, log), e2e_program(x, y, expected_log)
            self.assertEqual(result, expected, atol=0, rtol=0)
        self.assertEqual(log, expected_log)
        stats = compiled.stats()
        self.assertEqual(stats["inlined"], ["e2e_halve"])
        self.assertEqual(stats["unrolled_loops"], 1)
        self.assertGreater(stats["sites"], 0)
        self.assertEqual(stats["site_misses"], 0)
        # The batch of one failed the guard that batches of four had put on
        # the first block: that block, and no other, was compiled again.
        entry, rest = stats["blocks"][0], stats["blocks"][1:]
        failures = entry["guard_failures"] + entry["entry_guard_misses"]
        self.assertEqual((entry["versions"], failures), (2, 1))
        self.assertLessEqual(max(block["versions"] for block in rest), 1)

    @parametrize("optimizer", ("sgd", "adam"))
    def test_training_step(self, optimizer):
        def train(compile_step):
            torch.manual_seed(0)
            net = E2ENet()
            if optimizer == "sgd":
                opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
            else:
                opt = torch.optim.Adam(net.parameters(), lr=0.01)
            step = torch.rewrite(e2e_step, fallback=False) if compile_step else e2e_step
            data = [(torch.randn(16, 4), torch.randn(16, 1)) for _ in range(10)]
            losses = [step(net, opt, x, y).item() for x, y in data]
            return net.state_dict(), losses, step

        expected_state, expected_losses, _ = train(False)
        state, losses, step = train(True)
        self.assertEqual(losses, expected_losses)
        self.assertEqual(sorted(state), sorted(expected_state))
        for name, value in state.items():
            self.assertTrue(torch.equal(value, expected_state[name]), name)
        self.assertEqual(state["calls"].item(), 10)
        # The module, its submodules and the optimizer ran compiled.
        callees = step.stats()["compiled_callees"]
        optimizer_step = "SGD.step" if optimizer == "sgd" else "Adam.step"
        for name in (
            "E2ENet.forward",
            "Linear.forward",
            "Optimizer.zero_grad",
            optimizer_step,
        ):
            self.assertIn(name, callees)
        # The modules have no hooks: their calls went straight to forward.
        self.assertNotIn("Module._call_impl", callees)


instantiate_parametrized_tests(TestCfgEndToEnd)


class RegionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.fc = torch.nn.Linear(4 * 6 * 6, 5)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        return self.fc(torch.tanh(x.flatten(1)))


class RegionBranch(torch.nn.Module):
    # Branches on data between two stretches that trace.
    def __init__(self):
        super().__init__()
        self.first = RegionNet()
        self.second = torch.nn.Sequential(
            torch.nn.Linear(5, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        h = self.first(x)
        if h.sum() > 0:
            h = h * 2
        return self.second(h)


class RegionCounting(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = RegionNet()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.inner(x)


class RegionCountingQuietly(RegionCounting):
    def forward(self, x):
        # Not through a method of the module.
        vars(self)["calls"] += 1
        return self.inner(x)


class RegionLayers(torch.nn.Module):
    # A stack of layers behind a branch on data, so that each layer is a
    # region of its own.
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(6, 6),
                torch.nn.Tanh(),
                torch.nn.Linear(6, 6),
                torch.nn.ReLU(),
            )
            for _ in range(5)
        )

    def forward(self, x):
        if x.sum() > 1e6:
            return x
        for layer in self.layers:
            x = layer(x)
        return x


class RegionMade:
    """A pytree node whose ``__init__`` counts how often it ran."""

    made = 0

    def __init__(self, loc, scale):
        type(self).made += 1
        self.loc = loc
        self.scale = scale
        self.both = loc + scale


torch.utils._pytree.register_pytree_node(
    RegionMade,
    lambda made: ([made.loc, made.scale], None),
    lambda values, context: RegionMade(*values),
    serialized_type_name="test_rewrite.RegionMade",
)


class RegionCached:
    """An object that says of its copies that they come without its cache."""

    def __init__(self, value):
        self.value = value
        self.cache = None

    def __getstate__(self):
        return {"value": self.value}


class RegionItems(collections.OrderedDict):
    """What a model returns: a dict that has its items as attributes too."""

    made = 0

    def __init__(self, **items):
        type(self).made += 1
        super().__init__()
        for key, value in items.items():
            self[key] = value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)


def region_infer(model, x):
    return model(x)


def region_step(model, opt, x, y):
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    return loss


def region_step_plain(model, opt, x):
    opt.zero_grad()
    loss = model(x).pow(2).mean()
    loss.backward()
    opt.step()
    return loss


def region_logging(model, x, log):
    out = model(x)
    log.append(out.shape[0])
    return torch.sin(out).cos().exp().mean()


def region_small(x, y):
    return x + y


def region_chain(x, y):
    return ((x + y).sin() * y).cos().exp().tanh()


def region_frozen(model, x):
    with torch.no_grad():
        target = model(x)
    return ((model(x * 2) - target) ** 2).mean()


def region_loop(model, batches):
    total = None
    for x, scale in zip(batches, (1.0, 2.0, 3.0)):
        out = model(x).mul(scale).sin().sum()
        total = out if total is None else total + out
    return total


def region_accumulates(x, total):
    total.add_(x * 2)
    y = total * 3 + 1
    return y * 2 + x


class RegionOutput:
    # A plain object, as model code uses for results.
    def __init__(self, logits, hidden, name):
        self.logits = logits
        self.hidden = hidden
        self.name = name


class RegionCache:
    # State that outlives a call and is changed by it.
    def __init__(self):
        self.seen = []

    def update(self, value):
        self.seen.append(value)
        return len(self.seen)


class RegionStateful(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = RegionNet()
        self.scale = torch.tensor(2.0)

    def forward(self, x, cache=None):
        out = self.inner(x) * self.scale
        if cache is not None:
            cache.update(out.detach())
        return RegionOutput(out, [out.sin(), out.cos()], "out")


def region_cached(model, x, cache):
    return model(x, cache)


def region_counter(x, i):
    return ((x + i).sin() * i).cos().exp().tanh()


class RegionLinears(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(6, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 6),
            torch.nn.ReLU(),
        )
        self.layers = torch.nn.ModuleList(torch.nn.Linear(6, 6) for _ in range(5))

    def forward(self, x):
        return self.net(x)


class RegionAverage:
    # State that a call reads and replaces.
    def __init__(self):
        self.mean = torch.zeros(6)

    def update(self, value):
        self.mean = self.mean * 0.9 + value.mean(0) * 0.1
        return self.mean


class RegionRunningMean(RegionAverage):
    def __init__(self):
        super().__init__()
        self.count = 0

    def update(self, value):
        self.count = self.count + 1
        self.mean = self.mean + (value.mean(0) - self.mean) / self.count
        return self.mean


def region_running(model, x, stats):
    out = model(x)
    return out - stats.update(out)


def region_remember(model, x, memory):
    out = model(x)
    memory["last"] = out.detach()
    memory["sum"] = memory.get("sum", 0) + out.sum()
    return out * 2


class RegionState:
    def __init__(self):
        self.steps = []
        self.total = None


class RegionStarts(torch.nn.Module):
    # Makes an object that has nothing of its input in it, as a model does
    # that starts a cache.
    def __init__(self):
        super().__init__()
        self.net = RegionLinears()

    def forward(self, x):
        return self.net(x), RegionState()


class RegionMasked(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 6)
        self.mask = torch.empty(0)

    def forward(self, x):
        # Built on every call, and kept.
        self.mask = torch.tril(torch.ones(x.shape[0], x.shape[0]))
        return self.linear(self.mask @ x).tanh()


def region_note(log, x, out):
    log.append(out.shape[0])
    x.add_(1)
    return float(out.sum())


def region_partway(model, x, log, note):
    # The call is in another block than the model's, where a path comes by
    # it, and its callee is an argument, so that it is not inlined.
    out = model(x)
    total = 0.0
    if note is not None:
        total = note(log, x, out)
    return out * 2, total


def region_sized(model, x):
    out = model(x)
    return out.sum(0) / x.shape[0], x.shape[0] * 2


def region_branching(model, x):
    out = model(x)
    if x.shape[0] > 4:
        out = out * 2
    return out.sin()


def region_pick(model, x, i):
    return model.layers[i](x).sin().cos()


class RegionAsks(torch.nn.Module):
    # Code of a model that asks what a number is, in the way ``how`` says:
    # the number of rows of its input, or the one it is handed.
    def __init__(self, how):
        super().__init__()
        self.net = RegionLinears()
        self.how = how

    def forward(self, x, n):
        if n is None:
            n = x.shape[0]
        if self.how == "isinstance":
            whole = isinstance(n, int)
        elif self.how == "type":
            whole = type(n) is int
        else:
            whole = str(n).isdigit()
        if whole:
            x = x * n
        return self.net(x)


def region_asked(model, x, n):
    return model(x, n)


class RegionNames(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = RegionLinears()

    def forward(self, x):
        return {f"rows_{x.shape[0]}": self.net(x)}


def region_clock(model, x):
    start = time.perf_counter_ns()
    out = model(x)
    return out, start


REGION_CONSTANT = torch.full((5,), 3.0)


def region_global(model, x):
    out = model(x)
    return (out * REGION_CONSTANT).sin().cos().exp()


class RegionRecurrent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(6, 8, num_layers=2, bidirectional=True)
        self.out = torch.nn.Linear(16, 3)

    def forward(self, x):
        y, _ = self.lstm(x)
        return self.out(y).tanh()


RegionPair = collections.namedtuple("RegionPair", ["left", "right"])


def region_pair(pair, kinds):
    total = (pair.left * pair.right).sin().cos().exp()
    if "double" in kinds:
        total = total * 2
    return RegionPair(total, total.tanh().abs().sqrt().neg())


class RegionStages(torch.nn.Module):
    # Tensor work in several blocks: a comprehension, a loop, a branch on a
    # flag. `self` is a variable that a nested function shares.
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Linear(6, 6)
        self.stages = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Linear(6, 6), torch.nn.Tanh())
            for _ in range(3)
        )
        self.head = torch.nn.Linear(6, 3)
        self.doubled = False

    def forward(self, x, log):
        h = self.stem(x)
        feats = [stage(h) for stage in self.stages]
        h = feats[0]
        for feat in feats[1:]:
            h = h + feat
        if self.doubled:
            h = h * 2
        out = self.head(h).tanh()
        log.append(float(out.sum()))

        def result():
            return out, self.doubled

        return result()


def region_stages(model, x, log):
    return model(x, log)


class RegionRandom(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = RegionNet()

    def forward(self, x):
        return self.inner(x) * random.random()


class RegionCounted(torch.nn.Module):
    # Counts its calls in a buffer, whatever mode it is in.
    def __init__(self):
        super().__init__()
        self.inner = RegionNet()
        self.register_buffer("calls", torch.zeros((), dtype=torch.long))

    def forward(self, x):
        self.calls.add_(1)
        return self.inner(x) * self.calls


class RegionTwoHeads(torch.nn.Module):
    # One head is used by forward, the other is not; and a tensor that is
    # kept in a plain attribute.
    def __init__(self):
        super().__init__()
        self.used = RegionNet()
        self.unused = RegionNet()
        self.cache = torch.zeros(5)

    def forward(self, x):
        return self.used(x) + self.cache


def region_method(model, view, x):
    return model(view(2, 3, 6, 6) + x)


def region_loss(model, x, y):
    # As models of transformers do: the loss is a module made on the spot.
    logits = model(x)
    loss = torch.nn.CrossEntropyLoss()
    return loss(torch.tanh(logits) * 2 + 1, y) + logits.mean()


class RegionAlike(torch.nn.Module):
    # Layers that are built alike, with data deciding what comes between.
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([RegionNormed() for _ in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if x.sum() > 1e6:
                x = x / 2
        return x


class RegionNormed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        return torch.tanh(torch.relu(self.bn(self.conv(x))) + x)


def temporaries(x, steps):
    # Every step frees what the step before it made.
    for _ in range(steps):
        x = torch.tanh(x * 1.5 + 0.25)
    return x


@unittest.skipUnless(HAS_RUNTIME, "torch.rewrite runtime not built")
class TestCfgWorkspace(TestCase):
    """The memory a compiled function keeps from one call to the next."""

    hw_classification = HardwareClassification.GENERIC

    def test_tensors_outside_a_call_are_left_to_malloc(self):
        before = torch.randn(1 << 16)
        workspace = torch._C._cfg_workspace()
        if workspace is None:
            self.skipTest("the process has a CPU allocator of its own")
        after = torch.randn(1 << 16)
        self.assertEqual(before.sum() + after.sum(), after.sum() + before.sum())
        # Made before and after PyTorch's allocator was replaced: both are
        # freed by the one that made them.
        del before, after
        nothing = dict.fromkeys(("calls", "hits", "misses", "kept", "live", "peak"), 0)
        self.assertEqual(torch._C._cfg_workspace_stats(workspace), nothing)

    def test_memory_of_a_call_serves_the_next(self):
        compiled = torch.rewrite(temporaries, fallback=False)
        x = torch.randn(64, 1024)
        for _ in range(4):
            self.assertEqual(compiled(x, 6), temporaries(x, 6), atol=0, rtol=0)
        before = torch.rewrite.memory(compiled)
        self.assertEqual(before["live"], 0)
        self.assertGreaterEqual(before["kept"], x.numel() * 4)
        compiled(x, 6)
        after = torch.rewrite.memory(compiled)
        # Every tensor of the call found its memory in the workspace.
        self.assertEqual(after["misses"], before["misses"])
        self.assertGreaterEqual(after["hits"] - before["hits"], 6)
        self.assertEqual(after["calls"], before["calls"] + 1)

    def test_result_holds_its_memory(self):
        compiled = torch.rewrite(temporaries, fallback=False)
        x = torch.randn(64, 1024)
        expected = temporaries(x, 3)
        kept = [compiled(x, 3) for _ in range(3)]
        # What they asked for, rounded up to pages.
        live = torch.rewrite.memory(compiled)["live"]
        self.assertGreaterEqual(live, 3 * x.numel() * 4)
        self.assertLess(live, 3 * (x.numel() * 4 + 8192))
        # The function goes first: its results still own their memory.
        del compiled
        gc.collect()
        for result in kept:
            self.assertEqual(result, expected, atol=0, rtol=0)

    def test_result_freed_on_another_thread(self):
        compiled = torch.rewrite(temporaries, fallback=False)
        x = torch.randn(64, 1024)
        kept = [compiled(x, 2)]
        thread = threading.Thread(target=kept.clear)
        thread.start()
        thread.join()
        memory = torch.rewrite.memory(compiled)
        self.assertEqual(memory["live"], 0)
        self.assertGreaterEqual(memory["kept"], x.numel() * 4)

    def test_small_tensors_are_left_to_malloc(self):
        compiled = torch.rewrite(temporaries, fallback=False)
        x = torch.randn(16)
        for _ in range(3):
            self.assertEqual(compiled(x, 6), temporaries(x, 6), atol=0, rtol=0)
        memory = torch.rewrite.memory(compiled)
        self.assertEqual((memory["hits"], memory["misses"]), (0, 0))

    def test_memory_not_asked_for_is_given_back(self):
        compiled = torch.rewrite(temporaries, fallback=False)
        large, small = torch.randn(256, 1024), torch.randn(16)
        for _ in range(3):
            compiled(large, 6)
        self.assertGreaterEqual(
            torch.rewrite.memory(compiled)["kept"], large.numel() * 4
        )
        for _ in range(6):
            compiled(small, 6)
        self.assertEqual(torch.rewrite.memory(compiled)["kept"], 0)

    def test_memory_is_cut_to_the_size_that_is_asked_for(self):
        compiled = torch.rewrite(temporaries, fallback=False)
        large, small = torch.ones(8 << 20), torch.ones(1 << 20)
        for _ in range(2):
            compiled(large, 3)
        before = torch.rewrite.memory(compiled)
        # An eighth of what a large tensor left behind serves a small one.
        compiled(small, 3)
        after = torch.rewrite.memory(compiled)
        self.assertEqual(after["misses"], before["misses"])
        self.assertGreater(after["hits"], before["hits"])
        # What comes back is one piece again with the memory around it.
        compiled(large, 3)
        self.assertEqual(torch.rewrite.memory(compiled)["misses"], before["misses"])

    def test_holds_little_more_than_its_tensors_did_at_their_most(self):
        compiled = torch.rewrite(temporaries, fallback=False)
        # Every call needs pieces larger than any the calls before it left.
        for n in (1, 2, 6, 15, 37):
            compiled(torch.ones(n << 20), 3)
            memory = torch.rewrite.memory(compiled)
            # Half as much again, 256 MB at most; and 64 MB whatever its
            # tensors held.
            spare = min(memory["peak"] // 2, 1 << 28)
            limit = max(memory["peak"] + spare, 1 << 26)
            self.assertLessEqual(memory["live"] + memory["kept"], limit)
        # The memory of the last call is all there for the next.
        self.assertGreaterEqual(memory["kept"], 2 * (37 << 22))

    def test_without_a_workspace(self):
        compiled = torch.rewrite(temporaries, fallback=False, keep_memory=False)
        x = torch.randn(64, 1024)
        self.assertEqual(compiled(x, 6), temporaries(x, 6), atol=0, rtol=0)
        self.assertIsNone(torch.rewrite.memory(compiled))
        with unittest.mock.patch.dict(os.environ, {"TORCH_REWRITE_KEEP_MEMORY": "0"}):
            compiled = torch.rewrite(temporaries, fallback=False)
        self.assertIsNone(torch.rewrite.memory(compiled))

    def test_compiled_function_called_by_another(self):
        inner = torch.rewrite(temporaries, fallback=False)

        def outer(x):
            return inner(x, 3) + 1

        compiled = torch.rewrite(outer, fallback=False, callees=False)
        x = torch.randn(64, 1024)
        for _ in range(4):
            self.assertEqual(compiled(x), temporaries(x, 3) + 1, atol=0, rtol=0)
        # Back in the outer function, its own workspace is the one in use.
        self.assertGreater(torch.rewrite.memory(inner)["hits"], 0)
        self.assertGreater(torch.rewrite.memory(compiled)["hits"], 0)
        self.assertEqual(torch.rewrite.memory(compiled)["live"], 0)
        self.assertEqual(torch.rewrite.memory(inner)["live"], 0)


@unittest.skipUnless(HAS_RUNTIME, "torch.rewrite runtime not built")
class TestCfgRegions(TestCase):
    """Runs of a block compiled as one graph by a ``graphs`` pass."""

    hw_classification = HardwareClassification.GENERIC

    def compiled_regions(self, fn):
        return [r for r in torch.rewrite.regions(fn) if r["state"] == "compiled"]

    def left_to_vm(self, fn, where):
        return [
            r["reason"]
            for r in torch.rewrite.regions(fn)
            if r["state"] == "vm" and r["where"].startswith(where)
        ]

    def test_frame_readers_amid_tensor_work_find_their_function(self):
        # The instructions of a region run from frames of the region's own
        # until its graph is made: what they call is called by the function.
        graphs = lambda graph, example: graph  # noqa: E731
        fn = frame_reads_amid_tensor_work
        compiled = torch.rewrite(fn, passes=with_graphs(graphs), fallback=False)
        x = torch.ones(3)
        for _ in range(8):
            self.assertEqual(compiled(x), fn(x))
        self.assertEqual(compiled(x)[1][:2], (True, fn.__name__))
        self.assertEqual(len(self.compiled_regions(compiled)), 2)

    @parametrize(
        "f,args",
        ((frame_region, ()), (frame_path, (True,)), (frame_path, (False,))),
        name_fn=lambda f, args: f.__name__ + "".join(map(str, args)),
    )
    def test_region_binds_the_variables_it_gives(self, f, args):
        graphs = lambda graph, example: graph  # noqa: E731
        compiled = torch.rewrite(f, passes=with_graphs(graphs), fallback=False)
        refs = []
        for i in range(8):
            x = torch.full((3,), float(i))
            z, seen, *more = compiled(x, *args)
            want, there, *rest = f(x, *args)
            self.assertEqual([z, *more], [want, *rest])
            self.assertIs(seen["x"], x)
            self.assertIs(seen["z"], z)
            self.assertLessEqual(set(seen), set(there))
            for name in seen:
                self.assertEqual(seen[name], there[name])
            tensors = [t for t in seen.values() if isinstance(t, torch.Tensor)]
            refs.extend(weakref.ref(t) for t in tensors)
            del z, seen, more, want, there, rest, x, tensors
        # One whose instructions are replaced where they are, and one that
        # goes on into the blocks after it.
        ends = [bool(region["ends"]) for region in self.compiled_regions(compiled)]
        self.assertEqual(ends, [True] if f is frame_path else [False, True])
        # The first gives what the function keeps to its end, as the
        # frame does. The second gives what is read after it: a variable it
        # does not give has no value to be bound to.
        seen = compiled(torch.ones(3), *args)[1]
        bound = {"x", "z", "flag"} if f is frame_path else {"x", "y", "hidden", "z"}
        self.assertEqual(set(seen), bound)
        del seen
        # And no frame kept a tensor of an earlier call.
        gc.collect()
        self.assertEqual([ref() for ref in refs], [None] * len(refs))

    def test_graph_that_fails_on_its_first_run_does_not_write_twice(self):
        runs = []

        def backend(graph, example):
            def run(*args):
                runs.append(1)
                # Writes to `total`, as the instructions will when they run
                # in its place.
                graph(*args)
                raise RuntimeError("after the write")

            return run

        fn = region_accumulates
        compiled = torch.rewrite(fn, passes=with_graphs(backend), fallback=False)
        x, mine, theirs = torch.ones(3), torch.zeros(3), torch.zeros(3)
        for _ in range(8):
            self.assertEqual(compiled(x, mine), fn(x, theirs))
            self.assertEqual(mine, theirs)
        self.assertEqual(len(runs), 1)
        (reason,) = self.left_to_vm(compiled, "region_accumulates")
        self.assertIn("the compiled graph failed", reason)

    def test_module_call_is_one_graph(self):
        torch.manual_seed(0)
        model = RegionNet().eval()
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        with torch.no_grad():
            for _ in range(6):
                x = torch.randn(2, 3, 6, 6)
                self.assertEqual(
                    compiled(model, x), region_infer(model, x), atol=0, rtol=0
                )
        (region,) = self.compiled_regions(compiled)
        # The call in the function that was compiled took in everything
        # below it: the regions of the modules it calls were never tried.
        self.assertTrue(region["where"].startswith("region_infer"))
        self.assertEqual(region["variants"], 1)
        self.assertGreaterEqual(region["ops"], 6)
        self.assertIn("leap", compiled.dump())

    @parametrize("optimizer", ("sgd", "adam"))
    def test_training_step_returns_a_compiled_autograd_function(self, optimizer):
        def train(compile_step):
            torch.manual_seed(0)
            model = RegionNet().train()
            if optimizer == "sgd":
                opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            else:
                opt = torch.optim.Adam(model.parameters(), lr=0.01)
            step = region_step
            if compile_step:
                step = torch.rewrite(
                    step, passes=with_graphs("aot_eager"), fallback=False
                )
            losses, nodes = [], []
            for _ in range(10):
                x, y = torch.randn(4, 3, 6, 6), torch.randint(0, 5, (4,))
                loss = step(model, opt, x, y)
                losses.append(loss.item())
                nodes.append(type(loss.grad_fn).__name__)
            return model.state_dict(), losses, nodes, step

        expected_state, expected_losses, _, _ = train(False)
        state, losses, nodes, step = train(True)
        self.assertEqual(losses, expected_losses)
        # Parameters, and the running statistics the forward updates in place.
        for name, value in state.items():
            self.assertTrue(torch.equal(value, expected_state[name]), name)
        self.assertEqual(state["bn.num_batches_tracked"].item(), 10)
        # Forward and loss became one graph whose backward was traced ahead
        # of time; the engine runs it as a single node.
        self.assertEqual(nodes[-1], "CompiledFunctionBackward")
        (region,) = self.compiled_regions(step)
        self.assertTrue(region["where"].startswith("region_step"))
        # What a graph cannot hold stayed with the VM, each for its reason.
        reasons = " / ".join(self.left_to_vm(step, "region_step"))
        self.assertIn("Tensor.backward has an effect", reasons)
        self.assertIn("a method of a", reasons)

    def test_branch_on_data_stays_a_branch(self):
        torch.manual_seed(0)
        model = RegionBranch().eval()
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        signs = set()
        with torch.no_grad():
            for i in range(24):
                x = torch.randn(2, 3, 6, 6) * (1 if i % 3 else -1)
                expected = region_infer(model, x)
                signs.add(bool(model.first(x).sum() > 0))
                self.assertEqual(compiled(model, x), expected, atol=0, rtol=0)
        self.assertEqual(signs, {True, False})
        # The call of the model does not trace, so the VM runs its forward,
        # where the branch stays a branch. The stretch before it is a graph
        # that ends at the branch; the one after it is a graph of its own,
        # and once more as the end of the path through the arm that doubles.
        self.assertEqual(len(self.left_to_vm(compiled, "region_infer")), 1)
        where = sorted(
            r["where"].split(" (")[0] for r in self.compiled_regions(compiled)
        )
        self.assertEqual(where, ["RegionBranch.forward"] * 3)

    def test_inputs_select_a_variant(self):
        graphs = []

        def backend(graph, inputs):
            graphs.append([i.dtype for i in inputs])
            return graph

        compiled = torch.rewrite(
            region_chain, passes=with_graphs(backend, max_variants=3), fallback=False
        )
        f32, f64, i64, i32 = torch.float32, torch.float64, torch.int64, torch.int32
        for dtype in (f32, f32, f32, f32, f64, f32, f64, i64, i32, i64, i32, f32):
            x, y = torch.ones(2, dtype=dtype), torch.full((2,), 3, dtype=dtype)
            self.assertEqual(compiled(x, y), region_chain(x, y), atol=0, rtol=0)
        # One graph per kind of input, up to the limit; the kinds after that
        # ran the instructions as they are.
        self.assertEqual(graphs, [[f32, f32], [f64, f64], [i64, i64]])
        (region,) = self.compiled_regions(compiled)
        self.assertEqual(region["variants"], 3)

    def test_layers_built_alike_share_a_graph(self):
        graphs = []

        def backend(graph, inputs):
            graphs.append(graph)
            return graph

        torch.manual_seed(0)
        model = RegionLayers().eval()
        compiled = torch.rewrite(
            region_infer, passes=with_graphs(backend), fallback=False
        )
        with torch.no_grad():
            for _ in range(8):
                x = torch.randn(3, 6)
                self.assertEqual(compiled(model, x), model(x), atol=0, rtol=0)
        # The call in the loop is a region with one variant per layer, whose
        # parameters are inputs of its graph: the five traced to one graph.
        (region,) = self.compiled_regions(compiled)
        self.assertTrue(region["where"].startswith("RegionLayers.forward"))
        self.assertEqual((region["variants"], region["compiles"]), (5, 1))
        self.assertEqual(len(graphs), 1)

    def test_limit_is_on_graphs_not_on_variants(self):
        graphs = []

        def backend(graph, inputs):
            graphs.append(graph)
            return graph

        torch.manual_seed(0)
        model = RegionLayers().eval()
        compiled = torch.rewrite(
            region_infer, passes=with_graphs(backend, max_variants=2), fallback=False
        )
        with torch.no_grad():
            for n in (3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 4, 9):
                x = torch.randn(n, 6)
                self.assertEqual(compiled(model, x), model(x), atol=0, rtol=0)
        # Five layers, each with a variant for three rows and one that leaves
        # the rows open: ten variants of two graphs, which is as many graphs
        # as the region may have.
        (region,) = self.compiled_regions(compiled)
        self.assertEqual((region["variants"], region["compiles"]), (10, 2))
        self.assertEqual(len(region["open"]), 5)
        self.assertEqual(len(graphs), 2)
        # A third graph would be one too many: these inputs run the layers
        # as they are.
        model.double()
        with torch.no_grad():
            x = torch.randn(4, 6, dtype=torch.float64)
            self.assertEqual(compiled(model, x), model(x), atol=0, rtol=0)
        self.assertEqual(len(graphs), 2)

    def test_same_tensor_twice_is_its_own_variant(self):
        graphs = []

        def backend(graph, inputs):
            graphs.append(len(inputs))
            return graph

        compiled = torch.rewrite(
            region_chain, passes=with_graphs(backend), fallback=False
        )
        for _ in range(5):
            x, y = torch.randn(3), torch.randn(3)
            self.assertEqual(compiled(x, y), region_chain(x, y), atol=0, rtol=0)
            self.assertEqual(compiled(x, x), region_chain(x, x), atol=0, rtol=0)
        self.assertEqual(sorted(graphs), [1, 2])

    def test_small_region_is_left_to_the_vm(self):
        compiled = torch.rewrite(
            region_small, passes=with_graphs("aot_eager"), fallback=False
        )
        for _ in range(6):
            x, y = torch.randn(3), torch.randn(3)
            self.assertEqual(compiled(x, y), x + y, atol=0, rtol=0)
        self.assertEqual(self.compiled_regions(compiled), [])
        (reason,) = self.left_to_vm(compiled, "region_small")
        self.assertIn("not worth a graph", reason)
        # The block is specialized as it is without a backend.
        self.assertNotIn("call_object", compiled.dump())
        self.assertGreater(compiled.stats()["sites"], 0)

    def test_module_changes_are_noticed(self):
        torch.manual_seed(0)
        model = RegionNet()
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )

        def check():
            for _ in range(3):
                x = torch.randn(2, 3, 6, 6)
                reference = copy.deepcopy(model)
                expected = reference(x)
                self.assertEqual(compiled(model, x), expected, atol=0, rtol=0)
                for name, value in reference.state_dict().items():
                    self.assertTrue(torch.equal(value, model.state_dict()[name]), name)

        with torch.no_grad():
            model.eval()
            check()
            check()
            # Train mode updates the running statistics: another graph.
            model.train()
            check()
            # One submodule back in eval mode.
            model.bn.eval()
            check()
            # A parameter replaced by another tensor is read on the next call.
            model.fc.weight = torch.nn.Parameter(torch.randn(5, 4 * 6 * 6))
            check()
            # A submodule replaced by one of another shape.
            model.fc = torch.nn.Linear(4 * 6 * 6, 7)
            check()
            # A parameter taken away, without a method of the module.
            model.fc._parameters["bias"] = None
            check()
        (region,) = self.compiled_regions(compiled)
        self.assertEqual(region["variants"], 5)

    def test_frozen_parameter(self):
        torch.manual_seed(0)
        model = RegionNet().train()
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        for i in range(8):
            if i == 5:
                model.conv.weight.requires_grad_(False)
            x = torch.randn(2, 3, 6, 6)
            model.zero_grad()
            compiled(model, x).sum().backward()
            self.assertEqual(model.conv.weight.grad is None, i >= 5)
            self.assertIsNotNone(model.fc.weight.grad)

    def test_effects_on_python_objects_stay_in_the_vm(self):
        torch.manual_seed(0)
        model = RegionNet().eval()
        compiled = torch.rewrite(
            region_logging, passes=with_graphs("aot_eager"), fallback=False
        )
        log, expected_log = [], []
        with torch.no_grad():
            for n in (2, 2, 2, 3, 2, 2, 4, 2):
                x = torch.randn(n, 3, 6, 6)
                expected = region_logging(model, x, expected_log)
                self.assertEqual(compiled(model, x, log), expected, atol=0, rtol=0)
                self.assertEqual(log, expected_log)
        self.assertEqual(len(self.compiled_regions(compiled)), 2)
        reasons = " / ".join(self.left_to_vm(compiled, "region_logging"))
        self.assertIn("changes a list from before the region", reasons)

    def test_module_that_sets_an_attribute(self):
        torch.manual_seed(0)
        model = RegionCounting().eval()
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        with torch.no_grad():
            for i in range(8):
                x = torch.randn(2, 3, 6, 6)
                self.assertEqual(compiled(model, x), model.inner(x), atol=0, rtol=0)
                self.assertEqual(model.calls, i + 1)
        (reason,) = self.left_to_vm(compiled, "region_infer")
        self.assertIn("__setattr__ on a module the region was given", reason)
        # The call below the store is a graph.
        (region,) = self.compiled_regions(compiled)
        self.assertTrue(region["where"].startswith("RegionCounting.forward"))

    def test_module_changed_without_one_of_its_methods(self):
        # A trace runs the code of the call on the module itself. What it
        # does to the module is taken back, and the call is not a region.
        torch.manual_seed(0)
        model = RegionCountingQuietly().eval()
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        with torch.no_grad():
            for i in range(8):
                x = torch.randn(2, 3, 6, 6)
                self.assertEqual(compiled(model, x), model.inner(x), atol=0, rtol=0)
                self.assertEqual(model.calls, i + 1)
        (reason,) = self.left_to_vm(compiled, "region_infer")
        self.assertIn("changes a module the region was given", reason)
        (region,) = self.compiled_regions(compiled)
        self.assertTrue(region["where"].startswith("RegionCountingQuietly.forward"))

    def test_nothing_of_nn_module_is_replaced(self):
        # A region notices a change of a module by looking at the module, not
        # by wrapping what changes it: torch.compile, for one, traces
        # Module.__setattr__ and has to find the one it knows.
        class MakesAModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return torch.nn.Softmax(dim=-1)(self.linear(x).sin().cos())

        def f(model, x):
            return model(x)

        before = {
            name: vars(torch.nn.Module)[name]
            for name in ("__setattr__", "__delattr__", "add_module", "register_buffer")
        }
        backward = vars(torch.Tensor)["backward"]
        model, x = MakesAModule(), torch.randn(2, 4)
        compiled = torch.rewrite(f, passes=with_graphs("aot_eager"), fallback=False)
        for _ in range(5):
            self.assertEqual(compiled(model, x), f(model, x), atol=0, rtol=0)
        self.assertEqual(len(self.compiled_regions(compiled)), 1)
        for name, function in before.items():
            self.assertIs(vars(torch.nn.Module)[name], function)
        self.assertIs(vars(torch.Tensor)["backward"], backward)
        self.assertNotIn("grad", vars(torch.Tensor))
        dynamo = torch.compile(f, backend="eager")
        self.assertEqual(dynamo(model, x), f(model, x))

    def test_module_with_a_hook(self):
        torch.manual_seed(0)
        model = RegionNet().eval()
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        seen = []
        with torch.no_grad():
            for i in range(10):
                if i == 4:
                    # After the call was compiled: noticed on the next one.
                    handle = model.bn.register_forward_hook(
                        lambda module, args, output: seen.append(output.shape[0])
                    )
                if i == 8:
                    handle.remove()
                x = torch.randn(2 + i % 2, 3, 6, 6)
                expected = model(x)
                self.assertEqual(compiled(model, x), expected, atol=0, rtol=0)
        # Once for eager and once for the compiled function, on every call
        # while the hook was there.
        self.assertEqual(seen, [2, 2, 3, 3, 2, 2, 3, 3])
        reasons = " / ".join(
            r["reason"] for r in torch.rewrite.regions(compiled) if r["state"] == "vm"
        )
        self.assertIn("a module with hooks", reasons)

    def test_no_grad_section_is_not_differentiated(self):
        torch.manual_seed(0)
        model, reference = RegionNet().train(), RegionNet().train()
        reference.load_state_dict(model.state_dict())
        compiled = torch.rewrite(
            region_frozen, passes=with_graphs("aot_eager"), fallback=False
        )
        for _ in range(8):
            x = torch.randn(2, 3, 6, 6)
            model.zero_grad()
            reference.zero_grad()
            loss, expected = compiled(model, x), region_frozen(reference, x)
            loss.backward()
            expected.backward()
            self.assertEqual(loss, expected, atol=0, rtol=0)
            self.assertEqual(
                model.fc.weight.grad, reference.fc.weight.grad, atol=0, rtol=0
            )
        # With gradients off and with them on are two regions of the block.
        self.assertEqual(len(self.compiled_regions(compiled)), 2)

    def test_loop_variable_before_its_first_assignment(self):
        # `out` and `total` do not exist on the first trip through the loop,
        # and the block holds them alive: a region must not take them in.
        torch.manual_seed(0)
        model = RegionNet().eval()
        compiled = torch.rewrite(
            region_loop, passes=with_graphs("aot_eager"), fallback=False
        )
        with torch.no_grad():
            for _ in range(8):
                batches = [torch.randn(2, 3, 6, 6) for _ in range(3)]
                expected = region_loop(model, batches)
                self.assertEqual(compiled(model, batches), expected, atol=0, rtol=0)
        self.assertGreaterEqual(len(self.compiled_regions(compiled)), 1)

    def test_graph_that_fails_on_its_first_run(self):
        calls = []

        def backend(graph, inputs):
            def broken(*args):
                calls.append(len(args))
                raise RuntimeError("no such kernel")

            return broken

        compiled = torch.rewrite(
            region_chain, passes=with_graphs(backend), fallback=False
        )
        for _ in range(6):
            x, y = torch.randn(3), torch.randn(3)
            self.assertEqual(compiled(x, y), region_chain(x, y), atol=0, rtol=0)
        # Tried once; the instructions ran instead, that time and since.
        self.assertEqual(calls, [2])
        (reason,) = self.left_to_vm(compiled, "region_chain")
        self.assertIn("the compiled graph failed: RuntimeError: no such kernel", reason)

    def test_training_step_of_a_recurrent_model(self):
        # Whatever the backend makes of the RNN, the step is the step.
        def train(compile_step):
            torch.manual_seed(0)
            model = RegionRecurrent().train()
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            step = region_step_plain
            if compile_step:
                step = torch.rewrite(
                    step, passes=with_graphs("aot_eager"), fallback=False
                )
            losses = [step(model, opt, torch.randn(5, 4, 6)).item() for _ in range(8)]
            return model.state_dict(), losses

        expected_state, expected_losses = train(False)
        state, losses = train(True)
        self.assertEqual(losses, expected_losses)
        for name, value in state.items():
            self.assertTrue(torch.equal(value, expected_state[name]), name)

    def test_error_in_a_region(self):
        model = RegionNet().eval()
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        with torch.no_grad():
            for shape in [(2, 3, 6, 6)] * 4 + [
                (2, 3, 5, 5),
                (2, 3, 6, 6),
                (2, 4, 6, 6),
            ]:
                x = torch.randn(shape)
                outcome, expected = (
                    call_outcome(compiled, model, x),
                    call_outcome(region_infer, model, x),
                )
                self.assertEqual(outcome[0], expected[0])
                if outcome[0] == "raised":
                    self.assertEqual(outcome[1:], expected[1:])

    def test_measurement_starts_over_when_a_region_is_compiled(self):
        torch.manual_seed(0)
        model = RegionNet().eval()
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False, demote=True
        )
        with torch.no_grad():
            x = torch.randn(2, 3, 6, 6)
            for _ in range(8):
                compiled(model, x)
            self.assertTrue(compiled.stats()["measuring"])
            for _ in range(400):
                compiled(model, x)
        self.assertFalse(compiled.stats()["measuring"])

    def test_object_made_by_a_region_is_made_on_every_call(self):
        torch.manual_seed(0)
        model = RegionStateful().eval()
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        results = []
        with torch.no_grad():
            for _ in range(6):
                x = torch.randn(2, 3, 6, 6)
                result, expected = compiled(model, x), model(x)
                self.assertIs(type(result), RegionOutput)
                self.assertEqual(result.logits, expected.logits, atol=0, rtol=0)
                self.assertEqual(result.hidden, expected.hidden, atol=0, rtol=0)
                self.assertEqual(result.name, "out")
                results.append(result)
        (region,) = self.compiled_regions(compiled)
        self.assertTrue(region["where"].startswith("region_infer"))
        # Not one object handed out again and again.
        self.assertEqual(len({id(r) for r in results}), 6)
        self.assertEqual(len({id(r.hidden) for r in results}), 6)

    def test_object_that_grows_on_every_call_is_left_to_the_vm(self):
        torch.manual_seed(0)
        model = RegionStateful().eval()
        compiled = torch.rewrite(
            region_cached, passes=with_graphs("aot_eager"), fallback=False
        )
        cache, expected_cache = RegionCache(), RegionCache()
        with torch.no_grad():
            for _ in range(14):
                x = torch.randn(2, 3, 6, 6)
                expected = region_cached(model, x, expected_cache)
                result = compiled(model, x, cache)
                self.assertEqual(result.logits, expected.logits, atol=0, rtol=0)
                # A trace that ran the update would have left its own tensors
                # in the cache, or left it one entry short on later calls.
                self.assertEqual(cache.seen, expected_cache.seen, atol=0, rtol=0)
        # A region may change an object it is given, but this one is never
        # what a graph was made for: its list has one more item every time.
        reasons = " / ".join(self.left_to_vm(compiled, "region_cached"))
        self.assertIn("changes what it is given on every call", reasons)
        # The module below the call that takes the cache is a graph.
        self.assertGreaterEqual(len(self.compiled_regions(compiled)), 1)

    def test_region_goes_on_into_the_blocks_after_it(self):
        graphs = []

        def backend(graph, inputs):
            output = next(n for n in graph.graph.nodes if n.op == "output")
            graphs.append(len(output.args[0]))
            return graph

        torch.manual_seed(0)
        model = RegionStages().eval()
        compiled = torch.rewrite(
            region_stages, passes=with_graphs(backend), fallback=False
        )
        log, expected_log = [], []
        with torch.no_grad():
            for i in range(14):
                if i == 8:
                    # Not an input of anything: read by the path, and checked.
                    model.doubled = True
                x = torch.randn(2, 6)
                out, doubled = compiled(model, x, log)
                expected, _ = model(x, expected_log)
                self.assertEqual(out, expected, atol=0, rtol=0)
                self.assertEqual(doubled, i >= 8)
        self.assertEqual(log, expected_log)
        # The call of the model reads a tensor's data, so the VM runs its
        # forward. There, stem, comprehension, loop, branch and head are four
        # blocks and more, and one graph: the path through them, up to the
        # conversion to a float.
        (region,) = self.compiled_regions(compiled)
        self.assertTrue(region["where"].startswith("RegionStages.forward (block 0)"))
        self.assertGreaterEqual(region["ops"], 12)
        ((block, _),) = region["ends"]
        self.assertGreater(block, 0)
        # One graph per value of the flag. Each gives back what is read after
        # it and nothing else: not the list of the stages' outputs, which the
        # function only keeps until it returns.
        self.assertEqual(graphs, [1, 1])

    def test_variable_shared_with_a_nested_function_is_an_input(self):
        # A local that a nested function uses as well lives in a cell.
        # Reading it does not end a region: the cell is an input, and what
        # it holds is part of what a graph is compiled for.
        graphs = []

        def backend(graph, inputs):
            calls = [n for n in graph.graph.nodes if n.op == "call_function"]
            graphs.append(len(calls))
            return graph

        def f(x, scale):
            def shared():
                return scale

            return (x.sin().cos() * scale).exp().tanh(), shared

        compiled = torch.rewrite(f, passes=with_graphs(backend), fallback=False)
        x = torch.randn(4)
        for scale in (2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0):
            out, shared = compiled(x, scale)
            self.assertEqual(out, f(x, scale)[0], atol=0, rtol=0)
            self.assertEqual(shared(), scale)
        self.assertEqual(graphs, [5, 5])

    def test_view_is_handed_on_as_a_reshape(self):
        # A trace records `reshape` as the view it was for the tensor at
        # hand. A backend that lays the tensor out otherwise could not take
        # the view; it can always take the reshape.
        targets = []

        def backend(graph, inputs):
            targets.extend(
                str(n.target) for n in graph.graph.nodes if n.op == "call_function"
            )
            return graph

        def f(x):
            return x.reshape(2, -1).sin().view(-1, 2).cos().exp().tanh()

        compiled = torch.rewrite(f, passes=with_graphs(backend), fallback=False)
        for _ in range(5):
            x = torch.randn(4, 3)
            self.assertEqual(compiled(x), f(x), atol=0, rtol=0)
        self.assertEqual(len(self.compiled_regions(compiled)), 1)
        self.assertEqual([t for t in targets if "view" in t], [])
        self.assertEqual(len([t for t in targets if "reshape" in t]), 2)

    def test_named_tuples_and_sets_of_values_are_inputs(self):
        compiled = torch.rewrite(
            region_pair, passes=with_graphs("aot_eager"), fallback=False
        )
        for i in range(12):
            pair = RegionPair(torch.randn(3), torch.randn(3))
            kinds = {"double"} if i % 2 else {"single", "plain"}
            result, expected = compiled(pair, kinds), region_pair(pair, kinds)
            self.assertIs(type(result), RegionPair)
            self.assertEqual(tuple(result), tuple(expected), atol=0, rtol=0)
        # One region, which goes on through the branch: a graph per set.
        (region,) = self.compiled_regions(compiled)
        self.assertEqual(region["variants"], 2)
        # Neither was turned away as something a trace might change.
        reasons = " / ".join(self.left_to_vm(compiled, "region_pair"))
        self.assertNotIn("from before the region", reasons)

    def test_rnn_keeps_its_weights_in_a_list(self):
        # An RNN rebuilds the list when a weight was replaced, which a trace
        # does to all of them: the list is put back afterwards.
        torch.manual_seed(0)
        model = RegionRecurrent().eval()
        weights = list(model.lstm._flat_weights)
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        with torch.no_grad():
            for _ in range(6):
                x = torch.randn(5, 4, 6)
                self.assertEqual(compiled(model, x), model(x), atol=0, rtol=0)
        (region,) = self.compiled_regions(compiled)
        self.assertTrue(region["where"].startswith("region_infer"))
        self.assertTrue(all(a is b for a, b in zip(model.lstm._flat_weights, weights)))

    def test_tensor_kept_as_a_plain_attribute(self):
        torch.manual_seed(0)
        model = RegionStateful().eval()
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        with torch.no_grad():
            for i in range(8):
                if i == 5:
                    model.scale = torch.tensor(-1.0)
                x = torch.randn(2, 3, 6, 6)
                self.assertEqual(
                    compiled(model, x).logits, model(x).logits, atol=0, rtol=0
                )
        # The attribute is an input of the graph, not a constant in it.
        (region,) = self.compiled_regions(compiled)
        self.assertEqual(region["variants"], 1)

    def test_tensor_that_is_not_an_input_is_not_a_constant(self):
        torch.manual_seed(0)
        model = RegionNet().eval()
        compiled = torch.rewrite(
            region_global, passes=with_graphs("aot_eager"), fallback=False
        )
        with torch.no_grad():
            for i in range(8):
                if i == 5:
                    REGION_CONSTANT.fill_(0.5)
                x = torch.randn(2, 3, 6, 6)
                self.assertEqual(
                    compiled(model, x), region_global(model, x), atol=0, rtol=0
                )
        REGION_CONSTANT.fill_(3.0)
        # The VM reads the global, which makes the tensor an input of the
        # region that follows.
        (reason,) = self.left_to_vm(compiled, "region_global")
        self.assertIn("gives a tensor from outside the region", reason)
        self.assertEqual(len(self.compiled_regions(compiled)), 2)

    def test_values_that_keep_changing_do_not_multiply_graphs(self):
        graphs = []

        def backend(graph, inputs):
            graphs.append(1)
            return graph

        compiled = torch.rewrite(
            region_counter, passes=with_graphs(backend), fallback=False
        )
        x = torch.randn(4)
        for i in range(40):
            scale = i / 7
            self.assertEqual(
                compiled(x, scale), region_counter(x, scale), atol=0, rtol=0
            )
        self.assertLessEqual(len(graphs), 3)
        self.assertEqual(self.compiled_regions(compiled), [])
        (reason,) = self.left_to_vm(compiled, "region_counter")
        self.assertIn("ever new values", reason)

    def test_call_without_tensors_is_not_frozen(self):
        torch.manual_seed(0)
        model = RegionNet().eval()
        compiled = torch.rewrite(
            region_clock, passes=with_graphs("aot_eager"), fallback=False
        )
        starts = []
        with torch.no_grad():
            for _ in range(8):
                x = torch.randn(2, 3, 6, 6)
                out, start = compiled(model, x)
                self.assertEqual(out, model(x), atol=0, rtol=0)
                starts.append(start)
        self.assertEqual(sorted(set(starts)), starts)
        self.assertEqual(len(self.compiled_regions(compiled)), 1)

    def test_python_random_is_not_frozen(self):
        torch.manual_seed(0)
        model = RegionRandom().eval()
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        with torch.no_grad():
            x = torch.randn(2, 3, 6, 6)
            outputs = []
            for seed in range(8):
                random.seed(seed)
                expected = model(x)
                random.seed(seed)
                outputs.append(compiled(model, x))
                self.assertEqual(outputs[-1], expected, atol=0, rtol=0)
        self.assertGreater(len({float(o.sum()) for o in outputs}), 6)
        reasons = " / ".join(self.left_to_vm(compiled, "region_infer"))
        self.assertIn("draws from Python's random module", reasons)

    def test_library_code_around_tensors_gets_no_regions(self):
        torch.manual_seed(0)
        model = RegionNet().train()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        step = torch.rewrite(
            region_step, passes=with_graphs("aot_eager"), fallback=False
        )
        for _ in range(8):
            step(model, opt, torch.randn(4, 3, 6, 6), torch.randint(0, 5, (4,)))
        callees = step.stats()["compiled_callees"]
        self.assertIn("Adam.step", callees)
        where = {r["where"].split(" (")[0] for r in torch.rewrite.regions(step)}
        self.assertIn("region_step", where)
        # The optimizer is compiled by the VM, but nothing in it is a region.
        plumbing = ("Adam", "Optimizer", "_single_tensor", "_multi_tensor", "adam")
        self.assertEqual([w for w in where if w.startswith(plumbing)], [])

    def test_a_graph_is_made_for_the_default_device_it_runs_under(self):
        def made(x):
            return ((x * 2 + 1) * x - 3) / 2

        graphs = []
        passes = with_graphs(self.counting(graphs))
        compiled = torch.rewrite(made, passes=passes, fallback=False)
        x = torch.arange(3.0)
        for _ in range(8):
            self.assertEqual(compiled(x), made(x))
        self.assertEqual(len(graphs), 1)
        # What a default device does is built into a graph: there is one
        # graph for running under it and one for running without.
        with torch.device("cpu"):
            for _ in range(4):
                self.assertEqual(compiled(x), made(x))
        self.assertEqual(len(graphs), 2)
        for _ in range(3):
            self.assertEqual(compiled(x), made(x))
        self.assertEqual(len(graphs), 2)

    def test_a_graph_is_called_below_a_default_device(self):
        seen = []
        handle = DeviceContext.__torch_function__

        def counting(self, func, types, args=(), kwargs=None):
            seen.append(func)
            return handle(self, func, types, args, kwargs)

        def run(gm, example_inputs):
            # A backend whose code calls torch functions, as Inductor's does.
            return lambda *args: [torch.add(out, 0) for out in gm(*args)]

        model = RegionNet().eval()
        compiled = torch.rewrite(region_infer, passes=with_graphs(run), fallback=False)
        x = torch.randn(2, 3, 6, 6)
        with torch.no_grad(), torch.device("cpu"):
            for _ in range(8):
                self.assertEqual(compiled(model, x), region_infer(model, x))
            self.assertEqual(len(self.compiled_regions(compiled)), 1)
            with unittest.mock.patch.object(
                DeviceContext, "__torch_function__", counting
            ):
                out = compiled(model, x)
            self.assertEqual(out, region_infer(model, x))
            stack = torch.overrides._get_current_function_mode_stack()
            self.assertEqual([type(mode) for mode in stack], [DeviceContext])
        # The mode is not called for what the backend's code does.
        self.assertEqual(seen, [])

    def looks(self, compiled, *args):
        """How many tensors three calls each make their key of, and check
        against what a graph has built in."""
        keyed, stamped = [], []
        real_key, real_stamp = cfg_rs.tensor_key, cfg_rs.tensor_stamp

        def tensor_key(tensors, ints):
            keyed.append(len(tensors))
            return real_key(tensors, ints)

        def tensor_stamp(tensors):
            stamped.append(len(tensors))
            return real_stamp(tensors)

        counted = unittest.mock.patch.multiple(
            cfg_rs, tensor_key=tensor_key, tensor_stamp=tensor_stamp
        )
        with counted:
            for _ in range(3):
                compiled(*args)
        return keyed, stamped

    def test_the_tensors_of_a_module_are_looked_at_once_a_call(self):
        torch.manual_seed(0)
        model = RegionNet().eval()
        passes = with_graphs("aot_eager")
        compiled = torch.rewrite(region_infer, passes=passes, fallback=False)
        x = torch.randn(2, 3, 6, 6)
        with torch.no_grad():
            for _ in range(10):
                compiled(model, x)
            # The key of a call is made of its input: what the module holds
            # is what an earlier call found, and so is what the graph has
            # built in.
            self.assertEqual(self.looks(compiled, model, x), ([1, 1, 1], []))
            # A weight that is written to, replaced, or of another size is
            # found out all the same.
            model.fc.weight.mul_(2)
            self.assertEqual(compiled(model, x), region_infer(model, x))
            model.fc.bias = torch.nn.Parameter(torch.ones(5))
            self.assertEqual(compiled(model, x), region_infer(model, x))
            model.fc = torch.nn.Linear(4 * 6 * 6, 7)
            for _ in range(3):
                self.assertEqual(compiled(model, x), region_infer(model, x))
            model.bn.train()
            for _ in range(3):
                self.assertEqual(compiled(model, x), region_infer(model, x))

    def test_modules_that_share_tensors_or_lie_inside_one_another(self):
        torch.manual_seed(0)
        first, second = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
        second.weight = first.weight
        both = torch.nn.Sequential(first, second).eval()

        def outside_in(both, second, x):
            return both(x) + second(x) * 2 - x

        def inside_out(second, both, x):
            return both(x) + second(x) * 2 - x

        passes = with_graphs("aot_eager")
        x = torch.randn(3, 4)
        with torch.no_grad():
            for f, args in (
                (outside_in, (both, second, x)),
                (inside_out, (second, both, x)),
            ):
                compiled = torch.rewrite(f, passes=passes, fallback=False)
                for _ in range(10):
                    self.assertEqual(compiled(*args), f(*args))
                self.assertEqual(len(self.compiled_regions(compiled)), 1)
                # One module inside the other, and a weight that both layers
                # have: every tensor is still looked at once.
                self.assertEqual(self.looks(compiled, *args), ([1, 1, 1], []))
                first.weight.add_(1)
                self.assertEqual(compiled(*args), f(*args))
                second.bias = torch.nn.Parameter(torch.ones(4))
                self.assertEqual(compiled(*args), f(*args))

    def test_a_trace_does_not_use_up_an_iterator_of_the_call(self):
        def made(ks, x):
            y = x * 2 + 1
            gen = (k + 1 for k in ks)
            t = tuple(gen)
            return t, y

        def advanced(ks, x):
            y = x * 2 + 1
            it = iter(ks)
            first = next(it)
            rest = list(it)
            return first, rest, y

        x = torch.ones(3)
        for fn in (made, advanced):
            compiled = torch.rewrite(
                fn, passes=with_graphs("aot_eager"), fallback=False
            )
            # On every call, also the ones on which a region is traced: the
            # trace has the generator itself, which it cannot copy, and must
            # not run what takes from it.
            for _ in range(8):
                self.assertEqual(compiled([1, 2], x), fn([1, 2], x))

    def fake_inputs(self, seen):
        """A backend that notes whether it is given tensors of a trace that
        has a ShapeEnv, and runs the graph as it is."""
        from torch._guards import detect_fake_mode

        def backend(graph, inputs):
            mode = detect_fake_mode(inputs)
            seen.append(mode is not None and mode.shape_env is not None)
            return graph

        return backend

    def test_a_size_that_depends_on_data_stays_in_the_graph(self):
        def positive(x, y):
            kept = x[x > 0]
            return (kept * 2).sum() + y.sum(), kept.shape[0]

        def between(x, y):
            kept = x[x > 0] * 2 + 1
            first = kept.sum() + y.sum()
            time.perf_counter()
            more = y[y > 0] * 3 + 1
            return first + more.sum()

        torch.manual_seed(0)
        for fn, regions in ((positive, 1), (between, 2)):
            seen = []
            passes = with_graphs(self.fake_inputs(seen), min_ops=1)
            compiled = torch.rewrite(fn, passes=passes, fallback=False)
            # How many elements are positive is another number on every
            # call: one graph is for all of them.
            for _ in range(8):
                x, y = torch.randn(64), torch.randn(64)
                self.assertEqual(compiled(x, y), fn(x, y))
            self.assertEqual(len(self.compiled_regions(compiled)), regions)
            # The size of what a mask picks is a symbol of the trace. The
            # backend is handed the tensors of that trace, so that the
            # symbol is one of a ShapeEnv it has too.
            self.assertEqual(seen, [True] * regions)

    @slowTest
    def test_inductor_compiles_a_size_that_depends_on_data(self):
        def positive(x, y):
            kept = x[x > 0]
            return (kept * 2).sum() + y.sum(), kept.shape[0]

        torch.manual_seed(0)
        compiled = torch.rewrite(positive, passes=with_graphs("inductor", min_ops=1))
        for _ in range(8):
            x, y = torch.randn(64), torch.randn(64)
            self.assertEqual(compiled(x, y), positive(x, y))
        self.assertEqual(self.left_to_vm(compiled, "positive"), [])
        self.assertEqual(len(self.compiled_regions(compiled)), 1)

    def test_an_object_the_region_made_is_made_again_from_its_attributes(self):
        def made(x):
            one = RegionMade(x * 2, x + 1)
            one.extra = one.both.sum()
            return one

        def cached(x):
            return RegionCached(x * 2 + 1)

        def output(x):
            return RegionItems(logits=x * 2 + 1, loss=None)

        x = torch.arange(4.0)
        passes = with_graphs("aot_eager", min_ops=1)
        compiled = torch.rewrite(made, passes=passes, fallback=False)
        for _ in range(8):
            compiled(x)
        self.assertEqual(len(self.compiled_regions(compiled)), 1)
        # A pytree node is not made from what it flattens to: that would run
        # its __init__ on every call, and lose what was set afterwards.
        before = RegionMade.made
        for _ in range(3):
            got = compiled(x)
        self.assertEqual(RegionMade.made, before)
        self.assertIs(type(got), RegionMade)
        self.assertEqual(vars(got), vars(made(x)))

        # What a class says of its copies is not about the object it made.
        compiled = torch.rewrite(cached, passes=passes, fallback=False)
        for _ in range(8):
            got = compiled(x)
        self.assertEqual(len(self.compiled_regions(compiled)), 1)
        self.assertIs(type(got), RegionCached)
        self.assertEqual(vars(got), vars(cached(x)))

        # A dict of a class of its own: its items, in their order, and its
        # attributes.
        compiled = torch.rewrite(output, passes=passes, fallback=False)
        for _ in range(8):
            compiled(x)
        self.assertEqual(len(self.compiled_regions(compiled)), 1)
        before = RegionItems.made
        for _ in range(3):
            got = compiled(x)
        self.assertEqual(RegionItems.made, before)
        expected = output(x)
        self.assertIs(type(got), RegionItems)
        self.assertEqual(list(got.items()), list(expected.items()))
        self.assertEqual(vars(got), vars(expected))

    def test_the_dicts_of_a_module_are_walked_when_one_changed(self):
        torch.rewrite._load_runtime()
        torch.manual_seed(0)
        model = RegionNet().eval()
        types = (torch.Tensor, torch.nn.Parameter)
        stamp, epoch = cfg_rs.module_look(model, types, True)
        self.assertEqual(stamp, cfg_rs.module_stamp(model, types))
        self.assertEqual(cfg_rs.module_epoch(), epoch)
        # What is written to a tensor is in no dict.
        with torch.no_grad():
            model.fc.weight.mul_(2)
        self.assertEqual(cfg_rs.module_epoch(), epoch)
        changes = [
            lambda: setattr(model, "note", 1),
            lambda: model.bn.train(),
            lambda: model.register_buffer("more", torch.ones(1)),
            lambda: setattr(model.fc, "bias", torch.nn.Parameter(torch.ones(5))),
            lambda: model.fc.register_forward_hook(lambda *args: None),
            lambda: delattr(model, "note"),
            lambda: model.add_module("extra", torch.nn.ReLU()),
        ]
        for change in changes:
            before = cfg_rs.module_epoch()
            change()
            self.assertNotEqual(cfg_rs.module_epoch(), before)
            # The dicts that are new are watched from the next walk on.
            self.assertEqual(
                cfg_rs.module_look(model, types, True)[1], cfg_rs.module_epoch()
            )
        # Nothing is promised for dicts that are not watched, nor for a
        # module that keeps tensors in a list, which nothing watches.
        self.assertIsNone(cfg_rs.module_look(torch.nn.Linear(2, 2), types, False)[1])
        self.assertIsNone(cfg_rs.module_look(torch.nn.LSTM(2, 2), types, True)[1])

        # A region walks the dicts of the module it is handed again when
        # one of them changed, and not on the calls in between.
        model = RegionNet().eval()
        x = torch.randn(2, 3, 6, 6)
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        walks = []
        real = cfg_rs.module_look

        def module_look(*args):
            walks.append(args[0])
            return real(*args)

        with unittest.mock.patch.object(cfg_rs, "module_look", module_look):
            with torch.no_grad():
                for _ in range(10):
                    compiled(model, x)
                del walks[:]
                for _ in range(3):
                    self.assertEqual(compiled(model, x), region_infer(model, x))
                self.assertEqual(walks, [])
                model.note = 1
                for _ in range(3):
                    self.assertEqual(compiled(model, x), region_infer(model, x))
                self.assertEqual(walks, [model])

    def test_a_snapshot_holds_in_place(self):
        torch.rewrite._load_runtime()
        model = torch.nn.Linear(3, 2)
        places = [(model._parameters, "weight"), (model._parameters, "bias")]
        tensors = [model.weight, model.bias]
        snapshot = cfg_rs.snapshot(places, tensors)
        # What holds says, for tensors that are known to be at their places.
        self.assertEqual((snapshot.holds(places), snapshot.holds_in_place()), (2, 2))
        with torch.no_grad():
            model.weight.mul_(2)
        self.assertEqual((snapshot.holds(places), snapshot.holds_in_place()), (1, 1))
        model.bias.requires_grad_(False)
        self.assertEqual((snapshot.holds(places), snapshot.holds_in_place()), (0, 0))
        # That a place holds another tensor is what only a look at it finds.
        snapshot = cfg_rs.snapshot(places, tensors)
        model.bias = torch.nn.Parameter(torch.zeros(2))
        self.assertEqual((snapshot.holds(places), snapshot.holds_in_place()), (0, 2))

    def test_an_unknown_backend_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "unknown backend"):
            torch.rewrite(region_infer, passes=with_graphs("nope"))

    def counting(self, graphs):
        from torch._dynamo.backends.debugging import aot_eager

        def backend(graph, inputs):
            graphs.append(graph)
            return aot_eager(graph, inputs)

        return backend

    # --- what a region is given and changes

    def test_region_changes_an_object_it_is_given(self):
        torch.manual_seed(0)
        model = RegionLinears().eval()
        graphs = []
        compiled = torch.rewrite(
            region_running, passes=with_graphs(self.counting(graphs)), fallback=False
        )
        stats, expected_stats = RegionAverage(), RegionAverage()
        with torch.no_grad():
            for _ in range(10):
                x = torch.randn(4, 6)
                expected = region_running(model, x, expected_stats)
                self.assertEqual(compiled(model, x, stats), expected, atol=1e-6, rtol=0)
                # What the graph computed for the attribute, stored in the
                # object itself once the graph has run.
                self.assertEqual(stats.mean, expected_stats.mean, atol=1e-6, rtol=0)
                self.assertIs(type(stats.mean), torch.Tensor)
        (region,) = self.compiled_regions(compiled)
        self.assertTrue(region["where"].startswith("region_running"))
        self.assertEqual(len(graphs), 1)

    def test_region_stores_into_a_dict_it_is_given(self):
        torch.manual_seed(0)
        model = RegionLinears().eval()
        compiled = torch.rewrite(
            region_remember, passes=with_graphs("aot_eager"), fallback=False
        )
        memory, expected_memory = {}, {}
        with torch.no_grad():
            for _ in range(8):
                x = torch.randn(4, 6)
                expected = region_remember(model, x, expected_memory)
                self.assertEqual(compiled(model, x, memory), expected, atol=0, rtol=0)
                self.assertEqual(list(memory), list(expected_memory))
                self.assertEqual(
                    memory["last"], expected_memory["last"], atol=0, rtol=0
                )
                self.assertEqual(
                    memory["sum"], expected_memory["sum"], atol=1e-5, rtol=0
                )
        (region,) = self.compiled_regions(compiled)
        self.assertTrue(region["where"].startswith("region_remember"))

    def test_object_made_in_a_region_is_made_on_every_call(self):
        torch.manual_seed(0)
        model = RegionStarts().eval()
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        states = []
        with torch.no_grad():
            for _ in range(8):
                x = torch.randn(4, 6)
                out, state = compiled(model, x)
                self.assertEqual(out, model.net(x), atol=0, rtol=0)
                # Not the object of an earlier call, with what was done to it.
                self.assertIs(type(state), RegionState)
                self.assertEqual(state.steps, [])
                state.steps.append(out)
                states.append(state)
        self.assertEqual(len({id(state) for state in states}), 8)
        self.assertEqual(len({id(state.steps) for state in states}), 8)
        (region,) = self.compiled_regions(compiled)
        self.assertTrue(region["where"].startswith("region_infer"))

    def test_module_keeps_a_tensor_it_computes(self):
        torch.manual_seed(0)
        model, reference = RegionMasked().eval(), RegionMasked().eval()
        reference.load_state_dict(model.state_dict())
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        with torch.no_grad():
            for n in (4, 4, 4, 4, 4, 5, 6, 3, 4):
                x = torch.randn(n, 6)
                self.assertEqual(compiled(model, x), reference(x), atol=0, rtol=0)
                # What the graph computed for the attribute is stored in it.
                self.assertIs(type(model.mask), torch.Tensor)
                self.assertEqual(model.mask, reference.mask, atol=0, rtol=0)
        (region,) = self.compiled_regions(compiled)
        self.assertTrue(region["where"].startswith("region_infer"))
        self.assertEqual(self.left_to_vm(compiled, "RegionMasked"), [])

    def test_path_ends_before_a_call_that_fails_part_way(self):
        torch.manual_seed(0)
        model = RegionLinears().eval()
        compiled = torch.rewrite(
            region_partway, passes=with_graphs("aot_eager"), fallback=False
        )
        log, expected_log = [], []
        with torch.no_grad():
            for n in (4, 4, 4, 4, 4, 4, 5, 6):
                x = torch.randn(n, 6)
                expected_x = x.clone()
                expected = region_partway(model, expected_x, expected_log, region_note)
                out, total = compiled(model, x, log, region_note)
                self.assertEqual(out, expected[0], atol=0, rtol=0)
                self.assertEqual(total, expected[1])
                # The trace ran the append and the addition and then could
                # not go on. The VM runs the call, and the region does not
                # do the two as well.
                self.assertEqual(log, expected_log)
                self.assertEqual(x, expected_x, atol=0, rtol=0)
        # What came before the call is still a graph.
        where = [r["where"] for r in self.compiled_regions(compiled)]
        self.assertTrue(any(w.startswith("region_partway") for w in where), where)

    def test_size_that_varies_is_left_open(self):
        torch.manual_seed(0)
        model = RegionLinears().eval()
        graphs = []
        compiled = torch.rewrite(
            region_sized, passes=with_graphs(self.counting(graphs)), fallback=False
        )
        with torch.no_grad():
            for n in (4, 4, 4, 4, 5, 6, 7, 1, 9, 4, 12, 1, 300):
                x = torch.randn(n, 6)
                expected, _ = region_sized(model, x)
                mean, doubled = compiled(model, x)
                self.assertEqual(mean, expected, atol=1e-6, rtol=0)
                # Computed by the graph from the size, and a plain int again.
                self.assertIs(type(doubled), int)
                self.assertEqual(doubled, 2 * n)
        # One graph for the size of the first calls, one for every size from
        # 2 on, and one for a single row, which is not like the others.
        self.assertEqual(len(graphs), 3)
        (region,) = self.compiled_regions(compiled)
        self.assertEqual(region["variants"], 3)
        (conditions,) = region["open"]
        self.assertTrue(any(".size()[0]" in c for c in conditions), conditions)

    def test_branch_on_a_size_is_a_condition_of_the_variant(self):
        torch.manual_seed(0)
        model = RegionLinears().eval()
        graphs = []
        compiled = torch.rewrite(
            region_branching, passes=with_graphs(self.counting(graphs)), fallback=False
        )
        with torch.no_grad():
            for n in (2, 2, 2, 2, 3, 6, 8, 3, 4, 5, 2, 40):
                x = torch.randn(n, 6)
                expected = region_branching(model, x)
                self.assertEqual(compiled(model, x), expected, atol=0, rtol=0)
        # One for two rows; then one per way the branch goes.
        self.assertEqual(len(graphs), 3)
        (region,) = self.compiled_regions(compiled)
        below, above = region["open"]
        self.assertNotEqual(below, above)

    def test_integer_that_varies_is_an_input_of_the_graph(self):
        graphs = []
        compiled = torch.rewrite(
            region_counter, passes=with_graphs(self.counting(graphs)), fallback=False
        )
        x = torch.randn(4)
        for i in range(40):
            self.assertEqual(compiled(x, i), region_counter(x, i), atol=1e-6, rtol=0)
        self.assertLessEqual(len(graphs), 2)
        (region,) = self.compiled_regions(compiled)
        self.assertEqual(len(region["open"]), 1)

    def test_integer_kept_in_an_object_is_an_input_of_the_graph(self):
        torch.manual_seed(0)
        model = RegionLinears().eval()
        graphs = []
        compiled = torch.rewrite(
            region_running, passes=with_graphs(self.counting(graphs)), fallback=False
        )
        stats, expected_stats = RegionRunningMean(), RegionRunningMean()
        with torch.no_grad():
            for _ in range(10):
                x = torch.randn(4, 6)
                expected = region_running(model, x, expected_stats)
                self.assertEqual(compiled(model, x, stats), expected, atol=1e-6, rtol=0)
                self.assertEqual(stats.mean, expected_stats.mean, atol=1e-6, rtol=0)
        # The count is another integer on every call: the graph takes it,
        # and what the graph makes of it is stored as a plain int.
        self.assertEqual(stats.count, 10)
        self.assertIs(type(stats.count), int)
        (region,) = self.compiled_regions(compiled)
        self.assertTrue(region["where"].startswith("region_running"))
        self.assertLessEqual(len(graphs), 3)

    def test_code_that_asks_what_an_integer_is(self):
        torch.manual_seed(0)
        calls = [(4, n) for n in (1, 1, 1, 1, 2, 3, 4, 5, 6, 7)]
        self.assertEqual(len(self.asked("isinstance", calls)), 1)
        self.assertEqual(self.asked("type", calls), [])
        self.assertEqual(self.asked("text", calls), [])

    def test_integer_that_picks_a_layer_is_not_left_open(self):
        torch.manual_seed(0)
        model = RegionLinears().eval()
        graphs = []
        compiled = torch.rewrite(
            region_pick, passes=with_graphs(self.counting(graphs)), fallback=False
        )
        with torch.no_grad():
            for step in range(30):
                x, i = torch.randn(3, 6), step % 5
                self.assertEqual(
                    compiled(model, x, i), region_pick(model, x, i), atol=0, rtol=0
                )
        # An index is the number it is: no graph is good for another one,
        # and there is no end to them.
        self.assertLessEqual(len(graphs), 4)
        reasons = " / ".join(self.left_to_vm(compiled, "region_pick"))
        self.assertIn("ever new values", reasons)

    def asked(self, how, calls):
        """Runs a model that asks what a number is, and returns the
        conditions of the open variants of the region around its call."""
        model = RegionAsks(how).eval()
        compiled = torch.rewrite(
            region_asked, passes=with_graphs("aot_eager"), fallback=False
        )
        with torch.no_grad():
            for rows, n in calls:
                x = torch.randn(rows, 6)
                self.assertEqual(compiled(model, x, n), model(x, n), msg=how)
        return [
            conditions
            for region in self.compiled_regions(compiled)
            if region["where"].startswith("region_asked")
            for conditions in region["open"]
        ]

    def test_code_that_asks_what_a_size_is(self):
        torch.manual_seed(0)
        calls = [(rows, None) for rows in (2, 2, 2, 2, 3, 4, 5, 6, 7, 3)]
        # isinstance is told what it will be told when the code runs for
        # real, and the call of the model is one graph for all the sizes.
        self.assertEqual(len(self.asked("isinstance", calls)), 1)
        # type() and str() cannot be told: for them the size stays the
        # number it is, and every size has a graph of its own.
        self.assertEqual(self.asked("type", calls), [])
        self.assertEqual(self.asked("text", calls), [])

    def test_text_of_a_size(self):
        torch.manual_seed(0)
        model = RegionNames().eval()
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        with torch.no_grad():
            for n in (4, 4, 4, 4, 5, 6, 7, 5):
                x = torch.randn(n, 6)
                result = compiled(model, x)
                self.assertEqual(list(result), [f"rows_{n}"])
                self.assertEqual(result[f"rows_{n}"], model.net(x), atol=0, rtol=0)
        (region,) = self.compiled_regions(compiled)
        self.assertTrue(region["where"].startswith("region_infer"))
        self.assertEqual(region["open"], [])

    def test_strides_select_a_variant(self):
        compiled = torch.rewrite(
            region_chain, passes=with_graphs("aot_eager"), fallback=False
        )
        for _ in range(6):
            x, y = torch.randn(4, 4), torch.randn(4, 4)
            self.assertEqual(compiled(x, y), region_chain(x, y), atol=0, rtol=0)
            # The same sizes, laid out the other way.
            x = x.t()
            self.assertEqual(compiled(x, y), region_chain(x, y), atol=0, rtol=0)
        (region,) = self.compiled_regions(compiled)
        self.assertEqual(region["variants"], 2)

    def test_numbers_of_a_call(self):
        runtime = importlib.import_module("torch._cfg_rs")
        torch.rewrite(region_small)
        a, b = torch.randn(4, 12, 8), torch.nn.Parameter(torch.randn(3))
        view = a[:, 2:9]
        kinds, numbers, offsets = runtime.tensor_key([a, view, b], [5, -2])
        # The sizes and then the strides of each tensor, then the integers.
        self.assertEqual(
            memoryview(numbers).cast("q").tolist(),
            [4, 12, 8, 96, 8, 1, 4, 7, 8, 96, 8, 1, 3, 1, 5, -2],
        )
        self.assertEqual(memoryview(offsets).cast("q").tolist(), [0, 16, 0])
        # Other elements, or gradients, make another kind of tensor of the
        # same numbers.
        for other in (a.double(), a.clone().requires_grad_()):
            key = runtime.tensor_key([other, view, b], [5, -2])
            self.assertNotEqual(kinds, key[0])
            self.assertEqual(numbers, key[1])
        with self.assertRaises(TypeError):
            runtime.tensor_key([a, 3], [])

    def test_condition_on_the_numbers_of_a_call(self):
        runtime = importlib.import_module("torch._cfg_rs")
        torch.rewrite(region_small)
        a, b = torch.randn(4, 12, 8), torch.nn.Parameter(torch.randn(3))
        view = a[:, 2:9]
        _, numbers, offsets = runtime.tensor_key([a, view, b], [5, -2])
        # Open where the first tensor has 12, and what follows from it: 12
        # may be anything from 2 to 100 that is even.
        size = ("slot", 1)
        conditions = [
            ("and", [("le", ("const", 2), size), ("le", size, ("const", 100))]),
            ("eq", ("mod", size, ("const", 2)), ("const", 0)),
            ("eq", ("slot", 3), ("mul", ("const", 8), size)),
        ]
        guard = runtime.ShapeGuard(numbers, offsets, [1, 3], conditions)
        self.assertEqual(guard.open, 2)

        def holds(rows, tensors=None):
            c = torch.randn(4, rows, 8)
            key = runtime.tensor_key(tensors or [c, view, b], [5, -2])
            return guard.holds(key[1], key[2])

        self.assertTrue(holds(12))
        self.assertTrue(holds(30))
        self.assertFalse(holds(31))
        self.assertFalse(holds(102))
        self.assertFalse(holds(1))
        # A number that is not open is another one.
        self.assertFalse(holds(30, [torch.randn(5, 30, 8), view, b]))
        # Where a tensor starts in its storage is no part of the numbers,
        # but a condition may speak of it: it comes after them.
        self.assertTrue(holds(30, [torch.randn(4, 30, 8), a[:, 1:8], b]))
        conditions.append(("le", ("const", 2), ("slot", 16 + 1)))
        guard = runtime.ShapeGuard(numbers, offsets, [1, 3], conditions)
        self.assertTrue(holds(30))
        self.assertFalse(holds(30, [torch.randn(4, 30, 8), a[:, 0:7], b]))
        with self.assertRaises(ValueError):
            runtime.ShapeGuard(numbers, offsets, [], [("slot", 99)])
        with self.assertRaises(ValueError):
            runtime.ShapeGuard(numbers, offsets, [], [("power", size, size)])

    def test_stamp_of_tensors(self):
        stamp = importlib.import_module("torch._cfg_rs").tensor_stamp
        a, b = torch.nn.Parameter(torch.randn(3)), torch.randn(3)
        first = stamp([a, b])
        self.assertEqual(stamp([a, b]), first)
        self.assertNotEqual(stamp([b, a]), first)
        self.assertNotEqual(stamp([a, b.clone()]), first)
        b.add_(1)
        second = stamp([a, b])
        self.assertNotEqual(second, first)
        # Not an operation on the tensor: other memory.
        a.data = torch.randn(3)
        self.assertNotEqual(stamp([a, b]), second)
        with torch.inference_mode():
            unversioned = torch.randn(3)
        self.assertIsNone(stamp([a, unversioned]))

    def test_product_with_one_row_becomes_a_reduction(self):
        from torch.fx.experimental.proxy_tensor import make_fx

        regions = torch.rewrite._regions
        aten = torch.ops.aten
        row, matrix, bias = torch.randn(1, 640), torch.randn(640, 512), torch.randn(512)

        def products(row, matrix, bias):
            wide = torch.addmm(bias, row, matrix)
            return wide, torch.mm(row, matrix), torch.mm(matrix[:2], matrix.t()[:, :2])

        graph = make_fx(products)(row, matrix, bias)
        regions._rows_to_reductions(graph)
        targets = [node.target for node in graph.graph.nodes]
        # The product of two rows is left as it is.
        self.assertEqual(targets.count(aten.mm.default), 1)
        self.assertNotIn(aten.addmm.default, targets)
        self.assertEqual(targets.count(aten.sum.dim_IntList), 2)
        for got, expected in zip(graph(row, matrix, bias), products(row, matrix, bias)):
            self.assertEqual(got, expected, atol=1e-4, rtol=1e-4)

        # A matrix BLAS is through with before threads would pay.
        small = make_fx(products)(row[:, :64], matrix[:64], bias)
        regions._rows_to_reductions(small)
        self.assertNotIn(aten.sum.dim_IntList, [n.target for n in small.graph.nodes])

    def building_in(self):
        """Has ``aot_eager`` regions build weights in, as Inductor ones do."""
        regions = torch.rewrite._regions
        return unittest.mock.patch.dict(
            regions._BUILDS_IN, {regions._aot_eager: regions._aot_eager}
        )

    def test_weights_are_built_into_the_graph_of_an_inference_call(self):
        torch.manual_seed(0)
        model = RegionNet().eval()
        x = torch.randn(2, 3, 6, 6)
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        with self.building_in():
            with torch.no_grad():
                for _ in range(6):
                    self.assertEqual(compiled(model, x), model(x), atol=0, rtol=0)
            (region,) = self.compiled_regions(compiled)
            self.assertEqual((region["variants"], region["frozen"]), (1, 1))
            # A call that records gradients, and one of a model in training,
            # take the weights.
            self.assertTrue(compiled(model, x).requires_grad)
            with torch.no_grad():
                model.train()
                compiled(model, x)
            (region,) = self.compiled_regions(compiled)
            self.assertEqual((region["variants"], region["frozen"]), (3, 1))
            off = torch.rewrite(
                region_infer,
                passes=with_graphs("aot_eager", freeze=False),
                fallback=False,
            )
            with torch.no_grad():
                model.eval()
                for _ in range(6):
                    self.assertEqual(off(model, x), model(x), atol=0, rtol=0)
            (region,) = self.compiled_regions(off)
            self.assertEqual((region["variants"], region["frozen"]), (1, 0))

    @parametrize("write", ("in_place", "data", "data_swap", "optimizer", "load"))
    def test_write_to_a_built_in_weight_is_seen(self, write):
        torch.manual_seed(0)
        model = RegionNet().eval()
        x = torch.randn(2, 3, 6, 6)

        def change():
            if write == "in_place":
                with torch.no_grad():
                    model.conv.weight.mul_(1.5)
            elif write == "data":
                model.bn.running_mean.data.add_(0.5)
            elif write == "data_swap":
                model.conv.bias.data = torch.randn(4)
            elif write == "optimizer":
                opt = torch.optim.SGD(model.parameters(), lr=0.5)
                model(x).sum().backward()
                opt.step()
            else:
                model.load_state_dict(RegionNet().state_dict())

        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        with self.building_in():
            # Compiled again with what changed as an input: for "load", all.
            for compiles in (1, 2, 2, 2):
                with torch.no_grad():
                    for _ in range(6):
                        self.assertEqual(compiled(model, x), model(x), atol=0, rtol=0)
                (region,) = self.compiled_regions(compiled)
                frozen = 0 if write == "load" and compiles > 1 else 1
                seen = (region["variants"], region["compiles"], region["frozen"])
                self.assertEqual(seen, (1, compiles, frozen))
                change()

    def test_method_of_a_tensor_does_not_keep_the_tensor(self):
        torch.manual_seed(0)
        model = RegionNet().eval()
        compiled = torch.rewrite(
            region_method, passes=with_graphs("aot_eager"), fallback=False
        )
        x = torch.randn(2, 3, 6, 6)
        batches = []
        with torch.no_grad():
            for _ in range(8):
                batch = torch.randn(216)
                expected = region_method(model, batch.view, x)
                got = compiled(model, batch.view, x)
                self.assertEqual(got, expected, atol=0, rtol=0)
                batches.append(weakref.ref(batch))
                del batch
        gc.collect()
        # A method of a tensor is an input like the tensor: no variant holds
        # the one it was made with.
        self.assertEqual([ref() is None for ref in batches], [True] * 8)
        self.assertEqual(sum(r["variants"] for r in self.compiled_regions(compiled)), 1)

    def test_module_made_for_every_call_ends_the_region(self):
        torch.manual_seed(0)
        model = RegionNet().eval()
        compiled = torch.rewrite(
            region_loss, passes=with_graphs("aot_eager"), fallback=False
        )
        y = torch.randint(0, 5, (2,))
        with torch.no_grad():
            for _ in range(40):
                x = torch.randn(2, 3, 6, 6)
                expected = region_loss(model, x, y)
                self.assertEqual(compiled(model, x, y), expected, atol=0, rtol=0)
        regions = torch.rewrite.regions(compiled)
        # The loss was another object on every call: the region that was
        # handed it gave up, the one that is the model did not.
        reasons = [r["reason"] for r in regions if r["state"] == "vm"]
        self.assertIn("is given a new module on every call", reasons)
        compiled_regions = self.compiled_regions(compiled)
        self.assertGreaterEqual(len(compiled_regions), 1)
        self.assertLessEqual(max(r["variants"] for r in compiled_regions), 2)

    def test_only_parameters_and_buffers_a_graph_uses_are_built_in(self):
        torch.manual_seed(0)
        model = RegionTwoHeads().eval()
        x = torch.randn(2, 3, 6, 6)
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        with self.building_in(), torch.no_grad():
            for i in range(12):
                if i >= 6:
                    # Neither is part of the graph as a constant.
                    model.unused.conv.weight.mul_(2)
                    model.cache.add_(1)
                self.assertEqual(compiled(model, x), model(x), atol=0, rtol=0)
        (region,) = self.compiled_regions(compiled)
        self.assertEqual((region["compiles"], region["frozen"]), (1, 1))

    def test_buffer_a_region_writes_is_not_built_in(self):
        torch.manual_seed(0)
        model = RegionCounted().eval()
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        with self.building_in():
            with torch.no_grad():
                for _ in range(8):
                    x = torch.randn(2, 3, 6, 6)
                    reference = copy.deepcopy(model)
                    self.assertEqual(compiled(model, x), reference(x), atol=0, rtol=0)
                    self.assertEqual(model.calls, reference.calls)
        (region,) = self.compiled_regions(compiled)
        self.assertEqual((region["compiles"], region["frozen"]), (1, 1))

    def test_weights_that_were_only_moved_stay_built_in(self):
        torch.manual_seed(0)
        model = RegionNet().eval()
        x = torch.randn(2, 3, 6, 6)
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        with self.building_in(), torch.no_grad():
            for i in range(12):
                if i in (7, 9):
                    # Asks for pointers one could write through.
                    for tensor in model.state_dict().values():
                        tensor.data_ptr()
                self.assertEqual(compiled(model, x), model(x), atol=0, rtol=0)
            (region,) = self.compiled_regions(compiled)
            self.assertEqual((region["compiles"], region["frozen"]), (1, 1))
            # Watched as before.
            model.conv.weight.data.mul_(2)
            self.assertEqual(compiled(model, x), model(x), atol=0, rtol=0)
            (region,) = self.compiled_regions(compiled)
            self.assertEqual((region["compiles"], region["frozen"]), (2, 1))

    @unittest.skipUnless(TEST_NUMPY, "needs NumPy")
    def test_memory_of_a_numpy_array_is_not_built_in(self):
        import numpy as np

        torch.manual_seed(0)
        model = RegionNet().eval()
        array = np.random.rand(4, 3, 3, 3).astype(np.float32)
        model.conv.weight = torch.nn.Parameter(torch.from_numpy(array))
        x = torch.randn(2, 3, 6, 6)
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        with self.building_in(), torch.no_grad():
            for i in range(10):
                if i >= 6:
                    # Nothing PyTorch has says that this happened.
                    array *= 1.5
                self.assertEqual(compiled(model, x), model(x), atol=0, rtol=0)
        (region,) = self.compiled_regions(compiled)
        self.assertEqual((region["compiles"], region["frozen"]), (1, 1))

    def test_matrix_of_a_product_stays_an_input(self):
        from torch.fx.experimental.proxy_tensor import make_fx

        regions = torch.rewrite._regions

        def uses(weight, bias, table, positions, scale, ids, x):
            h = torch.nn.functional.linear(x, weight, bias)
            h = h + torch.nn.functional.embedding(ids, table)
            return h * scale + torch.nn.functional.embedding(positions, table)

        example = (
            torch.randn(8, 8),
            torch.randn(8),
            torch.randn(5, 8),
            torch.arange(4),
            torch.randn(1, 8),
            torch.tensor([0, 1, 2, 3]),
            torch.randn(4, 8),
        )
        graph = make_fx(uses)(*example)
        worth = regions._worth_building_in
        # The table is also looked up at built-in positions: that folds.
        self.assertEqual(worth(graph, [0, 1, 2, 3, 4]), [1, 2, 3, 4])
        self.assertEqual(worth(graph, [0, 1, 2, 4]), [1, 4])

    def test_layers_built_alike_share_a_graph_that_takes_their_weights(self):
        torch.manual_seed(0)
        model = RegionAlike().eval()
        x = torch.randn(2, 3, 6, 6)
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("aot_eager"), fallback=False
        )
        with self.building_in(), torch.no_grad():
            for _ in range(8):
                self.assertEqual(compiled(model, x), model(x), atol=0, rtol=0)
        regions = self.compiled_regions(compiled)
        self.assertEqual(sum(r["variants"] for r in regions), 3)
        # One for the first layer, one for all that came after it.
        self.assertEqual(sum(r["compiles"] for r in regions), 2)
        self.assertEqual(sum(r["frozen"] for r in regions), 1)

    @slowTest
    def test_inductor_folds_built_in_weights(self):
        torch.manual_seed(0)
        model = RegionNet().eval()
        compiled = torch.rewrite(
            region_infer, passes=with_graphs("inductor"), fallback=False
        )
        with torch.no_grad():
            model.bn.running_mean.uniform_(-1, 1)
            model.bn.running_var.uniform_(0.5, 2)
            for i in range(12):
                if i == 8:
                    model.bn.weight.data.mul_(3)
                x = torch.randn(2, 3, 6, 6)
                self.assertEqual(compiled(model, x), model(x), atol=1e-5, rtol=1e-5)
        (region,) = self.compiled_regions(compiled)
        self.assertEqual((region["variants"], region["frozen"]), (1, 1))
        self.assertEqual(region["compiles"], 2)

    @slowTest
    def test_inductor(self):
        def train(compile_step):
            torch.manual_seed(0)
            model = RegionNet().train()
            opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            step = region_step
            if compile_step:
                step = torch.rewrite(
                    step, passes=with_graphs("inductor"), fallback=False
                )
            losses = []
            for _ in range(8):
                x, y = torch.randn(4, 3, 6, 6), torch.randint(0, 5, (4,))
                loss = step(model, opt, x, y)
                losses.append(loss.item())
            return model.state_dict(), losses, type(loss.grad_fn).__name__

        expected_state, expected_losses, _ = train(False)
        state, losses, node = train(True)
        self.assertEqual(node, "CompiledFunctionBackward")
        self.assertEqual(losses, expected_losses, atol=1e-4, rtol=1e-4)
        for name, value in state.items():
            self.assertEqual(value, expected_state[name], atol=1e-4, rtol=1e-4)


instantiate_parametrized_tests(TestCfgRegions)


if __name__ == "__main__":
    run_tests()
