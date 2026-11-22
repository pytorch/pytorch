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
import types as types_module
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
    TEST_WITH_CROSSREF,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)
from torch.utils._python_dispatch import TorchDispatchMode


# These tests look at which kernels the dispatcher runs; the crossref and
# dynamo test configurations interpose on exactly that.
UNDISTURBED = not (TEST_WITH_CROSSREF or TEST_WITH_TORCHDYNAMO)

# The Rust runtime is only in builds made with USE_TORCH_VM=1, and only
# understands the bytecode of CPython 3.12 and 3.14.
try:
    cfg_rs = importlib.import_module("torch._cfg_rs")
except ImportError:
    cfg_rs = None
# The tests are about the VM: nothing is left to the interpreter for being
# faster there, unless a test asks for it.
os.environ["TORCH_VM_DEMOTE"] = "0"

HAS_RUNTIME = (
    UNDISTURBED and cfg_rs is not None and sys.version_info[:2] in ((3, 12), (3, 14))
)
P = torch.vm.passes
DEFAULT = torch.vm.DEFAULT_PASSES
# The default passes over the control-flow graph, which the tests of one pass over a block keep.
CFG = tuple(p for p in DEFAULT if p.kind == "cfg")


def without(*kinds):
    """The default passes without those of some kinds."""
    return tuple(p for p in DEFAULT if not isinstance(p, kinds))


def with_graphs(backend, **options):
    """The default passes, the graph pass, and behind it what a test names as
    the backend of its graphs: the pass that compiles them, or a callable of
    the test's own as a pass over graphs."""
    named = {"aot_eager": P.AOTAutograd, "inductor": P.Inductor}
    behind = named[backend]() if isinstance(backend, str) else backend
    return (*DEFAULT, P.Graphs(**options), behind)


def narrowing():
    """The default passes with NarrowDims in front of Fold."""
    at = DEFAULT.index(P.Fold())
    return (*DEFAULT[:at], P.NarrowDims(), *DEFAULT[at:])


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
    """The dispatcher hook that torch.vm records kernels with."""

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


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmFrontend(TestCase):
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

        compiled = torch.vm(fn, fallback=False)
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

        text = torch.vm(f, fallback=False).dump()
        self.assertIn("phi", text)
        self.assertIn("br_exhausted", text)
        self.assertIn("binary - ", text)
        self.assertIn("binary + ", text)

    def test_arguments(self):
        def f(a, b=2, *args, c=3, **kw):
            return a, b, args, c, sorted(kw.items())

        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(fact, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
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
        compiled = torch.vm(namespace["checked"], fallback=False)
        self.assertEqual(compiled(2), 2)
        with self.assertRaisesRegex(AssertionError, "not one"):
            compiled(1)

    def test_traceback_names_the_line(self):
        def f(x, y):
            z = x + 1
            return z @ y

        compiled = torch.vm(f, fallback=False)
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

        with self.assertRaisesRegex(torch.vm.Unsupported, "generators"):
            torch.vm(f, fallback=False)
        self.assertTrue(torch.vm.is_available())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIs(torch.vm(f), f)
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
        for compiled in (torch.vm(m), torch.vm(m.forward)):
            for _ in range(3):
                self.assertEqual(compiled(x), m(x))


instantiate_parametrized_tests(TestVmFrontend)


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmDevices(TestCase):
    """Profiling on real tensors and direct kernel calls."""

    hw_classification = HardwareClassification.ACCELERATOR

    def assertCompiledEqual(self, fn, argsets, **kwargs):
        compiled = torch.vm(fn, fallback=False, **kwargs)
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

        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
        x = torch.randn(3, device=device)
        compiled(x, x)
        compiled(x, x.clone().requires_grad_())
        compiled(x, x)
        (site,) = compiled.sites()
        self.assertEqual([v["entry_level"] for v in site["variants"]], [1, 0])
        self.assertEqual((site["hits"], site["misses"]), (1, 1))

        strict = torch.vm(f, fallback=False, skip_autograd=False)
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

        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(leaf, fallback=False)
        x = torch.randn(3, device=device, requires_grad=True)
        for _ in range(3):
            with self.assertRaisesRegex(RuntimeError, "leaf Variable"):
                compiled(x)

    def test_sites_are_polymorphic(self, device):
        def f(x, y):
            return (x * y + 2).sum()

        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
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

        boxed = torch.vm(f, fallback=False, unboxed=False)
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

        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
        x = torch.ones(2, device=device)
        self.assertEqual(compiled(x), x * 2)
        # The function ran once, on the real input: no separate tracing run.
        self.assertEqual(len(calls), 1)
        self.assertEqual(block_stats(compiled, "versions"), [1])

    def test_the_default_computes_what_eager_computes_bit_for_bit(self, device):
        layers = (torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.LayerNorm(32))
        model = torch.nn.Sequential(*layers, torch.nn.Linear(32, 4)).to(device)
        twin = copy.deepcopy(model)

        def step(model, opt, x, y):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            return loss

        compiled = torch.vm(step, fallback=False)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        twin_opt = torch.optim.Adam(twin.parameters(), lr=1e-2)
        x = torch.randn(8, 16, device=device)
        y = torch.randint(4, (8,), device=device)
        for _ in range(12):
            loss, expected = compiled(model, opt, x, y), step(twin, twin_opt, x, y)
            self.assertEqual(loss, expected, atol=0, rtol=0)
        for ours, theirs in zip(model.parameters(), twin.parameters()):
            self.assertTrue(torch.equal(ours, theirs))


instantiate_device_type_tests(TestVmDevices, globals())


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmPasses(TestCase):
    """The pass pipeline, guards, and recompilation of single blocks."""

    hw_classification = HardwareClassification.GENERIC

    def test_function_attributes_are_read_through(self):
        @unittest.expectedFailure
        def f(x, k=2):
            return x * k

        f.note = "set by a decorator"
        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(rank, fallback=False)
        for shape in ((2, 3), (2, 3), (4,), (2, 3, 4)):
            self.assertEqual(compiled(torch.ones(shape)), len(shape) + 1)
        self.assertIn("tensor_dim", compiled.dump())

    def test_what_passes_takes(self):
        self.assertNotIn(P.NarrowDims(), DEFAULT)
        with self.assertRaisesRegex(TypeError, "passes has a str"):
            torch.vm(lambda x: x, passes=["fold"])
        with self.assertRaisesRegex(TypeError, "passes has a int"):
            torch.vm(lambda x: x, passes=[3])
        with self.assertRaisesRegex(ValueError, "Fold is given more than once"):
            torch.vm(lambda x: x, passes=[P.Fold(), P.Fold()])
        self.assertEqual(repr(P.Fold()), "torch.vm.passes.Fold()")

    def test_passes_are_applied_in_the_order_of_the_list(self):
        def f(x):
            return x * 2 + 1

        def run(place):
            seen = []

            def look(region):
                seen.append({i["op"] for i in region.instructions()})
                return False

            passes = [P.ResolveDispatch()]
            passes.insert(place, look)
            compiled = torch.vm(f, passes=passes, fallback=False)
            for _ in range(3):
                self.assertEqual(compiled(torch.ones(2)), torch.full((2,), 3.0))
            return seen[0]

        # In front of resolve_dispatch a pass sees what the function wrote, and
        # behind it the kernel calls that those became.
        self.assertNotIn("dispatch", run(0))
        self.assertIn("dispatch", run(1))

    def test_a_pass_over_the_control_flow_graph_is_asked_for(self):
        def run(passes):
            compiled = torch.vm(inl_shifted, passes=passes, fallback=False)
            for _ in range(3):
                self.assertEqual(
                    compiled(torch.ones(2), torch.ones(2)),
                    inl_shifted(torch.ones(2), torch.ones(2)),
                )
            return compiled.stats()["inlined"]

        self.assertEqual(run([P.ResolveDispatch()]), [])
        self.assertEqual(run([P.Inline(), P.ResolveDispatch()]), ["inl_scale"])
        self.assertEqual(run([]), [])

    def test_generic_tier_only(self):
        def f(x, n):
            for _ in range(n):
                x = x * 2 + 1
            return x

        compiled = torch.vm(f, passes=[], fallback=False)
        for _ in range(3):
            self.assertEqual(compiled(torch.ones(2), 3), torch.full((2,), 15.0))
        self.assertEqual(sum(block_stats(compiled, "versions")), 0)
        self.assertEqual(sum(block_stats(compiled, "profile_runs")), 0)
        self.assertNotIn("specialized", compiled.dump())

    def test_without_passes_over_a_block_nothing_is_specialized(self):
        def f(x):
            return x + 1

        compiled = torch.vm(f, passes=CFG, fallback=False)
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

        compiled = torch.vm(f, passes=narrowing(), fallback=False)

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

        compiled = torch.vm(f, passes=narrowing(), fallback=False)
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

        passes = list(DEFAULT)
        passes.insert(passes.index(P.TensorAccessors()) + 1, pin_first_dim)
        compiled = torch.vm(f, passes=passes, fallback=False)
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

        compiled = torch.vm(f, passes=[*CFG, replace], fallback=False)
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
            "TestVmPasses.test_pass_replaces_a_run_by_a_call.<locals>.f (block 0)",
        )
        # Four operators, the constant one of them takes, and the two
        # variables they bind.
        self.assertEqual(len(run.replay.text(run.replay.start, run.replay.end)), 7)

    def test_replayed_instructions_are_called_from_their_function(self):
        # What a pass put in the place of instructions runs them itself: what
        # they call finds the function as its caller, at their line, and not
        # the frames of what replays them.
        class Asker:
            def __add__(self, other):
                frame = sys._getframe(1)
                return frame.f_code.co_name, frame.f_lineno

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

        made = []

        def replace(region):
            insts = region.instructions()
            if any(inst["op"] == "call_object" for inst in insts):
                return False
            (add,) = [i for i, inst in enumerate(insts) if inst["op"] == "binary"]
            made.append(Run(region.replayer(add, add + 1)))
            return region.fuse(add, add + 1, made[-1])

        def f(asker, k):
            found = asker + k
            return found

        line = f.__code__.co_firstlineno + 1
        compiled = torch.vm(f, passes=[*CFG, replace], fallback=False)
        for _ in range(4):
            self.assertEqual(compiled(Asker(), 1), ("f", line))
        (run,) = made
        # The first call profiled the block; the others went through `run`.
        self.assertEqual(run.calls, 3)

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

        compiled = torch.vm(f, passes=[*CFG, grab], fallback=False)
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

    def test_values_needed_and_values_read_behind_a_block(self):
        handles = []

        def grab(region):
            if not handles:
                end = len(region.instructions())
                handles.append((region.replayer(0, end), end))
            return False

        def f(xs, flag):
            total = 0
            for x in xs:
                total = total + x
            if flag:
                total = total * 2
            return total

        passes = [P.Inline(), P.ModuleShortcuts(), grab]
        compiled = torch.vm(f, passes=passes, fallback=False)
        self.assertEqual(compiled([1, 2, 3], True), 12)
        ((replay, end),) = handles
        # The first block takes the list, and makes the sum and the iterator.
        (xs,), made = replay.io(0, end)
        self.assertEqual(len(made), 2)
        needed, read = replay.live(0, end), replay.live(0, end, True)
        self.assertLessEqual(set(made), set(read))
        self.assertLess(set(read), set(needed))
        # The list is kept alive to the end of the function, as the frame of
        # the interpreter would keep it, and nothing reads it again.
        self.assertIn(xs, needed)
        self.assertNotIn(xs, read)
        with self.assertRaisesRegex(IndexError, "no such instruction"):
            replay.live(0, end + 1)

    def test_settle_specializes_the_block_again(self):
        seen = []

        def note_taker(region):
            seen.append(region.note)
            if region.note is None:
                region.note = [region.replayer(0, len(region.instructions()))]
            return False

        def f(x):
            return x * 2

        compiled = torch.vm(f, passes=[note_taker, *DEFAULT], fallback=False)
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

        compiled = torch.vm(
            outer, fallback=False, passes=without(P.Inline), callees=True
        )
        for _ in range(3):
            compiled(torch.ones(2))
        # The first run of a block is its profiling run: on the first call
        # that is outer's, inner's and the one of probe itself.
        self.assertEqual(depths, [3, 0, 0])
        self.assertEqual(cfg_rs.profiling_depth(), 0)

    def test_block_that_raises_every_time_is_specialized(self):
        depths = []

        class Missing:
            def __getitem__(self, key):
                depths.append(cfg_rs.profiling_depth())
                raise KeyError(key)

        def lookup(table, x):
            try:
                return table["scale"] * x
            except KeyError:
                return x + 1

        compiled = torch.vm(lookup, fallback=False)
        x = torch.ones(2)
        for _ in range(5):
            self.assertEqual(compiled(Missing(), x), x + 1)
        # The block of the lookup never gets to its end. It is specialized
        # from its second run, and what it calls is then no longer called
        # from a block that is profiling.
        self.assertEqual(block_stats(compiled, "profile_runs")[1], 1)
        self.assertEqual(depths[2:], [0, 0, 0])
        self.assertGreater(depths[0], 0)
        # A run that gets to the end goes on from where the others raised.
        self.assertEqual(compiled({"scale": 3.0}, x), x * 3)
        self.assertEqual(compiled(Missing(), x), x + 1)

    def test_region_is_only_valid_during_the_pass(self):
        kept = []

        def keep(region):
            kept.append(region)
            return False

        compiled = torch.vm(lambda x: x + 1, passes=[*CFG, keep], fallback=False)
        compiled(torch.ones(1))
        with self.assertRaisesRegex(RuntimeError, "only valid while its pass runs"):
            kept[0].instructions()

    def test_failing_pass_leaves_the_block_generic(self):
        def broken(region):
            raise KeyError("oops")

        def f(x):
            return x * 3

        compiled = torch.vm(f, passes=[*DEFAULT, broken], fallback=False)
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


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmUserPasses(TestCase):
    """Passes of the user's: a kernel in place of operations, by a pass
    written in Python or in Rust."""

    hw_classification = HardwareClassification.GENERIC
    _library = None

    def run_swapped(self, fn, passes, calls=12):
        compiled = torch.vm(fn, passes=[*passes, *DEFAULT], fallback=False)
        x, w = torch.randn(4, 8), torch.randn(8)
        for _ in range(calls):
            self.assertEqual(compiled(x, w), fn(x, w))
        return compiled, x, w

    def test_instructions_name_the_kernel_they_call(self):
        seen = []

        def look(region):
            seen.append([i["kernel"] for i in region.instructions() if i["kernel"]])
            return False

        compiled = torch.vm(swap_rms, passes=[look, *DEFAULT], fallback=False)
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
        swap = torch.vm.swap(SWAP_RMS, kernel, check=2)
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
        swap = torch.vm.swap(SWAP_RMS, kernel)
        compiled, x, w = self.run_swapped(swap_twice, [swap])
        before = kernel.calls
        compiled(x, w)
        self.assertEqual(kernel.calls - before, 2)

    def test_swap_checks_the_kernel_against_the_operations(self):
        swap = torch.vm.swap(SWAP_RMS, SwapKernel(shift=1e-2))
        with self.assertRaisesRegex(torch.vm.Mismatch, "differs by up to 0.01"):
            self.run_swapped(swap_rms, [swap])
        unchecked = torch.vm.swap(SWAP_RMS, SwapKernel(shift=1e-2), check=False)
        compiled = torch.vm(swap_rms, passes=[unchecked, *DEFAULT])
        x, w = torch.randn(4, 8), torch.randn(8)
        for _ in range(6):
            out = compiled(x, w)
        self.assertEqual(out, swap_rms(x, w) + 1e-2 * w)

    def test_swap_leaves_other_operations_alone(self):
        kernel = SwapKernel()
        swap = torch.vm.swap(["pow", "mean", "sub"], kernel)
        self.run_swapped(swap_rms, [swap])
        self.assertEqual(kernel.calls, 0)

    def library(self):
        """The passes of test/vm_passes, built with cargo."""
        cls = type(self)
        if cls._library is None:
            cargo = shutil.which("cargo")
            if cargo is None:
                self.skipTest("cargo is not installed")
            here = os.path.dirname(os.path.abspath(__file__))
            manifest = os.path.join(here, "vm_passes", "Cargo.toml")
            target = tempfile.mkdtemp(prefix="vm_passes_")
            command = [cargo, "build", "--release", "--offline", "--locked"]
            command += ["--manifest-path", manifest, "--target-dir", target]
            subprocess.run(command, check=True, capture_output=True)
            (cls._library,) = (
                os.path.join(target, "release", name)
                for name in os.listdir(os.path.join(target, "release"))
                if name.startswith("libvm_passes.") and not name.endswith(".d")
            )
        return torch.vm.load_passes(cls._library)

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
        with self.assertRaisesRegex(torch.vm.Mismatch, "differs by up to 0.01"):
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
            torch.vm.load_passes("/nowhere/libpasses.so")
        with self.assertRaisesRegex(RuntimeError, "exports no passes"):
            torch.vm.load_passes(torch._C.__file__)


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


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmGraphTransforms(TestCase):
    """Inlining and loop unrolling, which rewrite the graph before it runs."""

    hw_classification = HardwareClassification.GENERIC

    def test_inlines_calls_of_module_functions(self):
        compiled = torch.vm(inl_outer, fallback=False)
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
        compiled = torch.vm(inl_outer, fallback=False)
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
        recursive = torch.vm(inl_fact, fallback=False)
        self.assertEqual(recursive(5), 120)
        self.assertEqual(recursive.stats()["inlined"], [])
        # Defaults and keywords stay calls.
        defaults = torch.vm(inl_uses_defaults, fallback=False)
        self.assertEqual(defaults(torch.ones(2)), torch.full((2,), 5.0))
        self.assertEqual(defaults.stats()["inlined"], [])
        off = torch.vm(inl_outer, fallback=False, passes=without(P.Inline))
        x, y = torch.randn(3), torch.randn(3)
        self.assertEqual(off(x, y), inl_outer(x, y))
        self.assertEqual(off.stats()["inlined"], [])

    def test_unrolls_constant_range_loops(self):
        compiled = torch.vm(unr_loop, fallback=False)
        x, y = torch.randn(3), torch.randn(3)
        for _ in range(3):
            total, i = compiled(x, y)
            expected, expected_i = unr_loop(x, y)
            self.assertEqual(total, expected)
            self.assertEqual(i, expected_i)
        self.assertEqual(compiled.stats()["unrolled_loops"], 1)
        off = torch.vm(unr_loop, fallback=False, passes=without(P.Unroll))
        self.assertEqual(off(x, y)[0], unr_loop(x, y)[0])
        self.assertEqual(off.stats()["unrolled_loops"], 0)

    def test_unrolled_loop_is_guarded_on_range(self):
        compiled = torch.vm(unr_loop, fallback=False)
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
        compiled = torch.vm(nested, fallback=False)
        self.assertEqual(compiled(x), nested(x))
        self.assertGreaterEqual(compiled.stats()["unrolled_loops"], 3)
        compiled = torch.vm(empty, fallback=False)
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
            compiled = torch.vm(fn, fallback=False)
            self.assertEqual(compiled(*args), fn(*args))
            self.assertEqual(compiled.stats()["unrolled_loops"], 0)

    def test_error_in_inlined_function_has_both_frames(self):
        compiled = torch.vm(inl_calls_raises, fallback=False)
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


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmCallees(TestCase):
    """Python functions called from compiled code are compiled in turn."""

    hw_classification = HardwareClassification.GENERIC

    def callees(self, compiled):
        stats = compiled.stats()
        return stats["compiled_callees"], stats["interpreted_callees"]

    def test_functions_of_other_modules_are_compiled(self):
        def f(x):
            return torch.nn.functional.dropout(x, p=0.0, training=False) + 1

        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
        for args, kwargs in (((1,), {}), ((1, 5, 7, 8), {"scale": 3})):
            self.assertEqual(compiled(args, kwargs), f(args, kwargs))
        self.assertEqual(len(self.callees(compiled)[0]), 1)

    def test_unsupported_callee_runs_in_the_interpreter(self):
        def generator(n):
            yield from range(n)

        def f(n):
            return sum(generator(n))

        compiled = torch.vm(f, fallback=False)
        self.assertEqual([compiled(4), compiled(5)], [6, 10])
        compiled_names, interpreted = self.callees(compiled)
        self.assertEqual(compiled_names, [])
        self.assertEqual(len(interpreted), 1)

    def test_recursion_and_modules(self):
        compiled = torch.vm(callee_fact, fallback=False, passes=without(P.Inline))
        self.assertEqual(compiled(6), 720)
        self.assertEqual(self.callees(compiled)[0], ["callee_fact"])

        def run(module, x):
            return module.forward(x) + module(x)

        linear = torch.nn.Linear(3, 2)
        compiled = torch.vm(run, fallback=False)
        x = torch.randn(4, 3)
        for _ in range(3):
            self.assertEqual(compiled(linear, x), run(linear, x))
        self.assertIn("Linear.forward", self.callees(compiled)[0])

    def test_changed_function_is_compiled_again(self):
        def target(x, k=2):
            return x * k

        def f(x):
            return target(x)

        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
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
            compiled = torch.vm(fn, fallback=False)
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
        compiled = torch.vm(count, fallback=False)
        self.assertLess(abs(compiled(0) - count(0)), 5)

    def test_recursion_deeper_than_the_native_stack(self):
        def depth(n):
            return 0 if n == 0 else 1 + depth(n - 1)

        compiled = torch.vm(depth, fallback=False)
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
        compiled = torch.vm(sig, fallback=False)
        compiled_caller = torch.vm(call, fallback=False)
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
            compiled = torch.vm(fn, fallback=False)
            for args in argsets:
                self.assertEqual(call_outcome(compiled, *args), call_outcome(fn, *args))
        message = call_outcome(torch.vm(both, fallback=False), [1], {"k": 5})[1]
        self.assertTrue(
            message.endswith("target() got multiple values for keyword argument 'k'")
        )

    def test_callees_off(self):
        def f(obj, x):
            return obj.twice(x)

        compiled = torch.vm(f, fallback=False, callees=False)
        self.assertEqual(compiled(CalleeScale(2), 3), 12)
        self.assertEqual(self.callees(compiled), ([], []))

    def test_error_in_callee_has_both_frames(self):
        def f(x):
            return callee_raises(x)

        compiled = torch.vm(f, fallback=False, passes=without(P.Inline))
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


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmFrameHook(TestCase):
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
        compiled = torch.vm(fn, fallback=False)
        x = torch.arange(3.0)
        for _ in range(4):
            self.assertEqual(compiled(x), fn(x))
        self.assertIn(through, compiled.stats()["compiled_callees"])
        self.assertEqual(compiled.stats()["interpreted_callees"], [])

    def test_what_an_interpreted_function_calls_is_compiled(self):
        compiled = torch.vm(hook_drives_a_generator, fallback=False)
        x = torch.arange(3.0)
        for _ in range(4):
            self.assertEqual(compiled(x), hook_drives_a_generator(x))
        stats = compiled.stats()
        self.assertEqual(stats["interpreted_callees"], ["hook_generator"])
        self.assertIn("hook_bottom", stats["compiled_callees"])

    def test_arguments_are_bound_as_the_interpreter_binds_them(self):
        compiled = torch.vm(hook_binds_arguments, fallback=False)
        for _ in range(4):
            self.assertEqual(compiled(2), hook_binds_arguments(2))
        self.assertIn("hook_keywords", compiled.stats()["compiled_callees"])
        closures = torch.vm(hook_makes_closures, fallback=False)
        for _ in range(4):
            self.assertEqual(closures(2), hook_makes_closures(2))
        self.assertIn("hook_cell", closures.stats()["compiled_callees"])

    def test_an_exception_passes_through_the_frames(self):
        compiled = torch.vm(hook_reads_broken, fallback=False)

        def frames(fn):
            try:
                fn(HookFragile())
            except Exception as e:
                names = [f.name for f in traceback.extract_tb(e.__traceback__)]
                return type(e), names[:4]

        for _ in range(3):
            self.assertEqual(frames(compiled), frames(hook_reads_broken))
        self.assertIn("HookFragile.broken", compiled.stats()["compiled_callees"])
        endless = torch.vm(hook_reads_endless, fallback=False)
        self.assertEqual(frames(endless)[0], RecursionError)
        self.assertEqual(frames(compiled), frames(hook_reads_broken))

    def test_frames_are_offered_only_under_the_function(self):
        compiled = torch.vm(hook_asks, fallback=False)
        self.assertFalse(cfg_rs.frames_offered())
        # While a block is profiled, what its calls of C functions call is
        # the interpreter's.
        for _ in range(4):
            compiled(1)
        for _ in range(3):
            self.assertEqual(compiled(1), (True, True))
            self.assertFalse(cfg_rs.frames_offered())
        self.assertEqual(hook_asks(1), (False, False))
        alone = torch.vm(hook_asks, fallback=False, callees=False)
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

        compiled = torch.vm(f, passes=[look, *DEFAULT], fallback=False)
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
        compiled = torch.vm(
            hook_calls_slow_python,
            fallback=False,
            demote=True,
            passes=without(P.Inline),
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
        step = torch.vm(region_step, passes=passes, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
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


instantiate_parametrized_tests(TestVmFrameHook)


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmClosures(TestCase):
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

        compiled = torch.vm(counter, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
        x, y = torch.randn(3), torch.randn(3)
        self.assertEqual(compiled(x, y), f(x, y))

    def test_lambdas_with_defaults_and_annotations(self):
        def adders(ks):
            return [(lambda v, k=k: v + k) for k in ks]

        def annotated():
            def inner(a: int, b: "str" = "z") -> float:
                return 1.0

            return inner

        compiled = torch.vm(adders, fallback=False)
        self.assertEqual([add(10) for add in compiled([1, 2, 3])], [11, 12, 13])
        compiled = torch.vm(annotated, fallback=False)
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
        compiled = torch.vm(add, fallback=False)
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
            compiled = torch.vm(fn, fallback=False)
            with self.assertRaises(error) as expected:
                fn()
            with self.assertRaises(error) as got:
                compiled()
            self.assertEqual(str(got.exception), str(expected.exception))


def exc_double(x):
    if x < 0:
        raise ValueError("negative")
    return x * 2


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmExceptions(TestCase):
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
        compiled = torch.vm(fn, fallback=False, **options)
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

        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(f, fallback=False, passes=without(P.Inline), callees=False)
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


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmWith(TestCase):
    """with statements inside compiled functions."""

    hw_classification = HardwareClassification.GENERIC

    def test_with_calls_enter_and_exit(self):
        def f(x, log, suppress):
            y = 0
            with ExcManager(log, suppress) as m:
                y = m + exc_double(x)
            return y

        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
        with self.assertRaisesRegex(
            TypeError, "does not support the context manager protocol"
        ):
            compiled(3)

    def test_module_calls_are_compiled_all_the_way_down(self):
        def run(module, x):
            return module(x) + 1

        linear = torch.nn.Linear(3, 2)
        # Without the shortcut through Module.__call__, that is compiled too.
        compiled = torch.vm(run, fallback=False, passes=without(P.ModuleShortcuts))
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


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmMoreBytecodes(TestCase):
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

        compiled = torch.vm(f, fallback=False)
        self.assertEqual(compiled(2.5), f(2.5))
        with self.assertRaisesRegex(
            ImportError, "cannot import name 'nothing_here' from 'math'"
        ):
            torch.vm(missing, fallback=False)()

    def test_super(self):
        obj = BytecodeChild()
        for method in (BytecodeChild.scale, BytecodeChild.explicit):
            compiled = torch.vm(method, fallback=False)
            self.assertEqual(compiled(obj, 3), method(obj, 3))
        compiled = torch.vm(BytecodeChild.name.fget, fallback=False)
        self.assertEqual(compiled(obj), "child of base")

    def test_super_that_is_not_the_builtin(self):
        class Shadow:
            name = "something else"

        # A module that rebinds `super` gets what it asked for: its object is
        # called without arguments, as the source has it.
        method = BytecodeChild.name.fget
        compiled = torch.vm(method, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
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
        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(f, fallback=False)
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
            compiled = torch.vm(fn, fallback=False)
            self.assertEqual(call_outcome(compiled, *args), call_outcome(fn, *args))
        self.assertNotIn("bytecode_global", globals())

    def test_match_statement(self):
        compiled = torch.vm(match_describe, fallback=False)
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
            outcome = call_outcome(torch.vm(fn, fallback=False), *args)
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
        compiled = torch.vm(big, fallback=False)
        self.assertEqual(compiled(0, 3), big(0, 3))
        self.assertEqual(compiled(torch.zeros(2), 2), big(torch.zeros(2), 2))

    def test_optimizer_step_is_compiled(self):
        def step(opt):
            opt.zero_grad()
            opt.step()

        weight = torch.nn.Parameter(torch.ones(2))
        opt = torch.optim.SGD([weight], lr=0.5)
        compiled = torch.vm(step, fallback=False)
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


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmFrames(TestCase):
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
        compiled = torch.vm(f, fallback=False)
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
        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(
            frame_calls_fails,
            passes=DEFAULT if inline else without(P.Inline),
            fallback=False,
        )
        for _ in range(3):
            self.assertEqual(variables(compiled), variables(frame_calls_fails))
        self.assertEqual(variables(compiled), ({"a": 1, "d": 2}, {"b": 2, "c": [2]}))
        self.assertEqual(compiled.stats()["inlined"], ["frame_fails"] if inline else [])

    def test_tensor_variable_is_bound_when_something_is_called(self):
        compiled = torch.vm(frame_tensors, fallback=False)
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
        compiled = torch.vm(frame_tensor_loop, fallback=False, demote=False)
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
        compiled = torch.vm(f, fallback=False)
        for _ in range(3):
            self.assertEqual(compiled(*args), f(*args))

    def test_locals_holds_tensors_of_specialized_code(self):
        def f(x, y):
            z = x + y
            w = z * 2
            return locals()

        compiled = torch.vm(f, fallback=False)
        x, y = torch.ones(2), torch.full((2,), 3.0)
        for _ in range(5):
            got = compiled(x, y)
        self.assertEqual(list(got), ["x", "y", "z", "w"])
        self.assertEqual(got, f(x, y))

    def test_global_called_locals_that_is_something_else_is_called(self):
        def f(a):
            return locals()

        compiled = torch.vm(f, fallback=False)
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
        compiled = torch.vm(f, fallback=False)
        self.assertEqual(compiled(model, x), f(model, x))

    @parametrize("inline", (False, True))
    def test_the_stack_shows_compiled_functions(self, inline):
        compiled = torch.vm(
            frame_outer,
            passes=DEFAULT if inline else without(P.Inline),
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
        compiled = torch.vm(frame_new_classes, fallback=False)
        self.assertEqual(compiled(), (__name__, __name__))

    def test_warning_names_the_line_of_the_compiled_function(self):
        def caught(fn):
            with warnings.catch_warnings(record=True) as log:
                warnings.simplefilter("always")
                fn()
            return [(w.filename, w.lineno, str(w.message)) for w in log]

        compiled = torch.vm(frame_warns, fallback=False)
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

        compiled = torch.vm(
            frame_calls_divides,
            passes=DEFAULT if inline else without(P.Inline),
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

        self.assertEqual(where(torch.vm(fn, fallback=False)), where(fn))

    def test_name_error_names_the_variable(self):
        def outcome(f, *args):
            try:
                f(*args)
            except NameError as e:
                return type(e), str(e), e.name

        for fn, args in ((frame_missing_global, ()), (frame_unbound_local, (False,))):
            compiled = torch.vm(fn, fallback=False)
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

        for fn in (fails, torch.vm(fails, fallback=False)):
            try:
                fn()
            except ValueError as e:
                # The traceback has the frame, and the frame the local.
                self.assertIsNotNone(probes[-1]())
                traceback.clear_frames(e.__traceback__)
                self.assertIsNone(probes[-1]())

    @parametrize("inline", (False, True))
    def test_calls_made_under_a_trace_function_run_in_the_interpreter(self, inline):
        compiled = torch.vm(
            frame_sets_a_trace,
            passes=DEFAULT if inline else without(P.Inline),
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

        with self.assertRaisesRegex(torch.vm.Unsupported, "not a plain dict"):
            torch.vm(adds, fallback=False)
        compiled = torch.vm(f, fallback=False)
        self.assertEqual(compiled(adds), 8)
        self.assertEqual(len(compiled.stats()["interpreted_callees"]), 1)

    def test_call_of_a_c_function_does_not_keep_its_tensors_alive(self):
        # The first calls of a C function are recorded, to see whether it is
        # one kernel: when it is a whole computation, its tensors come and go.
        def f(x, log):
            return FrameWork.apply(x, log)

        compiled = torch.vm(f, fallback=False)
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

        with self.assertRaisesRegex(torch.vm.Unsupported, "stops a debugger"):
            torch.vm(enters, fallback=False)
        compiled = torch.vm(f, fallback=False)
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

        compiled = torch.vm(f, fallback=False, passes=without(P.Inline))
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
        compiled = torch.vm(f, fallback=False, passes=without(P.Inline))
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

        compiled = torch.vm(f, fallback=False)
        self.assertEqual(compiled(FrameChild(), 3), ("x=3", 7))
        self.assertEqual(compiled.stats()["interpreted_callees"], [])
        names = [
            name.rsplit(".", 1)[-1] for name in compiled.stats()["compiled_callees"]
        ]
        ours = [name for name in names if name in ("fmt", "scale")]
        self.assertEqual(sorted(ours), ["fmt", "scale", "scale"])

    @unittest.skipUnless(sys.platform.startswith("linux"), "counts page faults")
    @unittest.skipIf(sys.version_info < (3, 11), "frames are not kept in chunks")
    def test_calls_at_the_end_of_a_chunk_of_frames_map_no_memory(self):
        import resource

        def callee(a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0):
            return a

        def caller(n):
            for _ in range(n):
                callee()

        def deeper(depth, n):
            return caller(n) if depth == 0 else deeper(depth - 1, n)

        def faults():
            return resource.getrusage(resource.RUSAGE_SELF).ru_minflt

        self.assertEqual(torch.vm(lambda: 1, fallback=False)(), 1)
        # A chunk is 16 KB and a frame of `deeper` some 130 bytes: at one of
        # these depths the frame of `callee` is the first that does not fit.
        seen = []
        for depth in range(160):
            before = faults()
            deeper(depth, 2000)
            seen.append(faults() - before)
        self.assertLess(max(seen), 500)


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


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmEvalBreaker(TestCase):
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

        compiled = torch.vm(breaker_signals_itself, fallback=False)
        expected = [("handler", "breaker_signals_itself"), "after the call"]
        self.assertEqual(run(breaker_signals_itself), expected)
        self.assertEqual(run(compiled), expected)

    @unittest.skipUnless(hasattr(signal, "SIGUSR1"), "needs SIGUSR1")
    def test_exception_of_a_signal_handler_is_raised_by_the_call(self):
        def handler(signum, frame):
            raise KeyboardInterrupt("from the handler")

        compiled = torch.vm(breaker_signals_itself, fallback=False)
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

        compiled = torch.vm(breaker_spins, fallback=False)
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
        compiled = torch.vm(churn, fallback=False)
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


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmLifetimes(TestCase):
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

        compiled = torch.vm(
            fn, passes=DEFAULT if unroll else without(P.Unroll), fallback=False
        )
        expected = run(fn)
        self.assertEqual(run(compiled), expected)
        self.assertEqual(run(compiled), expected)

    def test_hold_in_the_graph(self):
        compiled = torch.vm(lifetime_rebound, fallback=False)
        dump = compiled.dump(versions=False)
        # One for the rebinding, one for the return.
        self.assertEqual(
            [line.strip() for line in dump.splitlines() if "hold" in line],
            ["hold %probe.4", "hold %log.1, %probe.10"],
        )


instantiate_parametrized_tests(TestVmFrames)
instantiate_parametrized_tests(TestVmLifetimes)


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


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmShortcuts(TestCase):
    """Module.__call__ and Module.__getattr__ are skipped when a few checks show what they would do."""

    hw_classification = HardwareClassification.GENERIC

    def test_nn_module_is_what_the_shortcuts_were_written_for(self):
        # If this fails, nn.Module.__call__ or __getattr__ changed: read the
        # change and bring torch/vm/_shortcuts.py in line with it.
        self.assertTrue(torch.vm._shortcuts.unchanged())

    def test_module_call_goes_straight_to_forward(self):
        net, x = ShortcutNet(), torch.randn(4, 3)
        compiled = torch.vm(shortcut_call, fallback=False)
        for _ in range(3):
            self.assertEqual(compiled(net, x), net(x))
        callees = compiled.stats()["compiled_callees"]
        self.assertEqual(callees, ["Linear.forward", "ShortcutNet.forward"])
        self.assertEqual(compiled.stats()["interpreted_callees"], [])

    def test_module_with_hooks_is_called_the_long_way(self):
        net, x = ShortcutNet(), torch.randn(4, 3)
        compiled = torch.vm(shortcut_call, fallback=False)
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
        compiled = torch.vm(shortcut_call, fallback=False)
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
        compiled = torch.vm(shortcut_attributes, fallback=False)
        for _ in range(2):
            for got, expected in zip(compiled(net), shortcut_attributes(net)):
                self.assertIs(got, expected)
        self.assertEqual(self.getattr_calls(compiled, net), 0)
        plain = torch.vm(
            shortcut_attributes, fallback=False, passes=without(P.ModuleShortcuts)
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
        missing = torch.vm(shortcut_missing, fallback=False)
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
        compiled = torch.vm(read, fallback=False)
        expected = call_outcome(read, module)
        self.assertEqual(Guarded.reads, 1)
        self.assertEqual(call_outcome(compiled, module), expected)
        self.assertEqual(Guarded.reads, 2)

    def test_without_shortcuts(self):
        net, x = ShortcutNet(), torch.randn(4, 3)
        compiled = torch.vm(
            shortcut_call, fallback=False, passes=without(P.ModuleShortcuts)
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


def demote_raises(n):
    total = 0
    for i in range(n):
        total += i * i % 7
    raise KeyError(total)


def demote_raises_caller(n, x):
    try:
        demote_raises(n)
    except KeyError as e:
        return e.args[0], x + 1


DEMOTE_LOUD = [False]


def demote_loud(n):
    total = 0
    for i in range(n):
        total += i * i % 7
    if DEMOTE_LOUD[0]:
        cfg_rs.unsettled()
    return total


def demote_loud_loop(n, x):
    total = 0
    for _ in range(500):
        total = demote_loud(n)
    return total, x + 1


def demote_chain_1(n):
    return demote_python(n)


def demote_chain_2(n):
    return demote_chain_1(n)


def demote_chain_3(n):
    return demote_chain_2(n)


def demote_chain_4(n):
    return demote_chain_3(n)


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmDemotion(TestCase):
    """What the interpreter runs faster is left to it."""

    hw_classification = HardwareClassification.GENERIC

    def test_python_callee_is_left_to_the_interpreter(self):
        compiled = torch.vm(
            demote_caller, fallback=False, demote=True, passes=without(P.Inline)
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
        compiled = torch.vm(
            demote_unsettled_caller,
            fallback=False,
            demote=True,
            passes=without(P.Inline),
        )
        x = torch.ones(2)
        for _ in range(200):
            total, _ = compiled(300, x)
            self.assertEqual(total, demote_python(300))
        self.assertEqual(compiled.stats()["demoted_callees"], [])

    def test_a_callee_that_raises_is_measured(self):
        compiled = torch.vm(
            demote_raises_caller, fallback=False, demote=True, passes=without(P.Inline)
        )
        x = torch.ones(2)
        for _ in range(200):
            total, y = compiled(300, x)
            self.assertEqual(total, demote_python(300))
            self.assertEqual(y, x + 1)
        stats = compiled.stats()
        # A call that ends in a raise is timed like one that returns.
        self.assertEqual(stats["demoted_callees"], ["demote_raises"])
        self.assertFalse(stats["measuring"])

    def test_a_measurement_that_ran_out_in_the_first_calls_starts_again(self):
        # The callee is called more often in the first call of the function
        # than a measurement lasts, and as it is then (something below it
        # is not what it will be), it cannot be given up.
        compiled = torch.vm(
            demote_loud_loop, fallback=False, demote=True, passes=without(P.Inline)
        )
        x = torch.ones(2)
        DEMOTE_LOUD[0] = True
        self.addCleanup(DEMOTE_LOUD.__setitem__, 0, False)
        for _ in range(2):
            compiled(50, x)
        self.assertEqual(compiled.stats()["demoted_callees"], [])
        DEMOTE_LOUD[0] = False
        for _ in range(4):
            total, _ = compiled(50, x)
            self.assertEqual(total, demote_python(50))
        # Measured again once that was over: the loop is slower in the VM.
        self.assertEqual(compiled.stats()["demoted_callees"], ["demote_loud"])

    def test_a_measurement_that_runs_out_goes_by_what_was_measured(self):
        # Every call of the callee says that it is not what it will be, so
        # no call of it counts against it; when that has gone on for as long
        # as a measurement lasts, the times there are decide.
        compiled = torch.vm(
            demote_unsettled_caller,
            fallback=False,
            demote=True,
            passes=without(P.Inline),
        )
        x = torch.ones(2)
        for _ in range(450):
            total, _ = compiled(300, x)
            self.assertEqual(total, demote_python(300))
        self.assertEqual(compiled.stats()["demoted_callees"], ["demote_unsettled"])

    def test_python_function_is_left_to_the_interpreter(self):
        compiled = torch.vm(demote_python, fallback=False, demote=True)
        self.assertFalse(compiled.stats()["demoted"])
        for _ in range(200):
            self.assertEqual(compiled(300), demote_python(300))
        self.assertTrue(compiled.stats()["demoted"])

    def test_a_measurement_waits_for_the_device(self):
        # What a graph starts on a GPU is not done when the call returns, so
        # the time of a call that is measured ends when a builtin that waits
        # for the device has returned. Here it is one that can be counted.
        torch.vm._load_runtime()
        waits = list(range(10000))
        cfg_rs.trial_settles(waits.pop)
        self.addCleanup(cfg_rs.trial_settles, None)
        compiled = torch.vm(demote_python, fallback=False, demote=True)
        for _ in range(200):
            self.assertEqual(compiled(300), demote_python(300))
        self.assertTrue(compiled.stats()["demoted"])
        measured = 10000 - len(waits)
        # Before and after each call that was timed, and not before the ones
        # that warm up.
        self.assertGreaterEqual(measured, 8)
        self.assertEqual(measured % 2, 0)
        # Once it is decided, nothing waits.
        for _ in range(20):
            compiled(300)
        self.assertEqual(10000 - len(waits), measured)
        with self.assertRaisesRegex(TypeError, "builtin"):
            cfg_rs.trial_settles(lambda: None)

    def test_measurement_is_reported(self):
        compiled = torch.vm(demote_python, fallback=False, demote=True)
        self.assertTrue(compiled.stats()["measuring"])
        for _ in range(200):
            compiled(300)
        stats = compiled.stats()
        self.assertEqual((stats["measuring"], stats["demoted"]), (False, True))
        unmeasured = torch.vm(demote_python, fallback=False, demote=False)
        self.assertFalse(unmeasured.stats()["measuring"])

    def test_chain_of_calls_is_measured_at_once(self):
        # Five functions deep. Were each measured from scratch once the one
        # it calls is settled, the outermost would still be at it.
        compiled = torch.vm(
            demote_chain_4, fallback=False, demote=True, passes=without(P.Inline)
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
        compiled = torch.vm(
            demote_tensor_caller, fallback=False, demote=True, passes=without(P.Inline)
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

        compiled = torch.vm(slow, fallback=False, demote=True)
        calls = 0
        while compiled.stats()["measuring"] and calls < 16:
            self.assertEqual(compiled(torch.ones(2)), torch.full((2,), 2.0))
            calls += 1
        self.assertFalse(compiled.stats()["measuring"])
        self.assertGreaterEqual(calls, 7)

    def test_measurement_starts_over_when_compiled_code_changes(self):
        compiled = torch.vm(demote_python, fallback=False, demote=True)
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
        compiled = torch.vm(
            demote_caller, fallback=False, demote=False, passes=without(P.Inline)
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

        compiled = torch.vm(
            caller, fallback=False, demote=True, passes=without(P.Inline)
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


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmEndToEnd(TestCase):
    """Whole programs, compared bit for bit with the interpreter."""

    hw_classification = HardwareClassification.GENERIC

    def test_tensor_program(self):
        # An inlined call, an unrolled loop, a branch on a shape that is
        # narrowed under a guard, a loop on tensor data and a Python side
        # effect, in one graph.
        compiled = torch.vm(e2e_program, passes=narrowing(), fallback=False)
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
            step = torch.vm(e2e_step, fallback=False) if compile_step else e2e_step
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


instantiate_parametrized_tests(TestVmEndToEnd)


class RegionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.fc = torch.nn.Linear(4 * 6 * 6, 5)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        return self.fc(torch.tanh(x.flatten(1)))


def region_ordered(model, feats):
    h = torch.relu(feats["a"]) + torch.tanh(feats["b"])
    return model.fc(torch.sigmoid(h).flatten(1))


class RegionOptions:
    """What a script was told on its command line."""

    def __init__(self):
        self.repeat = 3
        self.only = "model"
        self.devices = ["cpu"]
        self.threads = None


class RegionRunner:
    """What calls a model for a benchmark: an object with attributes, one of
    them a method of its own."""

    def __init__(self):
        self.options = RegionOptions()
        self.step = self.forward
        self.scale = 2.0

    def forward(self, model, x):
        return model(x) * self.scale

    def twice(self, model, x):
        return model(x) * (2 * self.scale)


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


class RegionOutput:
    # A plain object, as model code uses for results.
    def __init__(self, logits, hidden, name):
        self.logits = logits
        self.hidden = hidden
        self.name = name


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


def region_remember(model, x, memory):
    out = model(x)
    memory["last"] = out.detach()
    memory["sum"] = memory.get("sum", 0) + out.sum()
    return out * 2


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


def region_clock(model, x):
    start = time.perf_counter_ns()
    out = model(x)
    return out, start


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


def region_method(model, view, x):
    return model(view(2, 3, 6, 6) + x)


def temporaries(x, steps):
    # Every step frees what the step before it made.
    for _ in range(steps):
        x = torch.tanh(x * 1.5 + 0.25)
    return x


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmWorkspace(TestCase):
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
        compiled = torch.vm(temporaries, fallback=False)
        x = torch.randn(64, 1024)
        for _ in range(4):
            self.assertEqual(compiled(x, 6), temporaries(x, 6), atol=0, rtol=0)
        before = torch.vm.memory(compiled)
        self.assertEqual(before["live"], 0)
        self.assertGreaterEqual(before["kept"], x.numel() * 4)
        compiled(x, 6)
        after = torch.vm.memory(compiled)
        # Every tensor of the call found its memory in the workspace.
        self.assertEqual(after["misses"], before["misses"])
        self.assertGreaterEqual(after["hits"] - before["hits"], 6)
        self.assertEqual(after["calls"], before["calls"] + 1)

    def test_result_holds_its_memory(self):
        compiled = torch.vm(temporaries, fallback=False)
        x = torch.randn(64, 1024)
        expected = temporaries(x, 3)
        kept = [compiled(x, 3) for _ in range(3)]
        # What they asked for, rounded up to pages.
        live = torch.vm.memory(compiled)["live"]
        self.assertGreaterEqual(live, 3 * x.numel() * 4)
        self.assertLess(live, 3 * (x.numel() * 4 + 8192))
        # The function goes first: its results still own their memory.
        del compiled
        gc.collect()
        for result in kept:
            self.assertEqual(result, expected, atol=0, rtol=0)

    def test_result_freed_on_another_thread(self):
        compiled = torch.vm(temporaries, fallback=False)
        x = torch.randn(64, 1024)
        kept = [compiled(x, 2)]
        thread = threading.Thread(target=kept.clear)
        thread.start()
        thread.join()
        memory = torch.vm.memory(compiled)
        self.assertEqual(memory["live"], 0)
        self.assertGreaterEqual(memory["kept"], x.numel() * 4)

    def test_small_tensors_are_left_to_malloc(self):
        compiled = torch.vm(temporaries, fallback=False)
        x = torch.randn(16)
        for _ in range(3):
            self.assertEqual(compiled(x, 6), temporaries(x, 6), atol=0, rtol=0)
        memory = torch.vm.memory(compiled)
        self.assertEqual((memory["hits"], memory["misses"]), (0, 0))

    def test_memory_not_asked_for_is_given_back(self):
        compiled = torch.vm(temporaries, fallback=False)
        large, small = torch.randn(256, 1024), torch.randn(16)
        for _ in range(3):
            compiled(large, 6)
        self.assertGreaterEqual(torch.vm.memory(compiled)["kept"], large.numel() * 4)
        for _ in range(6):
            compiled(small, 6)
        self.assertEqual(torch.vm.memory(compiled)["kept"], 0)

    def test_memory_is_cut_to_the_size_that_is_asked_for(self):
        compiled = torch.vm(temporaries, fallback=False)
        large, small = torch.ones(8 << 20), torch.ones(1 << 20)
        for _ in range(2):
            compiled(large, 3)
        before = torch.vm.memory(compiled)
        # An eighth of what a large tensor left behind serves a small one.
        compiled(small, 3)
        after = torch.vm.memory(compiled)
        self.assertEqual(after["misses"], before["misses"])
        self.assertGreater(after["hits"], before["hits"])
        # What comes back is one piece again with the memory around it.
        compiled(large, 3)
        self.assertEqual(torch.vm.memory(compiled)["misses"], before["misses"])

    def test_holds_little_more_than_its_tensors_did_at_their_most(self):
        compiled = torch.vm(temporaries, fallback=False)
        # Every call needs pieces larger than any the calls before it left.
        for n in (1, 2, 6, 15, 37):
            compiled(torch.ones(n << 20), 3)
            memory = torch.vm.memory(compiled)
            # Half as much again, 256 MB at most; and 64 MB whatever its
            # tensors held.
            spare = min(memory["peak"] // 2, 1 << 28)
            limit = max(memory["peak"] + spare, 1 << 26)
            self.assertLessEqual(memory["live"] + memory["kept"], limit)
        # The memory of the last call is all there for the next.
        self.assertGreaterEqual(memory["kept"], 2 * (37 << 22))

    def test_without_a_workspace(self):
        compiled = torch.vm(temporaries, fallback=False, keep_memory=False)
        x = torch.randn(64, 1024)
        self.assertEqual(compiled(x, 6), temporaries(x, 6), atol=0, rtol=0)
        self.assertIsNone(torch.vm.memory(compiled))
        with unittest.mock.patch.dict(os.environ, {"TORCH_VM_KEEP_MEMORY": "0"}):
            compiled = torch.vm(temporaries, fallback=False)
        self.assertIsNone(torch.vm.memory(compiled))

    def test_compiled_function_called_by_another(self):
        inner = torch.vm(temporaries, fallback=False)

        def outer(x):
            return inner(x, 3) + 1

        compiled = torch.vm(outer, fallback=False, callees=False)
        x = torch.randn(64, 1024)
        for _ in range(4):
            self.assertEqual(compiled(x), temporaries(x, 3) + 1, atol=0, rtol=0)
        # Back in the outer function, its own workspace is the one in use.
        self.assertGreater(torch.vm.memory(inner)["hits"], 0)
        self.assertGreater(torch.vm.memory(compiled)["hits"], 0)
        self.assertEqual(torch.vm.memory(compiled)["live"], 0)
        self.assertEqual(torch.vm.memory(inner)["live"], 0)


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmRegions(TestCase):
    """Functions with graphs in them, run end to end, and the numbers of a
    call that a form of a graph is for."""

    hw_classification = HardwareClassification.GENERIC

    def compiled_regions(self, fn):
        return [r for r in torch.vm.regions(fn) if r["state"] == "compiled"]

    def left_to_vm(self, fn, where):
        return [
            r["reason"]
            for r in torch.vm.regions(fn)
            if r["state"] == "vm" and r["where"].startswith(where)
        ]

    def test_frame_readers_amid_tensor_work_find_their_function(self):
        # The instructions of a region run from frames of the region's own
        # until its graph is made: what they call is called by the function.
        graphs = lambda graph, example: graph  # noqa: E731
        fn = frame_reads_amid_tensor_work
        compiled = torch.vm(fn, passes=with_graphs(graphs), fallback=False)
        x = torch.ones(3)
        for _ in range(8):
            self.assertEqual(compiled(x), fn(x))
        self.assertEqual(compiled(x)[1][:2], (True, fn.__name__))
        self.assertEqual(len(self.compiled_regions(compiled)), 2)

    def test_inputs_select_a_variant(self):
        graphs = []

        def backend(graph, inputs):
            graphs.append([i.dtype for i in inputs])
            return graph

        compiled = torch.vm(
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

    def test_frozen_parameter(self):
        torch.manual_seed(0)
        model = RegionNet().train()
        compiled = torch.vm(
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

    def test_no_grad_section_is_not_differentiated(self):
        torch.manual_seed(0)
        model, reference = RegionNet().train(), RegionNet().train()
        reference.load_state_dict(model.state_dict())
        compiled = torch.vm(
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
        compiled = torch.vm(
            region_loop, passes=with_graphs("aot_eager"), fallback=False
        )
        with torch.no_grad():
            for _ in range(8):
                batches = [torch.randn(2, 3, 6, 6) for _ in range(3)]
                expected = region_loop(model, batches)
                self.assertEqual(compiled(model, batches), expected, atol=0, rtol=0)
        self.assertGreaterEqual(len(self.compiled_regions(compiled)), 1)

    def test_training_step_of_a_recurrent_model(self):
        # Whatever the backend makes of the RNN, the step is the step.
        def train(compile_step):
            torch.manual_seed(0)
            model = RegionRecurrent().train()
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            step = region_step_plain
            if compile_step:
                step = torch.vm(step, passes=with_graphs("aot_eager"), fallback=False)
            losses = [step(model, opt, torch.randn(5, 4, 6)).item() for _ in range(8)]
            return model.state_dict(), losses

        expected_state, expected_losses = train(False)
        state, losses = train(True)
        self.assertEqual(losses, expected_losses)
        for name, value in state.items():
            self.assertTrue(torch.equal(value, expected_state[name]), name)

    def test_error_in_a_region(self):
        model = RegionNet().eval()
        compiled = torch.vm(
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
        compiled = torch.vm(
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

    def test_region_that_is_handed_an_ordered_dict(self):
        torch.manual_seed(0)
        model = RegionNet().eval()
        compiled = torch.vm(
            region_ordered, passes=with_graphs("aot_eager"), fallback=False
        )
        with torch.no_grad():
            # Once the module has been looked at, a call is told from the
            # one before it by what it was handed: an OrderedDict is
            # described as a dict is, which the runtime took for a node of
            # another kind and raised TypeError.
            for _ in range(12):
                feats = collections.OrderedDict(
                    a=torch.randn(2, 4, 6, 6), b=torch.randn(2, 4, 6, 6)
                )
                expected = region_ordered(model, feats)
                self.assertEqual(compiled(model, feats), expected, atol=0, rtol=0)
        (region,) = self.compiled_regions(compiled)
        self.assertEqual(region["variants"], 1)

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

        compiled = torch.vm(f, passes=with_graphs(backend), fallback=False)
        x = torch.randn(4)
        for scale in (2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0):
            out, shared = compiled(x, scale)
            self.assertEqual(out, f(x, scale)[0], atol=0, rtol=0)
            self.assertEqual(shared(), scale)
        self.assertEqual(graphs, [5, 5])

    def test_named_tuples_and_sets_of_values_are_inputs(self):
        compiled = torch.vm(
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

    def test_tensor_kept_as_a_plain_attribute(self):
        torch.manual_seed(0)
        model = RegionStateful().eval()
        compiled = torch.vm(
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

    def test_values_that_keep_changing_do_not_multiply_graphs(self):
        graphs = []

        def backend(graph, inputs):
            graphs.append(1)
            return graph

        compiled = torch.vm(region_counter, passes=with_graphs(backend), fallback=False)
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
        compiled = torch.vm(
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

    def looks(self, compiled, *args):
        """How many tensors three calls each make their key of."""
        keyed = []
        real_key = cfg_rs.tensor_key

        def tensor_key(tensors, ints):
            keyed.append(len(tensors))
            return real_key(tensors, ints)

        with unittest.mock.patch.object(cfg_rs, "tensor_key", tensor_key):
            for _ in range(3):
                compiled(*args)
        return keyed

    def test_the_tensors_of_a_module_are_looked_at_once_a_call(self):
        torch.manual_seed(0)
        model = RegionNet().eval()
        passes = with_graphs("aot_eager")
        compiled = torch.vm(region_infer, passes=passes, fallback=False)
        x = torch.randn(2, 3, 6, 6)
        with torch.no_grad():
            for _ in range(10):
                compiled(model, x)
            # A call that is like the one before is told so in the runtime,
            # which makes its key of the input: what the module holds is
            # what an earlier call found, and so is what the graph has built
            # in.
            self.assertEqual(self.looks(compiled, model, x), [])
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
                compiled = torch.vm(f, passes=passes, fallback=False)
                for _ in range(10):
                    self.assertEqual(compiled(*args), f(*args))
                self.assertEqual(len(self.compiled_regions(compiled)), 1)
                # One module inside the other, and a weight that both layers
                # have: a call is still told to be like the one before.
                self.assertEqual(self.looks(compiled, *args), [])
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
            compiled = torch.vm(fn, passes=with_graphs("aot_eager"), fallback=False)
            # On every call, also the ones on which a region is traced: the
            # trace has the generator itself, which it cannot copy, and must
            # not run what takes from it.
            for _ in range(8):
                self.assertEqual(compiled([1, 2], x), fn([1, 2], x))

    def test_a_call_like_the_one_before_is_told_so_in_the_runtime(self):
        def mixed(model, x, options, more, scale, bias=None):
            y = model(x) * scale
            if options["double"]:
                y = y * 2
            for t in more:
                y = y + t.sum()
            if bias is not None:
                y = y + bias.sum()
            return y

        torch.manual_seed(0)
        model = RegionNet().eval()
        x, a, b = torch.randn(2, 3, 6, 6), torch.randn(3), torch.randn(4)
        compiled = torch.vm(mixed, passes=with_graphs("aot_eager"), fallback=False)
        usual = (model, x, {"double": True}, [a, b], 3)
        with torch.no_grad():
            for _ in range(10):
                self.assertEqual(compiled(*usual), mixed(*usual))
            self.assertEqual(self.looks(compiled, *usual), [])
            # A call that is not like it is described the long way, whatever
            # it is that differs: a value, a number, how a list or a dict is
            # built, a tensor of another size or kind or one that comes
            # twice, the modes of the thread, what the module holds.
            unusual = [
                (model, x, {"double": False}, [a, b], 3),
                (model, x, {"double": True}, [a], 3),
                (model, x, {"double": True}, [a, b, a], 3),
                (model, x, {"double": True}, [a, b], 5),
                (model, x, {"double": True}, [a, b], 2.5),
                (model, x, {"double": True, "more": 1}, [a, b], 3),
                (model, x, {"double": True}, [a, x], 3),
                (model, x, {"double": True}, [a, b], 3, b),
                (model, x.clone().requires_grad_(), {"double": True}, [a, b], 3),
                (model, torch.randn(5, 3, 6, 6), {"double": True}, [a, b], 3),
            ]
            for args in unusual:
                for _ in range(2):
                    self.assertEqual(compiled(*args), mixed(*args))
                self.assertEqual(compiled(*usual), mixed(*usual))
        for _ in range(2):
            self.assertEqual(compiled(*usual), mixed(*usual))
        with torch.no_grad():
            model.fc.weight.mul_(2)
            self.assertEqual(compiled(*usual), mixed(*usual))
            model.fc.weight.requires_grad_(False)
            self.assertEqual(compiled(*usual), mixed(*usual))
            model.fc.bias = torch.nn.Parameter(torch.ones(5))
            self.assertEqual(compiled(*usual), mixed(*usual))
            model.bn.train()
            self.assertEqual(compiled(*usual), mixed(*usual))
            model.bn.eval()
            handle = torch.nn.modules.module.register_module_forward_hook(
                lambda *args: None
            )
            try:
                self.assertEqual(compiled(*usual), mixed(*usual))
            finally:
                handle.remove()
            # And a call is like the one before again once it is.
            for _ in range(6):
                self.assertEqual(compiled(*usual), mixed(*usual))
            self.assertEqual(self.looks(compiled, *usual), [])

    def test_a_call_through_a_method_of_an_object_is_told_in_the_runtime(self):
        def measured(step, model, x):
            return step(model, x)

        torch.manual_seed(0)
        model = RegionNet().eval()
        x = torch.randn(2, 3, 6, 6)
        runner = RegionRunner()
        passes = with_graphs("aot_eager")
        compiled = torch.vm(measured, passes=passes, fallback=False)
        with torch.no_grad():
            for _ in range(10):
                expected = measured(runner.forward, model, x)
                self.assertEqual(compiled(runner.forward, model, x), expected)
            self.assertEqual(self.looks(compiled, runner.forward, model, x), [])
            # What the object holds is what a call depends on, as before.
            runner.scale = 3.0
            expected = measured(runner.forward, model, x)
            self.assertEqual(compiled(runner.forward, model, x), expected)
            runner.options.only = "another"
            self.assertEqual(compiled(runner.step, model, x), expected)
            expected = measured(runner.twice, model, x)
            self.assertEqual(compiled(runner.twice, model, x), expected)
            for _ in range(6):
                compiled(runner.forward, model, x)
            self.assertEqual(self.looks(compiled, runner.forward, model, x), [])

    def test_a_pass_over_the_graph_brings_its_guards(self):
        made = []

        def only_small(gm, input_types):
            x = next(node for node in gm.graph.nodes if node.op == "placeholder")
            made.append(torch.vm.materialize(input_types[0], at="hi").shape)
            torch.vm.guard(gm, x, torch.vm.Dim(0, range(1, 6)), torch.vm.Contiguous())
            torch.vm.guard(gm, x, *input_types[0].as_checks())
            for node in gm.graph.nodes:
                if node.target is torch.ops.aten.tanh.default:
                    node.target = torch.ops.aten.sigmoid.default
            return gm

        def f(x, w):
            return torch.tanh(x @ w).sum() + (x * 2).mean()

        compiled = torch.vm(f, passes=(*DEFAULT, only_small), fallback=False)
        w = torch.randn(8, 8)
        for n in (4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 4):
            x = torch.randn(n, 8)
            got = compiled(x, w)
            if n == 8:
                # A call that fails the guard runs what the function wrote.
                self.assertEqual(got, f(x, w))
        self.assertEqual(got, torch.sigmoid(x @ w).sum() + (x * 2).mean())
        self.assertEqual(made[0], (4, 8))
        gm = torch.fx.symbolic_trace(f)
        x = next(node for node in gm.graph.nodes if node.op == "placeholder")
        with self.assertRaisesRegex(TypeError, "takes checks"):
            torch.vm.guard(gm, x, 3)
        with self.assertRaisesRegex(ValueError, "is computed in it"):
            torch.vm.guard(gm, list(gm.graph.nodes)[-2], torch.vm.Contiguous())

    def test_what_types_and_checks_say(self):
        interval = torch.vm.Interval(16, 32)
        self.assertEqual(interval.samples(3), [16, 24, 32])
        self.assertEqual((20 in interval, 33 in interval), (True, False))
        t = torch.vm.TensorType(torch.float32, torch.device("cpu"), False, (interval,))
        self.assertEqual(
            (repr(t), t.nbytes("lo"), t.nbytes()), ("float32[16..32]", 64, 128)
        )
        self.assertEqual(torch.vm.materialize(t, dim0=24).shape, (24,))
        # The type as guards: what a pass that specializes to it adds.
        checks = t.as_checks()
        dtype, device = torch.vm.Dtype(torch.float32), torch.vm.Device(t.device)
        self.assertEqual(checks, [dtype, device, torch.vm.Dim(0, interval)])
        self.assertTrue(all(check.holds(torch.zeros(20)) for check in checks))
        self.assertFalse(all(check.holds(torch.zeros(40)) for check in checks))
        x = torch.zeros(4, 6)[:, ::2]
        self.assertTrue(torch.vm.Dim(-1, {3}).holds(x))
        self.assertTrue(torch.vm.Stride(-1, {2}).holds(x))
        self.assertFalse(torch.vm.Contiguous().holds(x))
        self.assertTrue(torch.vm.Dtype(torch.float32).holds(x))
        self.assertTrue(torch.vm.Device("cpu").holds(x))

    def test_a_pass_over_the_graph_that_raises_leaves_the_region_to_the_vm(self):
        def broken(gm, input_types):
            raise RuntimeError("not today")

        def f(x, w):
            return torch.tanh(x @ w).sum() + (x * 2).mean()

        compiled = torch.vm(f, passes=(*DEFAULT, broken), fallback=False)
        x, w = torch.randn(4, 8), torch.randn(8, 8)
        for _ in range(4):
            self.assertEqual(compiled(x, w), f(x, w), atol=0, rtol=0)
        regions = torch.vm.regions(compiled)
        reasons = [r["reason"] for r in regions if r["state"] == "vm"]
        self.assertIn("the pass broken raised RuntimeError: not today", reasons)

    def test_region_stores_into_a_dict_it_is_given(self):
        torch.manual_seed(0)
        model = RegionLinears().eval()
        compiled = torch.vm(
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

    def test_path_ends_before_a_call_that_fails_part_way(self):
        torch.manual_seed(0)
        model = RegionLinears().eval()
        compiled = torch.vm(
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

    def test_strides_select_a_variant(self):
        compiled = torch.vm(
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
        torch.vm(region_small)
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
        torch.vm(region_small)
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

    def test_method_of_a_tensor_does_not_keep_the_tensor(self):
        torch.manual_seed(0)
        model = RegionNet().eval()
        compiled = torch.vm(
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


def vm_twelve(x):
    for _ in range(12):
        x = x + 1
    return x


def vm_thirteen(x):
    return vm_twelve(x) + 1


def vm_scaled(x, y):
    return unr_loop(x, y)[0] * 2


class GraphProbe:
    """What the instructions of ``graph_facts`` look at."""

    def __init__(self):
        self.items = [1, 2, 3]
        self.plain = 2.0
        self.offered = []

    @property
    def scale(self):
        return 2.0

    def shift(self, x):
        return x + 1

    def __call__(self, x):
        return x - 1

    def __rmul__(self, other):
        self.offered.append(cfg_rs.frames_offered())
        return other * 3


def graph_plus_one(x):
    return x + 1


def graph_facts(x, probe, fn):
    y = x * probe.scale
    k = probe.plain
    n = len(probe.items)
    z = probe.shift(y)
    z = probe(z)
    z = fn(z)
    z.add_(n)
    w = x * probe
    return z.sum() + k + w.sum()


def graph_poly(x):
    return x.sin().cos() * 2 + x


def graph_two(x):
    y = x.sin().cos() * 2 + x
    print(end="")
    return (y.tanh() * y).exp().sum() - y.mean()


def graph_written(x):
    y = x.sin() * 2 + 1
    y.add_(1)
    return (y.cos() * y - y.tanh()).sum()


def graph_mixed(x, w):
    return torch.tanh(x @ w).sum() + (x * 2).mean()


def graph_sizes(x, w):
    y = torch.relu(x @ w)
    z = y.reshape(y.shape[0] * 2, -1)
    return (z * 2).sum(0) + z.shape[1]


def graph_normed(x, mean, var, training):
    y = torch.batch_norm(x * 2, None, None, mean, var, training, 0.1, 1e-5, True)
    return (torch.relu(y) + 1).mean((2, 3)).sum()


# What ``graph_reads`` reads between its operations.
graph_ops = types_module.SimpleNamespace(act=torch.tanh, scale=0.5)


def graph_reads(x):
    return graph_ops.act(x) * graph_ops.scale + x.sin() - x.cos()


class GraphNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
        self.fc = torch.nn.Linear(8, 5)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x * 2)))
        return self.fc(x.mean((2, 3)))


class GraphSeen(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.seen = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen.append(func)
        return func(*args, **(kwargs or {}))


# What ``graph_global`` reads between its operations.
GRAPH_SCALE = 0.5


def graph_global(x):
    return ((x.sin() * GRAPH_SCALE).cos() + GRAPH_SCALE).tanh() * 2


def graph_enclosed(scale):
    """A function that reads a variable of this one between its operations,
    and what rebinds the variable."""

    def run(x):
        return ((x.sin() * scale).cos() + scale).tanh() * 2

    def rebind(value):
        nonlocal scale
        scale = value

    return run, rebind


class GraphLeaky(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(6, 8)
        self.act = torch.nn.LeakyReLU(0.1)
        self.fc2 = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.act(self.fc2(self.act(self.fc1(x))))


def graph_infer(model, x):
    return model(x)


class GraphBox:
    """An object whose ``scale`` is an attribute until it is deleted: then
    its ``__getattr__`` makes one, another at every read."""

    def __init__(self):
        self.scale = 0.5
        self.made = 0

    def __getattr__(self, name):
        self.made += 1
        return 0.5 + self.made


def graph_boxed(x, box):
    return ((x.sin() * box.scale).cos() + box.scale).tanh() * 2


def graph_counter(x, i):
    return ((x + i).sin() * i).cos().exp().tanh()


def graph_cat(a, b):
    return torch.cat([a.sin(), b.cos()], 1).tanh() * 2


def graph_stack(a, b):
    return torch.stack((a.sin(), b, a * 2)).tanh().sum(0)


def graph_masked(x, mask):
    y = x.sin() * 2
    return (y[mask].cos() + 1).tanh()


def graph_subscripts(x, i):
    y = x.sin() * 2
    return (y[0] + y[1:, 2].sum() + y[i].mean() + y[:, i].sum()).tanh()


def graph_conv(x, w, b):
    y = torch.conv2d(x.sin(), w, b, 1, 1)
    return (torch.relu(y) * 2).tanh()


def graph_listed(xs, k):
    y = torch.cat(xs, 1).sin()
    return (torch.max_pool2d(y, k, 1, 0, 1, False) * 2).cos().tanh()


def graph_operands(case, device):
    """A function with an operation that is no call site, and tensors to
    call it with."""
    make = functools.partial(torch.randn, device=device)
    if case == "cat":
        return graph_cat, (make(3, 4), make(3, 4))
    if case == "stack":
        return graph_stack, (make(3, 4), make(3, 4))
    if case == "masked":
        x = make(4, 5)
        return graph_masked, (x, x > 0)
    if case == "subscripts":
        return graph_subscripts, (make(4, 5), torch.tensor([0, 2], device=device))
    if case == "listed":
        return graph_listed, ((make(2, 3, 8, 8), make(2, 3, 8, 8)), 3)
    return graph_conv, (make(2, 3, 8, 8), make(4, 3, 3, 3), make(4))


def graph_branches(x, flag):
    y = x.sin() * 2
    if flag:
        y = y.cos() + 1
    else:
        y = y.tanh() - 1
    return (y * y).exp().sum()


def graph_optional(x, bias=None):
    y = x.sin() * 2
    if bias is not None:
        y = y + bias
    return y.cos().tanh().sum()


def graph_loops(x, n):
    y = x.sin()
    for i in range(n):
        y = y * 0.5 + i
    return y.cos().sum()


def graph_collects(xs):
    out = []
    for x in xs:
        out.append(x.sin() * 2)
    return torch.cat(out).tanh().sum()


def graph_on_data(x):
    y = x.sin() * 2 + 1
    if y.sum() > 0:
        y = y.cos()
    return (y * y).tanh().exp().sum()


def graph_on_a_size(x):
    y = x.sin() * 2
    if y.shape[0] > 3:
        y = y.cos()
    return (y + 1).tanh().sum()


def graph_handled(x, table):
    y = x.sin() * 2
    try:
        k = table["k"]
        y = y.cos() + k
    except KeyError:
        y = y - 1
    return (y * 2).tanh().sum()


def graph_handled_then(x, table):
    try:
        y = x.sin() * 2
        z = (y + table["k"]).cos().tanh()
    except KeyError:
        z = x
    str(table)
    return z * 2


def graph_looks_up(x, table, key):
    y = x.sin() * 2
    if key:
        k = table[key]
        y = y + k
    return y.cos().tanh().sum()


def graph_printed(x):
    return str(x.sin() * 2)


def graph_unpacks(x, pair):
    y = x.sin() * 2
    a, b = pair
    return (y + a).cos().tanh() * b


class GraphScaled:
    """An object whose ``scale`` is an attribute, until its class gets a
    property of the name that counts its reads."""

    def __init__(self):
        self.scale = 0.5
        self.reads = 0


def graph_casts(x, w):
    return (torch.mm(torch.mm(x, w), w).relu() * 2 + 1).tanh()


class GraphCalls(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.seen = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.seen.append(getattr(func, "__name__", str(func)))
        return func(*args, **(kwargs or {}))


class GraphSub(torch.Tensor):
    pass


def graph_named(x, y):
    return torch.relu(x * y + y) - torch.cat([x, y], 1).sum()


def graph_reshaped(x, shape):
    return torch.flatten(x.sin().reshape(shape).cos().tanh(), 2).exp()


def graph_helper(x, scale=2.0, *, shift=1.0):
    return (x * scale + shift).tanh()


def graph_calls(x):
    y = graph_helper(x.sin())
    z = graph_helper(y, 3.0, shift=0.5)
    return graph_helper(z, scale=0.25).cos()


def graph_wraps(fn):
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)

    return inner


@graph_wraps
def graph_wrapped(x, scale=2.0, *, shift=1.0):
    return (x * scale + shift).tanh()


def graph_spreads(x):
    return graph_wrapped(graph_wrapped(x.sin(), 3.0), shift=0.5).cos()


class GraphTools:
    def __init__(self):
        self.k = 3.0
        self.log = []
        self.measured = False

    @staticmethod
    def double(x):
        return x * 2

    @classmethod
    def halve(cls, x):
        return x / 2

    @property
    def gain(self):
        return self.k

    def apply(self, x):
        return self.double(self.halve(x) + self.gain).tanh()

    def noted(self, x):
        y = x.sin() * 2
        # A call that a path does not go through (and that gives 0).
        self.log.append(1 + int(random.random()))
        return y.cos().tanh() + 1

    def picky(self, x, fail):
        y = x.sin() * 2
        if fail:
            raise ValueError("picky")
        return y.cos()


GRAPH_TOOLS = GraphTools()


def graph_methods(x):
    bound = GRAPH_TOOLS.apply
    return (bound(x.sin()) + GraphTools.double(x)).cos()


def graph_noted(x):
    return GRAPH_TOOLS.noted(x.exp()).tanh() * 3


def graph_picky(x, fail):
    try:
        y = GRAPH_TOOLS.picky(x.exp(), fail)
    except ValueError:
        y = x * 0
    return (y + 1).tanh().sin()


def graph_descends(x, n):
    if n == 0:
        return x.tanh()
    return graph_descends(x.sin() * 2, n - 1) + 1


class GraphStage(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = torch.nn.Conv2d(c, c, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(c)
        self.act = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = GraphStage(4)
        self.seq = torch.nn.Sequential(GraphStage(4), torch.nn.Tanh(), GraphStage(4))
        self.blocks = torch.nn.ModuleList([GraphStage(4) for _ in range(2)])
        self.head = torch.nn.Linear(4 * 8 * 8, 5)

    def forward(self, x):
        x = self.seq(self.a(x))
        for block in self.blocks:
            x = x + block(x)
        return self.head(x.flatten(1))


class GraphLayers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(3)])
        named = {"a": torch.nn.Linear(8, 8), "b": torch.nn.Tanh()}
        self.named = torch.nn.ModuleDict(named)
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x) * (i + 1)
        for a, b in zip(self.layers, self.layers):
            x = a(x) - b(x) + x
        x = self.layers[1](x) + len(self.layers)
        x = self.named["b"](self.named["a"](x))
        return self.drop(x)


class GraphPools(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2)
        self.norm = torch.nn.LayerNorm([4, 4, 4])

    def forward(self, x):
        functional = torch.nn.functional
        y = self.pool(functional.relu(self.conv(x)))
        y = functional.interpolate(self.norm(y), scale_factor=2.0, mode="nearest")
        pooled = functional.max_pool2d(y, 3, stride=1, padding=1)
        return pooled + functional.adaptive_avg_pool2d(y, 1)


def graph_flips(x, more):
    y = x.sin() * 2
    dims = [0, 1, 2, *more]
    return torch.flip(y, [0, 1, 2]).cos().tanh() + y.mean([0, 1, 2]) + y.sum(dims)


GRAPH_BIAS = 0.5


def graph_counts(n, tools):
    # A loop of integer arithmetic. While `measured` is set, its compiled
    # form sleeps: the measurement leaves it to the interpreter whatever
    # the clock says about the loop.
    total = 0
    for i in range(n):
        total += i * i % 7
    if tools.measured and cfg_rs.frames_offered():
        time.sleep(0.002)
    if tools.k > 4:
        return float(total) + GRAPH_BIAS
    return float(total)


def graph_counted(x, n):
    if GRAPH_TOOLS.measured:
        # And this one sleeps where the interpreter runs it: it is kept.
        if not cfg_rs.frames_offered():
            time.sleep(0.002)
        return graph_counts(n, GRAPH_TOOLS)
    return ((x.sin() * graph_counts(n, GRAPH_TOOLS)).cos() + 1).tanh().exp()


def graph_timed(x):
    # Its compiled form sleeps: slower than the interpreter by far.
    if cfg_rs.frames_offered():
        time.sleep(0.002)
    return ((x.sin() + 1).cos() * 2).tanh().exp()


class GraphLogs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers = [torch.nn.Linear(4, 4), torch.nn.Tanh()]
        layers += [torch.nn.Linear(4, 4), torch.nn.Tanh()]
        self.inner = torch.nn.Sequential(*layers)

    def forward(self, x):
        y = self.inner(x)
        GRAPH_TOOLS.log.append(1 + int(random.random()))
        return y.tanh() * 2


class GraphResidual(torch.nn.Module):
    """A block as torchvision's resnets have it: activations that write to
    what they are given, and a sum into what the block made."""

    def __init__(self, c=4):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(c, c, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(c)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(c, c, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(c)
        shortcut = (torch.nn.Conv2d(c, c, 1), torch.nn.BatchNorm2d(c))
        self.downsample = torch.nn.Sequential(*shortcut)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class GraphResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(GraphResidual(), GraphResidual())
        self.head = torch.nn.Linear(4 * 8 * 8, 5)

    def forward(self, x):
        return self.head(torch.flatten(self.layer(x), 1))


class GraphHead(torch.nn.Module):
    """What a detection head does to its output: in place, through views
    and subscripts."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 6, 1)
        self.stride = 8.0

    def forward(self, x):
        io = self.conv(x).permute(0, 2, 3, 1).contiguous()
        torch.nn.functional.leaky_relu(io, 0.1, inplace=True)
        torch.sigmoid_(io[..., 4:])
        io[..., :2] = io[..., :2] * 2 + 1
        io[..., :4] *= self.stride
        scale = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=x.device)
        return io.view(io.shape[0], -1, 6) * scale, io, io[..., :2]


def graph_writes_outside(x, buffer, fail):
    y = (x.sin() * 2 + 1).cos()
    y.tanh_()
    buffer.add_(y.mean())
    if fail:
        raise ValueError("after the write")
    view = x.view(-1)
    view.mul_(0.5)
    return (y + buffer).exp().sin().cos().tanh() + x


class GraphKeeps(torch.nn.Module):
    """Stores into itself, into a plain object and into containers it is
    given, between operations."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.calls = 0
        self.seen = []
        self.last = None

    def forward(self, x, notes, fail=False):
        y = (self.fc(x).sin() * 2).cos().tanh()
        self.calls += 1
        self.seen.append(y.shape[0])
        notes["calls"] = self.calls
        GRAPH_TOOLS.k = float(self.calls)
        if fail:
            raise ValueError("after the stores")
        self.last = y
        made = [y, {"n": self.calls}]
        return (y + 1).exp().sin().cos().tanh(), made


class GraphRouted(torch.nn.Module):
    """Keeps what its layers gave in a list, with a placeholder for the
    ones nothing reads, and reads the list further on."""

    def __init__(self):
        super().__init__()

        def conv():
            return torch.nn.Conv2d(4, 4, 3, padding=1)

        layers = [conv(), conv(), conv(), conv(), GraphLogs(), conv(), conv()]
        self.layers = torch.nn.ModuleList(layers)
        self.routes = {2: (-2,), 5: (-5, 2), 6: (-2, 0)}
        self.kept = [True, True, True, False, True, True, True]

    def forward(self, x):
        out = []
        for i, layer in enumerate(self.layers):
            x = layer(x) if i != 4 else x * 1
            for back in self.routes.get(i, ()):
                x = x + out[back]
            out.append(x if self.kept[i] else [])
            if i == 4:
                GRAPH_TOOLS.log.append(len(out) + int(random.random()))
        return x


class GraphHolder:
    def __init__(self):
        self.value = 0


class GraphStores(torch.nn.Module):
    """Stores of every kind between its operations, and reads behind them."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.register_buffer("grid", torch.zeros(4))
        self.calls = 0
        self.nx = self.ny = 0
        self.seen = []
        self.last = None
        self.box = GraphHolder()

    def forward(self, x, notes, other, fail=False):
        y = (self.fc(x).sin() * 2).cos()
        self.calls += 1
        self.nx, self.ny = x.shape[0], x.shape[1]
        self.grid = y.mean(0)
        z = y.view(self.nx, self.ny) + self.grid
        self.seen.append(self.calls)
        self.seen.extend([self.nx, self.ny])
        notes["calls"] = self.calls
        notes["next"] = notes["calls"] + 1
        self.box.value = self.calls
        other.value += 10
        self.seen[0] = -self.box.value
        if fail:
            raise ValueError("behind the stores")
        self.last = z
        self.calls += 1
        return (z + 1).exp().tanh(), [z, self.last]

    def state(self, notes, other):
        seen, noted = list(self.seen), dict(notes)
        kept = (self.calls, self.nx, self.ny, seen, noted, self.box.value, other.value)
        return kept, self.grid.clone(), self.last


class GraphGuarded(GraphStores):
    """What it stores goes through a `__setattr__` of its own."""

    def __setattr__(self, name, value):
        if name == "calls":
            value = value * 1
        super().__setattr__(name, value)


class GraphRebinds(torch.nn.Module):
    """Stores that `nn.Module.__setattr__` does more for than keep them."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.act = torch.nn.Tanh()
        self.count = 0

    def forward(self, x, fresh, layer):
        y = (x.sin() * self.weight).cos().tanh()
        self.count += 1
        if fresh is not None:
            self.weight = fresh
        if layer is not None:
            self.act = layer
        return self.act(y * self.weight).exp().sin()


class GraphAverages(torch.nn.Module):
    """Writes to its buffers and to what it is handed, between operations."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.register_buffer("ema", torch.zeros(4))
        self.register_buffer("last", torch.zeros(3, 4))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    def forward(self, x):
        y = (self.fc(x).sin() * 2).cos()
        self.ema.mul_(0.9).add_(y.mean(0), alpha=0.1)
        self.last[:] = y
        self.steps += 1
        x.add_(1)
        return (y + self.ema).exp().tanh() + self.last.sum() + x


def graph_shares(x, part):
    # `part` may be a view of `x`.
    y = (x.sin() * 2).cos()
    part.mul_(2)
    x.add_(y)
    return (x * part.sum()).tanh().exp() + y


def graph_saves(x):
    y = x.exp()
    y.add_(1)
    return (y * 2).sin().cos().sum()


def graph_several(x):
    y = (x.sin() * 2 + 1).cos()
    a, b = y.chunk(2, 1)
    top, where = torch.max(a, 1)
    lowest = torch.min(b, 1)
    parts = y.split(3, 1)
    first, second, third = (a + lowest.values.unsqueeze(1)).unbind(1)
    best = torch.topk(b, 2, 1)
    out = (a * b).exp() + top.unsqueeze(1) + first.unsqueeze(1) + third.unsqueeze(1)
    return out + parts[1] + torch.sort(b, 1).values + best.values.sum()


def graph_rated(x, rate):
    return ((x.sin() * rate).cos() + rate).tanh().exp()


def graph_inner(x):
    y = (x.sin() + 1).cos().tanh().exp()
    return y * random.random()


class GraphPair:
    def __init__(self):
        self.n = 0
        self.items = [0]


def graph_two_names(x, a, b):
    y = (x.sin() * 2 + 1).cos()
    a.n = 1
    b.n = 2
    b.items.append(a.n)
    return (y + a.n).exp().tanh() * b.n, a.n


def graph_two_lists(x, a, b):
    y = (x.sin() * 2 + 1).cos()
    a.items.append(y.shape[0])
    return (y * 2).exp().tanh().sum(), len(b.items), b.items[-1], sum(b.items)


def graph_through_dict(x, a, b):
    y = (x.sin() * 2 + 1).cos()
    a.__dict__["n"] = 7
    return (y * 2).exp().tanh().sum(), b.n


def graph_through_getattr(x, a, b):
    y = (x.sin() * 2 + 1).cos()
    a.n = 7
    return (y * 2).exp().tanh().sum(), getattr(b, "n"), hasattr(b, "n")  # noqa: B009


def graph_through_iterator(x, a, b):
    y = (x.sin() * 2 + 1).cos()
    over = iter(b.items)
    a.items.append(5)
    return (y * 2).exp().tanh().sum(), list(over)


def graph_dict_of(x, entries, b):
    y = (x.sin() * 2 + 1).cos()
    entries["n"] = 7
    return (y * 2).exp().tanh().sum(), b.n


def graph_versions(x, buffer):
    y = (x.sin() * 2 + 1).cos()
    buffer.add_(y.mean())
    return (y + buffer).exp().tanh().sum(), buffer._version


def graph_turns(x, buffer):
    y = (x.sin() * 2 + 1).cos()
    buffer.t_()
    return (y.sum() + buffer.sum()).exp().tanh() * buffer.shape[0], tuple(buffer.shape)


def graph_learns(x, w, buffer):
    y = (x.sin() * w + 1).cos()
    buffer.mul_(0.9).add_(y.mean(0), alpha=0.1)
    return (y + buffer).exp().tanh().sum(), buffer.requires_grad


class GraphCounts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.seen = []
        self.register_buffer("total", torch.zeros(()))

    def forward(self, x, index):
        y = (x.sin() * 2 + 1).cos()
        self.calls += 1
        self.total.add_(y.mean())
        self.seen.append(self.calls)
        return y[index].exp().sum() + self.total


def graph_repeats(x, buffer):
    y = (x.sin() * 2 + 1).cos()
    buffer.add_(1)
    more = y.repeat_interleave((y[:, 0] > 0).sum(), 0)
    for _ in range(more.shape[0] // 4):
        buffer.add_(1)
    return more.sum() + buffer


def graph_finds(x, buffer):
    y = (x.sin() * 2 + 1).cos()
    buffer.add_(1)
    rows, columns = torch.where(y > 0)
    if rows.shape[0] > 6:
        buffer.add_(10)
    return rows.sum() + buffer + y.sum()


def graph_weighs(x):
    y = (x.sin() * 2 + 1).cos()
    where = torch.argwhere(y > 0)
    if where.shape[0] > 6:
        return (y * 2).exp().tanh().sum()
    return (y * 3).exp().tanh().sum() - 1


def graph_fills(x, fresh):
    y = (x.sin() * 2 + 1).cos()
    fresh.append(y.shape[0])
    fresh[0] = 7
    return (y * 2).exp().tanh().sum()


def graph_notes(x, buffer):
    y = (x.sin() * 2 + 1).cos()
    buffer.add_(y.mean())
    y.note = 3
    return (y + buffer).exp().tanh().sum() * y.note


def graph_remembers(x, w):
    y = x
    for _ in range(40):
        y = y * w
    z = y.exp() + 0
    z.add_(1)
    return (z * 2).sin().cos().tanh()


class GraphPooled(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(torch.relu(self.conv(x))), 1))


def graph_grid(a, b):
    rows, columns = torch.meshgrid(a.sin(), b.cos())
    return (rows * columns).tanh().exp() + 1


def graph_grid_named(a, b):
    rows, columns = torch.meshgrid(a.sin(), b.cos(), indexing="ij")
    return (rows * columns).tanh().exp() + 1


def graph_outer(x):
    return graph_inner(x)


class GraphDoubles:
    """What is read of it is twice what was stored."""

    def __getattribute__(self, key):
        return 2 * object.__getattribute__(self, key)


def graph_reads_what_it_stored(x, doubles):
    y = (x.sin() + 1).cos()
    doubles.scale = 1.5
    return (y * doubles.scale).tanh().exp()


class GraphMidwayIndex:
    def __index__(self):
        return 4


class GraphMidwaySized:
    def __init__(self, count):
        self.count = count

    def __len__(self):
        id(self)
        return self.count


class GraphMidwayAsked(type):
    answer = True

    def __instancecheck__(cls, obj):
        id(cls)
        return GraphMidwayAsked.answer


class GraphMidwayKind(metaclass=GraphMidwayAsked):
    pass


class GraphMidwaySteps:
    """An iterator that cannot be asked for an iterator."""

    def __init__(self, items):
        self.items = list(items)

    def __next__(self):
        if not self.items:
            raise StopIteration
        return self.items.pop(0)


class GraphMidwayOver:
    """Its `__iter__` gives what `given` makes; `log` says what was asked."""

    def __init__(self, given, midway=True):
        self.given = given
        self.midway = midway
        self.log = []

    def __iter__(self):
        self.log.append("iter")
        if self.midway:
            id(self)
        return self.given()


class GraphMidwayOverSized(GraphMidwayOver):
    def __len__(self):
        self.log.append("len")
        return 2


class GraphMidwayOverCalled(GraphMidwayOver):
    def __call__(self):
        self.log.append("call")
        return 1.0


class GraphFallsBack:
    """A property that raises AttributeError, and a `__getattr__`."""

    def __init__(self, missing):
        self.missing = missing

    @property
    def scale(self):
        if self.missing:
            raise AttributeError("the property says no")
        return 2.0

    def __getattr__(self, name):
        return 3.0


class GraphFallsBackModule(torch.nn.Module):
    def __init__(self, missing):
        super().__init__()
        self.missing = missing

    @property
    def scale(self):
        if self.missing:
            return self.nothing_of_that_name
        return 2.0

    def forward(self, x):
        return ((x.sin() + 1).cos() * self.scale).tanh().exp()


def graph_counts_midway(x, sized):
    y = (x.sin() + 1).cos()
    count = len(sized)
    return (y * 2).tanh().exp() + count, count, type(count).__name__


def graph_is_of_a_midway_kind(x):
    y = (x.sin() + 1).cos()
    if isinstance(3, GraphMidwayKind):
        y = y + 1
    return (y * 2).tanh().exp(), isinstance(3, GraphMidwayKind)


def graph_goes_over_midway(x, over, how):
    y = (x.sin() + 1).cos()
    made = None
    if how == "for":
        for step in over:
            y = y + step
    elif how == "list":
        made = list(over)
    elif how == "tuple":
        made = tuple(over)
    elif how == "zip":
        made = list(zip(over, [5, 6]))
    elif how == "enumerate":
        made = list(enumerate(over, 3))
    elif how == "iter":
        made = list(iter(over))
    else:
        made = next(iter(over, 2.0))
    return (y * 2).tanh().exp(), made


def graph_reads_what_falls_back(x, obj):
    y = (x.sin() + 1).cos()
    return (y * obj.scale).tanh().exp(), obj.scale


class GraphCollides:
    """A key whose hash is that of "a", and that says when it is compared."""

    def __init__(self):
        self.log = []

    def __hash__(self):
        return hash("a")

    def __eq__(self, other):
        self.log.append("eq")
        return False


def graph_looks_up_a_key(x, table, members, asked, flag):
    y = (x.sin() + 1).cos()
    scale = table["a"]
    there = "a" in table, "a" in members
    got = table.get("a")
    if flag:
        y = y + 1
    return (y * scale).tanh().exp(), there, got


def graph_reads_a_table(x, table, members):
    y = x.sin()
    scale = table["a"]
    there = "a" in table, "a" in members
    return ((y + 1).cos() * scale).tanh().exp(), there


class GraphAnswers:
    """What the interpreter asks a class for an operator: each says so."""

    def __init__(self, n):
        self.n = n
        self.log = []

    def __bool__(self):
        self.log.append("bool")
        return self.n > 1

    def __eq__(self, other):
        self.log.append("eq")
        if isinstance(other, int):
            return other == 1 or NotImplemented
        return self.n == other.n

    def __lt__(self, other):
        self.log.append("lt")
        return NotImplemented

    def __gt__(self, other):
        self.log.append("gt")
        return self.n > other.n

    def __getitem__(self, at):
        self.log.append("getitem")
        if at > 1:
            raise IndexError(at)
        return float(at + self.n)

    def __enter__(self):
        self.log.append("enter")
        return 2.0

    def __exit__(self, kind, value, where):
        self.log.append("exit")
        return self.n > 1

    def __index__(self):
        self.log.append("index")
        return 1

    def __int__(self):
        self.log.append("int")
        return 2

    def __float__(self):
        self.log.append("float")
        return 2.5

    def __hash__(self):
        self.log.append("hash")
        return hash(1)

    def __add__(self, other):
        self.log.append("add")
        return other + 1.5


def graph_asks_for_operators(x, a, b):
    y = (x.sin() + 1).cos()
    if a:
        y = y + 1
    if a == b:
        y = y * 2
    if a < b:
        y = y * 3
    for item in a:
        y = y + item
    with a as scale:
        y = y * scale
        if a.n > 1:
            raise ValueError("inside")
        y = y + 1
    y = y * [1.0, 2.0, 3.0][a] + int(a) + float(a) + {1: 4.0}[a]
    total = a
    total += 2.0
    return (y * total).tanh().exp(), a == 3, 1.0 in a, list(range(a)), not a


def graph_parts(x, k):
    a = (x.sin() + 1).cos().tanh().exp()
    # Two tensors, or none: an operation of a graph only where it gives
    # some.
    cols = a[:, :k].unbind(1)
    b = a * 2
    if cols:
        b = b + cols[0].unsqueeze(1)
    return b.sin().cos().tanh()


def graph_asks(x, y):
    a = (x.sin() + y).cos().tanh().exp()
    same = a.equal(y)
    b = (a * 2).sin().cos().tanh()
    return b + 1 if same else b - 1


def graph_tops(x):
    a = (x.sin() + 1).cos().tanh().exp()
    top = a.max().item()
    return (a * top).sin().cos().tanh().exp()


def graph_halves(x):
    a = (x.sin() + 1).cos().tanh().exp()
    return a.chunk(2)


def graph_rows(x):
    a = (x.sin() + 1).cos().tanh().exp()
    total = a.sum()
    for row in a.unbind(0):
        total = total + row.mean()
    return total


def graph_gathers(x):
    outs = []
    for k in range(3):
        outs.append((x + k).sin())
    total = x.cos()
    for out in outs:
        total = total + out
    return total.tanh()


def graph_probed(x, n):
    y = x + 1
    if n > 2:
        y = y * 2
    count = int(random.random())
    return (y + count).sin().cos().tanh().exp()


def graph_sized(x, y):
    a = (x.sin() + 1).cos()
    if a.shape != y.shape:
        raise ValueError("sizes")
    b = (a * y).tanh()
    if b.shape == (2, 3):
        b = b + 1
    return b.exp()


def graph_children(model, x):
    for layer in model.children():
        x = layer(x)
    return x


def graph_paths(case, device):
    """A function whose graph goes through its blocks, and calls of it."""
    make = functools.partial(torch.randn, device=device)
    if case == "branches":
        flags = (True, True, True, False, False, True)
        return graph_branches, [(make(4), flag) for flag in flags]
    if case == "loops":
        return graph_loops, [(make(4), n) for n in (20, 20, 20, 12, 12, 20)]
    if case == "collects":
        counts = (3, 3, 3, 2, 2, 3)
        return graph_collects, [([make(3) for _ in range(n)],) for n in counts]
    return graph_on_a_size, [(make(n),) for n in (4, 4, 4, 2, 2, 4)]


GRAPHS = (*DEFAULT, P.Graphs())


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmGraphs(TestCase):
    """The graphs the engine makes of the tensor work of a block, and the
    passes over them."""

    hw_classification = HardwareClassification.GENERIC

    def graphs(self, fn, state=None):
        found = [r for r in torch.vm.regions(fn) if "ops" in r]
        return [r for r in found if state is None or r["state"] == state]

    def left(self, fn):
        return [r["reason"] for r in torch.vm.regions(fn) if "ops" not in r]

    def targets(self, gm):
        return [n.target for n in gm.graph.nodes if n.op == "call_function"]

    def test_what_an_instruction_did_is_taken_down(self):
        compiled = torch.vm(graph_facts, fallback=False)
        x, probe = torch.ones(3), GraphProbe()
        expected = graph_facts(x, GraphProbe(), graph_plus_one)
        for _ in range(3):
            self.assertEqual(compiled(x, probe, graph_plus_one), expected)
        facts = compiled.facts()
        self.assertTrue(all(fact["runs"] >= 1 for fact in facts))

        def one(text):
            (found,) = [fact for fact in facts if text in fact["instruction"]]
            return found

        # A property is a frame; an attribute of the object is neither a
        # frame nor a kernel.
        scale, plain = one("'scale'"), one("'plain'")
        self.assertEqual((scale["kernels"], scale["frames"]), ("none", True))
        self.assertEqual((plain["kernels"], plain["frames"]), ("none", False))
        calls = [fact for fact in facts if fact["instruction"].startswith("call ")]
        entered = [fact["callee"] for fact in calls if fact["callee"] is not None]
        found = [(callee["name"], callee["kind"]) for callee in entered]
        expected = [
            ("GraphProbe.shift", "method"),
            ("GraphProbe.__call__", "instance"),
            ("graph_plus_one", "function"),
        ]
        self.assertEqual(found, expected)
        self.assertIs(entered[2]["code"], graph_plus_one.__code__)
        self.assertTrue(all(fact["frames"] for fact in calls if fact["callee"]))
        # len: a call that runs no kernel and enters no frame.
        quiet = [f for f in calls if f["kernels"] == "none" and not f["frames"]]
        self.assertEqual(len(quiet), 1)
        # The call that writes to its receiver is the one of add_.
        (wrote,) = [fact for fact in facts if fact["wrote"]]
        self.assertEqual((wrote["kernels"], wrote["callee"]), ("one", None))
        self.assertIn(wrote, calls)
        # x * probe is one kernel, in a frame that C code entered while the
        # operation was looked at: the interpreter ran it.
        reflected = one("binary * %x.1 %probe")
        self.assertEqual((reflected["kernels"], reflected["frames"]), ("one", True))
        self.assertEqual(probe.offered[0], False)
        self.assertFalse(any(fact["raised"] for fact in facts))

    def test_a_pass_over_a_block_is_given_the_facts(self):
        seen = {}

        def look(region):
            for inst in region.instructions():
                seen[inst["text"].split("    ;")[0]] = inst["facts"]
            self.assertEqual(region.observed(2)["kind"], "object")
            return False

        compiled = torch.vm(graph_facts, passes=(*DEFAULT, look), fallback=False)
        compiled(torch.ones(3), GraphProbe(), graph_plus_one)
        self.assertEqual(seen["load_attr %probe.2 'scale'"]["frames"], True)
        self.assertEqual(seen["load_attr %probe.2 'plain'"]["frames"], False)
        self.assertEqual(seen["load_global 'len'"]["kernels"], "none")

    def test_a_call_site_says_what_a_graph_needs_of_it(self):
        x = torch.randn(3)
        made = torch._C._cfg_make_site
        site = made("binary", lambda: x * 2, (x, 2), constants=(False, True))[1]
        found = site.export()
        self.assertEqual((found["op"], found["overload"]), ("aten::mul", "Tensor"))
        operand = ("self", False, "operand", (0, "tensor"))
        self.assertEqual(found["args"], [operand, ("other", False, "const", 2)])
        self.assertEqual(found["guards"], ["tensor", "none"])
        self.assertEqual((found["returns"], found["writes"]), ("tensor", False))
        # A dtype is the object, not the number the kernel takes.
        double = torch.float64
        to = made("method", lambda: x.to(double), (x, double), callee=x.to)[1]
        self.assertIn(("dtype", False, "const", double), to.export()["args"])
        self.assertEqual(to.export()["guards"], ["tensor", "identity"])
        y = torch.randn(3)
        add_ = made("method", lambda: y.add_(1), (y, 1), callee=y.add_)[1]
        self.assertEqual(add_.export()["writes"], True)
        self.assertEqual(add_.info()["writes"], True)
        drop = torch.dropout
        operands = (drop, x, 0.5, True)
        drawn = made("call", lambda: drop(x, 0.5, True), operands, callee=drop)[1]
        self.assertEqual(drawn.export()["random"], True)
        self.assertEqual(site.info()["random"], False)
        positive = x.sum() > 0
        truth = made("unary", lambda: bool(positive), (positive,))[1]
        self.assertEqual(truth.info()["returns"], "bool")

    def test_a_block_is_one_graph(self):
        compiled = torch.vm(graph_poly, passes=GRAPHS, fallback=False)
        for _ in range(4):
            x = torch.randn(5)
            self.assertEqual(compiled(x), graph_poly(x), atol=0, rtol=0)
        (graph,) = torch.vm.regions(compiled)
        self.assertEqual(graph["state"], "built")
        self.assertTrue(graph["where"].startswith("graph_poly (block 0)["))
        operators = ["aten::sin", "aten::cos", "aten::mul.Tensor", "aten::add.Tensor"]
        self.assertEqual((graph["ops"], graph["operators"]), (4, operators))
        self.assertEqual((graph["inputs"], graph["outputs"]), (1, 1))
        # The engine runs it as the block: nothing stands in front of it.
        self.assertEqual((graph["variants"], graph["hits"]), (0, 0))
        self.assertNotIn("graph ", compiled.dump())
        # Without the pass there are no graphs.
        plain = torch.vm(graph_poly, fallback=False)
        plain(torch.randn(5))
        self.assertEqual(torch.vm.regions(plain), [])

    def test_what_is_no_part_of_a_graph_ends_it(self):
        compiled = torch.vm(graph_two, passes=GRAPHS, fallback=False)
        x = torch.randn(6)
        for _ in range(3):
            self.assertEqual(compiled(x), graph_two(x), atol=0, rtol=0)
        self.assertEqual([g["ops"] for g in self.graphs(compiled)], [4, 6])
        self.assertEqual(self.left(compiled), ["a call that runs no kernel"])
        (left,) = [r for r in torch.vm.regions(compiled) if r["state"] == "vm"]
        self.assertIn("call ", left["text"])
        # An operation that writes to a tensor the function made itself is
        # an operation of the graph, and so is one that writes to what the
        # function was given: see
        # test_a_write_to_a_tensor_from_outside_is_done_once.
        written = torch.vm(graph_written, passes=GRAPHS, fallback=False)
        for _ in range(3):
            self.assertEqual(written(x), graph_written(x), atol=0, rtol=0)
        (graph,) = self.graphs(written)
        self.assertEqual(graph["ops"], 9)
        self.assertIn("aten::add_.Tensor", graph["operators"])
        # Before a path is taken down, the block is the runs on either side
        # of the write, and a run of fewer operations than min_ops is none.
        small = (*DEFAULT, P.Graphs(min_ops=2))
        written = torch.vm(graph_written, passes=small, fallback=False)
        written(x)
        self.assertEqual([g["ops"] for g in self.graphs(written)], [3, 5])
        self.assertEqual(len(self.left(written)), 1)

    def test_the_graphs_of_what_a_function_calls_are_reported(self):
        torch.manual_seed(0)
        model = GraphNet().eval()

        def infer(model, x):
            return model(x)

        passes = (*DEFAULT, P.Graphs(min_ops=1))
        compiled = torch.vm(infer, passes=passes, fallback=False)
        x = torch.randn(2, 3, 6, 6)
        with torch.no_grad():
            for _ in range(4):
                self.assertEqual(compiled(model, x), model(x), atol=0, rtol=0)
        operators = [op for g in self.graphs(compiled) for op in g["operators"]]
        for expected in ("aten::conv2d", "aten::batch_norm", "aten::linear"):
            self.assertIn(expected, operators)
        where = {g["where"].split(" (")[0] for g in self.graphs(compiled)}
        self.assertIn("GraphNet.forward", where)
        # A call of a Python function that a path goes through is part of
        # the graph, and is not listed as left out.
        self.assertNotIn("calls Conv2d.forward", self.left(compiled))

    def test_no_graph_compiler_is_imported_for_the_graphs_of_the_engine(self):
        script = """
import json, sys, torch
P, DEFAULT = torch.vm.passes, torch.vm.DEFAULT_PASSES

def f(x, w):
    return torch.tanh(x @ w).sum() + (x * 2).mean()

def loaded(*more):
    names = ("torch._inductor", "torch._functorch.aot_autograd", "torch._dynamo")
    names += ("torch.vm._regions", *more)
    return [name for name in names if name in sys.modules]

x, w = torch.randn(4, 8), torch.randn(8, 8, requires_grad=True)
found = {}
compiled = torch.vm(f, passes=(*DEFAULT, P.Graphs()), fallback=False)
for _ in range(4):
    out = compiled(x, w)
out.backward()
found["graphs"] = [r["state"] for r in torch.vm.regions(compiled)]
found["engine"] = loaded("torch.vm._export")
keep = lambda gm, input_types: gm
compiled = torch.vm(f, passes=(*DEFAULT, keep), fallback=False)
for _ in range(4):
    compiled(x, w)
found["exported"] = [r["state"] for r in torch.vm.regions(compiled)]
found["pass"] = loaded()
compiled = torch.vm(f, passes=(*DEFAULT, P.Graphs(), P.AOTAutograd()), fallback=False)
found["listed"] = loaded()
for _ in range(4):
    compiled(x, w)
found["compiled"] = loaded()
print(json.dumps(found))
"""
        ran = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        found = __import__("json").loads(ran.stdout.strip().split("\n")[-1])
        self.assertEqual(found["graphs"], ["built"])
        self.assertEqual(found["engine"], [])
        # A pass over graphs of one's own brings the export, and nothing
        # else; a pass that compiles brings its compiler when it first runs.
        self.assertEqual(found["exported"], ["compiled"])
        self.assertEqual((found["pass"], found["listed"]), ([], []))
        self.assertIn("torch._functorch.aot_autograd", found["compiled"])
        self.assertNotIn("torch.vm._regions", found["compiled"])

    def test_a_pass_over_a_graph_rewrites_it(self):
        seen = []

        def sigmoid_for_tanh(gm, input_types):
            seen.append((self.targets(gm), input_types))
            for node in gm.graph.nodes:
                if node.target is torch.ops.aten.tanh.default:
                    node.target = torch.ops.aten.sigmoid.default
                    seen.append(torch.vm.type_of(gm, input_types, node))
            return gm

        passes = (*DEFAULT, P.Graphs(), sigmoid_for_tanh)
        compiled = torch.vm(graph_mixed, passes=passes, fallback=False)
        w = torch.randn(8, 8)
        for n in (4, 4, 4, 4, 6, 6, 6):
            x = torch.randn(n, 8)
            got = compiled(x, w)
        # What the pass made of the graph runs in place of its operations.
        expected = torch.sigmoid(x @ w).sum() + (x * 2).mean()
        self.assertEqual(got, expected, atol=0, rtol=0)
        (graph,) = self.graphs(compiled)
        # Exported once for each size, on the first call of the block that
        # ran its operations directly.
        self.assertEqual((graph["state"], graph["variants"]), ("compiled", 2))
        self.assertEqual((graph["hits"], graph["misses"]), (4, 2))
        self.assertIn("graph graph_mixed (block 0)", compiled.dump())
        targets, (x_type, w_type) = seen[0]
        aten = torch.ops.aten
        expected = [aten.matmul.default, aten.tanh.default, aten.sum.default]
        expected += [aten.mul.Tensor, aten.mean.default, aten.add.Tensor]
        self.assertEqual(targets, expected)
        self.assertIsInstance(x_type, torch.vm.TensorType)
        self.assertEqual((x_type.dtype, x_type.device), (torch.float32, x.device))
        said = (repr(x_type), repr(w_type))
        self.assertEqual(said, ("float32[4, 8]", "float32[8, 8]"))
        # An intermediate has the type it had in the call that was exported.
        self.assertEqual(repr(seen[1]), "float32[4, 8]")
        self.assertEqual(repr(seen[2][1][0]), "float32[6, 8]")

    def test_a_pass_over_a_graph_brings_its_guards(self):
        def only_small(gm, input_types):
            x = next(node for node in gm.graph.nodes if node.op == "placeholder")
            torch.vm.guard(gm, x, torch.vm.Dim(0, range(1, 6)))
            for node in gm.graph.nodes:
                if node.target is torch.ops.aten.tanh.default:
                    node.target = torch.ops.aten.sigmoid.default
            return gm

        passes = (*DEFAULT, P.Graphs(), only_small)
        compiled = torch.vm(graph_mixed, passes=passes, fallback=False)
        w = torch.randn(8, 8)
        for n in (4, 4, 4, 8, 8, 8, 8, 4):
            x = torch.randn(n, 8)
            got = compiled(x, w)
            if n == 8:
                # A call that fails the guard runs what the function wrote.
                self.assertEqual(got, graph_mixed(x, w), atol=0, rtol=0)
        self.assertEqual(got, torch.sigmoid(x @ w).sum() + (x * 2).mean())
        (graph,) = self.graphs(compiled, "compiled")
        self.assertEqual((graph["variants"], graph["hits"]), (2, 2))

    def test_a_graph_is_exported_for_the_values_its_operations_take(self):
        graphs = []

        def keep(gm, input_types):
            graphs.append(self.targets(gm))
            return gm

        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_reads, passes=passes, fallback=False)
        x = torch.randn(5)
        outer = (graph_ops.act, graph_ops.scale)
        try:
            for _ in range(4):
                self.assertEqual(compiled(x), graph_reads(x), atol=0, rtol=0)
            self.assertEqual(len(graphs), 1)
            self.assertIn(torch.ops.aten.tanh.default, graphs[0])
            # Another function, and then another number, where the block
            # reads them: the graph that has the old ones is not run.
            graph_ops.act = torch.sigmoid
            for _ in range(3):
                self.assertEqual(compiled(x), graph_reads(x), atol=0, rtol=0)
            self.assertIn(torch.ops.aten.sigmoid.default, graphs[1])
            graph_ops.scale = 0.25
            for _ in range(3):
                self.assertEqual(compiled(x), graph_reads(x), atol=0, rtol=0)
            self.assertEqual(len(graphs), 3)
            graph_ops.act, graph_ops.scale = outer
            hits = self.graphs(compiled)[0]["hits"]
            self.assertEqual(compiled(x), graph_reads(x), atol=0, rtol=0)
            self.assertEqual(self.graphs(compiled)[0]["hits"], hits + 1)
        finally:
            graph_ops.act, graph_ops.scale = outer
        self.assertEqual(len(graphs), 3)

    def test_a_graph_is_exported_for_the_modes_and_the_tensors_of_a_call(self):
        graphs = []

        def keep(gm, input_types):
            graphs.append([(t.dtype, t.requires_grad) for t in input_types])
            return gm

        passes = (*DEFAULT, P.Graphs(max_variants=3), keep)
        compiled = torch.vm(graph_poly, passes=passes, fallback=False)
        f32, f64 = torch.float32, torch.float64
        x, d, g = torch.randn(4), torch.randn(4, dtype=f64), torch.randn(4)
        g.requires_grad_()
        for value in (x, x, x, d, d, g, g, x, d):
            self.assertEqual(compiled(value), graph_poly(value), atol=0, rtol=0)
        kinds = [[(f32, False)], [(f64, False)], [(f32, True)]]
        self.assertEqual(graphs, kinds)
        # The same tensors with gradients off are a fourth kind of call,
        # one more than the graph is exported for: its operations run.
        with torch.no_grad():
            for _ in range(3):
                self.assertEqual(compiled(g), graph_poly(g), atol=0, rtol=0)
        self.assertEqual(len(graphs), 3)
        (graph,) = self.graphs(compiled, "compiled")
        # The three calls that were exported ran the operations as well.
        self.assertEqual((graph["variants"], graph["misses"]), (3, 6))
        # A view with other strides is not the tensor the graph was made
        # for either.
        strided = torch.randn(8)[::2]
        self.assertEqual(compiled(strided), graph_poly(strided), atol=0, rtol=0)
        self.assertEqual(self.graphs(compiled)[0]["misses"], 7)

    def test_a_call_whose_operations_write_to_what_they_are_given(self):
        graphs = []

        def keep(gm, input_types):
            graphs.append(self.targets(gm))
            return gm

        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_normed, passes=passes, fallback=False)
        ours, theirs = (torch.zeros(4), torch.ones(4)), (torch.zeros(4), torch.ones(4))
        for training in (True, True, True, True, False, False, False, True, False):
            x = torch.randn(2, 4, 5, 5)
            got = compiled(x, *ours, training)
            self.assertEqual(got, graph_normed(x, *theirs, training), atol=0, rtol=0)
            # The running statistics, which a batch norm that trains
            # updates without their version counters moving.
            self.assertEqual(ours, theirs, atol=0, rtol=0)
        self.assertEqual(ours[0]._version, 0)
        # Exported for the calls that train as well: the graph writes the
        # statistics, which are tensors it is given.
        self.assertEqual(len(graphs), 2)
        self.assertIn(torch.ops.aten.batch_norm.default, graphs[0])
        self.assertIn(torch.ops.aten.batch_norm.default, graphs[1])
        (graph,) = self.graphs(compiled, "compiled")
        self.assertEqual((graph["variants"], graph["hits"]), (2, 6))

    def test_what_a_dispatch_mode_is_to_see_is_not_run_as_a_graph(self):
        def sigmoid_for_tanh(gm, input_types):
            for node in gm.graph.nodes:
                if node.target is torch.ops.aten.tanh.default:
                    node.target = torch.ops.aten.sigmoid.default
            return gm

        passes = (*DEFAULT, P.Graphs(), sigmoid_for_tanh)
        compiled = torch.vm(graph_mixed, passes=passes, fallback=False)
        x, w = torch.randn(4, 8), torch.randn(8, 8)
        for _ in range(4):
            got = compiled(x, w)
        self.assertEqual(got, torch.sigmoid(x @ w).sum() + (x * 2).mean())
        # Under a mode the operations of the function run, and it sees them.
        with GraphSeen() as mode:
            self.assertEqual(compiled(x, w), graph_mixed(x, w), atol=0, rtol=0)
        self.assertIn(torch.ops.aten.tanh.default, mode.seen)
        self.assertNotIn(torch.ops.aten.sigmoid.default, mode.seen)
        self.assertEqual(compiled(x, w), got, atol=0, rtol=0)

    def test_the_integers_of_a_graph_are_computed_in_it(self):
        graphs = []

        def keep(gm, input_types):
            graphs.append(gm)
            return gm

        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_sizes, passes=passes, fallback=False)
        w = torch.randn(8, 8)
        for n in (4, 4, 4, 6, 6):
            x = torch.randn(n, 8)
            self.assertEqual(compiled(x, w), graph_sizes(x, w), atol=0, rtol=0)
        self.assertEqual(self.graphs(compiled)[0]["variants"], 2)
        targets = self.targets(graphs[0])
        self.assertIn(torch.ops.aten.sym_size.int, targets)
        self.assertIn(__import__("operator").mul, targets)
        # The graph gives the sum. The ranks that the block checks of the
        # two tensors the graph keeps to itself follow from the sizes of
        # what it is given, which the form checks: they are not asked of
        # the graph again.
        (given,) = [n.args[0] for n in graphs[0].graph.nodes if n.op == "output"]
        said = [node.target for node in given]
        self.assertEqual(said, [torch.ops.aten.add.Tensor])

    def test_a_pass_that_fails_leaves_the_graph_to_the_engine(self):
        def broken(gm, input_types):
            raise RuntimeError("not today")

        def wrong(gm, input_types):
            return gm.forward

        x, w = torch.randn(4, 8), torch.randn(8, 8)
        reasons = []
        for pass_ in (broken, wrong):
            passes = (*DEFAULT, P.Graphs(), pass_)
            compiled = torch.vm(graph_mixed, passes=passes, fallback=False)
            for _ in range(4):
                self.assertEqual(compiled(x, w), graph_mixed(x, w), atol=0, rtol=0)
            (graph,) = self.graphs(compiled)
            self.assertEqual((graph["state"], graph["ops"]), ("vm", 6))
            reasons.append(graph["reason"])
        self.assertEqual(reasons[0], "the pass broken raised RuntimeError: not today")
        self.assertEqual(reasons[1], "the pass wrong returned a method, not the graph")

    def test_what_was_made_of_a_graph_and_fails_is_taken_back(self):
        calls = []

        def failing(gm, input_types):
            def run(*args):
                calls.append(len(args))
                raise RuntimeError("no kernel for this")

            return torch.vm._export.calling(gm, run)

        passes = (*DEFAULT, P.Graphs(), failing)
        compiled = torch.vm(graph_mixed, passes=passes, fallback=False)
        x, w = torch.randn(4, 8), torch.randn(8, 8)
        for _ in range(6):
            self.assertEqual(compiled(x, w), graph_mixed(x, w), atol=0, rtol=0)
        # It ran once, raised, and the operations ran instead, then and
        # from then on.
        self.assertEqual(calls, [2])
        (graph,) = self.graphs(compiled)
        self.assertEqual(graph["state"], "vm")
        failed = "the compiled graph failed: RuntimeError: no kernel for this"
        self.assertEqual(graph["reason"], failed)

    def test_the_passes_over_a_graph_go_behind_the_graph_pass(self):
        def keep(gm, input_types):
            return gm

        f = graph_mixed
        aot, inductor = P.AOTAutograd(), P.Inductor()
        with self.assertRaisesRegex(ValueError, "keep in front of .*Graphs"):
            torch.vm(f, passes=(*DEFAULT, keep, P.Graphs()))
        with self.assertRaisesRegex(ValueError, "AOTAutograd.. in front of"):
            torch.vm(f, passes=(aot, *DEFAULT, P.Graphs()))
        with self.assertRaisesRegex(ValueError, "a graph is compiled once"):
            torch.vm(f, passes=(*DEFAULT, P.Graphs(), aot, inductor))
        with self.assertRaisesRegex(ValueError, "keep behind .*Inductor"):
            torch.vm(f, passes=(*DEFAULT, P.Graphs(), inductor, keep))
        with self.assertRaisesRegex(ValueError, "Inductor is given more than once"):
            torch.vm(f, passes=(*DEFAULT, P.Graphs(), inductor, P.Inductor()))
        with self.assertRaisesRegex(TypeError, "a pass of its own, behind it"):
            P.Graphs("inductor")
        with self.assertRaisesRegex(ValueError, "at least 1"):
            P.Graphs(min_ops=0)
        # Without the graph pass, the passes over a graph have it in front
        # of them.
        compiled = torch.vm(f, passes=(*DEFAULT, keep), fallback=False)
        self.assertIn("graphs", compiled.passes)
        x, w = torch.randn(4, 8), torch.randn(8, 8)
        for _ in range(3):
            self.assertEqual(compiled(x, w), f(x, w), atol=0, rtol=0)
        self.assertEqual(len(self.graphs(compiled, "compiled")), 1)
        # Where the graph pass stands among the passes over a block is all
        # the same.
        compiled = torch.vm(f, passes=(P.Graphs(), *DEFAULT), fallback=False)
        compiled(x, w)
        self.assertEqual(len(self.graphs(compiled, "built")), 1)

    def test_what_the_passes_over_a_graph_say_of_themselves(self):
        self.assertEqual(repr(P.Graphs()), "torch.vm.passes.Graphs()")
        said = "torch.vm.passes.Graphs(min_ops=2)"
        self.assertEqual(repr(P.Graphs(min_ops=2)), said)
        self.assertEqual(repr(P.Inductor()), "torch.vm.passes.Inductor()")
        for one in (P.AOTAutograd(), P.Inductor()):
            self.assertEqual((one.kind, one.exact), ("graph", False))
        # The engine runs its graphs with the kernels of eager mode. What a
        # pass over them returns is something else.
        self.assertTrue(torch.vm.is_exact(GRAPHS))
        self.assertFalse(torch.vm.is_exact((*GRAPHS, P.AOTAutograd())))
        self.assertFalse(torch.vm.is_exact((*GRAPHS, P.Inductor())))
        self.assertFalse(torch.vm.is_exact((*GRAPHS, lambda gm, input_types: gm)))

    def test_a_pass_that_compiles_returns_a_graph_that_calls_what_it_made(self):
        graph = torch.fx.Graph()
        x, w = graph.placeholder("x"), graph.placeholder("w")
        product = graph.call_function(torch.ops.aten.matmul.default, (x, w))
        graph.output((graph.call_function(torch.ops.aten.tanh.default, (product,)),))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        cpu = torch.device("cpu")
        sizes = lambda *ns: tuple(torch.vm.Interval(n, n) for n in ns)  # noqa: E731
        types = [
            torch.vm.TensorType(torch.float32, cpu, False, sizes(4, 8)),
            torch.vm.TensorType(torch.float32, cpu, True, sizes(8, 8)),
        ]
        made = P.AOTAutograd()(gm, types)
        self.assertIsInstance(made, torch.fx.GraphModule)
        nodes = [node.op for node in made.graph.nodes]
        self.assertEqual(nodes, ["placeholder", "placeholder", "call_module", "output"])
        a, b = torch.randn(4, 8), torch.randn(8, 8, requires_grad=True)
        (out,) = made(a, b)
        self.assertEqual(out, torch.tanh(a @ b))
        self.assertEqual(type(out.grad_fn).__name__, "CompiledFunctionBackward")

    # --- values that are read, kinds of calls, operations that are no call sites

    def forms(self, fn, operator):
        """The checks and the hits of the forms of the graph of ``fn`` that
        has an operator."""
        (graph,) = [g for g in self.graphs(fn) if operator in g["operators"]]
        return graph, [(form["checks"], form["hits"]) for form in graph["forms"]]

    def test_a_number_a_module_keeps_is_checked_where_a_graph_takes_it(self):
        torch.manual_seed(0)
        x = torch.randn(3, 6)
        for behind in ((), (P.Inductor(),)):
            model = GraphLeaky().eval()
            passes = (*DEFAULT, P.Graphs(min_ops=1), *behind)
            compiled = torch.vm(graph_infer, passes=passes, fallback=False)
            with torch.no_grad():
                for _ in range(5):
                    first = compiled(model, x)
                self.assertEqual(first, model(x), rtol=1e-5, atol=1e-6)
                steady = compiled.stats()["graphs"]
                # The layer is the same object, and its slope another number.
                model.act.negative_slope = 0.3
                for _ in range(3):
                    second = compiled(model, x)
                self.assertEqual(second, model(x), rtol=1e-5, atol=1e-6)
                self.assertNotEqual(first, second)
                changed = compiled.stats()["graphs"]
                model.act.negative_slope = 0.1
                self.assertEqual(compiled(model, x), first, atol=0, rtol=0)
                back = compiled.stats()["graphs"]
            # The graph is that of the function, which holds both calls of
            # the layer; the graph of the layer's own function waits.
            graphs = self.graphs(compiled)
            (graph,) = [g for g in graphs if g["where"].startswith("graph_infer")]
            self.assertEqual(graph["operators"].count("aten::leaky_relu"), 2)
            forms = [(form["checks"], form["hits"]) for form in graph["forms"]]
            if not behind:
                # The engine runs the operation with what it reads.
                self.assertEqual((graph["state"], back["exported"]), ("built", 0))
                continue
            self.assertEqual(graph["state"], "compiled")
            # A form with the slope it was exported for, which both calls
            # of the layer read, and then one more for the other.
            self.assertEqual(changed["exported"], steady["exported"] + 1)
            said = [check for checks, _ in forms for check in checks]
            self.assertEqual(sum("== 0.1" in check for check in said), 2)
            self.assertEqual(sum("== 0.3" in check for check in said), 2)
            # The first slope again is the forms that were made for it.
            self.assertEqual(back["exported"], changed["exported"])
            self.assertGreater(back["hits"], changed["hits"])
            self.assertEqual(back["misses"], changed["misses"])

    def test_a_global_and_a_variable_of_an_enclosing_function_are_checked(self):
        def keep(gm, input_types):
            return gm

        def set_global(value):
            globals()["GRAPH_SCALE"] = value

        x = torch.randn(5)
        try:
            for behind in (keep, P.Inductor()):
                run, rebind = graph_enclosed(0.5)
                for fn, change in ((graph_global, set_global), (run, rebind)):
                    passes = (*DEFAULT, P.Graphs(), behind)
                    compiled = torch.vm(fn, passes=passes, fallback=False)
                    exact = {} if behind is keep else {"rtol": 1e-5, "atol": 1e-6}
                    for _ in range(4):
                        first = compiled(x)
                    self.assertEqual(first, fn(x), **exact)
                    change(0.25)
                    for _ in range(3):
                        second = compiled(x)
                    self.assertEqual(second, fn(x), **exact)
                    self.assertNotEqual(first, second)
                    graph, forms = self.forms(compiled, "aten::tanh")
                    self.assertEqual((graph["exports"], graph["variants"]), (2, 2))
                    # The form that ran last is the first one.
                    self.assertTrue(any("== 0.25" in check for check in forms[0][0]))
                    self.assertTrue(any("== 0.5" in check for check in forms[1][0]))
                    hits = graph["hits"]
                    change(0.5)
                    self.assertEqual(compiled(x), first, atol=0, rtol=0)
                    graph, forms = self.forms(compiled, "aten::tanh")
                    self.assertEqual((graph["exports"], graph["hits"]), (2, hits + 1))
                    self.assertTrue(any("== 0.5" in check for check in forms[0][0]))
        finally:
            set_global(0.5)

    def test_a_read_that_runs_python_code_is_not_run_out_of_its_place(self):
        graphs = []

        def keep(gm, input_types):
            graphs.append(gm)
            return gm

        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_boxed, passes=passes, fallback=False)
        x, box, theirs = torch.randn(5), GraphBox(), GraphBox()
        for _ in range(4):
            self.assertEqual(compiled(x, box), graph_boxed(x, theirs), atol=0, rtol=0)
        (graph,) = self.graphs(compiled)
        found = (graph["state"], graph["hits"], len(graphs))
        self.assertEqual(found, ("compiled", 2, 1))
        # Without the attribute, a read is a call of a Python function, which
        # counts its calls and gives another number each time: it runs once
        # per read, where the function has it.
        del box.scale, theirs.scale
        for _ in range(6):
            self.assertEqual(compiled(x, box), graph_boxed(x, theirs), atol=0, rtol=0)
            self.assertEqual(box.made, theirs.made)
        self.assertEqual(box.made, 12)
        (graph,) = self.graphs(compiled)
        self.assertEqual((graph["state"], len(graphs)), ("vm", 1))
        self.assertEqual(graph["reason"], "load_attr could run Python code")
        self.assertEqual(compiled.stats()["graphs"]["given_up"], 1)

    def test_numbers_that_are_new_on_every_call_are_not_exported_for(self):
        graphs = []

        def keep(gm, input_types):
            graphs.append(gm)
            return gm

        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_counter, passes=passes, fallback=False)
        x = torch.randn(4)
        for i in range(50):
            scale = i / 7
            got = compiled(x, scale)
            self.assertEqual(got, graph_counter(x, scale), atol=0, rtol=0)
        # Three forms that never ran again, and a fourth number: no more.
        self.assertEqual(len(graphs), 3)
        (graph,) = self.graphs(compiled)
        reason = "its inputs take ever new values"
        self.assertEqual((graph["state"], graph["reason"]), ("vm", reason))
        counts = (graph["variants"], graph["exports"], graph["hits"])
        self.assertEqual(counts, (0, 3, 0))
        # The calls of this kind are told by their tensor alone.
        ((checks, _),) = self.forms(compiled, "aten::exp")[1]
        self.assertEqual(checks, ["%x.1: float32[4]"])
        # Another kind of call is exported as before.
        wide = torch.randn(4, dtype=torch.float64)
        for _ in range(3):
            got = compiled(wide, 0.5)
            self.assertEqual(got, graph_counter(wide, 0.5), atol=0, rtol=0)
        (graph,) = self.graphs(compiled)
        found = (graph["state"], graph["variants"], graph["hits"])
        self.assertEqual(found, ("compiled", 1, 2))
        self.assertEqual(len(graphs), 4)
        # Three numbers in turn are three forms, which run.
        graphs.clear()
        compiled = torch.vm(graph_counter, passes=passes, fallback=False)
        for i in range(13):
            scale = (0.5, 0.25, 0.125)[i % 3]
            got = compiled(x, scale)
            self.assertEqual(got, graph_counter(x, scale), atol=0, rtol=0)
        (graph,) = self.graphs(compiled)
        self.assertEqual((len(graphs), graph["variants"], graph["hits"]), (3, 3, 9))

    def test_the_forms_of_a_graph_are_tried_in_the_order_they_last_ran(self):
        graphs = []

        def keep(gm, input_types):
            graphs.append(input_types[0].dtype)
            return gm

        passes = (*DEFAULT, P.Graphs(max_variants=2), keep)
        compiled = torch.vm(graph_poly, passes=passes, fallback=False)
        single, double = torch.randn(4), torch.randn(4, dtype=torch.float64)

        def first():
            return self.forms(compiled, "aten::sin")[1][0]

        for value in (single, single, single, double, double):
            self.assertEqual(compiled(value), graph_poly(value), atol=0, rtol=0)
        self.assertEqual(first(), (["%x.1: float64[4]"], 1))
        compiled(single)
        self.assertEqual(first(), (["%x.1: float32[4]"], 2))
        compiled(double)
        self.assertEqual(first(), (["%x.1: float64[4]"], 2))
        # A third kind of call is one more than the graph is exported for.
        half = torch.randn(4, dtype=torch.float16)
        for _ in range(3):
            self.assertEqual(compiled(half), graph_poly(half), atol=0, rtol=0)
        graph, forms = self.forms(compiled, "aten::sin")
        self.assertEqual(graphs, [torch.float32, torch.float64])
        self.assertEqual((graph["variants"], graph["exports"], len(forms)), (2, 2, 2))
        stats = compiled.stats()["graphs"]
        self.assertEqual((stats["built"], stats["exported"]), (1, 2))
        counted = (graph["hits"], graph["misses"])
        self.assertEqual((stats["hits"], stats["misses"]), counted)
        self.assertEqual((graph["hits"], graph["misses"]), (4, 5))

    def test_a_forward_mode_gradient_does_not_end_the_exports_of_a_graph(self):
        import torch.autograd.forward_ad as forward_ad

        graphs = []

        def keep(gm, input_types):
            graphs.append(input_types[0].dtype)
            return gm

        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_poly, passes=passes, fallback=False)
        x = torch.randn(5)
        for _ in range(3):
            self.assertEqual(compiled(x), graph_poly(x), atol=0, rtol=0)
        for _ in range(5):
            with forward_ad.dual_level():
                dual = forward_ad.make_dual(x, torch.ones(5))
                got = forward_ad.unpack_dual(compiled(dual))
                expected = forward_ad.unpack_dual(graph_poly(dual))
                self.assertEqual(got.primal, expected.primal, atol=0, rtol=0)
                self.assertEqual(got.tangent, expected.tangent, atol=0, rtol=0)
        # Calls like those run the operations, and are looked at once.
        graph, forms = self.forms(compiled, "aten::sin")
        found = (graph["state"], graph["variants"], graph["misses"])
        self.assertEqual(found, ("compiled", 1, 6))
        closed = [form["reason"] for form in graph["forms"] if "reason" in form]
        reason = "a tensor of the call carries a forward-mode gradient"
        self.assertEqual(closed, [reason])
        # The graph goes on being exported for other calls.
        wide = torch.randn(5, dtype=torch.float64)
        for _ in range(3):
            self.assertEqual(compiled(wide), graph_poly(wide), atol=0, rtol=0)
        self.assertEqual(graphs, [torch.float32, torch.float64])
        graph, forms = self.forms(compiled, "aten::sin")
        self.assertEqual((graph["variants"], forms[0]), (2, (["%x.1: float64[5]"], 2)))
        self.assertEqual(compiled.stats()["graphs"]["given_up"], 0)

    def test_an_operation_that_is_no_call_site_is_exported_as_it_was_written(self):
        graphs = []

        def keep(gm, input_types):
            graphs.append(gm)
            return gm

        def calls(target):
            nodes = graphs[-1].graph.nodes
            return [n for n in nodes if n.op == "call_function" and n.target is target]

        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_cat, passes=passes, fallback=False)
        a, b = torch.randn(3, 4), torch.randn(3, 4)
        for _ in range(3):
            self.assertEqual(compiled(a, b), graph_cat(a, b), atol=0, rtol=0)
        # The list of the call is a list of two values of the graph.
        (cat,) = calls(torch.cat)
        parts, dim = cat.args
        self.assertEqual(([type(p) for p in parts], dim), ([torch.fx.Node] * 2, 1))
        targets = [p.target for p in parts]
        aten = torch.ops.aten
        self.assertEqual(targets, [aten.sin.default, aten.cos.default])
        self.assertIn("aten::cat", self.graphs(compiled)[0]["operators"])
        compiled = torch.vm(graph_conv, passes=passes, fallback=False)
        x, w, bias = torch.randn(2, 3, 8, 8), torch.randn(4, 3, 3, 3), torch.randn(4)
        for _ in range(3):
            got = compiled(x, w, bias)
            self.assertEqual(got, graph_conv(x, w, bias), atol=0, rtol=0)
        (conv,) = calls(torch.conv2d)
        self.assertEqual(conv.args[3:], (1, 1))
        # A subscript is the index the function wrote: an int, slices, and a
        # tensor, which is an input of the graph.
        compiled = torch.vm(graph_subscripts, passes=passes, fallback=False)
        x, index = torch.randn(4, 5), torch.tensor([0, 2])
        for _ in range(3):
            got = compiled(x, index)
            self.assertEqual(got, graph_subscripts(x, index), atol=0, rtol=0)
        indices = [n.args[1] for n in calls(__import__("operator").getitem)]
        self.assertEqual(indices[:2], [0, (slice(1, None, None), 2)])
        self.assertEqual([type(i) for i in indices[2:]], [torch.fx.Node, tuple])
        graph, forms = self.forms(compiled, "aten::index.Tensor")
        self.assertEqual(forms[0][0], ["%x.1: float32[4, 5]", "%i.2: int64[2]"])
        # Another kind of index is another form: its number is checked.
        for _ in range(3):
            self.assertEqual(compiled(x, 1), graph_subscripts(x, 1), atol=0, rtol=0)
        graph, forms = self.forms(compiled, "aten::index.Tensor")
        self.assertEqual(graph["variants"], 2)
        self.assertEqual(forms[0][0], ["%i.2 == 1", "%x.1: float32[4, 5]"])
        # A tuple of tensors that the function is handed: the graph takes
        # its items, and is for a tuple of as many.
        compiled = torch.vm(graph_listed, passes=passes, fallback=False)
        pair = (torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8))
        for _ in range(3):
            self.assertEqual(compiled(pair, 3), graph_listed(pair, 3), atol=0, rtol=0)
        (cat,) = calls(torch.cat)
        self.assertIs(type(cat.args[0]), tuple)
        self.assertEqual([p.op for p in cat.args[0]], ["placeholder"] * 2)
        graph, forms = self.forms(compiled, "aten::cat")
        self.assertIn("%xs.1 is a tuple of 2", forms[0][0])
        self.assertIn("%xs.1[1]: float32[2, 3, 8, 8]", forms[0][0])
        triple = (*pair, torch.randn(2, 3, 8, 8))
        for _ in range(3):
            got = compiled(triple, 3)
            self.assertEqual(got, graph_listed(triple, 3), atol=0, rtol=0)
        graph, forms = self.forms(compiled, "aten::cat")
        self.assertEqual((graph["variants"], graph["hits"]), (2, 3))
        self.assertIn("%xs.1 is a tuple of 3", forms[0][0])
        self.assertEqual(len(calls(torch.cat)[0].args[0]), 3)
        # The function of the module is looked up at every call.
        self.assertTrue(any(".max_pool2d is " in check for check in forms[0][0]))

    # --- paths through the blocks of a function

    def test_a_graph_goes_on_through_a_branch(self):
        graphs = []

        def keep(gm, input_types):
            graphs.append(gm)
            return gm

        for behind in ((), (keep,)):
            graphs.clear()
            passes = (*DEFAULT, P.Graphs(), *behind)
            compiled = torch.vm(graph_branches, passes=passes, fallback=False)
            for flag in (True, True, True, False, False, False, True):
                x = torch.randn(4)
                got = compiled(x, flag)
                self.assertEqual(got, graph_branches(x, flag), atol=0, rtol=0)
            # One graph, of the operations in front of the branch, on one
            # side of it and behind it: a path for each way it went.
            (graph,) = self.graphs(compiled)
            shape = (graph["ops"], graph["branches"], graph["paths"])
            self.assertEqual(shape, (7, 1, 2))
            self.assertTrue(graph["where"].startswith("graph_branches (block 0)["))
            ((block, _),) = graph["ends"]
            self.assertGreater(block, 0)
            self.assertEqual(graph["why"], "the function returns")
            self.assertEqual(self.left(compiled), [])
            if not behind:
                self.assertEqual((graph["state"], graph["hits"]), ("built", 0))
                continue
            # The flag is no input of either graph: the way the branch went
            # is checked on every call, in front of what was exported. The
            # first call the other way runs a block for the first time: the
            # path is taken down at the call after it.
            self.assertEqual((graph["state"], graph["variants"]), ("compiled", 2))
            counts = (graph["hits"], graph["misses"], graph["exports"])
            self.assertEqual(counts, (3, 3, 2))
            nodes = [gm.graph.nodes for gm in graphs]
            names = [[n.target for n in ns if n.op == "placeholder"] for ns in nodes]
            self.assertEqual(names, [["x"], ["x"]])
            cos, tanh = torch.ops.aten.cos.default, torch.ops.aten.tanh.default
            self.assertEqual([cos in self.targets(gm) for gm in graphs], [True, False])
            self.assertEqual([tanh in self.targets(gm) for gm in graphs], [False, True])
            stats = compiled.stats()["graphs"]
            totals = (stats["built"], stats["hits"], stats["misses"])
            self.assertEqual(totals, (1, 3, 3))

    def test_a_branch_on_none_and_one_on_a_size_are_ways_of_a_path(self):
        def keep(gm, input_types):
            return gm

        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_optional, passes=passes, fallback=False)
        x, bias = torch.randn(3), torch.randn(3)
        for args in ((x,), (x,), (x,), (x, bias), (x, bias), (x,)):
            self.assertEqual(compiled(*args), graph_optional(*args), atol=0, rtol=0)
        (graph,) = self.graphs(compiled)
        self.assertEqual((graph["paths"], graph["variants"], graph["hits"]), (2, 2, 2))
        # A size of a tensor the graph computes: the graph gives it, and the
        # branch is checked behind what was exported, before anything of it
        # is to be seen.
        compiled = torch.vm(graph_on_a_size, passes=passes, fallback=False)
        for n in (4, 4, 4, 2, 2, 4):
            x = torch.randn(n)
            self.assertEqual(compiled(x), graph_on_a_size(x), atol=0, rtol=0)
        (graph,) = self.graphs(compiled)
        self.assertEqual((graph["paths"], graph["branches"], graph["hits"]), (2, 1, 3))

    def test_a_graph_goes_round_a_loop(self):
        graphs = []

        def keep(gm, input_types):
            graphs.append(gm)
            return gm

        for behind in ((), (keep,)):
            passes = (*DEFAULT, P.Graphs(), *behind)
            compiled = torch.vm(graph_loops, passes=passes, fallback=False)
            x = torch.randn(4)
            for _ in range(4):
                self.assertEqual(compiled(x, 100), graph_loops(x, 100), atol=0, rtol=0)
            # Sine, a hundred times round, cosine and sum; the test of the
            # loop a hundred and one times.
            (graph,) = self.graphs(compiled)
            shape = (graph["ops"], graph["branches"], graph["paths"])
            self.assertEqual(shape, (203, 101, 1))
            if not behind:
                continue
            counts = (graph["variants"], graph["hits"], graph["misses"])
            self.assertEqual(counts, (1, 2, 1))
            # The number of times round is no input: it is the path. Another
            # number is another path, taken down at the call that has it.
            for _ in range(3):
                self.assertEqual(compiled(x, 7), graph_loops(x, 7), atol=0, rtol=0)
            (graph,) = self.graphs(compiled)
            shape = (graph["ops"], graph["paths"], graph["variants"])
            self.assertEqual(shape, (17, 2, 2))
            self.assertEqual((graph["hits"], graph["misses"]), (4, 2))
            self.assertEqual(compiled(x, 100), graph_loops(x, 100), atol=0, rtol=0)
            self.assertEqual(self.graphs(compiled)[0]["hits"], 5)
            self.assertEqual([len(self.targets(gm)) for gm in graphs], [203, 17])

    def test_a_loop_over_a_list_of_tensors_is_one_graph(self):
        graphs = []

        def keep(gm, input_types):
            graphs.append(gm)
            return gm

        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_collects, passes=passes, fallback=False)
        for n in (3, 3, 3, 2, 2, 3):
            xs = [torch.randn(3) for _ in range(n)]
            self.assertEqual(compiled(xs), graph_collects(xs), atol=0, rtol=0)
        # From the first instruction of the function: the list it fills is
        # one the graph makes, and what is concatenated is what the loop
        # appended to it.
        (graph,) = self.graphs(compiled)
        self.assertEqual((graph["ops"], graph["paths"], graph["variants"]), (9, 2, 2))
        self.assertEqual((graph["inputs"], graph["hits"]), (1, 3))
        nodes = graphs[0].graph.nodes
        placeholders = [n.target for n in nodes if n.op == "placeholder"]
        self.assertEqual(placeholders, ["x", "x_2", "x_3"])
        (cat,) = [n for n in graphs[0].graph.nodes if n.target is torch.cat]
        parts = [p.target for p in cat.args[0]]
        self.assertEqual(parts, [torch.ops.aten.mul.Tensor] * 3)
        self.assertEqual(len(self.targets(graphs[1])), 7)
        # The engine's own graph is the same one.
        compiled = torch.vm(graph_collects, passes=GRAPHS, fallback=False)
        xs = [torch.randn(3) for _ in range(3)]
        for _ in range(3):
            self.assertEqual(compiled(xs), graph_collects(xs), atol=0, rtol=0)
        (graph,) = self.graphs(compiled)
        self.assertEqual((graph["state"], graph["ops"]), ("built", 9))
        self.assertIn("aten::cat", graph["operators"])

    def test_a_branch_on_the_data_of_a_tensor_ends_a_graph(self):
        def keep(gm, input_types):
            return gm

        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_on_data, passes=passes, fallback=False)
        for sign in (1, 1, 1, -1, -1, 1):
            x = sign * torch.ones(3)
            self.assertEqual(compiled(x), graph_on_data(x), atol=0, rtol=0)
        first, second = self.graphs(compiled)
        read = "aten::is_nonzero gives a number read from a tensor"
        self.assertEqual((first["ops"], first["branches"], first["why"]), (5, 0, read))
        # The graph behind the branch has the side that has operations.
        self.assertEqual((second["ops"], second["why"]), (5, "the function returns"))
        self.assertEqual(self.left(compiled), [read])

    def test_a_graph_in_a_function_that_handles_exceptions(self):
        def keep(gm, input_types):
            return gm

        for behind in ((), (keep,)):
            passes = (*DEFAULT, P.Graphs(), *behind)
            compiled = torch.vm(graph_handled, passes=passes, fallback=False)
            x = torch.randn(4)
            for table in ({"k": 2.0}, {"k": 2.0}, {"k": 2.0}, {}, {}, {"k": 2.0}):
                got = compiled(x, table)
                self.assertEqual(got, graph_handled(x, table), atol=0, rtol=0)
            # The variables of such a function are kept where its handlers
            # find them; the graph goes on across their reads and writes.
            (graph,) = self.graphs(compiled)
            self.assertEqual((graph["ops"], graph["why"]), (7, "the function returns"))
            ((block, _),) = graph["ends"]
            self.assertGreater(block, 0)
            if behind:
                self.assertEqual((graph["variants"], graph["hits"]), (1, 2))
            # A variable that a handler finds, bound by a graph and read
            # behind it, where the function goes on.
            compiled = torch.vm(graph_handled_then, passes=passes, fallback=False)
            for table in ({"k": 2.0}, {"k": 2.0}, {"k": 2.0}, {}, {"k": 2.0}):
                got = compiled(x, table)
                self.assertEqual(got, graph_handled_then(x, table), atol=0, rtol=0)
            first = self.graphs(compiled)[0]
            ended = (first["ops"], first["why"])
            self.assertEqual(ended, (5, "a call that runs no kernel"))
            if behind:
                self.assertEqual(first["hits"], 2)

    def test_a_hook_for_every_module_is_not_for_what_was_made_of_a_graph(self):
        def keep(gm, input_types):
            return gm

        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_mixed, passes=passes, fallback=False)
        x, w = torch.randn(4, 8), torch.randn(8, 8)
        expected = graph_mixed(x, w)
        for _ in range(3):
            compiled(x, w)
        seen = []

        def hook(module, args, output):
            seen.append(type(module).__name__)
            return output * 0

        handle = torch.nn.modules.module.register_module_forward_hook(hook)
        try:
            got = compiled(x, w)
        finally:
            handle.remove()
        # The graph is a module to the pass; the engine calls its forward.
        self.assertEqual(got, expected, atol=0, rtol=0)
        self.assertEqual(seen, [])
        self.assertEqual(self.graphs(compiled)[0]["hits"], 2)

    def test_an_error_on_a_path_is_raised_where_the_function_raises_it(self):
        def keep(gm, input_types):
            return gm

        def where(fn, *args):
            try:
                fn(*args)
            except KeyError as e:
                frames = traceback.extract_tb(e.__traceback__)
                return [(frame.name, frame.lineno) for frame in frames][-1], str(e)
            return None

        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_looks_up, passes=passes, fallback=False)
        x, table = torch.randn(3), {"a": 1.0}
        for _ in range(4):
            got = compiled(x, table, "a")
            self.assertEqual(got, graph_looks_up(x, table, "a"), atol=0, rtol=0)
        (graph,) = self.graphs(compiled)
        self.assertEqual((graph["ops"], graph["branches"], graph["hits"]), (6, 1, 2))
        # The lookup, behind a branch of the path, runs in front of what
        # was exported; where it raises, the call is one for the
        # instructions.
        found = where(compiled, x, table, "b")
        self.assertEqual(found, where(graph_looks_up, x, table, "b"))
        self.assertEqual(found[0][0], "graph_looks_up")
        got = compiled(x, table, "a")
        self.assertEqual(got, graph_looks_up(x, table, "a"), atol=0, rtol=0)
        self.assertEqual(self.graphs(compiled)[0]["hits"], 3)

    def test_graphs_start_in_code_of_models_and_not_in_the_rest_of_torch(self):
        passes = (*DEFAULT, P.Graphs(min_ops=1))
        compiled = torch.vm(graph_printed, passes=passes, fallback=False)
        x = torch.randn(3)
        for _ in range(3):
            self.assertEqual(compiled(x), graph_printed(x))
        # What prints a tensor is tensor work in Python, in a file of torch
        # that is no code of a model: it runs, and has no graphs.
        called = compiled.stats()["compiled_callees"]
        self.assertIn("_Formatter.__init__", called)
        files = {r["where"].split(" (")[0] for r in torch.vm.regions(compiled)}
        self.assertIn("graph_printed", files)
        printing = ("_Formatter", "_tensor_str", "_str")
        self.assertEqual({f for f in files if f.startswith(printing)}, set())

    # --- what a graph leaves as the function has it

    def test_an_iterator_that_is_unpacked_is_used_once(self):
        def keep(gm, input_types):
            return gm

        passes = (*DEFAULT, P.Graphs(min_ops=1), keep)
        compiled = torch.vm(graph_unpacks, passes=passes, fallback=False)
        x = torch.randn(4)
        for k in range(6):
            got = compiled(x, iter((1.0 + k, 2.0)))
            expected = graph_unpacks(x, iter((1.0 + k, 2.0)))
            self.assertEqual(got, expected, atol=0, rtol=0)
        # The graph ends in front of what uses the iterator up.
        first = self.graphs(compiled)[0]
        ended = (first["ops"], first["why"])
        self.assertEqual(ended, (2, "unpack is no part of a graph"))

    def test_a_property_is_read_as_often_as_the_function_reads_it(self):
        def keep(gm, input_types):
            return gm

        def counted(self):
            self.reads += 1
            return 0.25 + self.reads

        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_boxed, passes=passes, fallback=False)
        x, box, theirs = torch.randn(5), GraphScaled(), GraphScaled()
        for _ in range(4):
            self.assertEqual(compiled(x, box), graph_boxed(x, theirs), atol=0, rtol=0)
        self.assertEqual(self.graphs(compiled)[0]["hits"], 2)
        try:
            # The attribute becomes a property that counts, and gives
            # another number at every read.
            del box.scale, theirs.scale
            GraphScaled.scale = property(counted)
            for _ in range(5):
                got = compiled(x, box)
                self.assertEqual(got, graph_boxed(x, theirs), atol=0, rtol=0)
                self.assertEqual(box.reads, theirs.reads)
            self.assertEqual(box.reads, 10)
        finally:
            del GraphScaled.scale

    def test_a_mode_sees_the_operations_of_the_function(self):
        def sigmoid_for_tanh(gm, input_types):
            for node in gm.graph.nodes:
                if node.target is torch.ops.aten.tanh.default:
                    node.target = torch.ops.aten.sigmoid.default
            return gm

        x, w = torch.randn(4, 8), torch.randn(8, 8)
        with GraphSeen() as mode:
            graph_mixed(x, w)
        eager = mode.seen
        for behind in (sigmoid_for_tanh, P.Inductor()):
            passes = (*DEFAULT, P.Graphs(), behind)
            compiled = torch.vm(graph_mixed, passes=passes, fallback=False)
            for _ in range(3):
                compiled(x, w)
            # Call after call under a mode: exactly what eager mode shows
            # it, and none of what was made of the graph.
            for _ in range(5):
                with GraphSeen() as mode:
                    got = compiled(x, w)
                self.assertEqual(got, graph_mixed(x, w), atol=0, rtol=0)
                self.assertEqual(mode.seen, eager)
            (graph,) = self.graphs(compiled)
            closed = [form["reason"] for form in graph["forms"] if "reason" in form]
            self.assertEqual(closed, ["a dispatch mode is to see the operations"])
            # A kind of call without the mode is exported as any other.
            exports = graph["exports"]
            wide = x.double(), w.double()
            for _ in range(3):
                compiled(*wide)
            (graph,) = self.graphs(compiled)
            self.assertEqual((graph["exports"], graph["variants"]), (exports + 1, 2))
        # A torch function mode sees the calls the function makes.
        with GraphCalls() as mode:
            graph_mixed(x, w)
        eager = mode.seen
        passes = (*DEFAULT, P.Graphs(), P.Inductor())
        compiled = torch.vm(graph_mixed, passes=passes, fallback=False)
        for _ in range(3):
            compiled(x, w)
        expected = graph_mixed(x, w)
        for _ in range(4):
            with GraphCalls() as mode:
                got = compiled(x, w)
            self.assertEqual(got, expected, atol=0, rtol=0)
            self.assertEqual(mode.seen, eager)
        # And a transform of torch.func gets the operations themselves.
        wrt = torch.randn(8, 8)
        for _ in range(4):
            got = torch.func.grad(compiled, argnums=1)(x, wrt)
            expected = torch.func.grad(graph_mixed, argnums=1)(x, wrt)
            self.assertEqual(got, expected, atol=0, rtol=0)

    def test_a_form_for_a_tensor_subclass_is_for_no_other_call(self):
        def keep(gm, input_types):
            return gm

        x, y = torch.randn(4, 4), torch.randn(4, 4)
        expected = graph_named(x, y)
        passes = (*DEFAULT, P.Graphs(), keep)
        for subclassed in ((0,), (1,), (0, 1)):
            compiled = torch.vm(graph_named, passes=passes, fallback=False)
            for _ in range(4):
                self.assertEqual(compiled(x, y), expected, atol=0, rtol=0)
            (graph,) = self.graphs(compiled)
            hits = graph["hits"]
            pair = enumerate((x, y))
            args = [t.as_subclass(GraphSub) if k in subclassed else t for k, t in pair]
            got = compiled(*args)
            self.assertIs(type(got), GraphSub)
            self.assertEqual(got.as_subclass(torch.Tensor), expected, atol=0, rtol=0)
            # The form of that call says which of its tensors were of the
            # subclass, so the calls with plain tensors still run the graph.
            (graph,) = self.graphs(compiled)
            (closed,) = [form for form in graph["forms"] if "reason" in form]
            for k, name in enumerate(("x", "y")):
                said = f"type(%{name}.{k + 1}) is GraphSub" in closed["checks"]
                self.assertEqual(said, k in subclassed)
            for _ in range(5):
                self.assertEqual(compiled(x, y), expected, atol=0, rtol=0)
            (graph,) = self.graphs(compiled)
            counts = graph["hits"], graph["exports"], graph["variants"]
            self.assertEqual(counts, (hits + 5, 1, 1))
            # And a call like the first is known.
            misses = graph["misses"]
            again = compiled(*args).as_subclass(torch.Tensor)
            self.assertEqual(again, expected, atol=0, rtol=0)
            (graph,) = self.graphs(compiled)
            self.assertEqual((graph["misses"], len(graph["forms"])), (misses + 1, 2))

    def test_the_text_of_a_check_is_the_same_in_every_run(self):
        def keep(gm, input_types):
            return gm

        def texts(fn, *args):
            compiled = torch.vm(fn, passes=(*DEFAULT, P.Graphs(), keep), fallback=False)
            for _ in range(4):
                compiled(*args)
            (graph,) = self.graphs(compiled)
            (form,) = graph["forms"]
            return form["checks"]

        x, y = torch.randn(4, 4), torch.randn(4, 4)
        # A function is named, not shown with where it is in memory.
        checks = texts(graph_named, x, y)
        named = sorted(check.split(" is ")[1] for check in checks if " is " in check)
        self.assertEqual(named, ["torch.cat", "torch.relu"])
        # A long value is cut, and the text says so.
        checks = texts(graph_reshaped, x, (1,) * 14 + (4, 4))
        cut = "%shape.2 == (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ..."
        self.assertIn(cut, checks)
        named = [check.split(" is ")[1] for check in checks if " is " in check]
        self.assertEqual(named, ["torch.flatten"])

    # --- paths across calls

    def whole(self, fn, args, ops, calls=8):
        """Compiles ``fn`` with a pass that keeps the graphs, calls it, with
        every result that of the function itself, and returns the compiled
        function, its one graph that ran, which has ``ops`` operations and
        was exported once, and the graph modules."""
        kept = []

        def keep(gm, input_types):
            kept.append(gm)
            return gm

        compiled = torch.vm(fn, passes=(*DEFAULT, P.Graphs(), keep), fallback=False)
        for _ in range(calls):
            self.assertEqual(compiled(*args), fn(*args), atol=0, rtol=0)
        (graph,) = [g for g in self.graphs(compiled) if g["hits"]]
        self.assertEqual((graph["ops"], graph["exports"], len(kept)), (ops, 1, 1))
        self.assertGreaterEqual(graph["hits"], calls - 4)
        return compiled, graph, kept

    def test_a_path_goes_through_the_calls_of_a_function(self):
        # Positional arguments, a default, keywords.
        _, graph, kept = self.whole(graph_calls, (torch.randn(4, 4),), 11)
        self.assertEqual(graph["why"], "the function returns")
        operations = [n.target for n in kept[0].graph.nodes if n.op.startswith("call")]
        self.assertEqual(len(operations), 11)
        # A wrapper that hands on what it was given with * and **.
        _, graph, _ = self.whole(graph_spreads, (torch.randn(4, 4),), 8)
        self.assertEqual(graph["why"], "the function returns")
        # A bound method, a staticmethod, a classmethod, a property.
        compiled, graph, _ = self.whole(graph_methods, (torch.randn(4, 4),), 8)
        # What the property reads is read at every call.
        x = torch.randn(4, 4)
        GRAPH_TOOLS.k = 5.0
        try:
            for _ in range(3):
                self.assertEqual(compiled(x), graph_methods(x), atol=0, rtol=0)
        finally:
            GRAPH_TOOLS.k = 3.0

    def test_a_model_is_one_graph(self):
        torch.manual_seed(0)
        x = torch.randn(2, 4, 8, 8)
        with torch.no_grad():
            # Modules that call modules, a Sequential, a loop over a ModuleList.
            model = GraphModel().eval()
            _, graph, _ = self.whole(model, (x,), 20)
            self.assertEqual(graph["operators"].count("aten::conv2d"), 5)
            # enumerate and zip over a ModuleList, an item of it, its length,
            # a ModuleDict, a dropout that drops nothing.
            _, graph, _ = self.whole(GraphLayers().eval(), (torch.randn(2, 8),), 23)
            self.assertIn("aten::dropout", graph["operators"])
            # Functions of torch.nn.functional that pick what they call.
            _, graph, _ = self.whole(GraphPools().eval(), (torch.randn(2, 3, 8, 8),), 8)
            self.assertEqual(graph["why"], "the function returns")
            # And with a compiler behind the graphs.
            passes = (*DEFAULT, P.Graphs(), P.Inductor())
            compiled = torch.vm(model, passes=passes, fallback=False)
            for _ in range(8):
                self.assertEqual(compiled(x), model(x), rtol=1e-4, atol=1e-5)
            (graph,) = [g for g in self.graphs(compiled) if g["hits"]]
            self.assertEqual((graph["ops"], graph["exports"]), (20, 1))

    def test_what_changes_after_a_model_is_warm_is_followed(self):
        def changes(model, x):
            yield "weight in place", lambda: model.a.conv.weight.mul_(1.5), x
            fresh = torch.nn.Parameter(torch.randn(4, 4, 3, 3))
            conv = model.a.conv
            yield "parameter replaced", lambda: setattr(conv, "weight", fresh), x
            yield "train", model.train, x
            yield "eval", model.eval, x
            yield "another batch size", lambda: None, torch.randn(5, 4, 8, 8)

            def resize(size):
                model.head = torch.nn.Linear(4 * size * size, 5)

            small = torch.randn(2, 4, 6, 6)
            yield "another image size", functools.partial(resize, 6), small
            yield "head replaced", functools.partial(resize, 8), x
            yield "slope", lambda: setattr(model.a.act, "negative_slope", 0.3), x
            hooks = []

            def doubled():
                hooks.append(model.seq[0].register_forward_hook(lambda m, i, o: o * 2))

            yield "forward hook", doubled, x
            yield "hook removed", lambda: hooks.pop().remove(), x

            def shifted():
                before = model.a.register_forward_pre_hook
                hooks.append(before(lambda m, i: (i[0] + 1,)))

            yield "forward pre-hook", shifted, x
            yield "pre-hook removed", lambda: hooks.pop().remove(), x
            everywhere = torch.nn.modules.module.register_module_forward_hook

            def all_shifted():
                hooks.append(everywhere(lambda m, i, o: o + 1))

            yield "hook for every module", all_shifted, x
            yield "that hook removed", lambda: hooks.pop().remove(), x

            def swapped():
                model.seq[1] = torch.nn.Sigmoid()

            yield "submodule replaced", swapped, x
            yield "another dtype", model.double, x.double()
            yield "the first dtype", model.float, x

        for behind, exact in (((), True), ((P.Inductor(),), False)):
            torch.manual_seed(0)
            model = GraphModel().eval()
            x = torch.randn(2, 4, 8, 8)
            passes = (*DEFAULT, P.Graphs(), *behind)
            compiled = torch.vm(model, passes=passes, fallback=False)
            tolerance = dict(atol=0, rtol=0) if exact else dict(rtol=1e-4, atol=1e-5)
            with torch.no_grad():
                for _ in range(8):
                    compiled(x)
                for what, change, given in changes(model, x):
                    change()
                    for _ in range(3):
                        # Train mode updates the statistics: both from the same.
                        reference = copy.deepcopy(model)
                        got, expected = compiled(given), reference(given)
                        self.assertEqual(got, expected, msg=what, **tolerance)
                        state = model.state_dict()
                        for name, value in reference.state_dict().items():
                            said = f"{what}: {name}"
                            self.assertEqual(state[name], value, msg=said, **tolerance)
            if behind:
                graph = max(self.graphs(compiled), key=lambda g: g["hits"])
                # The graph of the whole model ran again once things were
                # as before: after the hooks, and for the first dtype.
                self.assertGreaterEqual(graph["hits"], 20)
                self.assertLessEqual(graph["exports"], 8)

    def test_a_path_ends_in_front_of_a_call_it_cannot_go_through(self):
        def keep(gm, input_types):
            return gm

        passes = (*DEFAULT, P.Graphs(min_ops=1), keep)
        x = torch.randn(4, 4)
        # A store into an object, in a method: once a call, in its place.
        compiled = torch.vm(graph_noted, passes=passes, fallback=False)
        del GRAPH_TOOLS.log[:]
        for _ in range(8):
            self.assertEqual(compiled(x), graph_noted(x), atol=0, rtol=0)
        self.assertEqual(len(GRAPH_TOOLS.log), 16)
        (first, *_) = [g for g in self.graphs(compiled) if g["hits"]]
        self.assertEqual(first["operators"], ["aten::exp"])
        self.assertTrue(first["why"].startswith("in noted: "), first["why"])
        # The same behind a call that the path went through and came back
        # from: the path ends in front of the outer call all the same, and
        # the graph is that of the function that is called.
        torch.manual_seed(0)
        model = GraphLogs().eval()
        plain = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_infer, passes=plain, fallback=False)
        del GRAPH_TOOLS.log[:]
        with torch.no_grad():
            for _ in range(8):
                got = compiled(model, x)
                self.assertEqual(got, graph_infer(model, x), atol=0, rtol=0)
        self.assertEqual(len(GRAPH_TOOLS.log), 16)
        (inner,) = [g for g in self.graphs(compiled) if g["hits"]]
        self.assertTrue(inner["where"].startswith("GraphLogs.forward"), inner["where"])
        self.assertEqual((inner["ops"], inner["exports"]), (4, 1))
        # A call that raises part of the way, and is caught.
        compiled = torch.vm(graph_picky, passes=passes, fallback=False)
        for fail in (False, False, False, False, True, True, False, True, False):
            self.assertEqual(compiled(x, fail), graph_picky(x, fail), atol=0, rtol=0)
        # A function that calls itself is gone into again, a few times.
        compiled = torch.vm(graph_descends, passes=passes, fallback=False)
        for _ in range(8):
            self.assertEqual(compiled(x, 3), graph_descends(x, 3), atol=0, rtol=0)
        self.assertTrue(any(g["hits"] for g in self.graphs(compiled)))
        # A generator is left to the interpreter: the graphs are those of
        # what is called behind it, which no path of the caller holds.
        torch.manual_seed(0)
        model = torch.nn.Sequential(GraphStage(4), GraphStage(4)).eval()
        short = (*DEFAULT, P.Graphs(min_ops=3), keep)
        compiled = torch.vm(graph_children, passes=short, fallback=False)
        image = torch.randn(2, 4, 8, 8)
        with torch.no_grad():
            for _ in range(8):
                got = compiled(model, image)
                self.assertEqual(got, graph_children(model, image), atol=0, rtol=0)
        ran = [g for g in self.graphs(compiled) if g["hits"]]
        names = [g["where"].split(" ")[0] for g in ran]
        self.assertEqual(names, ["GraphStage.forward"])
        self.assertEqual(ran[0]["ops"], 3)

    def test_a_list_that_was_extended_is_handed_to_an_operation_as_it_is(self):
        # A list of three constants and more is built empty and extended.
        x = torch.randn(2, 3, 4, 5)
        compiled, graph, kept = self.whole(graph_flips, (x, [3]), 9)
        calls = [n for n in kept[0].graph.nodes if n.op == "call_function"]
        flips = [n for n in calls if "flip" in str(n.target)]
        self.assertEqual([list(n.args[1]) for n in flips], [[0, 1, 2]])
        # And one that the function extends by what it is handed is checked.
        for more in ([], [3], []):
            self.assertEqual(compiled(x, more), graph_flips(x, more), atol=0, rtol=0)

    def test_a_path_goes_into_a_function_that_is_left_to_the_interpreter(self):
        def keep(gm, input_types):
            return gm

        def graph_of(compiled):
            (graph,) = [g for g in self.graphs(compiled) if g["hits"]]
            return graph

        global GRAPH_BIAS
        passes = (*without(P.Inline), P.Graphs(), keep)
        compiled = torch.vm(graph_counted, passes=passes, fallback=False, demote=True)
        x = torch.randn(4, 4)
        # What the measurement finds is not left to the clock: while
        # `measured` is set, the compiled form of the helper sleeps, and
        # so does the function where the interpreter runs it.
        GRAPH_TOOLS.measured = True
        self.addCleanup(setattr, GRAPH_TOOLS, "measured", False)
        for _ in range(100):
            counted = compiled(x, 30)
        self.assertEqual(counted, graph_counted(x, 30))
        stats = compiled.stats()
        self.assertEqual(stats["demoted_callees"], ["graph_counts"])
        self.assertEqual((stats["demoted"], stats["measuring"]), (False, False))
        GRAPH_TOOLS.measured = False
        # The graph of the function holds a call of the helper, and goes
        # on running.
        for _ in range(8):
            self.assertEqual(compiled(x, 30), graph_counted(x, 30), atol=0, rtol=0)
        graph = graph_of(compiled)
        found = (graph["ops"], graph["why"], graph["exports"])
        self.assertEqual(found, (6, "the function returns", 1))
        hits = graph["hits"]
        for _ in range(20):
            self.assertEqual(compiled(x, 30), graph_counted(x, 30), atol=0, rtol=0)
        graph = graph_of(compiled)
        self.assertEqual((graph["hits"], graph["exports"]), (hits + 20, 1))
        # What the helper reads and tests is read and tested at every
        # call: an attribute it branches on, a global, its argument.
        try:
            GRAPH_TOOLS.k = 5.0
            for _ in range(3):
                self.assertEqual(compiled(x, 30), graph_counted(x, 30), atol=0, rtol=0)
            GRAPH_BIAS = 2.5
            for _ in range(3):
                self.assertEqual(compiled(x, 30), graph_counted(x, 30), atol=0, rtol=0)
            for _ in range(3):
                self.assertEqual(compiled(x, 20), graph_counted(x, 20), atol=0, rtol=0)
        finally:
            GRAPH_TOOLS.k = 3.0
            GRAPH_BIAS = 0.5
        for _ in range(3):
            self.assertEqual(compiled(x, 30), graph_counted(x, 30), atol=0, rtol=0)
        self.assertGreaterEqual(graph_of(compiled)["hits"], hits + 20 + 3)
        self.assertEqual(compiled.stats()["demoted_callees"], ["graph_counts"])

    def test_a_call_in_which_a_path_is_taken_down_is_no_sample_of_the_measurement(self):
        def keep(gm, input_types):
            return gm

        # The compiled form of the function sleeps, so every call that
        # counts finds it slower than the interpreter. A call with a
        # tensor of a class that was never seen is taken down (the form
        # for that kind of call is closed) and does not count.
        passes = (*DEFAULT, P.Graphs(max_variants=64), keep)
        compiled = torch.vm(graph_timed, passes=passes, fallback=False, demote=True)
        plain = torch.randn(4, 3)
        for _ in range(3):
            self.assertEqual(compiled(plain), graph_timed(plain), atol=0, rtol=0)
        for n in range(40):
            x = plain.as_subclass(type(f"GraphKind{n}", (torch.Tensor,), {}))
            self.assertEqual(compiled(x), graph_timed(x), atol=0, rtol=0)
        stats = compiled.stats()
        self.assertEqual((stats["demoted"], stats["measuring"]), (False, True))
        # The report names the classes, which are gone but for the forms.
        (graph,) = self.graphs(compiled)
        closed = [form["checks"] for form in graph["forms"] if "reason" in form]
        self.assertGreater(len(closed), 10)
        types = [check for checks in closed for check in checks]
        named = {check.split(" is ")[1] for check in types if check.startswith("type(")}
        kinds = [name for name in named if name.startswith("GraphKind")]
        self.assertTrue(named and len(kinds) == len(named), named)
        self.assertEqual(len(named), len(closed))
        # Calls of a kind it has a form for count: it is left to the
        # interpreter, as the measurement would have had it all along.
        for _ in range(80):
            self.assertEqual(compiled(plain), graph_timed(plain), atol=0, rtol=0)
        self.assertTrue(compiled.stats()["demoted"])

    def test_the_graphs_of_what_is_called_wait_for_the_graph_of_the_caller(self):
        exported = []

        def keep(gm, input_types):
            exported.append(len([n for n in gm.graph.nodes if n.op.startswith("call")]))
            return gm

        torch.manual_seed(0)
        model = GraphModel().eval()
        x = torch.randn(2, 4, 8, 8)
        passes = (*DEFAULT, P.Graphs(min_ops=3), keep)
        compiled = torch.vm(model, passes=passes, fallback=False)
        with torch.no_grad():
            for _ in range(10):
                self.assertEqual(compiled(x), model(x), atol=0, rtol=0)
        # The blocks have three operations each and are called first and
        # most: no graph was made of one of them.
        self.assertEqual(len(exported), 1)
        graphs = self.graphs(compiled)
        (whole,) = [g for g in graphs if g["hits"]]
        self.assertEqual((whole["ops"], whole["exports"]), (20, 1))
        inner = [g for g in graphs if g["where"].startswith("GraphStage.forward")]
        counts = [(g["hits"], g["exports"]) for g in inner]
        self.assertEqual(counts, [(0, 0)] * len(inner))

    # --- effects

    def test_a_function_is_gone_into_again_inside_itself(self):
        # A Sequential inside a block inside a Sequential.
        torch.manual_seed(0)
        with torch.no_grad():
            model = GraphResNet().eval()
            _, graph, kept = self.whole(model, (torch.randn(2, 4, 8, 8),), 20)
        self.assertEqual(graph["why"], "the function returns")

    def test_a_write_to_a_tensor_of_the_path_is_an_operation_of_the_graph(self):
        import operator

        torch.manual_seed(0)
        x = torch.randn(2, 4, 8, 8)
        with torch.no_grad():
            # Activations in place and `out += identity`.
            model = GraphResNet().eval()
            given = x.clone()
            _, graph, kept = self.whole(model, (given,), 20)
            self.assertEqual(given, x, atol=0, rtol=0)
            self.assertEqual(graph["operators"].count("aten::relu_"), 4)
            self.assertEqual(graph["operators"].count("aten::add_.Tensor"), 2)
            calls = [n for n in kept[0].graph.nodes if n.op == "call_function"]
            written = [n for n in calls if "relu_" in str(n.target)]
            self.assertEqual(len(written), 4)
            # Through views and subscripts, with the tensor that is written
            # to read later, returned, and returned as a view of itself.
            head = GraphHead().eval()
            compiled, graph, kept = self.whole(head, (x,), 17)
            self.assertEqual(graph["why"], "the function returns")
            calls = [n for n in kept[0].graph.nodes if n.op == "call_function"]
            stores = [n for n in calls if n.target is operator.setitem]
            self.assertEqual(len(stores), 2)
            for got, expected in ((compiled(x), head(x)),):
                whole, part = got[1], got[2]
                self.assertEqual(part._base is whole or part._base is whole._base, True)
                theirs, of = expected[2]._base, expected[1]
                self.assertEqual(theirs is of or theirs is of._base, True)
                part.zero_()
                self.assertEqual(whole[..., :2].abs().sum().item(), 0.0)
            # With what makes the writes functional, and a compiler.
            for behind in (P.AOTAutograd(), P.Inductor()):
                for net in (model, head):
                    passes = (*DEFAULT, P.Graphs(), behind)
                    compiled = torch.vm(net, passes=passes, fallback=False)
                    for _ in range(6):
                        got, expected = compiled(x), net(x)
                        self.assertEqual(got, expected, rtol=1e-4, atol=1e-5)
                    (graph,) = [g for g in self.graphs(compiled) if g["hits"]]
                    self.assertEqual(graph["exports"], 1)

    def test_a_write_to_a_tensor_from_outside_is_done_once(self):
        def keep(gm, input_types):
            return gm

        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_writes_outside, passes=passes, fallback=False)
        torch.manual_seed(0)
        mine, theirs = torch.randn(4, 4), None
        theirs = mine.clone()
        buffers = [torch.zeros(()), torch.zeros(())]
        # Hits, a call that raises behind the first write, another dtype
        # (a miss), and hits again: the argument and the buffer after each.
        kinds = [False] * 6 + [True, False, True] + ["double"] * 3 + [False] * 3
        for kind in kinds:
            if kind == "double":
                wide = [buffer.double() for buffer in buffers]
                args = (mine.double(), wide[0]), (theirs.double(), wide[1])
            else:
                args = (mine, buffers[0]), (theirs, buffers[1])
            results = []
            for fn, (x, buffer) in zip((compiled, graph_writes_outside), args):
                try:
                    results.append(fn(x, buffer, kind is True))
                except ValueError as raised:
                    results.append(str(raised))
                results.append((x.clone(), buffer.clone()))
            self.assertEqual(results[0], results[2], atol=0, rtol=0)
            self.assertEqual(results[1], results[3], atol=0, rtol=0)
        graphs = [g for g in self.graphs(compiled) if g["hits"]]
        # The writes are operations of the graph: the one to what the
        # function made, and the ones to what it was given.
        self.assertEqual([g["ops"] for g in graphs], [15])
        for written in ("aten::tanh_", "aten::add_.Tensor", "aten::mul_.Tensor"):
            self.assertIn(written, graphs[0]["operators"])
        # Batch norm in training mode: the statistics after every call.
        torch.manual_seed(0)
        model, reference = GraphResNet().train(), None
        reference = copy.deepcopy(model)
        compiled = torch.vm(model, passes=(*DEFAULT, P.Graphs(), keep), fallback=False)
        for _ in range(8):
            x = torch.randn(2, 4, 8, 8)
            self.assertEqual(compiled(x), reference(x), atol=0, rtol=0)
            state = model.state_dict()
            for name, value in reference.state_dict().items():
                self.assertEqual(state[name], value, atol=0, rtol=0, msg=name)

    def test_a_store_into_an_object_is_done_once_and_in_its_place(self):
        def keep(gm, input_types):
            return gm

        torch.manual_seed(0)
        model, reference = GraphKeeps().eval(), None
        reference = copy.deepcopy(model)
        compiled = torch.vm(model, passes=(*DEFAULT, P.Graphs(), keep), fallback=False)
        notes = [{}, {}]
        kinds = [False] * 6 + [True, False, True] + [5] * 3 + [False] * 3
        gains = []
        with torch.no_grad():
            for kind in kinds:
                x = torch.randn(5 if kind == 5 else 3, 4)
                results = []
                runs = (compiled, model, notes[0]), (reference, reference, notes[1])
                for fn, net, note in runs:
                    try:
                        results.append(fn(x, note, kind is True))
                    except ValueError as raised:
                        results.append(str(raised))
                    results.append((net.calls, list(net.seen), dict(note), net.last))
                    gains.append(GRAPH_TOOLS.k)
                self.assertEqual(results[0], results[2], atol=0, rtol=0)
                self.assertEqual(results[1], results[3], atol=0, rtol=0)
                self.assertEqual(gains[-1], gains[-2])
        GRAPH_TOOLS.k = 3.0
        self.assertEqual(model.calls, len(kinds))
        # The stores are effects of the path: the operations in front of
        # them and behind them are one graph.
        graphs = [g for g in self.graphs(compiled) if g["hits"]]
        self.assertEqual(sorted(g["ops"] for g in graphs), [10])

    def test_a_list_that_holds_tensors_and_other_things_lands_in_order(self):
        def keep(gm, input_types):
            return gm

        torch.manual_seed(0)
        model = GraphRouted().eval()
        x = torch.randn(2, 4, 8, 8)
        compiled = torch.vm(model, passes=(*DEFAULT, P.Graphs(), keep), fallback=False)
        del GRAPH_TOOLS.log[:]
        with torch.no_grad():
            for _ in range(8):
                self.assertEqual(compiled(x), model(x), atol=0, rtol=0)
        self.assertEqual(GRAPH_TOOLS.log, [5] * 16)
        graphs = [g for g in self.graphs(compiled) if g["hits"]]
        # Up to the store, where the list has four tensors and a
        # placeholder: what is read of it from there on is what the
        # instructions read.
        self.assertEqual([g["operators"].count("aten::conv2d") for g in graphs], [4])

    def test_training_steps_are_those_of_eager(self):
        def steps(make, net, batches, targets):
            torch.manual_seed(7)
            net = copy.deepcopy(net).train()
            fn = make(net)
            optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
            losses = []
            for x, y in zip(batches, targets):
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(fn(x).flatten(1)[:, :5], y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            return losses, net.state_dict()

        torch.manual_seed(0)
        blocks = (GraphResidual(), torch.nn.Dropout(0.2), GraphResidual())
        convnet = torch.nn.Sequential(*blocks)
        options = dict(dim_feedforward=16, dropout=0.1, batch_first=True)
        encoder = torch.nn.TransformerEncoderLayer(8, 2, **options)
        cases = [
            (convnet, [torch.randn(4, 4, 8, 8) for _ in range(30)]),
            (encoder, [torch.randn(4, 6, 8) for _ in range(30)]),
        ]
        targets = [torch.randint(0, 5, (4,)) for _ in range(30)]
        for net, batches in cases:
            eager, expected = steps(lambda net: net, net, batches, targets)
            again, _ = steps(lambda net: net, net, batches, targets)
            self.assertEqual(eager, again)

            def keep(gm, input_types):
                return gm

            compilers = ((P.AOTAutograd(),), False), ((P.Inductor(),), False)
            for behind, exact in (((), True), ((keep,), True), *compilers):
                passes = (*DEFAULT, P.Graphs(), *behind)

                def compiled(net, passes=passes):
                    return torch.vm(net, passes=passes, fallback=False)

                losses, state = steps(compiled, net, batches, targets)
                if not exact:
                    # What a compiler makes of a dropout draws its own
                    # numbers (`is_exact` says so): the steps run, and are
                    # not eager's.
                    self.assertFalse(torch.vm.is_exact(passes))
                    self.assertEqual(len(losses), len(eager))
                    self.assertTrue(all(math.isfinite(loss) for loss in losses))
                    continue
                said = f"{type(net).__name__} {behind}"
                self.assertEqual(losses, eager, msg=said, atol=0, rtol=0)
                for name, value in expected.items():
                    what = f"{said} {name}"
                    self.assertEqual(state[name], value, msg=what, atol=0, rtol=0)

    # --- stores as effects of a path

    def stores(self, make, passes, kinds):
        """Calls a model that stores (``make``), compiled with ``passes``,
        and a copy of it that is not: the results and what was stored
        after every call, for each of ``kinds`` (False: a call like the
        last; True: one that raises behind the stores; 5: another batch
        size; "self": the holder it is handed is its own)."""
        torch.manual_seed(0)
        model = make().eval()
        reference = copy.deepcopy(model)
        compiled = torch.vm(model, passes=passes, fallback=False)
        notes, holders = [{}, {}], [GraphHolder(), GraphHolder()]
        with torch.no_grad():
            for kind in kinds:
                x = torch.randn(5 if kind == 5 else 3, 4)
                found = []
                mine = (compiled, model, notes[0], holders[0])
                theirs = (reference, reference, notes[1], holders[1])
                for fn, net, note, holder in (mine, theirs):
                    other = net.box if kind == "self" else holder
                    try:
                        out, (first, second) = fn(x, note, other, kind is True)
                        found.append((out, first, first is second, second is net.last))
                    except ValueError as raised:
                        found.append(str(raised))
                    found.append(net.state(note, other))
                self.assertEqual(found[0], found[2], atol=0, rtol=0, msg=str(kind))
                self.assertEqual(found[1], found[3], atol=0, rtol=0, msg=str(kind))
        return compiled

    def test_stores_into_objects_are_effects_of_a_path(self):
        def keep(gm, input_types):
            return gm

        # Hits, a call that raises behind the stores, another batch size (a
        # miss), a holder that is the model's own (another path), and hits.
        kinds = [False] * 8 + [True, False, True, 5, 5, False]
        kinds += ["self", "self", False, False]
        compiled = self.stores(GraphStores, (*DEFAULT, P.Graphs(), keep), kinds)
        (graph,) = [g for g in self.graphs(compiled) if g["hits"]]
        # The stores end no graph: the operations in front of them and
        # behind them are one.
        self.assertEqual((graph["ops"], graph["why"]), (10, "the function returns"))
        self.assertGreaterEqual(graph["hits"], 10)
        # A number that is new on every call is stored, not checked: the
        # forms are those of the two batch sizes and the two holders.
        self.assertLessEqual(graph["exports"], 4)
        # With a compiler behind the graphs.
        passes = (*DEFAULT, P.Graphs(), P.AOTAutograd())
        compiled = self.stores(GraphStores, passes, kinds)
        self.assertTrue(any(g["hits"] for g in self.graphs(compiled)))

    def test_nothing_is_stored_for_a_graph_that_was_given_up(self):
        def fails(tensor):
            raise RuntimeError("what was made of the graph does not run")

        def breaks(gm, input_types):
            # The graph, with a call at its end that raises.
            first = next(n for n in gm.graph.nodes if n.op == "placeholder")
            last = next(n for n in gm.graph.nodes if n.op == "output")
            with gm.graph.inserting_before(last):
                gm.graph.call_function(fails, (first,))
            gm.recompile()
            return gm

        compiled = self.stores(GraphStores, (*DEFAULT, P.Graphs(), breaks), [False] * 8)
        (graph,) = [g for g in self.graphs(compiled) if g["ops"] == 10]
        self.assertEqual(graph["hits"], 0)
        self.assertIn("does not run", graph["reason"])

    def test_a_store_that_runs_python_code_ends_the_path(self):
        def keep(gm, input_types):
            return gm

        # A `__setattr__` of the class.
        compiled = self.stores(GraphGuarded, (*DEFAULT, P.Graphs(), keep), [False] * 8)
        graphs = [g for g in self.graphs(compiled) if g["hits"]]
        self.assertTrue(graphs)
        self.assertNotIn(10, [g["ops"] for g in graphs])
        self.assertIn("stores into an object", [g["why"] for g in graphs])
        # A parameter and a module that are assigned: `nn.Module` registers
        # them, which the instructions do.
        torch.manual_seed(0)
        model = GraphRebinds().eval()
        reference = copy.deepcopy(model)
        compiled = torch.vm(model, passes=(*DEFAULT, P.Graphs(), keep), fallback=False)
        x = torch.randn(3, 4)
        fresh = [torch.nn.Parameter(torch.full((4,), 2.0)), torch.nn.Sigmoid()]
        calls = [(None, None)] * 6 + [(fresh[0], None), (None, None)]
        calls += [(None, fresh[1]), (None, None), (None, None)]
        with torch.no_grad():
            for weight, layer in calls:
                got, expected = compiled(x, weight, layer), reference(x, weight, layer)
                self.assertEqual(got, expected, atol=0, rtol=0)
                self.assertEqual(model.count, reference.count)
                self.assertEqual(model.weight, reference.weight, atol=0, rtol=0)
                self.assertIs(type(model.act), type(reference.act))
                self.assertEqual(list(model._parameters), ["weight"])
                self.assertEqual(list(model._modules), ["act"])
        self.assertIs(model.weight, fresh[0])
        self.assertIs(model.act, fresh[1])
        # The counter in front of them is a store of the path: a call that
        # assigns neither is one graph.
        self.assertIn(8, [g["ops"] for g in self.graphs(compiled) if g["hits"]])

    # --- writes to tensors from outside, training steps, several results

    def test_writes_to_tensors_from_outside_are_operations_of_the_graph(self):
        def keep(gm, input_types):
            return gm

        def versions(net):
            return [b._version for b in net.buffers()]

        def moved(net, before):
            return [now - was for now, was in zip(versions(net), before)]

        compilers = ((P.AOTAutograd(),), False), ((P.Inductor(),), False)
        for behind, exact in (((keep,), True), *compilers):
            tolerance = dict(atol=0, rtol=0) if exact else dict(rtol=1e-5, atol=1e-6)
            torch.manual_seed(0)
            model = GraphAverages().eval()
            reference = copy.deepcopy(model)
            passes = (*DEFAULT, P.Graphs(), *behind)
            compiled = torch.vm(model, passes=passes, fallback=False)
            given = torch.randn(3, 4)
            with torch.no_grad():
                # Hits, and another dtype in between (a miss, another form).
                for kind in ["float"] * 8 + ["double"] * 3 + ["float"] * 3:
                    for net in (model, reference):
                        cast = net.double if kind == "double" else net.float
                        cast()
                    typed = given.to(model.ema.dtype)
                    mine, theirs = typed.clone(), typed.clone()
                    before = versions(model), versions(reference)
                    got, expected = compiled(mine), reference(theirs)
                    self.assertEqual(got, expected, msg=kind, **tolerance)
                    # What it was handed, and its buffers.
                    self.assertEqual(mine, theirs, msg=kind, **tolerance)
                    for name, value in reference.state_dict().items():
                        what, held = f"{kind} {name}", model.state_dict()[name]
                        self.assertEqual(held, value, msg=what, **tolerance)
                    # The version counters end where eager's writes leave
                    # them, also where what a compiler made of the graph
                    # writes a tensor back once.
                    counted = moved(model, before[0])
                    eager = moved(reference, before[1])
                    self.assertEqual(eager, [2, 1, 1])
                    self.assertEqual(counted, eager, msg=kind)
            graph = max(self.graphs(compiled), key=lambda g: g["hits"])
            self.assertEqual((graph["ops"], graph["why"]), (16, "the function returns"))
            self.assertGreaterEqual(graph["hits"], 8)
            # Two tensors that share memory, and two that do not: a form
            # each, and the writes go where eager's go.
            compiled = torch.vm(graph_shares, passes=passes, fallback=False)
            for shared in (True, True, True, True, False, False, False, True, False):
                bases = [torch.arange(12.0).reshape(3, 4) / 7 for _ in range(2)]
                parts = [base[1] if shared else torch.ones(4) for base in bases]
                got = compiled(bases[0], parts[0])
                expected = graph_shares(bases[1], parts[1])
                self.assertEqual(got, expected, msg=str(shared), **tolerance)
                self.assertEqual(bases[0], bases[1], msg=str(shared), **tolerance)
                self.assertEqual(parts[0], parts[1], msg=str(shared), **tolerance)
            (graph,) = [g for g in self.graphs(compiled) if g["hits"]]
            self.assertEqual((graph["ops"], graph["exports"]), (10, 2))

    def test_a_write_to_what_autograd_saved_raises_as_in_eager(self):
        def keep(gm, input_types):
            return gm

        for behind in ((keep,), (P.AOTAutograd(),)):
            passes = (*DEFAULT, P.Graphs(), *behind)
            compiled = torch.vm(graph_saves, passes=passes, fallback=False)
            for _ in range(6):
                x = torch.randn(4, requires_grad=True)
                loss = compiled(x)
                expected = graph_saves(x.detach().requires_grad_())
                self.assertEqual(loss, expected, atol=0, rtol=0)
                # `exp` kept its result for its derivative, and the result
                # was written to.
                raised = "modified by an inplace operation"
                with self.assertRaisesRegex(RuntimeError, raised):
                    loss.backward()
            # Without gradients the same function is a graph.
            with torch.no_grad():
                for _ in range(6):
                    x = torch.randn(4)
                    self.assertEqual(compiled(x), graph_saves(x), atol=0, rtol=0)
            ran = [g["ops"] for g in self.graphs(compiled) if g["hits"]]
            self.assertIn(6, ran)

    def test_operations_with_several_results_are_operations_of_a_graph(self):
        torch.manual_seed(0)
        x = torch.randn(4, 6)
        compiled, graph, kept = self.whole(graph_several, (x,), 25)
        self.assertEqual(graph["why"], "the function returns")
        for behind in (P.AOTAutograd(), P.Inductor()):
            passes = (*DEFAULT, P.Graphs(), behind)
            compiled = torch.vm(graph_several, passes=passes, fallback=False)
            for _ in range(6):
                self.assertEqual(compiled(x), graph_several(x), rtol=1e-5, atol=1e-6)
            (graph,) = [g for g in self.graphs(compiled) if g["hits"]]
            self.assertEqual(graph["exports"], 1)

    def test_a_training_step_is_a_graph(self):
        def keep(gm, input_types):
            return gm

        def step(fn, net, x, y, optimizer):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(fn(x).flatten(1)[:, :5], y)
            loss.backward()
            grads = [p.grad for p in net.parameters()]
            gradients = [None if grad is None else grad.clone() for grad in grads]
            optimizer.step()
            return loss.item(), gradients

        torch.manual_seed(0)
        batches = [torch.randn(4, 4, 8, 8) for _ in range(30)]
        targets = [torch.randint(0, 5, (4,)) for _ in range(30)]
        compilers = ((P.AOTAutograd(),), False), ((P.Inductor(),), False)
        for behind, exact in (((keep,), True), *compilers):
            tolerance = dict(atol=0, rtol=0) if exact else dict(rtol=1e-4, atol=1e-5)
            torch.manual_seed(1)
            model = GraphResNet().train()
            reference = copy.deepcopy(model)
            # How many steps run the graph is counted below: nothing is
            # left to the interpreter by measurement.
            passes = (*DEFAULT, P.Graphs(), *behind)
            compiled = torch.vm(model, passes=passes, fallback=False, demote=False)
            ours = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
            theirs = torch.optim.SGD(reference.parameters(), lr=0.05, momentum=0.9)
            for at, (x, y) in enumerate(zip(batches, targets)):
                if at == 20:
                    # Part of the model stops learning: another form.
                    for net in (model, reference):
                        net.layer[0].conv1.weight.requires_grad_(False)
                got = step(compiled, model, x, y, ours)
                expected = step(reference, reference, x, y, theirs)
                what = f"loss of step {at}"
                self.assertEqual(got[0], expected[0], msg=what, **tolerance)
                # The gradient of every parameter.
                what = f"gradients of step {at}"
                self.assertEqual(got[1], expected[1], msg=what, **tolerance)
            for name, value in reference.state_dict().items():
                self.assertEqual(model.state_dict()[name], value, msg=name, **tolerance)
            # The forward of a training step is one graph: batch norm writes
            # its statistics in it.
            graph = max(self.graphs(compiled), key=lambda g: g["hits"])
            self.assertEqual((graph["ops"], graph["why"]), (26, "the function returns"))
            self.assertGreaterEqual(graph["hits"], 20)
            self.assertIn("aten::add_.Tensor", graph["operators"])
            # Without gradients, and in inference mode: forms of their own.
            x = batches[0]
            for mode in (torch.no_grad, torch.inference_mode):
                for _ in range(3):
                    with mode():
                        got, expected = compiled(x), reference(x)
                    self.assertEqual(got, expected, **tolerance)
                    self.assertFalse(got.requires_grad)
            for name, value in reference.state_dict().items():
                self.assertEqual(model.state_dict()[name], value, msg=name, **tolerance)
            # An input that requires a gradient, and going backward twice.
            mine, other = x.clone().requires_grad_(), x.clone().requires_grad_()
            for _ in range(3):
                got, expected = compiled(mine).sum(), reference(other).sum()
                for loss in (got, expected):
                    loss.backward(retain_graph=True)
                    loss.backward()
                self.assertEqual(mine.grad, other.grad, **tolerance)

    # --- what the report says

    def test_a_training_graph_is_the_same_graph_under_autocast(self):
        # Autocast casts the first convolution's result. That the sizes of
        # the result follow from the sizes of what the convolution took is
        # asked on tensors that hold nothing, which autocast does not cast:
        # the dtypes are not compared there. Without that the count of a
        # batch norm, a write to a tensor from outside behind a check that
        # was taken for one that can fail, ended the graph.
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(4, 8, 3, padding=1)
                self.norm = torch.nn.BatchNorm2d(8)
                self.head = torch.nn.Linear(8 * 8 * 8, 5)

            def forward(self, x):
                h = torch.relu(self.norm(self.conv(x)))
                return self.head(torch.flatten(h, 1))

        def keep(gm, input_types):
            return gm

        x = torch.randn(4, 4, 8, 8)
        found = []
        for cast in (False, True):
            torch.manual_seed(1)
            model = Net().train()
            reference = copy.deepcopy(model)
            passes = (*DEFAULT, P.Graphs(), keep)
            compiled = torch.vm(model, passes=passes, fallback=False, demote=False)
            under = torch.autocast("cpu", dtype=torch.bfloat16, enabled=cast)
            with torch.no_grad(), under:
                for _ in range(6):
                    self.assertEqual(compiled(x), reference(x), atol=0, rtol=0)
            for name, value in reference.state_dict().items():
                self.assertEqual(model.state_dict()[name], value, msg=name)
            graphs = self.graphs(compiled, "compiled")
            found.append([(g["ops"], g["why"], g["operators"]) for g in graphs])
        ((ops, why, operators),) = found[0]
        self.assertEqual((ops, why), (6, "the function returns"))
        self.assertIn("aten::add_.Tensor", operators)
        self.assertEqual(found[1], found[0])

    def test_a_graph_has_one_name_and_the_report_counts_nothing_twice(self):
        def keep(gm, input_types):
            return gm

        # Three calls of a function, no branch: the graph is named by its
        # first operation at every call, whether or not its path, which
        # starts with the function, was taken down yet.
        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_calls, passes=passes, fallback=False)
        x = torch.randn(4, 4)
        names = set()
        for _ in range(8):
            compiled(x)
            names |= {g["where"] for g in self.graphs(compiled)}
        self.assertEqual(len(names), 1)
        (graph,) = self.graphs(compiled)
        self.assertEqual((graph["ops"], graph["branches"], graph["calls"]), (11, 0, 3))
        # The calls the path goes through are not listed as left out.
        calls = [left for left in self.left(compiled) if left.startswith("calls")]
        self.assertFalse(calls)
        # With the engine's own graphs, the graph of a function and that
        # of what it calls hold the same operations: the report says which
        # is within which.
        torch.manual_seed(0)
        model = GraphModel().eval()
        compiled = torch.vm(graph_infer, passes=(*DEFAULT, P.Graphs()), fallback=False)
        with torch.no_grad():
            for _ in range(8):
                compiled(model, torch.randn(2, 4, 8, 8))
        live = [g for g in self.graphs(compiled) if "within" not in g]
        found = [(g["where"].split(" ")[0], g["ops"]) for g in live]
        self.assertEqual(found, [("graph_infer", 20)])
        inside = [g for g in self.graphs(compiled) if "within" in g]
        held = {g["where"].split(" ")[0]: g["within"].split(" ")[0] for g in inside}
        self.assertEqual(held["GraphModel.forward"], "graph_infer")
        self.assertEqual(compiled.stats()["graphs"]["built"], 1)
        # A path of the caller that ended inside the called function is
        # cut back to in front of the call: the caller has no graph, and
        # the graph of the called function is within none.
        passes = (*without(P.Inline), P.Graphs(), keep)
        compiled = torch.vm(graph_outer, passes=passes, fallback=False)
        for _ in range(8):
            compiled(x)
        (graph,) = self.graphs(compiled)
        found = (graph["where"].split(" ")[0], graph["ops"])
        self.assertEqual(found, ("graph_inner", 5))
        self.assertNotIn("within", graph)
        self.assertGreater(graph["hits"], 0)
        self.assertEqual(compiled.stats()["graphs"]["built"], 1)

    def test_a_check_that_a_form_decides_keeps_no_tensor_in_the_results(self):
        kept = []

        def keep(gm, input_types):
            kept.append(gm)
            return gm

        torch.manual_seed(0)
        layers = [torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 4)]
        model = torch.nn.Sequential(*layers, torch.nn.Tanh()).eval()
        compiled = torch.vm(model, passes=(*DEFAULT, P.Graphs(), keep), fallback=False)
        x = torch.randn(3, 4)
        with torch.no_grad():
            for _ in range(6):
                self.assertEqual(compiled(x), model(x), atol=0, rtol=0)
        # `F.relu` asks whether its argument overrides torch functions:
        # for a tensor the graph gives, the answer follows from the form.
        (given,) = [n.args[0] for n in kept[0].graph.nodes if n.op == "output"]
        self.assertEqual([node.target for node in given], [torch.ops.aten.tanh.default])
        # The weights are named by where they were read, and the values of
        # a called function with the function.
        placeholders = [n.target for n in kept[0].graph.nodes if n.op == "placeholder"]
        self.assertEqual(placeholders[1:3], ["module_weight", "module_bias"])
        (graph,) = [g for g in self.graphs(compiled) if g["hits"]]
        checks = graph["forms"][0]["checks"]
        self.assertIn("module.weight: float32[4, 4], requires_grad", checks)
        called = [check for check in checks if check.startswith("Linear.forward:%")]
        self.assertTrue(called, checks)

    def test_a_kind_of_call_is_open_again_once_its_numbers_stay(self):
        exports = []

        def keep(gm, input_types):
            exports.append(1)
            return gm

        # The calls that run the graph are counted: no measurement.
        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_rated, passes=passes, fallback=False, demote=False)
        x = torch.randn(4, 4)
        # Ten rates, each new: the kind of call is closed, as before.
        for step in range(10):
            rate = 1.0 + step / 10
            self.assertEqual(compiled(x, rate), graph_rated(x, rate), atol=0, rtol=0)
        (graph,) = self.graphs(compiled)
        closed = [form["reason"] for form in graph["forms"] if "reason" in form]
        self.assertEqual(closed, ["its inputs take ever new values"])
        made, hits = len(exports), graph["hits"]
        # One rate from then on: it is exported once more, and runs.
        for _ in range(30):
            self.assertEqual(compiled(x, 3.0), graph_rated(x, 3.0), atol=0, rtol=0)
        (graph,) = self.graphs(compiled)
        self.assertEqual(len(exports), made + 1)
        self.assertGreaterEqual(graph["hits"], hits + 20)

    def test_what_an_operation_gives_shows_when_it_has_run(self):
        def keep(gm, input_types):
            return gm

        def run(fn, *args, passes=(keep,)):
            listed = (*DEFAULT, P.Graphs(), *passes)
            compiled = torch.vm(fn, passes=listed, fallback=False)
            for _ in range(8):
                self.assertEqual(compiled(*args), fn(*args), atol=0, rtol=0)
            ran = [g for g in self.graphs(compiled, "compiled") if g["hits"]]
            return [(g["ops"], g["why"]) for g in ran]

        x = torch.randn(4, 4)
        returns, number = "the function returns", "gives a number read from a tensor"
        for passes in ((keep,), (P.AOTAutograd(),)):
            # Two tensors: an operation of the graph, with the item that
            # is taken of them.
            found = run(graph_parts, x, 2, passes=passes)
            self.assertEqual(found, [(13, returns)])
            # None (an empty tuple): the path ends in front of it, and
            # the next graph starts behind it.
            ops = [ops for ops, _ in run(graph_parts, x, 0, passes=passes)]
            self.assertEqual(ops, [6, 4])
            # A bool, and a number: no operations of a graph either.
            found = run(graph_asks, x, torch.randn(4, 4), passes=passes)
            self.assertEqual(found, [(5, f"aten::equal {number}"), (5, returns)])
            found = run(graph_tops, x, passes=passes)
            self.assertEqual(found, [(6, f"aten::item {number}"), (5, returns)])
            # Several tensors that are needed whole, by the caller or by a
            # loop: a graph gives back tensors, so the path ends in front
            # of the operation.
            whole = "what an operation with several results gives is used whole"
            self.assertEqual(run(graph_halves, x, passes=passes), [(5, whole)])
            self.assertEqual(run(graph_rows, x, passes=passes), [(6, whole)])
        # One function, both ways and back.
        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_parts, passes=passes, fallback=False)
        for k in (2, 2, 2, 0, 0, 0, 2, 0, 1, 1, 2, 0):
            self.assertEqual(compiled(x, k), graph_parts(x, k), atol=0, rtol=0)

    def test_a_tensor_made_behind_a_graph_is_no_constant_of_it(self):
        def keep(gm, input_types):
            return gm

        # A loop over a list of what the operations gave takes its items
        # behind the graph: the path ends where an operation takes one.
        x = torch.randn(4, 4)
        for passes in ((keep,), (P.AOTAutograd(),)):
            listed = (*DEFAULT, P.Graphs(), *passes)
            compiled = torch.vm(graph_gathers, passes=listed, fallback=False)
            for scale in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
                got, expected = compiled(x * scale), graph_gathers(x * scale)
                self.assertEqual(got, expected, atol=0, rtol=0)
            (graph,) = [g for g in self.graphs(compiled) if g["hits"]]
            ends = "an operation takes what the graph cannot compute"
            self.assertEqual((graph["ops"], graph["why"]), (7, ends))

    def test_a_graph_that_turned_out_too_small_is_not_waited_for(self):
        def keep(gm, input_types):
            return gm

        # The first graph is taken down again at the second call, which
        # goes through a branch for the first time, and then has two
        # operations: the graph behind it does not go on waiting for it.
        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_probed, passes=passes, fallback=False, demote=False)
        x = torch.randn(4, 4)
        self.assertEqual(compiled(x, 1), graph_probed(x, 1), atol=0, rtol=0)
        for _ in range(12):
            self.assertEqual(compiled(x, 5), graph_probed(x, 5), atol=0, rtol=0)
        (graph,) = self.graphs(compiled)
        found = (graph["ops"], graph["state"], graph["exports"], graph["hits"])
        self.assertEqual(found, (5, "compiled", 1, 10))

    def test_a_comparison_of_sizes_is_part_of_a_path(self):
        def keep(gm, input_types):
            return gm

        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_sized, passes=passes, fallback=False)
        x, y = torch.randn(2, 3), torch.randn(2, 3)
        for _ in range(6):
            self.assertEqual(compiled(x, y), graph_sized(x, y), atol=0, rtol=0)
        (graph,) = self.graphs(compiled)
        found = (graph["ops"], graph["why"], graph["exports"])
        self.assertEqual(found, (7, "the function returns", 1))
        # Other sizes are another form, and where the function raises for
        # them it raises.
        x, y = torch.randn(4, 3), torch.randn(4, 3)
        for _ in range(3):
            self.assertEqual(compiled(x, y), graph_sized(x, y), atol=0, rtol=0)
        for _ in range(3):
            with self.assertRaisesRegex(ValueError, "sizes"):
                compiled(x, torch.randn(4, 1))
        x, y = torch.randn(2, 3), torch.randn(2, 3)
        self.assertEqual(compiled(x, y), graph_sized(x, y), atol=0, rtol=0)

    def test_what_a_path_takes_is_not_reported_as_left_out(self):
        def keep(gm, input_types):
            return gm

        # The writes of the layers and the single operations of the
        # functions the path goes through are operations of its graph.
        torch.manual_seed(0)
        model = GraphResNet().eval()
        compiled = torch.vm(model, passes=(*DEFAULT, P.Graphs(), keep), fallback=False)
        x = torch.randn(2, 4, 8, 8)
        with torch.no_grad():
            for _ in range(8):
                self.assertEqual(compiled(x), model(x), atol=0, rtol=0)
        (graph,) = [g for g in self.graphs(compiled) if g["hits"]]
        self.assertEqual(graph["ops"], 20)
        self.assertIn("aten::relu_", graph["operators"])
        self.assertEqual(self.left(compiled), [])
        # What no path takes is still listed.
        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_asks, passes=passes, fallback=False)
        for _ in range(6):
            compiled(x, x)
        left = ["aten::equal gives a number read from a tensor"]
        self.assertEqual(self.left(compiled), left)

    # --- what reads behind the stores and the writes of a path see

    def against_eager(self, fn, schedule, passes, exact=True):
        """Calls ``fn`` under torch.vm and in eager mode with what
        ``schedule`` makes for each call (twice, alike), and compares what
        they return and what they leave in what they were handed."""
        listed = (*DEFAULT, P.Graphs(), *passes)
        compiled = torch.vm(fn, passes=listed, fallback=False, demote=False)
        tolerance = dict(atol=0, rtol=0) if exact else dict(rtol=1e-4, atol=1e-5)
        for at, make in enumerate(schedule):
            mine, theirs = make(), make()
            outcomes = []
            for call, args in ((compiled, mine), (fn, theirs)):
                try:
                    outcomes.append(call(*args))
                except Exception as e:
                    outcomes.append(type(e).__name__)
            self.assertEqual(outcomes[0], outcomes[1], msg=f"call {at}", **tolerance)
            left, wanted = self.left_in(mine), self.left_in(theirs)
            self.assertEqual(left, wanted, msg=f"what call {at} left", **tolerance)
        return compiled

    def left_in(self, args):
        said = []
        for value in args:
            if isinstance(value, torch.Tensor):
                said.append(value.detach())
            elif isinstance(value, GraphPair):
                said.append((value.n, value.items))
            elif isinstance(value, (list, dict)):
                said.append(value)
        return said

    def test_a_form_of_a_path_that_writes_is_for_how_its_tensors_lie_in_memory(self):
        def given(row):
            def make():
                base = torch.arange(24.0).reshape(6, 4) / 7
                return (base[:3], torch.ones(4) if row is None else base[row])

            return make

        # A row of the first tensor, another row of it, a row of the same
        # memory behind it, and a tensor of its own: a form each. What a
        # compiler made for one writes where that one had the row.
        rows = [1, 1, 1, 1, 2, 2, 4, 4, None, 1, 2, 4]
        for behind in (P.AOTAutograd(), P.Inductor()):
            calls = [given(row) for row in rows]
            compiled = self.against_eager(graph_shares, calls, (behind,), exact=False)
            (graph,) = self.graphs(compiled)
            self.assertEqual(graph["exports"], 4)
            self.assertGreaterEqual(graph["hits"], 6)
            # The forms say how their tensors lie.
            checks = [check for form in graph["forms"] for check in form["checks"]]
            lie = sorted(check for check in checks if "memory" in check)
            starts = [4 * row for row in (1, 2, 4)]
            shared = [f"share memory: %x.1 at 0, %part.2 at {at}" for at in starts]
            self.assertEqual(lie, sorted(["no two tensors share memory"] + shared))

    def test_a_read_behind_a_store_goes_past_it_only_where_that_is_checked(self):
        def keep(gm, input_types):
            return gm

        x = torch.randn(4, 4)

        def boxes(shared, borg=False):
            def make():
                a, b = GraphPair(), GraphPair()
                if borg and shared:
                    b.__dict__ = a.__dict__
                return (x, a, a) if shared and not borg else (x, a, b)

            return make

        def with_dict(shared):
            def make():
                b = GraphPair()
                return (x, b.__dict__ if shared else {"n": 0}, b)

            return make

        apart_then_one = [False] * 5 + [True, True, False, True]
        cases = [
            # Two stores under one name through two arguments, and a read
            # through the first: the second store is the later one where
            # the two are one object.
            (graph_two_names, boxes, 8),
            # A list under two names; an iterator over it.
            (graph_two_lists, boxes, 8),
            (graph_through_iterator, boxes, 8),
            # The same attribute by other ways: the dict of the object,
            # `getattr`, two objects that keep their attributes in one dict,
            # a dict that is the one an object keeps them in.
            (graph_through_dict, boxes, 8),
            (graph_through_getattr, boxes, 8),
            (graph_through_getattr, lambda shared: boxes(shared, borg=True), 8),
            (graph_dict_of, with_dict, 8),
        ]
        for fn, make, ops in cases:
            calls = [make(shared) for shared in apart_then_one]
            compiled = self.against_eager(fn, calls, (keep,))
            # The calls with two objects run the graph.
            ran = [g for g in self.graphs(compiled) if g["hits"]]
            self.assertEqual([g["ops"] for g in ran], [ops], msg=fn.__name__)
            self.assertGreaterEqual(ran[0]["hits"], 4, msg=fn.__name__)

    def test_what_a_write_changes_of_a_tensor_is_not_read_in_front_of_it(self):
        def keep(gm, input_types):
            return gm

        x, w = torch.randn(4, 4), torch.randn(4, requires_grad=True)
        sizes = "changes the sizes of a tensor from outside"
        kinds = ((keep, True), (P.AOTAutograd(), False), (P.Inductor(), False))
        for behind, exact in kinds:
            # The version counter: read behind the write, and what the
            # write left it at.
            calls = [lambda: (x, torch.zeros(()))] * 8
            compiled = self.against_eager(graph_versions, calls, (behind,), exact)
            ends = "reads of a tensor what a write of the graph may have changed"
            self.assertEqual([g["why"] for g in self.graphs(compiled)], [ends])
            # A write that changes the sizes of what it is handed, or makes
            # it require a gradient, is no operation of a path: the graphs
            # in front of it and behind it run.
            turned = (graph_turns, lambda: (x, torch.arange(6.0).reshape(2, 3)))
            learned = (graph_learns, lambda: (x, w, torch.zeros(4)))
            for fn, make in (turned, learned):
                compiled = self.against_eager(fn, [make] * 8, (behind,), exact)
                graphs = self.graphs(compiled)
                whys = [g["why"] for g in graphs]
                self.assertIn(f"{sizes}, or whether it requires a gradient", whys)
                exports = [g["exports"] for g in graphs]
                self.assertEqual(exports, [1, 1], msg=fn.__name__)
                self.assertTrue(all(g["hits"] >= 4 for g in graphs), msg=fn.__name__)

    def test_a_call_that_raises_behind_a_write_leaves_what_eager_leaves(self):
        def keep(gm, input_types):
            return gm

        x = torch.randn(4, 4)
        good, bad = torch.tensor([0, 2]), torch.tensor([1, 7])
        outside = "a path that writes to a tensor from outside"
        kinds = ((keep, True), (P.AOTAutograd(), False), (P.Inductor(), False))
        for behind, exact in kinds:
            torch.manual_seed(1)
            model = GraphCounts().eval()
            reference = copy.deepcopy(model)
            passes = (*DEFAULT, P.Graphs(), behind)
            compiled = torch.vm(model, passes=passes, fallback=False, demote=False)
            tolerance = dict(atol=0, rtol=0) if exact else dict(rtol=1e-4, atol=1e-5)
            with torch.no_grad():
                for at, index in enumerate([good] * 6 + [bad, good, bad, bad, good]):
                    outcomes = []
                    for call in (compiled, reference):
                        try:
                            outcomes.append(call(x, index))
                        except Exception as e:
                            outcomes.append(type(e).__name__)
                    # The exception is eager's, and so is what the call
                    # left: the counter, the list, the buffer.
                    got, expected = outcomes
                    self.assertEqual(got, expected, msg=f"call {at}", **tolerance)
                    counted = (model.calls, model.seen)
                    wanted = (reference.calls, reference.seen)
                    self.assertEqual(counted, wanted, msg=f"call {at}")
                    total, wanted = model.total, reference.total
                    self.assertEqual(total, wanted, msg=f"call {at}", **tolerance)
            # The index, which can be out of range at any call, is behind
            # the graph that writes: that one cannot raise for it.
            (graph,) = [g for g in self.graphs(compiled) if g["hits"]]
            ends = f"can fail for what its tensors hold, on {outside}"
            self.assertEqual(graph["why"], ends)
            self.assertEqual((graph["ops"], graph["exports"]), (6, 1))
            self.assertNotIn("aten::index.Tensor", graph["operators"])

    def test_a_size_that_what_a_tensor_holds_decides_is_checked_at_every_call(self):
        def keep(gm, input_types):
            return gm

        torch.manual_seed(0)
        inputs = [torch.randn(4, 4) * scale for scale in (1, 1, 1, 1, 1, -1, 3, -1, 1)]
        passes, exact = (*DEFAULT, P.Graphs(), keep), dict(atol=0, rtol=0)
        # Behind a write to a tensor from outside, where a check cannot
        # fail: the path ends in front of what the size decides.
        for fn in (graph_repeats, graph_finds):
            mine, theirs = torch.zeros(()), torch.zeros(())
            compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
            for at, x in enumerate(inputs):
                what = f"{fn.__name__} {at}"
                got, expected = compiled(x, mine), fn(x, theirs)
                self.assertEqual(got, expected, msg=what, **exact)
                self.assertEqual(mine, theirs, msg=what, **exact)
        # Without one: the branch on the size is a check of every call.
        compiled = torch.vm(graph_weighs, passes=passes, fallback=False, demote=False)
        for at, x in enumerate(inputs):
            got, expected = compiled(x), graph_weighs(x)
            self.assertEqual(got, expected, msg=f"call {at}", **exact)

    def test_stores_that_follow_from_one_another_are_exported_once(self):
        exported = []

        def keep(gm, input_types):
            exported.append(1)
            return gm

        # An item of a list at a place that the path appended.
        calls = [lambda: (torch.ones(3, 4), [])] * 12
        compiled = self.against_eager(graph_fills, calls, (keep,))
        (graph,) = self.graphs(compiled)
        self.assertEqual((len(exported), graph["exports"], graph["hits"]), (1, 1, 10))
        # A store into what the graph gives, behind a write to a tensor
        # from outside: the path ends in front of it, and both graphs run.
        calls = [lambda: (torch.ones(4, 4), torch.zeros(()))] * 8
        compiled = self.against_eager(graph_notes, calls, (keep,))
        ran = [g for g in self.graphs(compiled) if g["hits"] >= 5]
        found = [(g["ops"], g["exports"], g["why"]) for g in ran]
        outside = "behind a write to a tensor from outside"
        ends = f"stores into what the graph gives, {outside}"
        self.assertEqual(found, [(6, 1, ends), (5, 1, "the function returns")])

    def test_why_a_graph_is_not_exported_is_said(self):
        def helper(gm):
            raise IndexError("list index out of range")

        def deep(gm, input_types):
            return helper(gm)

        def keep(gm, input_types):
            return gm

        # A pass that raises inside what it calls: the reason says where.
        passes = (*DEFAULT, P.Graphs(), deep)
        compiled = torch.vm(graph_rated, passes=passes, fallback=False)
        x = torch.randn(4, 4)
        for _ in range(4):
            self.assertEqual(compiled(x, 2.0), graph_rated(x, 2.0), atol=0, rtol=0)
        (graph,) = self.graphs(compiled)
        raised = "the pass deep raised IndexError: list index out of range"
        self.assertRegex(graph["reason"], rf"^{raised} \(in helper, test_vm.py:\d+\)$")
        # What autograd saved is looked at through the history of the run,
        # up to a bound: a longer one is not exported for.
        export = importlib.import_module("torch.vm._export")
        w = torch.randn(4, 4, requires_grad=True)
        passes = (*DEFAULT, P.Graphs(), keep)
        short = "what autograd saved is not looked at behind 8 functions"
        for bound, reason in ((export._HISTORY, None), (8, short)):
            with unittest.mock.patch.object(export, "_HISTORY", bound):
                compiled = torch.vm(graph_remembers, passes=passes, fallback=False)
                for _ in range(4):
                    got, expected = compiled(x, w), graph_remembers(x, w)
                    self.assertEqual(got, expected, atol=0, rtol=0)
                    mine = torch.autograd.grad(got.sum(), w)
                    theirs = torch.autograd.grad(expected.sum(), w)
                    self.assertEqual(mine, theirs, atol=0, rtol=0)
                forms = [form for g in self.graphs(compiled) for form in g["forms"]]
                closed = [form["reason"] for form in forms if "reason" in form]
                self.assertEqual(closed, [reason] if reason else [])

    def test_a_pool_to_one_cell_and_a_meshgrid_are_operations_of_a_small_graph(self):
        def keep(gm, input_types):
            return gm

        torch.manual_seed(0)
        model = GraphPooled().eval()
        compiled = torch.vm(model, passes=(*DEFAULT, P.Graphs(), keep), fallback=False)
        x = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            for _ in range(6):
                self.assertEqual(compiled(x), model(x), atol=0, rtol=0)
        (graph,) = [g for g in self.graphs(compiled) if g["hits"]]
        layers = ["conv2d", "relu", "adaptive_avg_pool2d", "flatten.using_ints"]
        wanted = [f"aten::{name}" for name in (*layers, "linear")]
        self.assertEqual(graph["operators"], wanted)
        self.assertEqual(graph["why"], "the function returns")
        # `torch.meshgrid` warns once in a process when it is called
        # without `indexing`; the call that warns runs Python code, which
        # the profile would take down for the instruction. Not here.
        a, b = torch.randn(3), torch.randn(5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = graph_grid(a, b)
        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_grid, passes=passes, fallback=False)
        for _ in range(6):
            self.assertEqual(compiled(a, b), expected, atol=0, rtol=0)
        (graph,) = [g for g in self.graphs(compiled) if g["hits"]]
        self.assertEqual((graph["ops"], graph["why"]), (7, "the function returns"))
        self.assertIn("aten::meshgrid", graph["operators"])
        # With `indexing` the C function is called with a `**kwargs` that
        # is not empty, which no path goes through: it ends in front.
        passes = (*DEFAULT, P.Graphs(min_ops=2), keep)
        compiled = torch.vm(graph_grid_named, passes=passes, fallback=False)
        for _ in range(6):
            self.assertEqual(compiled(a, b), graph_grid_named(a, b), atol=0, rtol=0)
        operators = [name for g in self.graphs(compiled) for name in g["operators"]]
        self.assertNotIn("aten::meshgrid", operators)


GRAPH_END_STATE = [0.5]
GRAPH_END_LOG = []


class GraphEnds:
    # Its end changes what the functions below read.
    def __del__(self):
        if GRAPH_END_LOG is None:
            return
        GRAPH_END_LOG.append("the end of an object")
        GRAPH_END_STATE[0] += 1.0


class GraphEndWatched:
    pass


def graph_end_callback(reference):
    GRAPH_END_LOG.append("a weak reference is told")
    GRAPH_END_STATE[0] += 1.0


class GraphEndBox:
    item = None


GRAPH_END_BOX = GraphEndBox()
GRAPH_END_DICT = {"k": None}
GRAPH_END_LIST = [None]
GRAPH_END_MODULE = torch.nn.Identity()
GRAPH_END_MODULE.item = None
GRAPH_END_HELD = [None]
GRAPH_END_REFS = []
GRAPH_END_WEAK = weakref.WeakKeyDictionary()


def graph_end_attribute(x, keep=None):
    h = torch.relu(x) + 1
    h = h * GRAPH_END_STATE[0]
    GRAPH_END_BOX.item = 0
    h = h + GRAPH_END_STATE[0]
    return (torch.tanh(h) * 2 - 1).sum() + GRAPH_END_STATE[0]


def graph_end_dict_item(x, keep=None):
    h = torch.relu(x) + 1
    h = h * GRAPH_END_STATE[0]
    GRAPH_END_DICT["k"] = 0
    h = h + GRAPH_END_STATE[0]
    return (torch.tanh(h) * 2 - 1).sum() + GRAPH_END_STATE[0]


def graph_end_list_item(x, keep=None):
    h = torch.relu(x) + 1
    h = h * GRAPH_END_STATE[0]
    GRAPH_END_LIST[0] = 0
    h = h + GRAPH_END_STATE[0]
    return (torch.tanh(h) * 2 - 1).sum() + GRAPH_END_STATE[0]


def graph_end_module_attribute(x, keep=None):
    h = torch.relu(x) + 1
    h = h * GRAPH_END_STATE[0]
    GRAPH_END_MODULE.item = 0
    h = h + GRAPH_END_STATE[0]
    return (torch.tanh(h) * 2 - 1).sum() + GRAPH_END_STATE[0]


def graph_end_variable(x, keep=None):
    # The variable holds the last reference to what `pop` gave, from in
    # front of the first operation on.
    held = GRAPH_END_HELD.pop()
    h = torch.relu(x) + 1
    h = h * GRAPH_END_STATE[0]
    held = None
    h = h + GRAPH_END_STATE[0]
    return (torch.tanh(h) * 2 - 1).sum() + GRAPH_END_STATE[0], held


def graph_end_deleted(x, keep=None):
    # The variable is deleted inside the path: the check that it is bound
    # reads it, so a call that takes the path down keeps it.
    held = GRAPH_END_HELD.pop()
    h = torch.relu(x) + 1
    h = h * GRAPH_END_STATE[0]
    del held
    h = h + GRAPH_END_STATE[0]
    return (torch.tanh(h) * 2 - 1).sum() + GRAPH_END_STATE[0]


def graph_end_in_no_variable(x, keep=None):
    # What `pop` gave is in no variable: it goes into the tuple and ends
    # with it, when the item behind it has been taken out.
    h = (GRAPH_END_HELD.pop(), torch.relu(x) + 1)[1]
    h = h * GRAPH_END_STATE[0]
    h = h + GRAPH_END_STATE[0]
    return (torch.tanh(h) * 2 - 1).sum() + GRAPH_END_STATE[0]


class GraphEndFunction(torch.autograd.Function):
    # What it keeps for its backward ends with the history of its result.
    @staticmethod
    def forward(ctx, t):
        ctx.held = GraphEnds()
        return t * 2

    @staticmethod
    def backward(ctx, grad):
        return grad * 2


def graph_runs_to_the_return(compiled):
    # How often the graphs that end where the function returns have run.
    listed = torch.vm.regions(compiled)
    graphs = [g for g in listed if "ops" in g and "within" not in g]
    return sum(g["hits"] for g in graphs if g["why"] == "the function returns")


def graph_weak_told():
    GRAPH_END_LOG.append("an entry is taken out")
    GRAPH_END_STATE[0] += 1.0


class GraphWeakAsks(weakref.WeakKeyDictionary):
    # A weak dictionary of the standard library's, of a class of the
    # user's: what its own callback asks of it runs this code.
    @property
    def _iterating(self):
        graph_weak_told()
        return self.__dict__["asked"]

    @_iterating.setter
    def _iterating(self, value):
        self.__dict__["asked"] = value


class GraphWeakTakesOut:
    # Line for line the dictionary by identity that `torch.compile` notes
    # modules in, with code of the user's where it takes an entry out.
    def __init__(self):
        self.values = {}
        self.refs = {}

    def __setitem__(self, key, value):
        idx = id(key)
        if idx not in self.refs:
            self.refs[idx] = weakref.ref(key, lambda ref: self._remove_id(idx))
        self.values[idx] = value

    def _remove_id(self, idx):
        graph_weak_told()
        if idx in self.values:
            del self.values[idx]
        if idx in self.refs:
            del self.refs[idx]


class GraphWeakDict(dict):
    def __delitem__(self, key):
        graph_weak_told()
        super().__delitem__(key)


class GraphEndBytes(bytearray):
    # Memory that a tensor is made around: its end runs this.
    def __del__(self):
        if GRAPH_END_LOG is not None:
            GRAPH_END_LOG.append("the end of a buffer")
            GRAPH_END_STATE[0] += 1.0


class GraphEndTensor(torch.Tensor):
    def __del__(self):
        if GRAPH_END_LOG is not None:
            GRAPH_END_LOG.append("the end of a tensor")
            GRAPH_END_STATE[0] += 1.0


class GraphKeyHashed:
    # A key whose hash is computed by this code, which is heard once it
    # is told to be.
    heard = False

    def __hash__(self):
        if GraphKeyHashed.heard:
            graph_weak_told()
        return 7


class GraphKeyCompared:
    # Two keys with one hash: looking one up compares it with the other.
    heard = False

    def __init__(self, tag):
        self.tag = tag

    def __hash__(self):
        return 11

    def __eq__(self, other):
        if GraphKeyCompared.heard:
            graph_weak_told()
        return self is other


class GraphUnusedPlain:
    pass


GRAPH_UNUSED_LOG = []


def graph_unused_told(self):
    GRAPH_UNUSED_LOG.append(("read", len(GRAPH_UNUSED_LOG)))
    return 1


# Functions with an instruction between their operations whose value
# nothing uses, and which raises (or goes another way) once what it reads
# is gone.
def graph_unused_attribute(obj, x):
    h = torch.relu(x) + 1
    obj.extra
    h = torch.tanh(h * 2 - 3) * 2 + 1
    return torch.relu(h) + h


def graph_unused_item(obj, x):
    h = torch.relu(x) + 1
    obj.table["extra"]
    h = torch.tanh(h * 2 - 3) * 2 + 1
    return torch.relu(h) + h


def graph_unused_place(obj, x):
    h = torch.relu(x) + 1
    obj.row[2]
    h = torch.tanh(h * 2 - 3) * 2 + 1
    return torch.relu(h) + h


def graph_unused_pair(obj, x):
    h = torch.relu(x) + 1
    first, second = obj.row[:2] if obj.extra else obj.row
    h = torch.tanh(h * 2 - 3) * 2 + 1
    return torch.relu(h) + h


def graph_unused_quotient(obj, x):
    h = torch.relu(x) + 1
    1 / obj.number
    h = torch.tanh(h * 2 - 3) * 2 + 1
    return torch.relu(h) + h


def graph_unused_global(obj, x):
    h = torch.relu(x) + 1
    GRAPH_UNUSED_NAME
    h = torch.tanh(h * 2 - 3) * 2 + 1
    return torch.relu(h) + h


def graph_unused_count(obj, x):
    h = torch.relu(x) + 1
    len(obj.extra)
    h = torch.tanh(h * 2 - 3) * 2 + 1
    return torch.relu(h) + h


GRAPH_UNUSED_NAME = 1


def graph_free_variable():
    scale = 2.0

    def inner(h):
        scale
        return h * 3

    def take():
        nonlocal scale
        del scale

    def put():
        nonlocal scale
        scale = 2.0

    return inner, take, put


GRAPH_FREE = graph_free_variable()


def graph_unused_free(obj, x):
    h = torch.relu(x) + 1
    h = GRAPH_FREE[0](h) * 2 - 3
    return torch.tanh(h) + h


def graph_defaulted(h, a=1.0, b=2.0):
    return h * a


def graph_defaulted_by_name(h, *, a=1.0, b=2.0):
    return h * a


def graph_default_moves(x):
    h = torch.relu(x) + 1
    h = graph_defaulted(h) * 2 - 3
    return torch.tanh(h) + h


def graph_default_by_name_goes(x):
    h = torch.relu(x) + 1
    h = graph_defaulted_by_name(h) * 2 - 3
    return torch.tanh(h) + h


class GraphChangeBox:
    pass


class GraphChangeThing:
    def __init__(self):
        self.value = 0


GRAPH_CHANGE_BOX = GraphChangeBox()
GRAPH_CHANGE_SEEN = []
GRAPH_CHANGE_TABLE = {}


def graph_change_tail(h):
    return (torch.tanh(h * 2) * 2 - 1).sum()


# Functions that change something they made after something else took it.
def graph_change_appends_behind_a_store(x, given):
    h = torch.relu(x) + 1
    made = []
    GRAPH_CHANGE_BOX.items = made
    made.append(given)
    return graph_change_tail(h)


def graph_change_appends_behind_a_list_from_outside(x, given):
    h = torch.relu(x) + 1
    made = []
    GRAPH_CHANGE_SEEN.append(made)
    made.append(given)
    return graph_change_tail(h)


def graph_change_appends_behind_a_dict_from_outside(x, given):
    h = torch.relu(x) + 1
    made = []
    GRAPH_CHANGE_TABLE["made"] = made
    made.append(given)
    return graph_change_tail(h)


def graph_change_appends_to_an_item_of_a_list(x, given):
    h = torch.relu(x) + 1
    inner = []
    outer = [inner]
    GRAPH_CHANGE_BOX.items = outer
    inner.append(given)
    return graph_change_tail(h)


def graph_change_appends_to_an_item_of_a_tuple(x, given):
    h = torch.relu(x) + 1
    inner = []
    GRAPH_CHANGE_BOX.items = (inner, 1)
    inner.append(given)
    return graph_change_tail(h)


def graph_change_appends_to_what_an_object_holds(x, given):
    h = torch.relu(x) + 1
    inner = []
    thing = GraphChangeThing()
    thing.value = inner
    GRAPH_CHANGE_BOX.items = thing
    inner.append(given)
    return graph_change_tail(h)


def graph_change_appends_around_a_store(x, given):
    h = torch.relu(x) + 1
    made = []
    made.append(given)
    GRAPH_CHANGE_BOX.items = made
    made.append(given + 1)
    return graph_change_tail(h)


def graph_change_appends_a_size(x, given):
    h = torch.relu(x) + 1
    made = []
    GRAPH_CHANGE_BOX.items = made
    made.append(h.shape[0] + given)
    return graph_change_tail(h)


def graph_change_appends_a_result(x, given):
    h = torch.relu(x) + 1
    made = []
    GRAPH_CHANGE_BOX.items = made
    made.append(h * given)
    return graph_change_tail(h)


def graph_change_stores_behind_a_store(x, given):
    h = torch.relu(x) + 1
    thing = GraphChangeThing()
    GRAPH_CHANGE_BOX.items = thing
    thing.value = given
    return graph_change_tail(h)


def graph_change_stores_behind_a_list_from_outside(x, given):
    h = torch.relu(x) + 1
    thing = GraphChangeThing()
    GRAPH_CHANGE_SEEN.append([thing])
    thing.value = given
    thing.other = given + 1
    return graph_change_tail(h)


def graph_change_ends_an_iterator(x, given):
    h = torch.relu(x) + 1
    going = iter((h.shape[0], h.shape[1]))
    GRAPH_CHANGE_BOX.items = going
    for step in going:
        pass
    return graph_change_tail(h)


def graph_change_goes_over_a_tuple(x, given):
    h = torch.relu(x) + 1
    for step in (given, given + 1.0):
        h = h + step
    return graph_change_tail(h)


# Functions with an `extend` of a list from outside, which waits for the
# path to land, in front of a change to the list it is given.
def graph_waits_for_a_list_given_back(x, given):
    h = torch.relu(x) + 1
    made = [0]
    GRAPH_CHANGE_SEEN.extend(made)
    made.append(given)
    return graph_change_tail(h), made


def graph_waits_for_a_list_that_is_stored(x, given):
    h = torch.relu(x) + 1
    made = [0]
    GRAPH_CHANGE_SEEN.extend(made)
    made.append(h.shape[1] + given)
    GRAPH_CHANGE_BOX.items = made
    return graph_change_tail(h)


def graph_waits_for_a_list_nothing_else_takes(x, given):
    h = torch.relu(x) + 1
    made = [0]
    GRAPH_CHANGE_SEEN.extend(made)
    made.append(given)
    return graph_change_tail(h)


def graph_waits_for_a_list_made_of_a_tuple(x, given):
    h = torch.relu(x) + 1
    pair = (given, 0)
    made = list(pair)
    GRAPH_CHANGE_SEEN.extend(made)
    made.append(given)
    return graph_change_tail(h), made


def graph_waits_for_a_list_from_outside(x, given):
    h = torch.relu(x) + 1
    row = GRAPH_CHANGE_TABLE["row"]
    row.append(given)
    GRAPH_CHANGE_SEEN.extend(row)
    row.append(given + 1)
    return graph_change_tail(h)


# Functions that change a list they made through another name for it.
def graph_other_way_through_a_dict(x, given):
    h = torch.relu(x) + 1
    made = [0]
    other = {"k": made}.get("k")
    other.append(1)
    made.append(given)
    return graph_change_tail(h), made


def graph_other_way_through_a_method(x, given):
    h = torch.relu(x) + 1
    made = [0]
    other = made.append.__self__
    other.append(1)
    made.append(given)
    return graph_change_tail(h), made


def graph_other_way_through_a_default(x, given):
    # The list itself where the dict from outside has no "k", and the
    # list that dict has there otherwise.
    h = torch.relu(x) + 1
    made = [0]
    other = GRAPH_CHANGE_TABLE.get("k", made)
    other.append(1)
    made.append(given)
    return graph_change_tail(h), made


def graph_other_way_extends(x, given):
    h = torch.relu(x) + 1
    made = [0]
    other = GRAPH_CHANGE_TABLE.get("k", made)
    GRAPH_CHANGE_SEEN.extend(other)
    made.append(given)
    return graph_change_tail(h), made


# Functions that go over what they are given.
def graph_goes_over(x, given):
    h = torch.relu(x) + 1
    for step in given:
        h = h + step
    return graph_change_tail(h)


def graph_goes_over_two(x, given):
    h = torch.relu(x) + 1
    for step, scale in zip(given, (1.0, 2.0, 3.0)):
        h = h + step * scale
    return graph_change_tail(h)


def graph_goes_over_under_tuple(x, given):
    h = torch.relu(x) + 1
    got = tuple(given)
    return graph_change_tail(h * got[0] + len(got))


def graph_gives_loudly():
    for step in (1.0, 2.0, 3.0):
        GRAPH_CHANGE_SEEN.append(step)
        yield step


def graph_pinned_scaled(h, scale=2.0, shift=1.0):
    return h * scale + shift


def graph_pins_a_place(x, given):
    h = torch.relu(x) + 1
    items = [h, h * 2, h * 3]
    return (items[given] * 0.5).sum()


def graph_pins_a_place_of_a_tuple(x, given):
    h = torch.relu(x) + 1
    items = (h, h * 2, h * 3)
    return (items[given] * 0.5).sum()


def graph_pins_a_number(x, given):
    h = torch.relu(x) + 1
    return (h * (h.shape[0] * given)).sum()


def graph_pins_a_rate(x, given):
    h = (torch.relu(x) + 1).reshape(1, 3, 4)
    attend = torch.nn.functional.scaled_dot_product_attention
    return attend(h, h, h, dropout_p=given).sum()


def graph_pins_names(x, given):
    h = torch.relu(x) + 1
    names = {name: 2.0 for name in (given,)}  # noqa: C420
    return graph_pinned_scaled(h, **names).sum()


def graph_pins_a_function(x, given):
    h = torch.relu(x) + 1
    return (h * given(h.shape[0], h.shape[1])).sum()


def graph_pins_a_class(x, given):
    h = torch.relu(x) + 1
    return (h * (2.0 if isinstance(h.shape, given) else 3.0)).sum()


def graph_pins_defaults(x, given):
    graph_pinned_scaled.__defaults__ = given
    return graph_pinned_scaled(torch.relu(x) + 1).sum()


def graph_pins_ten(x, given):
    h = torch.relu(x) + 1
    items = [h * (at + 1) for at in range(10)]
    return (items[given] * 0.5).sum()


def graph_spread_list(x, given):
    h = torch.relu(x) + 1
    made = [h, 0.5]
    spread = [*made]
    made.append(given)
    return (torch.tanh(torch.clamp(*spread) * 2) * 2 - 1).sum()


def graph_spread_tuple(x, given):
    h = torch.relu(x) + 1
    made = [h, 0.5]
    spread = (*made,)
    made.append(given)
    return (torch.tanh(torch.clamp(*spread) * 2) * 2 - 1).sum()


def graph_spread_twice(x, given):
    h = torch.relu(x) + 1
    made = [h]
    spread = [*made, 0.5]
    made.append(given)
    again = (*spread,)
    spread.append(given)
    return (torch.tanh(torch.clamp(*again) * 2) * 2 - 1).sum()


# How long one case of `GRAPH_THINGS` may take before it counts as one that
# does not come back.
GRAPH_CASE_SECONDS = 300
GRAPH_THINGS = '''"""The things a function makes between its operations (a list, a dict,
an object, a cell, an iterator), on a path: the cases of a group, each with
`Graphs()` alone, with a pass that returns the graph and with an input that
requires a gradient, twelve calls against eager from the same state. The
groups: `made` (things that hold themselves), `outside` (a list that holds
itself, handed to the function), `printed` (some of `made`), `sides` (what
reads and what changes one thing, on the two sides of the graph), `own`
(what comes from outside the path through an iterator, an item, an
attribute, a default or a variable of an outer function). What is compared
is the result or what is raised, and what was given back and left outside,
as shapes that say which object is which. One line `start NAME` before a
case and one `done NAME {...}` behind it: a case that does not come back is
the one started last."""
import gc
import json
import resource
import sys
import types

import torch
import torch.vm

P, DEFAULT = torch.vm.passes, torch.vm.DEFAULT_PASSES
DEVICE, WANTED = sys.argv[1], sys.argv[2:]


class Box:
    pass


class Thing:
    def method(self):
        return 1


class Ends:
    def __del__(self):
        LOG.append("ended")
        STATE[0] += 1.0


BOX = Box()
LOG = []
STATE = [0.5]
HELD = []


def keep(gm, input_types):
    return gm


def tail(h):
    return (torch.tanh(h * 2) * 2 - 1).sum()


def two_lists_given_back(x, given):
    h = torch.relu(x) + 1
    a = []
    b = [a]
    a.append(b)
    return tail(h), a


def two_lists_stored(x, given):
    h = torch.relu(x) + 1
    a = []
    b = [a]
    a.append(b)
    BOX.items = a
    b.append(given)
    a.append(given + 1)
    return tail(h)


def two_lists_dropped(x, given):
    h = torch.relu(x) + 1
    a = []
    b = [a]
    a.append(b)
    b.append(given)
    return tail(h)


def a_list_in_itself(x, given):
    h = torch.relu(x) + 1
    a = [given]
    a.append(a)
    BOX.items = a
    return tail(h)


def a_list_in_itself_given_back(x, given):
    h = torch.relu(x) + 1
    a = [given]
    a.append(a)
    return tail(h), a


def a_list_in_itself_many_times(x, given):
    h = torch.relu(x) + 1
    a = [given]
    for _ in range(40):
        a.append(a)
    BOX.items = len(a)
    return tail(h), a


def a_list_and_a_dict(x, given):
    h = torch.relu(x) + 1
    a = []
    d = {"a": a}
    a.append(d)
    BOX.items = d
    return tail(h)


def a_tuple_through_a_list(x, given):
    h = torch.relu(x) + 1
    a = [given]
    t = (a, given)
    a.append(t)
    BOX.items = t
    return tail(h), a


def a_set_through_an_object(x, given):
    h = torch.relu(x) + 1
    thing = Thing()
    members = {thing}
    thing.members = members
    thing.value = given
    BOX.items = thing
    return tail(h)


def an_object_in_itself(x, given):
    h = torch.relu(x) + 1
    thing = Thing()
    thing.me = thing
    thing.value = given
    BOX.items = thing
    return tail(h)


def a_cell_through_a_function(x, given):
    h = torch.relu(x) + 1

    def inner(n):
        return inner if n else given

    BOX.items = inner
    return tail(h)


def a_bound_method_in_its_list(x, given):
    h = torch.relu(x) + 1
    a = [given]
    a.append(a.append)
    BOX.items = a
    return tail(h)


def a_generator_that_holds_its_list(x, given):
    h = torch.relu(x) + 1
    out = [given]
    gen = (len(out) + v for v in (1, 2))
    out.append(gen)
    BOX.items = out
    return tail(h)


def a_cycle_read_through(x, given):
    h = torch.relu(x) + 1
    a = []
    b = [a]
    a.append(b)
    c = a[0][0][0]
    c.append(given)
    return tail(h), a


def a_cycle_with_a_size(x, given):
    h = torch.relu(x) + 1
    a = [h.shape[0] + given]
    a.append(a)
    return tail(h), a


def a_cycle_with_a_result(x, given):
    h = torch.relu(x) + 1
    a = [h]
    a.append(a)
    BOX.items = a
    return tail(h)


def a_cycle_spread(x, given):
    h = torch.relu(x) + 1
    a = [given]
    b = [*a, a]
    a.append(b)
    return tail(h), b


def a_cycle_spread_into_a_tuple(x, given):
    h = torch.relu(x) + 1
    a = [given]
    b = (*a, a)
    a.append(b)
    a.append(given + 1)
    return tail(h), b


def a_cycle_handed_to_an_operation(x, given):
    h = torch.relu(x) + 1
    parts = [h, h * 2]
    parts.append(parts)
    return tail(torch.cat(parts))


def a_cycle_as_the_sizes_of_an_operation(x, given):
    h = torch.relu(x) + 1
    sizes = [2, 6]
    sizes.append(sizes)
    return tail(h.reshape(sizes[0], sizes[1])), sizes


def a_cycle_gone_over(x, given):
    h = torch.relu(x) + 1
    a = [given]
    a.append(a)
    total = 0
    for item in a:
        if isinstance(item, list):
            total = total + len(item)
        else:
            total = total + item
    return tail(h) * total


def a_cycle_asked_which_it_is(x, given):
    h = torch.relu(x) + 1
    a = [given]
    a.append(a)
    same = a[1] is a
    return tail(h) + (1 if same else 0), a


def a_cycle_formatted(x, given):
    h = torch.relu(x) + 1
    a = [given]
    a.append(a)
    BOX.items = f"{a}"
    return tail(h)


def a_cycle_extended_outside(x, given):
    h = torch.relu(x) + 1
    a = [given]
    a.append(a)
    HELD.extend(a)
    a.append(given + 1)
    return tail(h)


# A holder, an alias or a read IN FRONT of what makes the thing late.
def holder_tuple_of_a_list_in_front(x, given):
    h = torch.relu(x) + 1
    a = [given]
    b = [a]
    c = (b, 1)
    a.append(h)
    BOX.items = c
    return tail(h)


def holder_unpacked_in_front(x, given):
    h = torch.relu(x) + 1
    a = [given]
    b = [a]
    c = (b, given)
    a.append(h)
    first, second = c
    BOX.items = first
    return tail(h) * second


def holder_dict_in_front(x, given):
    h = torch.relu(x) + 1
    a = [given]
    d = {"a": a}
    e = [d]
    a.append(h)
    BOX.items = e
    return tail(h)


def holder_object_in_front(x, given):
    h = torch.relu(x) + 1
    a = [given]
    thing = Thing()
    thing.items = a
    holder = [thing]
    a.append(h)
    BOX.items = holder
    return tail(h)


def iterator_in_front(x, given):
    h = torch.relu(x) + 1
    a = [given]
    it = iter(a)
    a.append(h)
    BOX.items = list(it)
    return tail(h)


def alias_in_front(x, given):
    h = torch.relu(x) + 1
    a = [given]
    other = {"k": a}.get("k")
    a.append(h)
    BOX.items = (len(other), other)
    return tail(h)


def length_in_front(x, given):
    h = torch.relu(x) + 1
    a = [given]
    b = [a]
    n = len(b)
    a.append(h)
    BOX.items = n
    return tail(h) * n


def slice_in_front(x, given):
    h = torch.relu(x) + 1
    a = [given]
    b = [a, given]
    c = b[:1]
    a.append(h)
    BOX.items = c
    return tail(h)


# The same BEHIND what makes the thing late.
def item_behind(x, given):
    h = torch.relu(x) + 1
    a = [given]
    b = [a, given]
    a.append(h)
    return tail(h) * b[-1], b


def loop_behind(x, given):
    h = torch.relu(x) + 1
    a = [given]
    b = [a]
    a.append(h)
    for item in b:
        BOX.items = item[-1]
    return tail(h)


def length_through_an_item_behind(x, given):
    h = torch.relu(x) + 1
    a = [given]
    b = [a]
    a.append(h)
    for item in b:
        BOX.items = len(item)
    return tail(h)


def dict_read_behind(x, given):
    h = torch.relu(x) + 1
    a = [given]
    d = {"a": a}
    a.append(h)
    BOX.items = d.get("a")[-1]
    return tail(h)


def holder_behind(x, given):
    h = torch.relu(x) + 1
    a = [given]
    a.append(h)
    b = [a]
    a.append(given + 1)
    c = (b, len(a))
    BOX.items = c
    return tail(h)


# A read that runs behind the graph, and a change that stands behind it in the function.
def joined_then_appended(x, given):
    h = torch.relu(x) + 1
    made = [given]
    joined = made + [h.shape[0]]
    made.append(given + 1)
    return tail(h), joined


def sliced_by_a_size_then_appended(x, given):
    h = torch.relu(x) + 1
    made = [given, given + 1, given + 2]
    part = made[-(h.shape[0] - 1):]
    made.append(given + 3)
    return tail(h), part


def extended_behind_a_size_then_appended(x, given):
    h = torch.relu(x) + 1
    made = [given]
    joined = [h.shape[0], *made]
    made.append(given + 1)
    return tail(h), joined


def extended_in_front_of_a_size_then_appended(x, given):
    h = torch.relu(x) + 1
    made = [given]
    joined = [*made, h.shape[0]]
    made.append(given + 1)
    return tail(h), joined


def spread_behind_a_result_then_appended(x, given):
    h = torch.relu(x) + 1
    bounds = [0.25, 1.75]
    args = (h, *bounds)
    bounds.append(given)
    return tail(torch.clamp(*args)), bounds


def iterator_drawn_behind_a_size_then_drawn(x, given):
    h = torch.relu(x) + 1
    it = iter([1.0, 2.0, 3.0, given])
    pairs = list(zip(it, h.shape))
    last = 0.0
    for last in it:
        break
    BOX.items = (pairs, last)
    return tail(h) * last


def dict_of_sizes(x, given):
    h = torch.relu(x) + 1
    made = {name: h.shape[at] + given for at, name in enumerate(("rows", "columns"))}
    holder = [made]
    BOX.items = holder
    return tail(h) * made["rows"]


def cell_set_behind(x, given):
    h = torch.relu(x) + 1
    count = given
    seen = [count]
    count = h.shape[0]
    seen.append(count)
    BOX.items = seen
    out = tail(h)

    def later():
        return count

    HELD.append(later)
    return out


def two_things_one_late(x, given):
    h = torch.relu(x) + 1
    plain = [given]
    late = [given]
    both = (plain, late)
    late.append(h)
    plain.append(given + 1)
    n = len(plain)
    BOX.items = both
    return tail(h) * n


def a_thing_put_into_a_late_one(x, given):
    h = torch.relu(x) + 1
    late = [h]
    plain = [given]
    late.append(plain)
    plain.append(given + 1)
    BOX.items = (late[1][-1], len(plain))
    return tail(h), late


THINGS = [Thing(), Thing()]
TABLE = {"a": Thing(), "b": Thing()}
HOLDER = Thing()
HOLDER.inner = Thing()
WATCHED = [THINGS[0], THINGS[1], TABLE["a"], TABLE["b"], HOLDER.inner]


# What an iterator of the path gives, an item, an attribute, a default or a
# variable of an outer function, where what it comes from is from outside the
# path: an object from outside, and a store into it waits.
def own_loop(x, given):
    h = torch.relu(x) + 1
    for thing in THINGS:
        thing.seen = given
    return tail(h)


def own_item(x, given):
    h = torch.relu(x) + 1
    THINGS[0].seen = given
    TABLE["b"].seen = given + 1
    return tail(h)


def own_attribute(x, given):
    h = torch.relu(x) + 1
    HOLDER.inner.seen = given
    return tail(h)


def stores_into_its_default(given, thing=THINGS[1]):
    thing.seen = given


def own_default(x, given):
    h = torch.relu(x) + 1
    stores_into_its_default(given)
    return tail(h)


def stores_into_an_outer_variable():
    thing = Thing()
    WATCHED.append(thing)

    def own_cell(x, given):
        h = torch.relu(x) + 1
        thing.seen = given
        return tail(h)

    return own_cell


own_cell = stores_into_an_outer_variable()


# The same through a holder the path built of things from outside: what is
# taken out of it is from outside.
def own_enumerate_of_a_list_of_outside(x, given):
    h = torch.relu(x) + 1
    for at, thing in enumerate([THINGS[1], THINGS[0]]):
        thing.seen = at + given
    return tail(h)


def own_pairs(x, given):
    h = torch.relu(x) + 1
    pairs = list(zip(THINGS, (1, 2)))
    for thing, n in pairs:
        thing.seen = n + given
    return tail(h)


def stores_into_what_it_is_handed_by_name(given, **kwargs):
    for name, thing in kwargs.items():
        thing.seen = (name, given)
    return kwargs.get("first")


def own_names(x, given):
    h = torch.relu(x) + 1
    things = {"first": THINGS[0], "second": THINGS[1]}
    first = stores_into_what_it_is_handed_by_name(given, **things)
    first.other = given
    return tail(h)


OWN = [
    own_loop, own_item, own_attribute, own_default, own_cell,
    own_enumerate_of_a_list_of_outside, own_pairs, own_names,
]


def ones():
    return torch.ones(3, 4, device=DEVICE)


def loop_of(*items):
    made = list(items)
    made.append(made)
    return made


def outside_read(x, given):
    h = torch.relu(x) + 1
    return tail(h * given[0]), given[-1] is given


def outside_counted(x, given):
    h = torch.relu(x) + 1
    return tail(h) * len(given)


def outside_stored(x, given):
    h = torch.relu(x) + 1
    BOX.items = given
    return tail(h)


def outside_formatted(x, given):
    h = torch.relu(x) + 1
    BOX.items = f"{given}"
    return tail(h)


def outside_handed_to_an_operation(x, given):
    h = torch.relu(x) + 1
    return tail(torch.cat([h, *given]))


def outside_as_the_sizes_of_an_operation(x, given):
    h = torch.relu(x) + 1
    return tail(h.reshape(given))


def outside_as_an_index(x, given):
    h = torch.relu(x) + 1
    return tail(h[given])


def outside_let_go_of(x, given):
    held = HELD.pop()
    h = (torch.relu(x) + 1) * STATE[0]
    held = None
    return tail(h + STATE[0]) + STATE[0], held


def shape(value, seen=None):
    """What a value is made of, with each object that is met a second time
    named by when it was met first."""
    seen = {} if seen is None else seen
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), round(float(value.detach().sum()), 4))
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if id(value) in seen:
        return ("met", seen[id(value)])
    seen[id(value)] = len(seen)
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, *[shape(item, seen) for item in value])
    if isinstance(value, dict):
        return ("dict", *[(key, shape(item, seen)) for key, item in value.items()])
    if isinstance(value, (set, frozenset)):
        return ("set", *sorted(repr(shape(item, seen)) for item in value))
    if isinstance(value, Thing):
        return ("Thing", shape(vars(value), seen))
    if isinstance(value, types.CellType):
        return ("cell", shape(value.cell_contents, seen))
    if isinstance(value, types.FunctionType):
        return ("function", value.__name__, shape(value.__closure__, seen))
    if isinstance(value, (types.BuiltinMethodType, types.MethodType)):
        return ("method", value.__name__, shape(value.__self__, seen))
    if isinstance(value, types.GeneratorType):
        return ("generator", shape(tuple(value), seen))
    return type(value).__name__


def reset():
    for thing in WATCHED:
        vars(thing).pop("seen", None)
        vars(thing).pop("other", None)
    BOX.items = None
    del LOG[:]
    del HELD[:]
    STATE[0] = 0.5


def calls(run, quiet, loud, graded, plant):
    reset()
    gc.collect()
    torch.manual_seed(1234)
    found = []
    for at in range(12):
        given = loud() if at in (7, 10) else quiet()
        if plant is not None:
            plant(at)
        x = torch.full((3, 4), 0.25 * (at + 1), device=DEVICE, requires_grad=graded)
        try:
            out = run(x, given)
            rest = out[1:] if isinstance(out, tuple) else ()
            out = out[0] if isinstance(out, tuple) else out
            if graded:
                out.backward()
            grad = None if x.grad is None else round(float(x.grad.sum()), 4)
            said = (round(out.item(), 4), shape(rest), grad)
        except Exception as error:
            said = f"{type(error).__name__}: {str(error)[:60]}"
        del given
        log = list(LOG)
        gc.collect()
        left = shape([BOX.items, list(HELD), WATCHED])
        found.append(repr((said, left, log, list(LOG), STATE[0])))
        BOX.items = None
    return found


def plant_cycle(at):
    # At calls 7 and 10 what the function lets go of is a list that holds
    # itself and an object whose end is heard; at the others a list of
    # numbers.
    held = [1.0]
    if at in (7, 10):
        held.append(held)
        held.append(Ends())
    HELD.append(held)


def one():
    return 1


def two():
    return 2


MADE = [
    two_lists_given_back, two_lists_stored, two_lists_dropped, a_list_in_itself,
    a_list_in_itself_given_back, a_list_in_itself_many_times, a_list_and_a_dict,
    a_tuple_through_a_list, a_set_through_an_object, an_object_in_itself,
    a_cell_through_a_function, a_bound_method_in_its_list,
    a_generator_that_holds_its_list, a_cycle_read_through, a_cycle_with_a_size,
    a_cycle_with_a_result, a_cycle_spread, a_cycle_spread_into_a_tuple,
    a_cycle_handed_to_an_operation, a_cycle_as_the_sizes_of_an_operation,
    a_cycle_gone_over, a_cycle_asked_which_it_is, a_cycle_formatted,
    a_cycle_extended_outside,
]
# What it is called, with the name of the function: the function, what it is
# given at most calls, what at calls 7 and 10.
OUTSIDE = [
    ("", outside_read, lambda: loop_of(2.0), lambda: loop_of(3.0)),
    ("", outside_counted, lambda: [1.0, 2.0], lambda: loop_of(1.0)),
    ("", outside_stored, lambda: loop_of(1.0), lambda: loop_of(2.0)),
    ("", outside_formatted, lambda: [1.0], lambda: loop_of(1.0)),
    ("_always", outside_formatted, lambda: loop_of(1.0), lambda: loop_of(1.0)),
    ("", outside_handed_to_an_operation, lambda: [ones()], lambda: loop_of(ones())),
    ("", outside_as_the_sizes_of_an_operation, lambda: [2, 6], lambda: loop_of(2, 6)),
    ("", outside_as_an_index, lambda: [0, 1], lambda: loop_of(0, 1)),
    ("", outside_let_go_of, one, two),
]


WAYS = (("graphs", (), False), ("pass", (keep,), False), ("gradient", (), True))


def try_it(name, fn, quiet, loud, plant):
    print(f"start {name}", flush=True)
    said = {}
    for label, behind, graded in WAYS:
        passes = (*DEFAULT, P.Graphs(), *behind)
        compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
        got = calls(compiled, quiet, loud, graded, plant)
        expected = calls(fn, quiet, loud, graded, plant)
        differ = [(at, a, b) for at, (a, b) in enumerate(zip(got, expected)) if a != b]
        found = torch.vm.regions(compiled)
        str(found)
        own = [g for g in found if g["where"].startswith(fn.__qualname__ + " ")]
        graphs = [(g["ops"], g.get("hits")) for g in own if "ops" in g]
        first = [str(part)[:300] for part in differ[:1]]
        said[label] = {"differ": len(differ), "first": first, "graphs": graphs}
    print(f"done {name} " + json.dumps(said), flush=True)


PRINTED = (
    "two_lists_given_back",
    "a_list_and_a_dict",
    "an_object_in_itself",
)
SIDES = [
    holder_tuple_of_a_list_in_front,
    holder_unpacked_in_front,
    holder_dict_in_front,
    holder_object_in_front,
    iterator_in_front,
    alias_in_front,
    length_in_front,
    slice_in_front,
    item_behind,
    loop_behind,
    length_through_an_item_behind,
    dict_read_behind,
    holder_behind,
    joined_then_appended,
    sliced_by_a_size_then_appended,
    extended_behind_a_size_then_appended,
    extended_in_front_of_a_size_then_appended,
    spread_behind_a_result_then_appended,
    iterator_drawn_behind_a_size_then_drawn,
    dict_of_sizes,
    cell_set_behind,
    two_things_one_late,
    a_thing_put_into_a_late_one,
]

if __name__ == "__main__":
    # A case that ends the process is to be seen at once, as that: no core is
    # written (where cores are handed to a collector, the process is held for
    # minutes and looks like a call that does not come back).
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    gc.disable()
    cases = []
    cases += [(fn.__name__, "made", fn, one, two) for fn in MADE]
    for more, fn, quiet, loud in OUTSIDE:
        cases.append((fn.__name__ + more, "outside", fn, quiet, loud))
    cases += [(fn.__name__, "sides", fn, one, two) for fn in SIDES]
    cases += [(fn.__name__, "own", fn, one, two) for fn in OWN]
    for name, group, fn, quiet, loud in cases:
        wanted = not WANTED or name in WANTED or group in WANTED
        if wanted or ("printed" in WANTED and name in PRINTED):
            plant = plant_cycle if fn is outside_let_go_of else None
            try_it(name, fn, quiet, loud, plant)
    print("all done", flush=True)
'''


def graph_key_dict_of_a_comprehension(x, given):
    h = torch.relu(x) + 1
    made = {key: 1 for key in (given,)}  # noqa: C420, F841
    return (torch.tanh(h * 2) * 2 - 1).sum()


def graph_key_list_of_what_is_spread(x, given):
    h = torch.relu(x) + 1
    made = [*given]  # noqa: F841
    return (torch.tanh(h * 2) * 2 - 1).sum()


def graph_key_dict_of_what_is_spread(x, given):
    h = torch.relu(x) + 1
    made = {**given}  # noqa: F841
    return (torch.tanh(h * 2) * 2 - 1).sum()


GRAPH_KEY_LOG = []


class GraphKeyText(str):
    __slots__ = ()

    def __hash__(self):
        GRAPH_KEY_LOG.append("hash")
        return str.__hash__(self)

    def __eq__(self, other):
        GRAPH_KEY_LOG.append("eq")
        return str.__eq__(self, other)


def graph_key_scaled(h, scale=2.0, shift=1.0):
    return h * scale + shift


def graph_key_stored_and_read(x, given):
    h = torch.relu(x) + 1
    made = {}
    made[given] = h * 2
    return (torch.tanh(made[given]) * 2 - 1).sum()


def graph_key_spread(x, given):
    h = torch.relu(x) + 1
    made = {}
    made[given] = 3.0
    return ((graph_key_scaled(h, **made) * 0.1).sin() * 2 - 1).sum()


class GraphWithinShared(torch.nn.Module):
    def forward(self, h):
        return torch.relu(h) + 1


class GraphWithinInner(torch.nn.Module):
    def __init__(self, loud):
        super().__init__()
        self.shared = GraphWithinShared()
        self.loud = loud

    def forward(self, h):
        h = torch.tanh(self.shared(h) * 2) - 1
        if self.loud:
            # No path goes on from here: one of a caller that came this far
            # is cut back in front of the call.
            print(end="")
        return h.cos()


class GraphWithinModel(torch.nn.Module):
    def __init__(self, loud):
        super().__init__()
        self.shared = GraphWithinShared()
        self.inner = GraphWithinInner(loud)

    def forward(self, x):
        h = torch.tanh(self.shared(x) * 2) - 1
        return self.inner(h).sum()


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmGraphResults(TestCase):
    """What a function computes with graphs: with those the engine runs
    itself, and with what a pass made of them."""

    hw_classification = HardwareClassification.ACCELERATOR

    def step(self, fn, x, w):
        out = fn(x, w)
        return out, torch.autograd.grad(out, (x, w))

    def test_the_engine_runs_a_graph_with_the_results_of_eager_mode(self, device):
        compiled = torch.vm(graph_mixed, passes=GRAPHS, fallback=False)
        for _ in range(5):
            x = torch.randn(4, 8, device=device, requires_grad=True)
            w = torch.randn(8, 8, device=device, requires_grad=True)
            out, grads = self.step(compiled, x, w)
            expected = self.step(graph_mixed, x, w)
            self.assertEqual(out, expected[0], atol=0, rtol=0)
            self.assertEqual(grads, expected[1], atol=0, rtol=0)
            # On the tape of eager mode: the node of the last operation.
            self.assertEqual(type(out.grad_fn).__name__, "AddBackward0")
        (graph,) = torch.vm.regions(compiled)
        self.assertEqual((graph["state"], graph["ops"]), ("built", 6))

    def test_an_exported_graph_runs_on_the_kernels_of_eager_mode(self, device):
        graphs = []

        def keep(gm, input_types):
            graphs.append([(t.dtype, str(t.device)) for t in input_types])
            return gm

        passes = (*DEFAULT, P.Graphs(), keep)
        compiled = torch.vm(graph_mixed, passes=passes, fallback=False)
        x = torch.randn(4, 8, device=device, requires_grad=True)
        w = torch.randn(8, 8, device=device, requires_grad=True)
        for _ in range(5):
            out, grads = self.step(compiled, x, w)
            expected = self.step(graph_mixed, x, w)
            self.assertEqual(out, expected[0], atol=0, rtol=0)
            self.assertEqual(grads, expected[1], atol=0, rtol=0)
        (graph,) = torch.vm.regions(compiled)
        self.assertEqual((graph["state"], graph["variants"]), ("compiled", 1))
        self.assertGreaterEqual(graph["hits"], 3)
        self.assertEqual(graphs, [[(torch.float32, str(x.device))] * 2])

    @parametrize("compiler", ("AOTAutograd", "Inductor"))
    def test_a_pass_that_compiles_a_graph(self, device, compiler):
        passes = (*DEFAULT, P.Graphs(), getattr(P, compiler)())
        compiled = torch.vm(graph_mixed, passes=passes, fallback=False)
        x = torch.randn(4, 8, device=device, requires_grad=True)
        w = torch.randn(8, 8, device=device, requires_grad=True)
        nodes = []
        for _ in range(6):
            out, grads = self.step(compiled, x, w)
            expected = self.step(graph_mixed, x, w)
            self.assertEqual(out, expected[0], rtol=1e-4, atol=1e-5)
            self.assertEqual(grads, expected[1], rtol=1e-4, atol=1e-5)
            nodes.append(type(out.grad_fn).__name__)
        # The profiling run and the run that was exported are the block's
        # own; then the backward is the one that was traced ahead of time.
        self.assertEqual(nodes[:2], ["AddBackward0"] * 2)
        self.assertEqual(nodes[2:], ["CompiledFunctionBackward"] * 4)
        (graph,) = torch.vm.regions(compiled)
        self.assertEqual((graph["state"], graph["variants"]), ("compiled", 1))
        self.assertEqual((graph["hits"], graph["misses"]), (4, 1))
        # Without gradients it is another graph, with no backward.
        with torch.no_grad():
            for _ in range(3):
                out = compiled(x, w)
                self.assertEqual(out, graph_mixed(x, w), rtol=1e-4, atol=1e-5)
        self.assertIsNone(out.grad_fn)
        self.assertEqual(torch.vm.regions(compiled)[0]["variants"], 2)

    @parametrize("case", ("cat", "stack", "masked", "subscripts", "conv", "listed"))
    def test_an_operation_that_is_no_call_site_is_a_node_of_a_graph(self, device, case):
        def keep(gm, input_types):
            return gm

        fn, args = graph_operands(case, device)
        operator = {"masked": "index.Tensor", "subscripts": "index.Tensor"}
        operator.update(conv="conv2d", listed="cat")
        operator = "aten::" + operator.get(case, case)
        expected = fn(*args)
        # With the kernels of eager mode: the engine's own graph, and the
        # one a pass over graphs is handed.
        for behind in ((), (keep,)):
            passes = (*DEFAULT, P.Graphs(), *behind)
            compiled = torch.vm(fn, passes=passes, fallback=False)
            for _ in range(4):
                self.assertEqual(compiled(*args), expected, atol=0, rtol=0)
            (graph,) = [r for r in torch.vm.regions(compiled) if "ops" in r]
            self.assertIn(operator, graph["operators"])
            self.assertEqual(graph["state"], "compiled" if behind else "built")
            self.assertEqual(graph["hits"], 2 if behind else 0)
        if case == "masked":
            # How many items a mask picks is not known to a graph compiler.
            return
        passes = (*DEFAULT, P.Graphs(), P.AOTAutograd())
        compiled = torch.vm(fn, passes=passes, fallback=False)
        for _ in range(4):
            self.assertEqual(compiled(*args), expected, rtol=1e-5, atol=1e-6)
        (graph,) = [r for r in torch.vm.regions(compiled) if "ops" in r]
        self.assertEqual((graph["state"], graph["hits"]), ("compiled", 2))

    @parametrize("case", ("branches", "loops", "collects", "size"))
    def test_a_path_computes_what_the_function_computes(self, device, case):
        def keep(gm, input_types):
            return gm

        fn, calls = graph_paths(case, device)
        for behind in ((), (keep,), (P.AOTAutograd(),), (P.Inductor(),)):
            passes = (*DEFAULT, P.Graphs(), *behind)
            compiled = torch.vm(fn, passes=passes, fallback=False)
            exact = behind in ((), (keep,))
            close = {"atol": 0, "rtol": 0} if exact else {"rtol": 1e-4, "atol": 1e-5}
            for args in calls:
                self.assertEqual(compiled(*args), fn(*args), **close)
            (graph,) = [r for r in torch.vm.regions(compiled) if "ops" in r]
            if not behind:
                self.assertEqual((graph["state"], graph["hits"]), ("built", 0))
                continue
            # Two ways through the function: two paths of one graph, each
            # with what was made of it.
            self.assertEqual((graph["state"], graph["paths"]), ("compiled", 2))
            self.assertEqual(graph["variants"], 2)
            self.assertGreaterEqual(graph["hits"], 2)

    def test_a_graph_is_for_the_dtype_autocast_casts_to(self, device):
        kind = torch.device(device).type
        passes = (*DEFAULT, P.Graphs(), P.Inductor())
        compiled = torch.vm(graph_casts, passes=passes, fallback=False)
        x, w = torch.randn(4, 8, device=device), torch.randn(8, 8, device=device)
        for dtype in (torch.bfloat16, torch.float16, torch.bfloat16):
            with torch.autocast(kind, dtype=dtype):
                for _ in range(4):
                    got = compiled(x, w)
                expected = graph_casts(x, w)
            self.assertEqual(got.dtype, expected.dtype)
            self.assertEqual(got.dtype, dtype)
            self.assertEqual(got.float(), expected.float(), rtol=0.05, atol=0.05)
        (graph,) = [r for r in torch.vm.regions(compiled) if "ops" in r]
        # One form per dtype; the first dtype again is the first form.
        self.assertEqual((graph["variants"], graph["exports"]), (2, 2))
        # Without autocast, and where operations have to be deterministic,
        # the graph is another one as well.
        for _ in range(3):
            self.assertEqual(compiled(x, w), graph_casts(x, w), rtol=1e-4, atol=1e-5)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            for _ in range(3):
                got = compiled(x, w)
                self.assertEqual(got, graph_casts(x, w), rtol=1e-4, atol=1e-5)
        finally:
            torch.use_deterministic_algorithms(False)
        (graph,) = [r for r in torch.vm.regions(compiled) if "ops" in r]
        self.assertEqual((graph["variants"], graph["exports"]), (4, 4))

    def test_writes_to_tensors_a_path_made(self, device):
        torch.manual_seed(0)
        x = torch.randn(2, 4, 8, 8, device=device)
        exact, close = dict(atol=0, rtol=0), dict(rtol=1e-4, atol=1e-5)
        with torch.no_grad():
            for net, ops in ((GraphResNet(), 20), (GraphHead(), 17)):
                net = net.eval().to(device)
                compilers = ((P.AOTAutograd(),), close), ((P.Inductor(),), close)
                for behind, tolerance in (((), exact), *compilers):
                    passes = (*DEFAULT, P.Graphs(), *behind)
                    compiled = torch.vm(net, passes=passes, fallback=False)
                    for _ in range(6):
                        self.assertEqual(compiled(x), net(x), **tolerance)
                    found = torch.vm.regions(compiled)
                    (graph,) = [g for g in found if g.get("ops") == ops]
                    self.assertEqual(graph["hits"] > 0, bool(behind))

    def test_what_changes_on_a_warm_model_that_writes_in_place(self, device):
        def keep(gm, input_types):
            return gm

        def changes(model, x):
            block = model.layer[0]
            yield "weight in place", lambda: block.conv1.weight.mul_(1.5), x
            made = dict(device=device, dtype=x.dtype)
            fresh = torch.nn.Parameter(torch.randn(4, 4, 3, 3, **made))
            yield "parameter replaced", lambda: setattr(block.conv1, "weight", fresh), x
            yield "train", model.train, x
            yield "eval", model.eval, x
            yield "another batch size", lambda: None, torch.randn(5, 4, 8, 8, **made)
            hooks = []

            def hook():
                hooks.append(block.register_forward_hook(lambda m, i, o: o * 2))

            def replace():
                model.layer[1] = GraphResidual().eval().to(x)

            yield "forward hook", hook, x
            yield "hook removed", lambda: hooks.pop().remove(), x
            plain = "an activation that does not write"
            yield plain, lambda: setattr(block, "relu", torch.nn.ReLU()), x
            yield "submodule replaced", replace, x
            # Other dtypes and back: half and bfloat16 where convolutions have them.
            on_gpu = torch.device(device).type == "cuda"
            others = (torch.float16, torch.bfloat16) if on_gpu else (torch.float64,)
            for dtype in (*others, torch.float32):
                yield f"{dtype}", lambda dtype=dtype: model.to(dtype), x.to(dtype)

        for behind, exact in (((keep,), True), ((P.Inductor(),), False)):
            torch.manual_seed(0)
            model = GraphResNet().eval().to(device)
            x = torch.randn(2, 4, 8, 8, device=device)
            # The calls that run a graph are counted: no measurement.
            passes = (*DEFAULT, P.Graphs(), *behind)
            compiled = torch.vm(model, passes=passes, fallback=False, demote=False)
            with torch.no_grad():
                for _ in range(8):
                    compiled(x)
                for what, change, given in changes(model, x):
                    change()
                    narrow = given.dtype in (torch.float16, torch.bfloat16)
                    loose = dict(rtol=5e-2, atol=5e-2)
                    close = loose if narrow else dict(rtol=1e-4, atol=1e-5)
                    tolerance = dict(atol=0, rtol=0) if exact else close
                    for _ in range(3):
                        reference = copy.deepcopy(model)
                        got, expected = compiled(given), reference(given)
                        self.assertEqual(got, expected, msg=what, **tolerance)
                        for name, value in reference.state_dict().items():
                            said, held = f"{what}: {name}", model.state_dict()[name]
                            self.assertEqual(held, value, msg=said, **tolerance)
            found = [g for g in torch.vm.regions(compiled) if "ops" in g]
            graph = max(found, key=lambda g: g["hits"])
            self.assertGreaterEqual(graph["hits"], 20)
            self.assertGreaterEqual(graph["ops"], 20)

    def test_what_a_class_reads_itself_is_not_what_a_path_stored(self, device):
        def keep(gm, input_types):
            return gm

        x = torch.randn(3, 4, device=device)
        # A class that reads its attributes in a function of its own gives
        # what that function gives, not what a path stored: such a read
        # behind a store is no part of the path.
        doubles = GraphDoubles()
        passes = (*DEFAULT, P.Graphs(), keep)
        stores = torch.vm(graph_reads_what_it_stored, passes=passes, fallback=False)
        for _ in range(6):
            expected = graph_reads_what_it_stored(x, GraphDoubles())
            self.assertEqual(stores(x, doubles), expected, atol=0, rtol=0)

    def same_as_eager(self, fn, make, calls=4):
        """`fn` under torch.vm, with `Graphs()` alone and with a pass that
        returns the graph, each on arguments that `make` makes anew for
        every call, against eager on others it makes: what is returned, or
        what is raised and with which text, and what the arguments say was
        asked of them (`log`)."""

        def keep(gm, input_types):
            return gm

        def outcome(function, args):
            try:
                return function(*args)
            except Exception as error:
                return type(error).__name__, str(error)

        def alike(got, expected):
            self.assertIs(type(got), type(expected), f"{got!r} for {expected!r}")
            if isinstance(expected, torch.Tensor):
                self.assertEqual(got, expected, atol=0, rtol=0)
            elif isinstance(expected, (tuple, list)):
                self.assertEqual(len(got), len(expected))
                for one, other in zip(got, expected):
                    alike(one, other)
            else:
                self.assertEqual(got, expected)

        for behind in ((), (keep,)):
            passes = (*DEFAULT, P.Graphs(), *behind)
            compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
            for _ in range(calls):
                mine, theirs = make(), make()
                alike(outcome(compiled, mine), outcome(fn, theirs))
                asked = [getattr(arg, "log", None) for arg in mine]
                self.assertEqual(asked, [getattr(arg, "log", None) for arg in theirs])

    def test_len_is_the_index_of_what_a_len_gives(self, device):
        x, same = torch.randn(3, 4, device=device), self.same_as_eager
        # A bool and what has an `__index__` are counts as ints; a negative
        # one and what is no number raise, as `len` raises.
        for count in (True, GraphMidwayIndex(), 3, -1, "a", -(2**70), 2**70):
            same(graph_counts_midway, lambda: (x, GraphMidwaySized(count)))

    def test_isinstance_is_whether_what_was_returned_is_true(self, device):
        x = torch.randn(3, 4, device=device)
        for answer in ("yes", [], 1, 0, None):
            GraphMidwayAsked.answer = answer
            self.same_as_eager(graph_is_of_a_midway_kind, lambda: (x,))
        GraphMidwayAsked.answer = True

    def test_what_an_iter_gives_is_asked_as_the_interpreter_asks_it(self, device):
        x, same = torch.randn(3, 4, device=device), self.same_as_eager
        fn = graph_goes_over_midway
        # What is no iterator raises; an iterator is not asked for an
        # iterator again (one of these cannot be, one starts over if it
        # is); whether the path comes back from the `__iter__` or not.
        gives = (lambda: [1.0, 2.0], lambda: GraphMidwaySteps([1.0, 2.0]))
        gives += (lambda: iter([1.0, 2.0]),)
        for how in ("for", "list", "tuple", "zip", "enumerate", "iter"):
            for given in gives:
                for midway in (True, False):
                    same(fn, lambda: (x, GraphMidwayOver(given, midway), how), calls=3)
        # `list` and `tuple` ask what has a `__len__` for it; `iter` with a
        # sentinel calls what it is given and does not go over it.
        for how in ("list", "tuple"):
            same(fn, lambda: (x, GraphMidwayOverSized(gives[2]), how))
        same(fn, lambda: (x, GraphMidwayOverCalled(gives[2]), "sentinel"))

    def test_a_property_that_raises_attribute_error_asks_getattr(self, device):
        def call(module, x):
            return module(x)

        x, same = torch.randn(3, 4, device=device), self.same_as_eager
        for missing in (True, False):
            same(graph_reads_what_falls_back, lambda: (x, GraphFallsBack(missing)))
            same(call, lambda: (GraphFallsBackModule(missing).to(device), x))

    def test_a_key_that_is_compared_in_python_is_compared_once(self, device):
        x = torch.randn(3, 4, device=device)
        # A dict and a set with a key of a class that has an `__eq__` and the
        # hash of the text that is looked up: looking the text up asks that
        # key, so it is no instruction of a path, which could run once more
        # where the path gives up behind it (the flag is another one then).
        made = []

        def make():
            made.append(None)
            key, flag = GraphCollides(), ((len(made) - 1) // 2) % 8 in (3, 5, 6)
            return x, {key: 1.0, "a": 2.0}, {key, "a"}, key, flag

        self.same_as_eager(graph_looks_up_a_key, make, calls=8)

    def test_a_key_that_comes_to_be_compared_in_python_is_compared_once(self, device):
        def keep(gm, input_types):
            return gm

        # The graph is found with a dict and a set of texts. Ones that then
        # hold a key of a class with an `__eq__` and the hash of the text
        # that is looked up ask that key: such a lookup is no instruction
        # that runs out of its place, where it would run once more on a call
        # that nothing was exported for (a tensor of another type).
        passes = (*DEFAULT, P.Graphs(), keep)
        fn = graph_reads_a_table
        compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
        x = torch.randn(3, 4, device=device)
        for _ in range(6):
            compiled(x, {"b": 1.0, "a": 2.0}, {"b", "a"})
        for other in (x, x.double(), x[0], x.long(), x):
            mine, theirs = GraphCollides(), GraphCollides()
            got, there = compiled(other, {mine: 1.0, "a": 2.0}, {mine, "a"})
            expected = fn(other, {theirs: 1.0, "a": 2.0}, {theirs, "a"})
            self.assertEqual(got, expected[0], atol=0, rtol=0)
            self.assertEqual((there, mine.log), (expected[1], theirs.log))

    def test_an_operator_that_a_class_answers_is_left_to_the_interpreter(self, device):
        x, same = torch.randn(3, 4, device=device), self.same_as_eager
        fn = graph_asks_for_operators
        # `if`, `==` (with NotImplemented and the other side asked), `<`,
        # `for` and `in` through `__getitem__`, `with`, an index, `int`,
        # `float`, a key of a dict, `+=` through `__add__`, `not`: no path
        # goes into these, and each is asked once.
        for n in (1, 2, 0):
            same(fn, lambda: (x, GraphAnswers(n), GraphAnswers(1)))

    def test_what_a_path_lets_go_of_ends_where_the_instructions_end_it(self, device):
        # A store replaces what was there, and a variable that is bound
        # anew lets go of what it held: where that was the last reference
        # to an object whose end runs Python code (a `__del__`, a callback
        # of a weak reference), the code runs where the instruction is,
        # and what the function reads behind it is what that code left.
        # A path does its stores where it lands, behind everything it
        # read: it is not for such a call.
        def keep(gm, input_types):
            return gm

        def plant(what):
            made = GraphEndWatched() if what in ("watched", "keyed") else GraphEnds()
            if what == "watched":
                GRAPH_END_REFS.append(weakref.ref(made, graph_end_callback))
            if what == "keyed":
                GRAPH_END_WEAK[made] = GraphEnds()
            return [made] if what == "inside" else made

        def empty():
            GRAPH_END_BOX.item, GRAPH_END_DICT["k"] = None, None
            GRAPH_END_LIST[0], GRAPH_END_MODULE.item = None, None
            del GRAPH_END_HELD[:], GRAPH_END_REFS[:], GRAPH_END_LOG[:]

        def calls(run, what, graded):
            empty()
            GRAPH_END_STATE[0] = 0.5
            found = []
            for at in range(12):
                planted = at in (7, 10)
                kept = plant(what) if planted and what == "kept" else None
                GRAPH_END_HELD.append(plant(what) if planted and what != "kept" else 0)
                if planted:
                    GRAPH_END_BOX.item = kept if what == "kept" else plant(what)
                    GRAPH_END_DICT["k"], GRAPH_END_LIST[0] = plant(what), plant(what)
                    GRAPH_END_MODULE.item = plant(what)
                torch.manual_seed(at)
                x = torch.randn(3, 4, device=device, requires_grad=graded)
                out = run(x, kept)
                out = out[0] if isinstance(out, tuple) else out
                if graded:
                    out.backward()
                found.append((out.item(), len(GRAPH_END_LOG), GRAPH_END_STATE[0]))
                del kept
            return found

        stores = (graph_end_attribute, graph_end_dict_item, graph_end_list_item)
        for fn in (*stores, graph_end_module_attribute, graph_end_variable):
            # An object with a `__del__`; one inside a list that goes with
            # it; one with a weak reference that is told of its end; one
            # that is a key of a weak dictionary whose value has a
            # `__del__`; and one that an argument of the call keeps alive
            # (a path is for that call: nothing ends).
            for what in ("ends", "inside", "watched", "keyed", "kept"):
                for behind, graded in (((), False), ((keep,), False), ((), True)):
                    passes = (*DEFAULT, P.Graphs(), *behind)
                    compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
                    got = calls(compiled, what, graded)
                    expected = calls(fn, what, graded)
                    name = f"{fn.__name__} {what} {behind} {graded}"
                    self.assertEqual(got, expected, msg=name)
        empty()

    def test_a_path_lets_go_of_what_holds_what_stays(self, device):
        # What a path lets go of may be the last reference to an object
        # that holds others which stay alive: a list that a slice makes of
        # a list of layers, a bound method of one of them. Its end runs no
        # Python code, though it refers to more than is looked through for
        # a finalizer: what it holds has other references and stays. A
        # path is for such a call. (A slice of an `nn.ModuleList` is a
        # module of its own; one that is made after `torch.compile` was
        # used has a weak reference that dynamo is told by.)
        def keep(gm, input_types):
            return gm

        layers = [torch.nn.Linear(4, 4) for _ in range(8)]
        fn = graph_end_variable
        for make in (lambda: layers[2:8], lambda: layers[0].forward):
            for behind in ((), (keep,)):
                passes = (*DEFAULT, P.Graphs(), *behind)
                compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
                found = []
                for run in (compiled, fn):
                    del GRAPH_END_HELD[:], GRAPH_END_LOG[:]
                    GRAPH_END_STATE[0] = 0.5
                    given = []
                    for at in range(12):
                        GRAPH_END_HELD.append(make())
                        torch.manual_seed(at)
                        x = torch.randn(3, 4, device=device)
                        given.append((run(x)[0], len(GRAPH_END_LOG)))
                    found.append(given)
                self.assertEqual(found[0], found[1], atol=0, rtol=0)
                if behind:
                    (graph,) = [g for g in torch.vm.regions(compiled) if "ops" in g]
                    self.assertGreaterEqual(graph["hits"], 8)
        del GRAPH_END_HELD[:]

    def test_what_a_path_lets_go_of_is_looked_at_at_every_call(self, device):
        # Whether a path may let go of an object is decided at every call,
        # on the object as it is then: nothing of an earlier call is
        # remembered. An object that was let go of quietly, or stayed
        # alive, for sixteen calls, and then: its class has a `__del__`;
        # the list has an item with one and the caller no longer keeps
        # it; a weak reference with a callback was made to it and the
        # caller no longer keeps it; the caller no longer keeps an object
        # with a `__del__`. From there on the instructions run and the
        # end is where they have it. An object with a `__del__` that the
        # caller keeps throughout is let go of by the path at every call.
        def keep(gm, input_types):
            return gm

        def ender(self):
            GRAPH_END_LOG.append("the end of an object")
            GRAPH_END_STATE[0] += 1.0

        def calls(run, what, graded, boxed):
            GRAPH_END_BOX.item = None
            del GRAPH_END_HELD[:], GRAPH_END_REFS[:], GRAPH_END_LOG[:]
            GRAPH_END_STATE[0] = 0.5
            later = type("GraphEndsLater", (), {})
            first = {"class": type(None), "item": list, "weak": GraphEndWatched}
            same = first.get(what, GraphEnds)()
            if what == "item":
                same.append(GraphEndWatched())
            fresh = {"item": list, "weak": GraphEndWatched, "count": GraphEnds}
            found = []
            for at in range(24):
                if what == "class":
                    if at == 16:
                        later.__del__ = ender
                    given = later()
                elif what == "kept" or at < 16:
                    given = same
                else:
                    # At the sixteenth call the object of all the calls
                    # before, then a new one at each.
                    given = same if at == 16 else fresh[what]()
                    same = None
                    if what == "item":
                        given.append(GraphEnds())
                    if what == "weak":
                        GRAPH_END_REFS.append(weakref.ref(given, graph_end_callback))
                if boxed:
                    GRAPH_END_BOX.item = given
                else:
                    GRAPH_END_HELD.append(given)
                del given
                torch.manual_seed(at)
                x = torch.randn(3, 4, device=device, requires_grad=graded)
                out = run(x)
                out = out[0] if isinstance(out, tuple) else out
                if graded:
                    out.backward()
                found.append((out.item(), len(GRAPH_END_LOG), GRAPH_END_STATE[0]))
                del GRAPH_END_HELD[:]
            del same
            GRAPH_END_BOX.item = None
            return found

        for fn in (graph_end_variable, graph_end_attribute):
            boxed = fn is graph_end_attribute
            for what in ("class", "item", "weak", "count", "kept"):
                for behind, graded in (((), False), ((keep,), False), ((), True)):
                    passes = (*DEFAULT, P.Graphs(), *behind)
                    compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
                    msg = f"{fn.__name__} {what} {behind} {graded}"
                    got = calls(compiled, what, graded, boxed)
                    expected = calls(fn, what, graded, boxed)
                    self.assertEqual(got, expected, msg=msg)
                    if behind:
                        # The calls before the change are the path's, and
                        # so are all where nothing changes.
                        ran = graph_runs_to_the_return(compiled)
                        least, most = (20, 24) if what == "kept" else (12, 16)
                        self.assertGreaterEqual(ran, least, msg=msg)
                        self.assertLessEqual(ran, most, msg=msg)
        del GRAPH_END_HELD[:], GRAPH_END_REFS[:], GRAPH_END_LOG[:]
        GRAPH_END_STATE[0] = 0.5

    def test_what_a_tensor_holds_out_of_sight_ends_in_its_place(self, device):
        # A tensor holds more than it shows: the hooks on it are held by
        # its history as well, and the history holds what functions
        # written in Python kept for their backward. A tensor that
        # requires a gradient is therefore let go of by a path only where
        # it stays alive, and what a tensor shows (the dict of its hooks,
        # its attributes) is looked into whatever its count of references
        # says.
        def keep(gm, input_types):
            return gm

        def plant(what):
            if what == "hooked":
                made = torch.ones(2, device=device, requires_grad=True) * 2
                made.register_hook(lambda grad, held=GraphEnds(): grad)
            elif what == "function":
                leaf = torch.ones(2, device=device, requires_grad=True)
                made = GraphEndFunction.apply(leaf)
            else:
                made = torch.ones(2, device=device)
                made.held = GraphEnds()
            return made

        def calls(run, what, kept, graded, boxed):
            GRAPH_END_BOX.item = None
            del GRAPH_END_HELD[:], GRAPH_END_LOG[:]
            GRAPH_END_STATE[0] = 0.5
            found = []
            for at in range(12):
                made = plant(what) if at in (7, 10) else 0
                if boxed:
                    GRAPH_END_BOX.item = made
                else:
                    GRAPH_END_HELD.append(made)
                if not kept:
                    del made
                torch.manual_seed(at)
                x = torch.randn(3, 4, device=device, requires_grad=graded)
                out = run(x)
                out = out[0] if isinstance(out, tuple) else out
                if graded:
                    out.backward()
                found.append((out.item(), len(GRAPH_END_LOG), GRAPH_END_STATE[0]))
                del GRAPH_END_HELD[:]
                made = None
            GRAPH_END_BOX.item = None
            return found

        def check(fn, what, kept, behind, graded):
            passes = (*DEFAULT, P.Graphs(), *behind)
            compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
            msg = f"{fn.__name__} {what} kept {kept} {behind} {graded}"
            boxed = fn is graph_end_attribute
            got = calls(compiled, what, kept, graded, boxed)
            expected = calls(fn, what, kept, graded, boxed)
            self.assertEqual(got, expected, msg=msg)
            if kept and behind:
                # The caller keeps the tensor: every call is the path's.
                ran = graph_runs_to_the_return(compiled)
                self.assertGreaterEqual(ran, 6, msg=msg)

        plain = (graph_end_attribute, graph_end_variable, graph_end_deleted)
        for fn in plain:
            for what in ("hooked", "function", "attribute"):
                for kept in (False, True):
                    for behind, graded in (((), False), ((keep,), False), ((), True)):
                        check(fn, what, kept, behind, graded)
        del GRAPH_END_HELD[:], GRAPH_END_LOG[:]
        GRAPH_END_STATE[0] = 0.5

    def test_a_call_that_takes_a_path_down_ends_objects_in_their_place(self, device):
        # A call in which a path is taken down keeps what the path reads
        # until the run is described. An object that the function lets
        # go of in such a call (a variable that is deleted, which the
        # check that it is bound has read; a tuple the function made and
        # dropped) ends where the instructions end it all the same:
        # where its end may run Python code, the path is not taken down
        # in that call. Another object ends outside the function at the
        # same calls and changes what the forms were made for, so that
        # those are calls no form is for.
        def keep(gm, input_types):
            return gm

        def calls(run, planted, graded):
            del GRAPH_END_HELD[:], GRAPH_END_LOG[:]
            GRAPH_END_STATE[0] = 0.5
            elsewhere, found = None, []
            for at in range(14):
                GRAPH_END_HELD.append(GraphEnds() if at in planted else 0)
                if at in planted:
                    elsewhere = GraphEnds()
                torch.manual_seed(at)
                x = torch.randn(3, 4, device=device, requires_grad=graded)
                out = run(x)
                if graded:
                    out.backward()
                found.append((out.item(), len(GRAPH_END_LOG), GRAPH_END_STATE[0]))
                del GRAPH_END_HELD[:]
            del elsewhere
            return found

        for fn in (graph_end_deleted, graph_end_in_no_variable):
            for planted in ((7, 10), (7, 8), (5, 7, 10), (7, 10, 12, 13)):
                for behind, graded in (((), False), ((keep,), False), ((), True)):
                    passes = (*DEFAULT, P.Graphs(), *behind)
                    compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
                    msg = f"{fn.__name__} {planted} {behind} {graded}"
                    got = calls(compiled, planted, graded)
                    expected = calls(fn, planted, graded)
                    self.assertEqual(got, expected, msg=msg)
        del GRAPH_END_HELD[:], GRAPH_END_LOG[:]
        GRAPH_END_STATE[0] = 0.5

    def test_a_callback_that_is_not_a_weak_containers_own_is_turned_down(self, device):
        # A weak reference with a callback is no reason against letting
        # go of an object where the callback is a weak container's own,
        # which takes the entry out and does nothing else. That is known
        # by what the callback is (the very code of the container's
        # class), by the exact class of the container and of the
        # reference, and by the exact types of what the callback reaches
        # of the container: not by names. A dictionary of the standard
        # library and the one by identity of `torch._dynamo.utils` pass,
        # and the path runs; a subclass whose attribute is a property, a
        # subclass with another method to take the entry out, an object
        # with such a method of its own, a class written line for line
        # like the original, and the original with a dict of another
        # class for its values all run code of the user's when the
        # object ends: the instructions run, and it ends in its place.
        from torch._dynamo.utils import ExactWeakKeyDictionary

        def keep(gm, input_types):
            return gm

        class TakesOut(ExactWeakKeyDictionary):
            def _remove_id(self, idx):
                graph_weak_told()
                super()._remove_id(idx)

        def own_method(made):
            # It takes the entry out as the class does (an entry that
            # stayed would keep a later object at the same place from
            # being noted), and runs code of the user's.
            def remove(idx):
                graph_weak_told()
                ExactWeakKeyDictionary._remove_id(made, idx)

            return remove

        classes = {"standard": weakref.WeakKeyDictionary, "asks": GraphWeakAsks}
        classes.update({"subclass": TakesOut, "alike": GraphWeakTakesOut})

        def container(what):
            made = classes.get(what, ExactWeakKeyDictionary)()
            if what == "own method":
                made._remove_id = own_method(made)
            if what == "own values":
                made.values = GraphWeakDict()
            return made

        def calls(run, what, graded, boxed):
            GRAPH_END_BOX.item = None
            del GRAPH_END_HELD[:], GRAPH_END_LOG[:]
            GRAPH_END_STATE[0] = 0.5
            noted, found = container(what), []
            for at in range(12):
                made = 0
                if at in (7, 10):
                    made = GraphEndWatched()
                    noted[made] = at
                if boxed:
                    GRAPH_END_BOX.item = made
                else:
                    GRAPH_END_HELD.append(made)
                del made
                torch.manual_seed(at)
                x = torch.randn(3, 4, device=device, requires_grad=graded)
                out = run(x)
                out = out[0] if isinstance(out, tuple) else out
                if graded:
                    out.backward()
                entries = noted.data if what in ("standard", "asks") else noted.values
                ended = (len(GRAPH_END_LOG), GRAPH_END_STATE[0])
                found.append((out.item(), *ended, len(entries)))
                del GRAPH_END_HELD[:]
            GRAPH_END_BOX.item = None
            return found

        kinds = ("standard", "identity", "asks", "subclass", "own method", "alike")
        for fn in (graph_end_variable, graph_end_attribute, graph_end_deleted):
            boxed = fn is graph_end_attribute
            for what in (*kinds, "own values"):
                for behind, graded in (((), False), ((keep,), False), ((), True)):
                    passes = (*DEFAULT, P.Graphs(), *behind)
                    compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
                    msg = f"{fn.__name__} {what} {behind} {graded}"
                    got = calls(compiled, what, graded, boxed)
                    expected = calls(fn, what, graded, boxed)
                    self.assertEqual(got, expected, msg=msg)
                    if behind and what in ("standard", "identity"):
                        # Every call behind the first two is the path's,
                        # those in which the object ends included.
                        listed = torch.vm.regions(compiled)
                        graphs = [g for g in listed if "ops" in g]
                        ran = sum(g["hits"] for g in graphs if "within" not in g)
                        self.assertGreaterEqual(ran, 10, msg=msg)
        del GRAPH_END_HELD[:], GRAPH_END_LOG[:]
        GRAPH_END_STATE[0] = 0.5

    def test_what_owns_the_memory_of_a_tensor_ends_in_its_place(self, device):
        # A tensor that requires no gradient can still keep an object
        # alive that it does not show: what owns its memory (a numpy array
        # over a buffer, the buffer of `torch.frombuffer`, the tensor a
        # DLPack capsule was made of), or its `.grad`. And an object that
        # shows nothing at all can (the capsule itself). A path lets go
        # of such a tensor only where it stays alive: its memory must be
        # torch's own allocation, its `.grad` is looked at like what it
        # refers to, and an object of a class that shows nothing must be
        # of one that is known to hold nothing.
        import numpy

        def keep(gm, input_types):
            return gm

        def plant(what):
            if what in ("numpy", "array"):
                buffer = GraphEndBytes(b"\0" * 16)
                array = numpy.frombuffer(buffer, dtype=numpy.float32)
                return array if what == "array" else torch.from_numpy(array)
            if what == "buffer":
                return torch.frombuffer(GraphEndBytes(b"\0" * 16), dtype=torch.float32)
            if what in ("capsule", "dlpack"):
                owned = torch.ones(2).as_subclass(GraphEndTensor)
                if what == "capsule":
                    return torch.utils.dlpack.to_dlpack(owned)
                return torch.from_dlpack(owned)
            loudly = torch.ones(2, requires_grad=True) * 2
            loudly.register_hook(lambda grad, held=GraphEnds(): grad)
            made = torch.ones(2)
            made.grad = loudly
            return made

        def calls(run, what, kept, graded, boxed):
            GRAPH_END_BOX.item = None
            del GRAPH_END_HELD[:], GRAPH_END_LOG[:]
            GRAPH_END_STATE[0] = 0.5
            found = []
            for at in range(12):
                made = plant(what) if at in (7, 10) else torch.ones(2)
                if boxed:
                    GRAPH_END_BOX.item = made
                else:
                    GRAPH_END_HELD.append(made)
                if not kept:
                    del made
                torch.manual_seed(at)
                x = torch.randn(3, 4, device=device, requires_grad=graded)
                out = run(x)
                out = out[0] if isinstance(out, tuple) else out
                if graded:
                    out.backward()
                found.append((out.item(), len(GRAPH_END_LOG), GRAPH_END_STATE[0]))
                del GRAPH_END_HELD[:]
                made = None
            GRAPH_END_BOX.item = None
            return found

        def check(fn, what, kept, behind, graded):
            passes = (*DEFAULT, P.Graphs(), *behind)
            compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
            msg = f"{fn.__name__} {what} kept {kept} {behind} {graded}"
            boxed = fn is graph_end_attribute
            got = calls(compiled, what, kept, graded, boxed)
            expected = calls(fn, what, kept, graded, boxed)
            self.assertEqual(got, expected, msg=msg)
            if behind:
                # A tensor of torch's own memory is let go of by the path
                # at every other call, and one that the caller keeps at
                # every call.
                listed = torch.vm.regions(compiled)
                graphs = [g for g in listed if "ops" in g and "within" not in g]
                ran = sum(g["hits"] for g in graphs)
                self.assertGreaterEqual(ran, 7 if kept else 4, msg=msg)

        for fn in (graph_end_attribute, graph_end_variable):
            for what in ("numpy", "buffer", "capsule", "dlpack", "array", "grad"):
                for kept in (False, True):
                    for behind, graded in (((), False), ((keep,), False), ((), True)):
                        check(fn, what, kept, behind, graded)
        del GRAPH_END_HELD[:], GRAPH_END_LOG[:]
        GRAPH_END_STATE[0] = 0.5

    def test_every_reference_that_goes_with_a_weak_entry_is_counted(self, device):
        # A path lets go of the last reference to a value of a
        # `weakref.WeakValueDictionary`. The entry goes with it, and the
        # entry holds its key twice: as the key of the dict and as the
        # `key` of the weak reference, which ends with the entry. A key
        # that nothing else holds ends there, and where it has a `__del__`
        # (itself, or an item of a tuple that is the key) that runs where
        # the instructions let go of the value. Taking the entry out also
        # hashes the key, and compares it with a key the dict keeps under
        # the same hash: code of the user's where the class of the key
        # says so. Every call is eager's: the result, what was logged,
        # the state that the ends change, how many entries are left.
        def keep(gm, input_types):
            return gm

        def heard(told):
            GraphKeyHashed.heard = GraphKeyCompared.heard = told

        def key(what, at):
            if what in ("an end", "a tuple"):
                return GraphEnds() if what == "an end" else (1, GraphEnds())
            if what in ("hashed", "a tuple hashed"):
                return GraphKeyHashed() if what == "hashed" else (1, GraphKeyHashed())
            return GraphKeyCompared(at) if what == "compared" else "k"

        def calls(run, what, graded, boxed):
            GRAPH_END_BOX.item = None
            del GRAPH_END_HELD[:], GRAPH_END_LOG[:]
            GRAPH_END_STATE[0] = 0.5
            heard(False)
            noted, found, other = weakref.WeakValueDictionary(), [], GraphEndWatched()
            for at in range(12):
                made = 0
                if at in (7, 10):
                    made = GraphEndWatched()
                    if what == "compared":
                        noted[GraphKeyCompared(-at)] = other
                    noted[key(what, at)] = made
                    heard(True)
                if boxed:
                    GRAPH_END_BOX.item = made
                else:
                    GRAPH_END_HELD.append(made)
                del made
                torch.manual_seed(at)
                x = torch.randn(3, 4, device=device, requires_grad=graded)
                out = run(x)
                out = out[0] if isinstance(out, tuple) else out
                if graded:
                    out.backward()
                heard(False)
                ended = (list(GRAPH_END_LOG), GRAPH_END_STATE[0])
                found.append((out.item(), *ended, len(noted.data)))
                del GRAPH_END_HELD[:]
            GRAPH_END_BOX.item = None
            return found

        kinds = ("a text", "an end", "a tuple", "hashed", "a tuple hashed")
        for fn in (graph_end_variable, graph_end_attribute, graph_end_deleted):
            boxed = fn is graph_end_attribute
            for what in (*kinds, "compared"):
                for behind, graded in (((), False), ((keep,), False), ((), True)):
                    passes = (*DEFAULT, P.Graphs(), *behind)
                    compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
                    msg = f"{fn.__name__} {what} {behind} {graded}"
                    got = calls(compiled, what, graded, boxed)
                    expected = calls(fn, what, graded, boxed)
                    self.assertEqual(got, expected, msg=msg)
                    if behind and what == "a text":
                        # Every call behind the first two is the path's,
                        # those in which the value ends included.
                        listed = torch.vm.regions(compiled)
                        graphs = [g for g in listed if "ops" in g]
                        ran = sum(g["hits"] for g in graphs if "within" not in g)
                        self.assertGreaterEqual(ran, 10, msg=msg)
        del GRAPH_END_HELD[:], GRAPH_END_LOG[:]
        GRAPH_END_STATE[0] = 0.5

    def test_an_instruction_that_nothing_uses_stays_where_it_can_fail(self, device):
        # An instruction of a path that gives what nothing uses is left out
        # of a run only if it cannot fail: an attribute, an item, a global
        # or a variable of an outer function that is read for nothing, a
        # division, an unpacking and a `len` show that the call gets as
        # far as the path does. What they read is there for
        # ten calls, gone for ten (eager raises, or takes the other way),
        # and back for ten; every call is eager's: the result, or what is
        # raised.
        def keep(gm, input_types):
            return gm

        def one(fn, obj, x):
            try:
                return fn(obj, x)
            except Exception as error:
                return type(error).__name__

        def there(obj):
            obj.extra, obj.table, obj.row, obj.number = [1], {"extra": 1}, [1, 2, 3], 1
            globals()["GRAPH_UNUSED_NAME"] = 1
            GRAPH_FREE[2]()

        def gone(obj):
            del obj.extra, obj.table["extra"], obj.row[1:]
            del globals()["GRAPH_UNUSED_NAME"]
            GRAPH_FREE[1]()

        def nothing(obj):
            obj.extra, obj.number = 0, 0

        cases = [
            (graph_unused_attribute, GraphUnusedPlain, gone, "AttributeError"),
            (graph_unused_item, GraphUnusedPlain, gone, "KeyError"),
            (graph_unused_place, GraphUnusedPlain, gone, "IndexError"),
            (graph_unused_pair, GraphUnusedPlain, nothing, "ValueError"),
            (graph_unused_quotient, GraphUnusedPlain, nothing, "ZeroDivisionError"),
            (graph_unused_count, GraphUnusedPlain, nothing, "TypeError"),
            (graph_unused_global, GraphUnusedPlain, gone, "NameError"),
            (graph_unused_free, GraphUnusedPlain, gone, "NameError"),
        ]
        x = torch.randn(3, 4, device=device)
        try:
            for behind in ((), (keep,)):
                for fn, cls, change, raised in cases:
                    obj = cls()
                    there(obj)
                    passes = (*DEFAULT, P.Graphs(), *behind)
                    compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
                    msg = f"{fn.__name__}, {cls.__name__}, with a pass: {bool(behind)}"
                    for at in range(30):
                        if at == 10:
                            change(obj)
                        if at == 20:
                            there(obj)
                        got, expected = one(compiled, obj, x), one(fn, obj, x)
                        said = f"{msg}, call {at}: {got}, {expected}"
                        self.assertEqual(type(got), type(expected), msg=said)
                        self.assertEqual(got, expected, atol=0, rtol=0, msg=said)
                        if raised is not None:
                            absent = 10 <= at < 20
                            self.assertEqual(expected == raised, absent, msg=said)
                    if behind:
                        # What was made of the graph ran while everything
                        # was there (and again later, unless an instruction
                        # that could not be shown to run no Python code
                        # had the graph given up meanwhile).
                        listed = torch.vm.regions(compiled)
                        graphs = [g for g in listed if "ops" in g and "within" not in g]
                        ran = sum(g["hits"] for g in graphs)
                        self.assertGreaterEqual(ran, 6, msg=f"{msg}: {graphs}")
            # What is read for nothing comes to run Python code (the class
            # gets a property under the name): it runs at every call, where
            # the instructions run it.
            fn = graph_unused_attribute
            for behind in ((), (keep,)):
                obj = GraphUnusedPlain()
                there(obj)
                passes = (*DEFAULT, P.Graphs(), *behind)
                compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
                seen = []
                for run in (compiled, fn):
                    del GRAPH_UNUSED_LOG[:]
                    obj.extra = [1]
                    given = []
                    for at in range(24):
                        if at == 12:
                            del obj.extra
                            GraphUnusedPlain.extra = property(graph_unused_told)
                        given.append((run(obj, x), list(GRAPH_UNUSED_LOG)))
                    del GraphUnusedPlain.extra
                    seen.append(given)
                for got, expected in zip(*seen):
                    self.assertEqual(got[1], expected[1])
                    self.assertEqual(got[0], expected[0], atol=0, rtol=0)
        finally:
            globals()["GRAPH_UNUSED_NAME"] = 1
            GRAPH_FREE[2]()
            if "extra" in vars(GraphUnusedPlain):
                del GraphUnusedPlain.extra
            del GRAPH_UNUSED_LOG[:]

    def test_a_function_a_path_went_into_binds_what_it_is_not_handed(self, device):
        # The defaults of a function that a path went into are read at
        # every call, and how many there are is checked: with fewer, the
        # default of the last parameter is that of another one and a
        # parameter has none, for which eager raises; a default by its
        # name that is taken away is missed although the function never
        # uses it. Every call is eager's: the result, or what is raised.
        def keep(gm, input_types):
            return gm

        def one(fn, x):
            try:
                return fn(x)
            except Exception as error:
                return type(error).__name__

        def put(defaults):
            graph_defaulted.__defaults__ = defaults[0]
            graph_defaulted_by_name.__kwdefaults__ = defaults[1]

        x = torch.randn(3, 4, device=device)
        whole = ((1.0, 2.0), {"a": 1.0, "b": 2.0})
        fewer, other = ((1.0,), {"a": 1.0}), ((3.0, 4.0), {"a": 3.0, "b": 4.0})
        schedule = [whole, fewer, other, whole]
        try:
            for behind in ((), (keep,)):
                for fn in (graph_default_moves, graph_default_by_name_goes):
                    put(whole)
                    passes = (*DEFAULT, P.Graphs(), *behind)
                    compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
                    raised = 0
                    for at in range(40):
                        put(schedule[at // 10])
                        got, expected = one(compiled, x), one(fn, x)
                        said = f"{fn.__name__}, call {at}: {got}, {expected}"
                        self.assertEqual(type(got), type(expected), msg=said)
                        self.assertEqual(got, expected, atol=0, rtol=0, msg=said)
                        raised += expected == "TypeError"
                    self.assertEqual(raised, 10)
                    if behind:
                        listed = torch.vm.regions(compiled)
                        graphs = [g for g in listed if "ops" in g and "within" not in g]
                        ran = sum(g["hits"] for g in graphs)
                        self.assertGreaterEqual(ran, 15, msg=str(graphs))
        finally:
            put(whole)

    def test_what_a_tensor_keeps_alive_through_its_base_ends_in_its_place(self, device):
        # A view keeps the tensor it is a view of alive from C++, with the
        # Python object of that tensor and what its attributes hold; a
        # storage keeps its Python object and that one's attributes; a
        # dual tensor its tangent. None of it is what a tensor shows. A
        # path lets go of such a tensor only where what it keeps that way
        # is looked at, or stays alive: every call is eager's.
        import torch.autograd.forward_ad as forward_ad

        def keep(gm, input_types):
            return gm

        def plant(what):
            base = torch.ones(4, 4)
            if what == "storage":
                made = torch.ones(2)
                made.untyped_storage().held = GraphEnds()
                return made
            base.held = GraphEnds()
            if what == "view":
                return base[:2]
            if what == "transposed":
                return base.T
            if what == "unbound":
                return base.unbind()[0]
            return (torch.ones(2), base[:2])

        def dual():
            tangent = torch.ones(2)
            tangent.held = GraphEnds()
            return forward_ad.make_dual(torch.ones(2), tangent)

        def calls(run, make, graded, boxed):
            GRAPH_END_BOX.item = None
            del GRAPH_END_HELD[:], GRAPH_END_LOG[:]
            GRAPH_END_STATE[0] = 0.5
            found = []
            for at in range(12):
                made = make() if at in (7, 10) else torch.ones(2)
                if boxed:
                    GRAPH_END_BOX.item = made
                else:
                    GRAPH_END_HELD.append(made)
                del made
                torch.manual_seed(at)
                x = torch.randn(3, 4, device=device, requires_grad=graded)
                out = run(x)
                out = out[0] if isinstance(out, tuple) else out
                if graded:
                    out.backward()
                found.append((out.item(), len(GRAPH_END_LOG), GRAPH_END_STATE[0]))
                del GRAPH_END_HELD[:]
            GRAPH_END_BOX.item = None
            return found

        def check(fn, what, behind, graded):
            passes = (*DEFAULT, P.Graphs(), *behind)
            compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
            msg = f"{fn.__name__} {what} {behind} {graded}"
            boxed = fn is graph_end_attribute
            if what == "a dual tensor":
                # A tangent that holds an object, of a dual tensor that the
                # path lets go of inside the level.
                with forward_ad.dual_level():
                    got = calls(compiled, dual, graded, boxed)
                    expected = calls(fn, dual, graded, boxed)
                self.assertEqual(got, expected, msg=msg)
                return
            got = calls(compiled, lambda: plant(what), graded, boxed)
            expected = calls(fn, lambda: plant(what), graded, boxed)
            self.assertEqual(got, expected, msg=msg)
            if behind:
                listed = torch.vm.regions(compiled)
                graphs = [g for g in listed if "ops" in g and "within" not in g]
                self.assertGreaterEqual(sum(g["hits"] for g in graphs), 4, msg=msg)

        kinds = ("view", "transposed", "unbound", "in a tuple", "storage")
        for fn in (graph_end_attribute, graph_end_variable):
            for what in kinds:
                for behind, graded in (((), False), ((keep,), False), ((), True)):
                    check(fn, what, behind, graded)
            for behind in ((), (keep,)):
                check(fn, "a dual tensor", behind, False)
        del GRAPH_END_HELD[:], GRAPH_END_LOG[:]
        GRAPH_END_STATE[0] = 0.5

    def test_what_watches_the_storage_of_a_tensor_is_told_in_its_place(self, device):
        # C++ keeps the Python object of a storage for the storage, and
        # it ends with it: with the tensor, where nothing else has the
        # storage. A weak reference on that object with a function to
        # tell, or a `weakref.finalize`, runs Python code there, as for
        # any object that ends. A path lets go of such a tensor (a
        # tensor, a view of one whose storage is watched) only where the
        # instructions do: every call is eager's, and what the function
        # reads behind the place is what it reads in eager mode.
        def keep(gm, input_types):
            return gm

        def told(ref=None):
            GRAPH_END_LOG.append("the storage ended")
            GRAPH_END_STATE[0] += 1.0

        def plant(what):
            made = torch.ones(4, 4)
            storage = made.untyped_storage()
            if what in ("a reference", "a view, a reference"):
                GRAPH_END_REFS.append(weakref.ref(storage, told))
            elif what in ("a finalizer", "a view, a finalizer"):
                weakref.finalize(storage, told)
            elif what == "an attribute that ends quietly":
                storage.note = [1, 2]
            return made[:2] if what.startswith("a view") else made

        def calls(run, what, graded, boxed):
            GRAPH_END_BOX.item = None
            del GRAPH_END_HELD[:], GRAPH_END_LOG[:], GRAPH_END_REFS[:]
            GRAPH_END_STATE[0] = 0.5
            found = []
            for at in range(12):
                made = plant(what) if at in (7, 10) else torch.ones(2)
                if boxed:
                    GRAPH_END_BOX.item = made
                else:
                    GRAPH_END_HELD.append(made)
                del made
                x = torch.full((3, 4), 0.25 * (at + 1), device=device)
                x.requires_grad_(graded)
                out = run(x)
                out = out[0] if isinstance(out, tuple) else out
                if graded:
                    out.backward()
                found.append((out.item(), len(GRAPH_END_LOG), GRAPH_END_STATE[0]))
                del GRAPH_END_HELD[:]
            GRAPH_END_BOX.item = None
            return found

        kinds = ("a reference", "a finalizer", "a view, a reference")
        kinds += ("a view, a finalizer", "an attribute that ends quietly")

        def check(fn, what, behind, graded):
            boxed = fn is graph_end_attribute
            passes = (*DEFAULT, P.Graphs(), *behind)
            compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
            msg = f"{fn.__name__}, {what}, a pass: {bool(behind)}, a gradient: {graded}"
            got = calls(compiled, what, graded, boxed)
            expected = calls(fn, what, graded, boxed)
            self.assertTrue(got == expected, msg=f"{msg}: {got} {expected}")
            if behind:
                listed = torch.vm.regions(compiled)
                graphs = [g for g in listed if "ops" in g and "within" not in g]
                # What ends quietly is no reason to turn a path down.
                least = 9 if what == kinds[-1] else 4
                hits = sum(g["hits"] for g in graphs)
                self.assertGreaterEqual(hits, least, msg=f"{msg}: {graphs}")

        try:
            for fn in (graph_end_attribute, graph_end_variable):
                for what in kinds:
                    for behind, graded in (((), False), ((keep,), False), ((), True)):
                        check(fn, what, behind, graded)
        finally:
            del GRAPH_END_HELD[:], GRAPH_END_LOG[:], GRAPH_END_REFS[:]
            GRAPH_END_STATE[0] = 0.5

    def graph_change_calls(self, run, graded, device, makes=()):
        # Twelve calls from the same state: what each gave or raised, and
        # what it left outside (as plain values; of an iterator, what is
        # left). The calls are given what the first of `makes` makes, the
        # eighth and the eleventh what the second makes.
        def plain(value):
            if isinstance(value, (list, tuple)):
                return [plain(item) for item in value]
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, GraphChangeThing):
                return plain(vars(value))
            if type(value).__name__.endswith("iterator"):
                return ["left", *value]
            if isinstance(value, torch.Tensor):
                return (value.tolist(), value.requires_grad, value.grad_fn is None)
            return value

        def one():
            return 1.0

        def two():
            return 2.0

        usual, other = makes or (one, two)
        GRAPH_CHANGE_BOX.items = None
        del GRAPH_CHANGE_SEEN[:]
        GRAPH_CHANGE_TABLE.clear()
        GRAPH_CHANGE_TABLE["row"] = []
        found = []
        for at in range(12):
            given = other() if at in (7, 10) else usual()
            x = torch.full((3, 4), 0.25 * (at + 1), device=device)
            x.requires_grad_(graded)
            try:
                out = run(x, given)
                rest = plain(out[1:]) if isinstance(out, tuple) else []
                out = out[0] if isinstance(out, tuple) else out
                if graded:
                    out.backward()
                gave = (out.item(), rest)
            except (TypeError, IndexError) as error:
                gave = type(error).__name__
            box, seen = plain(GRAPH_CHANGE_BOX.items), plain(GRAPH_CHANGE_SEEN)
            found.append((gave, box, seen, plain(GRAPH_CHANGE_TABLE)))
        return found

    def graph_change_as_eager(self, fns, device, makes=()):
        def keep(gm, input_types):
            return gm

        try:
            for fn in fns:
                for behind, graded in (((), False), ((keep,), False), ((), True)):
                    passes = (*DEFAULT, P.Graphs(), *behind)
                    compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
                    got = self.graph_change_calls(compiled, graded, device, makes)
                    expected = self.graph_change_calls(fn, graded, device, makes)
                    pairs = enumerate(zip(got, expected))
                    differ = [(at, a, b) for at, (a, b) in pairs if a != b]
                    msg = f"{fn.__name__}, a pass: {bool(behind)}, a gradient: {graded}"
                    self.assertEqual(differ[:1], [], msg=msg)
                    if behind:
                        listed = torch.vm.regions(compiled)
                        graphs = [g for g in listed if "ops" in g and "within" not in g]
                        hits = sum(g["hits"] for g in graphs)
                        self.assertGreaterEqual(hits, 6, msg=f"{msg}: {graphs}")
        finally:
            GRAPH_CHANGE_BOX.items = None
            del GRAPH_CHANGE_SEEN[:]
            GRAPH_CHANGE_TABLE.clear()

    def test_a_change_to_what_a_path_made_runs_where_the_thing_was_taken(self, device):
        # What changes something the function made between its operations
        # (an append to a list, a store into an object of a class it
        # called, the next item of an iterator) is seen by whatever took
        # the thing, in front of the change or behind it: a list that was
        # stored into an object, put into a list or a dict from outside,
        # or into another list, a tuple or an object the function made
        # that was, has the item (a number, a size, what the graph
        # gives), and an iterator whose last item is asked for has given
        # every item in front of it. A path leaves such a change out of a
        # run only together with the thing.
        fns = [
            graph_change_appends_behind_a_store,
            graph_change_appends_behind_a_list_from_outside,
            graph_change_appends_behind_a_dict_from_outside,
            graph_change_appends_to_an_item_of_a_list,
            graph_change_appends_to_an_item_of_a_tuple,
            graph_change_appends_to_what_an_object_holds,
            graph_change_appends_around_a_store,
            graph_change_appends_a_size,
            graph_change_appends_a_result,
            graph_change_stores_behind_a_store,
            graph_change_stores_behind_a_list_from_outside,
            graph_change_ends_an_iterator,
            graph_change_goes_over_a_tuple,
        ]
        self.graph_change_as_eager(fns, device)

    def test_an_extend_that_waits_takes_what_the_list_holds_at_its_place(self, device):
        # A store into an object from outside waits for the path to land;
        # what changes a list the path made is done where it stands. Of
        # all stores only `extend` reads what it is given, so of a list
        # the path made it takes the items the list holds at its place,
        # not those an append behind it adds. A list from outside is
        # changed by stores only, which are done in their order.
        fns = [
            graph_waits_for_a_list_given_back,
            graph_waits_for_a_list_that_is_stored,
            graph_waits_for_a_list_nothing_else_takes,
            graph_waits_for_a_list_made_of_a_tuple,
            graph_waits_for_a_list_from_outside,
        ]
        self.graph_change_as_eager(fns, device)

    def test_a_method_read_off_a_list_the_path_made_keeps_nothing_back(self, device):
        # A loop that fills a list of its own reads `out.append` off the
        # list each time round. The read is of a method of the builtin
        # class, off a list the path made: it cannot fail and nothing sees
        # it, so it is left out where nothing uses what it gives (the
        # append is the path's own instruction). With it kept, it would
        # run behind the graph, the appends with it, and the graph would
        # have to give every item of the list back as a result.
        graphs = []

        def keep(gm, input_types):
            graphs.append(gm)
            return gm

        passes = (*DEFAULT, P.Graphs(), keep)
        for fn in (graph_collects,):
            del graphs[:]
            compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
            xs = [torch.randn(3, device=device) for _ in range(3)]
            for _ in range(6):
                self.assertEqual(compiled(xs), fn(xs), atol=0, rtol=0)
            listed = torch.vm.regions(compiled)
            (found,) = [g for g in listed if "ops" in g and "within" not in g]
            self.assertGreaterEqual(found["hits"], 3, msg=str(found))
            nodes = graphs[0].graph.nodes
            (given,) = [n.args[0] for n in nodes if n.op == "output"]
            self.assertEqual(len(given), 1, msg=f"{fn.__name__}: {graphs[0].code}")

    def test_a_store_into_what_may_be_a_list_of_the_path_ends_the_path(self, device):
        # A store into a list from outside waits for the path to land,
        # and an append to a list the path made is done where it stands.
        # A list that is reached through another value (what a dict of
        # the path gives, the default of a `get`, what a method of the
        # list is bound to) may be the list the path made, at this call
        # or at a later one, and then the order of the two appends is the
        # function's only if both are done in place: such a store is none
        # of a path, and neither is an `extend` with such a list.
        def keyed():
            GRAPH_CHANGE_TABLE["k"] = GRAPH_CHANGE_TABLE["row"]
            return 1.0

        def bare():
            GRAPH_CHANGE_TABLE.pop("k", None)
            return 2.0

        fns = [graph_other_way_through_a_dict, graph_other_way_through_a_method]
        self.graph_change_as_eager(fns, device)
        fns = [graph_other_way_through_a_default, graph_other_way_extends]
        for usual, other in ((bare, bare), (keyed, bare), (bare, keyed)):
            self.graph_change_as_eager(fns, device, makes=(usual, other))

    def test_a_path_goes_over_no_iterator_but_its_own(self, device):
        # What goes over an iterator uses it up. A value that is a list
        # when the path is taken down may be an iterator of the caller's
        # at a later call: the instructions of the path do not go over
        # it (a path that then turned out not to be for the call would
        # leave the instructions an iterator that is used up, and a `zip`
        # over a generator runs the generator). Every call is eager's:
        # the result or what is raised, and what the generator logged.
        def row():
            return [1.0, 2.0]

        def three():
            return iter([1.0, 2.0, 3.0])

        def two():
            return iter([1.0, 2.0])

        def zipped():
            return zip(graph_gives_loudly())

        def generated():
            return (b for a, b in enumerate(graph_gives_loudly()))

        fns = [graph_goes_over, graph_goes_over_two, graph_goes_over_under_tuple]
        for other in (three, two, zipped, generated):
            self.graph_change_as_eager(fns, device, makes=(row, other))

    def graph_thing_cases(self, device, group, debug=False):
        # The cases of a group of `GRAPH_THINGS` in a process of their own:
        # one that does not come back within the time limit, or that ends
        # the process, is a failure and not a suite that stands still,
        # and the failure says which of the two it was (the process
        # writes no core, so that a crash is seen at once).
        import json
        import queue

        env = dict(os.environ, TORCH_VM_GRAPH_DEBUG="2") if debug else None
        command = [sys.executable, "-c", GRAPH_THINGS, device, group]
        pipes = {"stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL, "text": True}
        ran = subprocess.Popen(command, env=env, **pipes)
        lines = queue.Queue()

        def read():
            for line in ran.stdout:
                lines.put(line)
            lines.put(None)

        threading.Thread(target=read, daemon=True).start()
        started, found = None, {}
        while True:
            try:
                line = lines.get(timeout=GRAPH_CASE_SECONDS)
            except queue.Empty:
                ran.kill()
                ran.wait()
                limit = GRAPH_CASE_SECONDS
                self.fail(f"{started} did not come back within {limit} seconds")
            if line is None:
                break
            if line.startswith("start "):
                started = line.split()[1]
            elif line.startswith("done "):
                name, said = line[5:].split(" ", 1)
                found[name] = json.loads(said)
        code = ran.wait()
        how = f"signal {-code}" if code < 0 else f"exit {code}"
        self.assertEqual(code, 0, msg=f"{started} ended the process ({how})")
        self.assertIn(started, found, msg=f"{started} did not finish")
        for name, ways in found.items():
            for way, said in ways.items():
                msg = f"{name}, {way}: {said['first']}"
                self.assertEqual(said["differ"], 0, msg=msg)
        return found

    def test_an_operation_is_handed_what_a_list_held_where_it_was_spread(self, device):
        # `[*made]` and `(*made,)` take the items the list holds where the
        # instruction stands. An operation that such a copy is spread
        # into is handed those, whatever is appended to the list behind
        # the copy: `clamp(h, 0.5)`, not `clamp(h, 0.5, given)`.
        def keep(gm, input_types):
            return gm

        for fn in (graph_spread_list, graph_spread_tuple, graph_spread_twice):
            for behind, graded in (((), False), ((keep,), False), ((), True)):
                passes = (*DEFAULT, P.Graphs(), *behind)
                compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
                msg = f"{fn.__name__}, a pass: {bool(behind)}, a gradient: {graded}"
                for at in range(12):
                    x = torch.full((3, 4), 0.25 * (at + 1), device=device)
                    given = 0.75 if at not in (7, 10) else 1.25
                    grads = []
                    for run in (compiled, fn):
                        leaf = x.clone().requires_grad_(graded)
                        out = run(leaf, given)
                        if graded:
                            out.backward()
                        grads.append((out.detach(), leaf.grad))
                    where = f"{msg}, call {at}"
                    self.assertEqual(grads[0], grads[1], atol=0, rtol=0, msg=where)
                if behind:
                    found = torch.vm.regions(compiled)
                    graphs = [g for g in found if "ops" in g and "within" not in g]
                    ran = sum(g["hits"] for g in graphs)
                    self.assertGreaterEqual(ran, 6, msg=f"{msg}: {graphs}")

    def test_things_a_path_made_that_hold_each_other(self, device):
        # Lists that hold each other, a list in itself, a list and a
        # dict, a tuple closed through a list, objects of a class the
        # function calls, a set through an object, cells, a bound method
        # stored in what it is bound to, a generator that holds the list
        # it fills: made between the operations of a function, given
        # back, stored outside, dropped, read through, gone over,
        # spread, handed to an operation. A path makes them by the
        # instructions of the function in their order, and nothing that
        # looks into what a list or a tuple holds goes round one that
        # holds itself: every call is eager's, with `Graphs()` alone,
        # with a pass and with a gradient, and the graph around such a
        # thing runs.
        found = self.graph_thing_cases(device, "made")
        self.assertEqual(len(found), 24)
        runs = (
            "two_lists_given_back",
            "two_lists_stored",
            "two_lists_dropped",
            "a_list_in_itself",
            "a_list_in_itself_given_back",
            "a_list_in_itself_many_times",
            "a_list_and_a_dict",
            "a_tuple_through_a_list",
            "an_object_in_itself",
            "a_bound_method_in_its_list",
            "a_cycle_read_through",
            "a_cycle_with_a_result",
            "a_cycle_spread",
            "a_cycle_spread_into_a_tuple",
            "a_cycle_asked_which_it_is",
            "a_cycle_extended_outside",
        )
        for name in runs:
            for way in ("pass",):
                graphs = found[name][way]["graphs"]
                ran = sum(hits for _, hits in graphs)
                self.assertGreaterEqual(ran, 6, msg=f"{name}, {way}: {graphs}")
        # What is printed of such things ends as well.
        printed = self.graph_thing_cases(device, "printed", debug=True)
        self.assertEqual(len(printed), 3)

    def test_a_list_from_outside_that_holds_itself_ends_every_walk(self, device):
        # A list that holds itself, handed to a function: read, counted,
        # stored, turned into text, handed to an operation (which raises
        # what eager raises), let go of. Such a list is no value a form
        # is checked against or has as a constant, is not handed to an
        # operation of a graph and is not turned into text on a path.
        found = self.graph_thing_cases(device, "outside")
        self.assertEqual(len(found), 9)
        runs = (
            "outside_read",
            "outside_counted",
            "outside_stored",
        )
        for name in runs:
            for way in ("pass",):
                graphs = found[name][way]["graphs"]
                ran = sum(hits for _, hits in graphs)
                self.assertGreaterEqual(ran, 6, msg=f"{name}, {way}: {graphs}")

    def test_reads_and_changes_of_a_thing_keep_the_order_of_the_function(self, device):
        # A thing the function makes (a list, a dict, an object, a cell,
        # an iterator) runs in front of what stands for the graph until
        # an instruction that takes it has to run behind: a result of
        # the graph is put into it, or what reads it takes a size of the
        # graph's as well. From that instruction on everything that
        # takes the thing, or what may be it or hold it, runs behind the
        # graph; everything in front of it ran in front. So a holder
        # that was built of the thing before, an alias, a length or a
        # slice that was taken before are there for what uses them in
        # front of the graph, and a change that stands behind a read in
        # the function is not seen by the read. Every call is eager's,
        # and the graph runs.
        found = self.graph_thing_cases(device, "sides")
        self.assertEqual(len(found), 23)
        for name, ways in found.items():
            for way in ("pass",):
                graphs = ways[way]["graphs"]
                ran = sum(hits for _, hits in graphs)
                self.assertGreaterEqual(ran, 6, msg=f"{name}, {way}: {graphs}")

    def test_what_a_path_takes_out_of_outside_is_from_outside(self, device):
        # What an iterator that the path made gives is something of the
        # path's only where what it goes over may be or hold one: `for
        # i, layer in enumerate(layers)` over a container from outside
        # gives the layers, which are from outside. The same for an item
        # or an attribute of an object from outside, a default, a
        # variable of an outer function, an item of a list the path
        # built of such objects, a value of the `**kwargs` of a function
        # the path went into. A store into such an object is an effect
        # of the path, which waits, and no reason to end it: the graph
        # goes to the function's return, and what the stores leave is
        # eager's.
        found = self.graph_thing_cases(device, "own")
        self.assertEqual(len(found), 8)
        for name, ways in found.items():
            for way in ("pass",):
                graphs = ways[way]["graphs"]
                whole = sum(hits for ops, hits in graphs if ops >= 7)
                self.assertGreaterEqual(whole, 6, msg=f"{name}, {way}: {graphs}")

    def test_a_form_runs_only_under_the_checks_of_its_path(self, device):
        # Two calls that take the same instructions are one path only if
        # they were taken down with the same pinned values: the place of
        # an item that is taken out of a list or a tuple the path built,
        # a number or a name that a check made a constant, the rate of
        # an attention, a function or a class that something behind the
        # graph is computed with, how many defaults a function has. A
        # call with another one is a path of its own, with its own
        # checks, and what was made of its graph is never tried for a
        # call of the first. The pinned thing changes at the eighth and
        # the eleventh call and changes back: every call is eager's.
        def keep(gm, input_types):
            return gm

        cases = (
            (graph_pins_a_place, 0, 1),
            (graph_pins_a_place_of_a_tuple, 0, 2),
            (graph_pins_a_number, 2, 3),
            (graph_pins_a_rate, 0.0, 0.5),
            (graph_pins_names, "scale", "shift"),
            (graph_pins_a_function, max, min),
            (graph_pins_a_class, tuple, list),
            (graph_pins_defaults, (2.0, 1.0), (7.0, 3.0, 1.5)),
        )

        def calls(run, usual, other, graded):
            found = []
            for at in range(14):
                torch.manual_seed(100 + at)
                x = torch.full((3, 4), 0.25 * (at + 1), device=device)
                x.requires_grad_(graded)
                out = run(x, other if at in (7, 10) else usual)
                if graded:
                    out.backward()
                grad = None if x.grad is None else x.grad.sum().item()
                found.append((out.item(), grad))
            return found

        try:
            for fn, usual, other in cases:
                for behind, graded in (((), False), ((keep,), False), ((), True)):
                    passes = (*DEFAULT, P.Graphs(), *behind)
                    compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
                    msg = f"{fn.__name__}, a pass: {bool(behind)}, a gradient: {graded}"
                    got = calls(compiled, usual, other, graded)
                    expected = calls(fn, usual, other, graded)
                    self.assertTrue(got == expected, msg=f"{msg}: {got} {expected}")
                    if behind:
                        found = torch.vm.regions(compiled)
                        graphs = [g for g in found if "ops" in g and "within" not in g]
                        ran = sum(g["hits"] for g in graphs)
                        self.assertGreaterEqual(ran, 6, msg=f"{msg}: {graphs}")
            # Ten places, three times round: no more forms than the graph may
            # have, and the calls that none is for run the instructions.
            passes = (*DEFAULT, P.Graphs(), keep)
            ten = torch.vm(graph_pins_ten, passes=passes, fallback=False, demote=False)
            for at in range(30):
                x = torch.full((3, 4), 0.25 * (at + 1), device=device)
                got, expected = ten(x, at % 10), graph_pins_ten(x, at % 10)
                self.assertEqual(got, expected, atol=0, rtol=0)
            found = torch.vm.regions(ten)
            (graph,) = [g for g in found if "ops" in g and "within" not in g]
            self.assertLessEqual(len(graph["forms"]), 8, msg=str(graph))
            self.assertGreaterEqual(graph["hits"], 8, msg=str(graph))
        finally:
            graph_pinned_scaled.__defaults__ = (2.0, 1.0)

    def test_a_change_that_can_fail_is_not_left_out_with_what_it_changes(self, device):
        # What changes something the function made is left out of a run
        # together with the thing where nothing uses it, but only if it
        # cannot fail. An entry that is put into a dict under a key that
        # is no constant can: the key may be one that cannot be hashed.
        # So it runs in every run, the call in which it fails is not the
        # path's, and the instructions run and raise. The same for a
        # list or a dict that is filled from a value from outside
        # (`[*xs]`, `{**d}`) which cannot be gone over at some calls.
        def keep(gm, input_types):
            return gm

        cases = (
            (graph_key_dict_of_a_comprehension, lambda: 1, lambda: [1]),
            (graph_key_list_of_what_is_spread, lambda: (1, 2), lambda: 5),
            (graph_key_dict_of_what_is_spread, lambda: {"a": 1}, lambda: 5),
        )

        def calls(run, usual, other, graded):
            found = []
            for at in range(14):
                x = torch.full((3, 4), 0.25 * (at + 1), device=device)
                x.requires_grad_(graded)
                given = other() if at in (7, 10) else usual()
                try:
                    out = run(x, given)
                    if graded:
                        out.backward()
                    grad = None if x.grad is None else x.grad.sum().item()
                    found.append((out.item(), grad))
                except (TypeError, KeyError) as error:
                    found.append(f"{type(error).__name__}: {error}")
            return found

        for fn, usual, other in cases:
            for behind, graded in (((), False), ((keep,), False)):
                passes = (*DEFAULT, P.Graphs(), *behind)
                compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
                msg = f"{fn.__name__} with {usual()!r} and {other()!r}"
                msg += f", a pass: {bool(behind)}, a gradient: {graded}"
                got = calls(compiled, usual, other, graded)
                expected = calls(fn, usual, other, graded)
                self.assertTrue(got == expected, msg=f"{msg}: {got} {expected}")
                if behind or graded:
                    # With the usual value the graph runs.
                    found = torch.vm.regions(compiled)
                    ran = sum(g["hits"] for g in found if "ops" in g)
                    self.assertGreaterEqual(ran, 4, msg=f"{msg}: {found}")

    def test_the_code_of_a_key_runs_as_often_as_the_instructions_run_it(self, device):
        # Nothing of the engine touches a key of another class than the
        # exact ones of the interpreter (a number, a text, None, a tuple
        # of such). A key whose class has a `__hash__` or an `__eq__`
        # written in Python is hashed and compared by the instruction
        # that stores or reads under it, as often as without the engine:
        # what the class logs is what it logs there, also in the calls
        # in which a path is taken down.
        def keep(gm, input_types):
            return gm

        def logs(key):
            # A text of a class whose `__hash__` and `__eq__` log, new at
            # every call.
            return functools.partial(GraphKeyText, key)

        spread = graph_key_dict_of_what_is_spread
        cases = (
            (graph_key_stored_and_read, lambda: "k", logs("k")),
            (graph_key_stored_and_read, logs("k"), logs("k")),
            (graph_key_spread, logs("scale"), logs("shift")),
            (spread, lambda: {"a": 1}, lambda: {GraphKeyText("a"): 1}),
        )

        def calls(run, usual, other, graded):
            found = []
            for at in range(14):
                del GRAPH_KEY_LOG[:]
                x = torch.full((3, 4), 0.25 * (at + 1), device=device)
                x.requires_grad_(graded)
                given = other() if at in (7, 10) else usual()
                try:
                    out = run(x, given)
                    if graded:
                        out.backward()
                    grad = None if x.grad is None else x.grad.sum().item()
                    said = (out.item(), grad)
                except (TypeError, KeyError, ValueError) as error:
                    said = f"{type(error).__name__}: {error}"
                found.append((said, list(GRAPH_KEY_LOG)))
            return found

        try:
            for fn, usual, other in cases:
                for behind, graded in (((), False), ((keep,), False)):
                    passes = (*DEFAULT, P.Graphs(), *behind)
                    compiled = torch.vm(fn, passes=passes, fallback=False, demote=False)
                    msg = f"{fn.__name__} with {usual()!r} and {other()!r}"
                    msg += f", a pass: {bool(behind)}, a gradient: {graded}"
                    got = calls(compiled, usual, other, graded)
                    expected = calls(fn, usual, other, graded)
                    self.assertTrue(got == expected, msg=f"{msg}: {got} {expected}")
                    plain = not isinstance(usual(), (GraphKeyText, int))
                    if (behind or graded) and plain:
                        # With the usual key the graph runs.
                        found = torch.vm.regions(compiled)
                        ran = sum(g["hits"] for g in found if "ops" in g)
                        self.assertGreaterEqual(ran, 4, msg=f"{msg}: {found}")
        finally:
            del GRAPH_KEY_LOG[:]

    def test_a_graph_is_within_a_callers_only_where_its_path_goes_through(self, device):
        # What `regions` and `stats` say. A graph of a function is part
        # of a graph of a caller (`within`, and not counted) where a path
        # of the caller goes through the function to the first operation
        # of the graph. A path that went into the function, ended there
        # and was cut back in front of the call holds none of it, even
        # where the first operation is one of a module that the caller
        # calls too: the graph of the function is one of its own, which
        # runs at every call and is counted.
        def keep(gm, input_types):
            return gm

        models = ((GraphWithinModel, False), (GraphWithinModel, True))
        for model_class, loud in models:
            for graded in (False,):
                model = model_class(loud).to(device)
                passes = (*DEFAULT, P.Graphs(), keep)
                compiled = torch.vm(model, passes=passes, fallback=False, demote=False)
                msg = f"{model_class.__name__}, loud: {loud}, a gradient: {graded}"
                for at in range(12):
                    x = torch.full((3, 4), 0.25 * (at + 1), device=device)
                    x.requires_grad_(graded)
                    self.assertEqual(compiled(x), model(x), msg=msg)
                regions = [g for g in torch.vm.regions(compiled) if "ops" in g]
                said = [(g["where"], g["ops"], g["hits"]) for g in regions]
                parts = [g.get("within") for g in regions]
                msg = f"{msg}: {said}, within {parts}"

                def of(name):
                    start = f"{name}.forward "
                    found = [g for g in regions if g["where"].startswith(start)]
                    return [g for g in found if g["ops"] >= 5]

                outer, inner = of(model_class.__name__), of("GraphWithinInner")
                self.assertEqual(len(outer), 1, msg=msg)
                self.assertGreaterEqual(outer[0]["hits"], 6, msg=msg)
                self.assertNotIn("within", outer[0], msg=msg)
                built = compiled.stats()["graphs"]["built"]
                if loud:
                    self.assertEqual(len(inner), 1, msg=msg)
                    self.assertNotIn("within", inner[0], msg=msg)
                    self.assertGreaterEqual(inner[0]["hits"], 6, msg=msg)
                    self.assertEqual(built, 2, msg=msg)
                else:
                    # The path of the model goes through the function: what
                    # there is of a graph of it is part of that one.
                    self.assertEqual(outer[0]["ops"], 12, msg=msg)
                    within = [g.get("within") for g in inner]
                    self.assertEqual(within, [outer[0]["where"]] * len(inner), msg=msg)
                    self.assertEqual(built, 1, msg=msg)


instantiate_device_type_tests(TestVmGraphResults, globals())


@unittest.skipUnless(HAS_RUNTIME, "torch.vm runtime not built")
class TestVmApi(TestCase):
    """How torch.vm is asked for things."""

    hw_classification = HardwareClassification.GENERIC

    def test_the_decorator_with_and_without_arguments(self):
        @torch.vm
        def bare(x, y):
            return unr_loop(x, y)[0] * 2

        @torch.vm(passes=torch.vm.DEFAULT_PASSES)
        def spelled(x, y):
            return unr_loop(x, y)[0] * 2

        called = torch.vm(vm_scaled, fallback=False)
        x, y = torch.randn(3), torch.randn(3)
        for fn in (bare, spelled, called):
            for _ in range(3):
                self.assertEqual(fn(x, y), vm_scaled(x, y))
            # The defaults: the call is inlined, and the inlined loop unrolled.
            self.assertEqual(len(fn.stats()["inlined"]), 1)
            self.assertEqual(fn.stats()["unrolled_loops"], 1)

    def test_an_empty_list_changes_nothing(self):
        plain = torch.vm(vm_scaled, passes=(), fallback=False)
        x, y = torch.randn(3), torch.randn(3)
        self.assertEqual(plain(x, y), vm_scaled(x, y))
        self.assertEqual(plain.stats()["inlined"], [])
        self.assertEqual(plain.stats()["unrolled_loops"], 0)
        specialized = [b["specialized_executions"] for b in plain.stats()["blocks"]]
        self.assertEqual(sum(specialized), 0)

    def test_unroll_goes_behind_inline(self):
        with self.assertRaisesRegex(ValueError, "the engine inlines first"):
            torch.vm(lambda x: x, passes=(P.Unroll(), P.Inline()))

    def test_unroll_takes_the_number_of_iterations(self):
        x = torch.zeros(2)
        usual = torch.vm(vm_twelve, fallback=False)
        self.assertEqual(usual(x), x + 12)
        self.assertEqual(usual.stats()["unrolled_loops"], 0)
        for most, unrolled in ((11, 0), (12, 1), (16, 1)):
            passes = (P.Inline(), P.Unroll(most))
            fn = torch.vm(vm_twelve, passes=passes, fallback=False)
            self.assertEqual(fn(x), x + 12)
            self.assertEqual(fn.stats()["unrolled_loops"], unrolled)
        # A loop that comes in with an inlined call is unrolled as well.
        passes = (P.Inline(), P.Unroll(12))
        outer = torch.vm(vm_thirteen, passes=passes, fallback=False)
        self.assertEqual(outer(x), x + 13)
        self.assertEqual(len(outer.stats()["inlined"]), 1)
        self.assertEqual(outer.stats()["unrolled_loops"], 1)
        said = "torch.vm.passes.Unroll(max_trip_count=16)"
        self.assertEqual(repr(P.Unroll(16)), said)
        self.assertEqual(P.Unroll(8), P.Unroll())
        with self.assertRaisesRegex(ValueError, "at least one iteration"):
            P.Unroll(0)
        with self.assertRaisesRegex(TypeError, "number of iterations"):
            P.Unroll("8")

    def test_the_default_is_exact_and_a_pass_of_ones_own_is_not(self):
        self.assertTrue(torch.vm.is_exact())
        for one in DEFAULT:
            self.assertIs(one.exact, True, repr(one))
        # The passes of the engine that are no default keep a result as well.
        unrolling = [
            P.Unroll(64) if isinstance(p, P.Unroll) else p for p in narrowing()
        ]
        self.assertTrue(torch.vm.is_exact(unrolling))
        self.assertTrue(torch.vm.is_exact(()))
        # A pass of one's own is not taken to.
        self.assertFalse(torch.vm.is_exact([*DEFAULT, lambda region: False]))

    def test_the_default_is_exact_and_what_is_not_is_asked_for(self):
        self.assertTrue(torch.vm.is_exact())
        for one in DEFAULT:
            self.assertIs(one.exact, True, repr(one))
        # The passes of the engine that are no default keep a result as well.
        unrolling = [
            P.Unroll(64) if isinstance(p, P.Unroll) else p for p in narrowing()
        ]
        self.assertTrue(torch.vm.is_exact(unrolling))
        self.assertTrue(torch.vm.is_exact(()))
        # A graph compiler does not, and a pass of one's own is not taken to.
        self.assertFalse(torch.vm.is_exact(with_graphs("inductor")))
        self.assertFalse(torch.vm.is_exact(with_graphs("aot_eager")))
        self.assertFalse(torch.vm.is_exact([*DEFAULT, lambda region: False]))

    def test_a_module_is_called_with_its_hooks(self):
        model = torch.nn.Linear(3, 3)
        seen = []
        model.register_forward_pre_hook(lambda module, args: seen.append("before"))
        model.register_forward_hook(lambda module, args, output: output * 2)
        compiled = torch.vm(model, fallback=False)
        x = torch.randn(2, 3)
        for _ in range(4):
            self.assertEqual(compiled(x), model(x), atol=0, rtol=0)
        self.assertEqual(seen.count("before"), 8)


if __name__ == "__main__":
    run_tests()
