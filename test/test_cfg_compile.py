# Owner(s): ["module: dynamo"]

import collections
import copy
import dis
import gc
import importlib
import math
import os
import random
import signal
import sys
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
    TEST_WITH_CROSSREF,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)
from torch.utils._python_dispatch import TorchDispatchMode


# These tests look at which kernels the dispatcher runs; the crossref and
# dynamo test configurations interpose on exactly that.
UNDISTURBED = not (TEST_WITH_CROSSREF or TEST_WITH_TORCHDYNAMO)

# The Rust runtime is only in builds made with USE_CFG_COMPILE=1, and only
# understands CPython 3.12 bytecode.
try:
    cfg_rs = importlib.import_module("torch._cfg_rs")
except ImportError:
    cfg_rs = None
# The tests are about the VM: nothing is left to the interpreter for being
# faster there, unless a test asks for it.
os.environ["TORCH_CFG_COMPILE_DEMOTE"] = "0"

HAS_RUNTIME = UNDISTURBED and cfg_rs is not None and sys.version_info[:2] == (3, 12)
Pipeline = torch.cfg_compile.Pipeline


def call_outcome(fn, *args, **kwargs):
    """What a call returns, or the type and the message of what it raises."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return type(e), str(e)


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
    """The dispatcher hook that cfg_compile records kernels with."""

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


@unittest.skipUnless(HAS_RUNTIME, "cfg_compile runtime not built")
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

        compiled = torch.cfg_compile(fn, fallback=False)
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

        text = torch.cfg_compile(f, fallback=False).dump()
        self.assertIn("phi", text)
        self.assertIn("br_exhausted", text)
        self.assertIn("binary - ", text)
        self.assertIn("binary + ", text)

    def test_arguments(self):
        def f(a, b=2, *args, c=3, **kw):
            return a, b, args, c, sorted(kw.items())

        compiled = torch.cfg_compile(f, fallback=False)
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

        compiled = torch.cfg_compile(fact, fallback=False)
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

        compiled = torch.cfg_compile(f, fallback=False)
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
        compiled = torch.cfg_compile(namespace["checked"], fallback=False)
        self.assertEqual(compiled(2), 2)
        with self.assertRaisesRegex(AssertionError, "not one"):
            compiled(1)

    def test_traceback_names_the_line(self):
        def f(x, y):
            z = x + 1
            return z @ y

        compiled = torch.cfg_compile(f, fallback=False)
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

        with self.assertRaisesRegex(torch.cfg_compile.Unsupported, "generators"):
            torch.cfg_compile(f, fallback=False)
        self.assertTrue(torch.cfg_compile.is_available())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIs(torch.cfg_compile(f), f)
        self.assertIn(
            "left f uncompiled", str(caught[0].message).replace(f.__qualname__, "f")
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
        for compiled in (torch.cfg_compile(m), torch.cfg_compile(m.forward)):
            for _ in range(3):
                self.assertEqual(compiled(x), m(x))


instantiate_parametrized_tests(TestCfgFrontend)


@unittest.skipUnless(HAS_RUNTIME, "cfg_compile runtime not built")
class TestCfgCompile(TestCase):
    """Profiling on real tensors and direct kernel calls."""

    hw_classification = HardwareClassification.ACCELERATOR

    def assertCompiledEqual(self, fn, argsets, **kwargs):
        compiled = torch.cfg_compile(fn, fallback=False, **kwargs)
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

        compiled = torch.cfg_compile(f, fallback=False)
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

        compiled = torch.cfg_compile(f, fallback=False)
        x = torch.randn(3, device=device)
        compiled(x, x)
        compiled(x, x.clone().requires_grad_())
        compiled(x, x)
        (site,) = compiled.sites()
        self.assertEqual([v["entry_level"] for v in site["variants"]], [1, 0])
        self.assertEqual((site["hits"], site["misses"]), (1, 1))

        strict = torch.cfg_compile(f, fallback=False, skip_autograd=False)
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

        compiled = torch.cfg_compile(f, fallback=False)
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

        compiled = torch.cfg_compile(leaf, fallback=False)
        x = torch.randn(3, device=device, requires_grad=True)
        for _ in range(3):
            with self.assertRaisesRegex(RuntimeError, "leaf Variable"):
                compiled(x)

    def test_sites_are_polymorphic(self, device):
        def f(x, y):
            return (x * y + 2).sum()

        compiled = torch.cfg_compile(f, fallback=False)
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

        compiled = torch.cfg_compile(f, fallback=False)
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

        compiled = torch.cfg_compile(f, fallback=False)
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

        compiled = torch.cfg_compile(f, fallback=False)
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

        compiled = torch.cfg_compile(f, fallback=False)
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

        boxed = torch.cfg_compile(f, fallback=False, unboxed=False)
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

        compiled = torch.cfg_compile(f, fallback=False)
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

        compiled = torch.cfg_compile(f, fallback=False)
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

        compiled = torch.cfg_compile(f, fallback=False)
        x = torch.ones(2, device=device)
        self.assertEqual(compiled(x), x * 2)
        # The function ran once, on the real input: no separate tracing run.
        self.assertEqual(len(calls), 1)
        self.assertEqual(block_stats(compiled, "versions"), [1])


instantiate_device_type_tests(TestCfgCompile, globals())


@unittest.skipUnless(HAS_RUNTIME, "cfg_compile runtime not built")
class TestCfgPasses(TestCase):
    """The pass pipeline, guards, and recompilation of single blocks."""

    hw_classification = HardwareClassification.GENERIC

    def test_function_attributes_are_read_through(self):
        @unittest.expectedFailure
        def f(x, k=2):
            return x * k

        f.note = "set by a decorator"
        compiled = torch.cfg_compile(f, fallback=False)
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

        compiled = torch.cfg_compile(rank, fallback=False)
        for shape in ((2, 3), (2, 3), (4,), (2, 3, 4)):
            self.assertEqual(compiled(torch.ones(shape)), len(shape) + 1)
        self.assertIn("tensor_dim", compiled.dump())

    def test_pipeline_descriptor(self):
        default = Pipeline.default()
        self.assertNotIn("narrow_dims", default.passes)
        narrowing = Pipeline.narrowing()
        self.assertEqual(
            narrowing.passes.index("narrow_dims") + 1, narrowing.passes.index("fold")
        )
        self.assertEqual(default.remove("dce").append("dce").passes, default.passes)
        with self.assertRaisesRegex(ValueError, "unknown pass 'nope'"):
            torch.cfg_compile(lambda x: x, pipeline=Pipeline(["nope"]))
        with self.assertRaises(TypeError):
            Pipeline([3])

    def test_generic_tier_only(self):
        def f(x, n):
            for _ in range(n):
                x = x * 2 + 1
            return x

        compiled = torch.cfg_compile(f, specialize=False, fallback=False)
        for _ in range(3):
            self.assertEqual(compiled(torch.ones(2), 3), torch.full((2,), 15.0))
        self.assertEqual(sum(block_stats(compiled, "versions")), 0)
        self.assertEqual(sum(block_stats(compiled, "profile_runs")), 0)
        self.assertNotIn("specialized", compiled.dump())

    def test_without_passes_nothing_is_specialized(self):
        def f(x):
            return x + 1

        compiled = torch.cfg_compile(f, pipeline=Pipeline([]), fallback=False)
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

        compiled = torch.cfg_compile(f, pipeline=Pipeline.narrowing(), fallback=False)

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

        compiled = torch.cfg_compile(f, pipeline=Pipeline.narrowing(), fallback=False)
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

        pipeline = Pipeline.default().insert_after("tensor_accessors", pin_first_dim)
        compiled = torch.cfg_compile(f, pipeline=pipeline, fallback=False)
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

        compiled = torch.cfg_compile(f, pipeline=Pipeline([replace]), fallback=False)
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
        # Four operators and the constant one of them takes.
        self.assertEqual(len(run.replay.text(run.replay.start, run.replay.end)), 5)

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

        compiled = torch.cfg_compile(f, pipeline=Pipeline([grab]), fallback=False)
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

    def test_settle_specializes_the_block_again(self):
        seen = []

        def note_taker(region):
            seen.append(region.note)
            if region.note is None:
                region.note = [region.replayer(0, len(region.instructions()))]
            return False

        def f(x):
            return x * 2

        compiled = torch.cfg_compile(
            f,
            pipeline=Pipeline([note_taker, *Pipeline.default().passes]),
            fallback=False,
        )
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

        compiled = torch.cfg_compile(outer, fallback=False, inline=False, callees=True)
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

        compiled = torch.cfg_compile(
            lambda x: x + 1, pipeline=Pipeline([keep]), fallback=False
        )
        compiled(torch.ones(1))
        with self.assertRaisesRegex(RuntimeError, "only valid while its pass runs"):
            kept[0].instructions()

    def test_failing_pass_leaves_the_block_generic(self):
        def broken(region):
            raise KeyError("oops")

        def f(x):
            return x * 3

        compiled = torch.cfg_compile(
            f, pipeline=Pipeline.default().append(broken), fallback=False
        )
        for _ in range(3):
            self.assertEqual(compiled(torch.ones(2)), torch.full((2,), 3.0))
        stats = compiled.stats()
        self.assertEqual(block_stats(compiled, "versions"), [0])
        self.assertIn("pass failed", stats["blocks"][0]["rejected"])


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


@unittest.skipUnless(HAS_RUNTIME, "cfg_compile runtime not built")
class TestCfgGraphTransforms(TestCase):
    """Inlining and loop unrolling, which rewrite the graph before it runs."""

    hw_classification = HardwareClassification.GENERIC

    def test_inlines_calls_of_module_functions(self):
        compiled = torch.cfg_compile(inl_outer, fallback=False)
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
        compiled = torch.cfg_compile(inl_outer, fallback=False)
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
        recursive = torch.cfg_compile(inl_fact, fallback=False)
        self.assertEqual(recursive(5), 120)
        self.assertEqual(recursive.stats()["inlined"], [])
        # Defaults and keywords stay calls.
        defaults = torch.cfg_compile(inl_uses_defaults, fallback=False)
        self.assertEqual(defaults(torch.ones(2)), torch.full((2,), 5.0))
        self.assertEqual(defaults.stats()["inlined"], [])
        off = torch.cfg_compile(inl_outer, fallback=False, inline=False)
        x, y = torch.randn(3), torch.randn(3)
        self.assertEqual(off(x, y), inl_outer(x, y))
        self.assertEqual(off.stats()["inlined"], [])

    def test_unrolls_constant_range_loops(self):
        compiled = torch.cfg_compile(unr_loop, fallback=False)
        x, y = torch.randn(3), torch.randn(3)
        for _ in range(3):
            total, i = compiled(x, y)
            expected, expected_i = unr_loop(x, y)
            self.assertEqual(total, expected)
            self.assertEqual(i, expected_i)
        self.assertEqual(compiled.stats()["unrolled_loops"], 1)
        off = torch.cfg_compile(unr_loop, fallback=False, unroll=False)
        self.assertEqual(off(x, y)[0], unr_loop(x, y)[0])
        self.assertEqual(off.stats()["unrolled_loops"], 0)

    def test_unrolled_loop_is_guarded_on_range(self):
        compiled = torch.cfg_compile(unr_loop, fallback=False)
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
        compiled = torch.cfg_compile(nested, fallback=False)
        self.assertEqual(compiled(x), nested(x))
        self.assertGreaterEqual(compiled.stats()["unrolled_loops"], 3)
        compiled = torch.cfg_compile(empty, fallback=False)
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
            compiled = torch.cfg_compile(fn, fallback=False)
            self.assertEqual(compiled(*args), fn(*args))
            self.assertEqual(compiled.stats()["unrolled_loops"], 0)

    def test_error_in_inlined_function_has_both_frames(self):
        compiled = torch.cfg_compile(inl_calls_raises, fallback=False)
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


@unittest.skipUnless(HAS_RUNTIME, "cfg_compile runtime not built")
class TestCfgCallees(TestCase):
    """Python functions called from compiled code are compiled in turn."""

    hw_classification = HardwareClassification.GENERIC

    def callees(self, compiled):
        stats = compiled.stats()
        return stats["compiled_callees"], stats["interpreted_callees"]

    def test_functions_of_other_modules_are_compiled(self):
        def f(x):
            return torch.nn.functional.dropout(x, p=0.0, training=False) + 1

        compiled = torch.cfg_compile(f, fallback=False)
        x = torch.randn(3)
        for _ in range(3):
            self.assertEqual(compiled(x), f(x))
        self.assertEqual(self.callees(compiled), (["dropout"], []))

    def test_methods_and_callable_objects(self):
        def f(obj, x):
            bound = obj.twice
            return obj.twice(x) + bound(x, bias=1) + obj(x)

        compiled = torch.cfg_compile(f, fallback=False)
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

        compiled = torch.cfg_compile(f, fallback=False)
        for args, kwargs in (((1,), {}), ((1, 5, 7, 8), {"scale": 3})):
            self.assertEqual(compiled(args, kwargs), f(args, kwargs))
        self.assertEqual(len(self.callees(compiled)[0]), 1)

    def test_unsupported_callee_runs_in_the_interpreter(self):
        def generator(n):
            yield from range(n)

        def f(n):
            return sum(generator(n))

        compiled = torch.cfg_compile(f, fallback=False)
        self.assertEqual([compiled(4), compiled(5)], [6, 10])
        compiled_names, interpreted = self.callees(compiled)
        self.assertEqual(compiled_names, [])
        self.assertEqual(len(interpreted), 1)

    def test_recursion_and_modules(self):
        compiled = torch.cfg_compile(callee_fact, fallback=False, inline=False)
        self.assertEqual(compiled(6), 720)
        self.assertEqual(self.callees(compiled)[0], ["callee_fact"])

        def run(module, x):
            return module.forward(x) + module(x)

        linear = torch.nn.Linear(3, 2)
        compiled = torch.cfg_compile(run, fallback=False)
        x = torch.randn(4, 3)
        for _ in range(3):
            self.assertEqual(compiled(linear, x), run(linear, x))
        self.assertIn("Linear.forward", self.callees(compiled)[0])

    def test_changed_function_is_compiled_again(self):
        def target(x, k=2):
            return x * k

        def f(x):
            return target(x)

        compiled = torch.cfg_compile(f, fallback=False)
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

        compiled = torch.cfg_compile(f, fallback=False)
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

        compiled = torch.cfg_compile(f, fallback=False)
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

        compiled = torch.cfg_compile(f, fallback=False)
        self.assertEqual(compiled(20), ([True] * 20, 0))
        self.assertEqual(
            [name.rsplit(".", 1)[-1] for name in self.callees(compiled)[0]],
            ["outer", "inner"],
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
            compiled = torch.cfg_compile(fn, fallback=False)
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
        compiled = torch.cfg_compile(count, fallback=False)
        self.assertLess(abs(compiled(0) - count(0)), 5)

    def test_recursion_deeper_than_the_native_stack(self):
        def depth(n):
            return 0 if n == 0 else 1 + depth(n - 1)

        compiled = torch.cfg_compile(depth, fallback=False)
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
        compiled = torch.cfg_compile(sig, fallback=False)
        compiled_caller = torch.cfg_compile(call, fallback=False)
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
            compiled = torch.cfg_compile(fn, fallback=False)
            for args in argsets:
                self.assertEqual(call_outcome(compiled, *args), call_outcome(fn, *args))
        message = call_outcome(torch.cfg_compile(both, fallback=False), [1], {"k": 5})[
            1
        ]
        self.assertTrue(
            message.endswith("target() got multiple values for keyword argument 'k'")
        )

    def test_callees_off(self):
        def f(obj, x):
            return obj.twice(x)

        compiled = torch.cfg_compile(f, fallback=False, callees=False)
        self.assertEqual(compiled(CalleeScale(2), 3), 12)
        self.assertEqual(self.callees(compiled), ([], []))

    def test_error_in_callee_has_both_frames(self):
        def f(x):
            return callee_raises(x)

        compiled = torch.cfg_compile(f, fallback=False, inline=False)
        try:
            compiled(torch.ones(2))
        except NameError:
            frames = traceback.extract_tb(sys.exc_info()[2])
        else:
            self.fail("expected NameError")
        self.assertEqual([frame.name for frame in frames[-2:]], ["f", "callee_raises"])


@unittest.skipUnless(HAS_RUNTIME, "cfg_compile runtime not built")
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

        compiled = torch.cfg_compile(counter, fallback=False)
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

        compiled = torch.cfg_compile(f, fallback=False)
        x, y = torch.randn(3), torch.randn(3)
        self.assertEqual(compiled(x, y), f(x, y))

    def test_lambdas_with_defaults_and_annotations(self):
        def adders(ks):
            return [(lambda v, k=k: v + k) for k in ks]

        def annotated():
            def inner(a: int, b: "str" = "z") -> float:
                return 1.0

            return inner

        compiled = torch.cfg_compile(adders, fallback=False)
        self.assertEqual([add(10) for add in compiled([1, 2, 3])], [11, 12, 13])
        compiled = torch.cfg_compile(annotated, fallback=False)
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
        compiled = torch.cfg_compile(add, fallback=False)
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
            compiled = torch.cfg_compile(fn, fallback=False)
            with self.assertRaises(error) as expected:
                fn()
            with self.assertRaises(error) as got:
                compiled()
            self.assertEqual(str(got.exception), str(expected.exception))


def exc_double(x):
    if x < 0:
        raise ValueError("negative")
    return x * 2


@unittest.skipUnless(HAS_RUNTIME, "cfg_compile runtime not built")
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
        compiled = torch.cfg_compile(fn, fallback=False, **options)
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

        compiled = torch.cfg_compile(f, fallback=False)
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

        compiled = torch.cfg_compile(f, fallback=False, inline=False, callees=False)
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


@unittest.skipUnless(HAS_RUNTIME, "cfg_compile runtime not built")
class TestCfgWith(TestCase):
    """with statements inside compiled functions."""

    hw_classification = HardwareClassification.GENERIC

    def test_with_calls_enter_and_exit(self):
        def f(x, log, suppress):
            y = 0
            with ExcManager(log, suppress) as m:
                y = m + exc_double(x)
            return y

        compiled = torch.cfg_compile(f, fallback=False)
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

        compiled = torch.cfg_compile(f, fallback=False)
        for manager in (NoExit(), NoEnter(), 3):
            self.assertEqual(call_outcome(compiled, manager), call_outcome(f, manager))
        missed = "'NoExit' object does not support the context manager protocol (missed __exit__ method)"
        self.assertEqual(call_outcome(compiled, NoExit()), (TypeError, missed))

    def test_no_grad_block(self):
        def f(x):
            with torch.no_grad():
                y = x * 2
                z = y + 1
            return z, z.requires_grad, x * 2

        compiled = torch.cfg_compile(f, fallback=False)
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

        compiled = torch.cfg_compile(f, fallback=False)
        with self.assertRaisesRegex(
            TypeError, "does not support the context manager protocol"
        ):
            compiled(3)

    def test_module_calls_are_compiled_all_the_way_down(self):
        def run(module, x):
            return module(x) + 1

        linear = torch.nn.Linear(3, 2)
        # Without the shortcut through Module.__call__, that is compiled too.
        compiled = torch.cfg_compile(run, fallback=False, shortcuts=False)
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


@unittest.skipUnless(HAS_RUNTIME, "cfg_compile runtime not built")
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

        compiled = torch.cfg_compile(f, fallback=False)
        self.assertEqual(compiled(2.5), f(2.5))
        with self.assertRaisesRegex(
            ImportError, "cannot import name 'nothing_here' from 'math'"
        ):
            torch.cfg_compile(missing, fallback=False)()

    def test_super(self):
        obj = BytecodeChild()
        for method in (BytecodeChild.scale, BytecodeChild.explicit):
            compiled = torch.cfg_compile(method, fallback=False)
            self.assertEqual(compiled(obj, 3), method(obj, 3))
        compiled = torch.cfg_compile(BytecodeChild.name.fget, fallback=False)
        self.assertEqual(compiled(obj), "child of base")

    def test_super_that_is_not_the_builtin(self):
        class Shadow:
            name = "something else"

        # A module that rebinds `super` gets what it asked for: its object is
        # called without arguments, as the source has it.
        method = BytecodeChild.name.fget
        compiled = torch.cfg_compile(method, fallback=False)
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

        compiled = torch.cfg_compile(f, fallback=False)
        for seq in ([1, 2, 3, 4], (1, 2), "abc", range(5)):
            self.assertEqual(compiled(seq), f(seq))
        for bad, error in (([1], ValueError), (5, TypeError)):
            with self.assertRaises(error) as expected:
                f(bad)
            with self.assertRaises(error) as got:
                compiled(bad)
            self.assertEqual(str(got.exception), str(expected.exception))

    def test_class_statement(self):
        def f(x):
            class Point:
                scale = 2

                def __init__(self, v):
                    self.v = v

                def scaled(self):
                    return self.v * Point.scale

            return Point(x).scaled(), Point.__qualname__

        compiled = torch.cfg_compile(f, fallback=False)
        self.assertEqual(compiled(3), f(3))
        self.assertEqual(compiled(torch.ones(2))[0], torch.full((2,), 2.0))
        self.assertTrue(
            compiled.stats()["compiled_callees"][0].endswith("f.<locals>.Point.scaled")
        )

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
            compiled = torch.cfg_compile(fn, fallback=False)
            self.assertEqual(call_outcome(compiled, *args), call_outcome(fn, *args))
        self.assertNotIn("bytecode_global", globals())

    def test_match_statement(self):
        compiled = torch.cfg_compile(match_describe, fallback=False)
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
            outcome = call_outcome(torch.cfg_compile(fn, fallback=False), *args)
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
        compiled = torch.cfg_compile(big, fallback=False)
        self.assertEqual(compiled(0, 3), big(0, 3))
        self.assertEqual(compiled(torch.zeros(2), 2), big(torch.zeros(2), 2))

    def test_optimizer_step_is_compiled(self):
        def step(opt):
            opt.zero_grad()
            opt.step()

        weight = torch.nn.Parameter(torch.ones(2))
        opt = torch.optim.SGD([weight], lr=0.5)
        compiled = torch.cfg_compile(step, fallback=False)
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


@unittest.skipUnless(HAS_RUNTIME, "cfg_compile runtime not built")
class TestCfgFrames(TestCase):
    """A compiled function runs under a frame of the interpreter, with its locals in registers."""

    hw_classification = HardwareClassification.GENERIC

    def test_function_that_reads_its_frame_is_not_compiled(self):
        def uses_locals(a, b):
            return sorted(locals())

        def uses_eval(a):
            return eval("a + 1")

        for f, args, name in (
            (uses_locals, (1, 2), "locals"),
            (uses_eval, (1,), "eval"),
        ):
            with self.assertRaisesRegex(
                torch.cfg_compile.Unsupported,
                name + r"\(\) reads the local variables of the frame",
            ):
                torch.cfg_compile(f, fallback=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertEqual(torch.cfg_compile(f)(*args), f(*args))

    @parametrize("inline", (False, True))
    def test_the_stack_shows_compiled_functions(self, inline):
        compiled = torch.cfg_compile(frame_outer, fallback=False, inline=inline)
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
        compiled = torch.cfg_compile(frame_new_classes, fallback=False)
        self.assertEqual(compiled(), (__name__, __name__))

    def test_warning_names_the_line_of_the_compiled_function(self):
        def caught(fn):
            with warnings.catch_warnings(record=True) as log:
                warnings.simplefilter("always")
                fn()
            return [(w.filename, w.lineno, str(w.message)) for w in log]

        compiled = torch.cfg_compile(frame_warns, fallback=False)
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

        compiled = torch.cfg_compile(frame_calls_divides, fallback=False, inline=inline)
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

        self.assertEqual(where(torch.cfg_compile(fn, fallback=False)), where(fn))

    def test_name_error_names_the_variable(self):
        def outcome(f, *args):
            try:
                f(*args)
            except NameError as e:
                return type(e), str(e), e.name

        for fn, args in ((frame_missing_global, ()), (frame_unbound_local, (False,))):
            compiled = torch.cfg_compile(fn, fallback=False)
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

        for fn in (fails, torch.cfg_compile(fails, fallback=False)):
            try:
                fn()
            except ValueError as e:
                # The traceback has the frame, and the frame the local.
                self.assertIsNotNone(probes[-1]())
                traceback.clear_frames(e.__traceback__)
                self.assertIsNone(probes[-1]())

    def test_callee_that_reads_its_frame_runs_in_the_interpreter(self):
        def fmt(name, value):
            return "{name}={value}".format(**locals())

        def f(obj, x):
            return fmt("x", x), obj.scale(x)

        compiled = torch.cfg_compile(f, fallback=False)
        self.assertEqual(compiled(FrameChild(), 3), ("x=3", 7))
        # What an interpreted function calls is not compiled either.
        self.assertEqual(compiled.stats()["compiled_callees"], [])
        interpreted = compiled.stats()["interpreted_callees"]
        self.assertEqual(
            [name.rsplit(".", 1)[-1] for name in interpreted], ["scale", "fmt"]
        )


def breaker_signals_itself(log):
    os.kill(os.getpid(), signal.SIGUSR1)
    log.append("after the call")


def breaker_spins(flag, limit):
    n = 0
    while not flag and n < limit:
        n += 1
    return n


@unittest.skipUnless(HAS_RUNTIME, "cfg_compile runtime not built")
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

        compiled = torch.cfg_compile(breaker_signals_itself, fallback=False)
        expected = [("handler", "breaker_signals_itself"), "after the call"]
        self.assertEqual(run(breaker_signals_itself), expected)
        self.assertEqual(run(compiled), expected)

    @unittest.skipUnless(hasattr(signal, "SIGUSR1"), "needs SIGUSR1")
    def test_exception_of_a_signal_handler_is_raised_by_the_call(self):
        def handler(signum, frame):
            raise KeyboardInterrupt("from the handler")

        compiled = torch.cfg_compile(breaker_signals_itself, fallback=False)
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

        compiled = torch.cfg_compile(breaker_spins, fallback=False)
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
        compiled = torch.cfg_compile(churn, fallback=False)
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


@unittest.skipUnless(HAS_RUNTIME, "cfg_compile runtime not built")
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

        compiled = torch.cfg_compile(fn, fallback=False, unroll=unroll)
        expected = run(fn)
        self.assertEqual(run(compiled), expected)
        self.assertEqual(run(compiled), expected)

    def test_hold_in_the_graph(self):
        compiled = torch.cfg_compile(lifetime_rebound, fallback=False)
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


@unittest.skipUnless(HAS_RUNTIME, "cfg_compile runtime not built")
class TestCfgShortcuts(TestCase):
    """Module.__call__ and Module.__getattr__ are skipped when a few checks show what they would do."""

    hw_classification = HardwareClassification.GENERIC

    def test_nn_module_is_what_the_shortcuts_were_written_for(self):
        # If this fails, nn.Module.__call__ or __getattr__ changed: read the
        # change and bring torch/cfg_compile/_shortcuts.py in line with it.
        self.assertTrue(torch.cfg_compile._shortcuts.unchanged())

    def test_module_call_goes_straight_to_forward(self):
        net, x = ShortcutNet(), torch.randn(4, 3)
        compiled = torch.cfg_compile(shortcut_call, fallback=False)
        for _ in range(3):
            self.assertEqual(compiled(net, x), net(x))
        callees = compiled.stats()["compiled_callees"]
        self.assertEqual(callees, ["Linear.forward", "ShortcutNet.forward"])
        self.assertEqual(compiled.stats()["interpreted_callees"], [])

    def test_module_with_hooks_is_called_the_long_way(self):
        net, x = ShortcutNet(), torch.randn(4, 3)
        compiled = torch.cfg_compile(shortcut_call, fallback=False)
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
        compiled = torch.cfg_compile(shortcut_call, fallback=False)
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
        compiled = torch.cfg_compile(shortcut_attributes, fallback=False)
        for _ in range(2):
            for got, expected in zip(compiled(net), shortcut_attributes(net)):
                self.assertIs(got, expected)
        self.assertEqual(self.getattr_calls(compiled, net), 0)
        without = torch.cfg_compile(
            shortcut_attributes, fallback=False, shortcuts=False
        )
        self.assertEqual(self.getattr_calls(without, net), 3)
        net.scale = torch.nn.Parameter(torch.zeros(2))
        net.plain = 7
        self.assertIs(compiled(net)[1], net.scale)
        self.assertEqual(compiled(net)[3:6], (7, "net", 14))
        missing = torch.cfg_compile(shortcut_missing, fallback=False)
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
        compiled = torch.cfg_compile(read, fallback=False)
        expected = call_outcome(read, module)
        self.assertEqual(Guarded.reads, 1)
        self.assertEqual(call_outcome(compiled, module), expected)
        self.assertEqual(Guarded.reads, 2)

    def test_without_shortcuts(self):
        net, x = ShortcutNet(), torch.randn(4, 3)
        compiled = torch.cfg_compile(shortcut_call, fallback=False, shortcuts=False)
        self.assertEqual(compiled(net, x), net(x))
        self.assertIn("Module._call_impl", compiled.stats()["compiled_callees"])


def demote_python(n):
    total = 0
    for i in range(n):
        total += i * i % 7
    return total


def demote_caller(n, x):
    return demote_python(n), x + 1


def demote_chain_1(n):
    return demote_python(n)


def demote_chain_2(n):
    return demote_chain_1(n)


def demote_chain_3(n):
    return demote_chain_2(n)


def demote_chain_4(n):
    return demote_chain_3(n)


@unittest.skipUnless(HAS_RUNTIME, "cfg_compile runtime not built")
class TestCfgDemotion(TestCase):
    """What the interpreter runs faster is left to it."""

    hw_classification = HardwareClassification.GENERIC

    def test_python_callee_is_left_to_the_interpreter(self):
        compiled = torch.cfg_compile(
            demote_caller, fallback=False, demote=True, inline=False
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

    def test_python_function_is_left_to_the_interpreter(self):
        compiled = torch.cfg_compile(demote_python, fallback=False, demote=True)
        self.assertFalse(compiled.stats()["demoted"])
        for _ in range(200):
            self.assertEqual(compiled(300), demote_python(300))
        self.assertTrue(compiled.stats()["demoted"])

    def test_measurement_is_reported(self):
        compiled = torch.cfg_compile(demote_python, fallback=False, demote=True)
        self.assertTrue(compiled.stats()["measuring"])
        for _ in range(200):
            compiled(300)
        stats = compiled.stats()
        self.assertEqual((stats["measuring"], stats["demoted"]), (False, True))
        unmeasured = torch.cfg_compile(demote_python, fallback=False, demote=False)
        self.assertFalse(unmeasured.stats()["measuring"])

    def test_chain_of_calls_is_measured_at_once(self):
        # Five functions deep. Were each measured from scratch once the one
        # it calls is settled, the outermost would still be at it.
        compiled = torch.cfg_compile(
            demote_chain_4, fallback=False, demote=True, inline=False
        )
        for _ in range(150):
            self.assertEqual(compiled(300), demote_python(300))
        stats = compiled.stats()
        self.assertFalse(stats["measuring"])
        # The loop is with the interpreter: on its own, or along with
        # callers of it, which cost the same either way once it is.
        self.assertTrue(stats["demoted"] or stats["demoted_callees"])

    def test_measurement_starts_over_when_compiled_code_changes(self):
        compiled = torch.cfg_compile(demote_python, fallback=False, demote=True)
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
        compiled = torch.cfg_compile(
            demote_caller, fallback=False, demote=False, inline=False
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

        compiled = torch.cfg_compile(caller, fallback=False, demote=True, inline=False)
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


@unittest.skipUnless(HAS_RUNTIME, "cfg_compile runtime not built")
class TestCfgEndToEnd(TestCase):
    """Whole programs, compared bit for bit with the interpreter."""

    hw_classification = HardwareClassification.GENERIC

    def test_tensor_program(self):
        # An inlined call, an unrolled loop, a branch on a shape that is
        # narrowed under a guard, a loop on tensor data and a Python side
        # effect, in one graph.
        narrowing = torch.cfg_compile.Pipeline.narrowing()
        compiled = torch.cfg_compile(e2e_program, pipeline=narrowing, fallback=False)
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
            step = (
                torch.cfg_compile(e2e_step, fallback=False)
                if compile_step
                else e2e_step
            )
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


class RegionRandom(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = RegionNet()

    def forward(self, x):
        return self.inner(x) * random.random()


@unittest.skipUnless(HAS_RUNTIME, "cfg_compile runtime not built")
class TestCfgRegions(TestCase):
    """Runs of a block compiled as one graph by ``backend=``."""

    hw_classification = HardwareClassification.GENERIC

    def compiled_regions(self, fn):
        return [r for r in torch.cfg_compile.regions(fn) if r["state"] == "compiled"]

    def left_to_vm(self, fn, where):
        return [
            r["reason"]
            for r in torch.cfg_compile.regions(fn)
            if r["state"] == "vm" and r["where"].startswith(where)
        ]

    def test_module_call_is_one_graph(self):
        torch.manual_seed(0)
        model = RegionNet().eval()
        compiled = torch.cfg_compile(region_infer, backend="aot_eager", fallback=False)
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
        self.assertIn("call_object", compiled.dump())

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
                step = torch.cfg_compile(step, backend="aot_eager", fallback=False)
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
        compiled = torch.cfg_compile(region_infer, backend="aot_eager", fallback=False)
        signs = set()
        with torch.no_grad():
            for i in range(24):
                x = torch.randn(2, 3, 6, 6) * (1 if i % 3 else -1)
                expected = region_infer(model, x)
                signs.add(bool(model.first(x).sum() > 0))
                self.assertEqual(compiled(model, x), expected, atol=0, rtol=0)
        self.assertEqual(signs, {True, False})
        # The call of the model does not trace, so the VM runs its forward,
        # and what is on either side of the branch became graphs.
        self.assertEqual(len(self.left_to_vm(compiled, "region_infer")), 1)
        where = sorted(
            r["where"].split(" (")[0] for r in self.compiled_regions(compiled)
        )
        self.assertEqual(where, ["RegionBranch.forward", "RegionBranch.forward"])

    def test_inputs_select_a_variant(self):
        graphs = []

        def backend(graph, inputs):
            graphs.append([tuple(i.shape) for i in inputs])
            return graph

        compiled = torch.cfg_compile(
            region_chain, backend=backend, fallback=False, region_max_variants=3
        )
        sizes = [2, 2, 2, 2, 3, 2, 3, 4, 5, 6, 5, 2]
        for n in sizes:
            x, y = torch.randn(n), torch.randn(n)
            self.assertEqual(compiled(x, y), region_chain(x, y), atol=0, rtol=0)
        # One graph per shape, up to the limit; the shapes after that ran the
        # instructions as they are.
        self.assertEqual(graphs, [[(2,), (2,)], [(3,), (3,)], [(4,), (4,)]])
        x = torch.randn(2, dtype=torch.float64)
        self.assertEqual(compiled(x, x), region_chain(x, x), atol=0, rtol=0)
        (region,) = self.compiled_regions(compiled)
        self.assertEqual(region["variants"], 3)

    def test_layers_built_alike_share_a_graph(self):
        graphs = []

        def backend(graph, inputs):
            graphs.append(graph)
            return graph

        torch.manual_seed(0)
        model = RegionLayers().eval()
        compiled = torch.cfg_compile(region_infer, backend=backend, fallback=False)
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

    def test_same_tensor_twice_is_its_own_variant(self):
        graphs = []

        def backend(graph, inputs):
            graphs.append(len(inputs))
            return graph

        compiled = torch.cfg_compile(region_chain, backend=backend, fallback=False)
        for _ in range(5):
            x, y = torch.randn(3), torch.randn(3)
            self.assertEqual(compiled(x, y), region_chain(x, y), atol=0, rtol=0)
            self.assertEqual(compiled(x, x), region_chain(x, x), atol=0, rtol=0)
        self.assertEqual(sorted(graphs), [1, 2])

    def test_small_region_is_left_to_the_vm(self):
        compiled = torch.cfg_compile(region_small, backend="aot_eager", fallback=False)
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
        compiled = torch.cfg_compile(region_infer, backend="aot_eager", fallback=False)

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
        (region,) = self.compiled_regions(compiled)
        self.assertEqual(region["variants"], 4)

    def test_frozen_parameter(self):
        torch.manual_seed(0)
        model = RegionNet().train()
        compiled = torch.cfg_compile(region_infer, backend="aot_eager", fallback=False)
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
        compiled = torch.cfg_compile(
            region_logging, backend="aot_eager", fallback=False
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
        compiled = torch.cfg_compile(region_infer, backend="aot_eager", fallback=False)
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

    def test_module_with_a_hook(self):
        torch.manual_seed(0)
        model = RegionNet().eval()
        compiled = torch.cfg_compile(region_infer, backend="aot_eager", fallback=False)
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
            r["reason"]
            for r in torch.cfg_compile.regions(compiled)
            if r["state"] == "vm"
        )
        self.assertIn("a module with hooks", reasons)

    def test_no_grad_section_is_not_differentiated(self):
        torch.manual_seed(0)
        model, reference = RegionNet().train(), RegionNet().train()
        reference.load_state_dict(model.state_dict())
        compiled = torch.cfg_compile(region_frozen, backend="aot_eager", fallback=False)
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
        compiled = torch.cfg_compile(region_loop, backend="aot_eager", fallback=False)
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

        compiled = torch.cfg_compile(region_chain, backend=backend, fallback=False)
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
                step = torch.cfg_compile(step, backend="aot_eager", fallback=False)
            losses = [step(model, opt, torch.randn(5, 4, 6)).item() for _ in range(8)]
            return model.state_dict(), losses

        expected_state, expected_losses = train(False)
        state, losses = train(True)
        self.assertEqual(losses, expected_losses)
        for name, value in state.items():
            self.assertTrue(torch.equal(value, expected_state[name]), name)

    def test_error_in_a_region(self):
        model = RegionNet().eval()
        compiled = torch.cfg_compile(region_infer, backend="aot_eager", fallback=False)
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
        compiled = torch.cfg_compile(
            region_infer, backend="aot_eager", fallback=False, demote=True
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
        compiled = torch.cfg_compile(region_infer, backend="aot_eager", fallback=False)
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

    def test_object_from_before_a_region_is_left_alone(self):
        torch.manual_seed(0)
        model = RegionStateful().eval()
        compiled = torch.cfg_compile(region_cached, backend="aot_eager", fallback=False)
        cache, expected_cache = RegionCache(), RegionCache()
        with torch.no_grad():
            for _ in range(8):
                x = torch.randn(2, 3, 6, 6)
                expected = region_cached(model, x, expected_cache)
                result = compiled(model, x, cache)
                self.assertEqual(result.logits, expected.logits, atol=0, rtol=0)
                # A trace that ran the update would have left its own tensors
                # in the cache, or left it one entry short on later calls.
                self.assertEqual(cache.seen, expected_cache.seen, atol=0, rtol=0)
        reasons = " / ".join(self.left_to_vm(compiled, "region_cached"))
        self.assertIn("takes a RegionCache from before the region", reasons)
        # The module below the call that takes the cache is a graph.
        self.assertGreaterEqual(len(self.compiled_regions(compiled)), 1)

    def test_named_tuples_and_sets_of_values_are_inputs(self):
        compiled = torch.cfg_compile(region_pair, backend="aot_eager", fallback=False)
        for i in range(8):
            pair = RegionPair(torch.randn(3), torch.randn(3))
            kinds = {"double"} if i % 2 else {"single", "plain"}
            result, expected = compiled(pair, kinds), region_pair(pair, kinds)
            self.assertIs(type(result), RegionPair)
            self.assertEqual(tuple(result), tuple(expected), atol=0, rtol=0)
        self.assertGreaterEqual(len(self.compiled_regions(compiled)), 2)
        # Neither was turned away as something a trace might change.
        reasons = " / ".join(self.left_to_vm(compiled, "region_pair"))
        self.assertNotIn("from before the region", reasons)

    def test_rnn_keeps_its_weights_in_a_list(self):
        # An RNN rebuilds the list when a weight was replaced, which a trace
        # does to all of them: the list is put back afterwards.
        torch.manual_seed(0)
        model = RegionRecurrent().eval()
        weights = list(model.lstm._flat_weights)
        compiled = torch.cfg_compile(region_infer, backend="aot_eager", fallback=False)
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
        compiled = torch.cfg_compile(region_infer, backend="aot_eager", fallback=False)
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
        compiled = torch.cfg_compile(region_global, backend="aot_eager", fallback=False)
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

        compiled = torch.cfg_compile(region_counter, backend=backend, fallback=False)
        x = torch.randn(4)
        for i in range(40):
            self.assertEqual(compiled(x, i), region_counter(x, i), atol=0, rtol=0)
        self.assertLessEqual(len(graphs), 3)
        self.assertEqual(self.compiled_regions(compiled), [])
        (reason,) = self.left_to_vm(compiled, "region_counter")
        self.assertIn("ever new values", reason)

    def test_call_without_tensors_is_not_frozen(self):
        torch.manual_seed(0)
        model = RegionNet().eval()
        compiled = torch.cfg_compile(region_clock, backend="aot_eager", fallback=False)
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
        compiled = torch.cfg_compile(region_infer, backend="aot_eager", fallback=False)
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
        step = torch.cfg_compile(region_step, backend="aot_eager", fallback=False)
        for _ in range(8):
            step(model, opt, torch.randn(4, 3, 6, 6), torch.randint(0, 5, (4,)))
        callees = step.stats()["compiled_callees"]
        self.assertIn("Adam.step", callees)
        where = {r["where"].split(" (")[0] for r in torch.cfg_compile.regions(step)}
        self.assertIn("region_step", where)
        # The optimizer is compiled by the VM, but nothing in it is a region.
        plumbing = ("Adam", "Optimizer", "_single_tensor", "_multi_tensor", "adam")
        self.assertEqual([w for w in where if w.startswith(plumbing)], [])

    def test_backend_needs_specialization(self):
        with self.assertRaisesRegex(ValueError, "specialize=True"):
            torch.cfg_compile(region_infer, backend="aot_eager", specialize=False)
        with self.assertRaisesRegex(ValueError, "unknown backend"):
            torch.cfg_compile(region_infer, backend="nope")

    @slowTest
    def test_inductor(self):
        def train(compile_step):
            torch.manual_seed(0)
            model = RegionNet().train()
            opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            step = region_step
            if compile_step:
                step = torch.cfg_compile(step, backend="inductor", fallback=False)
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
