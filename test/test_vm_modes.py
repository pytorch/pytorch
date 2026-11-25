# Owner(s): ["module: dynamo"]

"""Tests of torch.vm under the thread state that sits above the level a region
is traced at: a region's graph is what ran below autocast, and it is run, and
lowered, without autocast treating its operators a second time.
"""

import contextlib
import functools
import os
import subprocess
import sys
import traceback
import unittest

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)


vm = torch.vm
HAS_VM = getattr(torch, "_rust", None) is not None and sys.version_info[:2] == (3, 12)

# The default pass list with regions formed early: types are observed on the
# third call of a function, the fourth call is traced, and the fifth is the
# first to run a region (the default schedule starts at call 24).
EARLY = vm.configure(vm.DEFAULT_PASSES, regions={"at": 3, "observe": 1}, pure_calls=1)

# Set in a child process to the id of the one test it is to run.
CHILD = "TORCH_VM_TEST_CHILD"


def isolated(test):
    """Runs a test method in a child process of its own: the executor's
    switches are process-wide, so a test that needs regions to form on a known
    call of its own function runs alone (as in test_vm.py)."""

    @functools.wraps(test)
    def wrapper(self, *args, **kwargs):
        if os.environ.get(CHILD) == self.id():
            method = getattr(type(self), self._testMethodName)
            if not getattr(method, "__unittest_expecting_failure__", False):
                return test(self, *args, **kwargs)
            # An expected failure is the parent's to report: the child only
            # says, by its exit status, whether the test failed.
            try:
                test(self, *args, **kwargs)
            except BaseException:
                traceback.print_exc()
                sys.stderr.flush()
                os._exit(1)
            os._exit(0)
        name = f"{type(self).__name__}.{self._testMethodName}"
        env = {**os.environ, CHILD: self.id()}
        # A crash is to end the child at once, without a core file.
        command = (
            f"ulimit -c 0; exec {sys.executable} {os.path.abspath(__file__)} {name}"
        )
        ran = subprocess.run(
            ["/bin/sh", "-c", command],
            env=env,
            capture_output=True,
            text=True,
            timeout=1200,
        )
        if ran.returncode != 0:
            how = (
                f"died of signal {-ran.returncode}"
                if ran.returncode < 0
                else f"exited with status {ran.returncode}"
            )
            self.fail(f"the child process {how}:\n{ran.stderr[-3000:]}")

    return wrapper


def region_runs(fn):
    """How many times the regions of ``fn`` have run."""
    return sum(r["runs"] for r in vm.regions(fn) if r["state"] == "region")


def keep(gm, input_types):
    return gm


def under_autocast(device, dtype=torch.bfloat16):
    """Autocast on ``device``, and the attention operator's math path (the
    one a build without a fused kernel runs)."""
    stack = contextlib.ExitStack()
    stack.enter_context(torch.autocast(torch.device(device).type, dtype=dtype))
    stack.enter_context(sdpa_kernel(SDPBackend.MATH))
    return stack


def modes_attend(q, k, v):
    # Autocast casts the three to its dtype and runs the operator with
    # autocast off; the operator's math path raises them to float32 for its
    # two matrix products.
    a = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return (a + q).tanh() * 2.0


def modes_attended(device):
    """Makes the three inputs of modes_attend in a dtype."""

    def make(dtype):
        return [torch.randn(2, 4, 16, 8, device=device).to(dtype) for _ in "qkv"]

    return make


def modes_distances(x, y):
    # Autocast runs cdist in float32, with autocast off for the matrix
    # product inside it.
    return (torch.cdist(x, y) + 1).tanh() * 2.0


class ModesProduct(torch.autograd.Function):
    """A product whose gradient is passed straight through."""

    @staticmethod
    def forward(ctx, x, w):
        return torch.mm(x, w)

    @staticmethod
    def backward(ctx, grad):
        return grad.float() * 0.5, None


def modes_function(x, w):
    y = ModesProduct.apply(x, w)
    a = torch.nn.functional.scaled_dot_product_attention(y[None], y[None], y[None])
    return (a[0].float() + x).tanh() * 2.0


modes_seen = []


def modes_note(y, w):
    modes_seen.append(torch.mm(y, w).dtype)


def modes_inner_notes(y, w):
    modes_note(y, w)
    a = torch.nn.functional.scaled_dot_product_attention(y[None], y[None], y[None])
    return a[0].float().tanh() * 2


def modes_outer_notes(x, w):
    return modes_inner_notes(x.sin() + 1, w).cos().exp()


class ModesNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.proj = torch.nn.Linear(8, 8)
        self.head = torch.nn.Linear(8, 4)

    def forward(self, x):
        h = self.conv(x).relu().flatten(2).transpose(1, 2)
        q = self.proj(h)[:, None]
        a = torch.nn.functional.scaled_dot_product_attention(q, q, q)[:, 0]
        return self.head((a + h).tanh()).mean(1)


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmAutocast(TestCase):
    """A region traced under autocast computes what eager computes there."""

    hw_classification = HardwareClassification.ACCELERATOR

    def calls(self, device, dtype, compiled, fn, make, exact=True, **tolerance):
        # Eight calls per kind of input: the fourth is traced, the later ones
        # run the region.
        for inputs in (torch.float32, dtype):
            before = region_runs(compiled)
            for call in range(8):
                args = make(inputs)
                # Eager first: a warning an operator gives once is eager's.
                with under_autocast(device, dtype):
                    want, got = fn(*args), compiled(*args)
                msg = f"{inputs} inputs, call {call}"
                self.assertEqual(got.dtype, want.dtype, msg=msg)
                if exact:
                    self.assertEqual(got, want, atol=0, rtol=0, msg=msg)
                else:
                    self.assertEqual(got.float(), want.float(), msg=msg, **tolerance)
            self.assertGreater(region_runs(compiled), before, msg=f"{inputs} inputs")

    @dtypes(torch.bfloat16, torch.float16)
    @isolated
    def test_attention_runs_its_products_as_eager_does(self, device, dtype):
        # Eager's two matrix products run in float32. The region's graph
        # holds them in float32; run under autocast again they would be cast
        # to the autocast dtype.
        compiled, make = vm(modes_attend, passes=EARLY), modes_attended(device)
        self.calls(device, dtype, compiled, modes_attend, make)

    @dtypes(torch.bfloat16, torch.float16)
    @isolated
    def test_operator_autocast_runs_in_float32_stays_in_float32(self, device, dtype):
        def make(inputs):
            return [torch.randn(40, 8, device=device).to(inputs) for _ in "xy"]

        compiled = vm(modes_distances, passes=EARLY)
        self.calls(device, dtype, compiled, modes_distances, make)

    @dtypes(torch.bfloat16, torch.float16)
    @isolated
    def test_block_pass_that_keeps_the_graph(self, device, dtype):
        compiled = vm(modes_attend, passes=(*EARLY, keep))
        self.calls(device, dtype, compiled, modes_attend, modes_attended(device))

    @dtypes(torch.bfloat16, torch.float16)
    @isolated
    def test_graph_lowered_by_aot_autograd(self, device, dtype):
        # AOT Autograd traces the region's graph again, under autocast: the
        # graph it is handed runs with autocast off. The versions are compared
        # with each other (verify) the way a region runs them.
        passes = (*EARLY, vm.passes.Autograd(), vm.passes.Partition())
        compiled = vm(modes_attend, passes=passes, verify=True)
        make, close = modes_attended(device), {"rtol": 1e-05, "atol": 1e-06}
        self.calls(device, dtype, compiled, modes_attend, make, False, **close)
        reports = compiled.__vm_program__.pipeline.reports.values()
        self.assertEqual([n for r in reports for n in r.notes if "verify" in n], [])
        self.assertEqual([r.name for r in reports if not r.versions[-1].hits], [])

    @dtypes(torch.bfloat16, torch.float16)
    @isolated
    def test_graph_lowered_by_inductor_is_what_torch_compile_gives(self, device, dtype):
        # Inductor has numerics of its own in the autocast dtype, so the
        # region is compared with the same function under torch.compile.
        compiled = vm(modes_attend, passes=(*EARLY, vm.passes.Inductor()))
        args = modes_attended(device)(torch.float32)
        with under_autocast(device, dtype):
            modes_attend(*args)
            for _ in range(8):
                got = compiled(*args)
            self.assertGreater(region_runs(compiled), 0)
            expected = torch.compile(modes_attend)(*args)
        reports = compiled.__vm_program__.pipeline.reports.values()
        self.assertEqual([n for r in reports for n in r.notes], [])
        self.assertEqual(got.dtype, expected.dtype)
        self.assertEqual(got, expected, rtol=1e-05, atol=1e-05)

    @isolated
    def test_lowered_graph_is_for_the_autocast_state_of_its_trace(self, device):
        # With inputs in the autocast dtype the attention's graph is the same
        # with autocast off and on. What AOT Autograd made of it with autocast
        # off does not stand for it under autocast: there its backward would
        # run with autocast on.
        passes = (*EARLY, vm.passes.Autograd(), vm.passes.Partition())
        compiled = vm(modes_attend, passes=passes)
        for call in range(16):
            q, k, v = modes_attended(device)(torch.bfloat16)
            leaves = [q.clone().requires_grad_(), q.clone().requires_grad_()]
            outs = []
            with under_autocast(device) if call >= 8 else sdpa_kernel(SDPBackend.MATH):
                for fn, leaf in zip((modes_attend, compiled), leaves):
                    outs.append(fn(leaf, k, v).float())
                    outs[-1].sum().backward()
            grads = [leaf.grad.float() for leaf in leaves]
            msg = f"call {call}"
            self.assertEqual(outs[1], outs[0], rtol=1e-05, atol=1e-06, msg=msg)
            self.assertEqual(grads[1], grads[0], rtol=1e-05, atol=1e-06, msg=msg)
        self.assertGreater(region_runs(compiled), 0)

    @parametrize("lowering", ("none", "autograd", "inductor"))
    @isolated
    def test_backward_inside_the_autocast_block(self, device, lowering):
        # The gradients of a step whose backward runs under autocast too:
        # eager's, bit for bit, with the default list; within the autocast
        # dtype's precision when AOT Autograd made the backward.
        kind = torch.device(device).type
        passes = {
            "none": EARLY,
            "autograd": (*EARLY, vm.passes.Autograd(), vm.passes.Partition()),
            "inductor": (*EARLY, vm.passes.Inductor()),
        }[lowering]

        def make():
            torch.manual_seed(0)
            net = ModesNet().to(device)

            def step(x, y):
                net.zero_grad(set_to_none=True)
                with torch.autocast(kind, dtype=torch.bfloat16):
                    loss = torch.nn.functional.cross_entropy(net(x), y)
                    loss.backward()
                return [loss.detach(), *(p.grad.clone() for p in net.parameters())]

            return step

        eager, compiled = make(), vm(make(), passes=passes)
        exact, close = {"atol": 0, "rtol": 0}, {"atol": 0.01, "rtol": 0.01}
        tolerance = exact if lowering == "none" else close
        with sdpa_kernel(SDPBackend.MATH):
            for call in range(9):
                torch.manual_seed(call)
                x = torch.randn(4, 3, 8, 8, device=device)
                y = torch.randint(0, 4, (4,), device=device)
                want, got = eager(x, y), compiled(x, y)
                self.assertEqual(got, want, msg=f"call {call}", **tolerance)
        self.assertGreater(region_runs(compiled), 0)

    @isolated
    def test_custom_function_runs_under_the_callers_autocast(self, device):
        # The region's graph calls the Function's apply as one node, and runs
        # with autocast off: the Function's forward is the program's own code,
        # and runs under autocast as eager runs it (its product in bfloat16).
        # The call of a Function is taken into a region when the pass list
        # says it only computes.
        apply = torch._C._FunctionBase.__dict__["apply"]
        passes = vm.configure(EARLY, pure_calls={"rounds": 1, "pure": (apply,)})
        compiled = vm(modes_function, passes=passes)
        w = torch.randn(8, 8, device=device)
        for call in range(8):
            x = torch.randn(8, 8, device=device)
            xs = [x.clone().requires_grad_(), x.clone().requires_grad_()]
            outs = []
            with under_autocast(device):
                for fn, leaf in zip((modes_function, compiled), xs):
                    outs.append(fn(leaf, w))
                    outs[-1].sum().backward()
            self.assertEqual(outs[1], outs[0], atol=0, rtol=0, msg=f"call {call}")
            self.assertEqual(xs[1].grad, xs[0].grad, atol=0, rtol=0, msg=f"call {call}")
        self.assertGreater(region_runs(compiled), 0)
        self.assertIn("apply_ModesProduct", vm.explain(compiled, view="fx"))

    @isolated
    def test_call_merged_as_it_is_runs_under_the_callers_autocast(self, device):
        def merge_note(cfg: vm.Cfg) -> None:
            for call in cfg.calls():
                if call.callee is modes_note:
                    cfg.merge_across(call)

        passes = (*vm.configure(EARLY, pure_calls=3), merge_note)
        compiled = vm(modes_outer_notes, passes=passes)
        w = torch.randn(8, 8, device=device)
        for call in range(8):
            x = torch.randn(8, 8, device=device)
            del modes_seen[:]
            with under_autocast(device):
                want, got = modes_outer_notes(x, w), compiled(x, w)
            # The call the pass merged is made once by each, under autocast.
            self.assertEqual(modes_seen, [torch.bfloat16] * 2, msg=f"call {call}")
            self.assertEqual(got, want, atol=0, rtol=0, msg=f"call {call}")
        self.assertGreater(region_runs(compiled), 0)
        self.assertIn("opaque_modes_note", vm.explain(compiled, view="fx"))


instantiate_device_type_tests(TestVmAutocast, globals())


if __name__ == "__main__":
    run_tests()
