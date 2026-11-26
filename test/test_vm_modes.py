# Owner(s): ["module: dynamo"]

"""Tests of torch.vm under the thread state that sits above the level a region
is traced at: a region's graph is what ran below autocast, and it is run, and
lowered, without autocast treating its operators a second time. Of what a
region keeps of the call it was traced on: none of that call's tensors. And
of a region's constants, the weights its graph reads: what the region checks
of them, a lowering that is bound to them, and a pass that keeps what it
derived from them while nothing wrote to them (vm.Unwritten). And of how a
call's tensors share memory: what is made of a region's graph runs only on
calls whose tensors overlap as the traced call's did. And of gradients: a
region with one input is differentiated, and a detach inside a region stops
the gradient on every list and can be named by a pattern. And of what C
code does to an operand: it is taken to have no effect only for the kinds of
operand the engine lists, so an iterator from outside, a C callable handed
to a builtin, a setter of torch's state and a defaultdict end a region; and
id() and hash() give which object their operand is on every call. And of
state a region reads without a check of its own: the backend flags are part
of the thread state a region is for, no region is traced or entered under a
torch function mode, and a method that tells of a tensor what a region does
not check ends the region also where it is taken from the class. And of a
nested tensor, which the executor's record of the operators that run does
not ask for sizes.
"""

import array
import collections
import contextlib
import ctypes
import functools
import gc
import itertools
import os
import subprocess
import sys
import time
import traceback
import unittest
import weakref

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    HardwareClassification,
    parametrize,
    run_tests,
    TEST_NUMPY,
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


def modes_weighted(x, w):
    return ((x * w).tanh().cos().sin() + x).sum()


def lowered(gm, input_types):
    # A lowering that keeps the graph it was given: what it returns calls it.
    return vm.lower_to(gm, gm.forward)


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmTracedTensors(TestCase):
    """The tensors of the call a region was traced on are the caller's to
    free, as in eager: the passes see them, the region does not keep them."""

    hw_classification = HardwareClassification.ACCELERATOR

    @parametrize("passes", ("default", "kept", "lowered", "autograd", "inductor"))
    @isolated
    def test_input_of_the_traced_call_is_freed(self, device, passes):
        behind = {
            "default": (),
            "kept": (keep,),
            "lowered": (lowered,),
            "autograd": (vm.passes.Autograd(), vm.passes.Partition()),
            "inductor": (vm.passes.Inductor(),),
        }[passes]
        exact = passes not in ("autograd", "inductor")
        tolerance = {"atol": 0, "rtol": 0} if exact else {"atol": 1e-05, "rtol": 1e-04}
        size = 1 << 20
        w = torch.full((size,), 1.5, device=device)
        compiled = vm(modes_weighted, passes=(*EARLY, *behind))
        inputs = []
        for call in range(12):
            # 4 MiB that only this call refers to.
            x = torch.randn(size, device=device)
            inputs.append(weakref.ref(x))
            got, want = compiled(x, w), modes_weighted(x, w)
            self.assertEqual(got, want, msg=f"call {call}", **tolerance)
            del x, got, want
        gc.collect()
        alive = [call for call, ref in enumerate(inputs) if ref() is not None]
        self.assertEqual(alive, [])
        # No other tensor of that size is left either (a copy of an input).
        left = [t for t in gc.get_objects() if type(t) is torch.Tensor and t is not w]
        self.assertEqual([t.shape for t in left if t.numel() == size], [])
        # The region runs on, and still says what it was traced for.
        before = region_runs(compiled)
        self.assertGreater(before, 0)
        x = torch.randn(size, device=device)
        self.assertEqual(compiled(x, w), modes_weighted(x, w), **tolerance)
        self.assertGreater(region_runs(compiled), before)
        self.assertIn(f"f32[{size}]", vm.explain(compiled))

    @isolated
    def test_gradients_of_the_traced_step_are_freed(self, device):
        # The optimizer's region is handed every gradient. With the gradients
        # set to None at the start of a step, only the last step's are alive.
        def make():
            torch.manual_seed(0)
            layers = [torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 8)]
            net = torch.nn.Sequential(*layers).to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

            def step(x):
                optimizer.zero_grad(set_to_none=True)
                loss = net(x).square().mean()
                loss.backward()
                optimizer.step()
                return loss.detach()

            return net, step

        (net, step), (eager_net, eager_step) = make(), make()
        compiled = vm(step, passes=EARLY)
        alive = []
        for call in range(12):
            x = torch.randn(8, 64, device=device)
            got, want = compiled(x), eager_step(x)
            self.assertEqual(got, want, atol=0, rtol=0, msg=f"call {call}")
            alive.append([weakref.ref(p.grad) for p in net.parameters()])
        weights, expected = list(net.parameters()), list(eager_net.parameters())
        self.assertEqual(weights, expected, atol=0, rtol=0)
        gc.collect()
        alive = [any(ref() is not None for ref in refs) for refs in alive]
        self.assertEqual(alive, [False] * 11 + [True])
        rows = [r for r in vm.regions(compiled) if r["state"] == "region"]
        self.assertTrue(any("adam" in r["where"] and r["runs"] for r in rows), msg=rows)


instantiate_device_type_tests(TestVmTracedTensors, globals())


class ModesWeights(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(8, 8))
        self.b = torch.nn.Parameter(torch.randn(8))

    def forward(self, x):
        y = torch.mm(x, self.w) + self.b
        return (y.reshape(y.shape[0] * y.shape[1]).tanh() * 2 + 1).cos()


def modes_chained(x, w):
    y = (x * w).sin().cos().tanh()
    # No region holds this call: the tensor work is two regions.
    time.perf_counter()
    return (y + x).sin().cos().tanh() * 2


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmConstants(TestCase):
    """A region's constants are the tensors its graph reads that are none of
    its arguments: a module's weights."""

    hw_classification = HardwareClassification.ACCELERATOR

    def bound(self, log):
        """A block pass whose lowering takes the constants once: it notes when
        it is bound and what it is called with, and does not look at the
        constants when it is called."""

        def lower(gm, input_types):
            def bind(consts):
                where = [vm.address(c) for c in consts]
                log.append(("bind", len(consts)))

                def call(*inputs):
                    log.append(("call", len(inputs)))
                    # Between two binds the constants are where they were.
                    now = [vm.address(c) for c in consts]
                    if now != where:
                        raise AssertionError("a constant moved and bind was not called")
                    return gm.forward(*inputs, *consts)

                return call

            return vm.lower_to(gm, bind=bind)

        return lower

    @isolated
    def test_weight_given_another_shape_is_traced_again(self, device):
        # `param.data = ...` leaves the parameter the object it was. The graph
        # is for its shape: a region is entered while its constants have the
        # metadata of the trace.
        net = ModesWeights().to(device)
        compiled = vm(net.forward, passes=EARLY)
        for call in range(14):
            x = torch.randn(4, 8, device=device)
            if call == 8:
                net.w.data = torch.randn(8, 8, device=device)
            if call == 11:
                net.w.data = torch.randn(8, 16, device=device)
                net.b.data = torch.randn(16, device=device)
            with torch.no_grad():
                want, got = net(x), compiled(x)
            self.assertEqual(got, want, atol=0, rtol=0, msg=f"call {call}")
        (row,) = (r for r in vm.regions(compiled) if r["state"] == "region")
        # The swap to the same shape is no miss; the other shape is one, and
        # gets a region of its own.
        self.assertEqual((row["variants"], sum(row["misses"].values())), (2, 1))

    @parametrize("lowering", ("autograd", "inductor"))
    @isolated
    def test_weight_that_starts_to_require_grad(self, device, lowering):
        # What AOT Autograd made of the region while the weight took no
        # gradient has no backward: the region is traced again when it does.
        behind = {
            "autograd": (vm.passes.Autograd(), vm.passes.Partition()),
            "inductor": (vm.passes.Inductor(),),
        }[lowering]
        net = ModesWeights().to(device).requires_grad_(False)
        compiled = vm(net.forward, passes=(*EARLY, *behind))
        for call in range(12):
            x = torch.randn(4, 8, device=device)
            if call == 8:
                net.w.requires_grad_(True)
            grads = []
            for fn in (net, compiled):
                out = fn(x).sum()
                self.assertEqual(out.requires_grad, call >= 8, msg=f"call {call}")
                net.w.grad = None
                if call >= 8:
                    out.backward()
                grads.append(net.w.grad)
            msg, close = f"call {call}", {"rtol": 1e-04, "atol": 1e-05}
            self.assertEqual(grads[1], grads[0], msg=msg, **close)
        self.assertGreater(region_runs(compiled), 0)

    @isolated
    def test_lowering_is_bound_to_the_constants(self, device):
        log = []
        net = ModesWeights().to(device)
        compiled = vm(net.forward, passes=(*EARLY, self.bound(log)))
        seen = []
        for call in range(16):
            x = torch.randn(4, 8, device=device)
            with torch.no_grad():
                if call == 8:
                    net.w.add_(1)
                if call == 10:
                    net.w.data = torch.randn(8, 8, device=device)
                if call == 13:
                    net.b.data = torch.randn(8, device=device)
                want = net(x)
                del log[:]
                got = compiled(x)
            self.assertEqual(got, want, atol=0, rtol=0, msg=f"call {call}")
            seen.append(list(log))
        # Bound at the trace (the fourth call), and again before the calls at
        # which a weight's memory was another; a weight written in place
        # (call 8) is read where it is. Every call hands over the one input
        # that is no constant.
        binds = [at for at, entries in enumerate(seen) if ("bind", 2) in entries]
        self.assertEqual(binds, [3, 10, 13])
        self.assertEqual(seen[4:10], [[("call", 1)]] * 6)
        self.assertEqual(seen[10], [("bind", 2), ("call", 1)])

    @isolated
    def test_bound_lowering_of_each_region_has_its_own_constants(self, device):
        # Two layers of one class have the same graph, and share what the
        # passes made of it: each region binds the lowering to its own weights.
        log = []
        first, second = ModesWeights().to(device), ModesWeights().to(device)

        def both(x):
            return first(x), second(x)

        compiled = vm(both, passes=(*EARLY, self.bound(log)))
        for call in range(8):
            x = torch.randn(4, 8, device=device)
            with torch.no_grad():
                want, got = both(x), compiled(x)
            self.assertEqual(got, want, atol=0, rtol=0, msg=f"call {call}")
        self.assertEqual(log.count(("bind", 2)), 2)
        self.assertGreater(region_runs(compiled), 0)

    def test_bound_lowering_called_with_every_input(self, device):
        # What `verify`, or a pass that wraps the lowering, does: then the
        # lowering looks at the constants itself.
        gm = torch.fx.symbolic_trace(lambda x, w: x * w + 1)
        gm.meta["vm_const_inputs"] = ["w"]
        binds = []

        def bind(consts):
            binds.append(list(consts))
            return lambda x: x * consts[0] + 1

        lowered = vm.lower_to(gm, bind=bind)
        x, w = torch.randn(4, device=device), torch.randn(4, device=device)
        other = torch.randn(4, device=device)
        self.assertEqual(lowered(x, w), x * w + 1)
        self.assertEqual(lowered(x + 1, w), (x + 1) * w + 1)
        self.assertEqual(len(binds), 1)
        self.assertEqual(lowered(x, other), x * other + 1)
        w.data = torch.randn(4, device=device)
        self.assertEqual(lowered(x, w), x * w + 1)
        self.assertEqual([c[0] is w for c in binds], [True, False, True])
        with self.assertRaisesRegex(TypeError, "bind="):
            vm.lower_to(gm)

    @isolated
    def test_input_type_says_which_region_made_the_input(self, device):
        seen = []

        def look(gm, input_types):
            seen.append([t.made_by for t in input_types])
            return gm

        compiled = vm(modes_chained, passes=(*EARLY, look))
        w = torch.randn(8, device=device)
        for _ in range(8):
            x = torch.randn(8, device=device)
            self.assertEqual(compiled(x, w), modes_chained(x, w), atol=0, rtol=0)
        rows = [r["where"] for r in vm.regions(compiled) if r["state"] == "region"]
        # The second region's first input is what the first region gave.
        self.assertEqual(seen, [[None, None], [(rows[0], 0), None]])


instantiate_device_type_tests(TestVmConstants, globals())


def modes_written(a, b):
    # Writes one input and then reads the other: what the read sees depends
    # on whether the two overlap.
    a.add_(1.0)
    c = b * 2.0
    a.mul_(c.sum().tanh())
    return (a - b).tanh() + c


class ModesShared(torch.nn.Module):
    """A weight in the middle of a larger piece of memory, so that an input
    can overlap it."""

    def __init__(self, device):
        super().__init__()
        self.store = torch.zeros(16, device=device)
        self.w = torch.nn.Parameter(self.store[4:12], requires_grad=False)

    def forward(self, x):
        x.add_(1.0)
        c = self.w * 2.0
        x.mul_(c.sum().tanh())
        return (x - self.w).tanh() + c


# Put on the test's device by the test.
MODES_READ = torch.arange(8.0) / 8


def modes_reading(x, w):
    # Reads a global tensor, and is given a tensor that may be that one.
    a = x * w
    b = a + MODES_READ.sum()
    return (b.tanh() - w) * 2.0 + MODES_READ


MODES_LISTS = {
    "default": EARLY,
    "autograd": (*EARLY, vm.passes.Autograd(), vm.passes.Partition()),
    "inductor": (*EARLY, vm.passes.Inductor()),
    "fast": vm.configure(vm.FAST, regions={"at": 3, "observe": 1}, pure_calls=1),
}


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmSharedMemory(TestCase):
    """A region whose graph block passes rewrote or lowered runs on calls
    whose tensors share memory as the traced call's did; another call is
    traced for itself. Eager's result either way, in the inputs too."""

    hw_classification = HardwareClassification.ACCELERATOR

    def inputs(self, device, call, apart, overlap, net):
        # The arguments of a call, and the memory to look at after it.
        torch.manual_seed(call)
        base, other = torch.randn(16, device=device), torch.randn(16, device=device)
        if overlap == "read":
            return (base[0:8], other[0:8] if apart else MODES_READ), [base]
        if net is not None:
            with torch.no_grad():
                net.store.copy_(torch.arange(16.0, device=device) / 16 + call / 8)
            if apart:
                return (base[0:8],), [base, net.store]
            x = net.store[0:8] if overlap == "partial" else net.w.detach()
            return (x,), [net.store]
        if apart:
            return (base[0:8], other[0:8]), [base, other]
        if overlap == "same":
            a = base[0:8]
            return (a, a), [base]
        return (base[0:8], base[4:12] if overlap == "partial" else base[0:8]), [base]

    def calls(self, device, passes, overlap, traced, module=None, fn=modes_written):
        nets = (module(device), module(device)) if module else (None, None)
        eager = nets[0] if module else fn
        compiled = vm(nets[1] if module else fn, passes=MODES_LISTS[passes])
        exact, close = {"atol": 0, "rtol": 0}, {"atol": 1e-05, "rtol": 1e-04}
        tolerance = exact if passes == "default" else close
        runs = []
        # Six calls as the region is traced (on the fourth), five the other
        # way, four as it was traced again.
        for call in range(15):
            apart = (traced == "apart") == (call < 6 or call >= 11)
            args, memory = self.inputs(device, call, apart, overlap, nets[0])
            want = [eager(*args), *memory]
            args, memory = self.inputs(device, call, apart, overlap, nets[1])
            got = [compiled(*args), *memory]
            self.assertEqual(got, want, msg=f"call {call}", **tolerance)
            runs.append(region_runs(compiled))
        # Regions ran each way.
        self.assertGreater(runs[5], 0)
        self.assertGreater(runs[10], runs[6])
        self.assertGreater(runs[14], runs[10])

    @parametrize("traced", ("apart", "overlapping"))
    @parametrize("passes", list(MODES_LISTS))
    @isolated
    def test_inputs_that_overlap_on_some_calls(self, device, passes, traced):
        self.calls(device, passes, "partial", traced)

    @parametrize("traced", ("apart", "overlapping"))
    @parametrize("passes", ("default", "inductor"))
    @isolated
    def test_one_tensor_given_twice_on_some_calls(self, device, passes, traced):
        # Traced when the two are one object, the graph has one input for
        # both: every list needs them to be one object again.
        self.calls(device, passes, "same", traced)

    @parametrize("traced", ("apart", "overlapping"))
    @parametrize("passes", list(MODES_LISTS))
    @isolated
    def test_input_over_a_weight_on_some_calls(self, device, passes, traced):
        self.calls(device, passes, "exact", traced, module=ModesShared)

    @isolated
    def test_input_that_overlaps_a_weight_in_part(self, device):
        self.calls(device, "inductor", "partial", "apart", module=ModesShared)

    @parametrize("traced", ("apart", "overlapping"))
    @parametrize("passes", ("default", "inductor"))
    @isolated
    def test_argument_that_is_a_tensor_the_body_reads(self, device, passes, traced):
        # Traced when the argument is the global tensor, the graph has one
        # input for the argument and for what the body read.
        global MODES_READ
        MODES_READ = MODES_READ.to(device)
        self.calls(device, passes, "read", traced, fn=modes_reading)

    def relation(self, inputs, consts=(), whole=True, read=()):
        return torch._rust.Overlap(list(inputs), list(consts), whole, list(read))

    def test_relation_traced_apart(self, device):
        a, b, w = (torch.zeros(8, device=device) for _ in range(3))
        r = self.relation([a, b], [w])
        base = torch.zeros(16, device=device)
        empty = torch.zeros(0, device=device)
        for inputs, holds in (
            ([torch.zeros(8, device=device), torch.zeros(8, device=device)], True),
            # One storage, and apart in it.
            ([base[0:8], base[8:16]], True),
            ([base[0:8], base[0:0]], True),
            ([a, empty], True),
            ([base[0:8], base[4:12]], False),
            ([base[0:8], base[0:8]], False),
            ([a, a], False),
            ([a, w], False),
            ([a, w.detach()], False),
            ([a, w[2:6]], False),
        ):
            self.assertEqual(r.holds(inputs), holds, msg=f"{inputs}")

    def test_relation_traced_overlapping(self, device):
        base = torch.zeros(16, device=device)
        r = self.relation([base[0:8], base[4:12]])
        other = torch.zeros(16, device=device)
        self.assertTrue(r.holds([other[0:8], other[4:12]]))
        # What is made of the graph is for where each is in the storage.
        self.assertFalse(r.holds([other[2:10], other[6:14]]))
        self.assertFalse(r.holds([other[0:8], other[2:10]]))
        self.assertFalse(r.holds([other[0:8], base[4:12]]))
        a = other[0:8]
        self.assertFalse(r.holds([a, a]))
        self.assertNotEqual(r.key(), self.relation([base[0:8], other[0:8]]).key())
        self.assertEqual(r.key(), self.relation([other[0:8], other[4:12]]).key())

    def test_relation_of_elements_that_interleave(self, device):
        # Spans are compared: two columns count as overlapping, and a call
        # stands as the trace did when they are the same two columns.
        m, n = torch.zeros(4, 4, device=device), torch.zeros(4, 4, device=device)
        r = self.relation([m[:, 0], m[:, 1]])
        self.assertTrue(r.holds([n[:, 0], n[:, 1]]))
        self.assertFalse(r.holds([n[:, 0], n[:, 2]]))
        self.assertFalse(r.holds([n[:, 0], m[:, 1]]))

    def test_relation_traced_with_one_tensor_twice(self, device):
        a, b = torch.zeros(8, device=device), torch.zeros(8, device=device)
        for whole in (True, False):
            r = self.relation([a, a, b], whole=whole)
            self.assertTrue(r.needed)
            self.assertTrue(r.holds([b, b, a]))
            self.assertFalse(r.holds([a, a.detach(), b]), msg=f"whole {whole}")
            self.assertFalse(r.holds([a, b, b]), msg=f"whole {whole}")
        # A graph run as traced takes the call's tensors as they come.
        r = self.relation([a, b], whole=False)
        self.assertFalse(r.needed)
        self.assertTrue(r.holds([a, a]))

    def test_relation_of_an_input_the_body_also_read(self, device):
        a, w = torch.zeros(8, device=device), torch.zeros(8, device=device)
        for whole in (True, False):
            r = self.relation([a, w], whole=whole, read=[1])
            self.assertTrue(r.needed)
            self.assertTrue(r.holds([torch.zeros(8, device=device), w]))
            self.assertFalse(r.holds([a, torch.zeros(8, device=device)]))

    def test_documented_limit_memory_that_two_storages_share(self, device):
        # Memory handed to PyTorch twice is two storages: nothing in PyTorch
        # knows that they overlap, and the relation goes by storages.
        if torch.device(device).type != "cpu":
            self.skipTest("memory from a buffer is the CPU's")
        memory = bytearray(64)
        a = torch.frombuffer(memory, dtype=torch.float32)
        b = torch.frombuffer(memory, dtype=torch.float32)
        r = self.relation([torch.zeros(16), torch.zeros(16)])
        self.assertTrue(r.holds([a, b]))

    def test_relation_follows_a_constant_whose_memory_moved(self, device):
        x, w, v = (torch.zeros(8, device=device) for _ in range(3))
        base = torch.zeros(16, device=device)
        r = self.relation([x], [w, v])
        self.assertTrue(r.holds([base[0:8]]))
        w.data = base[4:12]
        self.assertFalse(r.holds([base[0:8]]))
        self.assertTrue(r.holds([x]))
        v.data = base[8:16]
        self.assertFalse(r.holds([x]))
        v.data = torch.zeros(8, device=device)
        self.assertTrue(r.holds([x]))


instantiate_device_type_tests(TestVmSharedMemory, globals())


def modes_single(x):
    # One tensor in, and no weight.
    h = torch.tanh(x * 2.0)
    return ((h - 1.0) ** 2).sum() + h.mean()


def modes_detaching(device):
    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 8).to(device)

    def step(x):
        lin.zero_grad(set_to_none=True)
        h = torch.tanh(lin(x))
        # No gradient goes through the target.
        target = h.detach().mean(0)
        loss = ((h - target) ** 2).sum() + (h * target).sum()
        loss.backward()
        return loss.detach(), lin.weight.grad.clone()

    return step


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmGradients(TestCase):
    """What is made of a region's graph differentiates as the program does."""

    hw_classification = HardwareClassification.ACCELERATOR

    @isolated
    def test_region_with_one_input_is_differentiated(self, device):
        # The joint graph of such a region has more inputs than the region.
        compiled = vm(modes_single, passes=MODES_LISTS["autograd"])
        for call in range(9):
            torch.manual_seed(call)
            x = torch.randn(4, 8, device=device)
            a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
            want, got = modes_single(a), compiled(b)
            want.backward()
            got.backward()
            msg = f"call {call}"
            self.assertEqual(got, want, atol=1e-05, rtol=1e-04, msg=msg)
            self.assertEqual(b.grad, a.grad, atol=1e-05, rtol=1e-04, msg=msg)
        self.assertGreater(region_runs(compiled), 0)

    @parametrize("passes", list(MODES_LISTS))
    @isolated
    def test_gradient_stops_at_a_detach(self, device, passes):
        eager = modes_detaching(device)
        compiled = vm(modes_detaching(device), passes=MODES_LISTS[passes])
        exact, close = {"atol": 0, "rtol": 0}, {"atol": 1e-05, "rtol": 1e-04}
        tolerance = exact if passes == "default" else close
        for call in range(9):
            torch.manual_seed(call)
            x = torch.randn(4, 8, device=device)
            want, got = eager(x), compiled(x)
            self.assertEqual(got, want, msg=f"call {call}", **tolerance)
        self.assertGreater(region_runs(compiled), 0)

    @parametrize("passes", ("autograd", "inductor", "fast"))
    @isolated
    def test_backward_of_a_region_for_a_second_size(self, device, passes):
        # A backward graph is compiled when the backward first runs: for the
        # second size, in a call of the program's that traces nothing.
        eager = modes_detaching(device)
        compiled = vm(modes_detaching(device), passes=MODES_LISTS[passes])
        for call in range(10):
            torch.manual_seed(call)
            x = torch.randn(5 if call >= 6 else 4, 8, device=device)
            want, got = eager(x), compiled(x)
            self.assertEqual(got, want, msg=f"call {call}", atol=1e-05, rtol=1e-04)
        self.assertGreater(region_runs(compiled), 0)

    @isolated
    def test_pattern_with_a_detach_matches(self, device):
        found = []

        def look(gm, input_types):
            def pattern(h):
                return h.detach().mean(0)

            found.append(len(vm.match(gm, pattern)))
            return gm

        compiled = vm(modes_detaching(device), passes=(*EARLY, look))
        for call in range(9):
            compiled(torch.randn(4, 8, device=device))
        self.assertGreater(region_runs(compiled), 0)
        self.assertIn(1, found)


instantiate_device_type_tests(TestVmGradients, globals())


def modes_keeps(log):
    """A block pass that keeps a copy of each weight, under vm.Unwritten, and
    computes with the copies: what it derived from the weights' content. It
    notes when it kept copies and when it could not place the guard."""

    def keeps(gm, input_types):
        kept = {}
        placeholders = gm.graph.find_nodes(op="placeholder")
        for i, (placeholder, t) in enumerate(zip(placeholders, input_types)):
            if not t.is_const:
                continue
            try:
                vm.guard(gm, placeholder, vm.Unwritten())
            except vm.CannotGuard:
                log.append("cannot")
                continue
            kept[i] = t.const.detach().clone()
        if not kept:
            return gm
        log.append("kept")
        run = gm.forward

        def with_copies(*inputs):
            return run(*[kept.get(i, x) for i, x in enumerate(inputs)])

        return vm.lower_to(gm, with_copies)

    return keeps


class ModesWeight(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(8, 8) if weight is None else weight)

    def forward(self, x):
        return (torch.mm(x, self.w).tanh() * 2 + 1).cos()


def modes_on_storage(p):
    return torch.empty(0, device=p.device).set_(p.untyped_storage(), 0, p.shape)


# Ways of writing to a weight `p` through PyTorch.
MODES_WRITES = {
    "in_place": lambda p: p.add_(1),
    "through_a_view": lambda p: p[0].add_(1),
    "through_detach": lambda p: p.detach().add_(1),
    "through_data": lambda p: p.data.add_(1),
    "through_a_tensor_on_its_storage": lambda p: modes_on_storage(p).add_(1),
    "data_swapped": lambda p: setattr(p, "data", torch.randn_like(p)),
    "copied_into": lambda p: p.copy_(torch.randn_like(p)),
    "under_inference_mode": lambda p: torch.inference_mode()(lambda: p.add_(1))(),
}
# Ways of handing out the address of its memory, without a write.
MODES_EXPORTS = {
    "data_ptr": lambda p: p.data_ptr(),
    "numpy": lambda p: p.detach().numpy(),
    "dlpack": lambda p: torch.from_dlpack(p.detach()),
}


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmUnwritten(TestCase):
    """vm.Unwritten: what a pass derived from a weight's content stands until
    something writes to the weight through PyTorch."""

    hw_classification = HardwareClassification.ACCELERATOR

    def calls(self, device, net, compiled, at_7, n=12):
        """Calls 0 to n - 1 against eager's; ``at_7(net.w)`` before call 7."""
        equal = []
        for call in range(n):
            x = torch.randn(4, 8, device=device)
            with torch.no_grad():
                if call == 7:
                    at_7(net.w)
                want, got = net(x), compiled(x)
            equal.append(bool(torch.equal(got, want)))
        return equal

    @parametrize("way", list(MODES_WRITES))
    @isolated
    def test_write_fails_the_guard(self, device, way):
        log = []
        net = ModesWeight().to(device)
        compiled = vm(net.forward, passes=(*EARLY, modes_keeps(log)))
        equal = self.calls(device, net, compiled, MODES_WRITES[way])
        self.assertEqual(equal, [True] * 12)
        # The call after the write was traced again, and there the guard
        # could not be placed a second time: no copy is kept any more.
        self.assertEqual(log, ["kept", "cannot"])
        (row,) = (r for r in vm.regions(compiled) if r["state"] == "region")
        self.assertEqual((row["variants"], sum(row["misses"].values())), (2, 1))

    @parametrize("way", list(MODES_EXPORTS))
    @isolated
    def test_address_handed_out_fails_the_guard(self, device, way):
        if way != "data_ptr" and torch.device(device).type != "cpu":
            self.skipTest("NumPy and DLPack share the memory of CPU tensors")
        log = []
        net = ModesWeight().to(device)
        compiled = vm(net.forward, passes=(*EARLY, modes_keeps(log)))
        equal = self.calls(device, net, compiled, MODES_EXPORTS[way])
        self.assertEqual(equal, [True] * 12)
        self.assertEqual(log, ["kept", "cannot"])

    @isolated
    def test_reads_leave_the_guard(self, device):
        def reads(p):
            x = torch.randn(4, 8, device=p.device)
            return p @ p, torch.nn.functional.linear(x, p), p.sum(), p.clone(), p.half()

        log = []
        net = ModesWeight().to(device)
        compiled = vm(net.forward, passes=(*EARLY, modes_keeps(log)))
        self.assertEqual(self.calls(device, net, compiled, reads), [True] * 12)
        self.assertEqual(log, ["kept"])
        (row,) = (r for r in vm.regions(compiled) if r["state"] == "region")
        self.assertEqual((row["variants"], row["misses"]), (1, {}))

    @isolated
    def test_reported_write_places_the_guard_again(self, device):
        # What compiled code does after it wrote to a tensor: it bumps the
        # version counter, and nothing asked PyTorch for the memory. The
        # guard fails, and can be placed anew: the pass derives again.
        log = []
        net = ModesWeight().to(device)
        compiled = vm(net.forward, passes=(*EARLY, modes_keeps(log)))
        bump = torch.autograd.graph.increment_version
        self.assertEqual(self.calls(device, net, compiled, bump), [True] * 12)
        self.assertEqual(log, ["kept", "kept"])

    @isolated
    def test_memory_that_is_not_pytorchs_alone_takes_no_guard(self, device):
        if torch.device(device).type != "cpu":
            self.skipTest("memory shared with NumPy is CPU memory")
        # A NumPy array over the weight's memory, made before the pass looks:
        # a write through it could not be seen, so the guard is not placed.
        log = []
        net = ModesWeight()
        alias = net.w.detach().numpy()
        compiled = vm(net.forward, passes=(*EARLY, modes_keeps(log)))
        equal = self.calls(device, net, compiled, lambda p: alias.fill(1.0))
        self.assertEqual((equal, log), ([True] * 12, ["cannot"]))

    @isolated
    def test_documented_limit_write_through_an_address_taken_before(self, device):
        # THE LIMIT OF vm.Unwritten, shown, not covered: an address that
        # left PyTorch before the guard was placed. Nothing in PyTorch sees a
        # write through it (the version counter does not either), the guard
        # holds, and the region computes with what the pass derived.
        if torch.device(device).type != "cpu":
            self.skipTest("a write through a raw address of host memory")
        log = []
        net = ModesWeight()
        address = net.w.data_ptr()
        compiled = vm(net.forward, passes=(*EARLY, modes_keeps(log)))

        def poke(p):
            ctypes.memset(address, 0, p.numel() * p.element_size())

        equal = self.calls(device, net, compiled, poke)
        self.assertEqual(log, ["kept"])
        self.assertEqual(equal, [True] * 7 + [False] * 5)

    def test_guard_goes_on_a_weight_while_the_pass_runs(self, device):
        gm = torch.fx.symbolic_trace(lambda x, w: x * w + 1)
        x, w = gm.graph.find_nodes(op="placeholder")
        gm.meta["vm_const_inputs"] = ["w"]
        with self.assertRaisesRegex(vm.CannotGuard, "goes on a weight"):
            vm.guard(gm, x, vm.Unwritten())
        with self.assertRaisesRegex(vm.CannotGuard, "while the pass runs"):
            vm.guard(gm, w, vm.Unwritten())


instantiate_device_type_tests(TestVmUnwritten, globals())


def modes_around(x, k):
    # Four operations in front of the construct under test and four behind.
    return (((x.sin() * 2).cos() + 1).tanh() * k + 1).cos().tanh().exp()


class ModesBox:
    def __init__(self):
        self.n = 0.0


# State that outlives a call, made anew by modes_reset.
MODES_IT = MODES_KEPT = MODES_W = MODES_BOX = MODES_SEEN = None


def modes_reset(device):
    global MODES_IT, MODES_KEPT, MODES_W, MODES_BOX, MODES_SEEN
    MODES_IT = iter([float(i) for i in range(64)])
    MODES_KEPT = [3.0, 1.0, 2.0]
    MODES_W = torch.ones(8, device=device)
    MODES_BOX = ModesBox()
    MODES_SEEN = collections.defaultdict(float)


def modes_state():
    left = MODES_IT.__length_hint__()
    flags = (torch.is_grad_enabled(), MODES_W.requires_grad)
    return left, list(MODES_KEPT), flags, MODES_BOX.n, sorted(MODES_SEEN)


# C code takes items from an iterator that outlives the call.
def modes_unpack(x):
    (k,) = itertools.islice(MODES_IT, 1)
    return modes_around(x, k)


def modes_listed(x):
    return modes_around(x, list(itertools.islice(MODES_IT, 2))[1])


def modes_summed(x):
    return modes_around(x, sum(itertools.islice(MODES_IT, 2)))


def modes_starred(x):
    return modes_around(x, max(*itertools.islice(MODES_IT, 2)))


def modes_zipped(x):
    return modes_around(x, dict(zip("a", MODES_IT))["a"])


# A builtin calls a C callable that changes something.
def modes_mapped(x):
    list(map(MODES_KEPT.append, [1.5]))
    return modes_around(x, len(MODES_KEPT))


def modes_keyed(x):
    sorted([1.5], key=MODES_KEPT.append)
    return modes_around(x, len(MODES_KEPT))


def modes_sentinel(x):
    for _ in iter(MODES_KEPT.pop, 1.0):
        break
    MODES_KEPT.extend([4.0, 5.0])
    return modes_around(x, len(MODES_KEPT))


# A call sets state and runs no operator a graph would record.
def modes_grad_mode(x):
    torch._C._set_grad_enabled(False)
    return modes_around(x, 2.0)


def modes_requires_grad(x):
    MODES_W.requires_grad_(True)
    return modes_around(x, 2.0)


def modes_attribute(x):
    object.__setattr__(MODES_BOX, "n", MODES_BOX.n + 1.0)
    return modes_around(x, MODES_BOX.n)


def modes_detached(x):
    h = x * MODES_W
    h.detach_()
    return modes_around(h, 2.0)


# Reading a key a defaultdict lacks puts the key in.
def modes_missing(x):
    return modes_around(x, MODES_SEEN[len(MODES_SEEN)] + len(MODES_SEEN))


# Items taken from iterators the function makes itself: no effect.
def modes_made_zip(x):
    k = 0.0
    for u, v in zip(MODES_KEPT, (0.5, 0.25, 2.0)):
        k = k + u * v
    return modes_around(x, k)


def modes_made_generator(x):
    return modes_around(x, sum(u * 2 for u in MODES_KEPT) + max(MODES_KEPT))


def modes_made_enumerate(x):
    k = sum(i * u for i, u in enumerate(MODES_KEPT))
    return modes_around(x, k + len(list(reversed(MODES_KEPT))))


# A defaultdict read for keys that are there: the weights a list the
# function reads holds (an optimizer's state), and a string.
MODES_WEIGHTS = MODES_PER_WEIGHT = None


def modes_keyed_by_weights(x):
    y = ((x.sin() * 2).cos() + 1).tanh()
    k = MODES_PER_WEIGHT["steps"] + MODES_PER_WEIGHT[MODES_WEIGHTS[0]]
    return ((y * k + 1).cos().tanh().exp() * 2).sin()


MODES_ENDS = [
    modes_unpack,
    modes_listed,
    modes_summed,
    modes_starred,
    modes_zipped,
    modes_mapped,
    modes_keyed,
    modes_sentinel,
    modes_grad_mode,
    modes_requires_grad,
    modes_attribute,
    modes_detached,
    modes_missing,
]


# Every torchvision operator begins by telling a telemetry hook, once per
# process and key, that it was used. A region exists only after its code ran
# as written, so the report was made, and no program reads it.
def modes_logs_usage(x):
    torch._C._log_api_usage_once("test_vm_modes.modes_logs_usage")
    return modes_around(x, 2.0)


# torch.nn.functional.pad asks whether deterministic algorithms are on: a
# flag of the thread state every region is checked on.
def modes_padded(x):
    return modes_around(torch.nn.functional.pad(x, (0, 1))[:, 1:], 2.0)


MODES_STAYS = [
    modes_made_zip,
    modes_made_generator,
    modes_made_enumerate,
    modes_logs_usage,
    modes_padded,
]


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmEffects(TestCase):
    """What C code does to an operand has no effect only for the kinds of
    operand the engine lists (see exec/shown.rs): everything else ends the
    region, and the program's state is eager's after every call."""

    hw_classification = HardwareClassification.ACCELERATOR

    def log(self, device, fn, grad):
        # Nine calls from a fresh state: what each returned and left behind.
        modes_reset(device)
        out = []
        try:
            for call in range(9):
                torch.set_grad_enabled(True)
                MODES_W.requires_grad_(grad)
                torch.manual_seed(call)
                x = torch.randn(4, 8, device=device, requires_grad=grad)
                y = fn(x)
                out.append((y.detach(), y.requires_grad, modes_state()))
        finally:
            torch.set_grad_enabled(True)
        return out

    def both(self, device, fn):
        grad = fn is modes_detached
        want = self.log(device, fn, grad)
        compiled = vm(fn, passes=EARLY)
        self.assertEqual(self.log(device, compiled, grad), want, atol=0, rtol=0)
        return compiled

    @parametrize("fn", MODES_ENDS, name_fn=lambda fn: fn.__name__[6:])
    @isolated
    def test_effect_of_c_code_is_made_on_every_call(self, device, fn):
        self.both(device, fn)

    @parametrize("fn", MODES_STAYS, name_fn=lambda fn: fn.__name__[6:])
    @isolated
    def test_what_has_no_effect_stays_in_a_region(self, device, fn):
        # Iterators the function makes itself, a builtin of the table that
        # asks nothing of its arguments, a read of state regions are checked
        # on: one region for the function.
        compiled = self.both(device, fn)
        self.assertGreater(region_runs(compiled), 0)
        places = [r for r in vm.regions(compiled) if r["state"] == "region"]
        self.assertEqual(len(places), 1)

    @isolated
    def test_defaultdict_read_for_keys_that_are_there(self, device):
        # The region is found by the list's weights and by the dict's
        # content: the keys are there when it is entered, so nothing is put
        # in, and the read sits inside the region. A call on which a key is
        # not there is a call of its own, and the key is put in.
        global MODES_WEIGHTS, MODES_PER_WEIGHT
        MODES_WEIGHTS = [torch.ones(2, device=device)]
        MODES_PER_WEIGHT = collections.defaultdict(float, steps=0.5)
        MODES_PER_WEIGHT[MODES_WEIGHTS[0]] = 1.5
        compiled = vm(modes_keyed_by_weights, passes=EARLY)
        runs, places = [], 0
        for call in range(14):
            if call == 8:
                # The operations in front of the read and behind it are one
                # region so far. A weight the dict lacks:
                places = sum(r["state"] == "region" for r in vm.regions(compiled))
                MODES_WEIGHTS.insert(0, torch.ones(2, device=device))
            if call == 11:
                # A key that was there is gone.
                del MODES_PER_WEIGHT["steps"]
            torch.manual_seed(call)
            x = torch.randn(4, 8, device=device)
            got = compiled(x)
            # Its keys are the weights and "steps", after the call on the VM.
            keys = len(MODES_WEIGHTS) + 1
            self.assertEqual(len(MODES_PER_WEIGHT), keys, msg=f"call {call}")
            want = modes_keyed_by_weights(x)
            self.assertEqual(got, want, atol=0, rtol=0, msg=f"call {call}")
            runs.append(region_runs(compiled))
        self.assertGreater(runs[7], runs[4])
        self.assertGreater(runs[10], runs[8])
        self.assertGreater(runs[13], runs[11])
        self.assertEqual(places, 1)


instantiate_device_type_tests(TestVmEffects, globals())


def modes_work(x):
    return ((x.sin() * 2).cos() + 1).tanh().exp()


def modes_id(x):
    return modes_work(x), id(x)


def modes_hash(x):
    return modes_work(x), hash(x)


def modes_id_as_key(x):
    return modes_work(x), {id(x): 1}


def modes_id_of_made(x):
    y = modes_work(x)
    return y, id(y)


def modes_dunder_hash(x):
    return modes_work(x), x.__hash__()


def modes_id_of_list(x, items):
    return modes_work(x) + len(items), id(items)


def modes_id_of_read(x):
    return modes_work(x), id(MODES_BOX)


MODES_WHICH = [
    modes_id,
    modes_hash,
    modes_id_as_key,
    modes_id_of_made,
    modes_dunder_hash,
    modes_id_of_list,
    modes_id_of_read,
]


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmIdentity(TestCase):
    """id() and hash() have no effect, but what they give is which object
    their operand is: the call's own, not the traced call's. They stay in a
    region only where the region is found by that object."""

    hw_classification = HardwareClassification.ACCELERATOR

    @parametrize("fn", MODES_WHICH, name_fn=lambda fn: fn.__name__[6:])
    @isolated
    def test_which_object_is_this_calls(self, device, fn):
        modes_reset(device)
        compiled = vm(fn, passes=EARLY)
        for call in range(9):
            torch.manual_seed(call)
            x = torch.randn(4, 8, device=device)
            args = (x, [1, 2]) if fn is modes_id_of_list else (x,)
            want, got = fn(*args), compiled(*args)
            msg = f"call {call}"
            self.assertEqual(got[0], want[0], atol=0, rtol=0, msg=msg)
            # Of a tensor the call makes: the one the call returned.
            which = id(got[0]) if fn is modes_id_of_made else want[1]
            self.assertEqual(got[1], which, msg=msg)
        self.assertGreater(region_runs(compiled), 0)


instantiate_device_type_tests(TestVmIdentity, globals())


class ModesAddsOne(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        return out + 1 if func is torch.Tensor.cos else out


# No branch on what is read: the read is in the block of the operations.
def modes_reads_flag(x):
    return modes_around(x, (3.0, 2.0)[bool(torch.backends.cudnn.enabled)])


# A flag the thread state had before: its reader is a builtin of the table.
def modes_reads_deterministic(x):
    on = torch.are_deterministic_algorithms_enabled()
    return modes_around(x, (3.0, 2.0)[on])


# The method is taken from the class: no attribute of a tensor is loaded.
def modes_asks_the_class(x):
    return modes_around(x, 1.0 + torch.Tensor.storage_offset(x))


def modes_under_a_mode(x):
    return modes_around(x, 2.0)


MODES_READS = [
    modes_reads_flag,
    modes_reads_deterministic,
    modes_asks_the_class,
    modes_under_a_mode,
]


# An argument of a kind whose content no region checks, read by C code.
def modes_item_of(x, arg):
    return modes_work(x) * float(arg[0])


def modes_length_of(x, arg):
    return modes_work(x) * len(arg)


MODES_ARGUMENTS = {
    "numpy": (modes_item_of, lambda: torch.tensor([2.0, 1.0]).numpy()),
    "bytearray": (modes_item_of, lambda: bytearray([2, 1])),
    "deque": (modes_item_of, lambda: collections.deque([2.0, 1.0])),
    "array": (modes_item_of, lambda: array.array("d", [2.0, 1.0])),
    "deque_length": (modes_length_of, lambda: collections.deque([2.0, 1.0])),
}


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmStateRead(TestCase):
    """State a program reads and a region has no check of its own for: a
    call made in another state is not served by a region of the first."""

    hw_classification = HardwareClassification.ACCELERATOR

    def log(self, device, fn, what):
        # Six calls in the first state, three in the other, three in the
        # first again.
        out = []
        for call in range(12):
            other = 6 <= call < 9
            torch.manual_seed(call)
            x = torch.randn(4, 8, device=device)
            stack = contextlib.ExitStack()
            if what is modes_reads_flag:
                stack.enter_context(torch.backends.cudnn.flags(enabled=not other))
            elif what is modes_reads_deterministic:
                stack.enter_context(DeterministicGuard(other))
            elif what is modes_asks_the_class and other:
                # The same values, further into a storage.
                x = torch.cat([x, x])[4:]
            elif what is modes_under_a_mode and other:
                stack.enter_context(ModesAddsOne())
            with stack:
                out.append(fn(x).detach())
        return out

    @parametrize("fn", MODES_READS, name_fn=lambda fn: fn.__name__[6:])
    @isolated
    def test_call_in_another_state_is_eagers(self, device, fn):
        want = self.log(device, fn, fn)
        compiled = vm(fn, passes=EARLY)
        self.assertEqual(self.log(device, compiled, fn), want, atol=0, rtol=0)

    @parametrize("kind", list(MODES_ARGUMENTS))
    @isolated
    def test_argument_read_by_c_code_is_read_on_every_call(self, device, kind):
        if kind == "numpy" and not TEST_NUMPY:
            self.skipTest("needs numpy")
        fn, make = MODES_ARGUMENTS[kind]
        arg = make()
        compiled = vm(fn, passes=EARLY)
        for call in range(10):
            if call == 6:
                # The same object, with other content.
                if fn is modes_length_of:
                    arg.append(1.0)
                else:
                    arg[0] = 3
            torch.manual_seed(call)
            x = torch.randn(4, 8, device=device)
            got, want = compiled(x, arg), fn(x, arg)
            self.assertEqual(got, want, atol=0, rtol=0, msg=f"call {call}")
        # The operations in front of the read are still a region.
        self.assertGreater(region_runs(compiled), 0)


instantiate_device_type_tests(TestVmStateRead, globals())


def modes_nested(x, nt):
    return modes_work(x), (nt * 2).to_padded_tensor(0.0)


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmNestedTensors(TestCase):
    """A nested tensor has no sizes to ask for: the executor does not ask,
    and a function that is given one runs as eager runs it."""

    hw_classification = HardwareClassification.ACCELERATOR

    @isolated
    def test_nested_tensor_argument(self, device):
        compiled = vm(modes_nested, passes=EARLY)
        for call in range(8):
            torch.manual_seed(call)
            x = torch.randn(4, 8, device=device)
            parts = [torch.randn(n, 3, device=device) for n in (2, 4)]
            nt = torch.nested.nested_tensor(parts)
            got, want = compiled(x, nt), modes_nested(x, nt)
            self.assertEqual(got, want, atol=0, rtol=0, msg=f"call {call}")


instantiate_device_type_tests(TestVmNestedTensors, globals())


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestVmCudaGraphs(TestCase):
    """Lowerings that replay CUDA graphs of their own."""

    hw_classification = HardwareClassification.CUDA

    @isolated
    def test_cuda_graph_reads_the_constants_where_they_are(self):
        # The lowering a CUDA graph wants: the weights' addresses are in the
        # graph, and nothing compares them on a call.
        captures = []

        def graphed(gm, input_types):
            def bind(consts):
                state = {}

                def call(x):
                    if not state:
                        state["x"] = torch.empty_like(x)
                        state["graph"] = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(state["graph"]):
                            state["outs"] = gm.forward(state["x"], *consts)
                        captures.append(len(consts))
                    state["x"].copy_(x)
                    state["graph"].replay()
                    return [out.clone() for out in state["outs"]]

                return call

            return vm.lower_to(gm, bind=bind)

        device = "cuda"
        net = ModesWeights().to(device)
        compiled = vm(net.forward, passes=(*EARLY, graphed))
        with torch.no_grad():
            for call in range(14):
                x = torch.randn(4, 8, device=device)
                if call == 8:
                    net.w.add_(1)
                if call == 11:
                    net.w.data = torch.randn(8, 8, device=device)
                want, got = net(x), compiled(x)
                self.assertEqual(got, want, atol=0, rtol=0, msg=f"call {call}")
        self.assertEqual(captures, [2, 2])
        self.assertGreater(region_runs(compiled), 0)

    def unwritten(self, during):
        """Calls of a region whose pass keeps a copy of a CUDA weight under
        vm.Unwritten; ``during(net)`` runs before call 7. Returns what the
        pass noted."""
        log = []
        net = ModesWeight().to("cuda")
        compiled = vm(net.forward, passes=(*EARLY, modes_keeps(log)))
        with torch.no_grad():
            for call in range(12):
                x = torch.randn(4, 8, device="cuda")
                if call == 7:
                    during(net)
                want, got = net(x), compiled(x)
                self.assertEqual(got, want, atol=0, rtol=0, msg=f"call {call}")
        self.assertGreater(region_runs(compiled), 0)
        return log

    @isolated
    def test_capture_of_the_users_own_that_reads_a_guarded_weight(self):
        # PyTorch refuses a capture that reads a tensor with the guard's mark.
        # A capture begun from Python clears the marks first: it works as it
        # does without the VM, and the guard has failed.
        def capture(net):
            x = torch.randn(4, 8, device="cuda")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                y = torch.mm(x, net.w)
            graph.replay()
            self.assertEqual(y, torch.mm(x, net.w), atol=0, rtol=0)

        self.assertEqual(self.unwritten(capture), ["kept", "cannot"])

    @isolated
    def test_capture_that_keeps_the_guards(self):
        # A lowering whose graph reads what the pass derived, and no guarded
        # tensor, says so: its capture leaves the guards.
        def capture(net):
            x = torch.randn(4, 8, device="cuda")
            graph = torch.cuda.CUDAGraph()
            with vm.keep_unwritten(), torch.cuda.graph(graph):
                y = x * 2
            graph.replay()
            self.assertEqual(y, x * 2, atol=0, rtol=0)

        self.assertEqual(self.unwritten(capture), ["kept"])


if __name__ == "__main__":
    run_tests()
