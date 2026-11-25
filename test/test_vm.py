# Owner(s): ["module: dynamo"]

"""Tests of torch.vm: a function run on the torch._rust executor computes what
eager mode computes: the same results, gradients, writes to its inputs and
random draws, whatever the function does between its tensor operations.
"""

import abc
import collections
import collections.abc
import contextlib
import copy
import dis
import enum
import functools
import gc
import math
import os
import shutil
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

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils.checkpoint import checkpoint


vm = torch.vm
HAS_VM = getattr(torch, "_rust", None) is not None and sys.version_info[:2] == (3, 12)

# The default pass list with regions formed early: types are observed on the
# third call of a function, the fourth call is traced, and the fifth is the
# first to run a region (the default schedule starts at call 24).
EARLY = vm.configure(vm.DEFAULT_PASSES, regions={"at": 3, "observe": 1}, pure_calls=1)


def vm_early(fn):
    """``fn`` on the VM with the pass list EARLY."""
    return vm(fn, passes=EARLY)


# Set in a child process to the id of the one test it is to run.
CHILD = "TORCH_VM_TEST_CHILD"


def isolated(test):
    """Runs a test method in a child process of its own.

    The executor's switches are process-wide: once one function's schedule has
    turned regions on, every function run on the VM in that process is traced
    on its first call. A test that needs regions to form on a known call of
    its own function, or that may take the process down, therefore runs alone:
    the method starts this file again for just that test, and fails with what
    the child wrote if the child does not succeed.
    """

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


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmTracedCall(TestCase):
    """What the call that traces a region, and the calls that run it, give."""

    hw_classification = HardwareClassification.ACCELERATOR

    @isolated
    def test_loop_with_a_test_on_tensor_data(self, device):
        # The trace of a span cannot hold a branch on tensor data: it gives
        # up, and the call goes on as if nothing had been traced.
        class Holder:
            pass

        def f(x, holder, n):
            for i in range(n):
                y = torch.relu(x) * holder.scale
                if i + 1 < n and y.shape[0] > 0 and (y > -1).any():
                    x = y[0] + x * math.pi
            return x

        holder = Holder()
        holder.scale = 2.0
        x = torch.ones(8, device=device)
        g = vm(f, passes=EARLY)
        for _ in range(8):
            self.assertEqual(g(x, holder, 3), f(x, holder, 3), atol=0, rtol=0)

    @isolated
    def test_traced_call_takes_a_branch_never_taken_before(self, device):
        # The call that is traced leaves the loop body by a block that no
        # earlier call ran: the trace gives up there, and the call goes on.
        def f(x, n, stop):
            for i in range(n):
                x = (x.sin() * 2 + 1).tanh().cos()
                if i == stop:
                    x = x - 1
                    break
            return x.exp()

        x = torch.randn(8, device=device)
        g = vm(f, passes=EARLY)
        for call in range(8):
            stop = 1 if call == 3 else -1
            self.assertEqual(g(x, 3, stop), f(x, 3, stop), atol=0, rtol=0)

    @isolated
    def test_float_a_region_computes_is_a_float(self, device):
        # Floats are traced as a subclass of float. What the traced call hands
        # back to the program is a float again, wherever the function puts it.
        def f(x, rate):
            y = ((x.sin() * rate).cos() + rate).tanh() * 2
            return y, rate * 2.0, (rate + 1.0, [rate * 3.0], {"k": rate - 1.0})

        g = vm(f, passes=EARLY)
        x = torch.randn(4, device=device)
        for _ in range(8):
            y, doubled, (plus, (tripled,), table) = g(x, 0.5)
            got = [doubled, plus, tripled, table["k"]]
            self.assertEqual([type(v) for v in got], [float] * 4)
            self.assertEqual(got, [1.0, 1.5, 1.5, -0.5])
            self.assertEqual(y, f(x, 0.5)[0], atol=0, rtol=0)

    @isolated
    def test_device_argument_is_another_object_on_every_call(self, device):
        # `tensor.device` makes a new object on every read: the region is for
        # the device it names, and runs from the fifth call on.
        import torch._rust_inductor as ri

        def f(x, where):
            return (torch.arange(8, device=where) + x).sin().cos().tanh().exp()

        ri.COUNT = True
        g = vm(f, passes=EARLY)
        for call in range(8):
            x = torch.randn(4, 8, device=device)
            self.assertEqual(
                g(x, x.device), f(x, x.device), atol=0, rtol=0, msg=f"call {call}"
            )
        self.assertEqual(
            (region_runs(), ri.stats["compiled"] + ri.stats["reused"]), (4, 1)
        )

    @isolated
    def test_string_argument_is_compared_by_value(self, device):
        # A str made anew on every call, changed and changed back: each value
        # has its region, found again when the value comes back.
        import torch._rust_inductor as ri

        def f(x, how):
            y = (x.sin() * 2 + 1).cos().tanh()
            return y.sum() if how == "sum" else y.mean()

        ri.COUNT = True
        g = vm(f, passes=EARLY)
        runs, traced = [], []
        for how, calls in [("sum", 5), ("mean", 3), ("sum", 3), ("mean", 2)]:
            for _ in range(calls):
                x = torch.randn(4, 8, device=device)
                before = region_runs()
                made = "".join(list(how))
                self.assertIsNot(made, how)
                self.assertEqual(g(x, made), f(x, how), atol=0, rtol=0)
                runs.append(region_runs() - before)
            traced.append(ri.stats["traced"])
        self.assertGreater(runs[4], 0)
        self.assertGreater(runs[7], 0)
        self.assertEqual((runs[10], runs[12]), (runs[4], runs[7]))
        self.assertEqual(traced[3], traced[1])


instantiate_device_type_tests(TestVmTracedCall, globals())


class WritesBox:
    pass


class WritesSlotted:
    __slots__ = ("rate",)


class WritesModule(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.register_buffer("seen", torch.zeros(4, device=device))
        self.weight = torch.nn.Parameter(torch.ones(4, device=device))
        self.steps = 0.0


WRITES_TOTAL = 0.0


def writes_work(x, rate):
    return ((x.sin() * rate).cos() + rate).tanh() * 2


def writes_tail(y, rate):
    return (y * rate).relu() + y.sigmoid()


def writes_attribute(x, rate, box):
    y = writes_work(x, rate)
    box.rate = rate * 4.0
    return writes_tail(y, box.rate)


def writes_module_attribute(x, rate, module):
    y = writes_work(x, rate)
    module.steps = module.steps + rate
    return writes_tail(y, rate)


def writes_module_buffer(x, rate, module):
    y = writes_work(x, rate)
    module.seen = module.seen + y
    return writes_tail(y, rate)


def writes_module_parameter(x, rate, module):
    y = writes_work(x, rate)
    with torch.no_grad():
        module.weight.mul_(rate)
    return writes_tail(y, rate) * module.weight


def writes_item(x, rate, items):
    y = writes_work(x, rate)
    items["rate" if isinstance(items, dict) else 0] = rate * 4.0
    return writes_tail(y, rate)


def writes_global(x, rate, _):
    global WRITES_TOTAL
    y = writes_work(x, rate)
    WRITES_TOTAL = WRITES_TOTAL + rate
    return writes_tail(y, rate)


def writes_cell():
    total = 0.0

    def run(x, rate, _):
        nonlocal total
        y = writes_work(x, rate)
        total = total + rate
        return writes_tail(y, rate)

    return run, lambda _: total


def writes_in_place(x, rate, box):
    y = writes_work(x, rate)
    box.acc.add_(y)
    return writes_tail(y, rate) + box.acc


def writes_in_a_loop(x, rate, box):
    for i in range(3):
        x = writes_work(x, rate)
        box.rate = rate * (i + 1.0)
    return writes_tail(x, rate)


def writes_then_raises(x, rate, box):
    y = writes_work(x, rate)
    box.rate = rate * 4.0
    if rate > 0.9:
        raise ValueError("behind the store")
    return writes_tail(y, rate)


def writes_then_fails_in_a_kernel(x, rate, box):
    y = writes_work(x, rate)
    box.acc.add_(y)
    factor = torch.linalg.cholesky(torch.outer(y, y) + box.shift)
    return writes_tail(factor.diagonal(), rate) + box.acc.sum()


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmWrites(TestCase):
    """A write to something that outlives the call (an attribute, an item, a
    global, a cell, a tensor in place) is made on every call, in the order the
    function makes it and with the value that call computes."""

    hw_classification = HardwareClassification.ACCELERATOR

    def outcome(self, fn, *args):
        try:
            return fn(*args)
        except (ValueError, IndexError, RuntimeError) as e:
            return type(e).__name__

    def compare(self, fn, device, mine, theirs, read, eager=None, before=None):
        # Nine calls with three rates in turn: regions form on the fourth.
        g = vm(fn, passes=EARLY)
        x = torch.randn(4, device=device)
        for call in range(9):
            rate = (0.5, 0.75, 1.0)[call % 3]
            if before is not None:
                before(call, mine, theirs)
            got = self.outcome(g, x, rate, mine)
            want = self.outcome(eager or fn, x, rate, theirs)
            self.assertEqual(got, want, atol=0, rtol=0, msg=f"call {call}")
            ours, expected = read(mine), read(theirs)
            self.assertEqual(ours, expected, atol=0, rtol=0, msg=f"call {call}")
            if isinstance(expected, float):
                self.assertIs(type(ours), float, msg=f"call {call}")

    @isolated
    def test_attribute_of_an_object(self, device):
        for holder in (WritesBox, WritesSlotted):
            self.compare(
                writes_attribute, device, holder(), holder(), lambda box: box.rate
            )

    @isolated
    def test_attribute_of_a_module(self, device):
        mine, theirs = WritesModule(device), WritesModule(device)
        self.compare(writes_module_attribute, device, mine, theirs, lambda m: m.steps)

    @isolated
    def test_buffer_of_a_module_that_is_bound_anew(self, device):
        mine, theirs = WritesModule(device), WritesModule(device)
        self.compare(writes_module_buffer, device, mine, theirs, lambda m: m.seen)

    @isolated
    def test_parameter_of_a_module_written_in_place(self, device):
        mine, theirs = WritesModule(device), WritesModule(device)
        self.compare(writes_module_parameter, device, mine, theirs, lambda m: m.weight)

    @isolated
    def test_item_of_a_list(self, device):
        self.compare(
            writes_item, device, [0.0, 1.0], [0.0, 1.0], lambda items: items[0]
        )

    @isolated
    def test_item_of_a_dict(self, device):
        self.compare(
            writes_item,
            device,
            {"other": 1.0},
            {"other": 1.0},
            lambda items: items["rate"],
        )

    @isolated
    def test_global(self, device):
        # Eager's calls write the same global: each side keeps its own total.
        totals = {"mine": 0.0, "theirs": 0.0}

        def eager(x, rate, _):
            global WRITES_TOTAL
            totals["mine"], WRITES_TOTAL = WRITES_TOTAL, totals["theirs"]
            try:
                return writes_global(x, rate, None)
            finally:
                totals["theirs"], WRITES_TOTAL = WRITES_TOTAL, totals["mine"]

        self.compare(
            writes_global,
            device,
            "mine",
            "theirs",
            lambda side: WRITES_TOTAL if side == "mine" else totals["theirs"],
            eager=eager,
        )

    @isolated
    def test_closure_cell(self, device):
        (mine, read_mine), (theirs, read_theirs) = writes_cell(), writes_cell()
        self.compare(
            mine, device, read_mine, read_theirs, lambda read: read(None), eager=theirs
        )

    @isolated
    def test_tensor_written_in_place_through_an_object(self, device):
        mine, theirs = WritesBox(), WritesBox()
        mine.acc, theirs.acc = (
            torch.zeros(4, device=device),
            torch.zeros(4, device=device),
        )
        self.compare(writes_in_place, device, mine, theirs, lambda box: box.acc)

    @isolated
    def test_store_inside_a_loop(self, device):
        self.compare(
            writes_in_a_loop, device, WritesBox(), WritesBox(), lambda box: box.rate
        )

    @isolated
    def test_store_in_front_of_a_raise(self, device):
        # The third rate makes the function raise behind its store.
        self.compare(
            writes_then_raises, device, WritesBox(), WritesBox(), lambda box: box.rate
        )

    @isolated
    def test_write_in_place_in_front_of_a_kernel_that_raises(self, device):
        # On the seventh call the matrix is not positive definite: the kernel
        # behind the write in place raises, in eager mode and in a region
        # alike, and the tensor was written once.
        def before(call, *boxes):
            for box in boxes:
                box.shift = torch.eye(4, device=device) * (
                    -100.0 if call == 6 else 10.0
                )

        mine, theirs = WritesBox(), WritesBox()
        mine.acc, theirs.acc = (
            torch.zeros(4, device=device),
            torch.zeros(4, device=device),
        )
        self.compare(
            writes_then_fails_in_a_kernel,
            device,
            mine,
            theirs,
            lambda box: box.acc,
            before=before,
        )


instantiate_device_type_tests(TestVmWrites, globals())


CHANGES_FLAG = True


def changes_pick(y):
    if CHANGES_FLAG:
        return y * 2
    return y * 3


def changes_calls(x):
    return changes_pick(x.sin() + 1).tanh().cos().exp()


def changes_graded(y):
    if torch.is_grad_enabled():
        return y * 2
    return y * 3


def changes_calls_graded(x):
    return changes_graded(x.sin() + 1).tanh().cos().exp()


def changes_casts(x, w):
    return (torch.mm(torch.mm(x, w), w).relu() * 2 + 1).tanh()


def changes_closure():
    factor = 2.0

    def inner(y):
        return y * factor

    def outer(x):
        return inner(x.sin() + 1).tanh().cos().exp()

    def set_factor(value):
        nonlocal factor
        factor = value

    return outer, set_factor


class ChangesNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)
        self.norm = torch.nn.BatchNorm1d(8)
        self.drop = torch.nn.Dropout(0.5)
        self.scale = 2.0

    def forward(self, x):
        return self.drop(self.norm(torch.tanh(self.lin(x)))) * self.scale


def changes_forward(module, x):
    return module(x)


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmCalleeState(TestCase):
    """State that a callee reads between tensor operations, changed between
    calls after regions have formed: a region is entered only while what its
    trace read is as it was."""

    hw_classification = HardwareClassification.ACCELERATOR

    def module_calls(self, device, change):
        # Eleven calls of a module with dropout and batch norm, the same
        # random draws on both sides; `change(call, module)` before each.
        torch.manual_seed(0)
        mine = ChangesNet().to(device)
        theirs = copy.deepcopy(mine)
        g = vm(changes_forward, passes=EARLY)
        for call in range(11):
            change(call, mine)
            change(call, theirs)
            x = torch.randn(4, 8, device=device)
            torch.manual_seed(call)
            got = g(mine, x)
            torch.manual_seed(call)
            want = changes_forward(theirs, x)
            self.assertEqual(got, want, atol=0, rtol=0, msg=f"call {call}")
            self.assertEqual(
                mine.norm.running_mean,
                theirs.norm.running_mean,
                atol=0,
                rtol=0,
                msg=f"call {call}",
            )

    # The region traced in training mode has the dropout and the batch
    # statistics: evaluation mode gets a region of its own.
    @isolated
    def test_module_switched_between_train_and_eval(self, device):
        self.module_calls(
            device, lambda call, module: module.train(call < 5 or call >= 8)
        )

    @isolated
    def test_hyperparameter_set_from_outside(self, device):
        self.module_calls(
            device,
            lambda call, module: setattr(module, "scale", 5.0 if call >= 6 else 2.0),
        )

    @isolated
    def test_callee_that_branches_on_a_global(self, device):
        global CHANGES_FLAG
        g = vm(changes_calls, passes=EARLY)
        x = torch.randn(4, device=device)
        for call in range(9):
            CHANGES_FLAG = call < 6
            self.assertEqual(g(x), changes_calls(x), atol=0, rtol=0, msg=f"call {call}")

    @isolated
    def test_callee_that_branches_on_grad_mode(self, device):
        g = vm(changes_calls_graded, passes=EARLY)
        x = torch.randn(4, device=device)
        for call in range(9):
            with torch.set_grad_enabled(call < 6):
                self.assertEqual(
                    g(x), changes_calls_graded(x), atol=0, rtol=0, msg=f"call {call}"
                )

    # The dtype autocast casts to is thread state no instruction reads: a
    # region traced under bfloat16 is not the one for float16.
    @isolated
    def test_autocast_dtype_changed(self, device):
        g = vm(changes_casts, passes=EARLY)
        x, w = torch.randn(4, 8, device=device), torch.randn(8, 8, device=device)
        for call in range(9):
            with torch.autocast(
                torch.device(device).type,
                dtype=torch.bfloat16 if call < 6 else torch.float16,
            ):
                got, want = g(x, w), changes_casts(x, w)
            self.assertEqual(got.dtype, want.dtype, msg=f"call {call}")
            self.assertEqual(got, want, atol=0, rtol=0, msg=f"call {call}")

    @isolated
    def test_closure_variable_bound_anew(self, device):
        (mine, set_mine), (theirs, set_theirs) = changes_closure(), changes_closure()
        g = vm(mine, passes=EARLY)
        x = torch.randn(4, device=device)
        for call in range(9):
            if call == 6:
                set_mine(3.0)
                set_theirs(3.0)
            self.assertEqual(g(x), theirs(x), atol=0, rtol=0, msg=f"call {call}")


instantiate_device_type_tests(TestVmCalleeState, globals())


READS_FACTOR = 2
READS_HELPER = torch.sin
READS_TABLE = {"factor": 2.0}
READS_LIST = [2.0, 3.0]
READS_STAGES = [torch.sin]
READS_HOOKS = []
READS_MODULE = types_module.ModuleType("reads_module")
READS_MODULE.factor = 2.0


class ReadsConfig:
    bias = 1.0

    def __init__(self):
        self.factor = 2.0
        self.flip = False

    @property
    def doubled(self):
        return self.factor * 2

    def apply(self, y):
        return y * 2


class ReadsSlotted:
    __slots__ = ("factor",)

    def __init__(self):
        self.factor = 2.0


class ReadsNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)
        self.act = torch.nn.Tanh()
        self.stages = [torch.nn.Sigmoid()]

    def forward(self, y):
        y = self.act(self.lin(y))
        for stage in self.stages:
            y = stage(y)
        return y


READS_CONFIG = ReadsConfig()
READS_SLOTTED = ReadsSlotted()


def reads_through(inner):
    """A function with one region that is traced through a call of `inner`."""

    def outer(x):
        return inner(x.sin() + 1).tanh().cos().exp()

    return outer


def reads_apply_other(self, y):
    return y * 3


def region_runs():
    """How many times a region has run (counted by torch._rust_inductor once
    its COUNT is set)."""
    import torch._rust_inductor as ri

    return sum(
        n for name, n in ri.counts.items() if not name.endswith((" MISS", " CHECK"))
    )


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmReads(TestCase):
    """One test per kind of read a traced call makes of something that
    outlives it. The state is changed and changed back between calls; every
    call gives what eager gives, and each state has a region of its own (the
    regions are neither run with the other state nor given up)."""

    hw_classification = HardwareClassification.ACCELERATOR

    def alternating(self, device, fn, switch, eager=None):
        # Thirteen calls: five in the first state (the fourth is traced, the
        # fifth runs the region), three in the second, three in the first
        # again, two in the second again. `switch(second)` sets the state.
        import torch._rust_inductor as ri

        ri.COUNT = True
        eager = eager or fn
        g = vm(fn, passes=EARLY)
        runs, traced, call = [], [], 0
        for second, calls in [(False, 5), (True, 3), (False, 3), (True, 2)]:
            switch(second)
            for _ in range(calls):
                x = torch.randn(4, 8, device=device, dtype=torch.float32)
                before = region_runs()
                got, want = g(x), eager(x)
                self.assertEqual(got.dtype, want.dtype, msg=f"call {call}")
                self.assertEqual(got, want, atol=0, rtol=0, msg=f"call {call}")
                runs.append(region_runs() - before)
                call += 1
            traced.append(ri.stats["traced"])
        # Each state has its region, the same one when the state comes back.
        self.assertGreater(runs[4], 0)
        self.assertGreater(runs[7], 0)
        self.assertEqual((runs[10], runs[12]), (runs[4], runs[7]))
        self.assertEqual(traced[3], traced[1])

    @isolated
    def test_global_scalar(self, device):
        def switch(second):
            global READS_FACTOR
            READS_FACTOR = 5 if second else 2

        self.alternating(device, reads_through(lambda y: y * READS_FACTOR), switch)

    @isolated
    def test_global_function(self, device):
        def switch(second):
            global READS_HELPER
            READS_HELPER = torch.cos if second else torch.sin

        self.alternating(device, reads_through(lambda y: READS_HELPER(y)), switch)

    @isolated
    def test_attribute_of_an_object(self, device):
        self.alternating(
            device,
            reads_through(lambda y: y * READS_CONFIG.factor),
            lambda second: setattr(READS_CONFIG, "factor", 5.0 if second else 2.0),
        )

    @isolated
    def test_attribute_a_branch_is_taken_on(self, device):
        self.alternating(
            device,
            reads_through(lambda y: y * 3 if READS_CONFIG.flip else y - 1),
            lambda second: setattr(READS_CONFIG, "flip", second),
        )

    @isolated
    def test_attribute_of_a_class(self, device):
        self.alternating(
            device,
            reads_through(lambda y: y + READS_CONFIG.bias),
            lambda second: setattr(ReadsConfig, "bias", 3.0 if second else 1.0),
        )

    @isolated
    def test_attribute_of_a_class_hidden_by_the_instance(self, device):
        def switch(second):
            if second:
                READS_CONFIG.bias = 7.0
            else:
                vars(READS_CONFIG).pop("bias", None)

        self.alternating(device, reads_through(lambda y: y + READS_CONFIG.bias), switch)

    @isolated
    def test_attribute_behind_a_property(self, device):
        self.alternating(
            device,
            reads_through(lambda y: y * READS_CONFIG.doubled),
            lambda second: setattr(READS_CONFIG, "factor", 5.0 if second else 2.0),
        )

    @isolated
    def test_attribute_in_a_slot(self, device):
        self.alternating(
            device,
            reads_through(lambda y: y * READS_SLOTTED.factor),
            lambda second: setattr(READS_SLOTTED, "factor", 5.0 if second else 2.0),
        )

    @isolated
    def test_attribute_of_a_python_module(self, device):
        self.alternating(
            device,
            reads_through(lambda y: y * READS_MODULE.factor),
            lambda second: setattr(READS_MODULE, "factor", 5.0 if second else 2.0),
        )

    @isolated
    def test_attribute_asked_with_getattr(self, device):
        def switch(second):
            if second:
                READS_CONFIG.extra = 4.0
            else:
                vars(READS_CONFIG).pop("extra", None)

        self.alternating(
            device,
            reads_through(lambda y: y * getattr(READS_CONFIG, "extra", 1.5)),
            switch,
        )
        self.alternating(
            device,
            reads_through(lambda y: y * 3 if hasattr(READS_CONFIG, "extra") else y - 1),
            switch,
        )

    @isolated
    def test_method_replaced_on_the_class(self, device):
        first = ReadsConfig.apply
        self.alternating(
            device,
            reads_through(lambda y: READS_CONFIG.apply(y)),
            lambda second: setattr(
                ReadsConfig, "apply", reads_apply_other if second else first
            ),
        )

    @isolated
    def test_forward_replaced_on_the_instance(self, device):
        torch.manual_seed(0)
        net = ReadsNet().to(device)

        def other(y):
            return y * 3

        def switch(second):
            if second:
                net.forward = other
            else:
                vars(net).pop("forward", None)

        self.alternating(device, lambda x: net(x.sin() + 1).tanh().cos().exp(), switch)

    @isolated
    def test_submodule_replaced(self, device):
        torch.manual_seed(0)
        net = ReadsNet().to(device)
        first, other = net.act, torch.nn.Softsign()
        self.alternating(
            device,
            lambda x: net(x.sin() + 1).cos().exp(),
            lambda second: setattr(net, "act", other if second else first),
        )

    @isolated
    def test_list_of_layers_that_grows(self, device):
        torch.manual_seed(0)
        net = ReadsNet().to(device)
        more = torch.nn.Softsign()

        def switch(second):
            if second:
                net.stages.append(more)
            elif len(net.stages) > 1:
                net.stages.pop()

        self.alternating(device, lambda x: net(x.sin() + 1).cos().exp(), switch)

    @isolated
    def test_item_of_a_dict(self, device):
        self.alternating(
            device,
            reads_through(lambda y: y * READS_TABLE["factor"]),
            lambda second: READS_TABLE.update(factor=5.0 if second else 2.0),
        )

    @isolated
    def test_item_of_a_dict_that_comes_and_goes(self, device):
        def switch(second):
            if second:
                READS_TABLE["extra"] = 4.0
            else:
                READS_TABLE.pop("extra", None)

        self.alternating(
            device, reads_through(lambda y: y * READS_TABLE.get("extra", 1.5)), switch
        )
        self.alternating(
            device,
            reads_through(lambda y: y * 3 if "extra" in READS_TABLE else y - 1),
            switch,
        )

    @isolated
    def test_item_of_a_list(self, device):
        self.alternating(
            device,
            reads_through(lambda y: y * READS_LIST[0] + READS_LIST[-1]),
            lambda second: READS_LIST.__setitem__(0, 5.0 if second else 2.0),
        )

    @isolated
    def test_list_that_is_iterated(self, device):
        def inner(y):
            for stage in READS_STAGES:
                y = stage(y)
            return y

        def switch(second):
            if second:
                READS_STAGES.append(torch.cos)
            elif len(READS_STAGES) > 1:
                READS_STAGES.pop()

        self.alternating(device, reads_through(inner), switch)

    @isolated
    def test_length_and_truth_of_a_list(self, device):
        def switch(second):
            if second:
                READS_HOOKS.append(None)
            else:
                READS_HOOKS.clear()

        self.alternating(
            device, reads_through(lambda y: y * (len(READS_HOOKS) + 2)), switch
        )
        self.alternating(
            device, reads_through(lambda y: y * 3 if READS_HOOKS else y - 1), switch
        )

    @isolated
    def test_closure_variable(self, device):
        (mine, set_mine), (theirs, set_theirs) = changes_closure(), changes_closure()

        def switch(second):
            set_mine(3.0 if second else 2.0)
            set_theirs(3.0 if second else 2.0)

        self.alternating(device, mine, switch, eager=theirs)

    @isolated
    def test_default_dtype(self, device):
        def switch(second):
            torch.set_default_dtype(torch.float64 if second else torch.float32)

        try:
            self.alternating(
                device,
                reads_through(lambda y: y + torch.ones(8, device=y.device)),
                switch,
            )
        finally:
            torch.set_default_dtype(torch.float32)

    @isolated
    def test_autocast_switched_on_and_off(self, device):
        kind = torch.device(device).type
        w = torch.randn(8, 8, device=device)
        state = contextlib.ExitStack()

        def switch(second):
            state.close()
            if second:
                state.enter_context(torch.autocast(kind, dtype=torch.bfloat16))

        with state:
            self.alternating(
                device,
                lambda x: (torch.mm(torch.mm(x, w), w).relu() * 2 + 1).tanh(),
                switch,
            )

    @isolated
    def test_gradient_of_a_tensor(self, device):
        # What autograd keeps on a tensor is not what a region is traced for:
        # the block that reads it stays instructions.
        weight = torch.randn(8, device=device, requires_grad=True)
        fn = reads_through(lambda y: y * weight.grad)
        g = vm(fn, passes=EARLY)
        for call in range(9):
            weight.grad = torch.randn(8, device=device)
            x = torch.randn(4, 8, device=device)
            self.assertEqual(g(x), fn(x), atol=0, rtol=0, msg=f"call {call}")

    @isolated
    def test_function_the_vm_was_told_to_skip(self, device):
        # CPython runs it, so nothing sees what it reads: it is in no region.
        def skipped(y):
            return y * READS_FACTOR

        def switch(second):
            global READS_FACTOR
            READS_FACTOR = 5 if second else 2

        fn = reads_through(skipped)
        g = vm(fn, passes=(*EARLY, vm.passes.Skip(skipped)))
        for call in range(9):
            switch(3 <= call < 6)
            x = torch.randn(4, 8, device=device)
            self.assertEqual(g(x), fn(x), atol=0, rtol=0, msg=f"call {call}")

    @isolated
    def test_value_that_changes_on_every_call(self, device):
        # A float a callee reads is part of what a region was traced for: one
        # that never comes back never finds its region again, and the
        # instructions run. Every call still gives what eager gives.
        g = vm(reads_through(lambda y: y * READS_CONFIG.factor), passes=EARLY)
        eager = reads_through(lambda y: y * READS_CONFIG.factor)
        x = torch.randn(4, 8, device=device)
        for call in range(12):
            READS_CONFIG.factor = 0.25 * (call + 1)
            self.assertEqual(g(x), eager(x), atol=0, rtol=0, msg=f"call {call}")


instantiate_device_type_tests(TestVmReads, globals())


SWAP_OPS = ["pow", "mean", "add", "rsqrt", "mul"]


def swap_model(x, w):
    h = torch.tanh(x @ w)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    return (h.relu() + h.sigmoid()).sum(-1)


def swap_rms_norm(h):
    return h / h.square().mean(-1, keepdim=True).add(1e-6).sqrt()


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmSwap(TestCase):
    """vm.swap: a kernel in the place of a run of ops, checked against them."""

    hw_classification = HardwareClassification.ACCELERATOR

    def run_swapped(self, kernel, device, calls=8, **options):
        # What each call gives (or the exception it raises), and eager's.
        g = vm(swap_model, passes=(*EARLY, vm.swap(SWAP_OPS, kernel, **options)))
        torch.manual_seed(0)
        w = torch.randn(8, 8, device=device)
        got = []
        for _ in range(calls):
            x = torch.randn(4, 8, device=device)
            try:
                got.append((g(x, w), swap_model(x, w)))
            except vm.Mismatch as e:
                got.append(e)
        return got

    @isolated
    def test_kernel_runs_in_place_of_the_ops(self, device):
        calls = []

        def kernel(h):
            calls.append(tuple(h.shape))
            return swap_rms_norm(h)

        for ours, eagers in self.run_swapped(kernel, device):
            self.assertEqual(ours, eagers, rtol=1e-4, atol=1e-5)
        # The fourth call is traced; the kernel runs from the fifth on.
        self.assertEqual(calls, [(4, 8)] * 4)

    @isolated
    def test_kernel_that_gives_other_values_raises(self, device):
        got = self.run_swapped(lambda h: swap_rms_norm(h) * 1.5, device)
        self.assertNotIsInstance(got[3], Exception)
        self.assertIsInstance(got[4], vm.Mismatch)
        self.assertRegex(
            str(got[4]),
            r"result 0 differs by up to .* on call 1 \(the ops it replaced were run on the same inputs\)",
        )

    @isolated
    def test_kernel_that_gives_another_shape_or_dtype_raises(self, device):
        got = self.run_swapped(lambda h: swap_rms_norm(h)[:, :4], device)
        self.assertRegex(str(got[4]), r"result 0 has shape \(4, 4\), not \(4, 8\)")
        got = self.run_swapped(lambda h: swap_rms_norm(h).double(), device)
        self.assertRegex(
            str(got[4]), "result 0 is torch.float64 on .*, not torch.float32"
        )

    @isolated
    def test_the_check_stops_after_as_many_calls_as_asked(self, device):
        # Exact for the first two calls of the region, off by a factor after.
        seen = []

        def kernel(h):
            seen.append(len(seen))
            return swap_rms_norm(h) * (1.0 if len(seen) <= 2 else 1.5)

        got = self.run_swapped(kernel, device, check=2)
        self.assertFalse(any(isinstance(g, Exception) for g in got))
        for ours, eagers in got[:6]:
            self.assertEqual(ours, eagers, rtol=1e-4, atol=1e-5)
        self.assertFalse(torch.allclose(got[6][0], got[6][1], rtol=1e-4, atol=1e-5))

    @isolated
    def test_unchecked_kernel_is_trusted(self, device):
        got = self.run_swapped(lambda h: swap_rms_norm(h) * 1.5, device, check=False)
        self.assertFalse(any(isinstance(g, Exception) for g in got))
        self.assertFalse(torch.allclose(got[4][0], got[4][1], rtol=1e-4, atol=1e-5))

    @isolated
    def test_run_that_writes_in_place_is_left_as_it_is(self, device):
        def writes(x, w):
            h = torch.tanh(x @ w)
            h = h.pow(2).mean(-1, keepdim=True).add_(1e-6).rsqrt() * h
            return (h.relu() + h.sigmoid()).sum(-1)

        calls = []

        def kernel(*values):
            calls.append(len(values))
            return values[0]

        g = vm(
            writes,
            passes=(*EARLY, vm.swap(["pow", "mean", "add_", "rsqrt", "mul"], kernel)),
        )
        w = torch.randn(8, 8, device=device)
        for _ in range(8):
            x = torch.randn(4, 8, device=device)
            self.assertEqual(g(x, w), writes(x, w), atol=0, rtol=0)
        self.assertEqual(calls, [])

    @isolated
    def test_gradients_go_through_the_kernel(self, device):
        g = vm(swap_model, passes=(*EARLY, vm.swap(SWAP_OPS, swap_rms_norm)))
        w = torch.randn(8, 8, device=device, requires_grad=True)
        for _ in range(8):
            x = torch.randn(4, 8, device=device)
            (ours,) = torch.autograd.grad(g(x, w).sum(), w)
            (eagers,) = torch.autograd.grad(swap_model(x, w).sum(), w)
            self.assertEqual(ours, eagers, rtol=1e-4, atol=1e-5)


instantiate_device_type_tests(TestVmSwap, globals())


def rust_adds(x, w):
    return ((x @ w).sin() + x).cos().tanh().exp()


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmRustPasses(TestCase):
    """Passes written in Rust on the crate torch/_rust/sdk, loaded from the
    library cargo builds from test/vm_passes."""

    hw_classification = HardwareClassification.GENERIC

    def library(self):
        cargo = shutil.which("cargo")
        if cargo is None:
            self.skipTest("cargo is not installed")
        here = os.path.dirname(os.path.abspath(__file__))
        manifest = os.path.join(here, "vm_passes", "Cargo.toml")
        # One build for the tests of a run (each runs in a process of its own).
        target = os.path.join(tempfile.gettempdir(), f"vm_passes_{os.getuid()}")
        command = [cargo, "build", "--release", "--offline", "--locked"]
        command += ["--manifest-path", manifest, "--target-dir", target]
        subprocess.run(command, check=True, capture_output=True)
        (library,) = (
            os.path.join(target, "release", name)
            for name in os.listdir(os.path.join(target, "release"))
            if name.startswith("libvm_passes.") and not name.endswith(".d")
        )
        return library

    def run_with(self, fn, block_pass, calls=8):
        # What each call gives, eager's, and the warnings of the run.
        g = vm(fn, passes=(*EARLY, block_pass))
        torch.manual_seed(0)
        w = torch.randn(8, 8)
        got = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(calls):
                x = torch.randn(4, 8)
                got.append((g(x, w), fn(x, w)))
        return got, [str(w.message) for w in caught]

    @isolated
    def test_pass_puts_a_kernel_in(self):
        passes = vm.load_passes(self.library())
        self.assertEqual(sorted(passes), ["fuse_rmsnorm", "panics", "strict"])
        calls = []

        def kernel(h):
            calls.append(tuple(h.shape))
            return swap_rms_norm(h)

        fuse = passes["fuse_rmsnorm"].bind(kernel)
        self.assertEqual(fuse.__name__, "fuse_rmsnorm")
        self.assertEqual(vm.kind(fuse), "block")
        got, said = self.run_with(swap_model, fuse)
        for ours, eagers in got:
            self.assertEqual(ours, eagers, rtol=1e-4, atol=1e-5)
        # The fourth call is traced; the kernel runs from the fifth on.
        self.assertEqual(calls, [(4, 8)] * 4)
        self.assertEqual(said, [])

    @isolated
    def test_kernel_a_pass_puts_in_is_checked(self):
        passes = vm.load_passes(self.library())
        wrong = passes["fuse_rmsnorm"].bind(lambda h: swap_rms_norm(h) + 1e-2)
        with self.assertRaisesRegex(vm.Mismatch, "differs by up to 0.01"):
            self.run_with(swap_model, wrong)

    @isolated
    def test_pass_that_fails_leaves_the_region_as_it_was(self):
        passes = vm.load_passes(self.library())
        # `strict` fails where a region multiplies; an unbound pass has no
        # kernel; a panic is an error like another. Each is skipped for the
        # region with a warning that says why, and the results are eager's.
        for name, why in [
            ("strict", "a block that multiplies: aten::"),
            ("fuse_rmsnorm", "needs a kernel"),
            ("panics", "a pass panicked"),
        ]:
            got, said = self.run_with(swap_model, passes[name])
            for ours, eagers in got:
                self.assertEqual(ours, eagers, atol=0, rtol=0)
            self.assertTrue(
                any(why in text and "RuntimeError" in text for text in said),
                msg=f"{name}: {said}",
            )
        got, said = self.run_with(rust_adds, passes["strict"])
        for ours, eagers in got:
            self.assertEqual(ours, eagers, atol=0, rtol=0)
        self.assertEqual(said, [])

    def test_library_that_is_none_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "cannot load"):
            vm.load_passes("/nowhere/libpasses.so")
        with self.assertRaisesRegex(RuntimeError, "exports no passes"):
            vm.load_passes(torch._C.__file__)

    def test_library_of_another_interface_version_is_refused(self):
        library = self.library()
        with unittest.mock.patch.object(torch.vm._native, "ABI_VERSION", 2):
            with self.assertRaisesRegex(
                RuntimeError,
                "version 1 of the pass interface; this torch.vm has version 2",
            ):
                vm.load_passes(library)


def reasons_prints(x):
    y = x.sin() + 1
    print(end="")
    return (y.tanh().cos() * 2).exp()


def reasons_small(x):
    return x.sin().cos()


def reasons_branches(x):
    y = x.sin() + 1
    if (y > 0).any():
        y = y * 2
    return y.tanh().cos().exp()


def reasons_leaves(x, n, stop):
    for i in range(n):
        x = (x.sin() * 2 + 1).tanh().cos()
        if i == stop:
            x = x - 1
            break
    return x.exp()


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmRegionReasons(TestCase):
    """vm.regions: where the tensor work runs as a region and, as data, why
    the rest does not."""

    hw_classification = HardwareClassification.GENERIC

    def rows(self, fn, *args, calls=8, passes=EARLY, before=None):
        g = vm(fn, passes=passes)
        for call in range(calls):
            if before is not None:
                args = before(call) or args
            g(*args)
        rows = vm.regions(g)
        for row in rows:
            self.assertIn(
                row["state"],
                ("region", "too small", "refused", "gave up", "not movable"),
            )
            self.assertEqual(row["why"] is None, row["state"] == "region", msg=str(row))
        return rows

    @isolated
    def test_region_and_the_calls_it_turned_down(self):
        def before(call):
            global READS_FACTOR
            READS_FACTOR = 5 if call >= 5 else 2

        fn = reads_through(lambda y: y * READS_FACTOR)
        (row,) = (
            r
            for r in self.rows(fn, torch.randn(4, 8), before=before)
            if r["state"] == "region"
        )
        self.assertEqual(
            (row["kind"], row["function"]), ("block", "reads_through.<locals>.outer")
        )
        self.assertEqual(row["lines"], (fn.__code__.co_firstlineno + 1,) * 2)
        self.assertEqual(
            (row["ops"], row["inputs"], row["constants"], row["variants"]), (6, 1, 0, 2)
        )
        # Call 4 runs the first region; 5 finds the global changed (a miss)
        # and is traced again; 6 and 7 run the second, which is tried first.
        self.assertEqual(row["runs"], 3)
        self.assertEqual(row["misses"], {"read: a namespace or a container": 1})
        self.assertEqual(row["traces"], [("region", None)] * 2)
        self.assertGreater(row["reads"], 0)

    @isolated
    def test_block_with_too_few_operations(self):
        (row,) = self.rows(reasons_small, torch.randn(4, 8))
        self.assertEqual(
            (row["state"], row["ops"], row["why"]),
            ("too small", 2, "2 ops, fewer than 4"),
        )
        self.assertEqual(row["function"], "reasons_small")

    @isolated
    def test_trace_that_reads_what_no_check_covers(self):
        weight = torch.randn(8, requires_grad=True)
        weight.grad = torch.randn(8)
        rows = self.rows(reads_through(lambda y: y * weight.grad), torch.randn(4, 8))
        (row,) = (r for r in rows if r["state"] == "refused")
        self.assertIn("it reads a tensor's .grad", row["why"])
        self.assertEqual(row["kind"], "block")

    @isolated
    def test_call_that_is_not_movable(self):
        rows = self.rows(reasons_prints, torch.randn(4, 8))
        (row,) = (r for r in rows if r["state"] == "not movable")
        self.assertEqual((row["kind"], row["function"]), ("call", "reasons_prints"))
        self.assertIn("called <built-in function print>", row["why"])
        line = reasons_prints.__code__.co_firstlineno + 2
        self.assertEqual(
            (row["lines"], row["where"]), ((line, line), f"reasons_prints line {line}")
        )

    @isolated
    def test_cfg_pass_is_told_why_a_call_is_opaque(self):
        said = {}

        def look(cfg: vm.Cfg) -> None:
            for call in cfg.opaque_calls():
                said[(cfg.qualname, call.line)] = cfg.why_opaque(call)
            for call in cfg.calls():
                if call.movable:
                    self.assertIsNone(cfg.why_opaque(call))

        g = vm(reasons_prints, passes=(*EARLY, look))
        for _ in range(8):
            g(torch.randn(4, 8))
        line = reasons_prints.__code__.co_firstlineno + 2
        self.assertIn(
            "called <built-in function print>", said[("reasons_prints", line)]
        )
        # vm.explain lists what is in no region after the regions.
        text = vm.explain(g)
        self.assertIn("\nnot in a region:\n", text)
        self.assertIn(f"  reasons_prints line {line}: not movable: ", text)

    @isolated
    def test_branch_on_tensor_data(self):
        rows = self.rows(reasons_branches, torch.randn(4, 8))
        why = [
            r["why"]
            for r in rows
            if r["state"] == "not movable" and r["function"] == "reasons_branches"
        ]
        self.assertTrue(any(" ran aten::" in w for w in why), msg=str(rows))

    @isolated
    def test_span_that_is_left_by_another_way(self):
        # The traced call (the fourth) leaves the loop by a block that no
        # earlier call ran.
        rows = self.rows(
            reasons_leaves,
            torch.randn(8),
            3,
            -1,
            before=lambda call: (torch.randn(8), 3, 1 if call == 3 else -1),
        )
        (row,) = (r for r in rows if r["kind"] == "span")
        self.assertEqual((row["function"], row["lines"]), ("reasons_leaves", None))
        # The loop's first round was traced into a region; the second round
        # is the one that leaves.
        self.assertEqual([state for state, _ in row["traces"]], ["region", "gave up"])
        self.assertIn(
            "the span left by another way or ran a block never seen run",
            row["traces"][1][1],
        )
        self.assertEqual(row["state"], "region")

    @isolated
    def test_pass_that_raises_is_in_the_notes(self):
        def broken(gm, input_types):
            raise ValueError("no")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows = self.rows(
                reads_through(lambda y: y * 2),
                torch.randn(4, 8),
                passes=(*EARLY, broken),
            )
        (row,) = (r for r in rows if r["state"] == "region")
        self.assertTrue(
            any("raised ValueError: no" in note for note in row["notes"]), msg=str(row)
        )


class TestVmWithoutRuntime(TestCase):
    """A build of PyTorch that skipped the torch._rust crate (another Python
    than 3.12, another system than Linux, no cargo)."""

    hw_classification = HardwareClassification.GENERIC

    def test_error_names_the_reason(self):
        # In a process where the import of torch._rust fails.
        script = """
import sys


class Hidden:
    def find_spec(self, name, path=None, target=None):
        if name == "torch._rust":
            raise ImportError("hidden")


sys.meta_path.insert(0, Hidden())
import torch

assert torch._rust is None
try:
    torch.vm(lambda x: x + 1)
except RuntimeError as e:
    print(e)
"""
        ran = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
        )
        self.assertEqual(ran.returncode, 0, msg=ran.stderr[-2000:])
        self.assertIn(
            "torch.vm needs the torch._rust extension, which this build of PyTorch does not have",
            ran.stdout,
        )


def call_outcome(fn, *args, **kwargs):
    """What a call returns, or the type and the message of what it raises."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return type(e), str(e)


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


def unr_loop(x, y):
    total = x
    for i in range(3):
        total = total * 2 + y
        if i == 1:
            total = total - 1
    return total, i


def exc_double(x):
    if x < 0:
        raise ValueError("negative")
    return x * 2


class ExcManager:
    def __init__(self, log, suppress=False):
        self.log, self.suppress = log, suppress

    def __enter__(self):
        self.log.append("enter")
        return 7

    def __exit__(self, kind, value, tb):
        self.log.append(("exit", kind.__name__ if kind else None))
        return self.suppress


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


def frame_new_classes():
    pair = collections.namedtuple("pair", "a b")
    plain = type("plain", (), {})
    return pair.__module__, plain.__module__


def frame_missing_global():
    return frame_surely_not_defined  # noqa: F821


def frame_unbound_local(flag):
    if flag:
        value = 1
    return value


def breaker_spins(flag, limit):
    n = 0
    while not flag and n < limit:
        n += 1
    return n


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


class RegionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.fc = torch.nn.Linear(4 * 6 * 6, 5)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        return self.fc(torch.tanh(x.flatten(1)))


def region_infer(model, x):
    return model(x)


def region_step_plain(model, opt, x):
    opt.zero_grad()
    loss = model(x).pow(2).mean()
    loss.backward()
    opt.step()
    return loss


class RegionRecurrent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(6, 8, num_layers=2, bidirectional=True)
        self.out = torch.nn.Linear(16, 3)

    def forward(self, x):
        y, _ = self.lstm(x)
        return self.out(y).tanh()


def graph_mixed(x, w):
    return torch.tanh(x @ w).sum() + (x * 2).mean()


def graph_normed(x, mean, var, training):
    y = torch.batch_norm(x * 2, None, None, mean, var, training, 0.1, 1e-5, True)
    return (torch.relu(y) + 1).mean((2, 3)).sum()


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


def graph_collects(xs):
    out = []
    for x in xs:
        out.append(x.sin() * 2)
    return torch.cat(out).tanh().sum()


def graph_casts(x, w):
    return (torch.mm(torch.mm(x, w), w).relu() * 2 + 1).tanh()


class GraphLooksUp:
    """A class that reads its attributes in a function of its own, which
    goes to the one of the class behind it, which goes to `object`'s."""

    def __getattribute__(self, key):
        spec = super().__getattribute__("__dict__").get("_spec")
        if spec is not None:
            raise AttributeError(key)
        return super().__getattribute__(key)


class GraphSettings(GraphLooksUp):
    renamed = {"width": "hidden"}

    def __init__(self):
        self.hidden = 2.0
        self._kind = "plain"

    @property
    def kind(self):
        return self._kind

    def __getattribute__(self, key):
        if key != "renamed" and key in super().__getattribute__("renamed"):
            key = super().__getattribute__("renamed")[key]
        return super().__getattribute__(key)


class GraphSettingsWithDefault(GraphSettings):
    def __getattr__(self, key):
        return 3.0


class GraphDoubles:
    """What is read of it is twice what was stored."""

    def __getattribute__(self, key):
        return 2 * object.__getattribute__(self, key)


def graph_reads_settings(x, settings):
    y = (x.sin() + 1).cos()
    if settings.kind == "plain":
        y = y * settings.width
    return (y * settings.hidden).tanh().exp()


class GraphKeepsKeys:
    def __init__(self):
        self.keys = None
        self.updated = {}


def graph_keeps_and_reads(x, settings, kept):
    y = (x.sin() + 1).cos()
    kept.keys = y
    kept.updated[0] = True
    if settings.kind == "plain":
        y = y * settings.width
    return (y * settings.hidden).tanh().exp()


def graph_reads_what_it_stored(x, doubles):
    y = (x.sin() + 1).cos()
    doubles.scale = 1.5
    return (y * doubles.scale).tanh().exp()


def graph_attends_plainly(q, rate):
    y = (q.sin() + 1).cos()
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        return y
    out = torch.nn.functional.scaled_dot_product_attention(y, y, y)
    if rate is not None:
        again = torch.nn.functional.scaled_dot_product_attention(y, y, y, None, rate)
        out = out + again
    return (out * 2).tanh().exp()


GRAPH_CACHE_LOG = []


class GraphNotes:
    @functools.lru_cache(None)  # noqa: B019
    def once(self, text):
        GRAPH_CACHE_LOG.append(("once", text))


GRAPH_NOTES = GraphNotes()


def graph_cached_scale(kind):
    GRAPH_CACHE_LOG.append(("scale", kind))
    return {"small": 0.5, "large": 2.0}.get(kind, 1.0)


def graph_cached_bounded(n):
    GRAPH_CACHE_LOG.append(("bounded", n))
    return n + 1.0


def graph_cached_typed(a, b):
    GRAPH_CACHE_LOG.append(("typed", a, b))
    return float(a) + 1.0


def graph_cached_named(a, b=2):
    GRAPH_CACHE_LOG.append(("named", a, b))
    return float(a * b)


class GraphLoudKey:
    def __init__(self, n):
        self.n = n

    def __hash__(self):
        GRAPH_CACHE_LOG.append(("hash", self.n))
        return self.n

    def __eq__(self, other):
        GRAPH_CACHE_LOG.append(("eq", self.n))
        return isinstance(other, GraphLoudKey) and other.n == self.n


def graph_cached_of_key(key):
    GRAPH_CACHE_LOG.append(("of_key", key.n))
    return key.n * 0.5


GRAPH_LOUD_KEY = GraphLoudKey(3)


GRAPH_CACHES = (
    graph_cached_scale,
    graph_cached_bounded,
    graph_cached_typed,
    graph_cached_named,
    graph_cached_of_key,
    GraphNotes.once,
)


def graph_calls_cached(x, at=0):
    # A wrapper that is found on a class and bound to the object, and one
    # that is called three times, with what it gives going into operations.
    h = torch.relu(x) + 1
    GRAPH_NOTES.once(f"at {at % 2}")
    kind = "large" if at % 3 else "other"
    h = h * graph_cached_scale("small") + graph_cached_scale(kind)
    h = torch.tanh(h) * 2
    return h.sum() + graph_cached_scale("small")


def graph_cache_is_cleared(x, at=0):
    h = torch.relu(x) + 1
    if at in (5, 6, 9):
        graph_cached_scale.cache_clear()
    h = h * graph_cached_scale("small")
    h = torch.tanh(h) * 2 - 1
    return h.sum() * graph_cached_scale("small")


def graph_cache_is_bounded(x, at=0):
    h = (torch.relu(x) + 1) * 2 - 1
    h = h * graph_cached_bounded(at % 3)
    return (torch.tanh(h) * 2 - 1).sum() + 1


def graph_cache_is_typed(x, at=0):
    h = (torch.relu(x) + 1) * 2 - 1
    h = h * graph_cached_typed(1, 2) + graph_cached_typed(1.0, 2)
    h = h + graph_cached_typed(at % 2, int)
    return (torch.tanh(h) * 2 - 1).sum() + 1


def graph_cache_by_name(x, at=0):
    h = (torch.relu(x) + 1) * 2 - 1
    h = h * graph_cached_named(3) + graph_cached_named(3, b=2) + graph_cached_named(a=3)
    return (torch.tanh(h) * 2 - 1).sum() + 1


def graph_cache_of_loud_key(x, at=0):
    h = (torch.relu(x) + 1) * 2 - 1
    h = h * graph_cached_of_key(GRAPH_LOUD_KEY) + graph_cached_of_key(GraphLoudKey(3))
    return (torch.tanh(h) * 2 - 1).sum() + 1


def graph_cache_is_counted(x, at=0):
    h = torch.relu(x) + 1
    h = h * graph_cached_scale("small")
    info = graph_cached_scale.cache_info()
    return (h.sum() + 1) * 2 - 3, info.hits, info.misses, info.currsize


def graph_grows_a_list(x, shape):
    y = (x.sin() + 1).cos()
    shape += [2]
    return (y.view(shape) * 2).tanh().exp() + len(shape)


def graph_keeps_or_copies(x):
    y = (x.sin() + 1).cos()
    z = y.contiguous()
    z.add_(1)
    return (z * y).tanh().exp()


class GraphCollides:
    """A key whose hash is that of "a", and that says when it is compared."""

    def __init__(self):
        self.log = []

    def __hash__(self):
        return hash("a")

    def __eq__(self, other):
        self.log.append("eq")
        return False


def graph_reads_a_table(x, table, members):
    y = x.sin()
    scale = table["a"]
    there = "a" in table, "a" in members
    return ((y + 1).cos() * scale).tanh().exp(), there


class GraphGuarded2d(torch.nn.Module):
    """A layer as libraries of vision models have them: a convolution, a
    norm that checks what it is given with a message, an activation that
    writes."""

    def __init__(self, channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(channels)
        self.act = torch.nn.ReLU6(inplace=True)
        self.drop = torch.nn.Dropout(0.2)

    def forward(self, x):
        y = self.conv(x)
        torch._assert(y.ndim == 4, f"expected 4D input (got {y.ndim}D input)")
        return self.drop(self.act(self.bn(y))) + x


GRAPH_END_STATE = [0.5]


GRAPH_END_LOG = []


GRAPH_END_HELD = [None]


def graph_end_variable(x, keep=None):
    # The variable holds the last reference to what `pop` gave, from in
    # front of the first operation on.
    held = GRAPH_END_HELD.pop()
    h = torch.relu(x) + 1
    h = h * GRAPH_END_STATE[0]
    held = None
    h = h + GRAPH_END_STATE[0]
    return (torch.tanh(h) * 2 - 1).sum() + GRAPH_END_STATE[0], held


GRAPH_BEHIND_LOG = []


def graph_helper_catches(n):
    # Makes an exception, raises and catches it: no path goes through it.
    try:
        raise KeyError(n)
    except KeyError:
        GRAPH_BEHIND_LOG.append(n)
        return 2.0


def graph_layer_of_two(h, scale):
    return torch.tanh(h * scale)


def graph_goes_on_behind_a_call(x, at=0):
    # Behind the call that no path goes through, the function has no
    # operation of its own: only calls of functions that have some.
    h = (torch.relu(x) + 1) * 2 - 1
    scale = graph_helper_catches(at % 3)
    for _ in range(3):
        h = graph_layer_of_two(h, scale)
    return graph_layer_of_two(h, 0.5)


GRAPH_TEMPORARY_LAYERS = [torch.nn.Linear(4, 4) for _ in range(3)]


def graph_drops_temporaries(x, at=0):
    # A list that the slice makes and a bound method: each is held by the
    # function alone, from in front of the call that no path goes through,
    # and is let go of inside the graph that starts behind it.
    layers = GRAPH_TEMPORARY_LAYERS[:2]
    last = GRAPH_TEMPORARY_LAYERS[2].forward
    scale = graph_helper_catches(at % 3)
    h = x
    for layer in layers:
        h = layer(h)
    h = last(h)
    return torch.tanh(h) * scale


GRAPH_WIDE_TABLE = tuple((n, n + 1) for n in range(5000))


def graph_drops_beside_much(x, at=0):
    # As above, with a value among those of the path that holds thousands
    # of objects of its own (the table, which the path reads): they are
    # none of what the path lets go of.
    layers = GRAPH_TEMPORARY_LAYERS[:2]
    last = GRAPH_TEMPORARY_LAYERS[2].forward
    scale = graph_helper_catches(at % 3)
    table = GRAPH_WIDE_TABLE
    h = x
    for layer in layers:
        h = layer(h)
    h = last(h)
    return torch.tanh(h) * (len(table) // 2500 * scale)


def graph_starts_twice(x, at=0):
    # Behind the call that no path goes through, the function goes on
    # with operations of its own: a graph starts at the first anyway.
    scale = graph_helper_catches(at % 3)
    h = (torch.relu(x) + 1) * 2 - 1
    return torch.tanh(h * scale).sum()


class GraphUnusedPlain:
    pass


class GraphUnusedAsked:
    # A class that reads its attributes in a function of its own.
    def __getattribute__(self, key):
        return super().__getattribute__(key)


GRAPH_UNUSED_LOG = []


def graph_unused_told(self):
    GRAPH_UNUSED_LOG.append(("read", len(GRAPH_UNUSED_LOG)))
    return 1


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


def graph_unused_whether(obj, x):
    h = torch.relu(x) + 1
    if hasattr(obj, "extra"):
        h = h - 5
    h = torch.tanh(h * 2 - 3) * 2 + 1
    return torch.relu(h) + h


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


class GraphConfigBase:
    # A class that reads its attributes in a function of its own, which
    # ends in the lookup of `object`, as a base class does it for it too.
    def __getattribute__(self, key):
        spec = super().__getattribute__("__dict__").get("_spec")
        if spec is not None and key in spec:
            raise RuntimeError("not asked for")
        return super().__getattribute__(key)


class GraphConfig(GraphConfigBase):
    attribute_map = {"alias": "scale"}

    def __init__(self):
        self.scale = 2.0

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)


def graph_asks_a_config(config, x):
    h = torch.relu(x) + 1
    h = h * config.alias - 3
    if not getattr(config, "is_causal", True):
        return h.sum()
    h = torch.tanh(h) * 2 + 1
    if hasattr(config, "extra"):
        h = h - 5
    return torch.relu(h) + h


def graph_asks_a_plain_object(obj, x):
    h = torch.relu(x) + 1
    h = h * getattr(obj, "scale", 3.0) - 3
    if hasattr(obj, "extra"):
        h = h - 5
    return torch.tanh(h) + h


class GraphKept(abc.ABC):
    @abc.abstractmethod
    def size(self):
        pass


class GraphKeptOne(GraphKept):
    def size(self):
        return 1


class GraphKeptNot:
    pass


class GraphKeptSays:
    # What it says its class is runs this code.
    @property
    def __class__(self):
        GRAPH_UNUSED_LOG.append("asked for its class")
        return GraphKeptOne


def graph_asks_an_abstract_class(obj, x):
    h = torch.relu(x) + 1
    if isinstance(obj, GraphKept):
        h = h * 2 - 3
    return torch.tanh(h) + h


def graph_binds_a_method_of_a_dict(table, x, key):
    h = torch.relu(x) + 1
    get = table.get
    h = h * get(key, 3.0) - 3
    return torch.tanh(h) + h


def graph_draws_behind_a_branch(x):
    h = torch.tanh(torch.relu(x) + 1) * 2
    if h.requires_grad:
        h = h + 1
    h = torch.nn.functional.dropout(h * 2, 0.5)
    return torch.tanh(h) + 1


class GraphCounted:
    # Its truth is whether it holds anything.
    def __init__(self):
        self.items = []

    def __len__(self):
        return len(self.items)


class GraphCountedLoudly(GraphCounted):
    def __len__(self):
        GRAPH_UNUSED_LOG.append("asked how many")
        return len(self.items)


def graph_asks_whether_any(held, x):
    h = torch.relu(x) + 1
    if held and not x.requires_grad:
        h = h * 2 - 3
    if not held:
        h = h + 7
    return torch.tanh(h) + h


GRAPH_RAISE_OUT = {"there": 2.0}


GRAPH_RAISE_NUMBERS = {"there": 1, "missing": 0}


GRAPH_RAISE_ROWS = {"there": (2.0, 0.5), "missing": (2.0, 0.5, 1.0)}


GRAPH_RAISE_SEEN = []


def graph_raises_out(x, given):
    # An entry of a dict from outside, which is not there at some calls.
    h = torch.relu(x) + 1
    h = h * 2 + 1
    scale = GRAPH_RAISE_OUT[given]
    return (torch.tanh(h * scale) * 2 - 1).sum()


def graph_raises_and_handles(x, given):
    h = torch.relu(x) + 1
    h = h * 2 + 1
    try:
        scale = GRAPH_RAISE_OUT[given]
    except KeyError:
        scale = 3.0
    return (torch.tanh(h * scale) * 2 - 1).sum()


def graph_raises_dividing(x, given):
    h = torch.relu(x) + 1
    h = h * 2 + 1
    scale = 2.0 / GRAPH_RAISE_NUMBERS[given]
    return (torch.tanh(h * scale) * 2 - 1).sum()


def graph_raises_unpacking(x, given):
    h = torch.relu(x) + 1
    h = h * 2 + 1
    scale, shift = GRAPH_RAISE_ROWS[given]
    return (torch.tanh(h * scale + shift) * 2 - 1).sum()


def graph_stores_and_raises(x, given):
    # A store that waits for the path to land, in front of what raises.
    h = torch.relu(x) + 1
    h = h * 2 + 1
    GRAPH_RAISE_SEEN.append(given)
    scale = GRAPH_RAISE_OUT[given]
    return (torch.tanh(h * scale) * 2 - 1).sum()


def graph_raise_reads(key):
    return GRAPH_RAISE_OUT[key]


def graph_raises_in_what_it_calls(x, given):
    h = torch.relu(x) + 1
    h = h * 2 + 1
    scale = graph_raise_reads(given)
    return (torch.tanh(h * scale) * 2 - 1).sum()


def graph_short_way_returns(x, given):
    # A way through the function with fewer operations than a graph has.
    h = torch.relu(x) + 1
    if given != 1:
        return h.sum()
    h = (h * 2).sin() + 1
    return (torch.tanh(h * 2) * 2 - 1).sum()


def graph_short_way_raises(x, given):
    h = torch.relu(x) + 1
    if given != 1:
        raise ValueError
    h = (h * 2).sin() + 1
    return (torch.tanh(h * 2) * 2 - 1).sum()


VM_KLASS_LOG = []


VM_KLASS_TO = [torch.Tensor]


def vm_klass_true(name, tensor, *at):
    """What the class under every tensor gives for a read of a size."""
    found = getattr(torch._C.TensorBase, name)
    return found(tensor, *at) if name in ("size", "dim") else found.__get__(tensor)


class VmKlassLogs(torch.Tensor):
    """Says when its sizes are read, and gives them."""

    @property
    def shape(self):
        VM_KLASS_LOG.append("shape")
        return vm_klass_true("shape", self)

    @property
    def ndim(self):
        VM_KLASS_LOG.append("ndim")
        return vm_klass_true("ndim", self)

    def size(self, *at):
        VM_KLASS_LOG.append("size")
        return vm_klass_true("size", self, *at)

    def dim(self):
        VM_KLASS_LOG.append("dim")
        return vm_klass_true("dim", self)


class VmKlassLies(torch.Tensor):
    """Gives other sizes than it has."""

    @property
    def shape(self):
        true = vm_klass_true("shape", self)
        return torch.Size([2 * true[0], *true[1:]])

    @property
    def ndim(self):
        return vm_klass_true("ndim", self) + 1

    def size(self, *at):
        return 2 * vm_klass_true("size", self, *at)

    def dim(self):
        return vm_klass_true("dim", self) + 1


class VmKlassPlain(torch.Tensor):
    pass


def vm_klass_becomes(t):
    t.__class__ = VM_KLASS_TO[0]
    return 0


def vm_klass_back(t):
    t.__class__ = torch.Tensor
    return 0


def vm_klass_case_shape(x):
    a = x.shape[0]
    vm_klass_becomes(x)
    b = x.shape[0]
    vm_klass_back(x)
    h = (x * 2).sin() + 1
    return (h * (a + 10 * b)).sum()


def vm_asks_case_compared_behind_a_kernel(x):
    h = (x * 2).sin() + 1
    if x.size(0) > 3:
        return (h * 3).sum()
    return (h - 1).sum()


def vm_asks_case_in_a_loop(x):
    total = x * 0
    for _ in range(4):
        total = total + x * x.shape[0] + x.ndim
    return total.sum()


def vm_asks_case_reads(x):
    h = (x * 2).sin() + 1
    return (h * h.shape[0] + h.ndim + x.shape[0] + h.size(1) + x.dim()).sum()


VM_SAYS_LOG = []


class VmSaysLoud(Exception):
    def __str__(self):
        VM_SAYS_LOG.append("str")
        return "loud"

    def __repr__(self):
        VM_SAYS_LOG.append("repr")
        return "VmSaysLoud()"


class VmSaysPlain(RuntimeError):
    """An exception class with the text of the class it is made of."""


class VmSaysArgument:
    def __repr__(self):
        VM_SAYS_LOG.append("repr of an argument")
        return "argument"

    __str__ = __repr__


def vm_says_case(x):
    h = (x * 2).sin() + 1
    return (h.cos() * 3 + h).sum()


def vm_says_helper(gm):
    raise IndexError("deep")


def vm_says_pass_that_raises_deep(gm, input_types):
    return vm_says_helper(gm)


GRAPH_NAME_LOG = []


class GraphKeyLoud(str):
    """A text that logs when it is compared: as a key of a dict, whenever a
    lookup under an equal text comes to it."""

    __slots__ = ()

    def __eq__(self, other):
        GRAPH_NAME_LOG.append("eq")
        return str.__eq__(self, other)

    __hash__ = str.__hash__


class GraphKeyNever(str):
    """As `GraphKeyLoud`, and equal to nothing."""

    __slots__ = ()

    def __eq__(self, other):
        GRAPH_NAME_LOG.append("never")
        return False

    __hash__ = str.__hash__


class GraphKeyConfig:
    def __init__(self):
        self.width, self.scale = 2.0, 3.0


class GraphKeyOwnLookup(GraphKeyConfig):
    def __getattribute__(self, key):
        return super().__getattribute__(key)


GraphKeyBase = type("GraphKeyBase", (), {GraphKeyLoud("width"): 2.0, "scale": 3.0})


class GraphKeyChild(GraphKeyBase):
    @property
    def width(self):
        return super().width


def graph_key_reads(x, config):
    h = torch.relu(x) + 1
    h = h * config.width + 0.5
    return ((h * config.scale).sin() * 2 - 1).sum()


def graph_key_stores(x, config):
    h = torch.relu(x) + 1
    config.width = 2.0
    h = h * config.scale + 0.5
    return ((h * 2).sin() * 2 - 1).sum()


GRAPH_NAME_GLOBALS = {"torch": torch, "graph_key_width": 2.0}


exec(
    "def graph_key_reads_a_global(x, config):\n    h = torch.relu(x) + 1\n    h = h * graph_key_width + 0.5\n    return ((h * 3.0).sin() * 2 - 1).sum()\n",
    GRAPH_NAME_GLOBALS,
)


def graph_key_module():
    module = types_module.ModuleType("graph_key_module")
    module.width, module.scale = 2.0, 3.0
    return module


def graph_key_loud(held, name="width"):
    # The entry under the name, under a key that is equal and compares in
    # Python.
    held[GraphKeyLoud(name)] = held.pop(name)


def graph_key_never(config):
    # A key of the same hash that is equal to nothing, and the value in
    # the class.
    del vars(config)["width"]
    vars(config)[GraphKeyNever("width")] = 9.0
    type(config).width = 2.5


def graph_key_plain(held, name="width"):
    for key in [key for key in held if key == name]:
        held[name] = held.pop(key)


def graph_key_back(config):
    held = vars(config)
    for key in [key for key in held if type(key) is GraphKeyNever]:
        del held[key]
    held["width"] = 2.0
    del type(config).width


GRAPH_NAME_CASES = {
    "an attribute of an object": (
        graph_key_reads,
        GraphKeyConfig,
        lambda c: graph_key_loud(vars(c)),
        lambda c: graph_key_plain(vars(c)),
    ),
    "an attribute read by a function of the class": (
        graph_key_reads,
        GraphKeyOwnLookup,
        lambda c: graph_key_loud(vars(c)),
        lambda c: graph_key_plain(vars(c)),
    ),
    "a key that is equal to nothing": (
        graph_key_reads,
        GraphKeyConfig,
        graph_key_never,
        graph_key_back,
    ),
    "an attribute of a module": (
        graph_key_reads,
        graph_key_module,
        lambda m: graph_key_loud(vars(m)),
        lambda m: graph_key_plain(vars(m)),
    ),
    "a global of the function": (
        GRAPH_NAME_GLOBALS["graph_key_reads_a_global"],
        lambda: GRAPH_NAME_GLOBALS,
        lambda g: graph_key_loud(g, "graph_key_width"),
        lambda g: graph_key_plain(g, "graph_key_width"),
    ),
    "a store into an object": (
        graph_key_stores,
        GraphKeyConfig,
        lambda c: graph_key_loud(vars(c)),
        lambda c: graph_key_plain(vars(c)),
    ),
    "an attribute read through super()": (graph_key_reads, GraphKeyChild, None, None),
}


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


class GraphKeyRaises(str):
    __slots__ = ()

    def __hash__(self):
        GRAPH_KEY_LOG.append("hash")
        raise ValueError("a key that has no hash")


def graph_key_stored_and_dropped(x, given):
    h = torch.relu(x) + 1
    made = {}
    made[given] = 1
    return (torch.tanh(h * 2) * 2 - 1).sum()


def graph_key_popped_and_dropped(x, given):
    h = torch.relu(x) + 1
    made = {"k": 1}
    made.pop(given)
    return (torch.tanh(h * 2) * 2 - 1).sum()


def graph_key_popped_with_a_default(x, given):
    h = torch.relu(x) + 1
    made = {"k": 1}
    made.pop(given, None)
    return (torch.tanh(h * 2) * 2 - 1).sum()


class GraphKeyNumber(int):
    def __hash__(self):
        GRAPH_KEY_LOG.append("hash")
        return int.__hash__(self)

    def __eq__(self, other):
        GRAPH_KEY_LOG.append("eq")
        return int.__eq__(self, other)


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


GRAPH_OUTSIDE_LISTS = ([], [])


class GraphBareHolder:
    pass


def graph_appends_by_a_method_of_another_list(x, given):
    h = torch.relu(x) + 1
    holder = GraphBareHolder()
    holder.append = GRAPH_OUTSIDE_LISTS[given].append
    add = holder.append
    add(1.0)
    return (h * len(GRAPH_OUTSIDE_LISTS[given])).sum()


GRAPH_NAMED = {}


def graph_pops_by_a_method_read_earlier(x, given):
    h = torch.relu(x) + 1
    table = {"scale": given, "shift": 1.0}
    take = table.pop
    scale = take("scale")
    rest = take("offset", len(table))
    return (h * scale + rest).sum()


def graph_pops_by_a_method_that_alone_has_the_dict(x, given):
    h = torch.relu(x) + 1
    take = {"scale": given, "shift": 1.0}.pop
    return (h * take("scale") + take("shift")).sum()


def graph_takes_named(h, **kwargs):
    scale = kwargs.pop("scale", float(len(kwargs)))
    return graph_pinned_scaled(h, scale, **kwargs)


def graph_pops_with_a_default_it_calls_for(x, given):
    h = torch.relu(x) + 1
    if given:
        return graph_takes_named(h, scale=2.0, shift=0.5).sum()
    return graph_takes_named(h, shift=0.5).sum()


def graph_pops_by_a_method_of_another_dict(x, given):
    h = torch.relu(x) + 1
    holder = GraphBareHolder()
    holder.pop = GRAPH_NAMED.pop
    take = holder.pop
    return (h * take(given, 2.5)).sum()


GRAPH_META_LOG = []


class GraphMeta(type):
    pass


class GraphKinds(metaclass=GraphMeta):
    scale = 2.0


class GraphColor(enum.Enum):
    RED = 1
    BLUE = 2


def graph_reads_of_classes(x, color):
    h = torch.relu(x) + 1
    if color is GraphColor.RED:
        h = h * GraphKinds.scale
    out = (torch.tanh(h * 2) * 2 - 1).sum()
    return out + (2.0 if color is GraphColor.BLUE else 1.0)


class GraphWithinBehindAMember(GraphWithinModel):
    def forward(self, x):
        h = torch.tanh(self.shared(x) * 2) - 1
        if GraphColor.RED is GraphColor.BLUE:
            h = h + 1
        return self.inner(h).sum()


GRAPH_COMPARE_LOG = []


class GraphAction(str, enum.Enum):
    NONE = "none"
    NOTIFY = "notify"
    RAISE = "raise"


class GraphPlace(enum.Enum):
    ONE = 1
    TWO = 2


class GraphQuietText(str):
    __slots__ = ()


GRAPH_MIDDLE = GraphQuietText("m")


class GraphLoudText(str):
    __slots__ = ()

    def __eq__(self, other):
        GRAPH_COMPARE_LOG.append("eq")
        return str.__eq__(self, other)

    __hash__ = str.__hash__


def graph_loud_eq(self, other):
    GRAPH_COMPARE_LOG.append("planted")
    return str.__eq__(self, other)


def graph_compares_members(x, given):
    h = torch.relu(x) + 1
    action = GraphAction.NONE if given is None else GraphAction.NOTIFY
    if action == GraphAction.RAISE:
        raise ValueError(action)
    elif action in (GraphAction.NOTIFY, GraphAction.RAISE):
        h = h * 3
    return (h * 2).sum()


def graph_compares_which_it_is(x, given):
    h = torch.relu(x) + 1
    if given == GraphPlace.TWO or given != GraphPlace.ONE:
        h = h * 3
    if given in [GraphPlace.TWO, None]:
        h = h + 1
    return (h * 2).sum()


def graph_compares_texts(x, given):
    h = torch.relu(x) + 1
    if given == "wide" or given in ("deep", "tall"):
        h = h * 3
    return (h * 2).sum()


def graph_compares_an_order(x, given):
    h = torch.relu(x) + 1
    return (h * (2.0 if given < GRAPH_MIDDLE else 3.0)).sum()


class GraphLoudMeta(type):
    def __repr__(cls):
        GRAPH_COMPARE_LOG.append("repr")
        return "<class 'loud'>"


class GraphOfLoudMeta(metaclass=GraphLoudMeta):
    pass


def graph_text_of_a_class(x, given):
    h = torch.relu(x) + 1
    return (h * GRAPH_NAMED.get(str(given.__class__), 1.5)).sum()


GRAPH_LOOKUP_LOG = []


class GraphOwnLookup:
    """A class that reads its attributes in a function of its own, as the
    configuration of a library does."""

    names = {"wide": "width"}
    bias = 0.25

    def __init__(self):
        self.width, self.scale = 2.0, 3.0

    def __getattribute__(self, key):
        if key != "names" and key in super().__getattribute__("names"):
            key = super().__getattribute__("names")[key]
        return super().__getattribute__(key)


def graph_reads_by_its_own_lookup(x, given):
    config, options = given
    h = torch.relu(x) + 1
    h = h * config.wide + options.get("shift", 0.5) + config.bias
    return ((h * config.scale).sin() * 2 - (options.get("drop") or 1.0)).sum(), None


VM_GIVEN_MAKERS = {
    "zeros_like": lambda v: torch.zeros_like(v),
    "ones_like": lambda v: torch.ones_like(v),
    "empty_like": lambda v: torch.empty_like(v).zero_(),
    "full_like": lambda v: torch.full_like(v, 2.0),
    "rand_like": lambda v: torch.rand_like(v) * 0,
    "new_zeros": lambda v: v.new_zeros(4),
    "zeros": lambda v: torch.zeros(4, device=v.device) + v,
    "clone": lambda v: v.clone(),
}


def vm_given_function(name):
    make = VM_GIVEN_MAKERS[name]

    def given(x, v):
        h = (x * 0.5).cos() + x.sin()
        return h.sum() * 2.0, make(v)

    return given


def vm_given_outcome(call):
    """What a call gave, as sums, or what it raised."""
    try:
        return [round(float(t.sum()), 3) for t in call()]
    except Exception as error:
        return f"{type(error).__name__}: {str(error)[:60]}"


def vm_given_class(policy):
    """A class of tensors that wrap a tensor and keep what the dispatcher
    asks of them, and with `policy` what it asks about their sizes: the
    class and its log."""
    asked = []
    answers = {
        "aten.dim": lambda e: e.dim(),
        "aten.sym_size": lambda e: tuple(e.shape),
        "aten.size": lambda e: tuple(e.shape),
        "aten.sym_numel": lambda e: e.numel(),
        "aten.numel": lambda e: e.numel(),
        "aten.sym_stride": lambda e: tuple(e.stride()),
        "aten.stride": lambda e: tuple(e.stride()),
        "aten.is_contiguous": lambda e: e.is_contiguous(),
        "aten.sym_storage_offset": lambda e: e.storage_offset(),
        "aten.storage_offset": lambda e: e.storage_offset(),
        "aten.is_strides_like_format": lambda e: False,
        "aten.is_non_overlapping_and_dense": lambda e: True,
    }
    options = {"dispatch_sizes_strides_policy": policy} if policy else {}

    class Asked(torch.Tensor):
        @staticmethod
        def __new__(cls, elem):
            made = torch.Tensor._make_wrapper_subclass(
                cls,
                elem.size(),
                strides=elem.stride(),
                storage_offset=elem.storage_offset(),
                dtype=elem.dtype,
                layout=elem.layout,
                device=elem.device,
                requires_grad=elem.requires_grad,
                **options,
            )
            made.elem = elem
            return made

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            name = ".".join(str(func).split(".")[:2])
            asked.append(name)
            if name in answers:
                return answers[name](args[0].elem)

            def inner(value):
                return value.elem if isinstance(value, Asked) else value

            out = func(
                *[inner(a) for a in args],
                **{k: inner(a) for k, a in (kwargs or {}).items()},
            )
            return Asked(out) if isinstance(out, torch.Tensor) else out

    return Asked, asked


VM_GIVEN_PLAIN = {
    "zeros_like": lambda x: torch.zeros_like(x),
    "ones_like with a dtype": lambda x: torch.ones_like(x, dtype=torch.float64),
    "zeros_like that requires a gradient": lambda x: torch.zeros_like(
        x, requires_grad=True
    ),
    "zeros": lambda x: torch.zeros(4, device=x.device),
    "ones that requires a gradient": lambda x: torch.ones(
        2, 3, device=x.device, requires_grad=True
    ),
    "full": lambda x: torch.full((2,), 3.0, device=x.device),
    "arange": lambda x: torch.arange(4.0, device=x.device),
    "eye": lambda x: torch.eye(3, device=x.device),
    "rand_like": lambda x: torch.rand_like(x),
    "hann_window": lambda x: torch.hann_window(4, device=x.device),
}


def vm_given_plain(name):
    make = VM_GIVEN_PLAIN[name]

    def plain(x):
        h = (x * 2).sin() + 1
        return make(x), h.cos()

    return plain


def vm_given_made(run, device, at):
    """What a factory gave in a call: all that Python can ask of it."""
    torch.manual_seed(at)
    made = run(torch.ones(2, 2, device=device) * (at + 1))[0]
    return (
        str(made.dtype),
        made.device.type,
        made.requires_grad,
        made.is_leaf,
        made.grad_fn is None,
        made._version,
        made.tolist(),
    )


VM_GIVEN_USES = {
    "zeros_like": lambda v: torch.zeros_like(v),
    "new_zeros": lambda v: v.new_zeros(2),
    "a loop over it": lambda v: [e for e in v],  # noqa: C416
    "a write to it": lambda v: v.add_(1.0),
    "view": lambda v: v.view(-1),
    "reshape": lambda v: v.reshape(-1),
    "abs": lambda v: v.abs(),
}


def vm_given_user(name):
    use = VM_GIVEN_USES[name]

    def used(x, v):
        h = (x * 0.5).cos() + x.sin()
        return h.sum() * 2.0, use(v)

    return used


def vm_given_asks(run, x, klass, asked):
    """What a call with a tensor of `klass` gave or raised, as names, and
    what the class was asked in it."""
    del asked[:]
    try:
        out = run(x, klass(torch.ones(4, 4, device=x.device)))
        said = [type(part).__name__ for part in out]
    except Exception as error:
        said = f"{type(error).__name__}: {str(error)[:60]}"
    return said, list(asked)


def back_convolves(x, w0, w1, w2, w3):
    h = torch.nn.functional.conv2d(x, w0, stride=2, padding=3).relu()
    h = torch.nn.functional.conv2d(h, w1, stride=1, padding=1).relu()
    h = torch.nn.functional.conv2d(h, w2, stride=2, padding=1).relu()
    h = torch.nn.functional.conv2d(h, w3, stride=1, padding=1)
    return h.mean(), h.tanh().mean()


def back_reader_keeps(x, a):
    h = (x @ a).tanh()
    return h, (h * h).sum() + (a * a).sum()


def back_node_keeps(x, a):
    # One product for both results: with nodes apart eager would go backward
    # for the one behind the other, and the graph would keep eager's tape.
    m = x @ a
    h = m.tanh()
    return h, m.exp().sum() + a.sum()


def back_view_kept(x, a):
    h = (x @ a).tanh()
    v = h[0]
    return h, (v * v).sum() + a.sum()


class BackConvNet(torch.nn.Module):
    """A convnet as they are trained: batch norm, an activation in place, a
    max pool, a dropout, a loss."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
        self.act = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(2)
        self.drop = torch.nn.Dropout(0.3)
        self.head = torch.nn.Linear(8 * 4 * 4, 5)

    def forward(self, x, y):
        h = self.drop(self.pool(self.act(self.bn(self.conv(x)))))
        return torch.nn.functional.cross_entropy(self.head(h.flatten(1)), y)


class BackConvNetLean(BackConvNet):
    """The same convnet with nothing but the result bound where a path lands
    in front of the loss: its graph is one node."""

    def forward(self, x, y):
        h = self.head(self.drop(self.pool(self.act(self.bn(self.conv(x))))).flatten(1))
        return torch.nn.functional.cross_entropy(h, y)


BACK_CONV_NETS = (BackConvNet, BackConvNetLean)


ONE_NODE = ["the engine's backward"]


TWO_NODES = ["in parts", "the engine's backward", "the engine's backward"]


def vm_hooked_same(t):
    return t


def vm_hooked_writes(h, w):
    a = h * w + 1.0
    a.mul_(2.0)
    return (a * w).sin().exp().sum()


def vm_hooked_rooted(h, w):
    a = (h + w).tanh()
    b = a * w
    c = (b - 100.0).sqrt()
    d = c.sin()
    return d.sum()


def vm_hooked_leaves(device, at):
    torch.manual_seed(at)
    x = torch.randn(3, 4, device=device, requires_grad=True)
    w = torch.randn(3, 4, device=device, requires_grad=True)
    return x, w


def vm_hooked_rows(run, fn, device, case, calls):
    """What `case` gives for each call, or what the call raised."""
    rows = []
    for at in calls:
        try:
            rows.append(case(run, fn, device, at))
        except Exception as error:
            rows.append((type(error).__name__, str(error)[:120]))
    return rows


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmFrontend(TestCase):
    hw_classification = HardwareClassification.GENERIC

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

        compiled = vm(fn)
        for _ in range(2):
            self.assertEqual(outcome(compiled), outcome(fn))

    def test_arguments(self):
        def f(a, b=2, *args, c=3, **kw):
            return a, b, args, c, sorted(kw.items())

        compiled = vm(f)
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

        compiled = vm(fact)
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

        compiled = vm(f)
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
        compiled = vm(namespace["checked"])
        self.assertEqual(compiled(2), 2)
        with self.assertRaisesRegex(AssertionError, "not one"):
            compiled(1)

    def test_methods_and_modules(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.ones(3))

            def forward(self, x):
                return x * self.w + 1

        m = M()
        x = torch.randn(3)
        for compiled in (vm(m), vm(m.forward)):
            for _ in range(3):
                self.assertEqual(compiled(x), m(x))


instantiate_parametrized_tests(TestVmFrontend)


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmDevices(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @isolated
    def test_autograd(self, device):
        def f(x, w):
            y = (x @ w).tanh() * 2
            if y.sum() > 0:
                y = y + 1
            return (y * y).sum()

        compiled = vm_early(f)
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

    @isolated
    def test_inplace_and_views(self, device):
        def f(x, y):
            v = x.view(-1)
            v.add_(y.view(-1))
            x *= 2
            t = x.t()
            t.sub_(1)
            return x

        compiled = vm_early(f)
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

        compiled = vm_early(leaf)
        x = torch.randn(3, device=device, requires_grad=True)
        for _ in range(3):
            with self.assertRaisesRegex(RuntimeError, "leaf Variable"):
                compiled(x)

    @isolated
    def test_recursion_and_threads(self, device):
        def f(x, n):
            if n == 0:
                return x
            return compiled(x * 2, n - 1) + 1

        compiled = vm_early(f)
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

    @isolated
    def test_unboxed_tensors_are_released(self, device):
        def f(x, y, fail):
            a = x + y
            b = a * 2
            if fail:
                return b @ y.reshape(-1)[:1]
            return (a - b).sum()

        compiled = vm_early(f)
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

    @isolated
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

        compiled = vm_early(step)
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


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmGraphTransforms(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_inlined_call_is_guarded_on_the_callee(self):
        compiled = vm(inl_outer)
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

    def test_unrolled_loop_is_guarded_on_range(self):
        compiled = vm(unr_loop)
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


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmCallees(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_changed_function_is_compiled_again(self):
        def target(x, k=2):
            return x * k

        def f(x):
            return target(x)

        compiled = vm(f)
        self.assertEqual(compiled(3), 6)
        target.__defaults__ = (5,)
        self.assertEqual(compiled(3), 15)
        target.__code__ = (lambda x, k=2: x - k).__code__
        self.assertEqual(compiled(3), -2)

    def test_infinite_recursion(self):
        def forever():
            return forever()

        def guarded():
            try:
                return guarded()
            except ValueError:
                return -1

        for fn in (forever, guarded):
            compiled = vm(fn)
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
        compiled = vm(count)
        self.assertLess(abs(compiled(0) - count(0)), 5)

    def test_recursion_deeper_than_the_native_stack(self):
        def depth(n):
            return 0 if n == 0 else 1 + depth(n - 1)

        compiled = vm(depth)
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
        compiled = vm(sig)
        compiled_caller = vm(call)
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
            compiled = vm(fn)
            for args in argsets:
                self.assertEqual(call_outcome(compiled, *args), call_outcome(fn, *args))
        message = call_outcome(vm(both), [1], {"k": 5})[1]
        self.assertTrue(
            message.endswith("target() got multiple values for keyword argument 'k'")
        )


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmClosures(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_inner_function_shares_a_cell(self):
        def counter(start):
            count = start

            def bump(by=1, *, twice=False):
                nonlocal count
                count += by * (2 if twice else 1)
                return count

            return bump

        compiled = vm(counter)
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

        compiled = vm(f)
        x, y = torch.randn(3), torch.randn(3)
        self.assertEqual(compiled(x, y), f(x, y))

    def test_lambdas_with_defaults_and_annotations(self):
        def adders(ks):
            return [(lambda v, k=k: v + k) for k in ks]

        def annotated():
            def inner(a: int, b: "str" = "z") -> float:
                return 1.0

            return inner

        compiled = vm(adders)
        self.assertEqual([add(10) for add in compiled([1, 2, 3])], [11, 12, 13])
        compiled = vm(annotated)
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
        compiled = vm(add)
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
            compiled = vm(fn)
            with self.assertRaises(error) as expected:
                fn()
            with self.assertRaises(error) as got:
                compiled()
            self.assertEqual(str(got.exception), str(expected.exception))


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmExceptions(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def assertSameOutcome(self, fn, *argsets, **options):
        compiled = vm(fn)
        for args in argsets * 2:
            self.assertEqual(self.outcome(compiled, *args), self.outcome(fn, *args))
        return compiled

    def outcome(self, fn, *args):
        args = [list(a) if isinstance(a, list) else a for a in args]
        try:
            result = ("returned", fn(*args))
        except Exception as e:
            result = ("raised", type(e).__name__, str(e), type(e.__cause__).__name__)
        return (result, [a for a in args if isinstance(a, list)])

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

    def test_handled_exception_is_restored(self):
        def f(x):
            try:
                exc_double(x)
            except ValueError:
                inside = sys.exc_info()[1]
            return (type(inside).__name__, sys.exc_info()[1])

        self.assertSameOutcome(f, (-1,))


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmWith(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_with_calls_enter_and_exit(self):
        def f(x, log, suppress):
            y = 0
            with ExcManager(log, suppress) as m:
                y = m + exc_double(x)
            return y

        compiled = vm(f)
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

        compiled = vm(f)
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

        compiled = vm(f)
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

        compiled = vm(f)
        with self.assertRaisesRegex(
            TypeError, "does not support the context manager protocol"
        ):
            compiled(3)


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmMoreBytecodes(TestCase):
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

        compiled = vm(f)
        self.assertEqual(compiled(2.5), f(2.5))
        with self.assertRaisesRegex(
            ImportError, "cannot import name 'nothing_here' from 'math'"
        ):
            vm(missing)()

    def test_super(self):
        obj = BytecodeChild()
        for method in (BytecodeChild.scale, BytecodeChild.explicit):
            compiled = vm(method)
            self.assertEqual(compiled(obj, 3), method(obj, 3))
        compiled = vm(BytecodeChild.name.fget)
        self.assertEqual(compiled(obj), "child of base")

    def test_super_that_is_not_the_builtin(self):
        class Shadow:
            name = "something else"

        # A module that rebinds `super` gets what it asked for: its object is
        # called without arguments, as the source has it.
        method = BytecodeChild.name.fget
        compiled = vm(method)
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

        compiled = vm(f)
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
        compiled = vm(f)
        for xs in ([1, 2, 3], [], (0, 5)):
            self.assertEqual(compiled(xs), f(xs))

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
            compiled = vm(fn)
            self.assertEqual(call_outcome(compiled, *args), call_outcome(fn, *args))
        self.assertNotIn("bytecode_global", globals())

    def test_match_statement(self):
        compiled = vm(match_describe)
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
            outcome = call_outcome(vm(fn), *args)
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
        compiled = vm(big)
        self.assertEqual(compiled(0, 3), big(0, 3))
        self.assertEqual(compiled(torch.zeros(2), 2), big(torch.zeros(2), 2))


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmFrames(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_attribute_named_like_a_builtin_that_reads_the_frame(self):
        # `model.eval()` is not `eval()`.
        def f(model, x):
            model.eval()
            return model(x)

        model, x = torch.nn.Linear(2, 2), torch.ones(2)
        compiled = vm(f)
        self.assertEqual(compiled(model, x), f(model, x))

    def test_new_classes_belong_to_the_module_of_the_compiled_function(self):
        compiled = vm(frame_new_classes)
        self.assertEqual(compiled(), (__name__, __name__))

    def test_name_error_names_the_variable(self):
        def outcome(f, *args):
            try:
                f(*args)
            except NameError as e:
                return type(e), str(e), e.name

        for fn, args in ((frame_missing_global, ()), (frame_unbound_local, (False,))):
            compiled = vm(fn)
            self.assertEqual(outcome(compiled, *args), outcome(fn, *args))
        self.assertEqual(outcome(frame_missing_global)[2], "frame_surely_not_defined")


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmEvalBreaker(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_other_threads_run_during_a_compiled_loop(self):
        flag = []

        def worker():
            time.sleep(0.05)
            flag.append(1)

        compiled = vm(breaker_spins)
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
        compiled = vm(churn)
        self.assertGreater(collections(churn), 20)
        self.assertGreater(collections(compiled), 20)


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmShortcuts(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_forward_replaced_on_the_instance(self):
        net, x = ShortcutNet(), torch.randn(4, 3)
        compiled = vm(shortcut_call)
        self.assertEqual(compiled(net, x), net(x))
        net.forward = lambda x: x.sum()
        self.assertEqual(compiled(net, x), x.sum())
        del net.forward
        self.assertEqual(compiled(net, x), net(x))

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
        compiled = vm(read)
        expected = call_outcome(read, module)
        self.assertEqual(Guarded.reads, 1)
        self.assertEqual(call_outcome(compiled, module), expected)
        self.assertEqual(Guarded.reads, 2)


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmRegions(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @isolated
    def test_frozen_parameter(self):
        torch.manual_seed(0)
        model = RegionNet().train()
        compiled = vm(
            region_infer, passes=(*EARLY, vm.passes.Autograd(), vm.passes.Partition())
        )
        for i in range(8):
            if i == 5:
                model.conv.weight.requires_grad_(False)
            x = torch.randn(2, 3, 6, 6)
            model.zero_grad()
            compiled(model, x).sum().backward()
            self.assertEqual(model.conv.weight.grad is None, i >= 5)
            self.assertIsNotNone(model.fc.weight.grad)

    # With Autograd() and Partition(), the region AOT Autograd made fails in its forward: a value it saves for the backward is None.
    @unittest.expectedFailure
    @isolated
    def test_training_step_of_a_recurrent_model(self):
        def train(compile_step):
            torch.manual_seed(0)
            model = RegionRecurrent().train()
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            step = region_step_plain
            if compile_step:
                step = vm(
                    step, passes=(*EARLY, vm.passes.Autograd(), vm.passes.Partition())
                )
            losses = [step(model, opt, torch.randn(5, 4, 6)).item() for _ in range(8)]
            return (model.state_dict(), losses)

        expected_state, expected_losses = train(False)
        state, losses = train(True)
        self.assertEqual(losses, expected_losses)
        for name, value in state.items():
            self.assertTrue(torch.equal(value, expected_state[name]), name)

    @isolated
    def test_error_in_a_region(self):
        model = RegionNet().eval()
        compiled = vm(
            region_infer, passes=(*EARLY, vm.passes.Autograd(), vm.passes.Partition())
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

    @isolated
    def test_a_trace_does_not_use_up_an_iterator_of_the_call(self):
        def made(ks, x):
            y = x * 2 + 1
            gen = (k + 1 for k in ks)
            t = tuple(gen)
            return (t, y)

        def advanced(ks, x):
            y = x * 2 + 1
            it = iter(ks)
            first = next(it)
            rest = list(it)
            return (first, rest, y)

        x = torch.ones(3)
        for fn in (made, advanced):
            compiled = vm(
                fn, passes=(*EARLY, vm.passes.Autograd(), vm.passes.Partition())
            )
            for _ in range(8):
                self.assertEqual(compiled([1, 2], x), fn([1, 2], x))


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmGraphResults(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def step(self, fn, x, w):
        out = fn(x, w)
        return (out, torch.autograd.grad(out, (x, w)))

    @isolated
    def test_an_exported_graph_runs_on_the_kernels_of_eager_mode(self, device):
        graphs = []  # noqa: F841

        def keep(gm, input_types):
            return gm

        passes = (*EARLY, keep)
        compiled = vm(graph_mixed, passes=passes)
        x = torch.randn(4, 8, device=device, requires_grad=True)
        w = torch.randn(8, 8, device=device, requires_grad=True)
        for _ in range(5):
            out, grads = self.step(compiled, x, w)
            expected = self.step(graph_mixed, x, w)
            self.assertEqual(out, expected[0], atol=0, rtol=0)
            self.assertEqual(grads, expected[1], atol=0, rtol=0)
        seen = []  # noqa: F841

        def keeps(gm, input_types):
            return gm

        passes = (*EARLY, keeps)
        compiled = vm(graph_mixed, passes=passes)
        for _ in range(8):
            out, grads = self.step(compiled, x, w)
            expected = self.step(graph_mixed, x, w)
            self.assertEqual(out, expected[0], atol=0, rtol=0)
            self.assertEqual(grads, expected[1], atol=0, rtol=0)

    @parametrize("case", ("cat", "stack", "masked", "subscripts", "conv", "listed"))
    @isolated
    def test_an_operation_that_is_no_call_site_is_a_node_of_a_graph(self, device, case):
        def keep(gm, input_types):
            return gm

        fn, args = graph_operands(case, device)
        operator = {"masked": "index.Tensor", "subscripts": "index.Tensor"}
        operator.update(conv="conv2d", listed="cat")
        operator = "aten::" + operator.get(case, case)
        expected = fn(*args)
        for behind in ((), (keep,)):
            passes = (*EARLY, *behind)
            compiled = vm(fn, passes=passes)
            for _ in range(4):
                self.assertEqual(compiled(*args), expected, atol=0, rtol=0)
        if case == "masked":
            return
        passes = (*EARLY, vm.passes.Autograd(), vm.passes.Partition())
        compiled = vm(fn, passes=passes)
        for _ in range(4):
            self.assertEqual(compiled(*args), expected, rtol=1e-05, atol=1e-06)

    @isolated
    def test_a_graph_is_for_the_dtype_autocast_casts_to(self, device):
        kind = torch.device(device).type
        passes = (*EARLY, vm.passes.Inductor())
        compiled = vm(graph_casts, passes=passes)
        x, w = (torch.randn(4, 8, device=device), torch.randn(8, 8, device=device))
        for dtype in (torch.bfloat16, torch.float16, torch.bfloat16):
            with torch.autocast(kind, dtype=dtype):
                for _ in range(4):
                    got = compiled(x, w)
                expected = graph_casts(x, w)
            self.assertEqual(got.dtype, expected.dtype)
            self.assertEqual(got.dtype, dtype)
            self.assertEqual(got.float(), expected.float(), rtol=0.05, atol=0.05)
        for _ in range(3):
            self.assertEqual(compiled(x, w), graph_casts(x, w), rtol=0.0001, atol=1e-05)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            for _ in range(3):
                got = compiled(x, w)
                self.assertEqual(got, graph_casts(x, w), rtol=0.0001, atol=1e-05)
        finally:
            torch.use_deterministic_algorithms(False)

    @isolated
    def test_what_a_class_reads_itself_is_not_what_a_path_stored(self, device):
        def keep(gm, input_types):
            return gm

        x = torch.randn(3, 4, device=device)
        doubles = GraphDoubles()
        passes = (*EARLY, keep)
        stores = vm(graph_reads_what_it_stored, passes=passes)
        for _ in range(6):
            expected = graph_reads_what_it_stored(x, GraphDoubles())
            self.assertEqual(stores(x, doubles), expected, atol=0, rtol=0)

    @isolated
    def test_a_class_that_reads_its_attributes_in_a_function_of_its_own(self, device):
        x = torch.randn(3, 4, device=device)
        settings = GraphSettings()
        other_names = {"width": "_kind", "hidden": "hidden"}
        changes = [(settings, "hidden", 3.0), (settings, "_kind", "other")]
        changes += [(GraphSettings, "renamed", other_names), (settings, "_kind", 5.0)]
        for target, name, value in changes:
            setattr(target, name, value)
            for _ in range(2):
                expected = graph_reads_settings(x, settings)
        GraphSettings.renamed = {"width": "hidden"}
        other = GraphSettingsWithDefault()
        passes = (*EARLY,)
        left = vm(graph_reads_settings, passes=passes)
        for _ in range(6):
            expected = graph_reads_settings(x, other)
            self.assertEqual(left(x, other), expected, atol=0, rtol=0)

    # What the function stores into an object between its tensor operations is not what eager leaves there.
    @unittest.expectedFailure
    @isolated
    def test_a_class_that_reads_its_attributes_itself_behind_a_store(self, device):
        x = torch.randn(3, 4, device=device)
        settings, kept = (GraphSettings(), GraphKeepsKeys())
        self.assertEqual(kept.updated, {0: True})
        both = GraphSettings()
        both.updated = {}
        for args in ((x, both, both), (x, settings, kept), (x, both, both)):
            for _ in range(2):
                expected = graph_keeps_and_reads(*args)  # noqa: F841
        self.assertEqual(both.updated, {0: True})

    @isolated
    def test_an_attention_at_no_rate_and_a_question_to_cuda_end_no_graph(self, device):
        q = torch.randn(2, 2, 5, 4, device=device)
        fn, passes = (graph_attends_plainly, (*EARLY,))
        for rate, ops in ((None, 7), (0.0, 9)):
            compiled = vm(fn, passes=passes)
            for _ in range(6):
                self.assertEqual(compiled(q, rate), fn(q, rate), atol=0, rtol=0)
        compiled = vm(fn, passes=passes)
        for _ in range(6):
            torch.manual_seed(3)
            expected = fn(q, 0.5)
            torch.manual_seed(3)
            self.assertEqual(compiled(q, 0.5), expected, atol=0, rtol=0)
        draws = "aten::scaled_dot_product_attention draws random numbers"  # noqa: F841

    # What torch.vm returns does not forward `cache_clear` of the `lru_cache` function it was given.
    @unittest.expectedFailure
    @isolated
    def test_a_call_that_finds_its_entry_in_a_cache_is_part_of_a_path(self, device):
        def reset():
            GRAPH_CACHE_LOG.clear()
            for wrapper in GRAPH_CACHES:
                wrapper.cache_clear()

        def keep(gm, input_types):
            return gm

        for fn in (graph_calls_cached, graph_cache_is_cleared):
            reset()
            x = torch.randn(3, 4, device=device)
            fn(x)
        functions = (
            graph_calls_cached,
            graph_cache_is_cleared,
            graph_cache_is_bounded,
            graph_cache_is_typed,
            graph_cache_by_name,
            graph_cache_of_loud_key,
            graph_cache_is_counted,
        )
        for fn in functions:
            for behind in ((), (keep,)):
                passes = (*EARLY, *behind)  # noqa: F841
                found = []
                for got, expected in zip(found[0], found[1]):
                    self.assertEqual(got[1], expected[1])
                    self.assertEqual(got[2], expected[2])
                    self.assertEqual(got[0], expected[0], atol=0, rtol=0)
        reset()

    @isolated
    def test_a_list_that_is_added_to_in_place_is_changed_once(self, device):
        def keep(gm, input_types):
            return gm

        x = torch.randn(3, 4, device=device)
        passes = (*EARLY, keep)
        compiled = vm(graph_grows_a_list, passes=passes)
        for given in ([3, 2], [3, 2], [3, 2], [3, 2], [6, 1], [3, 2], [6, 1], [1, 6]):
            mine = list(given)
            expected = graph_grows_a_list(x, list(given))
            self.assertEqual(compiled(x, mine), expected, atol=0, rtol=0)
            self.assertEqual(mine, [*given, 2])

    @isolated
    def test_contiguous_copies_where_a_tensor_is_not(self, device):
        base = torch.randn(4, 4, device=device)
        passes = (*EARLY, vm.passes.Inductor())
        behind = vm(graph_keeps_or_copies, passes=passes)
        for x in (base, base, base, base.t(), base, base.t()):
            self.assertEqual(
                behind(x), graph_keeps_or_copies(x), rtol=0.0001, atol=1e-05
            )

    @isolated
    def test_a_key_that_comes_to_be_compared_in_python_is_compared_once(self, device):
        def keep(gm, input_types):
            return gm

        passes = (*EARLY, keep)
        fn = graph_reads_a_table
        compiled = vm(fn, passes=passes)
        x = torch.randn(3, 4, device=device)
        for _ in range(6):
            compiled(x, {"b": 1.0, "a": 2.0}, {"b", "a"})
        for other in (x, x.double(), x[0], x.long(), x):
            mine, theirs = (GraphCollides(), GraphCollides())
            got, there = compiled(other, {mine: 1.0, "a": 2.0}, {mine, "a"})
            expected = fn(other, {theirs: 1.0, "a": 2.0}, {theirs, "a"})
            self.assertEqual(got, expected[0], atol=0, rtol=0)
            self.assertEqual((there, mine.log), (expected[1], theirs.log))

    @isolated
    def test_a_layer_that_checks_its_input_with_a_message_is_one_graph(self, device):
        def keep(gm, input_types):
            return gm

        torch.manual_seed(0)
        layers = (GraphGuarded2d(4), GraphGuarded2d(4))
        model = torch.nn.Sequential(*layers).eval().to(device)
        x = torch.randn(2, 4, 8, 8, device=device)
        alone = vm(model, passes=(*EARLY,))
        with torch.no_grad():
            for _ in range(4):
                self.assertEqual(alone(x), model(x), atol=0, rtol=0)
        model.train()
        passes = (*EARLY, keep)  # noqa: F841
        reference = copy.deepcopy(model)
        for at in range(6):
            torch.manual_seed(at)
            torch.manual_seed(at)
        for name, value in reference.state_dict().items():
            self.assertEqual(model.state_dict()[name], value, msg=name, atol=0, rtol=0)
        alone = vm(model, passes=(*EARLY,))
        for _ in range(4):
            alone(x)

    @isolated
    def test_a_path_lets_go_of_what_holds_what_stays(self, device):
        def keep(gm, input_types):
            return gm

        layers = [torch.nn.Linear(4, 4) for _ in range(8)]
        fn = graph_end_variable
        for make in (lambda: layers[2:8], lambda: layers[0].forward):
            for behind in ((), (keep,)):
                passes = (*EARLY, *behind)
                compiled = vm(fn, passes=passes)
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
        del GRAPH_END_HELD[:]

    @isolated
    def test_a_graph_starts_behind_what_ended_a_path(self, device):
        def keep(gm, input_types):
            return gm

        fn = graph_goes_on_behind_a_call
        for behind in ((), (keep,)):
            passes = (*EARLY, *behind)
            compiled = vm(fn, passes=passes)
            found = []
            for run in (compiled, fn):
                GRAPH_BEHIND_LOG.clear()
                given = []
                for at in range(40):
                    torch.manual_seed(at)
                    x = torch.randn(3, 4, device=device)
                    given.append((run(x, at), list(GRAPH_BEHIND_LOG)))
                found.append(given)
            for got, expected in zip(found[0], found[1]):
                self.assertEqual(got[1], expected[1])
                self.assertEqual(got[0], expected[0], atol=0, rtol=0)
            first = (4, "in graph_helper_catches: a call that runs no kernel")  # noqa: F841
        GRAPH_BEHIND_LOG.clear()

    @isolated
    def test_a_path_lets_go_of_what_only_the_function_held(self, device):
        def keep(gm, input_types):
            return gm

        for layer in GRAPH_TEMPORARY_LAYERS:
            layer.to(device)
        fn = graph_drops_temporaries
        passes = (*EARLY, keep)
        compiled = vm(fn, passes=passes)
        for at in range(40):
            torch.manual_seed(at)
            x = torch.randn(3, 4, device=device)
            self.assertEqual(compiled(x, at), fn(x, at), atol=0, rtol=0)
        GRAPH_BEHIND_LOG.clear()
        for layer in GRAPH_TEMPORARY_LAYERS:
            layer.to("cpu")

    @isolated
    def test_what_a_path_holds_itself_has_no_part_in_what_it_lets_go_of(self, device):
        def keep(gm, input_types):
            return gm

        for layer in GRAPH_TEMPORARY_LAYERS:
            layer.to(device)
        fn = graph_drops_beside_much
        passes = (*EARLY, keep)
        compiled = vm(fn, passes=passes)
        for at in range(40):
            torch.manual_seed(at)
            x = torch.randn(3, 4, device=device)
            self.assertEqual(compiled(x, at), fn(x, at), atol=0, rtol=0)
        GRAPH_BEHIND_LOG.clear()
        for layer in GRAPH_TEMPORARY_LAYERS:
            layer.to("cpu")

    @isolated
    def test_a_graph_behind_a_call_is_not_made_twice(self, device):
        handed = []  # noqa: F841

        def keep(gm, input_types):
            calls = ("call_function", "call_method")  # noqa: F841
            return gm

        fn = graph_starts_twice
        passes = (*EARLY, keep)
        compiled = vm(fn, passes=passes)
        for at in range(40):
            torch.manual_seed(at)
            x = torch.randn(3, 4, device=device)
            self.assertEqual(compiled(x, at), fn(x, at), atol=0, rtol=0)
        GRAPH_BEHIND_LOG.clear()

    # A call returns a tensor where eager raises.
    @unittest.expectedFailure
    @isolated
    def test_an_instruction_that_nothing_uses_stays_where_it_can_fail(self, device):
        def keep(gm, input_types):
            return gm

        def one(fn, obj, x):
            try:
                return fn(obj, x)
            except Exception as error:
                return type(error).__name__

        def there(obj):
            obj.extra, obj.table, obj.row, obj.number = (
                [1],
                {"extra": 1},
                [1, 2, 3],
                1,
            )
            globals()["GRAPH_UNUSED_NAME"] = 1
            GRAPH_FREE[2]()

        def gone(obj):
            del obj.extra, obj.table["extra"], obj.row[1:]
            del globals()["GRAPH_UNUSED_NAME"]
            GRAPH_FREE[1]()

        def nothing(obj):
            obj.extra, obj.number = (0, 0)

        cases = [
            (graph_unused_attribute, GraphUnusedPlain, gone, "AttributeError"),
            (graph_unused_item, GraphUnusedPlain, gone, "KeyError"),
            (graph_unused_place, GraphUnusedPlain, gone, "IndexError"),
            (graph_unused_pair, GraphUnusedPlain, nothing, "ValueError"),
            (graph_unused_quotient, GraphUnusedPlain, nothing, "ZeroDivisionError"),
            (graph_unused_count, GraphUnusedPlain, nothing, "TypeError"),
            (graph_unused_global, GraphUnusedPlain, gone, "NameError"),
            (graph_unused_free, GraphUnusedPlain, gone, "NameError"),
            (graph_unused_attribute, GraphUnusedAsked, gone, "AttributeError"),
            (graph_unused_whether, GraphUnusedAsked, gone, None),
        ]
        x = torch.randn(3, 4, device=device)
        try:
            for behind in ((), (keep,)):
                for fn, cls, change, raised in cases:
                    obj = cls()
                    there(obj)
                    passes = (*EARLY, *behind)
                    compiled = vm(fn, passes=passes)
                    msg = f"{fn.__name__}, {cls.__name__}, with a pass: {bool(behind)}"
                    for at in range(30):
                        if at == 10:
                            change(obj)
                        if at == 20:
                            there(obj)
                        got, expected = (one(compiled, obj, x), one(fn, obj, x))
                        said = f"{msg}, call {at}: {got}, {expected}"
                        self.assertEqual(type(got), type(expected), msg=said)
                        self.assertEqual(got, expected, atol=0, rtol=0, msg=said)
                        if raised is not None:
                            absent = 10 <= at < 20
                            self.assertEqual(expected == raised, absent, msg=said)
            fn = graph_unused_attribute
            for behind in ((), (keep,)):
                obj = GraphUnusedPlain()
                there(obj)
                passes = (*EARLY, *behind)
                compiled = vm(fn, passes=passes)
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

    # A call returns a tensor where eager raises.
    @unittest.expectedFailure
    @isolated
    def test_a_function_a_path_went_into_binds_what_it_is_not_handed(self, device):
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
        fewer, other = (((1.0,), {"a": 1.0}), ((3.0, 4.0), {"a": 3.0, "b": 4.0}))
        schedule = [whole, fewer, other, whole]
        try:
            for behind in ((), (keep,)):
                for fn in (graph_default_moves, graph_default_by_name_goes):
                    put(whole)
                    passes = (*EARLY, *behind)
                    compiled = vm(fn, passes=passes)
                    raised = 0
                    for at in range(40):
                        put(schedule[at // 10])
                        got, expected = (one(compiled, x), one(fn, x))
                        said = f"{fn.__name__}, call {at}: {got}, {expected}"
                        self.assertEqual(type(got), type(expected), msg=said)
                        self.assertEqual(got, expected, atol=0, rtol=0, msg=said)
                        raised += expected == "TypeError"
                    self.assertEqual(raised, 10)
        finally:
            put(whole)

    @isolated
    def test_a_lookup_that_finds_nothing_under_getattr_with_a_default(self, device):
        def keep(gm, input_types):
            return gm

        x = torch.randn(3, 4, device=device)
        asked = (graph_asks_a_config, GraphConfig)
        for fn, made in (asked, (graph_asks_a_plain_object, GraphUnusedPlain)):
            for behind in ((), (keep,)):
                obj = made()
                passes = (*EARLY, *behind)
                compiled = vm(fn, passes=passes)
                for at in range(60):
                    if at == 40:
                        obj.is_causal, obj.scale = (False, 4.0)
                    if at == 44:
                        del obj.is_causal
                        obj.extra = 1
                        if made is GraphUnusedPlain:
                            del obj.scale
                    if at == 48:
                        del obj.extra
                        obj.scale = 2.0
                    msg = f"{fn.__name__}, call {at}"
                    got, expected = (compiled(obj, x), fn(obj, x))
                    self.assertEqual(got, expected, atol=0, rtol=0, msg=msg)

    @isolated
    def test_an_abstract_class_that_has_kept_its_answer_is_on_a_path(self, device):
        def keep(gm, input_types):
            return gm

        x = torch.randn(3, 4, device=device)
        fn = graph_asks_an_abstract_class
        for behind in ((), (keep,)):

            class Later:
                pass

            def given(at):
                kinds = (GraphKeptOne, GraphKeptNot, GraphKeptOne, Later, GraphKeptSays)
                return kinds[at % 5]()

            passes = (*EARLY, *behind)
            compiled = vm(fn, passes=passes)
            seen = []
            for run in (compiled, fn):
                del GRAPH_UNUSED_LOG[:]
                found = []
                for at in range(50):
                    if at == 25 and run is compiled:
                        GraphKept.register(Later)
                    out = run(given(at), x)
                    registered = isinstance(Later(), GraphKept)
                    found.append((out, registered, list(GRAPH_UNUSED_LOG)))
                seen.append(found)
            for at, (got, expected) in enumerate(zip(*seen)):
                said = f"call {at}"
                if at % 5 != 3 or at >= 25:
                    self.assertEqual(got[0], expected[0], atol=0, rtol=0, msg=said)
                self.assertEqual(got[2], expected[2], msg=said)
            self.assertTrue(all(row[1] for row in seen[0][25:]))
            self.assertFalse(any(row[1] for row in seen[0][:25]))
        del GRAPH_UNUSED_LOG[:]

    @isolated
    def test_a_method_of_a_dict_that_was_bound_earlier_is_the_method(self, device):
        def keep(gm, input_types):
            return gm

        x = torch.randn(3, 4, device=device)
        fn = graph_binds_a_method_of_a_dict
        for behind in ((), (keep,)):
            table = {"a": 2.0}
            passes = (*EARLY, *behind)
            compiled = vm(fn, passes=passes)
            for at in range(30):
                if at == 15:
                    table["b"] = 5.0
                key = "a" if at % 3 else "b"
                got, expected = (compiled(table, x, key), fn(table, x, key))
                self.assertEqual(got, expected, atol=0, rtol=0)

    @isolated
    def test_a_method_read_off_a_list_the_path_made_keeps_nothing_back(self, device):
        graphs = []  # noqa: F841

        def keep(gm, input_types):
            return gm

        def bound(xs):
            out, table = ([], {"scale": 2.0})
            put, get = (out.append, table.get)
            for x in xs:
                put(torch.sin(x) * get("scale"))
            return torch.tanh(torch.cat(out)).sum()

        passes = (*EARLY, keep)
        for fn in (graph_collects, bound):
            compiled = vm(fn, passes=passes)
            xs = [torch.randn(3, device=device) for _ in range(3)]
            for _ in range(6):
                self.assertEqual(compiled(xs), fn(xs), atol=0, rtol=0)

    @isolated
    def test_a_draw_that_a_path_does_not_take_says_what_is_in_its_way(self, device):
        def keep(gm, input_types):
            return gm

        x = torch.randn(3, 4, device=device)
        fn = graph_draws_behind_a_branch
        draws = "in dropout: aten::dropout draws random numbers"
        for behind, why in (((), draws), ((keep,), draws + " behind what can fail")):
            passes = (*EARLY, *behind)
            compiled = vm(fn, passes=passes)
            for _ in range(6):
                torch.manual_seed(3)
                expected = fn(x)
                torch.manual_seed(3)
                self.assertEqual(compiled(x), expected, atol=0, rtol=0)

    @isolated
    def test_the_truth_of_what_has_a_len_is_on_a_path(self, device):
        def keep(gm, input_types):
            return gm

        x = torch.randn(3, 4, device=device)
        fn = graph_asks_whether_any
        for made in (GraphCounted, GraphCountedLoudly):
            for behind in ((), (keep,)):
                passes = (*EARLY, *behind)
                compiled = vm(fn, passes=passes)
                seen = []
                for run in (compiled, fn):
                    del GRAPH_UNUSED_LOG[:]
                    held, given = (made(), [])
                    for at in range(40):
                        if at in (20, 30):
                            held.items.append(at)
                        if at == 35:
                            del held.items[:]
                        given.append((run(held, x), len(GRAPH_UNUSED_LOG)))
                    seen.append(given)
                for at, (got, expected) in enumerate(zip(*seen)):
                    said = f"call {at}"
                    self.assertEqual(got[1], expected[1], msg=said)
                    self.assertEqual(got[0], expected[0], atol=0, rtol=0, msg=said)
        del GRAPH_UNUSED_LOG[:]

    @isolated
    def test_the_truth_of_what_has_a_len_is_asked_of_the_class_every_call(self, device):
        def keep(gm, input_types):
            return gm

        def says_no(self):
            return False

        def branches(x, held):
            h = torch.relu(x) + 1
            if held:
                h = h * 3
            return (torch.tanh(h + 2) * 2 - 1).sum()

        def negates(x, held):
            h = torch.relu(x) + 1
            if not held:
                h = h * 3
            return (torch.tanh(h + 2) * 2 - 1).sum()

        def chooses(x, held):
            h = torch.relu(x) + 1
            return (torch.tanh(h + (held and 4.0)) * 2 - 1).sum()

        class Other:
            __bool__ = says_no

        def changes(lower, sized):
            def on_the_class():
                sized.__bool__ = says_no

            def none_on_the_class():
                sized.__bool__ = None

            def on_a_base():
                lower.__bool__ = says_no

            def other_bases():
                sized.__bases__ = (Other,)

            return [on_the_class, none_on_the_class, on_a_base, other_bases]

        def one(fn, x, held):
            try:
                return fn(x, held).item()
            except TypeError as error:
                return str(error)

        for fn in (branches, negates, chooses):
            for which in range(4):
                for behind in ((), (keep,)):

                    class Lower:
                        pass

                    class Sized(Lower):
                        def __len__(self):
                            return 3

                    change = changes(Lower, Sized)[which]
                    passes = (*EARLY, *behind)
                    compiled = vm(fn, passes=passes)
                    held = Sized()
                    msg = f"{fn.__name__}, {change.__name__}, a pass: {bool(behind)}"
                    for at in range(18):
                        if at == 7:
                            change()
                        if at == 12 and which == 0:
                            del Sized.__bool__
                        x = torch.full((3, 4), 0.25 * (at + 1), device=device)
                        got, expected = (one(compiled, x, held), one(fn, x, held))
                        said = f"{msg}, call {at}: {got}, {expected}"
                        self.assertEqual(type(got), type(expected), msg=said)
                        self.assertTrue(got == expected, msg=said)

    @isolated
    def test_an_operation_is_handed_what_a_list_held_where_it_was_spread(self, device):
        def keep(gm, input_types):
            return gm

        for fn in (graph_spread_list, graph_spread_tuple, graph_spread_twice):
            for behind, graded in (((), False), ((keep,), False), ((), True)):
                passes = (*EARLY, *behind)
                compiled = vm(fn, passes=passes)
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

    @isolated
    def test_what_stands_in_front_of_an_instruction_that_raises_is_a_path(self, device):
        def keep(gm, input_types):
            return gm

        schedules = {
            "always": lambda at: True,
            "every other call": lambda at: at % 2 == 0,
            "calls 7 and 10": lambda at: at in (7, 10),
            "calls 1 and 2": lambda at: at in (1, 2),
        }
        raised = (KeyError, ZeroDivisionError, ValueError)

        def calls(run, raises, graded, count):
            del GRAPH_RAISE_SEEN[:]
            found = []
            for at in range(count):
                x = torch.full((3, 4), 0.25 * (at + 1), device=device)
                x.requires_grad_(graded)
                try:
                    out = run(x, "missing" if raises(at) else "there")
                    if graded:
                        out.backward()
                    grad = None if x.grad is None else x.grad.sum().item()
                    found.append((out.item(), grad, list(GRAPH_RAISE_SEEN)))
                except raised as error:
                    said = (type(error).__name__, str(error))
                    found.append((*said, list(GRAPH_RAISE_SEEN)))
            return found

        functions = (
            graph_raises_out,
            graph_raises_and_handles,
            graph_raises_dividing,
            graph_raises_unpacking,
            graph_stores_and_raises,
            graph_raises_in_what_it_calls,
        )
        ways = (((), False), ((keep,), False), ((), True))
        for fn in functions:
            for when, raises in schedules.items():
                for behind, graded in ways:
                    exported = bool(behind) or graded
                    count = 24 if exported else 40
                    passes = (*EARLY, *behind)
                    compiled = vm(fn, passes=passes)
                    msg = f"{fn.__name__}, raises {when}, pass {bool(behind)}, gradient {graded}"
                    got = calls(compiled, raises, graded, count)
                    expected = calls(fn, raises, graded, count)
                    self.assertTrue(got == expected, msg=f"{msg}: {got} {expected}")
                    if not exported:
                        continue
                    returns = "the function returns"  # noqa: F841
                    if when == "calls 7 and 10":
                        if behind and fn is graph_raises_out:
                            GRAPH_RAISE_OUT["there"] = 3.0
                            try:
                                x = torch.full((3, 4), 0.5, device=device)
                                self.assertEqual(compiled(x, "there"), fn(x, "there"))
                            finally:
                                GRAPH_RAISE_OUT["there"] = 2.0

    @isolated
    def test_a_short_way_that_ends_the_call_is_not_taken_down_again(self, device):
        def keep(gm, input_types):
            return gm

        def calls(run, graded, count):
            found = []
            for at in range(count):
                value = 0.25 * (at % 3 + 1)
                x = torch.full((3, 4), value, device=device, requires_grad=graded)
                try:
                    out = run(x, 5 if at % 2 else 1)
                    if graded:
                        out.backward()
                    grad = None if x.grad is None else x.grad.sum().item()
                    found.append((out.item(), grad))
                except ValueError as error:
                    found.append((type(error).__name__, str(error)))
            return found

        count = 40
        for fn in (graph_short_way_returns, graph_short_way_raises):
            for behind, graded in (((), False), ((keep,), False), ((), True)):
                passes = (*EARLY, *behind)
                compiled = vm(fn, passes=passes)
                msg = f"{fn.__name__}, with a pass: {bool(behind)}, with a gradient: {graded}"
                got, expected = (
                    calls(compiled, graded, count),
                    calls(fn, graded, count),
                )
                self.assertTrue(got == expected, msg=f"{msg}: {got} {expected}")

    # The tensors the function writes to (the statistics of a batch norm) differ from eager's.
    @unittest.expectedFailure
    @isolated
    def test_a_take_that_writes_to_an_outside_tensor_is_a_path_of_its_own(self, device):
        orders = ((True, True), (True, False), (False, True), (False, False))
        for writes, first in orders:
            ours = (torch.zeros(4, device=device), torch.ones(4, device=device))
            theirs = (torch.zeros(4, device=device), torch.ones(4, device=device))
            raised = 0
            for at in range(12):
                training = first if at < 5 else not first
                x = torch.full((2, 4, 5, 5), 0.5 * (at + 1), device=device)
                x[0] += 1
                expected = graph_normed(x, *theirs, training)  # noqa: F841
                msg = f"what writes fails: {writes}, the first calls train: {first}"
                said = f"{msg}, call {at}"
                self.assertEqual(ours, theirs, atol=0, rtol=0, msg=said)
            if writes:
                self.assertGreaterEqual(raised, 1, msg=msg)
            else:
                self.assertEqual(raised, 0, msg=msg)

    @isolated
    def test_a_read_of_a_size_of_a_tensor_whose_class_was_assigned_runs_as_it_is_written(
        self, device
    ):
        def keep(gm, input_types):
            return gm

        def calls(fn, graded):
            rows = []
            for at in range(8):
                del VM_KLASS_LOG[:]
                x = (
                    torch.arange(6.0, device=device).reshape(2, 3) + at
                ).requires_grad_(graded)
                try:
                    out = fn(x)
                    grad = torch.autograd.grad(out, x)[0].tolist() if graded else None
                    rows.append(
                        (out.tolist(), grad, type(x).__name__, list(VM_KLASS_LOG))
                    )
                except Exception as error:
                    rows.append((type(error).__name__, str(error)))
                    x.__class__ = torch.Tensor
            return rows

        graphs = (*EARLY,)  # noqa: F841
        for to in (VmKlassLogs, VmKlassLies, VmKlassPlain):
            VM_KLASS_TO[0] = to
            self.assertEqual(calls(vm_klass_case_shape, False)[0][0] != 0, True)
        VM_KLASS_TO[0] = VmKlassLies
        lied = calls(vm_klass_case_shape, False)
        VM_KLASS_TO[0] = VmKlassPlain
        self.assertTrue(lied != calls(vm_klass_case_shape, False))

    # With an attribute put on the tensor class, the results differ from eager's.
    @unittest.expectedFailure
    @isolated
    def test_what_is_put_on_the_tensor_class_is_called_wherever_a_pass_stands_for_a_read(
        self, device
    ):
        def keep(gm, input_types):
            return gm

        def calls(fn, name, put):
            rows = []
            try:
                for at in range(10):
                    size = 2 + 3 * (at % 2)
                    x = (
                        torch.arange(float(size * 3), device=device).reshape(size, 3)
                        + at
                    ) / 7
                    if at == 4:
                        setattr(torch.Tensor, name, put)
                    if at == 8:
                        delattr(torch.Tensor, name)
                    rows.append(fn(x).tolist())
            finally:
                if name in torch.Tensor.__dict__:
                    delattr(torch.Tensor, name)
            return rows

        base = torch._C.TensorBase
        shape, ndim, size, dim = (base.shape, base.ndim, base.size, base.dim)
        puts = {
            "shape": property(
                lambda self: torch.Size(
                    [2 * shape.__get__(self)[0], *shape.__get__(self)[1:]]
                )
            ),
            "ndim": property(lambda self: ndim.__get__(self) + 1),
            "size": lambda self, *at: 2 * size(self, *at) if at else size(self),
            "dim": lambda self: dim(self) + 1,
        }
        graphs = (*EARLY,)
        ways = (
            ("the default passes", EARLY),
            ("graphs", graphs),
            ("a pass", (*graphs, keep)),
        )
        for fn in (
            vm_asks_case_reads,
            vm_asks_case_compared_behind_a_kernel,
            vm_asks_case_in_a_loop,
        ):
            plain = calls(fn, "shape", shape)
            for name, put in puts.items():
                for way, passes in ways:
                    compiled = vm(fn, passes=passes)
                    msg = f"{fn.__name__}, `{name}`, {way}"
                    expected = calls(fn, name, put)
                    self.assertEqual(calls(compiled, name, put), expected, msg=msg)
                    if fn is vm_asks_case_reads:
                        self.assertTrue(expected[4:8] != plain[4:8], msg=msg)
                    self.assertEqual(
                        expected[:4] + expected[8:], plain[:4] + plain[8:], msg=msg
                    )
                    if way == "the default passes":
                        compiled(torch.ones(2, 3, device=device))

    # The text of an exception is asked of Python code that eager does not run.
    @unittest.expectedFailure
    @isolated
    def test_the_text_of_what_was_raised_is_asked_only_where_no_code_makes_it(
        self, device
    ):
        def raises(error):
            def failing(gm, input_types):
                raise error

            return failing

        def made_raises(error):
            def wraps(gm, input_types):
                inner, ran = (gm.forward, [])

                def forward(*args):
                    if not ran:
                        ran.append(True)
                        raise error
                    return inner(*args)

                gm.forward = forward
                return gm

            return wraps

        graphs = (*EARLY,)
        cases = (
            (VmSaysLoud("x"), "VmSaysLoud"),
            (ValueError(VmSaysArgument()), "ValueError"),
            (OSError(2, "gone", VmSaysArgument()), "FileNotFoundError"),
            (UnicodeDecodeError("utf-8", b"x", 0, 1, "odd"), "UnicodeDecodeError"),
            (OSError(2, "gone", "there"), "FileNotFoundError: [Errno 2] gone: 'there'"),
            (VmSaysPlain("not today"), "VmSaysPlain: not today"),
            (KeyError("missing"), "KeyError: 'missing'"),
            (RuntimeError("two", 2), "RuntimeError: ('two', 2)"),
            (TypeError(), "TypeError: "),
        )
        places = (
            ("a pass", raises, "the pass failing raised "),
            ("what a pass made", made_raises, "the compiled graph failed: "),
        )
        for error, said in cases:
            for where, make, reason in places:
                msg = f"{type(error).__name__} raised by {where}"
                compiled = vm(vm_says_case, passes=(*graphs, make(error)))
                for at in range(6):
                    x = torch.arange(6.0, device=device) / 7 + at
                    del VM_SAYS_LOG[:]
                    self.assertEqual(
                        compiled(x).tolist(), vm_says_case(x).tolist(), msg=msg
                    )
                    self.assertEqual(VM_SAYS_LOG, [], msg=msg)
        with (
            unittest.mock.patch("linecache.getline") as getline,
            unittest.mock.patch("linecache.checkcache") as checkcache,
        ):
            compiled = vm(vm_says_case, passes=(*graphs, vm_says_pass_that_raises_deep))
            for at in range(4):
                x = torch.arange(6.0, device=device) / 7 + at
                self.assertEqual(compiled(x).tolist(), vm_says_case(x).tolist())
        line = vm_says_helper.__code__.co_firstlineno + 1  # noqa: F841
        self.assertEqual((getline.call_count, checkcache.call_count), (0, 0))

        class Run:
            def __init__(self, replay):
                self.replay = replay
                self.inputs, self.outputs = replay.io(replay.start, replay.end)

            def __repr__(self):
                VM_SAYS_LOG.append("repr of a callable")
                return "a run"

            def __call__(self, *args):
                env = dict(zip(self.inputs, args))
                self.replay.run(self.replay.start, self.replay.end, env)
                return tuple(env[value] for value in self.outputs)

        def scaled(x, y, k):
            a = x * y + k
            b = a - x
            return (a, b * 2)

        del VM_SAYS_LOG[:]
        for k in range(6):
            x, y = (torch.rand(3, device=device), torch.rand(3, device=device))  # noqa: F841
        self.assertEqual(VM_SAYS_LOG, [])

    # The results, or the calls of a key's `__eq__` between the tensor operations, differ from eager's.
    @unittest.expectedFailure
    @isolated
    def test_a_key_that_compares_in_python_is_compared_once_per_lookup(self, device):
        def keep(gm, input_types):
            return gm

        def shorter(a, b):
            return a if len(a[2]) <= len(b[2]) else b

        def calls(run, make, change, back, when, graded):
            found = []
            config = make()
            if when == "always" and change:
                change(config)
            try:
                for at in range(14):
                    late = when == "late" and at in (7, 10)
                    if late:
                        change(config)
                    x = torch.full((3, 4), 0.25 * (at + 1), device=device)
                    x.requires_grad_(graded)
                    del GRAPH_NAME_LOG[:]
                    try:
                        out = run(x, config)
                        if graded:
                            out.backward()
                        grad = None if x.grad is None else x.grad.sum().item()
                        found.append((out.item(), grad, list(GRAPH_NAME_LOG)))
                    finally:
                        if late:
                            back(config)
            finally:
                if when == "always" and back:
                    back(config)
                del GRAPH_NAME_LOG[:]
            return found

        ways = (((), False), ((keep,), False), ((), True))
        for name, (fn, make, change, back) in GRAPH_NAME_CASES.items():
            for when in ("always", "late") if change else ("always",):
                for behind, graded in ways:
                    passes = (*EARLY, *behind)
                    compiled = vm(fn, passes=passes)
                    msg = f"{name}, {when}, pass {bool(behind)}, gradient {graded}"
                    got = calls(compiled, make, change, back, when, graded)
                    eager = (fn, make, change, back, when, graded)
                    twice = [calls(*eager) for _ in range(2)]
                    expected = [shorter(a, b) for a, b in zip(*twice)]
                    self.assertTrue(got == expected, msg=f"{msg}: {got} {expected}")
                    there = [at for at, row in enumerate(expected) if row[2]]
                    wanted = list(range(14)) if when == "always" else [7, 10]
                    self.assertEqual(there, wanted, msg=msg)

    # Once a region runs, the results differ from eager's (first failing case: a function whose defaults are rebound).
    @unittest.expectedFailure
    @isolated
    def test_a_form_runs_only_under_the_checks_of_its_path(self, device):
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
                    passes = (*EARLY, *behind)
                    compiled = vm(fn, passes=passes)
                    msg = f"{fn.__name__}, a pass: {bool(behind)}, a gradient: {graded}"
                    got = calls(compiled, usual, other, graded)
                    expected = calls(fn, usual, other, graded)
                    self.assertTrue(got == expected, msg=f"{msg}: {got} {expected}")
            passes = (*EARLY, keep)
            ten = vm(graph_pins_ten, passes=passes)
            for at in range(30):
                x = torch.full((3, 4), 0.25 * (at + 1), device=device)
                got, expected = (ten(x, at % 10), graph_pins_ten(x, at % 10))
                self.assertEqual(got, expected, atol=0, rtol=0)
        finally:
            graph_pinned_scaled.__defaults__ = (2.0, 1.0)

    @isolated
    def test_a_change_that_can_fail_is_not_left_out_with_what_it_changes(self, device):
        def keep(gm, input_types):
            return gm

        cases = (
            (graph_key_dict_of_a_comprehension, lambda: 1, lambda: [1]),
            (graph_key_list_of_what_is_spread, lambda: (1, 2), lambda: 5),
            (graph_key_dict_of_what_is_spread, lambda: {"a": 1}, lambda: 5),
            (graph_key_stored_and_dropped, lambda: 1, lambda: [1]),
            (graph_key_popped_and_dropped, lambda: "k", lambda: [1]),
            (graph_key_popped_and_dropped, lambda: "k", lambda: "gone"),
            (graph_key_popped_with_a_default, lambda: "k", lambda: [1]),
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
            for behind, graded in (((), False), ((keep,), False), ((), True)):
                passes = (*EARLY, *behind)
                compiled = vm(fn, passes=passes)
                msg = f"{fn.__name__} with {usual()!r} and {other()!r}"
                msg += f", a pass: {bool(behind)}, a gradient: {graded}"
                got = calls(compiled, usual, other, graded)
                expected = calls(fn, usual, other, graded)
                self.assertTrue(got == expected, msg=f"{msg}: {got} {expected}")

    # The results, or how often the Python code of a dict key runs, differ from eager's.
    @unittest.expectedFailure
    @isolated
    def test_the_code_of_a_key_runs_as_often_as_the_instructions_run_it(self, device):
        def keep(gm, input_types):
            return gm

        def logs(key):
            return functools.partial(GraphKeyText, key)

        spread = graph_key_dict_of_what_is_spread
        cases = (
            (graph_key_stored_and_read, lambda: "k", logs("k")),
            (graph_key_stored_and_read, logs("k"), logs("k")),
            (graph_key_spread, logs("scale"), logs("shift")),
            (spread, lambda: {"a": 1}, lambda: {GraphKeyText("a"): 1}),
            (graph_key_stored_and_dropped, lambda: "k", lambda: GraphKeyRaises("k")),
            (graph_key_stored_and_dropped, lambda: "k", logs("k")),
            (graph_key_popped_with_a_default, lambda: "k", logs("k")),
            (graph_key_spread, lambda: "scale", logs("scale")),
            (graph_key_spread, lambda: 1, lambda: GraphKeyNumber(1)),
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
                for behind, graded in (((), False), ((keep,), False), ((), True)):
                    passes = (*EARLY, *behind)
                    compiled = vm(fn, passes=passes)
                    msg = f"{fn.__name__} with {usual()!r} and {other()!r}"
                    msg += f", a pass: {bool(behind)}, a gradient: {graded}"
                    got = calls(compiled, usual, other, graded)
                    expected = calls(fn, usual, other, graded)
                    self.assertTrue(got == expected, msg=f"{msg}: {got} {expected}")
                    plain = not isinstance(usual(), (GraphKeyText, int))  # noqa: F841
        finally:
            del GRAPH_KEY_LOG[:]

    @isolated
    def test_a_graph_is_within_a_callers_only_where_its_path_goes_through(self, device):
        def keep(gm, input_types):
            return gm

        member = (GraphWithinBehindAMember, True)
        models = ((GraphWithinModel, False), (GraphWithinModel, True), member)
        for model_class, loud in models:
            for graded in (False, True):
                model = model_class(loud).to(device)
                passes = (*EARLY, keep)
                compiled = vm(model, passes=passes)
                msg = f"{model_class.__name__}, loud: {loud}, a gradient: {graded}"
                for at in range(12):
                    x = torch.full((3, 4), 0.25 * (at + 1), device=device)
                    x.requires_grad_(graded)
                    self.assertEqual(compiled(x), model(x), msg=msg)

    @isolated
    def test_a_method_that_an_object_holds_appends_to_its_own_list(self, device):
        def keep(gm, input_types):
            return gm

        def calls(run, graded):
            for held in GRAPH_OUTSIDE_LISTS:
                del held[:]
            found = []
            for at in range(14):
                x = torch.full((3, 4), 0.25 * (at + 1), device=device)
                x.requires_grad_(graded)
                out = run(x, 1 if at in (7, 10) else 0)
                if graded:
                    out.backward()
                found.append((out.item(), [len(held) for held in GRAPH_OUTSIDE_LISTS]))
            return found

        fn = graph_appends_by_a_method_of_another_list
        try:
            for behind, graded in (((), False), ((keep,), False), ((), True)):
                passes = (*EARLY, *behind)
                compiled = vm(fn, passes=passes)
                msg = f"with a pass: {bool(behind)}, with a gradient: {graded}"
                got, expected = (calls(compiled, graded), calls(fn, graded))
                self.assertTrue(got == expected, msg=f"{msg}: {got} {expected}")
        finally:
            for held in GRAPH_OUTSIDE_LISTS:
                del held[:]

    @isolated
    def test_a_pop_by_a_method_read_off_earlier_is_the_change_pop_is(self, device):
        def keep(gm, input_types):
            return gm

        cases = (
            (graph_pops_by_a_method_read_earlier, 2.0, 3.0, True),
            (graph_pops_by_a_method_that_alone_has_the_dict, 2.0, 3.0, True),
            (graph_pops_with_a_default_it_calls_for, True, False, True),
            (graph_pops_by_a_method_of_another_dict, "gone", "kept", False),
        )

        def calls(run, usual, other, graded):
            GRAPH_NAMED.clear()
            GRAPH_NAMED["kept"] = 4.0
            found = []
            for at in range(14):
                x = torch.full((3, 4), 0.25 * (at + 1), device=device)
                x.requires_grad_(graded)
                out = run(x, other if at in (7, 10) else usual)
                if graded:
                    out.backward()
                grad = None if x.grad is None else x.grad.sum().item()
                found.append((out.item(), grad, sorted(GRAPH_NAMED)))
            return found

        try:
            for fn, usual, other, whole in cases:
                for behind, graded in (((), False), ((keep,), False), ((), True)):
                    passes = (*EARLY, *behind)
                    compiled = vm(fn, passes=passes)
                    msg = f"{fn.__name__}, a pass: {bool(behind)}, a gradient: {graded}"
                    got = calls(compiled, usual, other, graded)
                    expected = calls(fn, usual, other, graded)
                    self.assertTrue(got == expected, msg=f"{msg}: {got} {expected}")
        finally:
            GRAPH_NAMED.clear()

    @isolated
    def test_an_attribute_of_a_class_whose_class_reads_as_type_does(self, device):
        def keep(gm, input_types):
            return gm

        def logs(cls):
            GRAPH_META_LOG.append("read")
            return 3.0

        def calls(run, graded):
            del GRAPH_META_LOG[:]
            found = []
            for at in range(16):
                if at in (9, 10, 13):
                    GraphMeta.scale = property(logs)
                x = torch.full((3, 4), 0.25 * (at + 1), device=device)
                x.requires_grad_(graded)
                out = run(x, GraphColor.RED if at % 4 else GraphColor.BLUE)
                if graded:
                    out.backward()
                found.append((out.item(), len(GRAPH_META_LOG)))
                if "scale" in vars(GraphMeta):
                    del GraphMeta.scale
            return found

        fn = graph_reads_of_classes
        try:
            for behind, graded in (((), False), ((keep,), False), ((), True)):
                passes = (*EARLY, *behind)
                compiled = vm(fn, passes=passes)
                msg = f"with a pass: {bool(behind)}, with a gradient: {graded}"
                got, expected = (calls(compiled, graded), calls(fn, graded))
                self.assertTrue(got == expected, msg=f"{msg}: {got} {expected}")
        finally:
            if "scale" in vars(GraphMeta):
                del GraphMeta.scale
            del GRAPH_META_LOG[:]

    @isolated
    def test_a_comparison_whose_classes_compare_in_c_is_part_of_a_path(self, device):
        def keep(gm, input_types):
            return gm

        plain, wide = (GraphQuietText("narrow"), GraphQuietText("wide"))
        low, high = (GraphQuietText("a"), GraphQuietText("z"))
        cases = (
            (graph_compares_members, None, 1, False),
            (graph_compares_which_it_is, GraphPlace.ONE, GraphPlace.TWO, False),
            (graph_compares_which_it_is, GraphPlace.ONE, None, False),
            (graph_compares_texts, plain, wide, False),
            (graph_compares_texts, plain, GraphLoudText("wide"), False),
            (graph_compares_texts, plain, plain, True),
            (graph_compares_an_order, low, high, False),
            (graph_compares_an_order, low, None, False),
            (graph_compares_an_order, low, low, True),
        )

        def calls(run, usual, other, planted, graded):
            del GRAPH_COMPARE_LOG[:]
            found = []
            for at in range(14):
                x = torch.full((3, 4), 0.25 * (at + 1), device=device)
                x.requires_grad_(graded)
                if planted and at in (7, 10):
                    GraphQuietText.__eq__ = graph_loud_eq
                    GraphQuietText.__lt__ = graph_loud_eq
                try:
                    out = run(x, other if at in (7, 10) else usual)
                    if graded:
                        out.backward()
                    grad = None if x.grad is None else x.grad.sum().item()
                    said = (out.item(), grad)
                except (TypeError, ValueError) as error:
                    said = f"{type(error).__name__}: {error}"
                finally:
                    if "__eq__" in vars(GraphQuietText):
                        del GraphQuietText.__eq__
                        del GraphQuietText.__lt__
                found.append((said, list(GRAPH_COMPARE_LOG)))
            return found

        try:
            for fn, usual, other, planted in cases:
                for behind, graded in (((), False), ((keep,), False), ((), True)):
                    passes = (*EARLY, *behind)
                    compiled = vm(fn, passes=passes)
                    msg = f"{fn.__name__} with {usual!r} and {other!r}"
                    msg += f", planted: {planted}, a pass: {bool(behind)}"
                    msg += f", a gradient: {graded}"
                    got = calls(compiled, usual, other, planted, graded)
                    expected = calls(fn, usual, other, planted, graded)
                    self.assertTrue(got == expected, msg=f"{msg}: {got} {expected}")
        finally:
            if "__eq__" in vars(GraphQuietText):
                del GraphQuietText.__eq__
                del GraphQuietText.__lt__
            del GRAPH_COMPARE_LOG[:]

    # The results, or the calls of a metaclass's Python code between the tensor operations, differ from eager's.
    @unittest.expectedFailure
    @isolated
    def test_the_text_of_a_class_is_part_of_a_path(self, device):
        def keep(gm, input_types):
            return gm

        plain = GraphQuietText("narrow")
        cases = ((plain, GraphLoudText("a")), (plain, GraphOfLoudMeta()))

        def calls(run, usual, other, graded):
            del GRAPH_COMPARE_LOG[:]
            GRAPH_NAMED.clear()
            GRAPH_NAMED[str(GraphQuietText)] = 2.0
            GRAPH_NAMED[str(GraphLoudText)] = 3.0
            found = []
            for at in range(14):
                x = torch.full((3, 4), 0.25 * (at + 1), device=device)
                x.requires_grad_(graded)
                out = run(x, other if at in (7, 10) else usual)
                if graded:
                    out.backward()
                grad = None if x.grad is None else x.grad.sum().item()
                found.append((out.item(), grad, list(GRAPH_COMPARE_LOG)))
            return found

        fn = graph_text_of_a_class
        try:
            for usual, other in cases:
                for behind, graded in (((), False), ((keep,), False), ((), True)):
                    passes = (*EARLY, *behind)
                    compiled = vm(fn, passes=passes)
                    msg = f"with {type(other).__name__}, a pass: {bool(behind)}"
                    msg += f", a gradient: {graded}"
                    got = calls(compiled, usual, other, graded)
                    expected = calls(fn, usual, other, graded)
                    self.assertTrue(got == expected, msg=f"{msg}: {got} {expected}")
        finally:
            GRAPH_NAMED.clear()
            del GRAPH_COMPARE_LOG[:]

    @isolated
    def test_two_reads_found_once_give_what_the_instructions_give(self, device):
        def keep(gm, input_types):
            return gm

        class Loud(str):
            __slots__ = ()

            def __eq__(self, other):
                GRAPH_LOOKUP_LOG.append("eq")
                return str.__eq__(self, other)

            __hash__ = str.__hash__

        def scale(self):
            GRAPH_LOOKUP_LOG.append("scale")
            return 5.0

        config, options = (GraphOwnLookup(), {"shift": 0.75, "drop": 2.0})

        def values_back():
            config.width = 2.0
            options.update(shift=0.75)

        def class_back():
            del GraphOwnLookup.scale
            GraphOwnLookup.bias, GraphOwnLookup.names["wide"] = (0.25, "width")
            options.clear()
            options.update(shift=0.75, drop=2.0)

        def changes(at):
            if at == 7:
                config.width, options["shift"] = (4.0, 1.5)
                return values_back
            GraphOwnLookup.scale, GraphOwnLookup.bias = (property(scale), 0.5)
            del options["drop"]
            GraphOwnLookup.names["wide"] = "scale"
            options[Loud("shift")] = options.pop("shift")
            return class_back

        def calls(run, graded):
            del GRAPH_LOOKUP_LOG[:]
            found = []
            for at in range(14):
                x = torch.full((3, 4), 0.25 * (at + 1), device=device)
                x.requires_grad_(graded)
                back = changes(at) if at in (7, 10) else None
                try:
                    out, _ = run(x, (config, options))
                    if graded:
                        out.backward()
                    grad = None if x.grad is None else x.grad.sum().item()
                    found.append((out.item(), grad, list(GRAPH_LOOKUP_LOG)))
                finally:
                    if back is not None:
                        back()
            return found

        fn = graph_reads_by_its_own_lookup
        try:
            for behind, graded in (((), False), ((keep,), False), ((), True)):
                passes = (*EARLY, *behind)
                compiled = vm(fn, passes=passes)
                msg = f"with a pass: {bool(behind)}, with a gradient: {graded}"
                got, expected = (calls(compiled, graded), calls(fn, graded))
                self.assertTrue(got == expected, msg=f"{msg}: {got} {expected}")
                self.assertEqual(len({row[0] for row in expected}), 14, msg=msg)
        finally:
            del GRAPH_LOOKUP_LOG[:]

    @isolated
    def test_what_a_kernel_gives_is_its_callers_alone_while_a_block_is_profiled(
        self, device
    ):
        graphs = (*EARLY,)
        ways = (("the default passes", EARLY), ("graphs", graphs))
        x = torch.ones(4, 4, device=device)
        rows = torch.ones(3, 4, device=device)
        for name in VM_GIVEN_MAKERS:
            fn = vm_given_function(name)
            for way, passes in ways:
                msg = f"{name}, {way}"
                compiled = vm(fn, passes=passes)
                for at in range(6):
                    got = vm_given_outcome(
                        lambda: torch.func.vmap(lambda row: compiled(x, row))(rows)
                    )
                    expected = vm_given_outcome(
                        lambda: torch.func.vmap(lambda row: fn(x, row))(rows)
                    )
                    self.assertEqual(
                        got, expected, msg=f"{msg}, inside vmap, call {at}"
                    )
                compiled = vm(fn, passes=passes)
        for name in VM_GIVEN_PLAIN:
            fn = vm_given_plain(name)
            for way, passes in ways:
                compiled = vm(fn, passes=passes)
                for at in range(8):
                    msg = f"{name}, {way}, call {at}"
                    self.assertEqual(
                        vm_given_made(compiled, device, at),
                        vm_given_made(fn, device, at),
                        msg=msg,
                    )
        for name in VM_GIVEN_USES:
            fn = vm_given_user(name)
            for policy in (None, "sizes"):
                for way, passes in ways:
                    msg = (
                        f"{name}, a class that is asked (sizes policy {policy}), {way}"
                    )
                    klass, asked = vm_given_class(policy)
                    compiled = vm(fn, passes=passes)
                    for at in range(6):
                        got = vm_given_asks(compiled, x, klass, asked)


instantiate_device_type_tests(TestVmGraphResults, globals())


@unittest.skipUnless(HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmGraphBackward(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def alike(self, got, expected):
        self.assertIs(type(got), type(expected), f"{got!r} for {expected!r}")
        if isinstance(expected, torch.Tensor):
            self.assertEqual(got, expected, atol=0, rtol=0)
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(got), len(expected))
            for one, other in zip(got, expected):
                self.alike(one, other)
        else:
            self.assertEqual(got, expected)

    @isolated
    def test_a_tensor_written_before_a_backward_is_named_as_eager_names(self, device):
        def method(result, made):
            result.mul_(2)

        def item(result, made):
            with torch.no_grad():
                result[0].zero_()

        def operand(result, made):
            with torch.no_grad():
                made.mul_(2)

        def detached(result, made):
            result.detach_()

        def raised(fn, write, at, alone):
            torch.manual_seed(at)
            x = torch.randn(3, 4, device=device)
            w = torch.randn(4, 4, device=device, requires_grad=True)
            made = w * 2 if at % 2 else w + 2
            result, total = fn(x, made)
            write(result, made)
            with self.assertRaises(RuntimeError) as caught:
                (total if alone else total + result.sum()).backward()
            return str(caught.exception)

        cases = (
            (back_reader_keeps, method, False, "is output 0 of Mul,"),
            (back_reader_keeps, item, False, "is output 0 of Tanh,"),
            (back_reader_keeps, operand, False, "is output 0 of "),
            (back_node_keeps, method, False, "is output 0 of Tanh,"),
            (back_view_kept, method, False, "is output 0 of AsStrided,"),
            (back_reader_keeps, detached, True, "detached in-place"),
        )
        for fn, write, alone, text in cases:
            for at in range(8):
                expected = raised(fn, write, at, alone)
                self.assertIn(text, expected)
                if write is operand:
                    self.assertIn("of Mul," if at % 2 else "of Add,", expected)
            states = ONE_NODE if fn is back_node_keeps else TWO_NODES  # noqa: F841

    @isolated
    def test_a_backward_is_taken_down_whatever_cudnn_would_pick(self, device):
        torch.manual_seed(0)
        shapes = ((64, 3, 7, 7), (64, 64, 3, 3), (128, 64, 3, 3), (128, 128, 3, 3))
        made = [torch.randn(*shape, device=device) for shape in shapes]
        weights = [weight.requires_grad_() for weight in made]
        x = torch.randn(8, 3, 64, 64, device=device)
        cudnn = torch.backends.cudnn

        def step(fn):
            for w in weights:
                w.grad = None
            first, second = fn(x, *weights)
            (first + second).backward()
            return (first, second, [w.grad.clone() for w in weights])

        with cudnn.flags(benchmark=True, deterministic=False):
            for _ in range(3):
                step(back_convolves)
            for _ in range(8):
                self.assertEqual((cudnn.benchmark, cudnn.deterministic), (True, False))

    @parametrize("klass", BACK_CONV_NETS, name_fn=lambda net: net.__name__)
    @isolated
    def test_what_a_graph_gives_is_at_eagers_versions(self, device, klass):
        x = torch.randn(3, 4, device=device)
        torch.manual_seed(0)
        nets = [klass().to(device), klass().to(device)]
        nets[1].load_state_dict(nets[0].state_dict())
        before = [[buffer._version for buffer in net.buffers()] for net in nets]
        for at in range(10 if klass is BackConvNetLean else 16):
            torch.manual_seed(at)
            x = torch.randn(4, 3, 8, 8, device=device)  # noqa: F841
            y = torch.randint(0, 5, (4,), device=device)  # noqa: F841
            written = []
            for net, start in zip(nets, before):
                now = [buffer._version for buffer in net.buffers()]
                written.append([version - was for version, was in zip(now, start)])
            self.assertEqual(written[0], written[1])
        nodes = TWO_NODES if klass is BackConvNet else ONE_NODE  # noqa: F841

    @isolated
    def test_the_look_at_what_autograd_saved_runs_no_hook(self, device):
        def keep(gm, input_types):
            return gm

        graphs = (*EARLY,)
        ways = (
            ("graphs", graphs),
            ("a pass", (*graphs, keep)),
            ("eager's tape and a pass", (*EARLY, keep)),
        )

        def checkpointed(run, fn, device, at):
            ran = []

            def block(t, w):
                ran.append(at)
                return (t * w).cos() * w

            x, w = vm_hooked_leaves(device, at)
            h = checkpoint(block, x * 2.0, w, use_reentrant=False)
            y = run(h, w)
            forward = len(ran)
            y.backward()
            return (y.tolist(), x.grad.tolist(), w.grad.tolist(), forward, len(ran))

        def hooked(run, fn, device, at):
            seen = []

            def unpack(t):
                seen.append("unpack")
                return t

            x, w = vm_hooked_leaves(device, at)
            with torch.autograd.graph.saved_tensors_hooks(vm_hooked_same, unpack):
                h = (x * 2.0 * w).cos() * w
            y = run(h, w)
            forward = len(seen)
            y.backward()
            return (y.tolist(), x.grad.tolist(), w.grad.tolist(), forward, len(seen))

        for way, passes in ways:
            for case in (checkpointed, hooked):
                msg = f"{case.__name__}, {way}"
                compiled = vm(vm_hooked_writes, passes=passes)
                got = vm_hooked_rows(
                    compiled, vm_hooked_writes, device, case, range(12)
                )
                expected = vm_hooked_rows(
                    vm_hooked_writes, vm_hooked_writes, device, case, range(12)
                )
                self.assertEqual(got, expected, msg=msg)
                self.assertEqual(
                    [row[3] for row in expected],
                    [1 if case is checkpointed else 0] * 12,
                    msg=msg,
                )

    @isolated
    def test_anomaly_mode_finds_the_function_at_the_line_of_each_operation(
        self, device
    ):
        def keep(gm, input_types):
            return gm

        graphs = (*EARLY,)
        ways = (
            ("the default passes", EARLY),
            ("graphs", graphs),
            ("a pass", (*graphs, keep)),
            ("eager's tape and a pass", (*EARLY, keep)),
        )
        first = vm_hooked_rooted.__code__.co_firstlineno

        def case(under):
            def call(run, fn, device, at):
                x, w = vm_hooked_leaves(device, at)
                if not under(at):
                    y = run(x * 2.0, w)
                    y.backward()
                    return repr((y.tolist(), x.grad.tolist(), w.grad.tolist()))
                with warnings.catch_warnings(record=True) as shown:
                    warnings.simplefilter("always")
                    with torch.autograd.detect_anomaly():
                        y = run(x * 2.0, w)
                        with self.assertRaisesRegex(
                            RuntimeError, "returned nan values"
                        ):
                            y.backward()
                texts = [str(one.message) for one in shown]
                named = [
                    line
                    for text in texts
                    for line in text.split("\n")
                    if "in vm_hooked_rooted" in line
                ]
                return (
                    repr(y.tolist()),
                    [
                        int(line.split(", line ")[1].split(",")[0]) - first
                        for line in named
                    ],
                )

            return call

        for way, passes in ways:
            for name, under in (
                ("every call", lambda at: True),
                ("a warm function", lambda at: at >= 14),
                ("every other call", lambda at: at % 2 == 0),
            ):
                msg = f"{way}, {name}"
                compiled = vm(vm_hooked_rooted, passes=passes)
                got = vm_hooked_rows(
                    compiled, vm_hooked_rooted, device, case(under), range(20)
                )
                expected = vm_hooked_rows(
                    vm_hooked_rooted, vm_hooked_rooted, device, case(under), range(20)
                )
                self.assertEqual(got, expected, msg=msg)
                lines = [row[1] for at, row in enumerate(expected) if under(at)]
                self.assertTrue(
                    lines and all(found == [4] for found in lines),
                    msg=f"{msg}: {lines[:2]}",
                )


instantiate_device_type_tests(TestVmGraphBackward, globals())


if __name__ == "__main__":
    run_tests()
