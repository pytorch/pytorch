# Owner(s): ["module: dynamo"]

"""Tests of the soundness contract of torch.vm's core: what a region checks
before it runs, and what the engine may assume of a traced call.

Every class here states one rule of the contract and tests it over the whole
domain the rule speaks about (every field of what is compared for a tensor,
every instruction the engine knows), not over the cases somebody thought of.
"""

import ast
import builtins
import collections
import contextlib
import dis
import enum
import functools
import gc
import glob
import importlib
import json
import math
import opcode
import operator
import os
import re
import struct
import subprocess
import sys
import tempfile
import traceback
import types
import unittest
import unittest.mock
import warnings
import weakref

import numpy

import torch
from torch.overrides import TorchFunctionMode
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    _restore_fp32_precision,
    _snapshot_fp32_precision,
    DeterministicGuard,
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._python_dispatch import TorchDispatchMode


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.join(os.path.dirname(TEST_DIR), "torch", "_rust", "src")


class TestVmGateImports(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_torch_vm_is_importable(self):
        self.assertTrue(callable(torch.vm))


class TestVmGateClassNames(TestCase):
    """A test class of torch.vm has one name among the files that test it.
    Two classes of one name in one file: the second replaces the first, whose
    tests then never run and nothing says so. In two files: a run by class
    name, a log named after the class and a report by name cannot tell them
    apart.
    """

    hw_classification = HardwareClassification.GENERIC

    FILES = ("test_rust.py", "test_vm.py", "test_vm_modes.py", "test_vm_gate.py")
    # Defined in two files when this test was written. Nothing joins it.
    KNOWN = {"TestVmEffects": ["test_vm.py", "test_vm_modes.py"]}

    def test_no_two_test_classes_of_one_name(self):
        found = collections.defaultdict(list)
        for name in self.FILES:
            with open(os.path.join(TEST_DIR, name)) as f:
                tree = ast.parse(f.read())
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                    found[node.name].append(name)
        twice = {k: v for k, v in found.items() if len(v) > 1}
        self.assertEqual(twice, self.KNOWN)


class TestVmGateCallSites(TestCase):
    """The engine touches the program's objects by identity, by exact type and
    by C-level reads of containers. What runs code of the program (a
    ``__getattr__``, an ``__eq__``, a ``__hash__``, a ``__bool__``, a
    ``__getitem__``, any call) belongs where the executor runs an instruction
    of the program and nowhere else.

    That is not so yet. The test lists, per source file of the executor, how
    often it names one of those entries of the C API (or the pyo3 method that
    is a call of it), prints the table and holds the list of files against the
    one recorded here: a file that starts to call one fails, and a file that
    stops is taken off the record.
    """

    hw_classification = HardwareClassification.GENERIC

    # name -> (the C API by name, the pyo3 method that calls it)
    ENTRIES = {
        "GetAttr": (
            r"\bPyObject_(?:Generic)?GetAttr\w*",
            r"\.(?:getattr|getattr_opt|hasattr)\(",
        ),
        "RichCompare": (
            r"\bPyObject_RichCompare\w*",
            r"\.(?:rich_compare|eq|ne|lt|le|gt|ge)\(",
        ),
        "Hash": (r"\bPyObject_Hash\b", r"\.hash\(\)"),
        "IsTrue": (r"\bPyObject_IsTrue\b", r"\.is_truthy\(\)"),
        "GetItem": (
            r"\bPy(?:Object|Sequence|Mapping|Dict)_GetItem\w*",
            r"\.get_item\(",
        ),
        "Call": (
            r"\bPyObject_(?:Call|Vectorcall)\w*",
            r"(?<!\bself)\.call(?:0|1|_method[01]?)?\(",
        ),
    }
    CALLERS = [
        "bindings/tensor.rs",
        "bytecode_printer.rs",
        "cfg/python.rs",
        "exec/asked.rs",
        "exec/autograd.rs",
        "exec/backward.rs",
        "exec/built.rs",
        "exec/calls/run.rs",
        "exec/compilers.rs",
        "exec/covered.rs",
        "exec/effects.rs",
        "exec/graph.rs",
        "exec/joint.rs",
        "exec/made.rs",
        "exec/mod.rs",
        "exec/raises.rs",
        "exec/reads.rs",
        "exec/region.rs",
        "exec/run.rs",
        "exec/scalars.rs",
        "exec/shown.rs",
        "exec/torch_ops.rs",
        "exec/trampoline.rs",
        "exec/value_reads.rs",
        "exec/writes.rs",
    ]

    def test_callers_are_the_recorded_files(self):
        if not os.path.isdir(CRATE):
            self.skipTest("the executor's source is not beside the tests")
        table = {}
        for path in sorted(
            glob.glob(os.path.join(CRATE, "**", "*.rs"), recursive=True)
        ):
            with open(path) as f:
                text = re.sub(r"//[^\n]*", "", f.read())
            # A file that does not name pyo3 has none of its methods.
            uses_pyo3 = "pyo3" in text
            row = {
                name: len(re.findall(c_api, text))
                + (len(re.findall(method, text)) if uses_pyo3 else 0)
                for name, (c_api, method) in self.ENTRIES.items()
            }
            if any(row.values()):
                table[os.path.relpath(path, CRATE)] = row
        print("\ncalls that can run code of the program, per file of the executor:")
        print(f"{'':24}" + "".join(f"{name:>12}" for name in self.ENTRIES))
        for file, row in table.items():
            print(
                f"{file:24}"
                + "".join(f"{row[name] or '':>12}" for name in self.ENTRIES)
            )
        new = sorted(set(table) - set(self.CALLERS))
        gone = sorted(set(self.CALLERS) - set(table))
        self.assertEqual(
            (new, gone),
            ([], []),
            "files that call now and are not recorded; recorded files that no longer call",
        )


class TestVmGateHistogram(TestCase):
    """What the instructions under a trace are, counted when
    ``TORCH_RUST_GATE_HIST`` is set: every instruction of a traced body and of
    the frames under it is in a row with the classes of its operands, those
    the executor runs inline too. Without the variable nothing is counted,
    and with it the program computes what it computes without it. With
    ``TORCH_RUST_TRACE_PLAIN`` the frames under a body run their code as
    written: nothing the optimizer puts in place of an instruction.
    """

    hw_classification = HardwareClassification.GENERIC

    PROGRAM = """
import json, torch
from torch import vm

lin = torch.nn.Linear(4, 4)

def scaled(x, k):
    y = torch.relu(x) * k
    return y.sum(dim=-1)

@vm(passes=vm.configure(vm.DEFAULT_PASSES, regions={"at": 3, "observe": 1}, pure_calls=1))
def f(x):
    return scaled(lin(x), 2.0)

x = torch.ones(3, 4)
got = [f(x).tolist() for _ in range(12)]
ran = sum(r["runs"] for r in vm.regions(f) if r["state"] == "region")
print(json.dumps({"got": got, "want": scaled(lin(x), 2.0).tolist(), "ran": ran, "report": torch._rust._gate_report()}))
"""
    SWITCHES = ("GATE_HIST", "GATE_HIST_DIR", "TRACE_PLAIN")
    # What the optimizer puts in place of an instruction.
    REWRITTEN = ("Aten", "Lit", "Pinned", "Expect", "Region", "Autograd")

    def run_program(self, **switches):
        unset = [f"TORCH_RUST_{name}" for name in self.SWITCHES]
        env = {k: v for k, v in os.environ.items() if k not in unset} | switches
        command = [sys.executable, "-c", self.PROGRAM]
        ran = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=600
        )
        self.assertEqual(ran.returncode, 0, ran.stderr[-2000:])
        out = json.loads(ran.stdout.splitlines()[-1])
        # The program is the same program under every switch.
        self.assertEqual(out["got"], [out["want"]] * 12)
        self.assertGreater(out["ran"], 0)
        return out["report"]

    def test_nothing_is_counted_without_the_variable(self):
        empty = {"traces": 0, "rows": []}
        self.assertEqual(self.run_program(), empty)
        self.assertEqual(self.run_program(TORCH_RUST_GATE_HIST="0"), empty)

    def test_body_and_frames_under_it_are_counted(self):
        report = self.run_program(TORCH_RUST_GATE_HIST="1")
        self.assertGreaterEqual(report["traces"], 1)
        rows = report["rows"]
        for _, what, classes, traces, count, inline in rows:
            self.assertTrue(1 <= traces <= report["traces"], (what, classes))
            self.assertTrue(max(traces, inline) <= count, (what, classes))
            if inline:
                self.assertIn(what, ("Const", "LoadLocal", "StoreLocal"))
        # The body's own instructions and those of the frames it enters.
        self.assertTrue(any(not r[0] for r in rows))
        self.assertTrue(any(r[0] for r in rows))
        # The inline path is counted.
        self.assertGreater(sum(r[5] for r in rows), 0)
        # A call is a row by its callee and the exact class of each operand.
        keys = [(under, what, tuple(classes)) for under, what, classes, *_ in rows]
        tensor, weight = "torch.Tensor", "torch.nn.parameter.Parameter"
        call = "call c torch._C._nn.linear"
        linear = [k for k in keys if k[1] == call]
        self.assertEqual(linear, [(True, call, (tensor, weight, weight))])
        forward = "call py torch.nn.modules.linear.Linear.forward"
        self.assertIn(forward, [what for _, what, _ in keys])
        self.assertIn(("Binary(*)", (tensor, "float")), [k[1:] for k in keys])

    def test_frames_under_a_body_run_as_written_when_asked(self):
        report = self.run_program(TORCH_RUST_GATE_HIST="1", TORCH_RUST_TRACE_PLAIN="1")
        under = [r[1] for r in report["rows"] if r[0]]
        self.assertTrue(under)
        self.assertEqual([what for what in under if what in self.REWRITTEN], [])


vm = torch.vm
CHECKS_HAS_VM = getattr(torch, "_rust", None) is not None and sys.version_info[:2] == (
    3,
    12,
)

# The default pass list with regions formed early: the fourth call of a
# function is traced, and the fifth is the first to run a region.
CHECKS_EARLY = vm.configure(
    vm.DEFAULT_PASSES, regions={"at": 3, "observe": 1}, pure_calls=1
)

# Set in a child process to the id of the one test it is to run.
CHECKS_CHILD = "TORCH_VM_GATE_CHILD"


def checks_isolated(test):
    """Runs a test method in a child process of its own: the executor's
    switches are process-wide, and a test of what a region checks needs the
    regions of its own functions to form on calls it knows."""

    @functools.wraps(test)
    def wrapper(self, *args, **kwargs):
        if os.environ.get(CHECKS_CHILD) == self.id():
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
        env = {**os.environ, CHECKS_CHILD: self.id()}
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
            self.fail(f"the child process {how}:\n{ran.stderr[-6000:]}")

    return wrapper


def checks_region_runs(compiled, fn):
    """How many times the regions of the function ``fn`` have run."""
    return sum(
        r["runs"]
        for r in vm.regions(compiled)
        if r["function"] == fn.__name__ and r["state"] == "region"
    )


def checks_outcome(fn, *args):
    """What a call gives: its value, or the exception it raises."""
    try:
        return fn(*args)
    except Exception as e:
        return e


# What a region compares of a tensor: the fields of its key, as Python reads
# them. The sweep below changes one at a time.
KEY_FIELDS = {
    "class": type,
    "dtype": lambda t: t.dtype,
    "device": lambda t: t.device,
    "layout": lambda t: t.layout,
    "sizes": lambda t: tuple(t.shape),
    "strides": lambda t: t.stride() if t.layout is torch.strided else None,
    "requires_grad": lambda t: t.requires_grad,
    "conjugate_bit": lambda t: t.is_conj(),
    "negative_bit": lambda t: t.is_neg(),
    "inference": lambda t: t.is_inference(),
}


class KeySubclass(torch.Tensor):
    """A tensor class with nothing of its own."""


def key_plain(x):
    return x


def key_complex(x):
    return torch.complex(x, x.cos())


def key_made_in_inference_mode(x):
    with torch.inference_mode():
        return x.clone()


def key_other_device(x):
    # From the CPU: a tensor without data. From an accelerator: the CPU.
    return x.to("meta" if x.device.type == "cpu" else "cpu")


def key_escaped_from_vmap(x):
    kept = []
    torch.func.vmap(lambda row: kept.append(row) or row)(x.expand(2, *x.shape))
    return kept[0]


# By field of the key: the tensor of the traced call and one that differs
# from it in exactly that field, each made of a fresh (4, 8) float32 tensor.
KEY_SWEEP = {
    "class": (key_plain, lambda x: torch.nn.Parameter(x, requires_grad=False)),
    "dtype": (key_plain, lambda x: x.double()),
    "device": (key_plain, key_other_device),
    "sizes": (key_plain, lambda x: x[:1].clone()),
    "strides": (key_plain, lambda x: x.t().contiguous().t()),
    "requires_grad": (key_plain, lambda x: x.requires_grad_()),
    "conjugate_bit": (key_complex, lambda x: key_complex(x).conj()),
    "negative_bit": (key_plain, torch._neg_view),
    "inference": (key_plain, key_made_in_inference_mode),
}

# What the sweep cannot construct, and why.
KEY_NOT_SWEPT = {
    "layout": "a tensor that is not strided has no strides either, and no "
    "key at all: it is one of KEY_WITHOUT below",
    "device_index": "the device's index is a part of the device field; two "
    "tensors that differ in it alone need two devices of one type",
}

# Tensors that have no key: no region is entered with one, whatever it
# shares with the tensor of the trace. Operators refuse some of them: the
# call then raises what eager raises.
KEY_WITHOUT = {
    "a_subclass": lambda x: x.as_subclass(KeySubclass),
    "a_sparse_layout": lambda x: x.to_sparse(),
    "a_quantized_tensor": lambda x: torch.quantize_per_tensor(x, 0.1, 0, torch.qint8),
    "a_zero_tensor": lambda x: torch._efficientzerotensor(x.shape, device=x.device),
    "a_wrapper_of_functorch": key_escaped_from_vmap,
}


KEY_ZERO_FROM_STATE = ("a_zero_tensor", "read from a module attribute")


class KeyHolder(torch.nn.Module):
    pass


KEY_HELD = KeyHolder()


def key_argument(x, w):
    y = x.sin() * 2 + w
    return (y * 3 + 1).cos().tanh().exp()


def key_in_a_list(xs):
    y = xs[1].sin() * 2 + xs[0]
    return (y * 3 + 1).cos().tanh().exp()


def key_in_a_tuple(xs):
    y = xs[1].sin() * 2 + xs[0]
    return (y * 3 + 1).cos().tanh().exp()


def key_from_a_module(w):
    y = KEY_HELD.x.sin() * 2 + w
    return (y * 3 + 1).cos().tanh().exp()


def key_into_a_span(x, w):
    y = w
    for _ in range(3):
        y = (y * x + 1).cos().tanh()
    return y.exp()


# The ways a tensor enters a region: the function, how it is handed the
# tensor ``x`` under test and the other one, and the kind of its region.
KEY_WAYS = {
    "an argument": (key_argument, lambda x, w: (x, w), "block"),
    "a leaf of a list argument": (key_in_a_list, lambda x, w: ([w, x],), "block"),
    "a leaf of a tuple argument": (key_in_a_tuple, lambda x, w: ((w, x),), "block"),
    "read from a module attribute": (key_from_a_module, None, "block"),
    "the input of a span": (key_into_a_span, lambda x, w: (x, w), "span"),
}


def key_become(held, other):
    """Gives the object ``held`` every field of ``other`` in place (a tensor
    read from state is found again as the same object, so the field has to
    change under it); False where PyTorch has no way to: a tensor of another
    device type, layout or kind, and the caller binds ``other`` instead."""
    try:
        held.data = other
    except RuntimeError:
        return False
    held.__class__ = type(other)
    if held.requires_grad != other.requires_grad:
        held.requires_grad_(other.requires_grad)
    return True


@unittest.skipUnless(CHECKS_HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmTensorKey(TestCase):
    """Every tensor that enters a region is compared on one key: its exact
    class (Tensor or Parameter), dtype, device, sizes, strides,
    requires_grad, the conjugate and the negative bit and whether it is an
    inference tensor; a tensor that is not plain, dense and strided has no
    key and enters none. So a call that differs from the traced one in one
    field is a miss, whichever way the tensor comes in, and computes what
    eager computes."""

    hw_classification = HardwareClassification.ACCELERATOR

    def same(self, got, want, msg):
        if isinstance(want, Exception):
            self.assertIs(type(got), type(want), msg=f"{msg}: {got!r}")
            self.assertEqual(str(got), str(want), msg=msg)
            return
        self.assertIsInstance(got, torch.Tensor, msg=msg)
        for field, read in KEY_FIELDS.items():
            self.assertEqual(read(got), read(want), msg=f"{msg}: the {field}")
        self.assertEqual(
            got.grad_fn is None, want.grad_fn is None, msg=f"{msg}: the grad_fn"
        )
        if want.device.type != "meta":
            self.assertEqual(got, want, atol=0, rtol=0, msg=msg)

    def sweep(self, device, way, base, other):
        """Ten calls: the traced tensor on calls 0 to 5 and 8 and 9, the
        other one on calls 6 and 7."""
        fn, pack, kind = KEY_WAYS[way]
        compiled = vm(fn, passes=CHECKS_EARLY)
        held = None
        runs = []
        for call in range(10):
            torch.manual_seed(call)
            x = torch.randn(4, 8, device=device)
            w = base(torch.randn(4, 8, device=device))
            x = (other if call in (6, 7) else base)(x)
            args = (w,) if pack is None else pack(x, w)
            if pack is None and held is None:
                held = KEY_HELD.x = base(torch.ones(4, 8, device=device))
            elif pack is None and call == 6 and not key_become(held, x):
                # No way in place: the module holds another object, which
                # the region turns down for being another.
                KEY_HELD.x = x
            elif pack is None and call == 8:
                key_become(held, x)
                KEY_HELD.x = held
            before = checks_region_runs(compiled, fn)
            got = checks_outcome(compiled, *args)
            runs.append(checks_region_runs(compiled, fn) - before)
            self.same(got, checks_outcome(fn, *args), f"{way}, call {call}")
        rows = [r for r in vm.regions(compiled) if r["function"] == fn.__name__]
        self.assertEqual(
            {r["kind"] for r in rows if r["state"] == "region"}, {kind}, msg=str(rows)
        )
        # The region ran on the calls behind its trace, and does again once
        # the tensor is as it was.
        self.assertEqual(runs[5], 1, msg=f"{way}: {runs} {rows}")
        self.assertEqual(runs[8:], [1, 1], msg=f"{way}: {runs} {rows}")
        # The first call with the other tensor is a miss.
        self.assertEqual(runs[6], 0, msg=f"{way}: {runs} {rows}")

    @parametrize("field", list(KEY_SWEEP))
    def test_the_sweep_changes_one_field_at_a_time(self, device, field):
        base, other = KEY_SWEEP[field]
        x = torch.randn(4, 8, device=device)
        differ = [
            name
            for name, read in KEY_FIELDS.items()
            if read(base(x.clone())) != read(other(x.clone()))
        ]
        self.assertEqual(differ, [field])

    def test_every_field_is_swept_or_listed(self, device):
        self.assertEqual(
            sorted([*KEY_SWEEP, *KEY_NOT_SWEPT]), sorted([*KEY_FIELDS, "device_index"])
        )

    @parametrize("field", list(KEY_SWEEP))
    @checks_isolated
    def test_a_call_that_differs_in_one_field_is_a_miss(self, device, field):
        for way in KEY_WAYS:
            with self.subTest(way=way):
                self.sweep(device, way, *KEY_SWEEP[field])

    @parametrize("kind", list(KEY_WITHOUT))
    @checks_isolated
    def test_a_tensor_without_a_key_enters_no_region(self, device, kind):
        for way in KEY_WAYS:
            if (kind, way) != KEY_ZERO_FROM_STATE:
                with self.subTest(way=way):
                    self.sweep(device, way, key_plain, KEY_WITHOUT[kind])

    # The kernels of a zero tensor answer above the tracer, so the trace of
    # the call with the zero tensor has no operator that reads the module's
    # tensor and keeps no constant for it: what stands for the tensor is the
    # key it had when the call handed it to its first operator.
    @checks_isolated
    def test_a_zero_tensor_read_from_a_module_attribute(self, device):
        self.sweep(
            device, KEY_ZERO_FROM_STATE[1], key_plain, KEY_WITHOUT["a_zero_tensor"]
        )

    @unittest.skipUnless(torch.cuda.device_count() > 1, "needs two devices of one type")
    @checks_isolated
    def test_a_call_that_differs_in_the_device_index_is_a_miss(self, device):
        if torch.device(device).type != "cuda":
            self.skipTest("the device under test has one index")
        for way in KEY_WAYS:
            with self.subTest(way=way):
                self.sweep(device, way, key_plain, lambda x: x.to("cuda:1"))


instantiate_device_type_tests(TestVmTensorKey, globals())


# What a region compares of a value that is no tensor.


class ExactStr(str):
    """A str class that prints itself as one."""

    __slots__ = ()

    def __repr__(self):
        return f"ExactStr({str.__repr__(self)})"


class ExactEnum(enum.IntEnum):
    ONE = 1


# Pairs that `==` calls equal and that are not one value to the code that
# reads them: the value of the traced call, then the other.
EXACT_PAIRS = {
    "zero_and_negative_zero": (0.0, -0.0),
    "negative_zero_and_zero": (-0.0, 0.0),
    "one_and_true": (1, True),
    "true_and_one": (True, 1),
    "int_and_float": (1, 1.0),
    "float_and_int": (1.0, 1),
    "float_and_numpy_float": (1.0, numpy.float64(1.0)),
    "str_and_str_subclass": ("a", ExactStr("a")),
    "tuple_of_ints_and_with_a_float": ((1, 2), (1, 2.0)),
    "tuple_with_a_float_and_of_ints": ((1, 2.0), (1, 2)),
    "int_and_int_enum": (1, ExactEnum.ONE),
}

# Values that are the same though they are two objects: a check that told
# them apart would only cost regions.
EXACT_SAME = (
    (1.5, float("1.5")),
    (float("nan"), struct.unpack("d", struct.pack("d", float("nan")))[0]),
    (10**30, int("1" + "0" * 30)),
    ("ab", "".join(["a", "b"])),
    (b"ab", b"".join([b"a", b"b"])),
    ((1, "a", (2.5, None)), (1, "".join(["a"]), (float("2.5"), None))),
    ([1, [True]], [1, [True]]),
    (torch.Size([2, 3]), torch.Size([2, 3])),
    (torch.device("cpu"), torch.device("cpu")),
)


class ExactHolder(torch.nn.Module):
    pass


EXACT_HELD = ExactHolder()


def exact_argument(x, v):
    y = x.sin() * 2 + 1
    return (y * 3 + 1).cos().tanh().exp(), repr(v)


def exact_argument_of_a_span(x, v):
    y = x
    for _ in range(3):
        y = (y * x + 1).cos().tanh()
        text = repr(v)
    return y.exp(), text


def exact_key_of_an_argument(x, d):
    y = x.sin() * 2 + 1
    for k in d:
        y = y + d[k]
    return (y * 3).cos().tanh().exp(), [repr(k) for k in d], [type(k) for k in d]


def exact_from_state(x):
    y = x.sin() * 2 + 1
    return (y * 3 + 1).cos().tanh().exp(), repr(EXACT_HELD.v), type(EXACT_HELD.v)


def exact_under_a_guard(x, v):
    y = x.sin().abs() * 2 + 1
    return (y * v).reciprocal(), (y + v).cos().tanh().exp()


def exact_bakes_floats(gm, input_types):
    """A block pass that bakes the float inputs of a graph into it and guards
    their values with vm.Value, as a compiler does."""
    return vm.specialize_floats(gm, vm._examples.raw(gm))


def exact_guarded_hits(compiled, fn):
    """How many times the versions of ``fn``'s regions that a pass put a
    guard in front of have run."""
    return sum(v.hits for v in vm.versions(compiled) if v.guards)


def exact_handed(x, v):
    return (x, v)


def exact_keyed(x, v):
    return (x, {v: torch.ones_like(x), "z": torch.ones_like(x)})


def exact_held(x, v):
    EXACT_HELD.v = v
    return (x,)


# The ways a value comes to a check: the function, how the call hands it
# the value, the pass list, and what counts the runs of what the check
# stands in front of (a region; for a pass's guard, the pass's version).
EXACT_WAYS = {
    "an argument": (exact_argument, exact_handed, (), checks_region_runs),
    "an argument of a span": (
        exact_argument_of_a_span,
        exact_handed,
        (),
        checks_region_runs,
    ),
    "a dict key of an argument": (
        exact_key_of_an_argument,
        exact_keyed,
        (),
        checks_region_runs,
    ),
    "read from state": (exact_from_state, exact_held, (), checks_region_runs),
    "the guard value of a pass": (
        exact_under_a_guard,
        exact_handed,
        (exact_bakes_floats,),
        exact_guarded_hits,
    ),
}

# By pair, the ways in which the traced value gets nothing to run, so that
# there is nothing the other value could wrongly run: a dict with the key
# 0.0 or -0.0 is refused at the trace, and the pass bakes floats only.
EXACT_NO_REGION = {
    **dict.fromkeys(
        ("zero_and_negative_zero", "negative_zero_and_zero"),
        ("a dict key of an argument",),
    ),
    **dict.fromkeys(
        (
            "one_and_true",
            "true_and_one",
            "int_and_float",
            "str_and_str_subclass",
            "tuple_of_ints_and_with_a_float",
            "tuple_with_a_float_and_of_ints",
            "int_and_int_enum",
        ),
        ("the guard value of a pass",),
    ),
}


@unittest.skipUnless(CHECKS_HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmExactChecks(TestCase):
    """A check that does not compare by identity compares the exact class
    and the value bit for bit, never with `==`: 0.0 and -0.0, True and 1, 1
    and 1.0, 1.0 and a numpy.float64, "a" and an instance of a str subclass,
    (1, 2) and (1, 2.0), an int and an IntEnum member are equal, and a
    program tells each pair apart. So a region traced on one of a pair does
    not run on the other, whichever way the value comes to it."""

    hw_classification = HardwareClassification.ACCELERATOR

    def same(self, got, want, msg):
        if isinstance(want, Exception):
            self.assertIs(type(got), type(want), msg=f"{msg}: {got!r}")
            self.assertEqual(str(got), str(want), msg=msg)
        elif isinstance(want, torch.Tensor):
            self.assertIsInstance(got, torch.Tensor, msg=msg)
            self.assertEqual(got.dtype, want.dtype, msg=msg)
            self.assertEqual(got, want, atol=0, rtol=0, msg=msg)
            # A zero and a negative zero are equal.
            self.assertEqual(torch.signbit(got), torch.signbit(want), msg=msg)
        elif isinstance(want, (tuple, list)):
            self.assertIs(type(got), type(want), msg=msg)
            self.assertEqual(len(got), len(want), msg=msg)
            for g, w in zip(got, want):
                self.same(g, w, msg)
        else:
            self.assertIs(type(got), type(want), msg=msg)
            self.assertEqual(repr(got), repr(want), msg=msg)

    def sweep(self, device, way, first, other):
        """Ten calls: the traced value on calls 0 to 5 and 8 and 9, the
        other one on calls 6 and 7. True if a region ran on the traced
        value: it then never runs on the first call with the other."""
        fn, hand, more, count = EXACT_WAYS[way]
        compiled = vm(fn, passes=(*CHECKS_EARLY, *more))
        runs = []
        for call in range(10):
            torch.manual_seed(call)
            x = torch.randn(4, 8, device=device)
            args = hand(x, other if call in (6, 7) else first)
            before = count(compiled, fn)
            got = checks_outcome(compiled, *args)
            runs.append(count(compiled, fn) - before)
            self.same(got, checks_outcome(fn, *args), f"{way}, call {call}")
        if runs[5]:
            self.assertEqual(runs[6], 0, msg=f"{way}: {runs}")
            self.assertEqual(runs[8:], [runs[5]] * 2, msg=f"{way}: {runs}")
        return bool(runs[5])

    @parametrize("pair", list(EXACT_PAIRS))
    @checks_isolated
    def test_a_region_traced_on_one_of_a_pair_misses_the_other(self, device, pair):
        ran = {}
        for way in EXACT_WAYS:
            with self.subTest(way=way):
                ran[way] = self.sweep(device, way, *EXACT_PAIRS[pair])
        # The sweep is of regions that ran: by way, the pairs whose first
        # value gets none (EXACT_NO_REGION says why).
        none = sorted(way for way, did in ran.items() if not did)
        self.assertEqual(none, sorted(EXACT_NO_REGION.get(pair, ())))

    @parametrize("pair", list(EXACT_PAIRS))
    def test_the_pair_is_equal_and_not_the_same(self, device, pair):
        first, other = EXACT_PAIRS[pair]
        self.assertTrue(first == other)
        self.assertFalse(torch._rust._same(first, other))
        self.assertFalse(torch._rust._same(other, first))
        self.assertTrue(torch._rust._same(first, first))
        # In a container, and nested.
        self.assertFalse(torch._rust._same((first,), (other,)))
        self.assertFalse(torch._rust._same([0, [first]], [0, [other]]))
        # As a pass's guard value.
        self.assertFalse(vm.Value(first).holds(other, None))
        self.assertTrue(vm.Value(first).holds(first, None))
        # As a leaf and as a dict key of a container argument, where the
        # executor compares the container itself.
        x = torch.ones(2, device=device)
        for build in (lambda v: [x, v], lambda v: {v: x}):
            try:
                checks = torch._rust.TreeChecks(build(first))
            except TypeError:
                # Not a value it compares: the Python walk decides.
                continue
            self.assertIsNone(checks.leaves(build(other)))
            self.assertIsNotNone(checks.leaves(build(first)))

    def test_two_objects_of_one_value_are_the_same(self, device):
        for a, b in EXACT_SAME:
            self.assertIsNot(a, b)
            self.assertTrue(torch._rust._same(a, b), msg=repr(a))
        # Everything else is the same as itself only.
        others = (complex(1, 2), complex("1+2j")), (frozenset([1]), frozenset([1]))
        for a, b in (*others, ({1: 2}, {1: 2})):
            self.assertFalse(torch._rust._same(a, b), msg=repr(a))
            self.assertTrue(torch._rust._same(a, a), msg=repr(a))


instantiate_device_type_tests(TestVmExactChecks, globals())


# The state around a call: the word a region is for, and what must be neutral.


def ambient_body(x):
    y = x.sin() * 2 + torch.ones(8, device=x.device)
    return (y @ y.t()).cos().tanh().exp()


def ambient_body_from_the_start(x):
    y = x.sin() * 2 + torch.ones(8, device=x.device)
    return (y @ y.t()).cos().tanh().exp()


def ambient_plainly(fn, x):
    return fn(x)


@contextlib.contextmanager
def ambient_set(get, put, value):
    old = get()
    put(value)
    try:
        yield
    finally:
        put(old)


def ambient_fill_uninitialized_memory(value):
    def get():
        return torch.utils.deterministic.fill_uninitialized_memory

    def put(v):
        torch.utils.deterministic.fill_uninitialized_memory = v

    return ambient_set(get, put, value)


@contextlib.contextmanager
def ambient_float32_matmul_precision(value):
    # Setting it writes the matmul entries of two backends too: all go back.
    snapshot = _snapshot_fp32_precision()
    torch.set_float32_matmul_precision(value)
    try:
        yield
    finally:
        _restore_fp32_precision(snapshot)


# The items of the word: the state of the traced call, and another. Each is
# a context manager the call runs under (None: as the process starts).
AMBIENT_WORD = {
    "autocast": (None, lambda: torch.autocast("cpu", dtype=torch.bfloat16)),
    "autocast_dtype": (
        lambda: torch.autocast("cpu", dtype=torch.bfloat16),
        lambda: torch.autocast("cpu", dtype=torch.float16),
    ),
    "default_dtype": (
        None,
        lambda: ambient_set(
            torch.get_default_dtype, torch.set_default_dtype, torch.float64
        ),
    ),
    "deterministic_algorithms": (None, lambda: DeterministicGuard(True)),
    "deterministic_algorithms_warn_only": (
        lambda: DeterministicGuard(True),
        lambda: DeterministicGuard(True, warn_only=True),
    ),
    "deterministic_fill_uninitialized_memory": (
        None,
        lambda: ambient_fill_uninitialized_memory(False),
    ),
    "cudnn_enabled": (None, lambda: torch.backends.cudnn.flags(enabled=False)),
    "cudnn_benchmark": (None, lambda: torch.backends.cudnn.flags(benchmark=True)),
    "cudnn_deterministic": (
        None,
        lambda: torch.backends.cudnn.flags(deterministic=True),
    ),
    "mkldnn_enabled": (None, lambda: torch.backends.mkldnn.flags(enabled=False)),
    "float32_matmul_precision": (
        None,
        lambda: ambient_float32_matmul_precision("high"),
    ),
    "grad_mode": (None, torch.no_grad),
    "inference_mode": (torch.no_grad, torch.inference_mode),
}

# What the word has and the sweep cannot set, and why.
AMBIENT_NOT_SWEPT = {
    "autocast for cuda, xpu and mps": "as for the CPU, on a device of that type",
    "the current CUDA device": "needs two CUDA devices: see "
    "test_the_current_cuda_device_is_in_the_word",
}


def ambient_under_a_dispatch_mode(fn, x):
    seen = []

    class SeesOperators(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            seen.append(func.name())
            return func(*args, **(kwargs or {}))

    with SeesOperators():
        return fn(x), seen


def ambient_under_a_function_mode(fn, x):
    seen = []

    class SeesFunctions(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            seen.append(getattr(func, "__name__", repr(func)))
            return func(*args, **(kwargs or {}))

    with SeesFunctions():
        return fn(x), seen


def ambient_under_vmap(fn, x):
    return torch.func.vmap(fn)(x.expand(2, *x.shape))


def ambient_under_a_forward_ad_level(fn, x):
    with torch.autograd.forward_ad.dual_level():
        return fn(x)


def ambient_under_saved_tensor_hooks(fn, x):
    packed = []

    def pack(t):
        packed.append(tuple(t.shape))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        return fn(x), packed


def ambient_under_fx_tracing(fn, x):
    traced = torch.fx.symbolic_trace(fn)
    return traced(x), traced.code


def ambient_under_anomaly_mode(fn, x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.autograd.detect_anomaly(check_nan=False):
            return fn(x)


# What must be neutral: how a call runs under each item, giving what the
# program computes and what the item saw of it.
AMBIENT_NEUTRAL = {
    "a_dispatch_mode": ambient_under_a_dispatch_mode,
    "a_torch_function_mode": ambient_under_a_function_mode,
    "a_functorch_transform": ambient_under_vmap,
    "a_forward_ad_level": ambient_under_a_forward_ad_level,
    "saved_tensor_hooks": ambient_under_saved_tensor_hooks,
    "fx_tracing": ambient_under_fx_tracing,
    "anomaly_mode": ambient_under_anomaly_mode,
}

AMBIENT_NEUTRAL_NOT_SWEPT = {
    "JIT tracing": "this build has no torch.jit.trace to start it",
    "autocast for a device type the word does not name": "torch.autocast "
    "takes it only where a backend of that type is registered",
}


@unittest.skipUnless(CHECKS_HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmAmbient(TestCase):
    """A region is entered only under the ambient word of its trace
    (autocast, the default dtype, the determinism and backend flags, grad
    mode, inference mode, the current CUDA device), and only while
    everything else is neutral: under a dispatch mode, a torch function
    mode, a functorch transform, a forward-AD level, saved-tensor hooks, JIT
    or fx tracing or anomaly mode no trace starts and no region is entered,
    and the code runs as written."""

    hw_classification = HardwareClassification.ACCELERATOR

    def same(self, got, want, msg):
        if isinstance(want, torch.Tensor):
            self.assertIsInstance(got, torch.Tensor, msg=msg)
            self.assertEqual(got.dtype, want.dtype, msg=msg)
            self.assertEqual(got.requires_grad, want.requires_grad, msg=msg)
            self.assertEqual(got.is_inference(), want.is_inference(), msg=msg)
            self.assertEqual(got, want, atol=0, rtol=0, msg=msg)
        elif isinstance(want, tuple):
            self.assertEqual(len(got), len(want), msg=msg)
            for g, w in zip(got, want):
                self.same(g, w, msg)
        else:
            self.assertEqual(got, want, msg=msg)

    def calls(self, device, fn, how):
        """Ten calls of ``fn`` on the VM and in eager, each made by
        ``how(call)(function, x)``: how many regions each of the VM's ran."""
        compiled = vm(fn, passes=CHECKS_EARLY)
        runs = []
        for call in range(10):
            torch.manual_seed(call)
            x = torch.randn(4, 8, device=device, requires_grad=True)
            before = checks_region_runs(compiled, fn)
            got = how(call)(compiled, x)
            runs.append(checks_region_runs(compiled, fn) - before)
            self.same(got, how(call)(fn, x), f"call {call}")
        return runs, compiled

    @parametrize("item", list(AMBIENT_WORD))
    @checks_isolated
    def test_a_call_under_another_word_is_a_miss(self, device, item):
        first, other = AMBIENT_WORD[item]

        def how(call):
            def run(fn, x):
                enter = other if call in (6, 7) else first
                with contextlib.ExitStack() if enter is None else enter():
                    return fn(x)

            return run

        runs, _ = self.calls(device, ambient_body, how)
        self.assertEqual(runs[5:7], [1, 0], msg=str(runs))
        self.assertEqual(runs[8:], [1, 1], msg=str(runs))

    @parametrize("item", list(AMBIENT_NEUTRAL))
    @checks_isolated
    def test_no_region_is_entered_under(self, device, item):
        def how(call):
            return AMBIENT_NEUTRAL[item] if call in (6, 7) else ambient_plainly

        runs, _ = self.calls(device, ambient_body, how)
        self.assertEqual(runs[5:8], [1, 0, 0], msg=str(runs))
        self.assertEqual(runs[8:], [1, 1], msg=str(runs))

    @parametrize("item", list(AMBIENT_NEUTRAL))
    @checks_isolated
    def test_no_trace_starts_under(self, device, item):
        fn = ambient_body_from_the_start
        runs, compiled = self.calls(device, fn, lambda call: AMBIENT_NEUTRAL[item])
        self.assertEqual(runs, [0] * 10)
        rows = [r for r in vm.regions(compiled) if r["function"] == fn.__name__]
        self.assertNotIn("region", {r["state"] for r in rows}, msg=str(rows))

    @checks_isolated
    def test_a_default_device_keeps_every_region_away(self, device):
        # The known cost: torch.set_default_device puts a torch function
        # mode on for good, so a program that sets one is never traced.
        torch.set_default_device(device)
        try:
            fn = ambient_body_from_the_start
            runs, _ = self.calls(device, fn, lambda call: ambient_plainly)
        finally:
            torch.set_default_device(None)
        self.assertEqual(runs, [0] * 10)

    @unittest.skipUnless(
        torch.cuda.device_count() > 1, "needs two CUDA devices; this machine has one"
    )
    @checks_isolated
    def test_the_current_cuda_device_is_in_the_word(self, device):
        if torch.device(device).type != "cuda":
            self.skipTest("the current CUDA device is a CUDA matter")

        def ambient_made_on_the_current_device(x):
            y = x.sin() * 2 + torch.ones(8, device="cuda").to(x.device)
            return (y @ y.t()).cos().tanh().exp(), torch.ones(2, device="cuda").device

        def how(call):
            def run(fn, x):
                with torch.cuda.device(1 if call in (6, 7) else 0):
                    return fn(x)

            return run

        runs, _ = self.calls(device, ambient_made_on_the_current_device, how)
        self.assertEqual(runs[5:7], [1, 0], msg=str(runs))
        self.assertEqual(runs[8:], [1, 1], msg=str(runs))


instantiate_device_type_tests(TestVmAmbient, globals())


if CHECKS_HAS_VM:
    from torch._rust import _calls_make, _calls_run, _calls_text


def _cl_classes(t, one):
    """The classes of the operand template ``t`` with every leaf of class
    ``one``: None for a null, a tuple for what is built."""
    kind = t[0] if isinstance(t, tuple) and t else None
    if kind in ("seq", "map") and len(t) == 3:
        return tuple(_cl_classes(part, one) for part in t[2])
    if kind == "slice":
        return tuple(_cl_classes(part, one) for part in t[1:])
    return None if kind == "null" else one


def _cl_entry(
    callee, args, kwnames=None, out=None, at=0, flags=(), asserts=(), classes=None
):
    """classes: the tuple parallel to ``args``, or one class for every
    operand (torch.Tensor if not given)."""
    if not isinstance(classes, tuple):
        one = torch.Tensor if classes is None else classes
        classes = tuple(_cl_classes(t, one) for t in args)
    return (
        callee,
        list(args),
        kwnames,
        out,
        at,
        tuple(flags),
        tuple(asserts),
        classes,
    )


def _cl_drop(*slots, at=0):
    return ("drop", slots, at)


def _cl_frames(f, *args):
    """How many Python frames ``f(*args)`` enters: 0 for C code only."""
    entered = []

    def profile(frame, event, arg):
        if event == "call":
            entered.append(frame.f_code.co_name)

    sys.setprofile(profile)
    try:
        f(*args)
    finally:
        sys.setprofile(None)
    return len(entered)


class _ClFinal:
    """An object whose death is seen: it writes its name to a log."""

    def __init__(self, name, log):
        self.name, self.log = name, log

    def __del__(self):
        self.log.append(self.name)


def _cl_vector(f):
    return ("vector", f, f.__qualname__)


def _cl_run(spec, inputs=(), pins=()):
    return _calls_run(_calls_make(spec), tuple(inputs), tuple(pins))


_CL_BINARY = {
    "+": operator.add, "&": operator.and_, "//": operator.floordiv, "<<": operator.lshift,
    "@": operator.matmul, "*": operator.mul, "%": operator.mod, "|": operator.or_,
    "**": operator.pow, ">>": operator.rshift, "-": operator.sub, "/": operator.truediv,
    "^": operator.xor, "+=": operator.iadd, "&=": operator.iand, "//=": operator.ifloordiv,
    "<<=": operator.ilshift, "@=": operator.imatmul, "*=": operator.imul, "%=": operator.imod,
    "|=": operator.ior, "**=": operator.ipow, ">>=": operator.irshift, "-=": operator.isub,
    "/=": operator.itruediv, "^=": operator.ixor,
}  # fmt: skip
_CL_COMPARE = {
    "<": operator.lt, "<=": operator.le, "==": operator.eq,
    "!=": operator.ne, ">": operator.gt, ">=": operator.ge,
}  # fmt: skip
_CL_UNARY = {"-": operator.neg, "~": operator.invert, "+": operator.pos}


@unittest.skipUnless(CHECKS_HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmCallLists(TestCase):
    """A call list is a recorded run of C calls: the callee by identity and
    templates of its operands, run again in order with no frame. The rule:
    running a list does what the plain Python calls do (values, state, the
    exception of a call that raises, where and in which order values die),
    and a list that is not understood in full is refused when it is made: so
    is a slot callee that would run a method written in Python for the
    classes its operands are pinned to.
    """

    hw_classification = HardwareClassification.GENERIC

    def one(self, callee, inputs, kwnames=None):
        args = [("in", k) for k in range(len(inputs))]
        spec = (
            len(inputs),
            0,
            [],
            [
                _cl_entry(
                    callee,
                    args,
                    kwnames,
                    out=0,
                    classes=tuple(type(v) for v in inputs),
                )
            ],
            [("res", 0)],
        )
        return _cl_run(spec, inputs)[0]

    def test_every_binary_operator(self):
        ints = (torch.tensor([[7, 9], [12, 5]]), torch.tensor([[2, 1], [3, 2]]))
        floats = (
            torch.tensor([[7.5, 9.0], [12.0, 5.25]]),
            torch.tensor([[2.0, 1.5], [3.0, 2.0]]),
        )
        self.assertEqual(len(_CL_BINARY), 26)
        for text, f in _CL_BINARY.items():
            with self.subTest(op=text):
                if not text.startswith("@"):
                    self.assertEqual(self.one(("binary", text), (7, 2)), f(7, 2))
                x, y = (
                    ints if text.rstrip("=") in ("&", "<<", "|", ">>", "^") else floats
                )
                # Tensor's `**` and `//` are functions written in Python: a
                # frame in the plain program, so no entry. `//=` has a C
                # method, but the `__floordiv__` it could hand over to is not.
                if _cl_frames(f, x.clone(), y) or text == "//=":
                    with self.assertRaisesRegex(
                        ValueError, "slot would run Python.*no C function"
                    ):
                        self.one(("binary", text), (x.clone(), y))
                    continue
                mine, eager = x.clone(), x.clone()
                got, want = self.one(("binary", text), (mine, y)), f(eager, y)
                self.assertTrue(torch.equal(got, want))
                # An in-place operator hands back its operand, written.
                self.assertEqual(got is mine, want is eager)
                self.assertTrue(torch.equal(mine, eager))

    def test_every_comparison(self):
        x, y = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 2.0, 1.0])
        self.assertEqual(len(_CL_COMPARE), 6)
        for text, f in _CL_COMPARE.items():
            with self.subTest(op=text):
                self.assertIs(self.one(("compare", text), (2, 3)), f(2, 3))
                self.assertTrue(
                    torch.equal(self.one(("compare", text), (x, y)), f(x, y))
                )

    def test_every_unary_operator(self):
        x = torch.tensor([1, -2, 3])
        for text, f in _CL_UNARY.items():
            with self.subTest(op=text):
                self.assertEqual(self.one(("unary", text), (5,)), f(5))
                self.assertTrue(torch.equal(self.one(("unary", text), (x,)), f(x)))

    def test_every_kind_of_c_callable(self):
        x = torch.tensor([1.0, 2.0])
        pattern, target = re.compile("b+"), []
        # A builtin method that knows its class, and one of an object.
        got = self.one(_cl_vector(pattern.search), ("abbc",))
        self.assertEqual((type(got), got.span()), (re.Match, (1, 3)))
        self.assertIsNone(self.one(_cl_vector(target.append), (3,)))
        self.assertEqual(target, [3])
        # A builtin function, a method descriptor, a slot wrapper, a C type.
        for f, args in [
            (torch.add, (x, x)),
            (torch.Tensor.neg, (x,)),
            (float.__mul__, (1.5, 4.0)),
            (float, (3,)),
        ]:
            with self.subTest(callee=f.__qualname__):
                got, want = self.one(_cl_vector(f), args), f(*args)
                self.assertIs(type(got), type(want))
                self.assertEqual(got, want)

    def test_keyword_names(self):
        x = torch.arange(6.0).reshape(2, 3)
        got = self.one(_cl_vector(torch.sum), (x, 1, True), kwnames=("dim", "keepdim"))
        self.assertTrue(torch.equal(got, torch.sum(x, dim=1, keepdim=True)))
        spec = (
            2,
            0,
            [1],
            [
                _cl_entry(
                    _cl_vector(torch.cat),
                    [("seq", "list", [("in", 0), ("in", 1)]), ("const", 0)],
                    ("dim",),
                    out=0,
                    classes=((torch.Tensor, torch.Tensor), int),
                )
            ],
            [("res", 0)],
        )
        self.assertTrue(torch.equal(_cl_run(spec, (x, x))[0], torch.cat([x, x], dim=1)))

    def test_subscripts_and_attributes(self):
        x, T = torch.arange(24.0).reshape(2, 3, 4), torch.Tensor
        rows = torch.tensor([1, 0])
        box, names, items = types.SimpleNamespace(), {"a": 1, "b": 2}, [10, 20, 30]
        key = (
            "seq",
            "tuple",
            [
                ("slice", ("const", 0), ("null",), ("null",)),
                ("const", 1),
                ("const", 2),
                ("in", 1),
            ],
        )
        spec = (2, 3, [1, None, ..., "b", 0, 5.0], [
            _cl_entry(("getitem",), [("in", 0), key], out=0, asserts=[("class", T)], classes=(T, ((int, None, None), type(None), type(...), T))),
            _cl_entry(("getitem",), [("in", 0), ("slice", ("null",), ("null",), ("const", 0))], out=1, asserts=[("class", T)], classes=(T, (None, None, int))),
            _cl_entry(("getattr", "mT"), [("res", 0)], out=2, asserts=[("class", T)]),
            _cl_entry(("setitem",), [("const", 5), ("res", 1), ("const", 4)], classes=(float, T, int)),
            _cl_entry(("setattr", "kept"), [("res", 2), ("pin", 0)], classes=(T, types.SimpleNamespace)),
            _cl_entry(("setitem",), [("res", 0), ("pin", 1), ("const", 3)], classes=(T, dict, str)),
            _cl_entry(("delitem",), [("pin", 2), ("const", 4)], classes=(list, int)),
        ], [("res", 0), ("res", 1), ("res", 2)])  # fmt: skip
        mine = x.clone()
        got = _cl_run(spec, (mine, rows), (box, names, items))
        want0, want1 = x[1:, None, ..., rows], x.clone()[::1]
        want1[0] = 5.0
        self.assertTrue(torch.equal(got[0], want0))
        self.assertTrue(torch.equal(got[1], want1))
        # The store went through the view into the input.
        self.assertTrue(torch.equal(mine, want1))
        self.assertTrue(torch.equal(got[2], want0.mT))
        self.assertIs(box.kept, got[2])
        self.assertIs(names["b"], got[0])
        self.assertEqual(items, [20, 30])

    def test_every_template(self):
        x, y, w = (
            torch.tensor([1.0, 4.0]),
            torch.tensor([2.0, 3.0]),
            torch.tensor([10.0, 20.0]),
        )
        nested = (
            "seq",
            "tuple",
            [("seq", "tuple", [("in", 0), ("pin", 0)]), ("in", 1)],
        )
        spec = (2, 1, [0, "k", 2.5], [
            # tuple() of an exact tuple is that tuple: ((x, w), y)
            _cl_entry(_cl_vector(tuple), [nested], out=0, asserts=[("class", tuple)]),
            _cl_entry(_cl_vector(torch.max), [("in", 0), ("const", 0)], out=1, asserts=[("class", torch.return_types.max)], classes=(torch.Tensor, int)),
            _cl_entry(_cl_vector(torch.add), [("part", 0, (0, 1)), ("part", 1, (0,))], out=2),
            _cl_drop(1),
        ], [
            ("res", 2),
            ("part", 0, (1,)),
            ("seq", "list", [("in", 0), ("seq", "tuple", [("res", 2)]), ("const", 2)]),
            ("map", ("k", 7), [("pin", 0), ("seq", "list", [])]),
            ("slice", ("const", 0), ("null",), ("in", 1)),
            ("const", 1),
        ])  # fmt: skip
        h = _calls_make(spec)
        first, second = _calls_run(h, (x, y), (w,)), _calls_run(h, (x, y), (w,))
        for got in (first, second):
            self.assertTrue(torch.equal(got[0], w + torch.max(x, 0)[0]))
            self.assertIs(got[1], y)
            self.assertIs(type(got[2]), list)
            self.assertIs(type(got[2][1]), tuple)
            self.assertEqual(
                (got[2][0] is x, got[2][1][0] is got[0], got[2][2]), (True, True, 2.5)
            )
            self.assertIs(type(got[3]), dict)
            self.assertEqual(
                (list(got[3]), got[3]["k"] is w, got[3][7]), (["k", 7], True, [])
            )
            self.assertEqual(
                (type(got[4]), got[4].start, got[4].stop, got[4].step is y),
                (slice, 0, None, True),
            )
            self.assertEqual(got[5], "k")
        # What a template builds is built for each run.
        for k in (0, 2, 3, 4):
            self.assertIsNot(first[k], second[k])
        self.assertIsNot(first[3][7], second[3][7])

    def test_text(self):
        x, T = torch.zeros(2, 3), torch.Tensor
        spec = (2, 1, [2.0, (1, 0), "k", x, b"k"], [
            _cl_entry(("vector", torch.add, "torch.add"), [("in", 0), ("in", 1)], out=0, flags=["pure", "fresh"], asserts=[("key", x)]),
            _cl_entry(("vector", torch.Tensor.mul_, "Tensor.mul_"), [("move", 0), ("pin", 0)], out=1, at=1, flags=[("writes", (0,))], asserts=[("is", 0), ("class", T)]),
            _cl_entry(("binary", "*="), [("res", 1), ("const", 0)], out=2, at=2, flags=["opaque", "covered", ("writes", (0,)), ("rekeys", (0, 1))], asserts=[("class", T)], classes=(T, float)),
            _cl_entry(("vector", torch.permute, "torch.permute"), [("res", 2), ("const", 1)], ("dims",), out=3, at=3, flags=["draws", "effect"], asserts=[("class", T)], classes=(T, tuple)),
            _cl_entry(("vector", torch.max, "torch.max"), [("move", 3), ("const", 3)], out=4, at=4, asserts=[("count", 2), ("class", torch.return_types.max)]),
            _cl_drop(1, at=4),
            _cl_entry(("compare", "<="), [("part", 4, (0,)), ("in", 0)], at=5),
            _cl_entry(("unary", "-"), [("res", 2)], out=7, at=6, asserts=[("class", T)]),
            _cl_entry(("getitem",), [("res", 7), ("seq", "tuple", [("slice", ("const", 0), ("null",), ("null",)), ("seq", "list", [("in", 1)])])], out=8, at=7, asserts=[("class", T)], classes=(T, ((float, None, None), (T,)))),
            _cl_entry(("setitem",), [("in", 0), ("res", 8), ("map", ("k", 1, b"k"), [("in", 1), ("seq", "tuple", [("const", 2)]), ("const", 4)])], at=8, classes=(T, T, (T, (str,), bytes))),
            _cl_entry(("delitem",), [("pin", 0), ("const", 2)], at=9, classes=(T, str)),
            _cl_entry(("getattr", "mT"), [("res", 8)], out=11, at=10, asserts=[("class", T)]),
            _cl_entry(("setattr", "grad"), [("move", 11), ("pin", 0)], at=11),
            _cl_drop(8, 7, 4, 2, at=12),
        ], [])  # fmt: skip
        self.assertEqual(
            _calls_text(_calls_make(spec)).split("\n"),
            [
                "  0  r0 = torch.add(in0, in1) of (Tensor, Tensor)  @0 [pure fresh] assert key=tensor(f32 cpu [2, 3] stride [3, 1])",
                "  1  r1 = Tensor.mul_(move r0, pin0) of (Tensor, Tensor)  @1 [writes(0)] assert class=Tensor is(operand 0)",
                "  2  r2 = r1 *= 2.0 of (Tensor, float)  @2 [opaque covered writes(0) rekeys(0, 1)] assert class=Tensor",
                "  3  r3 = torch.permute(r2, dims=(1, 0)) of (Tensor, tuple)  @3 [draws effect] assert class=Tensor",
                "  4  r4 = torch.max(move r3, c3) of (Tensor, Tensor)  @4 assert class=max count=2",
                "  5  drop r1  @4",
                "  6  r4.0 <= in0 of (Tensor, Tensor)  @5",
                "  7  r7 = -r2 of (Tensor)  @6 assert class=Tensor",
                "  8  r8 = r7[(2.0::, [in1])] of (Tensor, float, -, -, Tensor)  @7 assert class=Tensor",
                "  9  r8[{'k': in1, 1: ('k',), b'k': b'k'}] = in0 of (Tensor, Tensor, Tensor, str, bytes)  @8",
                " 10  del pin0['k'] of (Tensor, str)  @9",
                " 11  r11 = r8.mT of (Tensor)  @10 assert class=Tensor",
                " 12  pin0.grad = move r11 of (Tensor, Tensor)  @11",
                " 13  drop r8 r7 r4 r2  @12",
            ],
        )

    def test_a_list_that_is_not_understood_is_refused(self):
        add, x = _cl_vector(torch.add), ("in", 0)

        def spec(*entries, inputs=1, pins=1, consts=(1,), outputs=()):
            return (inputs, pins, list(consts), list(entries), list(outputs))

        first = _cl_entry(
            add, [x, x], out=0, asserts=[("class", torch.Tensor)], classes=int
        )
        unknown = _cl_entry(add, [x, x], out=0, classes=int)
        self.assertEqual(
            _calls_run(_calls_make(spec(first, outputs=[("res", 0)])), (1,), (None,)),
            (2,),
        )

        def on(*args, **more):
            return _cl_entry(add, list(args), **more)

        class Named:
            @property
            def shown(self):
                return 1

            def __getitem__(self, key):
                return key

            def __lt__(self, other):
                return True

            def __neg__(self):
                return self

        class Quiet(list):
            __slots__ = ()

        Tensor = torch.Tensor

        class Callable:
            def __call__(self, a):
                return a

        class Text(str):
            __slots__ = ()

        class Pair(tuple):
            __slots__ = ()

        cases = [
            # the callee
            ('no callee kind "python"', spec(_cl_entry(("python", len), [x]))),
            (
                "the callee .* is no builtin",
                spec(_cl_entry(_cl_vector(torch.nn.functional.relu), [x])),
            ),
            (
                "the callee .* is no builtin",
                spec(_cl_entry(("vector", lambda a: a, "f"), [x])),
            ),
            (
                "the callee .* is no builtin",
                spec(_cl_entry(("vector", Callable(), "f"), [x])),
            ),
            (
                "the callee .* is no builtin",
                spec(_cl_entry(("vector", Callable, "f"), [x])),
            ),
            (
                "the callee .* is no builtin",
                spec(_cl_entry(("vector", torch.Tensor, "f"), [x])),
            ),
            (
                "the callee .* is no builtin",
                spec(_cl_entry(("vector", functools.partial, "f"), [x])),
            ),
            (
                "the callee .* is no builtin",
                spec(_cl_entry(("vector", functools.partial(len), "f"), [x])),
            ),
            (
                "the callee .* is no builtin",
                spec(_cl_entry(("vector", (1.0).__add__, "f"), [x])),
            ),
            (
                "the callee .* is no builtin",
                spec(_cl_entry(("vector", torch.nn.ReLU().forward, "f"), [x])),
            ),
            (
                "the callee .* is no builtin",
                spec(_cl_entry(("vector", torch.ops.aten.relu, "f"), [x])),
            ),
            (
                "a callee with no name",
                spec(_cl_entry(("vector", torch.add, ""), [x, x])),
            ),
            ("vector takes 2 fields", spec(_cl_entry(("vector", torch.add), [x, x]))),
            (
                "a state setter: unimplemented",
                spec(_cl_entry(("state", torch._C._set_grad_enabled), [("const", 0)])),
            ),
            ('no binary operator "and"', spec(_cl_entry(("binary", "and"), [x, x]))),
            ('no compare operator "is"', spec(_cl_entry(("compare", "is"), [x, x]))),
            ('no unary operator "not"', spec(_cl_entry(("unary", "not"), [x]))),
            ("getitem takes 0 fields", spec(_cl_entry(("getitem", 1), [x, x]))),
            ("an attribute name must be a str", spec(_cl_entry(("getattr", 1), [x]))),
            (
                "an attribute name must be a str",
                spec(_cl_entry(("getattr", Text("T")), [x])),
            ),
            # the count of operands
            (
                "1 operands where the callee takes 2",
                spec(_cl_entry(("binary", "+"), [x])),
            ),
            (
                "3 operands where the callee takes 2",
                spec(_cl_entry(("compare", "<"), [x, x, x])),
            ),
            (
                "2 operands where the callee takes 1",
                spec(_cl_entry(("unary", "-"), [x, x])),
            ),
            ("1 operands where the callee takes 2", spec(_cl_entry(("getitem",), [x]))),
            (
                "2 operands where the callee takes 3",
                spec(_cl_entry(("setitem",), [x, x])),
            ),
            ("1 operands where the callee takes 2", spec(_cl_entry(("delitem",), [x]))),
            (
                "2 operands where the callee takes 1",
                spec(_cl_entry(("getattr", "T"), [x, x])),
            ),
            (
                "1 operands where the callee takes 2",
                spec(_cl_entry(("setattr", "T"), [x])),
            ),
            # templates
            ('no template kind "local"', spec(_cl_entry(add, [x, ("local", 0)]))),
            ("in takes 1 fields", spec(_cl_entry(add, [x, ("in", 0, 1)]))),
            ("input 1 of 1", spec(_cl_entry(add, [x, ("in", 1)]))),
            ("pin 1 of 1", spec(_cl_entry(add, [x, ("pin", 1)]))),
            ("constant 1 of 1", spec(_cl_entry(add, [x, ("const", 1)]))),
            ("an input must be an int from 0", spec(_cl_entry(add, [x, ("in", -1)]))),
            ("an input must be an int from 0", spec(_cl_entry(add, [x, ("in", True)]))),
            (
                "an input must be an int from 0",
                spec(_cl_entry(add, [x, ("in", 2**40)])),
            ),
            (
                "r0 is the result of this or a later entry",
                spec(_cl_entry(add, [x, ("res", 0)])),
            ),
            (
                "r1 is the result of this or a later entry",
                spec(first, _cl_entry(add, [x, ("res", 1)], out=1)),
            ),
            (
                "r2 is the result of this or a later entry",
                spec(first, _cl_entry(add, [x, ("part", 2, (0,))], out=1), first),
            ),
            (
                "r0 is not held",
                spec(_cl_entry(add, [x, x]), _cl_entry(add, [x, ("res", 0)])),
            ),
            # a use behind the slot's release
            ("r0 is not held", spec(first, _cl_drop(0), on(x, ("res", 0)))),
            (
                "r0 is not held",
                spec(first, _cl_drop(0), on(x, ("seq", "list", [("part", 0, (1,))]))),
            ),
            ("r0 is not held", spec(first, on(x, ("move", 0)), on(x, ("res", 0)))),
            ("r0 is not held", spec(first, on(x, ("move", 0)), on(x, ("move", 0)))),
            (
                "output 0: r0 is not held",
                spec(first, _cl_drop(0), outputs=[("res", 0)]),
            ),
            (
                "output 0: r0 is not held",
                spec(first, on(x, ("move", 0)), outputs=[("res", 0)]),
            ),
            # a slot with two releases, or with none
            (
                "drops r0, which is not an earlier",
                spec(first, on(x, ("move", 0)), _cl_drop(0)),
            ),
            ("r0 is moved and named again", spec(first, on(("move", 0), ("move", 0)))),
            ("r0 is moved and named again", spec(first, on(("res", 0), ("move", 0)))),
            (
                "r0 is moved and named again",
                spec(first, on(x, ("seq", "list", [("move", 0), ("res", 0)]))),
            ),
            ("output 0: an output moves nothing", spec(first, outputs=[("move", 0)])),
            ("r0 is never released", spec(first)),
            (
                "r1 is never released",
                spec(first, _cl_entry(add, [x, x], out=1), outputs=[("res", 0)]),
            ),
            # classes
            ("1 classes for 2 operands", spec(_cl_entry(add, [x, x], classes=(int,)))),
            (
                "3 classes for 2 operands",
                spec(_cl_entry(add, [x, x], classes=(int, int, int))),
            ),
            ("classes must be a tuple or a list", spec(first[:7] + (None,))),
            (
                "the class of an operand must be a type",
                spec(_cl_entry(add, [x, x], classes=(int, 1))),
            ),
            (
                "the class of an operand must be a type",
                spec(_cl_entry(add, [x, x], classes=(int, None))),
            ),
            (
                "the class of an operand must be a type",
                spec(_cl_entry(add, [x, x], classes=(int, (int,)))),
            ),
            (
                "the class of a null must be None",
                spec(
                    _cl_entry(
                        ("getitem",),
                        [x, ("slice", ("null",), x, x)],
                        classes=(list, (int, int, int)),
                    )
                ),
            ),
            (
                "2 classes for 3 parts",
                spec(
                    _cl_entry(
                        ("getitem",),
                        [x, ("slice", ("null",), x, x)],
                        classes=(list, (None, int)),
                    )
                ),
            ),
            (
                "the classes of the parts of a seq, map or slice must be a tuple or a list",
                spec(_cl_entry(add, [x, ("seq", "list", [x])], classes=(int, int))),
            ),
            (
                "constant 0 is not of the class it is pinned to",
                spec(_cl_entry(add, [x, ("const", 0)], classes=(int, float))),
            ),
            (
                "r0 is an operand and its entry asserts no class of it",
                spec(unknown, on(x, ("res", 0)), outputs=[("res", 0)]),
            ),
            (
                "r0 is an operand and its entry asserts no class of it",
                spec(unknown, on(x, ("move", 0))),
            ),
            (
                "r0 is pinned to another class than its entry asserts",
                spec(first, _cl_entry(add, [x, ("move", 0)], classes=(int, float))),
            ),
            (
                "a part of r0, which its entry does not assert to be a tuple",
                spec(first, on(x, ("part", 0, (0,))), _cl_drop(0)),
            ),
            (
                "a part of r0, which its entry does not assert to be a tuple",
                spec(unknown, on(x, ("part", 0, (0,))), _cl_drop(0)),
            ),
            # a slot that would run Python, or is not covered
            (
                "slot would run Python.*no C function: __getitem__ of the class .*Named",
                spec(_cl_entry(("getitem",), [x, x], classes=(Named, int))),
            ),
            (
                "slot would run Python.*no C function: __setitem__ of the class .*ModuleList",
                spec(
                    _cl_entry(
                        ("setitem",), [x, x, x], classes=(int, torch.nn.ModuleList, int)
                    )
                ),
            ),
            (
                "slot would run Python.*no C function: __delitem__ of the class .*ModuleList",
                spec(
                    _cl_entry(("delitem",), [x, x], classes=(torch.nn.ModuleList, int))
                ),
            ),
            (
                "slot would run Python.*__class_getitem__",
                spec(_cl_entry(("getitem",), [x, x], classes=(type, int))),
            ),
            (
                "slot would run Python.*no C function: __lt__ of the class .*Named",
                spec(_cl_entry(("compare", "<"), [x, x], classes=(Named, int))),
            ),
            (
                "slot would run Python.*no C function: __lt__ of the class .*Named",
                spec(_cl_entry(("compare", ">"), [x, x], classes=(int, Named))),
            ),
            (
                "slot would run Python.*no C function: __neg__ of the class .*Named",
                spec(_cl_entry(("unary", "-"), [x], classes=(Named,))),
            ),
            (
                "slot would run Python.*no C function: __rsub__ of the class Tensor",
                spec(_cl_entry(("binary", "-"), [x, x], classes=(float, Tensor))),
            ),
            (
                "slot would run Python.*no C function: __ipow__ of the class Tensor",
                spec(_cl_entry(("binary", "**="), [x, x], classes=(Tensor, int))),
            ),
            (
                "not covered: the __get__ of what the name is",
                spec(_cl_entry(("getattr", "shown"), [x], classes=(Named,))),
            ),
            (
                "not covered: its own __getattribute__ or __getattr__",
                spec(_cl_entry(("getattr", "weight"), [x], classes=(torch.nn.Linear,))),
            ),
            (
                "not covered: its own __getattribute__ or __getattr__",
                spec(
                    _cl_entry(("getattr", "float32"), [x], classes=(types.ModuleType,))
                ),
            ),
            (
                "not covered: its own __setattr__",
                spec(
                    _cl_entry(
                        ("setattr", "weight"), [x, x], classes=(int, torch.nn.Linear)
                    )
                ),
            ),
            (
                "not covered: the __set__ of what the name is",
                spec(_cl_entry(("setattr", "shown"), [x, x], classes=(int, Named))),
            ),
            ("output 1: input 3 of 1", spec(first, outputs=[x, ("in", 3)])),
            (
                "a part of r0 with an empty path",
                spec(first, _cl_entry(add, [x, ("part", 0, ())])),
            ),
            (
                'a seq is a list or a tuple, not "set"',
                spec(_cl_entry(add, [x, ("seq", "set", [x])])),
            ),
            (
                "a map of 2 keys and 1 values",
                spec(_cl_entry(add, [x, ("map", ("a", "b"), [x])])),
            ),
            (
                "a map key that is no exact str",
                spec(_cl_entry(add, [x, ("map", (("a",),), [x])])),
            ),
            (
                "a map key that is no exact str",
                spec(_cl_entry(add, [x, ("map", (Text("a"),), [x])])),
            ),
            (
                "a null operand that is no bound of a slice",
                spec(_cl_entry(add, [x, ("null",)])),
            ),
            (
                "a null operand that is no bound of a slice",
                spec(_cl_entry(add, [x, ("seq", "list", [("null",)])])),
            ),
            (
                "a null operand that is no bound of a slice",
                spec(first, outputs=[("null",)]),
            ),
            ("slice takes 3 fields", spec(_cl_entry(add, [x, ("slice", x, x)]))),
            # keyword names
            ("keyword names", spec(_cl_entry(add, [x, x], ["alpha"]))),
            ("keyword names", spec(_cl_entry(add, [x, x], (1,)))),
            ("keyword names", spec(_cl_entry(add, [x, x], (Text("alpha"),)))),
            ("keyword names", spec(_cl_entry(add, [x], ("other", "alpha")))),
            ("keyword names", spec(_cl_entry(("binary", "+"), [x, x], ("other",)))),
            (
                "keyword names of which two are the same",
                spec(_cl_entry(add, [x, x, x], ("other", "".join(["oth", "er"])))),
            ),
            # out, drops
            (
                "out 1: the slot of an entry is its own index",
                spec(_cl_entry(add, [x, x], out=1)),
            ),
            (
                "out 0: the slot of an entry is its own index",
                spec(_cl_entry(("setitem",), [x, x, x], out=0)),
            ),
            (
                "out 0: the slot of an entry is its own index",
                spec(_cl_entry(("delitem",), [x, x], out=0)),
            ),
            (
                "out 0: the slot of an entry is its own index",
                spec(_cl_entry(("setattr", "a"), [x, x], out=0)),
            ),
            ("drops r0, which is not an earlier", spec(_cl_drop(0))),
            (
                "drops r1, which is not an earlier",
                spec(first, _cl_entry(add, [x, x], out=1), _cl_drop(1, 1)),
            ),
            (
                "drops r0, which is not an earlier",
                spec(_cl_entry(add, [x, x]), _cl_drop(0)),
            ),
            ("drops r5, which is not an earlier", spec(first, _cl_drop(5))),
            ("drops r1, which is not an earlier", spec(first, _cl_drop(0, 1))),
            (
                "an entry that starts with a str is ..drop., slots, at.",
                spec(("drop", (0,))),
            ),
            (
                "an entry that starts with a str is ..drop., slots, at.",
                spec(("keep", (0,), 0)),
            ),
            ("a dropped slot must be an int", spec(first, ("drop", ("0",), 0))),
            # flags
            (
                'no flag "idempotent"',
                spec(_cl_entry(add, [x, x], flags=["idempotent"])),
            ),
            ('no flag "reads"', spec(_cl_entry(add, [x, x], flags=[("reads", (0,))]))),
            (
                "a written or rekeyed operand that is not there",
                spec(_cl_entry(add, [x, x], flags=[("writes", (2,))])),
            ),
            (
                "a written or rekeyed operand that is not there",
                spec(_cl_entry(add, [x, x], flags=[("rekeys", (2,))])),
            ),
            (
                "the state flag on a callee that is no state setter",
                spec(_cl_entry(add, [x, x], flags=["state"])),
            ),
            # asserts
            (
                'no assert kind "value"',
                spec(_cl_entry(add, [x, x], asserts=[("value", 1)])),
            ),
            (
                "key takes 1 fields",
                spec(
                    _cl_entry(add, [x, x], asserts=[("key", ("sizes",), torch.ones(1))])
                ),
            ),
            (
                "the key of an assert is read off a tensor that has one",
                spec(_cl_entry(add, [x, x], asserts=[("key", (1,))])),
            ),
            (
                "the class of an assert must be a type",
                spec(_cl_entry(add, [x, x], asserts=[("class", "int")])),
            ),
            (
                "a result has one class",
                spec(
                    _cl_entry(
                        add, [x, x], asserts=[("class", Tensor), ("key", torch.ones(1))]
                    )
                ),
            ),
            (
                "a result has one class",
                spec(_cl_entry(add, [x, x], asserts=[("class", int), ("class", int)])),
            ),
            (
                "an assert where there is no result",
                spec(_cl_entry(("setitem",), [x, x, x], asserts=[("count", 1)])),
            ),
            (
                "on an operand that is not there",
                spec(_cl_entry(add, [x, x], asserts=[("is", 2)])),
            ),
            # the shape of the spec
            ("a call list has 5 fields, not 4", (1, 1, [], [])),
            ("a call list must be a tuple or a list", {"entries": []}),
            ("a call list must be a tuple or a list", Pair((1, 1, [], [], []))),
            ("an entry has 8 fields, not 7", spec(first[:7])),
            ("an entry must be a tuple or a list", spec(Pair(first))),
            ("a template must start with its kind", spec(_cl_entry(add, [x, ()]))),
            ("a template must be a str", spec(_cl_entry(add, [x, (1, 0)]))),
            ("a template must be a str", spec(_cl_entry(add, [x, (Text("in"), 0)]))),
            ("the count of inputs must be an int", ("1", 1, [], [], [])),
            ("at must be an int", spec(_cl_entry(add, [x, x], at=None))),
        ]
        for name, without in KEY_WITHOUT.items():
            if name != "a_wrapper_of_functorch":
                cases.append(
                    (
                        "the key of an assert is read off a tensor that has one",
                        spec(
                            _cl_entry(
                                add,
                                [x, x],
                                asserts=[("key", without(torch.ones(2, 2)))],
                            )
                        ),
                    )
                )
        for flag in ("draws", "effect", "opaque", ("writes", (0,)), ("rekeys", (0,))):
            cases.append(
                ("pure beside", spec(_cl_entry(add, [x, x], flags=["pure", flag])))
            )
        for why, bad in cases:
            with self.subTest(why=why):
                with self.assertRaisesRegex(ValueError, "torch.vm call list: .*" + why):
                    _calls_make(bad)
        # What is like a refused class and is C code is taken: a subclass that
        # adds nothing, a plain value on the class, a method.
        for callee, classes in [
            (("getitem",), (Quiet, int)),
            (("getattr", "append"), (Quiet,)),
            (("getattr", "__doc__"), (Named,)),
            (("getattr", "missing"), (Named,)),
            (("setattr", "missing"), (int, Named)),
            (("compare", "=="), (Named, int)),
            (("binary", "-"), (Tensor, float)),
        ]:
            with self.subTest(taken=callee, classes=classes):
                _calls_make(
                    spec(_cl_entry(callee, [x] * len(classes), classes=classes))
                )
        h = _calls_make(spec(first, outputs=[("res", 0)]))
        one = torch.ones(1)
        for inputs, pins in [
            ((), (one,)),
            ((one,), ()),
            ([one], (one,)),
            ((one,), [one]),
            (Pair((one,)), (one,)),
            ((one, one), (one,)),
        ]:
            with self.assertRaisesRegex(ValueError, "must be a tuple of 1"):
                _calls_run(h, inputs, pins)

    def step_list(self, second, flags=(("writes", (0,)),), asserts=()):
        """x.add_(1.0); t = <second>(x, y); x.mul_(2.0); seen.append(t)"""
        T = torch.Tensor
        return _calls_make((2, 1, [1.0, 2.0], [
            _cl_entry(_cl_vector(torch.Tensor.add_), [("in", 0), ("const", 0)], flags=flags, classes=(T, float)),
            _cl_entry(_cl_vector(second), [("in", 0), ("in", 1)], out=1, flags=["pure"], asserts=asserts or [("class", T)]),
            _cl_entry(_cl_vector(torch.Tensor.mul_), [("in", 0), ("const", 1)], flags=[("writes", (0,))], classes=(T, float)),
            _cl_entry(_cl_vector(list.append), [("pin", 0), ("move", 1)], flags=["effect"], classes=(list, T)),
        ], [("in", 0)]))  # fmt: skip

    def test_an_entry_that_raises(self):
        x, y, seen = torch.ones(2, 3), torch.ones(4, 5), []
        h = self.step_list(torch.matmul)
        with self.assertRaises(RuntimeError) as eager:
            torch.matmul(x, y)
        with self.assertRaises(RuntimeError) as mine:
            _calls_run(h, (x, y), (seen,))
        # The callee's own exception, and nothing of the engine's around it.
        self.assertEqual(str(mine.exception), str(eager.exception))
        self.assertIsNone(mine.exception.__cause__)
        self.assertIsNone(mine.exception.__context__)
        # What ran before the entry stays done; nothing behind it ran.
        self.assertTrue(torch.equal(x, torch.full((2, 3), 2.0)))
        self.assertEqual(seen, [])
        self.assertEqual(h.stopped, (1, "raised", False, {}))
        # The same list on operands that fit: the run goes through.
        y2 = torch.ones(3, 2)
        self.assertIs(_calls_run(h, (x, y2), (seen,))[0], x)
        self.assertIsNone(h.stopped)
        self.assertTrue(torch.equal(x, torch.full((2, 3), 6.0)))
        self.assertTrue(torch.equal(seen[0], torch.full((2, 2), 9.0)))
        # With nothing in front that cannot be taken back, the raise says so.
        h = self.step_list(torch.matmul, flags=["pure"])
        with self.assertRaises(RuntimeError):
            _calls_run(h, (x, y), (seen,))
        self.assertEqual(h.stopped, (1, "raised", True, {}))
        # An exception of any class comes out as it is.
        h = _calls_make(
            (
                1,
                0,
                ["k"],
                [
                    _cl_entry(
                        ("getitem",),
                        [("in", 0), ("const", 0)],
                        out=0,
                        classes=(dict, str),
                    )
                ],
                [("res", 0)],
            )
        )
        with self.assertRaisesRegex(KeyError, "'k'"):
            _calls_run(h, ({},), ())
        self.assertEqual(h.stopped, (0, "raised", False, {}))

    def test_an_assert_that_fails(self):
        y, seen = torch.ones(3, 2), []
        fits, other = torch.empty(2, 2), torch.empty(2, 5)
        for flags, restartable in ((["pure"], True), ([("writes", (0,))], False)):
            with self.subTest(restartable=restartable):
                x = torch.ones(2, 3)
                h = self.step_list(torch.matmul, flags=flags, asserts=[("key", fits)])
                self.assertIs(_calls_run(h, (x, y), (seen,))[0], x)
                self.assertIsNone(h.stopped)
                x = torch.ones(2, 3)
                del seen[:]
                h = self.step_list(torch.matmul, flags=flags, asserts=[("key", other)])
                if restartable:
                    # A miss: no outputs, no exception.
                    self.assertIsNone(_calls_run(h, (x, y), (seen,)))
                else:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        r"torch.vm: call list entry 1: the result is not what the traced call read of it \(key=tensor\(f32 cpu \[2, 5\] stride \[5, 1\]\)\), and what ran before cannot be taken back: 1  r1 = .*matmul\(in0, in1\)",
                    ):
                        _calls_run(h, (x, y), (seen,))
                # The result that failed is in its slot: the caller's.
                entry, what, again, live = h.stopped
                self.assertEqual((entry, what, again), (1, "assert", restartable))
                self.assertEqual(
                    {j: tuple(t.shape) for j, t in live.items()}, {1: (2, 2)}
                )
                del live
                # Nothing behind the entry ran.
                self.assertTrue(torch.equal(x, torch.full((2, 3), 2.0)))
                self.assertEqual(seen, [])

    def held(self, value, asserts, callee=None, operands=None):
        """(outputs, stopped without the slots) of one pure entry that hands
        back ``value`` (or of ``callee`` on ``operands``) under ``asserts``."""
        operands = ({"k": value}, "k") if operands is None else operands
        entry = _cl_entry(
            callee or _cl_vector(dict.get),
            [("in", k) for k in range(len(operands))],
            out=0,
            flags=["pure"],
            asserts=asserts,
            classes=tuple(type(v) for v in operands),
        )
        h = _calls_make((len(operands), 0, [], [entry, _cl_drop(0)], []))
        return _calls_run(h, tuple(operands), ()), h.stopped and h.stopped[:3]

    def test_an_assert_of_a_key_is_the_whole_key(self):
        missed = (None, (0, "assert", True))
        for field, (base, other) in KEY_SWEEP.items():
            with self.subTest(field=field):
                x = torch.randn(4, 8)
                traced, same, differs = base(x.clone()), base(x.clone()), other(x)
                self.assertEqual(self.held(same, [("key", traced)]), ((), None))
                # One field of the key is another: the assert fails.
                self.assertEqual(self.held(differs, [("key", traced)]), missed)
                self.assertEqual(self.held(same, [("key", differs)]), missed)
        # A result that has no key has no key to compare.
        x = torch.ones(2, 2)
        for name, without in KEY_WITHOUT.items():
            with self.subTest(without=name):
                self.assertEqual(self.held(without(x), [("key", x)]), missed)
        self.assertEqual(self.held(1.0, [("key", x)]), missed)

    def test_the_other_asserts(self):
        missed = (None, (0, "assert", True))
        x, T = torch.ones(2, 3), torch.Tensor

        class Pair(tuple):
            __slots__ = ()

        # The exact class.
        self.assertEqual(self.held(x, [("class", T)]), ((), None))
        self.assertEqual(self.held(torch.nn.Parameter(x), [("class", T)]), missed)
        self.assertEqual(self.held(True, [("class", int)]), missed)
        self.assertEqual(self.held(Pair((1, 2)), [("class", tuple)]), missed)
        # The count: of an exact tuple or a struct sequence only.
        pair = torch.max(x, 0)
        self.assertEqual(self.held(pair, [("count", 2)]), ((), None))
        self.assertEqual(self.held(pair, [("count", 3)]), missed)
        self.assertEqual(self.held((1, 2), [("count", 2)]), ((), None))
        self.assertEqual(self.held(Pair((1, 2)), [("count", 2)]), missed)
        self.assertEqual(self.held([1, 2], [("count", 2)]), missed)
        self.assertEqual(self.held(x, [("count", 2)]), missed)
        # The result is an operand.
        add_, add = _cl_vector(torch.Tensor.add_), _cl_vector(torch.add)
        self.assertEqual(
            self.held(None, [("is", 0)], add_, (torch.ones(2), 1.0)), ((), None)
        )
        self.assertEqual(
            self.held(None, [("is", 1)], add_, (torch.ones(2), torch.ones(2))), missed
        )
        self.assertEqual(self.held(None, [("is", 0)], add, (x, x)), missed)
        # Several at once: each is checked.
        both = (torch.ones(2, 3), 1.0)
        self.assertEqual(
            self.held(None, [("is", 0), ("key", x)], add_, both), ((), None)
        )
        self.assertEqual(
            self.held(None, [("is", 0), ("key", torch.ones(3, 2))], add_, both), missed
        )

    def test_a_part_that_is_not_there(self):
        x, pair = torch.ones(2, 3), torch.return_types.max

        class Pair(tuple):
            __slots__ = ()

            def __getitem__(self, i):
                return "python"

        for flags, restartable in ((["pure"], True), ([], False)):
            for path in [(2,), (0, 0)]:  # too short; a tensor is no tuple
                with self.subTest(restartable=restartable, path=path):
                    first = _cl_entry(
                        _cl_vector(torch.max),
                        [("in", 0), ("const", 0)],
                        out=0,
                        flags=flags,
                        asserts=[("class", pair)],
                        classes=(torch.Tensor, int),
                    )
                    wants = _cl_entry(_cl_vector(torch.neg), [("part", 0, path)], out=1)
                    h = _calls_make(
                        (1, 0, [0], [first, wants, _cl_drop(0)], [("res", 1)])
                    )
                    if restartable:
                        self.assertIsNone(_calls_run(h, (x,), ()))
                    else:
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "entry 1: a part of a result that the entry is called on is not there",
                        ):
                            _calls_run(h, (x,), ())
                    # The entry that wanted it has not run: its slot is empty.
                    self.assertEqual(h.stopped[:3], (1, "part", restartable))
                    self.assertEqual(list(h.stopped[3]), [0])
                    h = _calls_make((1, 0, [0], [first], [("part", 0, path)]))
                    if restartable:
                        self.assertIsNone(_calls_run(h, (x,), ()))
                    else:
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "entry 1: a part of a result that an output is made of is not there",
                        ):
                            _calls_run(h, (x,), ())
                    self.assertEqual(h.stopped[:3], (1, "part", restartable))
        # A part is read off an exact tuple or a struct sequence: a list has
        # none, and a subclass of tuple could answer `[0]` with its own code.
        for value in ([1, 2], Pair((1, 2)), {0: 1}, "ab"):
            with self.subTest(value=type(value).__name__):
                got = _calls_make((1, 0, ["k"], [
                    _cl_entry(_cl_vector(dict.get), [("in", 0), ("const", 0)], out=0, flags=["pure"], classes=(dict, str)),
                ], [("part", 0, (0,))]))  # fmt: skip
                self.assertIsNone(_calls_run(got, ({"k": value},), ()))
                self.assertEqual(got.stopped[:3], (1, "part", True))
        self.assertEqual(Pair((1, 2))[0], "python")

    def test_restartable_is_every_entry_up_to_the_one_that_stopped(self):
        x, y = torch.ones(2, 3), torch.ones(4, 5)
        known = [("class", torch.Tensor)]
        fresh = _cl_entry(
            _cl_vector(torch.add),
            [("in", 0), ("in", 0)],
            out=0,
            flags=["pure", "fresh"],
            asserts=known,
        )
        made = _cl_entry(
            _cl_vector(torch.add),
            [("in", 0), ("in", 0)],
            out=0,
            flags=["pure"],
            asserts=known,
        )
        fails = _cl_entry(
            _cl_vector(torch.matmul), [("in", 0), ("in", 1)], flags=["pure"]
        )

        def write(target, *flags):
            return _cl_entry(
                _cl_vector(torch.Tensor.add_),
                [target, ("in", 0)],
                flags=[("writes", (0,)), *flags],
            )

        seq = _cl_entry(
            _cl_vector(torch._foreach_neg_),
            [("seq", "list", [("res", 0)])],
            flags=[("writes", (0,))],
        )
        for restartable, entries in [
            (True, [fresh, write(("res", 0)), fails]),  # a write to what the run made
            (False, [made, write(("res", 0)), fails]),  # not known to be the run's own
            (False, [fresh, write(("in", 0)), fails]),  # a write to an input
            (False, [fresh, write(("res", 0), "draws"), fails]),
            (False, [fresh, write(("res", 0), "effect"), fails]),
            (False, [fresh, write(("res", 0), "opaque"), fails]),
            (False, [fresh, write(("res", 0), ("rekeys", (0,))), fails]),
            (False, [fresh, seq, fails]),  # a written container of results
            (
                False,
                [fresh, _cl_entry(_cl_vector(torch.neg), [("res", 0)]), fails],
            ),  # no flag: nothing is known
            (
                False,
                [fresh, _cl_entry(_cl_vector(torch.matmul), [("in", 0), ("in", 1)])],
            ),  # the entry that raised counts
            (True, [fresh, fails, write(("in", 0))]),  # what is behind does not
        ]:
            entries = entries + [_cl_drop(0)]
            with self.subTest(text=_calls_text(_calls_make((2, 0, [], entries, [])))):
                h = _calls_make((2, 0, [], entries, []))
                with self.assertRaises(RuntimeError):
                    _calls_run(h, (x.clone(), y), ())
                self.assertEqual(h.stopped[1:3], ("raised", restartable))
        # A class that is not the pinned one stops in front of its entry: what
        # counts is what ran before it. A drop takes nothing back.
        for restartable, entries in [
            (True, [fresh, _cl_drop(0), write(("in", 1))]),
            (False, [fresh, write(("in", 0)), _cl_drop(0), write(("in", 1))]),
        ]:
            with self.subTest(restartable=restartable):
                h = _calls_make((2, 0, [], entries, []))
                if restartable:
                    self.assertIsNone(_calls_run(h, (x.clone(), 1.0), ()))
                else:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "entry 3: an operand is not of the class it had at the trace",
                    ):
                        _calls_run(h, (x.clone(), 1.0), ())
                self.assertEqual(
                    h.stopped, (len(entries) - 1, "class", restartable, {})
                )

    def logs(self, body, entries, operands, consts=()):
        """What dies when: the log of the plain program ``body(*pins, log)``
        and the log of the list on pins made the same way."""
        both = []
        for run in ("eager", "list"):
            log = []
            pins = operands(log)
            if run == "eager":
                body(*pins, log)
            else:
                spec = (0, len(pins) + 1, list(consts), entries, [])
                self.assertEqual(_cl_run(spec, (), (*pins, log)), ())
            del pins
            both.append([x if isinstance(x, str) else "?" for x in log])
        return both

    def test_a_temporary_dies_when_its_call_returns(self):
        """An operand that is a temporary goes when the callee has returned,
        before the result is dropped; a local goes where it is deleted."""
        F = _ClFinal

        def temporary(src, d, log):
            d.pop("k", src.pop())

        def local(src, d, log):
            a = src.pop()
            d.pop("k", a)
            del a

        pop = _cl_entry(_cl_vector(list.pop), [("pin", 0)], out=0, asserts=[("class", F)], classes=(list,))  # fmt: skip

        def call(operand):
            return _cl_entry(_cl_vector(dict.pop), [("pin", 1), ("const", 0), operand], classes=(dict, str, F))  # fmt: skip

        for body, entries, want in [
            (temporary, [pop, call(("move", 0))], ["operand", "result"]),
            (local, [pop, call(("res", 0)), _cl_drop(0)], ["result", "operand"]),
        ]:
            with self.subTest(body=body.__name__):
                eager, mine = self.logs(
                    body,
                    entries,
                    lambda log: ([F("operand", log)], {"k": F("result", log)}),
                    ["k"],
                )
                self.assertEqual(eager, want)
                self.assertEqual(mine, eager)

    def test_arguments_die_in_their_order(self):
        """What a call was given goes in the order of its arguments, and what
        was built for it goes there as the built object lets go of it."""
        F = _ClFinal

        def body(src, log):
            operator.is_([src.pop(), src.pop()], src.pop())
            log.append("returned")
            operator.is_(src.pop(), src.pop())

        def pop(k):
            return _cl_entry(_cl_vector(list.pop), [("pin", 0)], out=k, asserts=[("class", F)], classes=(list,))  # fmt: skip

        is_ = _cl_vector(operator.is_)
        entries = [
            pop(0),
            pop(1),
            pop(2),
            _cl_entry(is_, [("seq", "list", [("move", 0), ("move", 1)]), ("move", 2)], classes=((F, F), F)),
            _cl_entry(_cl_vector(list.append), [("pin", 1), ("const", 0)], classes=(list, str)),
            pop(5),
            pop(6),
            _cl_entry(is_, [("move", 5), ("move", 6)], classes=(F, F)),
        ]  # fmt: skip
        eager, mine = self.logs(
            body,
            entries,
            lambda log: ([F(name, log) for name in ("e", "d", "c", "b", "a")],),
            ["returned"],
        )
        self.assertEqual(eager, ["b", "a", "c", "returned", "d", "e"])
        self.assertEqual(mine, eager)

    def test_a_tensor_dies_where_the_program_lets_go_of_it(self):
        """u = t * 2.0; v = u - 1.0 with t a temporary of the multiply, or a
        local deleted behind the subtract. A weak reference's callback says
        when t went."""
        T = torch.Tensor

        def temporary(src, log):
            u = src.pop() * 2.0
            log.append("multiplied")
            v = u - 1.0
            log.append("subtracted")

        def local(src, log):
            t = src.pop()
            u = t * 2.0
            log.append("multiplied")
            v = u - 1.0
            del t
            log.append("subtracted")

        def mark(k):
            return _cl_entry(_cl_vector(list.append), [("pin", 1), ("const", k)], classes=(list, str))  # fmt: skip

        pop = _cl_entry(_cl_vector(list.pop), [("pin", 0)], out=0, asserts=[("class", T)], classes=(list,))  # fmt: skip

        def times(operand):
            return _cl_entry(("binary", "*"), [operand, ("const", 2)], out=1, asserts=[("class", T)], classes=(T, float))  # fmt: skip

        minus = _cl_entry(("binary", "-"), [("move", 1), ("const", 3)], out=3, classes=(T, float))  # fmt: skip
        keep = []

        def operands(log):
            t = torch.ones(3)
            keep.append(weakref.ref(t, lambda gone: log.append("t dies")))
            return ([t],)

        for body, entries, want in [
            (
                temporary,
                [pop, times(("move", 0)), mark(0), minus, mark(1), _cl_drop(3)],
                ["t dies", "multiplied", "subtracted"],
            ),
            (
                local,
                [
                    pop,
                    times(("res", 0)),
                    mark(0),
                    minus,
                    _cl_drop(0),
                    mark(1),
                    _cl_drop(3),
                ],
                ["multiplied", "t dies", "subtracted"],
            ),
        ]:
            with self.subTest(body=body.__name__):
                eager, mine = self.logs(
                    body, entries, operands, ["multiplied", "subtracted", 2.0, 1.0]
                )
                self.assertEqual(eager, want)
                self.assertEqual(mine, eager)

    def test_a_local_that_is_overwritten(self):
        F = _ClFinal

        def body(src, log):
            a = src.pop()
            log.append("popped")
            a = src.pop()
            log.append("rebound")

        def pop(k):
            return _cl_entry(_cl_vector(list.pop), [("pin", 0)], out=k, classes=(list,))

        def mark(k):
            return _cl_entry(_cl_vector(list.append), [("pin", 1), ("const", k)], classes=(list, str))  # fmt: skip

        # The new value is stored, then the old one goes; the last one goes
        # when the body ends.
        entries = [pop(0), mark(0), pop(2), _cl_drop(0), mark(1), _cl_drop(2)]
        eager, mine = self.logs(
            body,
            entries,
            lambda log: ([F("second", log), F("first", log)],),
            ["popped", "rebound"],
        )
        self.assertEqual(eager, ["popped", "first", "rebound", "second"])
        self.assertEqual(mine, eager)

    def test_a_tuple_result_partly_used(self):
        F = _ClFinal

        def unpacked(d, log):
            a, b = d.pop("k")
            del a
            log.append("deleted")

        def read(d, log):
            t = d.pop("k")
            log.append(t[1].name + " read")
            del t
            log.append("deleted")

        pop = _cl_entry(_cl_vector(dict.pop), [("pin", 0), ("const", 0)], out=0, asserts=[("class", tuple)], classes=(dict, str))  # fmt: skip

        def item(operand, i, k):
            return _cl_entry(("getitem",), [operand, ("const", i)], out=k, asserts=[("class", F)], classes=(tuple, int))  # fmt: skip

        def mark(operand, classes):
            return _cl_entry(
                _cl_vector(list.append), [("pin", 1), operand], classes=classes
            )

        name = _cl_entry(("getattr", "name"), [("part", 0, (1,))], out=1, asserts=[("class", str)], classes=(F,))  # fmt: skip
        text = _cl_entry(("binary", "+"), [("move", 1), ("const", 4)], out=2, asserts=[("class", str)], classes=(str, str))  # fmt: skip
        for body, entries, want in [
            (
                # The tuple goes with the unpacking: each item is on its own.
                unpacked,
                [
                    pop,
                    item(("res", 0), 1, 1),
                    item(("move", 0), 2, 2),
                    _cl_drop(1),
                    mark(("const", 3), (list, str)),
                    _cl_drop(2),
                ],
                ["first", "deleted", "second"],
            ),
            (
                # A part is read off the tuple, which stays until it is deleted.
                read,
                [
                    pop,
                    name,
                    text,
                    mark(("move", 2), (list, str)),
                    _cl_drop(0),
                    mark(("const", 3), (list, str)),
                ],
                ["second read", "second", "first", "deleted"],
            ),
        ]:
            with self.subTest(body=body.__name__):
                eager, mine = self.logs(
                    body,
                    entries,
                    lambda log: ({"k": (F("first", log), F("second", log))},),
                    ["k", 0, 1, "deleted", " read"],
                )
                self.assertEqual(eager, want)
                self.assertEqual(mine, eager)

    def test_a_stop_keeps_its_slots(self):
        """a = src.pop(); b = src.pop(); torch.add(x, "bad") under a handler:
        the frame the traceback holds keeps a and b until the handler is
        done. A list that stops hands its slots over for that."""
        x, F = torch.ones(2), _ClFinal

        def body(src):
            a = src.pop()
            b = src.pop()
            torch.add(x, "bad")

        def pop(k):
            return _cl_entry(_cl_vector(list.pop), [("pin", 0)], out=k, classes=(list,))

        fails = _cl_entry(_cl_vector(torch.add), [("in", 0), ("const", 0)], classes=(torch.Tensor, str))  # fmt: skip
        h = _calls_make((1, 1, ["bad"], [pop(0), pop(1), fails, _cl_drop(0, 1)], []))
        eager, mine = [], []
        try:
            body([F("b", eager), F("a", eager)])
        except TypeError:
            eager.append("handler")
        eager.append("after")
        try:
            _calls_run(h, (x,), ([F("b", mine), F("a", mine)],))
        except TypeError:
            mine.append("handler")
            entry, what, restartable, live = h.stopped
            self.assertEqual((entry, what, restartable), (2, "raised", False))
            self.assertEqual({j: f.name for j, f in live.items()}, {0: "a", 1: "b"})
            del live
            h.release()
        mine.append("after")
        self.assertEqual(eager, ["handler", "a", "b", "after"])
        self.assertEqual(mine, eager)
        self.assertEqual(h.stopped, (2, "raised", False, {}))
        # Not released by hand, they stay until the next run starts.
        del mine[:]
        with self.assertRaises(TypeError):
            _calls_run(h, (x,), ([F("b", mine), F("a", mine)],))
        self.assertEqual(mine, [])
        with self.assertRaises(IndexError):
            _calls_run(h, (x,), ([],))
        self.assertEqual(mine, ["a", "b"])
        self.assertEqual(h.stopped, (0, "raised", False, {}))

    def test_stopped_is_cleared_when_a_run_starts(self):
        x = torch.ones(2)
        entry = _cl_entry(_cl_vector(torch.add), [("in", 0), ("in", 1)], out=0, classes=(torch.Tensor, str))  # fmt: skip
        h = _calls_make((2, 0, [], [entry], [("res", 0)]))
        with self.assertRaises(TypeError):
            _calls_run(h, (x, "bad"), ())
        self.assertEqual(h.stopped, (0, "raised", False, {}))
        # A call that is turned down before anything runs says nothing of the
        # run before it.
        with self.assertRaisesRegex(ValueError, "inputs must be a tuple of 2"):
            _calls_run(h, (x,), ())
        self.assertIsNone(h.stopped)

    def test_an_operand_of_another_class_stops_its_entry(self):
        ran, T = [], torch.Tensor

        class Loud:
            def __add__(self, other):
                ran.append("__add__")
                return 1

            def __index__(self):
                ran.append("__index__")
                return 0

        def run(callee, classes, operands):
            args = [("in", k) for k in range(len(operands))]
            entry = _cl_entry(callee, args, out=0, flags=["pure"], classes=classes)
            h = _calls_make((len(operands), 0, [], [entry], [("res", 0)]))
            return _calls_run(h, tuple(operands), ()), h.stopped

        missed = (None, (0, "class", True, {}))
        x, w = torch.ones(2), torch.nn.Parameter(torch.ones(2))
        self.assertEqual(run(("binary", "+"), (int, int), (2, 3)), ((5,), None))
        for callee, classes, operands in [
            (("binary", "+"), (int, int), (Loud(), 1)),
            (("binary", "+"), (int, int), (True, 1)),
            (("getitem",), (list, int), ([1, 2], Loud())),
            (_cl_vector(torch.add), (T, T), (x, 2.0)),
            (_cl_vector(torch.add), (T, T), (w, x)),
            (_cl_vector(torch.add), (T, T), (x, None)),
        ]:
            with self.subTest(callee=callee[0], operands=operands):
                self.assertEqual(run(callee, classes, operands), missed)
        # A part of what is built for the call is an operand too.
        stack = _cl_entry(_cl_vector(torch.stack), [("seq", "list", [("in", 0), ("in", 1)])], out=0, flags=["pure"])  # fmt: skip
        h = _calls_make((2, 0, [], [stack], [("res", 0)]))
        self.assertTrue(torch.equal(_calls_run(h, (x, x), ())[0], torch.stack([x, x])))
        self.assertIsNone(_calls_run(h, (x, w), ()))
        self.assertEqual(h.stopped, (0, "class", True, {}))
        self.assertEqual(ran, [])

    def test_a_class_that_changed_since_the_list_was_made(self):
        ran = []

        class Box(list):
            pass

        entry = _cl_entry(("getitem",), [("in", 0), ("in", 1)], out=0, flags=["pure"], classes=(Box, int))  # fmt: skip
        h = _calls_make((2, 0, [], [entry], [("res", 0)]))
        self.assertEqual(_calls_run(h, (Box([5, 6]), 1), ()), (6,))
        Box.__getitem__ = lambda self, i: ran.append(i)
        # The class now answers a subscript with its own code: the list, which
        # was made for the C one, stops in front of the entry.
        self.assertIsNone(_calls_run(h, (Box([5, 6]), 1), ()))
        self.assertEqual(h.stopped, (0, "class", True, {}))
        self.assertEqual(ran, [])
        with self.assertRaisesRegex(ValueError, "no C function: __getitem__"):
            _calls_make((2, 0, [], [entry], [("res", 0)]))

    def test_a_list_that_is_entered_while_it_runs(self):
        count = []
        nothing = _calls_make((0, 0, [], [], []))
        CallList = type(nothing)
        again = _cl_entry(_cl_vector(_calls_run), [("pin", 0), ("seq", "tuple", []), ("seq", "tuple", [("pin", 0), ("pin", 1)])], classes=(CallList, (), (CallList, list)))  # fmt: skip
        h = _calls_make((0, 2, [1], [
            _cl_entry(_cl_vector(list.append), [("pin", 1), ("const", 0)], classes=(list, int)),
            again,
        ], []))  # fmt: skip
        with self.assertRaisesRegex(RuntimeError, "entered while this thread runs it"):
            _calls_run(h, (), (h, count))
        self.assertEqual(count, [1])
        self.assertEqual(h.stopped, (1, "raised", False, {}))
        # Lists that run one another: so many deep, and no deeper.
        inner = _cl_entry(_cl_vector(_calls_run), [("pin", 0), ("seq", "tuple", []), ("pin", 1)], classes=(CallList, (), tuple))  # fmt: skip

        def chain(depth):
            first, pins = nothing, ()
            for _ in range(depth):
                first, pins = _calls_make((0, 2, [], [inner], [])), (first, pins)
            return first, pins

        first, pins = chain(31)
        self.assertEqual(_calls_run(first, (), pins), ())
        first, pins = chain(40)
        with self.assertRaisesRegex(RecursionError, "call lists nested deeper than 32"):
            _calls_run(first, (), pins)
        # The marks are gone with the runs.
        self.assertEqual(_calls_run(nothing, (), ()), ())
        with self.assertRaisesRegex(RuntimeError, "entered while"):
            _calls_run(h, (), (h, count))


@unittest.skipUnless(CHECKS_HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmCallListPrograms(TestCase):
    """Three programs written down as call lists by hand, each run against
    the plain Python it was taken from: the results are the same bits and so
    is every tensor the program writes.
    """

    hw_classification = HardwareClassification.ACCELERATOR

    def test_conv_block(self, device):
        gen = torch.Generator().manual_seed(0)

        def rand(*shape):
            return torch.randn(*shape, generator=gen).to(device)

        def state():
            # conv weight, then weight, bias, running mean, running var of a batch norm, twice
            gen.manual_seed(1)
            return [
                rand(4, 4, 3, 3),
                rand(4),
                rand(4),
                rand(4),
                rand(4).abs() + 0.5,
                rand(4, 4, 3, 3),
                rand(4),
                rand(4),
                rand(4),
                rand(4).abs() + 0.5,
            ]

        cudnn = torch.backends.cudnn.enabled

        def block(x, p, training):
            out = torch.conv2d(x, p[0], None, (1, 1), (1, 1), (1, 1), 1)
            out = torch.batch_norm(
                out, p[1], p[2], p[3], p[4], training, 0.1, 1e-05, cudnn
            )
            out = torch.relu_(out)
            out = torch.conv2d(out, p[5], None, (1, 1), (1, 1), (1, 1), 1)
            out = torch.batch_norm(
                out, p[6], p[7], p[8], p[9], training, 0.1, 1e-05, cudnn
            )
            out += x
            return torch.relu_(out)

        def calls(training):
            conv, bn, relu_ = (
                _cl_vector(torch.conv2d),
                _cl_vector(torch.batch_norm),
                _cl_vector(torch.relu_),
            )
            tail = [("const", k) for k in (0, 1, 1, 1, 2)]
            norm = [("const", k) for k in (3, 4, 5, 6)]

            def pin(*ks):
                return [("pin", k) for k in ks]

            T, known = torch.Tensor, [("class", torch.Tensor)]
            same = [("is", 0), *known]
            as_conv = (T, T, type(None), tuple, tuple, tuple, int)
            as_bn = (T, T, T, T, T, bool, float, float, bool)
            # Each `out` is a temporary of the next call: moved into it.
            return _calls_make((1, 10, [None, (1, 1), 1, training, 0.1, 1e-05, cudnn], [
                _cl_entry(conv, [("in", 0), ("pin", 0), *tail], out=0, at=0, asserts=known, classes=as_conv),
                _cl_entry(bn, [("move", 0), *pin(1, 2, 3, 4), *norm], out=1, at=1, asserts=known, classes=as_bn),
                _cl_entry(relu_, [("move", 1)], out=2, at=2, asserts=same),
                _cl_entry(conv, [("move", 2), ("pin", 5), *tail], out=3, at=3, asserts=known, classes=as_conv),
                _cl_entry(bn, [("move", 3), *pin(6, 7, 8, 9), *norm], out=4, at=4, asserts=known, classes=as_bn),
                _cl_entry(("binary", "+="), [("move", 4), ("in", 0)], out=5, at=5, asserts=same),
                _cl_entry(relu_, [("move", 5)], out=6, at=6),
            ], [("res", 6)]))  # fmt: skip

        for training in (False, True):
            with self.subTest(training=training):
                h, mine, eager = calls(training), state(), state()
                for seed in (2, 3):
                    gen.manual_seed(seed)
                    x = rand(2, 4, 8, 8)
                    x_mine = x.clone()
                    got, want = (
                        _calls_run(h, (x_mine,), tuple(mine))[0],
                        block(x, eager, training),
                    )
                    self.assertTrue(torch.equal(got, want))
                    self.assertTrue(torch.equal(x_mine, x))
                    # The running statistics: untouched in eval mode, written
                    # once per call, as the plain call writes them, in training.
                    for p, q in zip(mine, eager):
                        self.assertTrue(torch.equal(p, q))
                self.assertEqual(torch.equal(mine[3], state()[3]), not training)

    def test_attention_chain(self, device):
        gen = torch.Generator().manual_seed(0)
        batch, length, width, heads = 2, 5, 8, 2
        linear = torch._C._nn.linear
        p = [
            torch.randn(*shape, generator=gen).to(device)
            for shape in [(width, width), (width,)] * 4
        ]
        scale = math.sqrt(width // heads)

        def layer(h):
            q = (
                linear(h, p[0], p[1])
                .view(batch, length, heads, width // heads)
                .transpose(1, 2)
            )
            k = (
                linear(h, p[2], p[3])
                .view(batch, length, heads, width // heads)
                .transpose(1, 2)
            )
            v = (
                linear(h, p[4], p[5])
                .view(batch, length, heads, width // heads)
                .transpose(1, 2)
            )
            scores = torch.matmul(q, k.transpose(-1, -2)) / scale
            probs = scores.softmax(dim=-1)
            context = (
                torch.matmul(probs, v)
                .transpose(1, 2)
                .contiguous()
                .view(batch, length, width)
            )
            return linear(context, p[6], p[7]) + h

        split = [("const", k) for k in (0, 1, 2, 3)]
        lin, view, transpose = (
            _cl_vector(linear),
            _cl_vector(torch.Tensor.view),
            _cl_vector(torch.Tensor.transpose),
        )
        matmul = _cl_vector(torch.matmul)
        entries = []

        consts = [batch, length, heads, width // heads, 1, -1, -2, scale, width, 2]

        def call(callee, operands, kwnames=None):
            # Every intermediate is a temporary of the one call that takes it.
            entries.append(
                _cl_entry(
                    callee,
                    operands,
                    kwnames,
                    out=len(entries),
                    at=len(entries),
                    flags=["pure"],
                    asserts=[("class", torch.Tensor)],
                    classes=tuple(
                        type(consts[t[1]]) if t[0] == "const" else torch.Tensor
                        for t in operands
                    ),
                )
            )
            return ("move", len(entries) - 1)

        def head(weight):
            flat = call(lin, [("in", 0), ("pin", weight), ("pin", weight + 1)])
            split_ = call(view, [flat, *split])
            return call(transpose, [split_, ("const", 4), ("const", 9)])

        q, k, v = head(0), head(2), head(4)
        kt = call(transpose, [k, ("const", 5), ("const", 6)])
        raw = call(matmul, [q, kt])
        scores = call(("binary", "/"), [raw, ("const", 7)])
        probs = call(_cl_vector(torch.Tensor.softmax), [scores, ("const", 5)], ("dim",))
        mixed = call(matmul, [probs, v])
        moved = call(transpose, [mixed, ("const", 4), ("const", 9)])
        packed = call(_cl_vector(torch.Tensor.contiguous), [moved])
        context = call(view, [packed, ("const", 0), ("const", 1), ("const", 8)])
        dense = call(lin, [context, ("pin", 6), ("pin", 7)])
        out = call(("binary", "+"), [dense, ("in", 0)])
        h = _calls_make((1, 8, consts, entries, [("res", out[1])]))
        self.assertEqual(len(entries), 19)
        for seed in (1, 2):
            gen.manual_seed(seed)
            x = torch.randn(batch, length, width, generator=gen).to(device)
            self.assertTrue(torch.equal(_calls_run(h, (x,), tuple(p))[0], layer(x)))

    def test_foreach_optimizer_step(self, device):
        gen = torch.Generator().manual_seed(0)
        shapes = [(3, 4), (4,), (2, 3, 2)]

        def tensors():
            return [torch.randn(*shape, generator=gen).to(device) for shape in shapes]

        def step(params, grads, bufs, momentum=0.9, lr=0.1, decay=0.01):
            grads = torch._foreach_add(grads, params, alpha=decay)
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, [grads[0], grads[1], grads[2]])
            torch._foreach_add_(params, bufs, alpha=-lr)

        n = len(shapes)
        params = ("seq", "list", [("pin", k) for k in range(n)])
        bufs = ("seq", "list", [("pin", n + k) for k in range(n)])
        grads = ("seq", "list", [("in", k) for k in range(n)])
        decayed = ("seq", "list", [("part", 0, (k,)) for k in range(n)])
        three = (torch.Tensor,) * n
        h = _calls_make((n, 2 * n, [0.01, 0.9, -0.1], [
            _cl_entry(_cl_vector(torch._foreach_add), [grads, params, ("const", 0)], ("alpha",), out=0, flags=["pure", "fresh"], asserts=[("count", n), ("class", tuple)], classes=(three, three, float)),
            _cl_entry(_cl_vector(torch._foreach_mul_), [bufs, ("const", 1)], at=1, flags=[("writes", (0,))], classes=(three, float)),
            _cl_entry(_cl_vector(torch._foreach_add_), [bufs, decayed], at=2, flags=[("writes", (0,))]),
            _cl_entry(_cl_vector(torch._foreach_add_), [params, bufs, ("const", 2)], ("alpha",), at=3, flags=[("writes", (0,))], classes=(three, three, float)),
            # `grads`, the local, goes when the body ends.
            _cl_drop(0, at=4),
        ], []))  # fmt: skip
        gen.manual_seed(1)
        mine = tensors() + tensors()
        eager = [t.clone() for t in mine]
        for _ in range(3):
            g = tensors()
            self.assertEqual(_calls_run(h, tuple(g), tuple(mine)), ())
            step(eager[:n], g, eager[n:])
            for p, q in zip(mine, eager):
                self.assertTrue(torch.equal(p, q))


instantiate_device_type_tests(TestVmCallListPrograms, globals())


import contextlib
import io

from torch._rust import _witness, _witness_names, _witness_pause


class _WtMake:
    """Operands for one sample: the same values every time it is built."""

    def __init__(self, device):
        self.device = device
        self.gen = torch.Generator().manual_seed(0)

    def f(self, *shape):
        return torch.randn(*shape, generator=self.gen).to(self.device)

    def pos(self, *shape):
        return self.f(*shape).abs() + 0.5

    def i(self, high, *shape):
        return torch.randint(high, shape, generator=self.gen).to(self.device)

    def fs(self, positive=False):
        make = self.pos if positive else self.f
        return [make(3), make(2, 2)]

    def lstm(self):
        sizes = [(20, 4), (20, 5), (20,), (20,), (20, 5), (20, 5), (20,), (20,)]
        return [self.f(*size) * 0.1 for size in sizes]


# The first batch of the measured table of torch's C callees: one sample each.
_WT_SAMPLES = {
    "torch.conv2d": lambda m: ((m.f(2, 3, 8, 8), m.f(4, 3, 3, 3), m.f(4)), {"padding": 1}),
    "torch.batch_norm": lambda m: ((m.f(4, 3, 5, 5), m.f(3), m.f(3), m.f(3), m.pos(3), True, 0.1, 1e-5, True), {}),
    "torch.relu_": lambda m: ((m.f(3, 4),), {}),
    "torch.relu": lambda m: ((m.f(3, 4),), {}),
    "torch.max_pool2d": lambda m: ((m.f(2, 3, 8, 8), 2), {}),
    "torch._C._nn.adaptive_avg_pool2d": lambda m: ((m.f(2, 3, 8, 8), (1, 1)), {}),
    "torch.flatten": lambda m: ((m.f(2, 3, 4), 1), {}),
    "torch._C._nn.linear": lambda m: ((m.f(5, 4), m.f(3, 4), m.f(3)), {}),
    "torch.Tensor.__iadd__": lambda m: ((m.f(3, 4), m.f(3, 4)), {}),
    "torch.Tensor.add_": lambda m: ((m.f(3, 4), m.f(3, 4)), {"alpha": 2}),
    "torch.Tensor.add": lambda m: ((m.f(3, 4), m.f(4)), {}),
    "torch.layer_norm": lambda m: ((m.f(2, 5, 4), (4,), m.f(4), m.f(4), 1e-5, False), {}),
    "torch._C._nn.gelu": lambda m: ((m.f(3, 4),), {}),
    "torch.dropout": lambda m: ((m.f(8, 8), 0.5, True), {}),
    "torch.matmul": lambda m: ((m.f(2, 3, 4), m.f(4, 5)), {}),
    "torch.Tensor.matmul": lambda m: ((m.f(3, 4), m.f(4)), {}),
    "torch.softmax": lambda m: ((m.f(3, 4), -1), {}),
    "torch.Tensor.softmax": lambda m: ((m.f(3, 4), 0), {}),
    "torch.Tensor.view": lambda m: ((m.f(3, 4), 2, 6), {}),
    "torch.Tensor.permute": lambda m: ((m.f(2, 3, 4), 2, 0, 1), {}),
    "torch.Tensor.transpose": lambda m: ((m.f(2, 3, 4), 0, 2), {}),
    "torch.Tensor.reshape": lambda m: ((m.f(3, 4).t(), 12), {}),
    "torch.Tensor.contiguous": lambda m: ((m.f(3, 4).t(),), {}),
    "torch.Tensor.size": lambda m: ((m.f(3, 4),), {}),
    "torch.Tensor.dim": lambda m: ((m.f(3, 4),), {}),
    "torch.Tensor.item": lambda m: ((m.f(1),), {}),
    "torch.Tensor.mul": lambda m: ((m.f(3, 4), 2.5), {}),
    "torch.Tensor.shape.__get__": lambda m: ((m.f(3, 4),), {}),
    "torch.Tensor.device.__get__": lambda m: ((m.f(3, 4),), {}),
    "torch.Tensor.grad.__get__": lambda m: ((m.f(3, 4),), {}),
    "torch.Tensor.is_cpu.__get__": lambda m: ((m.f(3, 4),), {}),
    "torch.Tensor.is_sparse.__get__": lambda m: ((m.f(3, 4),), {}),
    "torch.is_complex": lambda m: ((m.f(3),), {}),
    "torch.tensor": lambda m: (([[1.0, 2.0], [3.0, 4.0]],), {"device": m.device}),
    "torch.embedding": lambda m: ((m.f(10, 4), m.i(10, 2, 3)), {}),
    "torch._foreach_add_": lambda m: ((m.fs(), m.fs()), {"alpha": 0.5}),
    "torch._foreach_add": lambda m: ((m.fs(), 1.5), {}),
    "torch._foreach_mul_": lambda m: ((m.fs(), 0.9), {}),
    "torch._foreach_addcmul_": lambda m: ((m.fs(), m.fs(), m.fs()), {"value": 0.1}),
    "torch._foreach_addcdiv_": lambda m: ((m.fs(), m.fs(), m.fs(True)), {"value": 0.1}),
    "torch._foreach_sqrt": lambda m: ((m.fs(True),), {}),
    "torch._foreach_lerp_": lambda m: ((m.fs(), m.fs(), 0.3), {}),
    "torch._foreach_div_": lambda m: ((m.fs(), m.fs(True)), {}),
    "torch._foreach_neg": lambda m: ((m.fs(),), {}),
    "torch._foreach_maximum_": lambda m: ((m.fs(), m.fs()), {}),
    "torch.Tensor.mul_": lambda m: ((m.f(3, 4), m.f(3, 4)), {}),
    "torch.Tensor.addcdiv_": lambda m: ((m.f(3, 4), m.f(3, 4), m.pos(3, 4)), {"value": 0.5}),
    "torch.Tensor.lerp_": lambda m: ((m.f(3, 4), m.f(3, 4), 0.25), {}),
    # Dropout between the layers where a reseeded generator repeats it (cuDNN keeps a state of its own).
    "torch._VF.lstm": lambda m: ((m.f(6, 3, 4), (m.f(2, 3, 5), m.f(2, 3, 5)), m.lstm(), True, 2, 0.5 * (m.device == "cpu"), m.device == "cpu", False, False), {}),
    "torch.add": lambda m: ((m.f(3, 4), m.f(3, 4)), {}),
    "torch.nonzero": lambda m: ((m.f(4, 4).relu(),), {}),
    "torch._VF._unique2": lambda m: ((m.i(4, 12), True, True, True), {}),
    "torch.masked_select": lambda m: ((m.f(3, 4), m.f(3, 4) > 0), {}),
    "torch.Tensor.resize_": lambda m: ((m.f(3, 4), 2, 3), {}),
    "torch.Tensor.requires_grad_": lambda m: ((m.f(3, 4),), {}),
    "torch.Tensor.set_": lambda m: ((m.f(3, 4), m.f(2, 6)), {}),
    "torch.linalg.svdvals": lambda m: ((m.f(6, 6),), {}),
}  # fmt: skip
# Members of the first batch with no sample here: torchvision is not a
# dependency of the tests.
_WT_NOT_COVERED = ["torchvision.ops.nms", "torch.ops.torchvision.nms"]


def _wt_bits(x):
    if isinstance(x, torch.Tensor):
        flat = x.detach().cpu().contiguous().reshape(-1)
        return (
            str(x.dtype),
            tuple(x.shape),
            x.requires_grad,
            flat.view(torch.uint8).tolist(),
        )
    if isinstance(x, (list, tuple)) and not isinstance(x, torch.Size):
        return [type(x).__name__] + [_wt_bits(v) for v in x]
    return (type(x).__name__, repr(x))


class TestVmWitness(TestCase):
    """The witness says what the dispatcher and the frame hook saw while one
    C call ran: its top-level operators in order, the arguments their schemas
    say they write (mapped to the call's operands, or to what the call made,
    or outside), whether an operator tagged as drawing, as handing out a value
    or as sizing its result by data ran at any depth, whether Python was
    entered. The rules: it changes nothing a program computes; what it cannot
    classify is opaque, and opaque is everything at once and stays; what no
    schema declares and what is no operator it does not see, and the tests
    say so: "saw_nothing" is what it saw, never what the call did.
    """

    hw_classification = HardwareClassification.ACCELERATOR

    def pair(self, device):
        gen = torch.Generator().manual_seed(0)
        return [torch.randn(3, 4, generator=gen).to(device) for _ in range(2)]

    def test_add_does_nothing(self, device):
        a, b = self.pair(device)
        r, d = _witness(torch.add, a, b)
        self.assertEqual(_wt_bits(r), _wt_bits(a + b))
        self.assertEqual(
            d,
            {"ops": ["aten::add.Tensor"], "below": 0, "wrote": [], "wrote_outside": False, "opaque": False, "why": "",
             "drew": False, "read_value": False, "read_value_top": False, "data_sized": False, "entered_python": False,
             "saw_nothing": True},
        )  # fmt: skip

    def test_a_declared_write_is_mapped_to_the_operand(self, device):
        a, b = self.pair(device)
        _, d = _witness(torch.Tensor.add_, a, b)
        self.assertEqual(
            (d["ops"], d["wrote"]), (["aten::add_.Tensor"], [(0, 0, 0, False)])
        )
        self.assertEqual((d["saw_nothing"], d["wrote_outside"]), (False, False))
        # By keyword; a member of a list; an operand written through a view
        # the kernel makes (found by its storage).
        _, d = _witness(torch.add, a, b, out=torch.empty_like(a))
        self.assertEqual(
            (d["ops"], d["wrote"]), (["aten::add.out"], [(0, 3, "out", False)])
        )
        _, d = _witness(torch._foreach_neg_, [a, b])
        self.assertEqual(
            [w[:3] for w in d["wrote"] if not w[3]], [(0, 0, (0, 0)), (0, 0, (0, 1))]
        )
        _, d = _witness(torch.Tensor.fill_diagonal_, a, 1.0)
        self.assertEqual(d["wrote"][0], (0, 0, 0, False))
        self.assertEqual({w[2] for w in d["wrote"]}, {0})
        self.assertTrue(any(w[3] for w in d["wrote"]))
        # A write to a tensor the call made itself names no operand and asks
        # for no license.
        _, d = _witness(torch.mul, a, 2)
        self.assertEqual([w[2] for w in d["wrote"]], [None] * len(d["wrote"]))
        self.assertEqual((d["saw_nothing"], d["wrote_outside"]), (True, False))

    def test_batch_norm_writes_its_statistics_unseen(self, device):
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(4, 3, 5, 5, generator=gen).to(device)
        mean, var = torch.zeros(3, device=device), torch.ones(3, device=device)
        _, d = _witness(
            torch.batch_norm, x, None, None, mean, var, True, 0.1, 1e-5, True
        )
        self.assertNotEqual(_wt_bits(mean), _wt_bits(torch.zeros(3, device=device)))
        # No schema declares that write: the measured table knows it, the witness cannot.
        self.assertEqual([w for w in d["wrote"] if w[2] is not None], [])
        self.assertEqual(
            (d["ops"], d["saw_nothing"], d["wrote_outside"]),
            (["aten::batch_norm"], True, False),
        )

    def test_draws(self, device):
        a, _ = self.pair(device)
        _, d = _witness(torch.dropout, a, 0.5, True)
        self.assertEqual(
            (d["ops"], d["drew"], d["saw_nothing"]), (["aten::dropout"], True, False)
        )
        self.assertGreater(d["below"], 0)
        # The tag is given by the operator's name, not by what a call does.
        _, d = _witness(torch.dropout, a, 0.5, False)
        self.assertEqual(
            (d["ops"], d["drew"], d["below"]), (["aten::dropout"], True, 0)
        )
        m = _WtMake(device)
        args = (
            m.f(6, 3, 4),
            (m.f(2, 3, 5), m.f(2, 3, 5)),
            m.lstm(),
            True,
            2,
            0.5,
            True,
            False,
            False,
        )
        _, d = _witness(torch._VF.lstm, *args)
        self.assertEqual(
            (d["ops"], d["drew"], d["opaque"]),
            (["aten::lstm.input"], True, False),
            d["why"],
        )
        self.assertGreater(d["below"], 0)

    def test_value_reads(self, device):
        a, b = self.pair(device)
        r, d = _witness(torch.Tensor.item, a[0, 0])
        self.assertEqual(r, a[0, 0].item())
        self.assertEqual(
            (d["ops"], d["read_value"], d["read_value_top"]),
            (["aten::item"], True, True),
        )
        # aten::_local_scalar_dense, below aten::item.
        self.assertGreaterEqual(d["below"], 1)
        for fn, args, op in (
            (torch.equal, (a, b), "aten::equal"),
            (torch.allclose, (a, b), "aten::allclose"),
            (torch.is_nonzero, (a[0, 0],), "aten::is_nonzero"),
            (torch.Tensor.__bool__, (a[0, 0],), "aten::is_nonzero"),
        ):
            _, d = _witness(fn, *args)
            self.assertEqual(
                (d["ops"], d["read_value_top"], d["saw_nothing"]), ([op], True, False)
            )
        # A read below the top-level operator (a kernel checking LAPACK's
        # status) is no value of the call's.
        _, d = _witness(torch.linalg.inv, a[:3, :3])
        self.assertEqual((d["read_value"], d["read_value_top"]), (True, False))
        # A fact of the tensor's key is no value read.
        _, d = _witness(torch.Tensor.is_same_size, a, b)
        self.assertEqual((d["read_value"], d["opaque"]), (False, False))
        # A read that is no operator is not seen.
        r, d = _witness(torch.Tensor.tolist, a)
        self.assertEqual((len(r), d["read_value"], d["saw_nothing"]), (3, False, True))

    def test_operators_in_order(self, device):
        a, _ = self.pair(device)
        m = _WtMake(device)
        w, bias = m.f(5, 4), m.f(5)
        r, d = _witness(torch.nn.functional.linear, a, w, bias)
        self.assertEqual(_wt_bits(r), _wt_bits(torch.nn.functional.linear(a, w, bias)))
        self.assertEqual((d["ops"], d["entered_python"]), (["aten::linear"], False))
        self.assertGreater(d["below"], 0)

        def three():
            return torch.nn.functional.linear(a, w, bias).relu().sum(0)

        _, d = _witness(three)
        self.assertEqual(
            d["ops"], ["aten::linear", "aten::relu", "aten::sum.dim_IntList"]
        )

    def test_python_frames(self, device):
        a, b = self.pair(device)

        class Twice(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, g):
                return g * 2

        class Sub(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return super().__torch_function__(func, types, args, kwargs)

        r, d = _witness(Twice.apply, a)
        self.assertEqual(_wt_bits(r), _wt_bits(a * 2))
        self.assertEqual((d["entered_python"], d["saw_nothing"]), (True, False))
        _, d = _witness(torch.add, a.as_subclass(Sub), b)
        self.assertEqual((d["ops"][0], d["entered_python"]), ("aten::add.Tensor", True))
        for fn, args in (
            (torch.add, (a, b)),
            (torch.Tensor.item, (a[0, 0],)),
            (len, ([a],)),
        ):
            _, d = _witness(fn, *args)
            self.assertFalse(d["entered_python"])

    def test_pause_and_resume(self, device):
        a, _ = self.pair(device)

        def body():
            torch.neg(a)
            _witness_pause(True)
            included = torch._C._dispatch_tls_is_dispatch_key_included("PreDispatch")
            torch.abs(a)
            a[0, 0].item()
            torch.dropout(a, 0.5, True)
            _witness_pause(False)
            torch.exp(a)
            return included

        included, d = _witness(body)
        self.assertFalse(included)
        self.assertEqual(
            (d["ops"], d["drew"], d["read_value"]),
            (["aten::neg", "aten::exp"], False, False),
        )

    def test_a_witness_inside_a_witness(self, device):
        a, _ = self.pair(device)
        inner = []

        class Callback(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = torch.neg(x)
                inner.append(_witness(torch.dropout, y, 0.5, True)[1])
                return torch.exp(y)

            @staticmethod
            def backward(ctx, g):
                return g

        _, d = _witness(Callback.apply, a)
        self.assertEqual(
            (inner[0]["ops"], inner[0]["drew"], inner[0]["entered_python"]),
            (["aten::dropout"], True, False),
        )
        # The one around it saw all that ran while it was open, and goes on
        # behind the inner one.
        self.assertEqual(d["ops"][:3], ["aten::neg", "aten::dropout", "aten::exp"])
        self.assertEqual(
            (d["drew"], d["entered_python"], d["below"] >= inner[0]["below"]),
            (True, True, True),
        )
        self.assertFalse(torch._C._dispatch_tls_is_dispatch_key_included("PreDispatch"))

    def test_beside_the_recorder_of_effects(self, device):
        # The hook has two users; neither turns the other off. First the
        # witness inside a call the executor prints the operators of.
        a, b = self.pair(device)
        seen = []

        def f():
            torch.neg(a)
            seen.append(_witness(torch.nn.functional.linear, a, b)[1])
            seen.append(_witness(torch.add, a, b)[1])
            torch.exp(a)

        out = io.StringIO()
        torch._rust.set_print_torch_ops(True)
        try:
            with contextlib.redirect_stdout(out):
                torch._rust.call_with_cfg_execution(f)
        finally:
            torch._rust.set_print_torch_ops(False)
        # Top-level operators only, each once, and still behind the witness.
        printed = re.findall(r"torch op: ([\w:.]+)\(", out.getvalue())
        self.assertEqual(
            printed, ["aten::neg", "aten::linear", "aten::add.Tensor", "aten::exp"]
        )
        self.assertEqual(
            [d["ops"] for d in seen], [["aten::linear"], ["aten::add.Tensor"]]
        )
        self.assertGreater(seen[0]["below"], 0)

        # Then the executor, recording effects, inside the witness: when its
        # recorder ends, the witness still sees.
        def g():
            return torch.neg(a).abs()

        def body():
            torch._rust.call_with_cfg_execution(g)
            included = torch._C._dispatch_tls_is_dispatch_key_included("PreDispatch")
            torch.exp(a)
            return included

        included, d = _witness(body)
        self.assertTrue(included)
        self.assertEqual(d["ops"], ["aten::neg", "aten::abs", "aten::exp"])
        self.assertFalse(d["opaque"], d["why"])
        self.assertFalse(torch._C._dispatch_tls_is_dispatch_key_included("PreDispatch"))

    def test_what_a_program_can_see(self, device):
        # No mode, no fx tracing, nn.Module.__call__ untouched. The one thing
        # that differs: the PreDispatch key is in the thread's include set.
        call = torch.nn.Module.__call__

        def body():
            return (
                torch._C._len_torch_dispatch_stack(),
                torch._C._len_torch_function_stack(),
                torch.fx._symbolic_trace.is_fx_tracing(),
                torch.nn.Module.__call__ is call,
                torch._C._dispatch_tls_is_dispatch_key_included("PreDispatch"),
            )

        self.assertEqual(body(), (0, 0, False, True, False))
        self.assertEqual(_witness(body)[0], (0, 0, False, True, True))
        self.assertEqual(body(), (0, 0, False, True, False))
        for fn in (
            torch._C._len_torch_dispatch_stack,
            torch._C._len_torch_function_stack,
        ):
            r, d = _witness(fn)
            self.assertEqual((r, d["entered_python"]), (0, False))

    def test_what_is_not_classified_is_everything(self, device):
        a, b = self.pair(device)
        everything = {
            "opaque": True,
            "drew": True,
            "read_value": True,
            "read_value_top": True,
            "data_sized": True,
            "entered_python": True,
            "wrote_outside": True,
            "saw_nothing": False,
        }
        # An operator outside aten: its tags say nothing.
        enter = torch._C._jit_get_operation("profiler::_record_function_enter_new")[0]
        _, d = _witness(enter, "x", None)
        self.assertEqual({k: d[k] for k in everything}, everything)
        self.assertIn("profiler::_record_function_enter_new", d["why"])
        # A result that is no tensor, from an operator on neither list.
        _, d = _witness(
            torch.Tensor.is_coalesced,
            torch.sparse_coo_tensor(size=(2, 2), device=device),
        )
        self.assertEqual({k: d[k] for k in everything}, everything)
        self.assertIn("aten::is_coalesced", d["why"])
        # A witness that cannot see: the key is excluded on this thread.
        with torch._C._DisableTorchDispatch():
            _, d = _witness(torch.add, a, b)
        self.assertEqual(
            {k: d[k] for k in everything} | {"ops": d["ops"]}, everything | {"ops": []}
        )
        # One the program blinds while it is open.
        _, d = _witness(
            torch._C._dispatch_tls_set_dispatch_key_included, "PreDispatch", False
        )
        self.assertEqual({k: d[k] for k in everything}, everything)
        self.assertFalse(torch._C._dispatch_tls_is_dispatch_key_included("PreDispatch"))

    def test_every_operator_with_a_result_that_is_no_tensor_is_classified(self, device):
        # Over the dispatcher's whole registry: such an operator carries the
        # tag, or is on one of the witness's two lists, or is opaque. The
        # opaque ones are recorded here, so that a new one is seen.
        reads, no_data = _witness_names()
        registered = set(torch._C._dispatch_get_all_op_names())
        found = collections.defaultdict(set)
        for schema in torch._C._jit_get_all_schemas():
            full = schema.name + (
                "." + schema.overload_name if schema.overload_name else ""
            )
            tensors = all(
                re.fullmatch(r"(Optional\[)?(List\[)?Tensor\]*", str(r.type))
                for r in schema.returns
            )
            if (
                tensors
                or not schema.name.startswith("aten::")
                or full not in registered
            ):
                continue
            overload = getattr(
                getattr(torch.ops.aten, schema.name[6:]),
                schema.overload_name or "default",
            )
            if torch.Tag.data_dependent_output in overload.tags:
                found["tag"].add(schema.name)
            elif schema.name in reads:
                found["reads"].add(schema.name)
            elif schema.name in no_data:
                found["no data"].add(schema.name)
            else:
                found["opaque"].add(schema.name[6:])
        self.assertEqual(
            sorted(found["tag"]),
            [
                "aten::_local_scalar_dense",
                "aten::allclose",
                "aten::equal",
                "aten::item",
            ],
        )
        self.assertEqual(
            (sorted(found["reads"]), sorted(found["no data"])),
            (sorted(reads), sorted(no_data)),
        )
        self.assertEqual(
            sorted(found["opaque"]),
            ["_cslt_sparse_mm_search", "_cudnn_attention_forward", "_cufft_get_plan_cache_max_size", "_cufft_get_plan_cache_size",
             "_debug_has_internal_overlap", "_dimI", "_dimV", "_efficient_attention_forward", "_fused_sdp_choice",
             "_has_compatible_shallow_copy_type", "_has_same_storage_numel", "_nested_get_ragged_idx",
             "_nested_tensor_from_mask_left_aligned", "_nnpack_available", "_nnz", "_scaled_dot_product_cudnn_attention",
             "_scaled_dot_product_flash_attention", "_scaled_dot_product_fused_attention_overrideable",
             "_use_cudnn_rnn_flatten_weight", "_use_miopen_ctc_loss", "_version", "dense_dim", "fbgemm_linear_quantize_weight",
             "get_gradients", "is_coalesced", "is_distributed", "is_pinned", "is_set_to", "is_vulkan_available", "output_nr",
             "q_per_channel_axis", "q_scale", "q_zero_point", "qscheme", "retains_grad", "sparse_dim"],
        )  # fmt: skip

    def test_no_member_computes_anything_else_under_the_witness(self, device):
        first_batch = sorted(_WT_SAMPLES) + _WT_NOT_COVERED
        self.assertEqual((len(first_batch), len(set(first_batch))), (59, 59))
        for name, make in _WT_SAMPLES.items():
            fn = functools.reduce(getattr, name.split(".")[1:], torch)
            with self.subTest(member=name):
                results = []
                for under in (False, True):
                    args, kwargs = make(_WtMake(device))
                    torch.manual_seed(0)
                    r = (
                        _witness(fn, *args, **kwargs)
                        if under
                        else (fn(*args, **kwargs), None)
                    )
                    results.append(
                        (_wt_bits(r[0]), _wt_bits([args, list(kwargs.values())]), r[1])
                    )
                self.assertEqual(results[0][:2], results[1][:2])
                self.assertFalse(results[1][2]["opaque"], results[1][2]["why"])

    def test_the_programs_a_dispatch_mode_changes(self, device):
        # Three callees whose kernels take another path when a dispatch mode
        # is on, or when fx tracing is: under the witness they take eager's.
        def both(fn, *args, **kwargs):
            plain = fn(*args, **kwargs)
            under, d = _witness(fn, *args, **kwargs)
            self.assertEqual(_wt_bits(plain), _wt_bits(under))
            return d

        gen = torch.Generator().manual_seed(0)
        for _ in range(4):
            both(torch.linalg.svdvals, torch.randn(9, 9, generator=gen).to(device))
        torch.manual_seed(0)
        mha = torch.nn.MultiheadAttention(16, 4, batch_first=True).eval().to(device)
        x = torch.randn(2, 5, 16, generator=gen).to(device)
        with torch.no_grad():
            d = both(mha, x, x, x, need_weights=False)
            self.assertIn("aten::_native_multi_head_attention", d["ops"])
            d = both(
                torch.nn.functional.multi_head_attention_forward, x, x, x, 16, 4, mha.in_proj_weight, mha.in_proj_bias,
                None, None, False, 0.0, mha.out_proj.weight, mha.out_proj.bias, training=False, need_weights=False,
            )  # fmt: skip
            self.assertNotIn("aten::_native_multi_head_attention", d["ops"])
        lstm = torch.nn.LSTM(4, 6).to(device)
        data, zeros = (
            torch.randn(8, 4, generator=gen).to(device),
            torch.zeros(1, 4, 6, device=device),
        )
        for sizes in ([4, 2, 2], [4, 3, 1]):
            both(
                torch._VF.lstm,
                data,
                torch.tensor(sizes),
                (zeros, zeros),
                lstm._flat_weights,
                True,
                1,
                0.0,
                False,
                False,
            )

    def test_the_object_a_builtin_is_bound_to_is_an_operand(self, device):
        a, b = self.pair(device)
        want = a + b
        # `a.add_` holds `a`: the call's first operand, written.
        for bound in (a.add_, a.__iadd__):
            before = a.clone()
            r, d = _witness(bound, b)
            self.assertIs(r, a)
            self.assertEqual(_wt_bits(a), _wt_bits(before + b))
            self.assertEqual(
                (d["ops"], d["wrote"], d["wrote_outside"], d["saw_nothing"]),
                (["aten::add_.Tensor"], [(0, 0, 0, False)], False, False),
            )
        self.assertEqual(_wt_bits(a), _wt_bits(want + b))
        # A function of a module is bound to the module, a static one to
        # nothing: the positions are the arguments'.
        for fn in (torch._C._nn.elu_, torch.sigmoid_):
            _, d = _witness(fn, b)
            self.assertEqual(
                (d["wrote"], d["wrote_outside"]), ([(0, 0, 0, False)], False)
            )
        # What another kind of callable holds is no operand: a write to it is
        # a write outside, which nothing licenses.
        r, d = _witness(a.__setitem__, 0, 3.0)
        self.assertEqual(_wt_bits(a[0]), _wt_bits(torch.full_like(a[0], 3.0)))
        self.assertEqual(
            (d["wrote"], d["wrote_outside"], d["opaque"], d["saw_nothing"]),
            ([(1, 0, None, False)], True, False, False),
        )

    def test_a_write_to_what_the_call_made_needs_no_license(self, device):
        a, b = self.pair(device)
        m = _WtMake(device)
        # linear over a 4-d input adds the bias in place into the product it
        # made: a nested operator writing a temporary.
        x, weight, bias = m.f(2, 2, 3, 4), m.f(5, 4), m.f(5)
        r, d = _witness(torch._C._nn.linear, x, weight, bias)
        self.assertEqual(_wt_bits(r), _wt_bits(torch._C._nn.linear(x, weight, bias)))
        on_cpu = torch.device(device).type == "cpu"
        self.assertEqual(
            {(e[2], e[3]) for e in d["wrote"]}, {(None, True)} if on_cpu else set()
        )
        self.assertEqual(
            (d["ops"], d["wrote_outside"], d["saw_nothing"]),
            (["aten::linear"], False, True),
        )
        # A reduction over integers fills its result through operators of its
        # own before it returns it: known to be made only when it returns.
        mask = a > 0
        r, d = _witness(torch.sum, mask)
        self.assertEqual(_wt_bits(r), _wt_bits(torch.sum(mask)))
        self.assertIn((0, 0, None, True), d["wrote"])
        self.assertEqual(
            (d["ops"], d["wrote_outside"], d["saw_nothing"]),
            (["aten::sum"], False, True),
        )

        # Made by storage: a view of what the call made is what the call made.
        def body():
            t = torch.zeros(3, 4, device=device)
            t.view(-1).add_(b.view(-1))
            inner.append(_witness(lambda: t.add_(b))[1])
            return t

        inner = []
        r, d = _witness(body)
        self.assertEqual(_wt_bits(r), _wt_bits(torch.zeros_like(b) + b + b))
        self.assertEqual(
            ([e[2] for e in d["wrote"]], d["wrote_outside"]), ([None] * 3, False)
        )
        # For a witness opened later, what was made before it is outside.
        self.assertEqual(
            (inner[0]["wrote"], inner[0]["wrote_outside"]),
            ([(0, 0, None, False)], True),
        )

    def test_a_write_outside_is_never_licensed(self, device):
        a, b = self.pair(device)
        # The autograd engine adds into a gradient that is there: a tensor no
        # operand leads to. Through Tensor.backward, and at the engine's own
        # C entry, where no frame is entered and nothing else is seen.
        # KNOWN HOLE, asserted so that it is seen when it closes: the engine
        # runs a graph of another device than the CPU on a thread of its own,
        # where no witness is. Of such a backward nothing is seen.
        seen = torch.device(device).type == "cpu"
        leaf = a.clone().requires_grad_()
        leaf.grad = torch.ones_like(a)
        _, d = _witness(torch.Tensor.backward, (leaf * 2).sum())
        self.assertEqual(_wt_bits(leaf.grad), _wt_bits(torch.full_like(a, 3.0)))
        self.assertEqual((d["wrote_outside"], d["saw_nothing"]), (seen, False))
        run = torch.autograd.Variable._execution_engine.run_backward
        loss = (leaf * 2).sum()
        ones = (torch.ones_like(loss),)
        _, d = _witness(
            run, (loss,), ones, False, False, (), allow_unreachable=True, accumulate_grad=True
        )  # fmt: skip
        self.assertEqual(_wt_bits(leaf.grad), _wt_bits(torch.full_like(a, 5.0)))
        added = [(d["ops"].index("aten::add_.Tensor"), 0, None, False)] if seen else []
        self.assertEqual([e for e in d["wrote"] if not e[3]], added)
        self.assertEqual(
            (d["entered_python"], d["opaque"], d["wrote_outside"], d["saw_nothing"]),
            (False, False, seen, not seen),
        )
        # NOT SEEN: the first gradient is attached, not added. No operator
        # writes, so the witness has nothing to say about it.
        leaf.grad = None
        loss = (leaf * 2).sum()
        _, d = _witness(
            run, (loss,), ones, False, False, (), allow_unreachable=True, accumulate_grad=True
        )  # fmt: skip
        self.assertEqual(_wt_bits(leaf.grad), _wt_bits(torch.full_like(a, 2.0)))
        self.assertEqual((d["wrote_outside"], d["saw_nothing"]), (False, True))

        # A view, made inside the call, of a tensor that a closure leads to.
        outside = torch.zeros(3, 4, device=device)

        class WritesOutside(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                outside.view(-1).add_(1)
                return x * 2

            @staticmethod
            def backward(ctx, g):
                return g * 2

        _, d = _witness(WritesOutside.apply, a)
        self.assertEqual(_wt_bits(outside), _wt_bits(torch.ones_like(a)))
        self.assertIn((d["ops"].index("aten::add_.Tensor"), 0, None, False), d["wrote"])
        self.assertEqual((d["wrote_outside"], d["saw_nothing"]), (True, False))
        # The same of an operand is a write to that operand.
        _, d = _witness(torch.Tensor.__setitem__, a, 0, 3.0)
        self.assertEqual(
            (d["ops"], d["wrote"], d["wrote_outside"], d["saw_nothing"]),
            (
                ["aten::select.int", "aten::fill_.Scalar"],
                [(1, 0, 0, False)],
                False,
                False,
            ),
        )

        # A tensor inside a container the caller did not open is no operand.
        def zero(box):
            box["t"].zero_()

        _, d = _witness(zero, {"t": b})
        self.assertEqual(
            (d["wrote"], d["wrote_outside"]), ([(0, 0, None, False)], True)
        )

    def test_state_an_operator_hands_out_is_not_what_it_made(self, device):
        # "Made" is a new tensor on a storage none of the operator's arguments
        # holds. Three ways an operator returns memory that was there before,
        # each written afterwards: outside, and the witness can tell (it is
        # not opaque).
        values = torch.ones(2, device=device)
        index = torch.tensor([[0, 1]], device=device)
        sparse = torch.sparse_coo_tensor(index, values, (3,), check_invariants=False)
        scale = torch.tensor([0.1, 0.2], dtype=torch.float64)
        quantized = torch.quantize_per_channel(
            torch.ones(2, 2), scale, torch.zeros(2, dtype=torch.long), 0, torch.quint8
        )
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "mapped")
            with open(path, "wb") as f:
                f.write(bytes(16))

            # The parts of a sparse tensor (it names no storage), a tensor a
            # quantizer keeps (somebody else holds it), a file's memory.
            def parts():
                sparse._values().add_(1)

            def kept():
                quantized.q_per_channel_scales().add_(1)

            def mapped():
                torch.from_file(path, shared=True, size=4).add_(1)

            for body in (parts, kept, mapped):
                _, d = _witness(body)
                self.assertEqual(
                    (d["ops"][1:], d["opaque"], d["wrote_outside"]),
                    (["aten::add_.Tensor"], False, True),
                    body.__name__,
                )
            with open(path, "rb") as f:
                self.assertEqual(f.read(), struct.pack("4f", 1.0, 1.0, 1.0, 1.0))
        self.assertEqual(_wt_bits(sparse._values()), _wt_bits(values))
        self.assertEqual(_wt_bits(values), _wt_bits(torch.full_like(values, 2.0)))
        self.assertEqual(quantized.q_per_channel_scales().tolist(), [1.1, 1.2])

    def test_a_result_sized_by_data(self, device):
        a, b = self.pair(device)
        index = torch.tensor([0, 2], device=device)
        for fn, args in (
            (torch.nonzero, (a,)),
            (torch.masked_select, (a, a > 0)),
            (torch.Tensor.__getitem__, (a, a > 0)),
            (torch.bincount, (index,)),
            # Below the top-level operator: where(condition) runs nonzero.
            (torch.where, (a > 0,)),
            # The tag is the operator's, as the draw tag: also an index by
            # integers, whose size follows no data.
            (torch.Tensor.__getitem__, (a, index)),
        ):
            _, d = _witness(fn, *args)
            self.assertEqual(
                (d["data_sized"], d["saw_nothing"], d["opaque"]),
                (True, False, False),
                fn,
            )
        _, d = _witness(torch.Tensor.__getitem__, a, 0)
        self.assertEqual((d["data_sized"], d["saw_nothing"]), (False, True))

    def test_a_witness_blinded_in_the_middle_stays_opaque(self, device):
        a, b = self.pair(device)
        set_key = torch._C._dispatch_tls_set_dispatch_key_included
        is_key = torch._C._dispatch_tls_is_dispatch_key_included
        inner = []

        # No Python frame is entered between the clearing and what finds it.
        def a_witness_inside():
            set_key("PreDispatch", False)
            a.add_(b)
            inner.append(_witness(torch.neg, a)[1])

        def a_pause():
            set_key("PreDispatch", False)
            a.add_(b)
            _witness_pause(True)
            _witness_pause(False)

        def nothing():
            pass

        def a_frame():
            set_key("PreDispatch", False)
            a.add_(b)
            nothing()
            set_key("PreDispatch", True)

        # torch/_ops.py clears the key when its last pre-dispatch mode goes.
        def a_tracer():
            from torch.fx.experimental.proxy_tensor import make_fx

            make_fx(torch.neg, pre_dispatch=True)(b)
            inner.append(is_key("PreDispatch"))
            a.add_(b)
            set_key("PreDispatch", True)

        for body, at in (
            (a_witness_inside, "a begin"),
            (a_pause, "a pause"),
            (a_frame, "a Python frame"),
            (a_tracer, "a Python frame"),
        ):
            before = a.clone()
            _, d = _witness(body)
            self.assertEqual(_wt_bits(a), _wt_bits(before + b))
            self.assertEqual(
                (
                    d["opaque"],
                    d["why"],
                    d["saw_nothing"],
                    "aten::add_.Tensor" in d["ops"],
                ),
                (True, "could not see at " + at, False, False),
                body.__name__,
            )
            self.assertFalse(is_key("PreDispatch"))
        # The witness inside saw what ran while it was open; the tracer left
        # the key cleared.
        self.assertEqual(
            (inner[0]["ops"], inner[0]["opaque"], inner[1]),
            (["aten::neg"], False, False),
        )

        # A handler of a dispatch mode runs with the key excluded: what it
        # runs is not seen, and the frame it enters says so.
        class Writes(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                a.add_(b)
                return func(*args, **(kwargs or {}))

        def under_a_mode():
            with Writes():
                torch.neg(b)

        before = a.clone()
        _, d = _witness(under_a_mode)
        self.assertEqual(_wt_bits(a), _wt_bits(before + b))
        self.assertEqual(
            (d["ops"], d["opaque"], d["why"]),
            (["aten::neg"], True, "could not see at a Python frame"),
        )

    def test_known_hole_the_key_cleared_and_set_again_unseen(self, device):
        # KNOWN HOLE, asserted so that it is seen when it closes. The key is
        # taken away and put back with no call of the witness and no Python
        # frame entered in between: the operators in between are not seen and
        # nothing says so. Seeing it needs a change to c10 or to the binding.
        a, b = self.pair(device)
        set_key = torch._C._dispatch_tls_set_dispatch_key_included
        hole = ([], [], False, False, True)

        def what(d):
            return (
                d["ops"],
                d["wrote"],
                d["opaque"],
                d["wrote_outside"],
                d["saw_nothing"],
            )

        # Inside one C call: a call list of three C calls.
        set_entry = ("vector", set_key, "set")
        add_entry = ("vector", torch.Tensor.add_, "add_")
        T = torch.Tensor
        both = _calls_make((2, 0, ["PreDispatch", False, True], [
            (set_entry, [("const", 0), ("const", 1)], None, None, 0, (), (), (str, bool)),
            (add_entry, [("in", 0), ("in", 1)], None, None, 0, (), (), (T, T)),
            (set_entry, [("const", 0), ("const", 2)], None, None, 0, (), (), (str, bool)),
        ], []))  # fmt: skip
        before = a.clone()
        _, d = _witness(_calls_run, both, (a, b), ())
        self.assertEqual(_wt_bits(a), _wt_bits(before + b))
        self.assertEqual(what(d), hole)

        # C calls of one frame: the key excluded and included again.
        def guarded():
            with torch._C._DisableTorchDispatch():
                a.add_(b)

        before = a.clone()
        _, d = _witness(guarded)
        self.assertEqual(_wt_bits(a), _wt_bits(before + b))
        self.assertEqual(what(d)[:4], hole[:4])
        self.assertFalse(torch._C._dispatch_tls_is_dispatch_key_included("PreDispatch"))

    def test_a_resume_inside_an_operator_in_flight(self, device):
        # An operator with a kernel in Python, whose operators come to the
        # hook below it. A witness paused when that operator began and resumed
        # inside it did not see it begin: what runs below is no part of the
        # last operator the witness saw, and is not top-level.
        a, _ = self.pair(device)

        def kernel(x):
            _witness_pause(False)
            return torch.neg(x).add_(1)

        def body(first):
            if first:
                torch.exp(a)
            _witness_pause(True)
            return torch.ops.vmwitness.resumes(a)

        with torch.library._scoped_library("vmwitness", "FRAGMENT") as lib:
            lib.define("resumes(Tensor x) -> Tensor")
            lib.impl("resumes", kernel, "CompositeExplicitAutograd")
            for first in (False, True):
                # Around it a witness that is not paused: the operator comes
                # to the hook.
                (r, d), _ = _witness(_witness, body, first)
                self.assertEqual(_wt_bits(r), _wt_bits(torch.neg(a) + 1))
                self.assertEqual(
                    (d["ops"], d["opaque"], d["why"]),
                    (["aten::exp"] * first, True, "resumed inside an operator that began while it was paused"),
                )  # fmt: skip

        # Paused and resumed with no operator in flight: nothing to refuse.
        def between():
            _witness_pause(True)
            torch.neg(a)
            _witness_pause(False)
            torch.exp(a)

        _, d = _witness(between)
        self.assertEqual((d["ops"], d["opaque"]), (["aten::exp"], False))


instantiate_device_type_tests(TestVmWitness, globals())


# The families of the rule table that are not written yet: an instruction
# kind that is "later" names one of them, and the code that decided before
# the table decides until that family is there.
RULES_LATER = (
    "torch",
    "builtins",
    "descriptors",
    "classes",
    "attributes",
    "subscripts",
    "iteration",
    "functions",
    "stores",
    "recorder",
)
# The families that are written.
RULES_FAMILIES = ("ops", "imports")
RULES_COLUMNS = {
    "reads": (
        "Slot",
        "Constant",
        "Global",
        "Cell",
        "Closure",
        "Identity",
        "Class",
        "Text",
        "Handled",
        "Pending",
        "Modules",
        "Attribute",
    ),
    "writes": r"None|Slot|Handled|Fills\(\d+\)",
    "result": r"None|Lit|Entry|Slot|Part|Made|Pinned|Return|Operand\(\d+\)",
    "enters": r"Never|Named|Foreign",
    "scalars": r"None|Operand|Parts|Specialize",
}
# What the executor runs without asking anything (its inline path).
RULES_INLINE = ("Const", "LoadLocal", "StoreLocal")
# An instruction kind that the table stops, by the opcode that becomes it.
RULES_LOWERED = {"LoadFromDictOrDeref": "LOAD_FROM_DICT_OR_DEREF"}

RULES_K = "K = [2.0]"
RULES_K_CHANGE = "K[0] = 5.0 if second else 2.0"
RULES_MANAGER = """class Manager:
    def __enter__(self):
        return self.k
    def __exit__(self, *exc):
        return True
CM = Manager()
CM.k = 2.0"""
RULES_MANAGER_CHANGE = "CM.k = 5.0 if second else 2.0"
RULES_DRIVE = """try:
    outer().send(None)
except StopIteration as done:
    k = done.value"""
RULES_ASYNC_MANAGER = """class Manager:
    async def __aenter__(self):
        return self.k
    async def __aexit__(self, *exc):
        return False
CM = Manager()
CM.k = 2.0
async def outer():
    async with CM as v:
        return v"""
RULES_ASYNC_ITEMS = """class Items:
    def __aiter__(self):
        self.left = 1
        return self
    async def __anext__(self):
        if not self.left:
            raise StopAsyncIteration
        self.left = 0
        return K[0]
async def outer():
    async for v in Items():
        got = v
    return got"""
RULES_MATCH = "match {}:\n    case {}:\n        k = {}\n    case _:\n        k = 1.0"

# A program per stop of the rule table that Python source can make: (the
# stops it holds, what the module has, the construct, which sets `k` in a
# function with tensor work around it, what changes between calls, the state
# the calls leave).
RULES_STOPS = {
    "match_class": (
        ("MATCH_CLASS",),
        "class Box:\n    k = 2.0\nBOX = Box()",
        RULES_MATCH.format("BOX", "Box(k=v)", "v"),
        "BOX.k = 5.0 if second else 2.0",
        "None",
    ),
    "match_class_positional": (
        ("MATCH_CLASS",),
        "import dataclasses\n@dataclasses.dataclass\n"
        "class DCls:\n    k: float = 2.0\nDC = DCls()",
        RULES_MATCH.format("DC", "DCls(v)", "v"),
        "DC.k = 5.0 if second else 2.0",
        "None",
    ),
    "match_mapping": (
        ("MATCH_MAPPING", "MATCH_KEYS"),
        "D = {'k': 2.0}",
        RULES_MATCH.format("D", "{'k': v}", "v"),
        "D['k'] = 5.0 if second else 2.0",
        "None",
    ),
    "match_sequence": (
        ("MATCH_SEQUENCE", "GET_LEN"),
        "S = [2.0, 1.0]",
        RULES_MATCH.format("S", "[a, b]", "a * b"),
        "S[0] = 5.0 if second else 2.0",
        "None",
    ),
    "store_global": (
        ("STORE_GLOBAL",),
        "G = 2.0\nSTEP = [1.0]",
        "global G\nG = G + STEP[0]\nk = G",
        "STEP[0] = 3.0 if second else 1.0",
        "G",
    ),
    "delete_global": (
        ("DELETE_GLOBAL",),
        "T = 2.0",
        "global T\nk = T\ndel T",
        "globals()['T'] = 5.0 if second else 2.0",
        "'T' in globals()",
    ),
    "with": (
        ("BEFORE_WITH",),
        RULES_MANAGER,
        "with CM as k:\n    pass",
        RULES_MANAGER_CHANGE,
        "None",
    ),
    "with_raises": (
        ("WITH_EXCEPT_START",),
        RULES_MANAGER,
        "with CM as k:\n    raise KeyError",
        RULES_MANAGER_CHANGE,
        "None",
    ),
    "class_statement": (
        ("LOAD_BUILD_CLASS", "LOAD_NAME", "STORE_NAME"),
        RULES_K,
        "class Local:\n    k = K[0]\nk = Local.k",
        RULES_K_CHANGE,
        "None",
    ),
    "class_deletes_a_name": (
        ("DELETE_NAME",),
        RULES_K,
        "class Local:\n    tmp = K[0]\n    k = tmp\n    del tmp\nk = Local.k",
        RULES_K_CHANGE,
        "None",
    ),
    "class_annotations": (
        ("SETUP_ANNOTATIONS",),
        RULES_K,
        "class Local:\n    k: float = K[0]\nk = Local.k",
        RULES_K_CHANGE,
        "None",
    ),
    "class_reads_a_local": (
        ("LOAD_LOCALS", "LoadFromDictOrDeref"),
        RULES_K,
        "base = K[0]\nclass Local:\n    k = base\nk = Local.k",
        RULES_K_CHANGE,
        "None",
    ),
    # CLEANUP_THROW is in the handler of every await: the program holds it,
    # and only a throw() into the coroutine would run it.
    "await": (
        ("GET_AWAITABLE", "CLEANUP_THROW"),
        RULES_K
        + "\nasync def inner():\n    return K[0]\n"
        + "async def outer():\n    return await inner()",
        RULES_DRIVE,
        RULES_K_CHANGE,
        "None",
    ),
    "yield_from": (
        ("GET_YIELD_FROM_ITER",),
        RULES_K
        + "\ndef inner():\n    yield K[0]\ndef outer():\n    yield from inner()",
        "k = next(outer())",
        RULES_K_CHANGE,
        "None",
    ),
    "async_with": (
        ("BEFORE_ASYNC_WITH",),
        RULES_ASYNC_MANAGER,
        RULES_DRIVE,
        RULES_MANAGER_CHANGE,
        "None",
    ),
    "async_for": (
        ("GET_AITER", "GET_ANEXT", "END_ASYNC_FOR"),
        RULES_K + "\n" + RULES_ASYNC_ITEMS,
        RULES_DRIVE,
        RULES_K_CHANGE,
        "None",
    ),
    "except_star": (
        ("CHECK_EG_MATCH", "CALL_INTRINSIC_2"),
        RULES_K,
        "try:\n    raise ExceptionGroup('g', [ValueError(K[0])])\n"
        "except* ValueError as group:\n    k = group.exceptions[0].args[0]",
        RULES_K_CHANGE,
        "None",
    ),
    "failed_assert": (
        ("LOAD_ASSERTION_ERROR",),
        RULES_K,
        "try:\n    assert K[0] < 0.0\nexcept AssertionError:\n    k = K[0]",
        RULES_K_CHANGE,
        "None",
    ),
    "type_parameter": (
        ("CALL_INTRINSIC_1", "CALL_INTRINSIC_2"),
        RULES_K,
        "def same[T](a: T) -> T:\n    return a\nk = same(K[0])",
        RULES_K_CHANGE,
        "None",
    ),
}
# The stops that have no program, and why.
RULES_NOT_WRITTEN = {
    "LOAD_FROM_DICT_OR_GLOBALS": "only under a type parameter in a class body",
    "RESERVED": "no source compiles to it",
    "INTERPRETER_EXIT": "no source compiles to it",
}


def rules_source(name, programs=RULES_STOPS):
    _, setup, construct, change, state = programs[name][:5]
    body = "\n".join(f"    {line}" for line in construct.splitlines())
    change = "\n".join(f"    {line}" for line in change.splitlines())
    return (
        f"import torch\n{setup}\n"
        f"def f(x):\n    y = torch.sin(x) * 2.0 + torch.cos(x)\n{body}\n"
        "    return (y * k + torch.tanh(y)).sum(-1)\n"
        f"def change(second):\n{change}\n"
        f"def state():\n    return {state}\n"
    )


def rules_opnames(code):
    """The opcodes of a code object and of every code object in it."""
    names = {i.opname for i in dis.get_instructions(code)}
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= rules_opnames(const)
    return names


@instantiate_parametrized_tests
@unittest.skipUnless(CHECKS_HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmRulesOps(TestCase):
    """The rule table has no row for "anything else". Every kind of
    instruction of the executor and every opcode of CPython is a row (what it
    does is a fact of its kind), an opcode that becomes such kinds, a stop
    with its reason, or names the family of rules that will take it; an
    opcode in none of these is refused. What the table stops runs under no
    trace: a program that uses it computes what eager computes whatever
    changes between its calls, and no region replaces the construct."""

    hw_classification = HardwareClassification.GENERIC

    def statuses(self):
        table = collections.defaultdict(list)
        for name, status in torch._rust._rules_kinds():
            table[name].append(status)
        return table

    def listed(self, table, name, lowered=True):
        """Whether the table has an answer for `name` that is not a default."""
        for status in table[name]:
            how, _, rest = status.partition(":")
            if how in ("row", "stop"):
                return True
            if how == "as" and lowered:
                return all(
                    self.listed(table, k, lowered=False) for k in rest.split(",")
                )
        later = [s.partition(":")[2] for s in table[name] if s.startswith("later:")]
        return bool(later) and all(family in RULES_LATER for family in later)

    def test_every_kind_and_opcode_is_listed(self):
        table = self.statuses()
        unlisted = sorted(n for n in table if not self.listed(table, n))
        # The numbers that are no opcode of the executor's are refused.
        self.assertTrue(all(re.fullmatch(r"<\d+>", n) for n in unlisted), unlisted)
        for name in unlisted:
            self.assertEqual(table[name], ["unlisted"])
        for name, number in opcode.opmap.items():
            with self.subTest(opcode=name):
                if number >= opcode.MIN_PSEUDO_OPCODE:
                    # The compiler's own: no code object holds one.
                    continue
                if name not in table:
                    # Put in place by sys.monitoring: the executor does not
                    # know them, and the table refuses the number.
                    self.assertTrue(name.startswith("INSTRUMENTED_"))
                    self.assertEqual(table[f"<{number}>"], ["unlisted"])
                    continue
                self.assertTrue(self.listed(table, name), table[name])
        for kind in RULES_INLINE:
            self.assertEqual(table[kind], ["row"])

    def test_every_op_of_the_executor_is_a_kind(self):
        path = os.path.join(CRATE, "cfg", "ir.rs")
        if not os.path.exists(path):
            self.skipTest("the executor's source is not beside the tests")
        with open(path) as f:
            text = re.sub(r"//[^\n]*", "", f.read())
        body = re.search(r"pub enum Op \{(.*?)\n\}", text, re.DOTALL).group(1)
        ops = set(re.findall(r"^    ([A-Z]\w*)", body, re.MULTILINE)) - {"Bytecode"}
        table = self.statuses()
        # An opcode's name is in capitals, a number's in brackets.
        kinds = {n for n in table if not n.isupper() and not n.startswith("<")}
        self.assertEqual(kinds, ops)
        self.assertGreater(len(ops), 30)
        for op in ops:
            self.assertTrue(self.listed(table, op), op)

    def test_the_dump_is_regular(self):
        fields = {"row": 10, "stop": 4, "later": 4, "as": 4, "group": 4}
        table = self.statuses()
        seen = collections.Counter()
        dumped = set()
        for line in torch._rust._rules_dump().splitlines():
            parts = line.split("\t")
            with self.subTest(line=line):
                self.assertIn(parts[0], fields)
                self.assertEqual(len(parts), fields[parts[0]])
                self.assertTrue(all(parts))
                seen[parts[0]] += 1
                if parts[0] == "group":
                    continue
                how, family, what, *rest = parts
                self.assertIn(family, RULES_FAMILIES)
                dumped.add(
                    (what, how if how in ("row", "stop") else f"{how}:{rest[0]}")
                )
                if how == "later":
                    self.assertIn(rest[0], (*RULES_FAMILIES, *RULES_LATER))
                if how in ("row", "stop"):
                    # One sentence, or one reason.
                    self.assertGreater(len(rest[-1]), 20)
                if how != "row":
                    continue
                operands, reads, writes, result, enters, scalars, why = rest
                self.assertRegex(operands, r"^(-|(-?\d+:\w+)(,-?\d+:\w+)*)$")
                for read in reads.split(","):
                    self.assertIn(read, ("-", *RULES_COLUMNS["reads"]))
                for column, value in zip(
                    ("writes", "result", "enters", "scalars"), rest[2:6]
                ):
                    self.assertRegex(value, f"^({RULES_COLUMNS[column]})$")
                self.assertTrue(why.endswith("."), why)
        # The dump is the table: every answer is a line, and no line more.
        answers = {(n, s) for n, ss in table.items() for s in ss if s != "unlisted"}
        self.assertEqual(dumped, answers)
        self.assertGreater(seen["stop"], 30)
        self.assertGreater(seen["row"], 30)
        self.assertGreater(seen["group"], 6)

    def test_a_group_is_a_list_of_exact_classes(self):
        class Int(int):
            pass

        class Tensor(torch.Tensor):
            pass

        groups = torch._rust._rules_groups
        for value in (1, 1.5, True, "a", b"a", None, ...):
            self.assertEqual(groups(value), ["Immutable"], msg=repr(value))
        self.assertEqual(groups([]), ["ExactList"])
        self.assertEqual(groups(()), ["ExactTuple"])
        self.assertEqual(groups({}), ["ExactDict"])
        self.assertEqual(groups(set()), ["ExactSet"])
        self.assertEqual(groups(torch.ones(1)), ["PlainTensor"])
        self.assertEqual(groups(torch.nn.Parameter(torch.ones(1))), ["PlainTensor"])
        # A subclass of a member is no member, nor is any other class.
        Point = collections.namedtuple("Point", "x")
        others = (Int(1), Tensor(), Point(1), collections.OrderedDict(), frozenset())
        for value in (*others, 1j, object(), int, numpy.float64(1.0)):
            self.assertEqual(groups(value), [], msg=repr(value))

    def test_every_stop_has_a_program(self):
        table = self.statuses()
        stops = {
            RULES_LOWERED.get(n, n).partition(" ")[0]
            for n, statuses in table.items()
            if "stop" in statuses and (n.isupper() or n in RULES_LOWERED)
        }
        written = {RULES_LOWERED.get(n, n) for p in RULES_STOPS.values() for n in p[0]}
        self.assertEqual(stops, written | set(RULES_NOT_WRITTEN))
        self.assertEqual(written & set(RULES_NOT_WRITTEN), set())
        for name, program in RULES_STOPS.items():
            held = rules_opnames(compile(rules_source(name), "<rules>", "exec"))
            for stop in program[0]:
                self.assertIn(RULES_LOWERED.get(stop, stop), held, msg=name)

    @parametrize("name", list(RULES_STOPS))
    @checks_isolated
    def test_a_stopped_instruction_sits_in_no_region(self, name):
        # Counted from the first trace on: which instructions traces met.
        os.environ["TORCH_RUST_GATE_HIST"] = "1"
        source = rules_source(name)
        eager, traced = {}, {}
        exec(compile(source, "<rules eager>", "exec"), eager)
        exec(compile(source, "<rules vm>", "exec"), traced)
        compiled = vm(traced["f"], passes=CHECKS_EARLY)
        for call in range(13):
            # The value the construct reads: A, then B, then A again.
            second = 7 <= call < 10
            eager["change"](second)
            traced["change"](second)
            torch.manual_seed(call)
            x = torch.randn(4, 8)
            want = checks_outcome(eager["f"], x)
            got = checks_outcome(compiled, x)
            self.assertIsInstance(want, torch.Tensor, msg=f"call {call}")
            self.assertIsInstance(got, torch.Tensor, msg=f"call {call}: {got!r}")
            self.assertEqual(got, want, atol=0, rtol=0, msg=f"call {call}")
            self.assertEqual(traced["state"](), eager["state"](), msg=f"call {call}")
        # A region is what a trace that was not refused ran. Every trace
        # that met a stopped instruction was refused, for that instruction.
        rows = torch._rust._gate_report()["rows"]
        refusals = [
            str(why)
            for place in vm.regions(compiled)
            for state, why in place.get("traces", ())
            if state != "region"
        ]
        for stop in RULES_STOPS[name][0]:
            met = sum(traces for _, what, _, traces, _, _ in rows if what == stop)
            named = sum(f"no row for: {stop} " in why for why in refusals)
            self.assertEqual(met, named, msg=f"{stop}: {refusals}")


# How a program comes to hand a region an object the region then knows by
# identity, one entry per class of object and per way it is held: the
# function, and what makes a fresh object where the program keeps it (the
# arguments that follow the tensor, and the objects to watch by name).
class PinCfg:
    def __init__(self):
        self.k = 2.0

    def scale(self, x):
        return x * self.k


class PinSlots:
    __slots__ = ("k", "content")

    def __init__(self):
        self.k = 2.0
        self.content = PinCfg()


class PinModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((8,), 2.0))

    def forward(self, x):
        return x * self.weight


class PinBox:
    pass


PIN_STATE = {}
PIN_LIST = [None]
PIN_BOX = PinBox()
# Where the test's tensors are.
PIN_DEVICE = ["cpu"]


def pin_module():
    return PinModule().to(PIN_DEVICE[0])


def pin_helper():
    def helper(y):
        return y * 2.0

    return helper


def pin_argument(x, cfg):
    return ((x.sin() * cfg.k).cos() + 1).tanh().exp()


def pin_argument_in_a_tuple(x, pair):
    return ((x.sin() * pair[0].k).cos() + 1).tanh().exp()


def pin_called(x, call):
    return (call(x.sin()).cos() + 1).tanh().exp()


def pin_from_a_dict(x):
    return ((x.sin() * PIN_STATE["cfg"].k).cos() + 1).tanh().exp()


def pin_from_a_global(x):
    return ((x.sin() * PIN_GLOBAL.k).cos() + 1).tanh().exp()


def pin_from_an_attribute(x):
    return ((x.sin() * PIN_BOX.cfg.k).cos() + 1).tanh().exp()


def pin_from_a_list(x):
    return ((x.sin() * PIN_LIST[0].k).cos() + 1).tanh().exp()


def pin_called_from_a_global(x):
    return (PIN_CALLED(x.sin()).cos() + 1).tanh().exp()


def pin_tensor_from_a_global(x):
    return ((x.sin() * PIN_TENSOR).cos() + 1).tanh().exp()


def pin_tensor_also_handed(x, t):
    return ((x.sin() * t + PIN_TENSOR).cos() + 1).tanh().exp()


def pin_dict_from_a_global(x):
    return ((x.sin() * PIN_DICT["k"]).cos() + 1).tanh().exp()


def pin_tuple_from_a_global(x):
    return ((x.sin() * PIN_TUPLE[0]).cos() + 1).tanh().exp()


def pin_set_from_a_global(x):
    return ((x.sin() * len(PIN_SET)).cos() + 1).tanh().exp()


def pin_class_from_a_global(x):
    return ((x.sin() * PIN_CLASS.k).cos() + 1).tanh().exp()


def pin_cell():
    cfg = PinCfg()

    def pin_from_a_cell(x):
        return ((x.sin() * cfg.k).cos() + 1).tanh().exp()

    return pin_from_a_cell


def pin_as_argument(make, watch=None):
    def fresh():
        o = make()
        return (o,), {"object": o} if watch is None else watch(o)

    return fresh


def pin_as_global(name, make, watch=None):
    def fresh():
        o = globals()[name] = make()
        return (), {"object": o} if watch is None else watch(o)

    return fresh


def pin_as_global_and_argument(name, make):
    def fresh():
        o = globals()[name] = make()
        return (o,), {"object": o}

    return fresh


def pin_put(put, make):
    def fresh():
        o = make()
        put(o)
        return (), {"object": o}

    return fresh


def pin_module_parts(m):
    return {"object": m, "weight": m.weight}


def pin_method_parts(method):
    return {"object": method.__self__}


def pin_content(holder):
    if type(holder) is PinSlots:
        return {"content": holder.content}
    if type(holder) is set:
        return {"object": holder, "content": next(iter(holder))}
    return {"content": holder["s"] if type(holder) is dict else holder[1]}


def pin_holding():
    cfg = PinCfg()
    cfg.content = PinCfg()
    return cfg


def pin_held(cfg):
    return {"object": cfg, "content": cfg.content}


def pin_runs(compiled):
    """How many times the regions of a function have run."""
    return sum(r["runs"] for r in vm.regions(compiled) if r["state"] == "region")


# How a reader comes by the __dict__ of an instance without the instance:
# what the program keeps (made of `vars(OBJ)`), what a call computes with it,
# and how the program changes the dict it reads and puts it back. In place
# of the dict itself in the last two, `{d}` is the dict to change.
PIN_DICT_READERS = {
    "items_view": ("KEPT = vars(OBJ).items()", "dict(KEPT)['k']"),
    "values_view": ("KEPT = vars(OBJ).values()", "list(KEPT)[0]"),
    "keys_view": (
        "KEPT = vars(OBJ).keys()",
        "2.0 + float('z' in KEPT)",
        "{d}['z'] = 1",
        "del {d}['z']",
    ),
    "mappingproxy": ("KEPT = types.MappingProxyType(vars(OBJ))", "KEPT['k']"),
    "bound_method": ("KEPT = vars(OBJ).get", "KEPT('k')"),
    "the_dict": ("KEPT = vars(OBJ)", "KEPT['k']"),
    "the_instance": ("KEPT = None", "OBJ.k", "OBJ.k = 3.0", "OBJ.k = 2.0"),
}
# What becomes of the instance's dict before the change: nothing, or the
# instance is given an equal one (the reader still has the old one).
PIN_DICT_FATES = {
    "in_place": "pass",
    "instance_given_an_equal_dict": "OBJ.__dict__ = dict(OLD)",
}
PIN_DICT_PROGRAM = """
class Cfg:
    def __init__(self):
        self.k = 2.0
        self.j = 1.0


OBJ = Cfg()
OLD = vars(OBJ)
{kept}


def program(x):
    y = (x.sin() * OBJ.j).cos()
    k = {fetch}
    return ((y * k).cos() + 1).tanh().exp()


def change():
    {fate}
    {change}


def back():
    {back}
"""


def pin_frame_body(x):
    return ((x.sin() * 2.0).cos() + 1).tanh().exp()


def pin_frame_of_a_module():
    module = sys.modules["vm_pin_frame"] = types.ModuleType("vm_pin_frame")
    module.__builtins__ = builtins
    module.content = PinCfg()
    fn = types.FunctionType(pin_frame_body.__code__, vars(module))
    return fn, {"object": module, "content": module.content}


def pin_frame_of_no_module():
    names = {"__name__": "vm_pin_frame_of_none", "__builtins__": builtins}
    names["content"] = PinCfg()
    fn = types.FunctionType(pin_frame_body.__code__, names)
    return fn, {"content": names["content"]}


# A function of one code object whose frames have globals of their own, by
# what those globals are: what makes a fresh function, and the objects to
# watch. Its code loads no global, so that no read holds the namespace.
PIN_FRAMES = {
    "frame_of_a_loaded_module": pin_frame_of_a_module,
    "frame_of_no_module": pin_frame_of_no_module,
}
PIN_FROM_A_CELL = pin_cell()
PIN_WAYS = {
    "argument": (pin_argument, pin_as_argument(PinCfg)),
    "argument_with_slots": (pin_argument, pin_as_argument(PinSlots, pin_content)),
    # What the instance's __dict__ holds goes with the instance: the dict
    # the region read `k` from is not kept either.
    "argument_holding_an_object": (
        pin_argument,
        pin_as_argument(pin_holding, pin_held),
    ),
    "global_holding_an_object": (
        pin_from_a_global,
        pin_as_global("PIN_GLOBAL", pin_holding, pin_held),
    ),
    "argument_in_a_tuple": (
        pin_argument_in_a_tuple,
        pin_as_argument(lambda: (PinCfg(), 1), lambda pair: {"object": pair[0]}),
    ),
    "bound_method_argument": (
        pin_called,
        pin_as_argument(lambda: PinCfg().scale, pin_method_parts),
    ),
    "module_argument": (pin_called, pin_as_argument(pin_module, pin_module_parts)),
    "function_argument": (pin_called, pin_as_argument(pin_helper)),
    "item_of_a_dict": (
        pin_from_a_dict,
        pin_put(lambda o: PIN_STATE.update(cfg=o), PinCfg),
    ),
    "global": (pin_from_a_global, pin_as_global("PIN_GLOBAL", PinCfg)),
    "attribute": (
        pin_from_an_attribute,
        pin_put(lambda o: setattr(PIN_BOX, "cfg", o), PinCfg),
    ),
    "item_of_a_list": (
        pin_from_a_list,
        pin_put(lambda o: PIN_LIST.__setitem__(0, o), PinCfg),
    ),
    "content_of_a_cell": (
        PIN_FROM_A_CELL,
        pin_put(
            lambda o: setattr(PIN_FROM_A_CELL.__closure__[0], "cell_contents", o),
            PinCfg,
        ),
    ),
    "global_function": (
        pin_called_from_a_global,
        pin_as_global("PIN_CALLED", pin_helper),
    ),
    "global_module": (
        pin_called_from_a_global,
        pin_as_global("PIN_CALLED", pin_module, pin_module_parts),
    ),
    "global_tensor": (
        pin_tensor_from_a_global,
        pin_as_global(
            "PIN_TENSOR", lambda: torch.full((8,), 2.0, device=PIN_DEVICE[0])
        ),
    ),
    "global_tensor_also_an_argument": (
        pin_tensor_also_handed,
        pin_as_global_and_argument(
            "PIN_TENSOR", lambda: torch.full((8,), 2.0, device=PIN_DEVICE[0])
        ),
    ),
    "global_dict": (
        pin_dict_from_a_global,
        pin_as_global("PIN_DICT", lambda: {"k": 2.0, "s": PinCfg()}, pin_content),
    ),
    "global_tuple": (
        pin_tuple_from_a_global,
        pin_as_global("PIN_TUPLE", lambda: (2.0, PinCfg()), pin_content),
    ),
    "global_set": (
        pin_set_from_a_global,
        pin_as_global("PIN_SET", lambda: {PinCfg()}, pin_content),
    ),
    "global_class": (
        pin_class_from_a_global,
        pin_as_global("PIN_CLASS", lambda: type("PinMade", (), {"k": 2.0})),
    ),
}
# What a region still keeps alive after the program dropped it, by way and
# watched object, with the reason: for as long as the place keeps the region
# (or, for a constant, the record of its trace). Everything else is gone the
# moment the program lets go of it.
PIN_CONSTANT = (
    "a tensor the region's graph reads as a constant is an argument of the "
    "compiled graph and is kept by the type the trace's record has for it"
)
PIN_KEPT = {
    ("argument_with_slots", "content"): (
        "an instance of a class with __slots__ and no __weakref__ takes no "
        "weak reference: the region counts it, with what it holds"
    ),
    ("global_dict", "content"): (
        "an exact dict that is no instance's __dict__ takes no weak reference "
        "and has no owner to find it through: counted, with what it holds"
    ),
    ("global_tuple", "content"): (
        "a tuple takes no weak reference: counted, with what it holds"
    ),
    ("global_tensor", "object"): PIN_CONSTANT,
    ("global_tensor_also_an_argument", "object"): (
        "an input the trace also read from state is counted by the check of "
        "how the region's tensors share memory"
    ),
    ("global_module", "weight"): PIN_CONSTANT,
    ("module_argument", "weight"): PIN_CONSTANT,
    ("frame_of_no_module", "content"): (
        "a region is for the frames that have the globals of the frame it was "
        "traced in: a dict takes no weak reference, and one that is the dict "
        "of no loaded module has no owner to find it through: counted, with "
        "what it holds"
    ),
}


class TestVmPins(TestCase):
    """A region keeps no object of the program alive. What it compares by
    identity or reads from when it is entered (an argument, a module, a
    function, what a namespace, an attribute, an item or a cell holds) it
    holds by a weak reference, so the object dies when the program drops it;
    a region whose object died can never run again and is dropped when its
    place is next entered, and no check compares the address of an object
    that died. PIN_KEPT names what is still kept, and why."""

    hw_classification = HardwareClassification.ACCELERATOR

    def calls(self, compiled, fn, extra, first, n):
        for call in range(first, first + n):
            torch.manual_seed(call)
            x = torch.randn(4, 8, device=PIN_DEVICE[0])
            self.assertEqual(compiled(x, *extra), fn(x, *extra), atol=0, rtol=0)

    @parametrize("way", list(PIN_WAYS))
    @checks_isolated
    def test_an_object_the_program_drops_is_gone(self, device, way):
        PIN_DEVICE[0] = device
        fn, fresh = PIN_WAYS[way]
        compiled = vm(fn, passes=CHECKS_EARLY)
        extra, watched = fresh()
        watched = {name: weakref.ref(o) for name, o in watched.items()}
        self.calls(compiled, fn, extra, 0, 6)
        ran = pin_runs(compiled)
        self.assertGreater(ran, 0, msg="no region ran on the object")
        # The program's last references to the first object go here.
        extra, _ = fresh()
        gc.collect()
        kept = {name for name, ref in watched.items() if ref() is not None}
        listed = {name for name in watched if (way, name) in PIN_KEPT}
        self.assertEqual(kept, listed, msg=f"{way}: kept alive {sorted(kept)}")
        self.calls(compiled, fn, extra, 6, 8)
        states = [r["state"] for r in vm.regions(compiled)]
        self.assertGreater(pin_runs(compiled), 0, msg=f"none on the second: {states}")

    def test_what_is_kept_names_a_way(self, device):
        ways = set(PIN_WAYS) | set(PIN_FRAMES)
        self.assertLessEqual({way for way, _ in PIN_KEPT}, ways)

    @parametrize("way", list(PIN_FRAMES))
    @checks_isolated
    def test_the_namespace_of_a_function_the_program_drops(self, device, way):
        """A region is entered only by a frame with the globals and the
        builtins of the frame it was traced in. It finds the dict of a
        loaded module through the module, which it holds weakly: the
        namespace dies with the program's last function of it."""
        PIN_DEVICE[0] = device
        fn, watched = PIN_FRAMES[way]()
        compiled = vm(fn, passes=CHECKS_EARLY)
        watched = {name: weakref.ref(o) for name, o in watched.items()}
        self.calls(compiled, fn, (), 0, 6)
        ran = pin_runs(compiled)
        self.assertGreater(ran, 0, msg="no region ran in the function's frame")
        # The program's last references to the first function go here.
        fn, _ = PIN_FRAMES[way]()
        compiled = vm(fn, passes=CHECKS_EARLY)
        gc.collect()
        kept = {name for name, ref in watched.items() if ref() is not None}
        listed = {name for name in watched if (way, name) in PIN_KEPT}
        self.assertEqual(kept, listed, msg=f"{way}: kept alive {sorted(kept)}")
        self.calls(compiled, fn, (), 6, 8)
        states = [r["state"] for r in vm.regions(compiled)]
        self.assertGreater(pin_runs(compiled), ran, msg=f"none on the second: {states}")

    @checks_isolated
    def test_a_new_object_on_every_call_is_not_traced_for_ever(self, device):
        """Each call's object dies with the call, and its region with it.
        The place gives up after a few calls in a row that each brought an
        object it had no region for, as it does for objects that live on."""
        import torch._rust_inductor as tracer

        PIN_DEVICE[0] = device
        fn, fresh = PIN_WAYS["argument"]
        compiled = vm(fn, passes=CHECKS_EARLY)
        before = tracer.stats["traced"]
        for call in range(16):
            extra, _ = fresh()
            self.calls(compiled, fn, extra, call, 1)
        self.assertLessEqual(tracer.stats["traced"] - before, 4)

    @checks_isolated
    def test_dropping_a_region_runs_no_code_as_the_program(self, device):
        """A place tells the tracer of a region it dropped by calling a
        Python function of the engine's: no frame of the program's, so
        nothing of it is among the program's calls in vm.regions."""
        import torch._rust_inductor as tracer

        PIN_DEVICE[0] = device
        fn, fresh = PIN_WAYS["argument"]
        compiled = vm(fn, passes=CHECKS_EARLY)
        extra, _ = fresh()
        self.calls(compiled, fn, extra, 0, 6)
        told, dropped = [], tracer._dropped

        def noting(func, why):
            told.append(why)
            dropped(func, why)

        with unittest.mock.patch.object(tracer, "_dropped", noting):
            extra, _ = fresh()
            self.calls(compiled, fn, extra, 6, 2)
        self.assertEqual(told, ["gone"])
        where = [r["where"] for r in vm.regions(compiled)]
        mine = [w for w in where if "_dropped" in w or "noting" in w]
        self.assertEqual(mine, [], msg=str(where))

    @parametrize("reader", list(PIN_DICT_READERS))
    @parametrize("fate", list(PIN_DICT_FATES))
    @checks_isolated
    def test_dict_of_an_instance_read_without_the_instance(self, device, reader, fate):
        """A region that reads an attribute through the instance compares
        what it read in the dict the instance has, and holds neither. What it
        read of that dict some other way (a view, a mappingproxy, a bound
        method, the dict itself, kept by the program) is of the dict that was
        read: once the instance has another dict, however equal, the region
        is not entered on the instance's word."""
        kept, fetch, *how = PIN_DICT_READERS[reader]
        change, back = how or ("{d}['k'] = 3.0", "{d}['k'] = 2.0")
        source = PIN_DICT_PROGRAM.format(
            kept=kept,
            fetch=fetch,
            fate=PIN_DICT_FATES[fate],
            change=change.format(d="OLD"),
            back=back.format(d="OLD"),
        )
        mine, theirs = {"types": types}, {"types": types}
        exec(source, mine)
        exec(source, theirs)
        compiled = vm(mine["program"], passes=CHECKS_EARLY)
        # The schedule A, B, A.
        for phase, calls in enumerate((6, 3, 4)):
            for names in (mine, theirs):
                if phase:
                    names["change" if phase == 1 else "back"]()
            for call in range(calls):
                torch.manual_seed(10 * phase + call)
                x = torch.randn(4, 8, device=device)
                msg = f"phase {phase}, call {call}"
                want = theirs["program"](x)
                self.assertEqual(compiled(x), want, atol=0, rtol=0, msg=msg)
            if phase == 0:
                self.assertGreater(pin_runs(compiled), 0, msg="no region ran")

    @parametrize("way", ["argument", "item_of_a_dict"])
    @checks_isolated
    def test_an_object_at_the_address_of_one_that_died_enters_no_region(
        self, device, way
    ):
        PIN_DEVICE[0] = device
        fn, fresh = PIN_WAYS[way]
        compiled = vm(fn, passes=CHECKS_EARLY)
        # Among objects of its class that live on, so that the memory of the
        # one that dies is the next to be handed out for another. Which
        # memory an object gets is the allocator's to say: tried again, with
        # the next object, when it was another.
        around = [PinCfg() for _ in range(1024)]
        for attempt in range(8):
            extra, watched = fresh()
            around += [PinCfg() for _ in range(1024)]
            address = id(watched["object"])
            self.calls(compiled, fn, extra, 0, 6)
            self.assertGreater(pin_runs(compiled), 0, msg=f"attempt {attempt}")
            del watched
            extra = None
            PIN_STATE.clear()
            extra, watched = fresh()
            if id(watched["object"]) != address:
                around.append(extra)
                continue
            # A region entered for it would compute with the 2.0 of the
            # object that died.
            watched["object"].k = 3.0
            ran = pin_runs(compiled)
            self.calls(compiled, fn, extra, 6, 1)
            self.assertLessEqual(pin_runs(compiled), ran)
            return
        self.fail("no object was made at the address of one that died")


instantiate_device_type_tests(TestVmPins, globals())


# The ways two tensors of 16 elements' worth of memory overlap: what each
# gives for (the first tensor's memory, the memory the second is in when the
# two are apart, whether the two share). The second has the same sizes and
# strides either way.
MEMORY_OVERLAPS = {
    "one_object": lambda base, other, shared: (
        (base[0:8],) * 2 if shared else (base[0:8], other[0:8])
    ),
    "one_memory": lambda base, other, shared: (
        base[0:8],
        (base if shared else other)[0:8],
    ),
    "detached": lambda base, other, shared: (
        base[0:8],
        (base if shared else other)[0:8].detach(),
    ),
    "partial": lambda base, other, shared: (
        base[0:6],
        (base if shared else other)[4:10],
    ),
    "interleaved": lambda base, other, shared: (
        base[0::2],
        (base if shared else other)[1::2],
    ),
    "expanded": lambda base, other, shared: (
        base[0:8],
        (base if shared else other)[2:3].expand(8),
    ),
    "transposed": lambda base, other, shared: (
        base.view(4, 4),
        (base if shared else other).view(4, 4).T,
    ),
    "two_storages": lambda base, other, shared: (
        base[0:8],
        torch.from_numpy((base if shared else other).numpy())[0:8],
    ),
}
# What cannot be: an expanded tensor is not written in place, and memory
# from numpy is the CPU's.
MEMORY_NOT_WRITTEN = {"expanded"}
MEMORY_CPU_ONLY = {"two_storages"}


def memory_first_written(a, b):
    a.add_(1.0)
    c = b * 2.0
    a.mul_(c.sum().tanh())
    return (a - b).tanh() + c


def memory_second_written(a, b):
    c = a * 2.0
    b.add_(c)
    d = (a + 1.0).cos()
    b.mul_(0.5)
    return (a - b).tanh() + d


def memory_neither_written(a, b):
    c = a * 2.0 + b
    return (c - b).tanh() + (a * b).cos()


class MemoryHolder(torch.nn.Module):
    pass


# Its `store` and `w` (elements 4 to 11 of the store) are put on the test's
# device by the test.
MEMORY_HELD = MemoryHolder()


def memory_input_written(x, g):
    x.sub_(g * 0.1)
    c = MEMORY_HELD.w * 2.0
    return (x - MEMORY_HELD.w).tanh() + c


def memory_state_written(x, g):
    # As an optimizer's step: the weight is updated in place.
    y = x * 2.0
    MEMORY_HELD.w.sub_(g * 0.1)
    return (x - MEMORY_HELD.w).tanh() + y


def memory_state_read(x, g):
    y = x * 2.0 + MEMORY_HELD.w
    return (y - g).tanh() + (x * MEMORY_HELD.w).cos()


MEMORY_WRITTEN = {
    "first": (memory_first_written, memory_input_written),
    "second": (memory_second_written, memory_state_written),
    "neither": (memory_neither_written, memory_state_read),
}
# How an input stands to the weight when the two share: what it is made of.
MEMORY_OF_STATE = {
    "one_object": lambda held: held.w,
    "one_memory": lambda held: held.store[4:12],
    "detached": lambda held: held.w.detach(),
    "partial": lambda held: held.store[0:8],
    "strided_through": lambda held: held.store[1::2],
}
MEMORY_APART_OF_STATE = {"strided_through": lambda base: base[1::2]}
MEMORY_LISTS = {
    "default": CHECKS_EARLY,
    "effects": vm.configure(
        (*vm.DEFAULT_PASSES, vm.passes.Effects()),
        regions={"at": 3, "observe": 1},
        pure_calls=1,
    ),
}
MEMORY_MISS = "how its tensors share memory"


def memory_unplaced(device):
    """Tensors whose place in memory cannot be said, by kind."""
    return {
        "sparse": torch.zeros(8, device=device).to_sparse(),
        "meta": torch.zeros(8, device="meta"),
        "zero_tensor": torch._efficientzerotensor((8,), device=device),
        "no_tensor": 3.0,
    }


def memory_with_a_meta_tensor(x, m):
    y = x.sin() * 2 + 1
    n = m * 2 + 1
    return (y @ y.t()).cos().tanh().exp(), n.shape


@unittest.skipUnless(CHECKS_HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmMemoryRelation(TestCase):
    """A region is entered only by a call whose tensors (its inputs and the
    tensors its graph reads from state) stand to one another in memory as
    the traced call's did, for every pair of them and on every pass list:
    the same ones are one object, the same ones overlap and as far apart,
    all the others are apart. Another call is a miss and is traced for
    itself. A trace with a tensor whose place cannot be said is refused."""

    hw_classification = HardwareClassification.ACCELERATOR

    def counts(self, compiled, fn):
        rows = [
            r
            for r in vm.regions(compiled)
            if r["function"] == fn.__name__ and r["state"] == "region"
        ]
        missed = sum(dict(r.get("misses") or {}).get(MEMORY_MISS, 0) for r in rows)
        return sum(r["runs"] for r in rows), missed

    def phases(self, fn, passes, traced, make):
        """Sixteen calls of ``fn`` on the VM and in eager: six with the
        tensors standing as ``traced`` says, five the other way, five the
        first way again. ``make(call, shared)`` gives a call's arguments
        and the memory to compare after it."""
        compiled = vm(fn, passes=MEMORY_LISTS[passes])
        seen = []
        for call in range(16):
            shared = (6 <= call < 11) == (traced == "apart")
            outcomes = []
            for run in (fn, compiled):
                args, memory = make(call, shared)
                out = checks_outcome(run, *args)
                self.assertNotIsInstance(out, Exception, msg=f"call {call}: {out!r}")
                outcomes.append([out, *(m.clone() for m in memory)])
            want, got = outcomes
            self.assertEqual(got, want, atol=0, rtol=0, msg=f"call {call}")
            seen.append(self.counts(compiled, fn))
        self.ran_each_way(seen)

    def ran_each_way(self, seen):
        """``seen``: the counts after each of sixteen calls, of which the
        calls 6 to 10 had the tensors standing the other way."""
        (runs1, missed1), (runs2, missed2), (runs3, _) = seen[5], seen[10], seen[15]
        # A region ran while the tensors stood as traced, and no call missed.
        self.assertGreater(runs1, 0, msg=str(seen))
        self.assertEqual(missed1, 0, msg=str(seen))
        # The other way is a miss, and a region of its own from then on.
        self.assertGreater(missed2, 0, msg=str(seen))
        self.assertGreater(runs2, runs1, msg=str(seen))
        # The first way again: its region is still there.
        self.assertGreater(runs3, runs2, msg=str(seen))

    def two_inputs(self, device, overlap, written, traced, passes):
        if overlap in MEMORY_CPU_ONLY and torch.device(device).type != "cpu":
            self.skipTest("memory handed over twice by numpy is the CPU's")

        def make(call, shared):
            torch.manual_seed(call)
            base, other = (torch.randn(16, device=device) for _ in range(2))
            return MEMORY_OVERLAPS[overlap](base, other, shared), [base, other]

        self.phases(MEMORY_WRITTEN[written][0], passes, traced, make)

    @parametrize("traced", ("apart", "shared"))
    @parametrize("written", ("first", "second", "neither"))
    @parametrize("overlap", list(MEMORY_OVERLAPS))
    @checks_isolated
    def test_two_inputs(self, device, overlap, written, traced):
        if written == "second" and overlap in MEMORY_NOT_WRITTEN:
            self.skipTest("an expanded tensor is not written in place")
        self.two_inputs(device, overlap, written, traced, "default")

    @parametrize("overlap", list(MEMORY_OVERLAPS))
    @checks_isolated
    def test_two_inputs_with_effects_held(self, device, overlap):
        self.two_inputs(device, overlap, "first", "apart", "effects")

    def input_and_state(self, device, overlap, written, traced, passes):
        held = MEMORY_HELD
        held.store = torch.zeros(16, device=device)
        held.w = held.store[4:12]

        def make(call, shared):
            torch.manual_seed(call)
            base = torch.randn(16, device=device)
            g = torch.randn(8, device=device)
            held.store.copy_(torch.arange(16.0, device=device) / 16 + call / 8)
            apart = MEMORY_APART_OF_STATE.get(overlap, lambda base: base[0:8])
            x = MEMORY_OF_STATE[overlap](held) if shared else apart(base)
            return (x, g), [base, held.store]

        self.phases(MEMORY_WRITTEN[written][1], passes, traced, make)

    @parametrize("traced", ("apart", "shared"))
    @parametrize("written", ("first", "second", "neither"))
    @parametrize("overlap", list(MEMORY_OF_STATE))
    @checks_isolated
    def test_an_input_and_a_tensor_read_from_state(
        self, device, overlap, written, traced
    ):
        self.input_and_state(device, overlap, written, traced, "default")

    @parametrize("overlap", list(MEMORY_OF_STATE))
    @checks_isolated
    def test_an_input_and_a_tensor_read_from_state_with_effects_held(
        self, device, overlap
    ):
        self.input_and_state(device, overlap, "second", "apart", "effects")

    @checks_isolated
    def test_a_weight_whose_memory_moves_onto_an_input(self, device):
        # The tensors of every call are the same objects: what changes is
        # where the weight's data is (the reads tell the relation).
        held, fn = MEMORY_HELD, memory_state_written
        held.store = torch.zeros(16, device=device)
        held.w = torch.zeros(8, device=device)
        compiled = vm(fn, passes=CHECKS_EARLY)
        seen = []
        for call in range(16):
            if call == 6:
                held.w.data = held.store[4:12]
            if call == 11:
                held.w.data = torch.zeros(8, device=device)
            outcomes = []
            for run in (fn, compiled):
                torch.manual_seed(call)
                g = torch.randn(8, device=device)
                held.store.copy_(torch.arange(16.0, device=device) / 16 + call / 8)
                if not 6 <= call < 11:
                    held.w.fill_(call / 4)
                out = run(held.store[0:8], g)
                outcomes.append([out, held.store.clone(), held.w.clone()])
            want, got = outcomes
            self.assertEqual(got, want, atol=0, rtol=0, msg=f"call {call}")
            seen.append(self.counts(compiled, fn))
        self.ran_each_way(seen)

    def test_every_overlap_is_told_from_apart(self, device):
        # The relation itself, for every way of overlapping, both ways.
        for overlap, make in MEMORY_OVERLAPS.items():
            if overlap in MEMORY_CPU_ONLY and torch.device(device).type != "cpu":
                continue
            base, other = (torch.randn(16, device=device) for _ in range(2))
            shared, apart = make(base, other, True), make(base, other, False)
            ways = ((shared, shared, apart), (apart, apart, shared))
            for traced, same, differs in ways:
                relation = torch._rust.Overlap(list(traced), [])
                self.assertIsNone(relation.unplaced, msg=overlap)
                self.assertTrue(relation.needed, msg=overlap)
                self.assertTrue(relation.holds(list(same)), msg=overlap)
                self.assertFalse(relation.holds(list(differs)), msg=overlap)
                # With the second among the constants, the same.
                relation = torch._rust.Overlap([traced[0]], [traced[1]])
                self.assertTrue(relation.holds([same[0]]), msg=overlap)
                if traced is apart:
                    self.assertFalse(relation.holds([apart[1]]), msg=overlap)

    def test_overlapping_tensors_are_where_they_were(self, device):
        base, other = (torch.zeros(16, device=device) for _ in range(2))
        relation = torch._rust.Overlap([base[0:8], base[4:12]], [])
        self.assertTrue(relation.holds([other[0:8], other[4:12]]))
        # As far apart, somewhere else in the storage; nearer; the other
        # way around; in two storages.
        self.assertFalse(relation.holds([other[2:10], other[6:14]]))
        self.assertFalse(relation.holds([other[0:8], other[2:10]]))
        self.assertFalse(relation.holds([other[4:12], other[0:8]]))
        self.assertFalse(relation.holds([other[0:8], base[4:12]]))
        # Three in a row, of which the outer two do not meet.
        relation = torch._rust.Overlap([base[0:6], base[4:10], base[8:14]], [])
        self.assertTrue(relation.holds([other[0:6], other[4:10], other[8:14]]))
        self.assertFalse(relation.holds([other[0:6], other[4:10], base[8:14]]))
        self.assertFalse(relation.holds([other[0:6], other[4:10], other[10:16]]))

    def test_a_tensor_without_elements_overlaps_none(self, device):
        base = torch.zeros(16, device=device)
        a, b = torch.zeros(8, device=device), torch.zeros(0, device=device)
        relation = torch._rust.Overlap([a, b], [])
        self.assertTrue(relation.holds([base[0:8], base[4:4]]))
        # It is still an object, which another tensor is or is not.
        self.assertFalse(torch._rust.Overlap([b, b], []).holds([b, base[0:0]]))
        self.assertFalse(torch._rust.Overlap([b, base[0:0]], []).holds([b, b]))

    def test_a_tensor_whose_place_cannot_be_said(self, device):
        a = torch.zeros(8, device=device)
        for kind, t in memory_unplaced(device).items():
            for inputs, consts in (([a, t], []), ([t, a], []), ([a], [t])):
                relation = torch._rust.Overlap(inputs, consts)
                self.assertIsNotNone(relation.unplaced, msg=kind)
                self.assertFalse(relation.holds(inputs), msg=kind)
            # Asked of a call, it is a miss.
            relation = torch._rust.Overlap([a, torch.zeros(8, device=device)], [])
            self.assertIsNone(relation.unplaced)
            self.assertFalse(relation.holds([a, t]), msg=kind)
            # One tensor stands to no other: nothing is asked of it.
            relation = torch._rust.Overlap([t], [])
            self.assertIsNone(relation.unplaced, msg=kind)
            self.assertFalse(relation.needed, msg=kind)

    def test_a_constant_that_moves_is_looked_at_again(self, device):
        x, w, v = (torch.zeros(8, device=device) for _ in range(3))
        base = torch.zeros(16, device=device)
        relation = torch._rust.Overlap([x], [w, v])
        self.assertTrue(relation.holds([base[0:8]]))
        w.data = base[4:12]
        self.assertFalse(relation.holds([base[0:8]]))
        self.assertTrue(relation.holds([x]))
        v.data = base[8:16]
        self.assertFalse(relation.holds([x]))
        v.data = torch.zeros(8, device=device)
        self.assertTrue(relation.holds([x]))

    @checks_isolated
    def test_no_region_for_a_tensor_whose_place_cannot_be_said(self, device):
        fn = memory_with_a_meta_tensor
        compiled = vm(fn, passes=CHECKS_EARLY)
        for call in range(8):
            torch.manual_seed(call)
            x, m = torch.randn(4, 8, device=device), torch.zeros(8, device="meta")
            got, want = compiled(x, m), fn(x, m)
            self.assertEqual(got, want, atol=0, rtol=0, msg=f"call {call}")
        rows = [r for r in vm.regions(compiled) if r["function"] == fn.__name__]
        whys = [why for r in rows for _, why in r.get("traces") or ()]
        unplaced = [why for why in whys if "cannot be said" in (why or "")]
        self.assertTrue(unplaced, msg=str(rows))
        self.assertEqual(checks_region_runs(compiled, fn), 0, msg=str(rows))


instantiate_device_type_tests(TestVmMemoryRelation, globals())


IMPORTS_SETUP = """import builtins, importlib, os, sys, tempfile, types
def module(name, **items):
    m = types.ModuleType(name)
    m.__dict__.update(items)
    return m
def package(name, k):
    p = module(name)
    p.sub = module(name + '.sub', K=k)
    return p
def put(p):
    sys.modules[p.__name__] = p
    sys.modules[p.sub.__name__] = p.sub
"""
IMPORTS_MODULE = IMPORTS_SETUP + (
    "A = module('vm_imp_m', K=2.0)\nB = module('vm_imp_m', K=5.0)\n"
    "sys.modules['vm_imp_m'] = A"
)
IMPORTS_PACKAGE = IMPORTS_SETUP + (
    "P = package('vm_imp_p', 2.0)\nP2 = package('vm_imp_p', 5.0)\nput(P)"
)
# A module that the import machinery loads from a file, which reads its K
# from where the test puts it.
IMPORTS_FILE = (
    IMPORTS_SETUP
    + """DIR = tempfile.TemporaryDirectory()
with open(os.path.join(DIR.name, '{0}.py'), 'w') as file:
    file.write('import sys\\nK = sys.vm_imp_k\\n')
sys.dont_write_bytecode = True
sys.path.insert(0, DIR.name)
sys.vm_imp_k = 2.0
"""
)
# Modules whose spec is importlib's, made by hand: it has no `_initializing`
# of its own, as the spec of a reloaded module has none. The import waits for
# a module whose spec says it is being initialized: WAITED counts that.
IMPORTS_SPEC = (
    IMPORTS_MODULE
    + """
Spec = importlib.machinery.ModuleSpec
A.__spec__ = Spec('vm_imp_m', None)
B.__spec__ = Spec('vm_imp_m', None)
OWN_WAIT = importlib._bootstrap._lock_unlock_module
WAITED = []
def waited(name):
    if name == 'vm_imp_m':
        WAITED.append(name)
    return OWN_WAIT(name)
def waits():
    sys.modules['vm_imp_m'] = A
    importlib._bootstrap._lock_unlock_module = waited
    A.__spec__.__dict__.pop('_initializing', None)
    if '_initializing' in vars(Spec):
        del Spec._initializing"""
)
# Classes a module can be given (`m.__class__ = Sub`), whose own code then
# answers for the module's attributes: RAN has what that code was asked for.
IMPORTS_CLASS = """
RAN = []
class WithProperty(types.ModuleType):
    @property
    def K(self):
        RAN.append('K')
        return 5.0
class WithGetattribute(types.ModuleType):
    def __getattribute__(self, name):
        RAN.append(name)
        return 7.0 if name == 'K' else super().__getattribute__(name)
def become(m, cls, second):
    m.__class__ = cls if second else types.ModuleType"""
IMPORTS_ONE = "import vm_imp_m\nk = vm_imp_m.K"
IMPORTS_SUB = "import vm_imp_p.sub\nk = vm_imp_p.sub.K"
IMPORTS_FROM = "from vm_imp_m import K\nk = K"

# A program per way an import can differ between two calls: (the opcodes it
# holds, what the module has, the construct, which sets `k`, what changes
# before a call (`second`: for calls 7 to 9), the state the calls leave,
# whether the import has a row as the program first runs it).
IMPORTS_PROGRAMS = {
    "module_replaced": (
        ("IMPORT_NAME",),
        IMPORTS_MODULE,
        IMPORTS_ONE,
        "sys.modules['vm_imp_m'] = B if second else A",
        "None",
        True,
    ),
    "module_removed": (
        ("IMPORT_NAME",),
        IMPORTS_MODULE,
        IMPORTS_ONE,
        "sys.modules['vm_imp_m'] = A\nif second:\n    del sys.modules['vm_imp_m']",
        "None",
        True,
    ),
    "module_is_none": (
        ("IMPORT_NAME",),
        IMPORTS_MODULE,
        IMPORTS_ONE,
        "sys.modules['vm_imp_m'] = None if second else A",
        "None",
        True,
    ),
    "import_function_replaced": (
        ("IMPORT_NAME",),
        IMPORTS_MODULE
        + "\nOWN = builtins.__import__\nSEEN = []\n"
        + "def other(name, *args, **kwargs):\n"
        + "    if name != 'vm_imp_m':\n        return OWN(name, *args, **kwargs)\n"
        + "    SEEN.append(name)\n    return B",
        IMPORTS_ONE,
        "sys.modules['vm_imp_m'] = A\nbuiltins.__import__ = other if second else OWN",
        "len(SEEN)",
        True,
    ),
    "package_replaced": (
        ("IMPORT_NAME",),
        IMPORTS_PACKAGE,
        IMPORTS_SUB,
        "put(P2 if second else P)",
        "None",
        True,
    ),
    "submodule_removed": (
        ("IMPORT_NAME",),
        IMPORTS_PACKAGE,
        IMPORTS_SUB,
        "put(P)\nif second:\n    del sys.modules['vm_imp_p.sub']",
        "None",
        True,
    ),
    "three_names": (
        ("IMPORT_NAME",),
        IMPORTS_SETUP
        + "def deep(k):\n    x = module('vm_imp_x')\n"
        + "    x.y = module('vm_imp_x.y')\n"
        + "    x.y.z = module('vm_imp_x.y.z', K=k)\n    return x\n"
        + "def put(x):\n    sys.modules['vm_imp_x'] = x\n"
        + "    sys.modules['vm_imp_x.y'] = x.y\n"
        + "    sys.modules['vm_imp_x.y.z'] = x.y.z\n"
        + "X = deep(2.0)\nX2 = deep(5.0)\nput(X)",
        "import vm_imp_x.y.z\nk = vm_imp_x.y.z.K",
        "put(X2 if second else X)",
        "None",
        True,
    ),
    "submodule_under_another_name": (
        ("IMPORT_NAME", "IMPORT_FROM"),
        IMPORTS_PACKAGE + "\nS = P.sub",
        "import vm_imp_p.sub as other\nk = other.K",
        "P.sub = S\nput(P)\nP.sub = P2.sub if second else S",
        "None",
        True,
    ),
    "name_replaced": (
        ("IMPORT_NAME", "IMPORT_FROM"),
        IMPORTS_MODULE,
        IMPORTS_FROM,
        "sys.modules['vm_imp_m'] = A\nA.K = 5.0 if second else 2.0",
        "None",
        True,
    ),
    "name_deleted": (
        ("IMPORT_NAME", "IMPORT_FROM"),
        IMPORTS_MODULE,
        IMPORTS_FROM,
        "sys.modules['vm_imp_m'] = A\nA.K = 2.0\nif second:\n    del A.K",
        "None",
        True,
    ),
    "name_from_a_submodule_in_sys_modules": (
        ("IMPORT_NAME", "IMPORT_FROM"),
        IMPORTS_PACKAGE + "\ndel P.sub\nS = sys.modules['vm_imp_p.sub']",
        "from vm_imp_p import sub\nk = sub.K",
        "sys.modules['vm_imp_p'] = P\n"
        "sys.modules['vm_imp_p.sub'] = P2.sub if second else S",
        "None",
        True,
    ),
    "module_gets_a_class_with_a_property": (
        ("IMPORT_NAME", "IMPORT_FROM"),
        IMPORTS_MODULE + IMPORTS_CLASS,
        IMPORTS_FROM,
        "sys.modules['vm_imp_m'] = A\nbecome(A, WithProperty, second)",
        "RAN[:]",
        True,
    ),
    "module_gets_a_class_with_getattribute": (
        ("IMPORT_NAME", "IMPORT_FROM"),
        IMPORTS_MODULE + IMPORTS_CLASS,
        IMPORTS_FROM,
        "sys.modules['vm_imp_m'] = A\nbecome(A, WithGetattribute, second)",
        "RAN[:]",
        True,
    ),
    "module_only_imported_gets_a_class": (
        ("IMPORT_NAME",),
        IMPORTS_MODULE + IMPORTS_CLASS,
        "import vm_imp_m\nk = 3.0",
        "sys.modules['vm_imp_m'] = A\nbecome(A, WithGetattribute, second)",
        "RAN[:]",
        True,
    ),
    "package_gets_a_class": (
        ("IMPORT_NAME",),
        IMPORTS_PACKAGE + IMPORTS_CLASS,
        "import vm_imp_p.sub\nk = 3.0",
        "put(P)\nbecome(P, WithGetattribute, second)",
        "RAN[:]",
        True,
    ),
    "submodule_gets_a_class": (
        ("IMPORT_NAME",),
        IMPORTS_PACKAGE + IMPORTS_CLASS + "\nS = P.sub",
        "import vm_imp_p.sub\nk = 3.0",
        "put(P)\nbecome(S, WithGetattribute, second)",
        "RAN[:]",
        True,
    ),
    "module_with_a_submodule_in_sys_modules_gets_a_class": (
        ("IMPORT_NAME", "IMPORT_FROM"),
        IMPORTS_PACKAGE
        + IMPORTS_CLASS
        + "\ndel P.sub\nS = sys.modules['vm_imp_p.sub']",
        "from vm_imp_p import sub\nk = sub.K",
        "sys.modules['vm_imp_p'] = P\nsys.modules['vm_imp_p.sub'] = S\n"
        "become(P, WithGetattribute, second)",
        "RAN[:]",
        True,
    ),
    # The names of a package are taken by importlib's code, under no trace:
    # the module is then an input of the region that holds IMPORT_FROM.
    "package_a_name_is_taken_from_gets_a_class": (
        ("IMPORT_FROM", "IMPORT_NAME"),
        IMPORTS_SETUP
        + "G = module('vm_imp_g', K=2.0, __path__=[])\nsys.modules['vm_imp_g'] = G"
        + IMPORTS_CLASS,
        "from vm_imp_g import K\nk = K",
        "sys.modules['vm_imp_g'] = G\nbecome(G, WithProperty, second)",
        "RAN[:]",
        True,
    ),
    "reloaded": (
        ("IMPORT_NAME",),
        IMPORTS_FILE.format("vm_imp_file") + "import vm_imp_file",
        "import vm_imp_file\nk = vm_imp_file.K",
        "if sys.vm_imp_k != (5.0 if second else 2.0):\n"
        "    sys.vm_imp_k = 5.0 if second else 2.0\n"
        "    importlib.reload(sys.modules['vm_imp_file'])",
        "None",
        True,
    ),
    "spec_without_initializing_replaced": (
        ("IMPORT_NAME",),
        IMPORTS_SPEC,
        IMPORTS_ONE,
        "waits()\nsys.modules['vm_imp_m'] = B if second else A",
        "len(WAITED)",
        True,
    ),
    "spec_gets_initializing": (
        ("IMPORT_NAME",),
        IMPORTS_SPEC,
        IMPORTS_ONE,
        "waits()\nif second:\n    A.__spec__._initializing = True",
        "len(WAITED)",
        True,
    ),
    "spec_class_gets_initializing": (
        ("IMPORT_NAME",),
        IMPORTS_SPEC,
        IMPORTS_ONE,
        "waits()\nif second:\n    Spec._initializing = True",
        "len(WAITED)",
        True,
    ),
    "spec_class_gets_getattr": (
        ("IMPORT_NAME",),
        IMPORTS_SPEC + "\ndef busy(self, name):\n    return True",
        IMPORTS_ONE,
        "waits()\nif '__getattr__' in vars(Spec):\n    del Spec.__getattr__\n"
        "if second:\n    Spec.__getattr__ = busy",
        "len(WAITED)",
        True,
    ),
    "spec_class_has_initializing": (
        ("IMPORT_NAME",),
        IMPORTS_SPEC,
        IMPORTS_ONE,
        "waits()\nSpec._initializing = True\n"
        "sys.modules['vm_imp_m'] = B if second else A",
        "len(WAITED)",
        False,
    ),
    "spec_of_another_class": (
        ("IMPORT_NAME",),
        IMPORTS_MODULE + "\nA.__spec__ = B.__spec__ = types.SimpleNamespace()",
        IMPORTS_ONE,
        "sys.modules['vm_imp_m'] = B if second else A",
        "None",
        False,
    ),
    "name_from_a_module_getattr": (
        ("IMPORT_NAME",),
        IMPORTS_SETUP
        + "STATE = [2.0]\ndef provide(name):\n    if name == 'K':\n"
        + "        return STATE[0]\n    raise AttributeError(name)\n"
        + "G = module('vm_imp_g', __getattr__=provide)\nsys.modules['vm_imp_g'] = G",
        "from vm_imp_g import K\nk = K",
        "sys.modules['vm_imp_g'] = G\nSTATE[0] = 5.0 if second else 2.0",
        "None",
        False,
    ),
    "relative": (
        ("IMPORT_NAME",),
        IMPORTS_PACKAGE + "\n__package__ = 'vm_imp_p'",
        "from . import sub\nk = sub.K",
        "put(P2 if second else P)",
        "None",
        False,
    ),
    "first_import": (
        ("IMPORT_NAME",),
        IMPORTS_FILE.format("vm_imp_first"),
        "import vm_imp_first\nk = vm_imp_first.K",
        "sys.vm_imp_k = 5.0 if second else 2.0\nsys.modules.pop('vm_imp_first', None)",
        "None",
        False,
    ),
}


@instantiate_parametrized_tests
@unittest.skipUnless(CHECKS_HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmRulesImports(TestCase):
    """An import of a loaded module has a row: it reads ``__import__`` of the
    builtins and items of ``sys.modules``, runs no code and writes nothing.
    What it read is what a region over it is checked on, so a region runs
    over the import while nothing changes, and whatever changes between two
    calls the program computes what eager computes. Any other import (a
    module that is not loaded, a relative one, names taken from a module with
    a ``__getattr__``) runs under no trace."""

    hw_classification = HardwareClassification.GENERIC

    def test_the_imports_have_rows(self):
        table = collections.defaultdict(list)
        for name, status in torch._rust._rules_kinds():
            table[name].append(status)
        for name in ("IMPORT_NAME", "IMPORT_FROM"):
            self.assertEqual(sorted(table[name]), ["later:imports", "row"])
        rows = [
            line.split("\t")
            for line in torch._rust._rules_dump().splitlines()
            if line.startswith("row\timports\t")
        ]
        self.assertEqual(
            sorted((r[2], r[3]) for r in rows),
            [
                ("IMPORT_FROM", "0:ExactModule"),
                ("IMPORT_NAME", "0:Immutable,1:ExactTuple"),
                ("IMPORT_NAME", "0:Immutable,1:Immutable"),
            ],
        )
        for row in rows:
            # They read more than their operands, enter nothing, write nothing.
            self.assertIn("Modules", row[4].split(","))
            # A module's attribute is its dict's item for one class only.
            self.assertIn("Class", row[4].split(","))
            self.assertEqual(row[5:8], ["None", "Pinned", "Never"])

        class Lazy(types.ModuleType):
            pass

        self.assertEqual(torch._rust._rules_groups(types), ["ExactModule"])
        self.assertEqual(torch._rust._rules_groups(Lazy("lazy")), [])

    def test_every_way_has_a_program(self):
        for name, program in IMPORTS_PROGRAMS.items():
            source = rules_source(name, IMPORTS_PROGRAMS)
            held = rules_opnames(compile(source, "<imports>", "exec"))
            self.assertLessEqual(set(program[0]), held, msg=name)
            self.assertLessEqual(set(program[0]), {"IMPORT_NAME", "IMPORT_FROM"})

    def outcome(self, fn, x, call):
        got = checks_outcome(fn, x)
        if isinstance(got, Exception):
            return type(got), str(got)
        self.assertIsInstance(got, torch.Tensor, msg=f"call {call}")
        return got

    @parametrize("name", list(IMPORTS_PROGRAMS))
    @checks_isolated
    def test_an_import_between_calls(self, name):
        os.environ["TORCH_RUST_GATE_HIST"] = "1"
        opcodes, _, construct, _, _, has_row = IMPORTS_PROGRAMS[name]
        source = rules_source(name, IMPORTS_PROGRAMS)
        eager, traced = {}, {}
        exec(compile(source, "<imports eager>", "exec"), eager)
        exec(compile(source, "<imports vm>", "exec"), traced)
        compiled = vm(traced["f"], passes=CHECKS_EARLY)
        own_import = builtins.__import__
        own_wait = importlib._bootstrap._lock_unlock_module
        raised = 0
        try:
            for call in range(13):
                # What the import finds: A, then B, then A again.
                second = 7 <= call < 10
                torch.manual_seed(call)
                x = torch.randn(4, 8)
                eager["change"](second)
                want = self.outcome(eager["f"], x, call)
                traced["change"](second)
                got = self.outcome(compiled, x, call)
                raised += isinstance(want, tuple)
                self.assertEqual(type(got), type(want), msg=f"call {call}: {got!r}")
                if isinstance(want, tuple):
                    self.assertTrue(got == want, msg=f"call {call}: {got!r}")
                else:
                    self.assertEqual(got, want, atol=0, rtol=0, msg=f"call {call}")
                self.assertEqual(
                    traced["state"](), eager["state"](), msg=f"call {call}"
                )
        finally:
            builtins.__import__ = own_import
            importlib._bootstrap._lock_unlock_module = own_wait
            for added in ("_initializing", "__getattr__"):
                if added in vars(importlib.machinery.ModuleSpec):
                    delattr(importlib.machinery.ModuleSpec, added)
        # Eager raises only where the change took away what the import finds.
        self.assertIn(raised, (0, 3))
        rows = torch._rust._gate_report()["rows"]
        refusals = [
            str(why)
            for place in vm.regions(compiled)
            for state, why in place.get("traces", ())
            if state != "region"
        ]
        met = sum(traces for _, what, _, traces, _, _ in rows if what == opcodes[0])
        named = sum(f"no row for: {opcodes[0]} " in why for why in refusals)
        if not has_row:
            # Every trace that met the import was refused for it.
            self.assertEqual(met, named, msg=str(refusals))
            return
        line = source.splitlines().index("    " + construct.splitlines()[0]) + 1
        runs = sum(
            r["runs"]
            for r in vm.regions(compiled)
            if r["function"] == "f"
            and r["state"] == "region"
            and r["lines"][0] <= line <= r["lines"][1]
        )
        self.assertGreater(met, named, msg=str(refusals))
        self.assertGreater(runs, 0, msg=str(vm.regions(compiled)))

    @checks_isolated
    def test_an_import_while_its_module_is_reloaded(self):
        """importlib.reload gives the module a new spec, which has no
        ``_initializing``, and runs the module's code with the module in
        ``sys.modules``: an import of it from that code finds it at once."""
        setup = IMPORTS_FILE.format("vm_imp_during").replace(
            "K = sys.vm_imp_k", "K = sys.vm_imp_k\\nsys.vm_imp_body()\\nK += 1.0"
        )
        construct = "import vm_imp_during\nk = vm_imp_during.K"
        source = rules_source(
            "during", {"during": ((), setup, construct, "pass", "None")}
        )
        space = {}
        exec(compile(source, "<imports during>", "exec"), space)
        compiled = vm(space["f"], passes=CHECKS_EARLY)

        def calls(fn):
            """What 13 calls give, the module reloaded before calls 6 and
            10, and what six calls from the module's code give meanwhile."""
            seen, during = [], []
            sys.vm_imp_k = 2.0
            sys.vm_imp_body = lambda: None
            sys.modules.pop("vm_imp_during", None)
            module = importlib.import_module("vm_imp_during")
            for call in range(13):
                torch.manual_seed(call)
                x = torch.randn(4, 8)
                if call in (6, 10):
                    sys.vm_imp_k = float(call)
                    sys.vm_imp_body = lambda x=x: seen.extend(fn(x) for _ in range(6))
                    before = checks_region_runs(compiled, space["f"])
                    importlib.reload(module)
                    during.append(checks_region_runs(compiled, space["f"]) - before)
                    sys.vm_imp_body = lambda: None
                seen.append(fn(x))
            return seen, during

        want, _ = calls(space["f"])
        got, during = calls(compiled)
        self.assertEqual(len(got), len(want))
        for at, (a, b) in enumerate(zip(got, want)):
            self.assertEqual(a, b, atol=0, rtol=0, msg=f"value {at}")
        # A region ran over the import while the module was being reloaded.
        self.assertGreater(during[0], 0, msg=str(vm.regions(compiled)))


if CHECKS_HAS_VM:
    from torch._rust_inductor import lower as _lw


def _lw_keys(tensors):
    return [x if type(x) is float else _lw.key_of(x) for x in tensors]


def _lw_key_tree(x):
    if isinstance(x, torch.Tensor):
        return _lw.key_of(x)
    return type(x)(_lw_key_tree(i) for i in x) if type(x) in (tuple, list) else x


def _lw_copy(x):
    if not isinstance(x, torch.Tensor):
        return x
    if type(x) is torch.nn.Parameter:
        return torch.nn.Parameter(x.detach().clone(), requires_grad=x.requires_grad)
    return x.detach().clone().requires_grad_(x.requires_grad)


def _lw_leaves(t):
    """The templates under ``t`` that are no container."""
    if t[0] not in ("seq", "map", "slice"):
        return [t]
    parts = t[2] if t[0] != "slice" else t[1:]
    return [leaf for part in parts for leaf in _lw_leaves(part)]


def _lw_stated(spec, inputs=(), pins=()):
    """``spec``, written with what a test is about, in the form a list has:
    every operand pinned to its class (an input, a pin and a constant to the
    class of the object, torch.Tensor where none is given; a result to what
    its entry asserts), every kept result asserted to be of a class
    (torch.Tensor if the entry names none), and every slot that no entry
    moves or drops and no output names dropped behind the last entry."""
    ninputs, npins, consts, entries, outputs = spec
    asserted, live, stated = {}, set(), []
    given = {"in": inputs, "pin": pins, "const": consts}

    def pinned(t):
        kind = t[0]
        if kind in ("seq", "map", "slice"):
            return tuple(pinned(part) for part in (t[2] if kind != "slice" else t[1:]))
        if kind in given:
            return type(given[kind][t[1]]) if t[1] < len(given[kind]) else torch.Tensor
        if kind == "move":
            live.discard(t[1])
        if kind in ("res", "move"):
            return asserted.get(t[1], torch.Tensor)
        return None if kind == "null" else torch.Tensor

    for k, e in enumerate(entries):
        if e[0] == "drop":
            live.difference_update(e[1])
            stated.append(e)
            continue
        callee, args, kwnames, out, at, flags, asserts, _ = e
        classes = tuple(pinned(t) for t in args)
        if out is not None:
            named = [
                a[1] if a[0] == "class" else type(a[1])
                for a in asserts
                if a[0] in ("class", "key")
            ]
            asserted[k] = named[0] if named else torch.Tensor
            asserts = asserts if named else (*asserts, ("class", torch.Tensor))
            live.add(k)
        stated.append((callee, args, kwnames, out, at, flags, asserts, classes))
    kept = {t[1] for o in outputs for t in _lw_leaves(o) if t[0] in ("res", "part")}
    if live - kept:
        stated.append(_cl_drop(*sorted(live - kept)))
    return (ninputs, npins, consts, stated, outputs)


def _lw_lower(spec, inputs, pins=(), facts=None):
    """The list of ``spec`` (stated), and its lowering from the keys of
    ``inputs`` and ``pins`` and from what one real run of the list on copies
    of them left (``facts``: a function of that, to say something else)."""
    spec = _lw_stated(spec, inputs, pins)
    h = _calls_make(spec)
    a, b = [_lw_copy(x) for x in inputs], [_lw_copy(x) for x in pins]
    before = _lw_keys(a), _lw_keys(b)
    out = _calls_run(h, tuple(a), tuple(b))
    left = _lw.Traced(
        tuple(_lw_key_tree(o) for o in out), tuple(_lw_keys(a)), tuple(_lw_keys(b))
    )
    return h, _lw.lower(h, spec, *before, left if facts is None else facts(left))


def _lw_modes():
    return (
        torch._C._len_torch_dispatch_stack(),
        torch._C._len_torch_function_stack(),
    )


@unittest.skipUnless(CHECKS_HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmLower(TestCase):
    """The lowering makes the aten-level graph of a call list from a
    data-free run of the list. The rule: a graph it hands out computes what
    the list computes (results bit for bit, and every tensor the list
    writes), and a list it does not understand in full, or whose data-free
    run is not what the traced run was, gets no graph but a reason that
    names the entry.
    """

    hw_classification = HardwareClassification.ACCELERATOR

    def lw_agrees(self, spec, make, calls=None):
        """Lowers ``spec`` for the tensors of ``make(0)`` and holds the graph
        against the list on those of three other seeds."""
        inputs, pins = make(0)
        h, low = _lw_lower(spec, inputs, pins)
        self.assertIsInstance(low, _lw.Lowered, msg=getattr(low, "reason", ""))
        self.assertEqual(_lw_modes(), (0, 0))
        for seed in (1, 2, 3):
            mine, theirs = make(seed), make(seed)
            torch.manual_seed(seed)
            want = _calls_run(h, tuple(theirs[0]), tuple(theirs[1]))
            torch.manual_seed(seed)
            got = low.gm(*mine[0], *mine[1], *low.constants)
            self.assertEqual(_wt_bits(got), _wt_bits(want))
            self.assertEqual(_wt_bits(mine), _wt_bits(theirs))
        return low

    def lw_tensors(self, device, *shapes, pins=(), grad=False):
        def make(seed):
            gen = torch.Generator().manual_seed(seed)
            inputs = [torch.randn(*s, generator=gen).to(device) for s in shapes]
            made = [torch.randn(*s, generator=gen).to(device) for s in pins]
            return inputs, [torch.nn.Parameter(p, requires_grad=grad) for p in made]

        return make

    def test_in_place_operators(self, device):
        spec = (2, 1, [0.5], [
            _cl_entry(("binary", "*"), [("in", 0), ("pin", 0)], out=0, flags=["pure", "fresh"]),
            _cl_entry(("binary", "+="), [("res", 0), ("in", 1)], out=1, flags=[("writes", (0,))]),
            _cl_entry(_cl_vector(torch.Tensor.mul_), [("res", 1), ("const", 0)], out=2, flags=[("writes", (0,))]),
            _cl_entry(("binary", "-="), [("in", 1), ("res", 2)], out=3, flags=[("writes", (0,))]),
            _cl_entry(_cl_vector(torch.relu_), [("in", 0)], flags=[("writes", (0,))]),
        ], [("res", 2), ("res", 3)])  # fmt: skip
        low = self.lw_agrees(spec, self.lw_tensors(device, (3, 4), (3, 4), pins=[(4,)]))
        self.assertEqual(low.list_facts, {"writes_input"})
        self.assertEqual(
            [n.name for n in low.gm.graph.find_nodes(op="placeholder")],
            ["arg0_1", "arg1_1", "const_0"],
        )
        # The second output is the input the list wrote, the object itself.
        (x, y), (w,) = self.lw_tensors(device, (3, 4), (3, 4), pins=[(4,)])(5)
        self.assertIs(low.gm(x, y, w)[1], y)

    def test_a_view_chain_written_through(self, device):
        tail = ("slice", ("const", 1), ("null",), ("null",))
        spec = (1, 0, [2, -1, 0, 1, 1.0], [
            _cl_entry(_cl_vector(torch.Tensor.view), [("in", 0), ("const", 0), ("const", 1)], out=0, flags=["pure"]),
            _cl_entry(_cl_vector(torch.Tensor.transpose), [("res", 0), ("const", 2), ("const", 3)], out=1, flags=["pure"]),
            _cl_entry(("getitem",), [("res", 1), tail], out=2, flags=["pure"]),
            _cl_entry(_cl_vector(torch.Tensor.add_), [("res", 2), ("const", 4)], out=3, flags=[("writes", (0,))]),
            _cl_entry(("getattr", "T"), [("res", 0)], out=4, flags=["pure"]),
        ], [("res", 2), ("res", 4)])  # fmt: skip
        low = self.lw_agrees(spec, self.lw_tensors(device, (4, 3)))
        # Written: a result that is not known to be over memory of its own.
        self.assertEqual(low.list_facts, {"writes_input"})

    def test_cat_of_a_list_and_keywords(self, device):
        both = ("seq", "list", [("in", 0), ("in", 1)])
        twice = ("seq", "tuple", [("res", 0), ("res", 0)])
        spec = (2, 0, [1, -1, True], [
            _cl_entry(_cl_vector(torch.cat), [both, ("const", 0)], ("dim",), out=0, flags=["pure", "fresh"]),
            _cl_entry(_cl_vector(torch.stack), [twice], out=1, flags=["pure", "fresh"]),
            _cl_entry(_cl_vector(torch.Tensor.softmax), [("res", 1), ("const", 1)], ("dim",), out=2, flags=["pure", "fresh"]),
            _cl_entry(_cl_vector(torch.sum), [("res", 2), ("const", 0), ("const", 2)], ("dim", "keepdim"), out=3, flags=["pure", "fresh"]),
        ], [("res", 3), ("res", 0)])  # fmt: skip
        low = self.lw_agrees(spec, self.lw_tensors(device, (3, 2), (3, 5)))
        self.assertEqual(low.list_facts, frozenset())

    def test_a_tuple_result_and_its_parts(self, device):
        pair = ("seq", "list", [("in", 0), ("res", 1)])
        spec = (1, 0, [1, 2.0], [
            _cl_entry(_cl_vector(torch.max), [("in", 0), ("const", 0)], out=0, flags=["pure"], asserts=[("class", torch.return_types.max), ("count", 2)]),
            _cl_entry(("binary", "*"), [("part", 0, (0,)), ("part", 0, (1,))], out=1, flags=["pure", "fresh"]),
            _cl_entry(_cl_vector(torch._foreach_mul), [pair, ("const", 1)], out=2, flags=["pure"], asserts=[("class", tuple)]),
        ], [("part", 0, (1,)), ("res", 1), ("res", 2), ("part", 2, (0,))])  # fmt: skip
        self.lw_agrees(spec, self.lw_tensors(device, (4, 4)))

    def test_dropout_in_training(self, device):
        spec = (1, 1, [0.5, True], [
            _cl_entry(_cl_vector(torch.dropout), [("in", 0), ("const", 0), ("const", 1)], out=0, flags=["draws", "fresh"]),
            _cl_entry(("binary", "*"), [("res", 0), ("pin", 0)], out=1, flags=["pure", "fresh"]),
        ], [("res", 1)])  # fmt: skip
        low = self.lw_agrees(spec, self.lw_tensors(device, (8, 8), pins=[(8,)]))
        self.assertEqual(low.list_facts, {"draws"})

    def test_items_comparison_and_negation(self, device):
        column = (
            "seq",
            "tuple",
            [("slice", ("null",), ("null",), ("null",)), ("const", 0)],
        )
        spec = (2, 0, [1], [
            _cl_entry(("getitem",), [("in", 1), column], out=0, flags=["pure"]),
            _cl_entry(("setitem",), [("res", 0), ("in", 0), column], flags=[("writes", (1,))]),
            _cl_entry(("unary", "-"), [("in", 0)], out=2, flags=["pure", "fresh"]),
            _cl_entry(("compare", "<"), [("res", 2), ("in", 1)], out=3, flags=["pure", "fresh"]),
            _cl_entry(_cl_vector(torch.where), [("res", 3), ("in", 0), ("in", 1)], out=4, flags=["pure", "fresh"]),
        ], [("res", 3), ("res", 4)])  # fmt: skip
        low = self.lw_agrees(spec, self.lw_tensors(device, (3, 4), (3, 4)))
        self.assertEqual(low.list_facts, {"writes_input"})

    def test_gradients_are_asked_of_the_graph_as_of_the_list(self, device):
        spec = (1, 2, [], [
            _cl_entry(_cl_vector(torch._C._nn.linear), [("in", 0), ("pin", 0), ("pin", 1)], out=0, flags=["pure", "fresh"]),
            _cl_entry(_cl_vector(torch.relu_), [("res", 0)], out=1, flags=[("writes", (0,))]),
            _cl_entry(_cl_vector(torch.Tensor.sum), [("res", 1)], out=2, flags=["pure", "fresh"]),
        ], [("res", 2)])  # fmt: skip
        make = self.lw_tensors(device, (5, 4), pins=[(3, 4), (3,)], grad=True)
        low = self.lw_agrees(spec, make)
        h = _calls_make(_lw_stated(spec, *make(0)))
        (x,), mine = make(7)
        _, theirs = make(7)
        low.gm(x, *mine)[0].backward()
        _calls_run(h, (x,), tuple(theirs))[0].backward()
        self.assertEqual(
            _wt_bits([p.grad for p in mine]), _wt_bits([p.grad for p in theirs])
        )

    def test_a_float_input_is_a_constant_of_the_graph(self, device):
        spec = (3, 0, [], [
            _cl_entry(("binary", "*"), [("in", 0), ("in", 1)], out=0, flags=["pure", "fresh"]),
            _cl_entry(_cl_vector(torch.add), [("res", 0), ("in", 0), ("in", 1)], ("alpha",), out=1, flags=["pure", "fresh"]),
        ], [("res", 1)])  # fmt: skip
        make = self.lw_tensors(device, (3, 4))
        low = self.lw_agrees(spec, lambda seed: ([*make(seed)[0], 2.5, 7.0], []))
        # The float the list uses is named; the one it does not use is not.
        self.assertEqual(low.specialized, (1,))
        (x,), _ = make(4)
        self.assertEqual(_wt_bits(low.gm(x, 3.5, 7.0)), _wt_bits(low.gm(x, 2.5, 7.0)))

    def test_a_tensor_made_of_constants_is_a_trailing_placeholder(self, device):
        spec = (1, 1, [2.0, device], [
            _cl_entry(_cl_vector(torch.tensor), [("const", 0), ("const", 1)], ("device",), out=0, flags=["pure", "fresh"]),
            _cl_entry(("binary", "*"), [("in", 0), ("res", 0)], out=1, flags=["pure", "fresh"]),
            _cl_entry(("binary", "+"), [("res", 1), ("pin", 0)], out=2, flags=["pure", "fresh"]),
        ], [("res", 2)])  # fmt: skip
        low = self.lw_agrees(spec, self.lw_tensors(device, (3, 4), pins=[(4,)]))
        self.assertEqual(_wt_bits(list(low.constants)), _wt_bits([torch.tensor(2.0)]))
        self.assertEqual(
            [n.name for n in low.gm.graph.find_nodes(op="placeholder")],
            ["arg0_1", "const_0", "const_1"],
        )
        self.assertEqual(list(low.gm.graph.find_nodes(op="get_attr")), [])

    def lw_refused(self, spec, inputs, pins=(), facts=None):
        h, low = _lw_lower(spec, inputs, pins, facts)
        self.assertIsInstance(low, _lw.NotLowerable)
        self.assertEqual(_lw_modes(), (0, 0))
        return low

    def test_what_has_no_graph_and_why(self, device):
        x = torch.randn(3, 4, device=device)
        one = torch.randn((), device=device)

        def single(
            callee, args, flags=("pure",), out=0, consts=(), outputs=None, cls=None
        ):
            asserts = [("class", cls)] if cls else []
            entries = [_cl_entry(callee, args, out=out, flags=flags, asserts=asserts)]
            return (
                2,
                0,
                list(consts),
                entries,
                [("in", 0)] if outputs is None else outputs,
            )

        add = _cl_vector(torch.add)
        ops = [("in", 0), ("in", 0)]
        cases = {
            "an opaque call": single(add, ops, ["opaque"]),
            "an effect": single(add, ops, ["effect"]),
            "changes a tensor's key in place": single(_cl_vector(torch.Tensor.t_), ops[:1], [("writes", (0,)), ("rekeys", (0,))]),
            "no flag says what the entry does": single(add, ops, []),
            "a 'setattr' entry is not lowered": single(("setattr", "requires_grad"), [("const", 0), ("in", 0)], [("writes", (1,))], None, [True]),
            "a 'delitem' entry is not lowered": single(("delitem",), [("in", 0), ("const", 0)], [("writes", (0,))], None, [0]),
            "a read of .data": single(("getattr", "data"), ops[:1]),
            "depends on a tensor's data (DataDependentOutputException": single(_cl_vector(torch.Tensor.is_nonzero), [("in", 1)], cls=bool),
            "depends on a tensor's data (DynamicOutputShapeException": single(_cl_vector(torch.nonzero), ops[:1]),
            "it has no fake kernel": single(_cl_vector(torch.Tensor.to_sparse), ops[:1]),
            "Please convert all Tensors to FakeTensors": single(add, [("in", 0), ("const", 0)], consts=[torch.ones(4, device=device)]),
            "a part of a result is not there": single(_cl_vector(torch.max), [("in", 0), ("const", 0)], ["pure", "fresh"], consts=[1], outputs=[("part", 0, (2,))], cls=torch.return_types.max),
            "are not compared": single(add, ops, consts=[1.5], outputs=[("const", 0)]),
            "differ in what the tensor is": single(add, ops, consts=[torch.ones(2)], outputs=[("const", 0)]),
        }  # fmt: skip
        for why, spec in cases.items():
            with self.subTest(why=why):
                # The list itself runs (but for the two that raise as
                # written): what is refused is its graph.
                if "a part" in why or "delitem" in why:
                    keys = _lw_keys([x, one])
                    left = _lw.Traced((keys[0],), tuple(keys), ())
                    spec = _lw_stated(spec, (x, one))
                    low = _lw.lower(_calls_make(spec), spec, keys, [], left)
                else:
                    low = self.lw_refused(spec, (x, one))
                self.assertIn(why, low.reason)
                no_entry = ("compared", "what the tensor is", "a part")
                at_output = any(text in why for text in no_entry)
                self.assertEqual(low.entry, None if at_output else 0)
                self.assertEqual(_lw_modes(), (0, 0))
        self.assertEqual(
            self.lw_refused(cases["an effect"], (x, one)).list_facts, {"effects"}
        )
        self.assertEqual(
            self.lw_refused(cases["a read of .data"], (x, one)).list_facts,
            {"data_attr"},
        )
        reads = "depends on a tensor's data (DataDependentOutputException"
        self.assertEqual(
            self.lw_refused(cases[reads], (x, one)).list_facts, {"reads_value"}
        )

    def test_keys_that_are_not_lowered(self, device):
        x = torch.randn(3, 4, device=device)
        spec = _lw_stated(
            (
                1,
                0,
                [],
                [
                    _cl_entry(
                        _cl_vector(torch.add),
                        [("in", 0), ("in", 0)],
                        out=0,
                        flags=["pure"],
                    )
                ],
                [("res", 0)],
            )
        )
        h = _calls_make(spec)
        key = _lw.key_of(x)
        out = _lw.Traced((key,), (key,), ())
        self.assertIsInstance(_lw.lower(h, spec, [key], [], out), _lw.Lowered)
        for why, given in {
            "has no key": None,
            "conjugate": key._replace(conj=True),
            "negative bit": key._replace(neg=True),
            "inference tensor": key._replace(inference=True),
            "has no key (": (3, 4),
        }.items():
            with self.subTest(why=why):
                low = _lw.lower(h, spec, [given], [], out)
                self.assertIsInstance(low, _lw.NotLowerable)
                self.assertIn(why, low.reason)
        # What has a key: the exact classes, dense and strided.
        self.assertIsNone(_lw.key_of(x.to_sparse()))
        self.assertIsNone(_lw.key_of(x.as_subclass(KeySubclass)))
        self.assertIsNone(_lw.key_of(2.5))
        for count in ([key, key], []):
            low = _lw.lower(h, spec, count, [], out)
            self.assertIn("not as many", low.reason)
        low = _lw.lower(h, spec, [key], [], _lw.Traced((key, key), (key,), ()))
        self.assertIn("not as many", low.reason)
        with TorchFunctionMode():
            low = _lw.lower(h, spec, [key], [], out)
        self.assertIn("a function mode is on", low.reason)
        # A fake tensor that requires grad is a leaf: the key does not say
        # whether the traced tensor was one, and a leaf is not written.
        spec = _lw_stated(
            (
                1,
                0,
                [],
                [
                    _cl_entry(
                        _cl_vector(torch.relu_),
                        [("in", 0)],
                        out=0,
                        flags=[("writes", (0,))],
                    )
                ],
                [("res", 0)],
            )
        )
        key = key._replace(requires_grad=True)
        low = _lw.lower(
            _calls_make(spec), spec, [key], [], _lw.Traced((key,), (key,), ())
        )
        self.assertIn("a leaf Variable that requires grad", low.reason)
        self.assertEqual(_lw_modes(), (0, 0))

    def test_no_mode_is_left_on_a_stack(self, device):
        x = torch.randn(3, 4, device=device)
        for push, pop, mode in (
            (torch._C._push_on_torch_dispatch_stack, lambda: torch._C._pop_torch_dispatch_stack(None), TorchDispatchMode()),
            (torch._C._push_on_torch_function_stack, torch._C._pop_torch_function_stack, TorchFunctionMode()),
        ):  # fmt: skip
            with self.subTest(push=push.__name__):
                # A list that pushes a mode and goes on: the next operator
                # fails under it, or the run ends with a mode on the stack.
                low, left = None, None
                spec = _lw_stated((1, 0, [mode], [
                    _cl_entry(_cl_vector(push), [("const", 0)], flags=["pure"]),
                    _cl_entry(_cl_vector(torch.add), [("in", 0), ("in", 0)], out=1, flags=["pure"]),
                ], [("res", 1)]))  # fmt: skip
                key = _lw.key_of(x)
                try:
                    low = _lw.lower(
                        _calls_make(spec),
                        spec,
                        [key],
                        [],
                        _lw.Traced((key,), (key,), ()),
                    )
                    left = _lw_modes()
                finally:
                    while _lw_modes() != (0, 0):
                        pop()
                self.assertEqual(left, (0, 0))
                self.assertIsInstance(low, _lw.NotLowerable)

    def test_a_run_that_is_not_the_traced_one_has_no_graph(self, device):
        # L2. The traced run made float64 (the default dtype then); the
        # data-free run, under another default, makes float32.
        spec = _lw_stated((1, 0, [3, device], [
            _cl_entry(_cl_vector(torch.zeros), [("const", 0), ("const", 1)], ("device",), out=0, flags=["pure", "fresh"]),
            _cl_entry(("binary", "+"), [("res", 0), ("in", 0)], out=1, flags=["pure", "fresh"]),
        ], [("res", 0), ("res", 1)]))  # fmt: skip
        x = torch.randn(3, device=device)
        h = _calls_make(spec)
        torch.set_default_dtype(torch.float64)
        try:
            out = _calls_run(h, (x,), ())
        finally:
            torch.set_default_dtype(torch.float32)
        key = _lw.key_of(x)
        left = _lw.Traced(tuple(map(_lw.key_of, out)), (key,), ())
        low = _lw.lower(h, spec, [key], [], left)
        self.assertIsInstance(low, _lw.NotLowerable)
        self.assertEqual(
            low.reason, "output 0: the data-free run and the traced run differ in dtype"
        )
        torch.set_default_dtype(torch.float64)
        try:
            low = _lw.lower(h, spec, [key], [], left)
        finally:
            torch.set_default_dtype(torch.float32)
        self.assertIsInstance(low, _lw.Lowered)

    def test_every_field_of_every_recorded_key_is_compared(self, device):
        spec = (1, 1, [], [
            _cl_entry(_cl_vector(torch.Tensor.add_), [("in", 0), ("pin", 0)], out=0, flags=[("writes", (0,))]),
            _cl_entry(_cl_vector(torch.Tensor.t), [("pin", 0)], out=1, flags=["pure"]),
            _cl_entry(_cl_vector(torch._foreach_neg), [("seq", "list", [("res", 1)])], out=2, flags=["pure"], asserts=[("class", tuple)]),
        ], [("res", 0), ("pin", 0), ("res", 2)])  # fmt: skip
        x = torch.randn(4, 4, device=device)
        w = torch.nn.Parameter(torch.randn(4, 4, device=device))
        self.assertIsInstance(_lw_lower(spec, [x], [w])[1], _lw.Lowered)
        other = {
            "parameter": lambda k: not k.parameter,
            "dtype": lambda k: torch.float64,
            "device": lambda k: torch.device("meta"),
            "sizes": lambda k: (4, 5),
            "strides": lambda k: (1, 4) if k.strides == (4, 1) else (4, 1),
            "requires_grad": lambda k: not k.requires_grad,
            "conj": lambda k: True,
            "neg": lambda k: True,
            "inference": lambda k: True,
        }
        self.assertEqual(set(other), set(_lw.Key._fields))

        def changed(where, field):
            def facts(left):
                trees = {
                    "outputs": list(left.outputs),
                    "inputs": list(left.inputs),
                    "pins": list(left.pins),
                }
                group, i = where
                key = trees[group][i]
                inner = type(key) is not _lw.Key
                key = key[0] if inner else key
                new = key._replace(**{field: other[field](key)})
                trees[group][i] = type(trees[group][i])([new]) if inner else new
                return _lw.Traced(
                    *(tuple(trees[g]) for g in ("outputs", "inputs", "pins"))
                )

            return facts

        places = [
            ("outputs", 0),
            ("outputs", 1),
            ("outputs", 2),
            ("inputs", 0),
            ("pins", 0),
        ]
        for where in places:
            for field in _lw.Key._fields:
                with self.subTest(where=where, field=field):
                    low = self.lw_refused(spec, [x], [w], changed(where, field))
                    self.assertIn(f"differ in {field}", low.reason)
        # The shape of what is handed out: a list is no tuple, None no tensor.
        for facts in (
            lambda left: left._replace(
                outputs=(
                    *left.outputs[:2],
                    (list if type(left.outputs[2]) is tuple else tuple)(
                        left.outputs[2]
                    ),
                )
            ),
            lambda left: left._replace(
                outputs=(*left.outputs[:2], [*left.outputs[2], left.outputs[2][0]])
            ),
            lambda left: left._replace(outputs=(None, *left.outputs[1:])),
            lambda left: left._replace(inputs=(None,)),
        ):
            self.lw_refused(spec, [x], [w], facts)

    def test_what_an_assert_says_is_compared(self, device):
        x = torch.randn(3, 4, device=device)
        wide = torch.randn(3, 4, device=device, dtype=torch.float64)
        T, P = torch.Tensor, torch.nn.Parameter

        def spec(callee, flags, *asserts):
            args = [("in", 0), ("const", 0)]
            entry = _cl_entry(callee, args, out=0, flags=flags, asserts=asserts)
            return _lw_stated((1, 0, [1], [entry], [("in", 0)]))

        add, add_, top = (
            _cl_vector(torch.add),
            _cl_vector(torch.Tensor.add_),
            _cl_vector(torch.max),
        )
        pure, writes = ["pure"], [("writes", (0,))]
        pair = ("class", torch.return_types.max)
        holds = [
            spec(add, pure, ("key", x)),
            spec(add, pure, ("class", T)),
            spec(add_, writes, ("is", 0), ("class", T)),
            spec(top, pure, ("count", 2), pair),
        ]
        fails = [
            spec(add, pure, ("key", wide)),
            spec(add, pure, ("key", x[0])),
            spec(add, pure, ("key", x.to("meta"))),
            spec(add, pure, ("key", x.clone().requires_grad_())),
            spec(add, pure, ("key", torch.randn(4, 3, device=device).t())),
            spec(add, pure, ("key", P(x, requires_grad=False))),
            spec(add, pure, ("class", P)),
            spec(add, pure, pair),
            spec(add, pure, ("class", float)),
            spec(top, pure, ("class", T)),
            spec(add, pure, ("is", 0), ("class", T)),
            spec(add_, writes, ("is", 1), ("class", T)),
            spec(top, pure, ("count", 3), pair),
            spec(add, pure, ("count", 2)),
        ]
        key = _lw.key_of(x)
        out = _lw.Traced((key,), (key,), ())
        for s in holds:
            low = _lw.lower(_calls_make(s), s, [key], [], out)
            self.assertIsInstance(low, _lw.Lowered, msg=getattr(low, "reason", ""))
        for k, s in enumerate(fails):
            with self.subTest(k=k):
                low = _lw.lower(_calls_make(s), s, [key], [], out)
                self.assertIsInstance(low, _lw.NotLowerable)
                self.assertEqual(low.entry, 0)
                self.assertEqual(_lw_modes(), (0, 0))

    def test_the_class_an_operand_is_pinned_to_is_compared(self, device):
        # A tensor is a fake tensor in the data-free run, whatever its class:
        # its pin is held against the class its recorded key names. What is
        # no tensor is the object itself there, and must be of its pin.
        x = torch.randn(3, 4, device=device)
        w = torch.nn.Parameter(torch.randn(3, 4, device=device), requires_grad=False)
        T, P = torch.Tensor, torch.nn.Parameter
        kx, kw = _lw.key_of(x), _lw.key_of(w)

        def spec(result=P, pin=P, tensor=T, number=float):
            return (2, 1, [], [
                _cl_entry(_cl_vector(torch.Tensor.add_), [("pin", 0), ("in", 0)], out=0, flags=[("writes", (0,))], asserts=[("class", result), ("is", 0)], classes=(pin, tensor)),
                _cl_entry(_cl_vector(torch.mul), [("res", 0), ("in", 1)], out=1, flags=["pure", "fresh"], asserts=[("class", T)], classes=(result, number)),
                _cl_drop(0),
            ], [("res", 1)])  # fmt: skip

        def low(s, keys=(kx, 2.5), pins=(kw,)):
            left = _lw.Traced((kx,), tuple(keys), tuple(pins))
            return _lw.lower(_calls_make(s), s, list(keys), list(pins), left)

        good = spec()
        graph = low(good)
        self.assertIsInstance(graph, _lw.Lowered, msg=getattr(graph, "reason", ""))
        h = _calls_make(good)
        mine, theirs = [_lw_copy(x), _lw_copy(w)], [_lw_copy(x), _lw_copy(w)]
        want = _calls_run(h, (theirs[0], 2.5), (theirs[1],))
        self.assertEqual(_wt_bits(graph.gm(mine[0], 2.5, mine[1])), _wt_bits(want))
        self.assertEqual(_wt_bits(mine), _wt_bits(theirs))
        for why, entry, refused in (
            ("an operand is pinned to be a Tensor", 0, low(spec(pin=T))),
            ("an operand is pinned to be a Parameter", 0, low(spec(tensor=P))),
            ("the result is asserted to be a Tensor", 0, low(spec(result=T))),
            ("is not of the class it is pinned to", 1, low(spec(number=int))),
            ("is not of the class it is pinned to", 1, low(spec(number=T))),
            # The same pins, and keys that name the other classes.
            ("an operand is pinned to be a Parameter", 0, low(good, pins=(kx,))),
            ("an operand is pinned to be a Tensor", 0, low(good, keys=(kw, 2.5))),
        ):
            with self.subTest(why=why):
                self.assertIsInstance(refused, _lw.NotLowerable)
                self.assertIn(why, refused.reason)
                self.assertEqual(refused.entry, entry)
                self.assertEqual(_lw_modes(), (0, 0))

    def test_where_a_list_releases_is_nothing_in_its_graph(self, device):
        # a = x * w; b = relu_(a); c = b + y; d = stack([c, b]); d.sum(0), c
        mul, relu_ = _cl_vector(torch.mul), _cl_vector(torch.relu_)
        add, stack = _cl_vector(torch.add), _cl_vector(torch.stack)
        total = _cl_vector(torch.Tensor.sum)
        made, wrote, same = ["pure", "fresh"], [("writes", (0,))], [("is", 0)]
        at_the_end = (2, 1, [0], [
            _cl_entry(mul, [("in", 0), ("pin", 0)], out=0, flags=made),
            _cl_entry(relu_, [("res", 0)], out=1, flags=wrote, asserts=same),
            _cl_entry(add, [("res", 1), ("in", 1)], out=2, flags=made),
            _cl_entry(stack, [("seq", "list", [("res", 2), ("res", 1)])], out=3, flags=made),
            _cl_entry(total, [("res", 3), ("const", 0)], out=4, flags=made),
        ], [("res", 4), ("res", 2)])  # fmt: skip
        moved = (2, 1, [0], [
            _cl_entry(mul, [("in", 0), ("pin", 0)], out=0, flags=made),
            _cl_entry(relu_, [("move", 0)], out=1, flags=wrote, asserts=same),
            _cl_entry(add, [("res", 1), ("in", 1)], out=2, flags=made),
            _cl_entry(stack, [("seq", "list", [("res", 2), ("move", 1)])], out=3, flags=made),
            _cl_entry(total, [("move", 3), ("const", 0)], out=4, flags=made),
        ], [("res", 4), ("res", 2)])  # fmt: skip
        dropped = (2, 1, [0], [
            _cl_entry(mul, [("in", 0), ("pin", 0)], out=0, flags=made),
            _cl_entry(relu_, [("res", 0)], out=1, flags=wrote, asserts=same),
            _cl_drop(0),
            _cl_entry(add, [("res", 1), ("in", 1)], out=3, flags=made),
            _cl_entry(stack, [("seq", "list", [("res", 3), ("res", 1)])], out=4, flags=made),
            _cl_drop(1),
            _cl_entry(total, [("res", 4), ("const", 0)], out=6, flags=made),
            _cl_drop(4),
        ], [("res", 6), ("res", 3)])  # fmt: skip
        make = self.lw_tensors(device, (3, 4), (3, 4), pins=[(4,)])
        graphs = [self.lw_agrees(spec, make) for spec in (at_the_end, moved, dropped)]
        self.assertEqual(len({low.gm.code for low in graphs}), 1)
        self.assertEqual({low.list_facts for low in graphs}, {frozenset()})
        # The stated lists do release where they say.
        stated = [_lw_stated(spec, *make(0)) for spec in (moved, dropped)]
        texts = [_calls_text(_calls_make(spec)) for spec in stated]
        self.assertEqual(texts[0].count("move r"), 3)
        self.assertEqual(texts[1].count("drop r"), 3)

    def test_a_release_that_could_be_seen_has_no_graph(self, device):
        # A graph has no place for a release. Of a tensor of the run, a plain
        # value and a tuple of those nothing sees one; of any other object
        # (a weak reference to it, its class's deallocator) that is not
        # known, and the list keeps its releases by having no graph.
        x = torch.randn(3, 4, device=device)
        x0, log = ("in", 0), []

        class Finalized:
            def __del__(self):
                log.append("released")

        # A finalizer, a weak reference, a class that takes weak references.
        seen = [
            (object.__new__, ("const", 0), Finalized),
            (weakref.ref, x0, weakref.ReferenceType),
            (set, None, set),
        ]
        unseen = [
            (torch.Tensor.dim, x0, int),
            (torch.Tensor.unbind, x0, tuple),
            (torch.Tensor.size, x0, torch.Size),
            (torch.Tensor.sort, x0, torch.return_types.sort),
            (torch.Tensor.neg, x0, torch.Tensor),
            (list, ("seq", "tuple", [x0]), list),
        ]

        def spec(case, release):
            callee, operand, cls = case
            out = None if release == "not kept" else 0
            asserts = [("class", cls)] if out == 0 else []
            first = _cl_entry(_cl_vector(callee), [operand] if operand else [], out=out, flags=["pure"], asserts=asserts)  # fmt: skip
            then = {
                "dropped": [_cl_drop(0)],
                "moved": [_cl_entry(_cl_vector(id), [("move", 0)], flags=["pure"])],
            }.get(release, [])
            out = [("res", 0)] if release == "handed out" else [("in", 0)]
            return (1, 0, [Finalized], [first, *then], out)

        for release in ("dropped", "moved", "not kept"):
            for case in seen:
                with self.subTest(release=release, cls=case[2].__name__):
                    low = self.lw_refused(spec(case, release), [x])
                    said = f"the list releases its result, a {case[2].__name__}"
                    self.assertIn(said, low.reason)
                    self.assertEqual(low.entry, 0)
            for case in unseen:
                with self.subTest(release=release, cls=case[2].__name__):
                    low = _lw_lower(spec(case, release), [x])[1]
                    self.assertIsInstance(
                        low, _lw.Lowered, msg=getattr(low, "reason", "")
                    )
        # Handed out, a release is the caller's: a list of tensors is a form
        # of output. The others are not, so they have no graph either way.
        low = _lw_lower(spec(unseen[-1], "handed out"), [x])[1]
        self.assertIsInstance(low, _lw.Lowered, msg=getattr(low, "reason", ""))
        self.assertIs(low.gm(x)[0][0], x)
        for case in seen:
            self.lw_refused(spec(case, "handed out"), [x])
        # No fact says "its releases can be seen": no pass is given such a list.
        self.assertNotIn("release", " ".join(_lw.FACTS))

    def test_what_is_not_understood_has_no_graph(self, device):
        x = torch.randn(3, 4, device=device)
        key = _lw.key_of(x)
        out = _lw.Traced((key,), (key,), ())
        T = torch.Tensor

        def spec(
            callee=None, args=None, flags=("pure",), fields=8, top=5, more=(), **given
        ):
            given = {"asserts": [("class", T)], **given}
            entry = _cl_entry(
                callee or _cl_vector(torch.add),
                args or [("in", 0), ("in", 0)],
                out=0,
                flags=flags,
                **given,
            )
            entry = (*entry, *[()] * (fields - 8))[:fields]
            whole = (1, 0, [], [entry, *more], [("res", 0)]) + (None,) * (top - 5)
            return whole[:top]

        known = spec()
        h = _calls_make(known)
        self.assertIsInstance(_lw.lower(h, known, [key], [], out), _lw.Lowered)
        # A spec the list was not made of, in a form of later: each is
        # refused by the lowering itself, whatever the interpreter takes.
        for why, later in {
            "the flag 'later' is not understood": spec(flags=["pure", "later"]),
            "the flag 'moves' is not understood": spec(flags=["pure", ("moves", (0,))]),
            "the operand template 'moved' is not understood": spec(
                args=[("moved", 0), ("in", 0)]
            ),
            "a 'drop' entry is not lowered": spec(callee=("drop",)),
            "a 'state' entry is not lowered": spec(
                callee=("state", torch._C._set_grad_enabled)
            ),
            "the assert 'later' is not understood": spec(asserts=[("later", 1)]),
            "entries of 8": spec(fields=7),
            "entries of 8 ": spec(fields=9),
            "drops of 3": spec(more=[("drop", ())]),
            "drops of 3 ": spec(more=[("drop", (), 0, ())]),
            "a spec that is not understood: ValueError": spec(classes=(T,)),
            "a spec that is not understood: ValueError ": spec(classes=(T, (T,))),
            "a spec of 5 fields": spec(top=4),
            "a spec that is not understood": None,
        }.items():
            with self.subTest(why=why):
                low = _lw.lower(h, later, [key], [], out)
                self.assertIsInstance(low, _lw.NotLowerable)
                self.assertIn(why.strip(), low.reason)

    def test_the_facts_of_a_list(self, device):
        x = torch.randn(3, 4, device=device)
        w = torch.randn(3, 4, device=device)
        mul_ = _cl_vector(torch.Tensor.mul_)
        made = _cl_entry(
            _cl_vector(torch.add),
            [("in", 0), ("pin", 0)],
            out=0,
            flags=["pure", "fresh"],
        )
        view = _cl_entry(
            _cl_vector(torch.Tensor.view_as),
            [("in", 0), ("pin", 0)],
            out=0,
            flags=["pure"],
        )
        parts = _cl_entry(
            _cl_vector(torch.Tensor.unbind),
            [("in", 0)],
            out=0,
            flags=["pure"],
            asserts=[("class", tuple)],
        )
        both = ("seq", "list", [("res", 0), ("pin", 0)])

        def written(first, operand, callee=mul_, rest=(("const", 0),)):
            return (
                1,
                1,
                [2.0],
                [first, _cl_entry(callee, [operand, *rest], flags=[("writes", (0,))])],
                [("res", 0)],
            )

        for spec, facts in (
            (written(made, ("res", 0)), set()),
            (written(view, ("res", 0)), {"writes_input"}),
            (written(made, ("in", 0)), {"writes_input"}),
            (written(made, ("pin", 0)), {"writes_input"}),
            (written(parts, ("part", 0, (0,))), {"writes_input"}),
            (written(made, both, _cl_vector(torch._foreach_mul_)), {"writes_input"}),
            (
                written(
                    made, ("seq", "list", [("res", 0)]), _cl_vector(torch._foreach_mul_)
                ),
                set(),
            ),
        ):
            with self.subTest(
                spec=_calls_text(_calls_make(_lw_stated(spec, [x], [w])))
            ):
                low = _lw_lower(spec, [x], [w])[1]
                self.assertIsInstance(low, _lw.Lowered, msg=getattr(low, "reason", ""))
                self.assertEqual(low.list_facts, facts)
                self.assertLessEqual(low.list_facts, set(_lw.FACTS))

    def test_the_module_names_nothing_of_the_tracer(self, device):
        with open(_lw.__file__) as f:
            tree = ast.parse(f.read())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add("." * node.level + (node.module or ""))
        self.assertEqual(
            names,
            {
                "dataclasses",
                "typing",
                "torch",
                "torch._rust",
                "torch._subclasses",
                "torch.fx.experimental.proxy_tensor",
                "torch.fx.experimental.symbolic_shapes",
                ".lower_scalars",
            },
        )


instantiate_device_type_tests(TestVmLower, globals())


@unittest.skipUnless(CHECKS_HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmLowerScalars(TestCase):
    """A list may read a number out of a tensor and compute with it. The
    rule: the graph of such a list computes the number of every call, as
    the list does (nothing of the traced value is in it; what was recorded
    is the number's class), and a list that needs the value where a graph
    cannot have it (a size, an index, a branch inside a callee, the number
    as an output) gets no graph but a reason.
    """

    hw_classification = HardwareClassification.ACCELERATOR

    def lws_agrees(self, spec, make):
        """Lowers ``spec`` for the tensors of ``make(0)`` and holds the graph
        against the list on those of three other seeds: other values in
        the tensors that are read."""
        h, low = _lw_lower(spec, *make(0))
        self.assertIsInstance(low, _lw.Lowered, msg=getattr(low, "reason", ""))
        self.assertEqual(_lw_modes(), (0, 0))
        self.assertIn("reads_value", low.list_facts)
        for seed in (1, 2, 3):
            mine, theirs = make(seed), make(seed)
            want = _calls_run(h, tuple(theirs[0]), tuple(theirs[1]))
            got = low.gm(*mine[0], *mine[1], *low.constants)
            self.assertEqual(_wt_bits(got), _wt_bits(want))
            self.assertEqual(_wt_bits(mine), _wt_bits(theirs))
        return low

    def lws_tensors(self, device, *values):
        """make(seed): one input of (3, 4) and a scalar pin per value (a
        function of the seed)."""

        def make(seed):
            gen = torch.Generator().manual_seed(seed)
            x = torch.randn(3, 4, generator=gen).to(device)
            return [x], [torch.tensor(v(seed), device=device) for v in values]

        return make

    def lws_reads(self, low):
        read = torch.ops.aten._local_scalar_dense.default
        return len(low.gm.graph.find_nodes(op="call_function", target=read))

    def test_arithmetic_on_a_read_is_done_by_the_graph(self, device):
        # s = t.item(); w = 0.001 / (1.0 - 0.9 ** s); k = n.item()
        # (x * w + k) * -s, the operators as instructions and as C calls.
        item = _cl_vector(torch.Tensor.item)
        number, count = [("class", float)], [("class", int)]
        spec = (1, 2, [0.9, 1.0, 0.001], [
            _cl_entry(item, [("pin", 0)], out=0, flags=["pure"], asserts=number),
            _cl_entry(("binary", "**"), [("const", 0), ("res", 0)], out=1, flags=["pure"], asserts=number),
            _cl_entry(("binary", "-"), [("const", 1), ("move", 1)], out=2, flags=["pure"], asserts=number),
            _cl_entry(_cl_vector(operator.truediv), [("const", 2), ("move", 2)], out=3, flags=["pure"], asserts=number),
            _cl_entry(("binary", "*"), [("in", 0), ("move", 3)], out=4, flags=["pure", "fresh"]),
            _cl_entry(item, [("pin", 1)], out=5, flags=["pure"], asserts=count),
            _cl_entry(_cl_vector(torch.add), [("move", 4), ("move", 5)], out=6, flags=["pure", "fresh"]),
            _cl_entry(("unary", "-"), [("move", 0)], out=7, flags=["pure"], asserts=number),
            _cl_entry(_cl_vector(torch.Tensor.mul_), [("move", 6), ("move", 7)], out=8, flags=[("writes", (0,))]),
        ], [("res", 8)])  # fmt: skip
        values = (lambda seed: 1.5 + seed, lambda seed: 2 + seed)
        low = self.lws_agrees(spec, self.lws_tensors(device, *values))
        self.assertEqual(low.list_facts, {"reads_value"})
        self.assertEqual(low.specialized, ())
        self.assertEqual(self.lws_reads(low), 2)
        # No number of the traced call (2.5, 2 and what was made of them).
        text = low.gm.code
        for traced in ("2.5", "0.76", "0.0012", "1.5"):
            self.assertNotIn(traced, text)

    def test_a_tensor_written_between_two_reads_is_read_twice(self, device):
        item, mul = _cl_vector(torch.Tensor.item), _cl_vector(torch.mul)
        number = [("class", float)]

        def spec(written):
            write = _cl_entry(_cl_vector(torch.Tensor.add_), [("pin", 0), ("const", 0)], flags=[("writes", (0,))])  # fmt: skip
            again = _cl_entry(
                _cl_vector(torch.Tensor.abs), [("pin", 0)], flags=["pure"]
            )
            return (1, 1, [1.0], [
                _cl_entry(item, [("pin", 0)], out=0, flags=["pure"], asserts=number),
                write if written else again,
                _cl_entry(item, [("pin", 0)], out=2, flags=["pure"], asserts=number),
                _cl_entry(mul, [("in", 0), ("move", 0)], out=3, flags=["pure", "fresh"]),
                _cl_entry(mul, [("in", 0), ("move", 2)], out=4, flags=["pure", "fresh"]),
            ], [("res", 3), ("res", 4)])  # fmt: skip

        make = self.lws_tensors(device, lambda seed: 0.5 * seed)
        low = self.lws_agrees(spec(True), make)
        self.assertEqual(self.lws_reads(low), 2)
        self.assertEqual(low.list_facts, {"reads_value", "writes_input"})
        self.assertEqual(self.lws_reads(self.lws_agrees(spec(False), make)), 2)

    def test_every_read_of_the_list_is_a_read_of_the_graph(self, device):
        # s = a.item(); b.add_(1.0); t = a.item(); x * s, x * t
        item, mul = _cl_vector(torch.Tensor.item), _cl_vector(torch.mul)
        number = [("class", float)]
        spec = (1, 2, [1.0], [
            _cl_entry(item, [("pin", 0)], out=0, flags=["pure"], asserts=number),
            _cl_entry(_cl_vector(torch.Tensor.add_), [("pin", 1), ("const", 0)], flags=[("writes", (0,))]),
            _cl_entry(item, [("pin", 0)], out=2, flags=["pure"], asserts=number),
            _cl_entry(mul, [("in", 0), ("move", 0)], out=3, flags=["pure", "fresh"]),
            _cl_entry(mul, [("in", 0), ("move", 2)], out=4, flags=["pure", "fresh"]),
        ], [("res", 3), ("res", 4)])  # fmt: skip
        make = self.lws_tensors(
            device, lambda seed: 0.5 * seed, lambda seed: 2.0 * seed
        )
        low = self.lws_agrees(spec, make)
        self.assertEqual(self.lws_reads(low), 2)
        # The two tensors of a later call may be one memory: its write is
        # then between the two reads, in the graph as in the list.
        (x,), (a, _) = make(1)
        (y,), (b, _) = make(1)
        h = _calls_make(_lw_stated(spec, [x], [a, a]))
        want = _calls_run(h, (x,), (a, a.view(())))
        got = low.gm(y, b, b.view(()))
        self.assertEqual(_wt_bits(got), _wt_bits(want))
        self.assertNotEqual(_wt_bits(got[0]), _wt_bits(got[1]))

    def test_what_is_recorded_of_a_read_is_its_class(self, device):
        item = _cl_vector(torch.Tensor.item)
        values = {
            float: torch.tensor(2.5, device=device),
            int: torch.tensor(3, device=device),
            bool: torch.tensor(True, device=device),
        }
        x = torch.randn(3, 4, device=device)

        def spec(asserts, out=0):
            return _lw_stated((1, 1, [], [
                _cl_entry(item, [("pin", 0)], out=out, flags=["pure"], asserts=asserts),
                _cl_entry(_cl_vector(torch.neg), [("in", 0)], out=1, flags=["pure", "fresh"]),
            ], [("res", 1)]), [x], [x])  # fmt: skip

        def low(s, pin):
            keys = _lw_keys([x, pin])
            left = _lw.Traced((keys[0],), (keys[0],), (keys[1],))
            return _lw.lower(_calls_make(s), s, keys[:1], keys[1:], left)

        for cls, pin in values.items():
            for said in values:
                with self.subTest(read=cls.__name__, said=said.__name__):
                    got = low(spec([("class", said)]), pin)
                    if said is cls:
                        self.assertIsInstance(
                            got, _lw.Lowered, msg=getattr(got, "reason", "")
                        )
                        self.assertEqual(got.list_facts, {"reads_value"})
                    else:
                        self.assertIsInstance(got, _lw.NotLowerable)
                        self.assertEqual(got.entry, 0)
        # A read of which no class was recorded: kept or not.
        for s in (spec([]), spec([], out=None), spec([("class", torch.Tensor)])):
            got = low(s, values[float])
            self.assertIsInstance(got, _lw.NotLowerable)
            self.assertIn("of no asserted class", got.reason)
            self.assertEqual((got.entry, got.list_facts), (0, {"reads_value"}))
        self.assertEqual(_lw_modes(), (0, 0))

    def test_where_the_value_is_needed_there_is_no_graph(self, device):
        item = _cl_vector(torch.Tensor.item)
        x = torch.randn(3, 4, device=device)
        pins = [torch.tensor(1, device=device), torch.tensor(0.25, device=device)]
        count, number = ("pin", 0), ("pin", 1)
        r = ("res", 0)

        def spec(
            read, callee, args, flags=("pure",), out=None, outputs=None, asserts=()
        ):
            cls = int if read is count else float
            return (1, 2, [0, -1, 0.5], [
                _cl_entry(item, [read], out=0, flags=["pure"], asserts=[("class", cls)]),
                _cl_entry(callee, args, out=out, flags=flags, asserts=asserts),
            ], [("in", 0)] if outputs is None else outputs)  # fmt: skip

        bound = ("slice", ("null",), r, ("null",))
        writes = [("writes", (1,))]
        needs = "it needs the value of a number read from a tensor"
        cases = {
            "a size": (spec(count, _cl_vector(torch.zeros), [r], out=1), "a size of its result", 1),
            "a view's size": (spec(count, _cl_vector(torch.Tensor.view), [("in", 0), ("const", 1), r], out=1), "a size of its result", 1),
            "a size in a tuple": (spec(count, _cl_vector(torch.topk), [("in", 0), r], out=1, asserts=[("class", torch.return_types.topk)]), "a size of its result", 1),
            "an index": (spec(count, ("getitem",), [("in", 0), r], out=1), "is an index", 1),
            "a bound of a slice": (spec(count, ("getitem",), [("in", 0), bound], out=1), "is an index", 1),
            "an index of a store": (spec(count, ("setitem",), [("const", 2), ("in", 0), r], writes), "is an index", 1),
            "range": (spec(count, _cl_vector(range), [r]), needs, 1),
            "int": (spec(number, _cl_vector(int), [r]), needs, 1),
            "float": (spec(number, _cl_vector(float), [r]), needs, 1),
            "bool": (spec(number, _cl_vector(bool), [r]), needs, 1),
            "math.sqrt": (spec(number, _cl_vector(math.sqrt), [r]), needs, 1),
            "max": (spec(number, _cl_vector(max), [r, ("const", 2)]), needs, 1),
            "an output": (spec(number, _cl_vector(torch.neg), [("in", 0)], outputs=[r]), "are not compared", None),
        }  # fmt: skip
        for what, (s, why, entry) in cases.items():
            with self.subTest(what=what):
                low = _lw_lower(s, [x], pins)[1]
                self.assertIsInstance(low, _lw.NotLowerable)
                self.assertIn(why, low.reason)
                self.assertEqual(low.entry, entry)
                self.assertEqual(_lw_modes(), (0, 0))
        # The same numbers where no fact of the run hangs on the value: what
        # a tensor is filled with, the index a callee is handed (the kernel
        # of every call checks it, as the list's does).
        fill = spec(number, _cl_vector(torch.full), [("seq", "tuple", [("const", 2)]), r], out=1)  # fmt: skip
        fill = (1, 2, [0, -1, 2], *fill[3:])
        low = _lw_lower(fill, [x], pins)[1]
        self.assertIsInstance(low, _lw.Lowered, msg=getattr(low, "reason", ""))
        row = spec(count, _cl_vector(torch.select), [("in", 0), ("const", 0), r], out=1, outputs=[("res", 1)])  # fmt: skip
        h, low = _lw_lower(row, [x], pins)
        self.assertIsInstance(low, _lw.Lowered, msg=getattr(low, "reason", ""))
        for n in (0, 2, -1):
            at = [torch.tensor(n, device=device), pins[1]]
            self.assertEqual(
                _wt_bits(low.gm(x, *at)), _wt_bits(_calls_run(h, (x,), tuple(at)))
            )
        outside = [torch.tensor(3, device=device), pins[1]]
        with self.assertRaises(IndexError):
            _calls_run(h, (x,), tuple(outside))
        h.release()
        with self.assertRaises(IndexError):
            low.gm(x, *outside)

    def test_what_the_run_assumed_of_a_float_is_an_assert_of_the_graph(self, device):
        # x * (1.0 - 0.5 ** s) ** 0.5: a float while 0.5 ** s <= 1.0, as at
        # the trace; a complex number for a negative s.
        number = [("class", float)]

        def spec(base):
            return (1, 1, [1.0, 0.5], [
                _cl_entry(_cl_vector(torch.Tensor.item), [("pin", 0)], out=0, flags=["pure"], asserts=number),
                _cl_entry(("binary", "**"), [("const", 1), ("move", 0)], out=1, flags=["pure"], asserts=number),
                _cl_entry(("binary", "-"), [("const", 0), ("move", 1)], out=2, flags=["pure"], asserts=number),
                _cl_entry(("binary", "**"), [(base, 2), ("const", 1)], out=3, flags=["pure"], asserts=number),
                _cl_entry(("binary", "*"), [("in", 0), ("move", 3)], out=4, flags=["pure", "fresh"]),
            ], [("res", 4)])  # fmt: skip

        make = self.lws_tensors(device, lambda seed: 0.25 * seed)
        low = self.lws_agrees(spec("move"), make)
        check = torch.ops.aten._assert_scalar.default
        self.assertEqual(
            len(low.gm.graph.find_nodes(op="call_function", target=check)), 1
        )
        # A call whose number is outside it: the list stops at the entry
        # whose result is no float (a miss: nothing was written), the graph
        # raises. Neither hands out a value.
        (x,), _ = make(1)
        s = torch.tensor(-2.0, device=device)
        h = _calls_make(_lw_stated(spec("move"), [x], [s]))
        self.assertIsNone(_calls_run(h, (x,), (s,)))
        self.assertEqual(h.stopped[:2], (3, "assert"))
        h.release()
        with self.assertRaises(RuntimeError):
            low.gm(x, s)
        # An assumption that cannot be made an assert (here: of the read
        # number itself, s <= 1.0 for (1.0 - s) ** 0.5): no graph.
        bare = spec("move")
        bare = (*bare[:3], [bare[3][0], _cl_drop(0), *bare[3][2:]], bare[4])
        bare[3][2] = _cl_entry(("binary", "-"), [("const", 0), ("res", 0)], out=2, flags=["pure"], asserts=number)  # fmt: skip
        bare[3][1] = _cl_entry(
            ("binary", "**"), [("const", 1), ("res", 0)], flags=["pure"]
        )
        refused = _lw_lower(bare, *make(1))[1]
        self.assertIsInstance(refused, _lw.NotLowerable)
        self.assertIn("could not be made an assert of the graph", refused.reason)

    def test_the_module_of_the_scalars_names_nothing_of_the_tracer(self, device):
        beside = os.path.join(os.path.dirname(_lw.__file__), "lower_scalars.py")
        with open(beside) as f:
            tree = ast.parse(f.read())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add("." * node.level + (node.module or ""))
        self.assertEqual(
            names,
            {
                "operator",
                "torch",
                "torch._subclasses.fake_tensor",
                "torch.fx.experimental.symbolic_shapes",
                "torch.fx.passes.runtime_assert",
            },
        )


instantiate_device_type_tests(TestVmLowerScalars, globals())


NAMESPACES_SETUP = """import builtins, functools, sys, types
import torch
def names(**items):
    return {'__builtins__': vars(builtins), '__name__': 'vm_ns', 'torch': torch, **items}
def of(function, **items):
    return types.FunctionType(function.__code__, names(**items))
def before():
    pass
def state():
    return None
"""
NAMESPACES_TEMPLATE = """
def template(x):
    y = torch.sin(x) * 2.0 + torch.cos(x)
    return (y * {} + torch.tanh(y)).sum(-1)
"""
# Two functions, `A` and `B`, whose frames run the same source in namespaces
# that are not the same: (what makes them; which of them get a region; whether
# a region traced for one is to run for the other).
NAMESPACES_WAYS = {
    "global_read": (
        NAMESPACES_TEMPLATE.format("K")
        + "A = of(template, K=2.0)\nB = of(template, K=5.0)",
        "AB",
        False,
    ),
    "global_call": (
        NAMESPACES_TEMPLATE.format("scale()")
        + "A = of(template, scale=lambda: 2.0)\nB = of(template, scale=lambda: 5.0)",
        "AB",
        False,
    ),
    "builtin_shadowed": (
        NAMESPACES_TEMPLATE.format("abs(-2.0)")
        + "A = of(template)\nB = of(template, abs=lambda value: 5.0)",
        "AB",
        False,
    ),
    # The interpreter's own `__import__` finds M1; the other one is Python
    # code, which runs under no region, and returns M2.
    "import_with_another_import_function": (
        """
M1 = types.ModuleType('vm_ns_m')
M1.K = 2.0
M2 = types.ModuleType('vm_ns_m')
M2.K = 5.0
OWN = builtins.__import__
SEEN = []
def other(name, *args, **kwargs):
    if name != 'vm_ns_m':
        return OWN(name, *args, **kwargs)
    SEEN.append(name)
    return M2
def before():
    sys.modules['vm_ns_m'] = M1
def state():
    return len(SEEN)
def template(x):
    y = torch.sin(x) * 2.0 + torch.cos(x)
    import vm_ns_m
    return (y * vm_ns_m.K + torch.tanh(y)).sum(-1)
A = of(template)
B = of(template, __builtins__={**vars(builtins), '__import__': other})""",
        "A",
        False,
    ),
    "copied_with_update_wrapper": (
        NAMESPACES_TEMPLATE.format("K")
        + "A = of(template, K=2.0)\n"
        + "B = functools.update_wrapper(of(template, K=5.0), A)",
        "AB",
        False,
    ),
    # Two code objects: each function has places of its own.
    "exec_twins": (
        f"TWIN = {NAMESPACES_TEMPLATE.format('K')!r}"
        + """
def twin(k):
    space = names(K=k)
    exec(TWIN, space)
    return space['template']
A = twin(2.0)
B = twin(5.0)""",
        "AB",
        False,
    ),
    # One namespace, other cells: what a cell holds is an input of the region.
    "closures_of_one_factory": (
        """
def factory(t):
    def inner(x):
        y = torch.sin(x) * 2.0 + torch.cos(x)
        return (y * t + torch.tanh(y)).sum(-1)
    return inner
A = factory(torch.full((8,), 2.0))
B = factory(torch.full((8,), 5.0))""",
        "AB",
        True,
    ),
}


@instantiate_parametrized_tests
@unittest.skipUnless(CHECKS_HAS_VM, "needs the torch._rust extension and CPython 3.12")
class TestVmFrameNamespaces(TestCase):
    """A region is entered only by a frame that has the globals and the
    builtins of the frame it was traced in. A code object can be the code of
    several functions, each with namespaces of its own, and what a trace read
    of the frame's namespaces is checked in the dicts it was read from: a
    function with other namespaces gets its own trace, and computes what
    eager computes. Functions of one code object in one namespace (the
    closures a factory returns) go on sharing their regions."""

    hw_classification = HardwareClassification.GENERIC

    def runs(self, compiled):
        return sum(r.get("runs") or 0 for r in vm.regions(compiled))

    @parametrize("way", list(NAMESPACES_WAYS))
    @checks_isolated
    def test_two_functions_called_in_turn(self, way):
        import torch._rust_inductor as tracer

        source, with_regions, shared = NAMESPACES_WAYS[way]
        eager, traced = {}, {}
        exec(compile(NAMESPACES_SETUP + source, "<namespaces eager>", "exec"), eager)
        exec(compile(NAMESPACES_SETUP + source, "<namespaces vm>", "exec"), traced)
        compiled = {name: vm(traced[name], passes=CHECKS_EARLY) for name in "AB"}
        ran = {"A": 0, "B": 0}
        # Which function the program calls: A, then B, then A again.
        schedule = "A" * 7 + "B" * 7 + "A" * 5
        for call, name in enumerate(schedule):
            torch.manual_seed(call)
            x = torch.randn(4, 8)
            eager["before"]()
            want = eager[name](x)
            traced["before"]()
            runs, traces = self.runs(compiled["A"]), tracer.stats["traced"]
            got = compiled[name](x)
            msg = f"call {call}, of {name}"
            self.assertEqual(got, want, atol=0, rtol=0, msg=msg)
            self.assertEqual(traced["state"](), eager["state"](), msg=msg)
            ran_now = self.runs(compiled["A"]) - runs
            ran[name] += ran_now
            if shared and call in (7, 14):
                # The first call of the other function runs the regions
                # that were traced for the one before it.
                self.assertGreater(ran_now, 0, msg=msg)
                self.assertEqual(tracer.stats["traced"], traces, msg=msg)
        places = vm.regions(compiled["A"])
        self.assertEqual(
            {name for name in ran if ran[name]}, set(with_regions), msg=str(places)
        )


if __name__ == "__main__":
    run_tests()
