# Owner(s): ["module: dynamo"]

"""Tests of the soundness contract of torch.vm's core: what a region checks
before it runs, and what the engine may assume of a traced call.

Every class here states one rule of the contract and tests it over the whole
domain the rule speaks about (every field of what is compared for a tensor,
every instruction the engine knows), not over the cases somebody thought of.
"""

import ast
import collections
import contextlib
import dis
import enum
import functools
import glob
import json
import math
import opcode
import operator
import os
import re
import struct
import subprocess
import sys
import traceback
import types
import unittest
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


def _cl_entry(
    callee, args, kwnames=None, out=None, at=0, flags=(), asserts=(), kills=()
):
    return (
        callee,
        list(args),
        kwnames,
        out,
        at,
        tuple(flags),
        tuple(asserts),
        tuple(kills),
    )


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
    exception of a call that raises, where an intermediate dies), and a list
    that is not understood in full is refused when it is made.
    """

    hw_classification = HardwareClassification.GENERIC

    def one(self, callee, inputs, kwnames=None):
        args = [("in", k) for k in range(len(inputs))]
        spec = (
            len(inputs),
            0,
            [],
            [_cl_entry(callee, args, kwnames, out=0)],
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
                )
            ],
            [("res", 0)],
        )
        self.assertTrue(torch.equal(_cl_run(spec, (x, x))[0], torch.cat([x, x], dim=1)))

    def test_subscripts_and_attributes(self):
        x = torch.arange(24.0).reshape(2, 3, 4)
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
            _cl_entry(("getitem",), [("in", 0), key], out=0),
            _cl_entry(("getitem",), [("in", 0), ("slice", ("null",), ("null",), ("const", 0))], out=1),
            _cl_entry(("getattr", "mT"), [("res", 0)], out=2),
            _cl_entry(("setitem",), [("const", 5), ("res", 1), ("const", 4)]),
            _cl_entry(("setattr", "kept"), [("res", 2), ("pin", 0)]),
            _cl_entry(("setitem",), [("res", 0), ("pin", 1), ("const", 3)]),
            _cl_entry(("delitem",), [("pin", 2), ("const", 4)]),
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
            _cl_entry(_cl_vector(tuple), [nested], out=0),
            _cl_entry(_cl_vector(torch.max), [("in", 0), ("const", 0)], out=1),
            _cl_entry(_cl_vector(torch.add), [("part", 0, (0, 1)), ("part", 1, (0,))], out=2, kills=[1]),
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
        x = torch.zeros(2, 3)
        spec = (2, 1, [2.0, (1, 0), "k", x], [
            _cl_entry(("vector", torch.add, "torch.add"), [("in", 0), ("in", 1)], out=0, flags=["pure", "fresh"], asserts=[("key", ("sizes", "dtype", "device", "requires_grad"), x)]),
            _cl_entry(("vector", torch.Tensor.mul_, "Tensor.mul_"), [("res", 0), ("pin", 0)], out=1, at=1, flags=[("writes", (0,))], asserts=[("is", 0)], kills=[0]),
            _cl_entry(("binary", "*="), [("res", 1), ("const", 0)], out=2, at=2, flags=["opaque", "covered", ("writes", (0,)), ("rekeys", (0, 1))]),
            _cl_entry(("vector", torch.permute, "torch.permute"), [("res", 2), ("const", 1)], ("dims",), out=3, at=3, flags=["draws", "effect"]),
            _cl_entry(("vector", torch.max, "torch.max"), [("res", 3), ("const", 3)], out=4, at=4, asserts=[("count", 2)], kills=[1, 3]),
            _cl_entry(("compare", "<="), [("part", 4, (0,)), ("in", 0)], at=5),
            _cl_entry(("unary", "-"), [("res", 2)], out=6, at=6),
            _cl_entry(("getitem",), [("res", 6), ("seq", "tuple", [("slice", ("const", 0), ("null",), ("null",)), ("seq", "list", [("in", 1)])])], out=7, at=7),
            _cl_entry(("setitem",), [("in", 0), ("res", 7), ("map", ("k", 1), [("in", 1), ("seq", "tuple", [("const", 2)])])], at=8),
            _cl_entry(("delitem",), [("pin", 0), ("const", 2)], at=9),
            _cl_entry(("getattr", "mT"), [("res", 7)], out=10, at=10),
            _cl_entry(("setattr", "grad"), [("res", 10), ("pin", 0)], at=11, kills=[10, 7]),
        ], [])  # fmt: skip
        self.assertEqual(
            _calls_text(_calls_make(spec)).split("\n"),
            [
                "  0  r0 = torch.add(in0, in1)  @0 [pure fresh] assert sizes=[2, 3] dtype=f32 device=cpu requires_grad=false",
                "  1  r1 = Tensor.mul_(r0, pin0)  @1 [writes(0)] assert is(operand 0) kills r0",
                "  2  r2 = r1 *= 2.0  @2 [opaque covered writes(0) rekeys(0, 1)]",
                "  3  r3 = torch.permute(r2, dims=(1, 0))  @3 [draws effect]",
                "  4  r4 = torch.max(r3, c3)  @4 assert count=2 kills r1 r3",
                "  5  r4.0 <= in0  @5",
                "  6  r6 = -r2  @6",
                "  7  r7 = r6[(2.0::, [in1])]  @7",
                "  8  r7[{'k': in1, 1: ('k',)}] = in0  @8",
                "  9  del pin0['k']  @9",
                " 10  r10 = r7.mT  @10",
                " 11  pin0.grad = r10  @11 kills r10 r7",
            ],
        )

    def test_a_list_that_is_not_understood_is_refused(self):
        add, x = _cl_vector(torch.add), ("in", 0)

        def spec(*entries, inputs=1, pins=1, consts=(1,), outputs=()):
            return (inputs, pins, list(consts), list(entries), list(outputs))

        first = _cl_entry(add, [x, x], out=0)
        self.assertEqual(
            _calls_run(_calls_make(spec(first, outputs=[("res", 0)])), (1,), (None,)),
            (2,),
        )

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
            (
                "r0 is not held",
                spec(
                    first,
                    _cl_entry(add, [x, x], kills=[0]),
                    _cl_entry(add, [x, ("res", 0)]),
                ),
            ),
            (
                "r0 is not held",
                spec(
                    first,
                    _cl_entry(add, [x, x], kills=[0]),
                    _cl_entry(add, [x, ("seq", "list", [("part", 0, (1,))])]),
                ),
            ),
            (
                "output 0: r0 is not held",
                spec(first, _cl_entry(add, [x, x], kills=[0]), outputs=[("res", 0)]),
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
            # out, kills
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
            (
                "kills r0, which is not an earlier",
                spec(_cl_entry(add, [x, x], out=0, kills=[0])),
            ),
            (
                "kills r1, which is not an earlier",
                spec(
                    first,
                    _cl_entry(add, [x, x], out=1),
                    _cl_entry(add, [x, x], kills=[1, 1]),
                ),
            ),
            (
                "kills r0, which is not an earlier",
                spec(_cl_entry(add, [x, x]), _cl_entry(add, [x, x], kills=[0])),
            ),
            (
                "kills r5, which is not an earlier",
                spec(first, _cl_entry(add, [x, x], kills=[5])),
            ),
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
                'no key field "strides"',
                spec(
                    _cl_entry(
                        add, [x, x], asserts=[("key", ("strides",), torch.ones(1))]
                    )
                ),
            ),
            (
                "the key of an assert is read off a tensor",
                spec(_cl_entry(add, [x, x], asserts=[("key", ("sizes",), (1,))])),
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
        for flag in ("draws", "effect", "opaque", ("writes", (0,)), ("rekeys", (0,))):
            cases.append(
                ("pure beside", spec(_cl_entry(add, [x, x], flags=["pure", flag])))
            )
        for why, bad in cases:
            with self.subTest(why=why):
                with self.assertRaisesRegex(ValueError, "torch.vm call list: .*" + why):
                    _calls_make(bad)
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
        """x.add_(1.0); <second>(x, y); x.mul_(2.0); seen.append(x)"""
        return _calls_make((2, 1, [1.0, 2.0], [
            _cl_entry(_cl_vector(torch.Tensor.add_), [("in", 0), ("const", 0)], flags=flags),
            _cl_entry(_cl_vector(second), [("in", 0), ("in", 1)], out=1, flags=["pure"], asserts=asserts),
            _cl_entry(_cl_vector(torch.Tensor.mul_), [("in", 0), ("const", 1)], flags=[("writes", (0,))]),
            _cl_entry(_cl_vector(list.append), [("pin", 0), ("res", 1)], flags=["effect"], kills=[1]),
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
        self.assertEqual(h.stopped, (1, "raised", False))
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
        self.assertEqual(h.stopped, (1, "raised", True))
        # An exception of any class comes out as it is.
        h = _calls_make(
            (
                1,
                0,
                ["k"],
                [_cl_entry(("getitem",), [("in", 0), ("const", 0)], out=0)],
                [("res", 0)],
            )
        )
        with self.assertRaisesRegex(KeyError, "'k'"):
            _calls_run(h, ({},), ())
        self.assertEqual(h.stopped, (0, "raised", False))

    def test_an_assert_that_fails(self):
        y, seen = torch.ones(3, 2), []
        fits, other = torch.empty(2, 2), torch.empty(2, 5)
        for flags, restartable in ((["pure"], True), ([("writes", (0,))], False)):
            with self.subTest(restartable=restartable):
                x = torch.ones(2, 3)
                h = self.step_list(
                    torch.matmul, flags=flags, asserts=[("key", ("sizes",), fits)]
                )
                self.assertIs(_calls_run(h, (x, y), (seen,))[0], x)
                self.assertIsNone(h.stopped)
                x = torch.ones(2, 3)
                del seen[:]
                h = self.step_list(
                    torch.matmul, flags=flags, asserts=[("key", ("sizes",), other)]
                )
                if restartable:
                    # A miss: no outputs, no exception.
                    self.assertIsNone(_calls_run(h, (x, y), (seen,)))
                else:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        r"torch.vm: call list entry 1: the result is not what the traced call read of it \(sizes=\[2, 5\]\), and what ran before cannot be taken back: 1  r1 = .*matmul\(in0, in1\)",
                    ):
                        _calls_run(h, (x, y), (seen,))
                self.assertEqual(h.stopped, (1, "assert", restartable))
                # Nothing behind the entry ran.
                self.assertTrue(torch.equal(x, torch.full((2, 3), 2.0)))
                self.assertEqual(seen, [])

    def test_every_field_of_an_assert(self):
        x = torch.ones(2, 3)
        grad = torch.ones(2, 3, requires_grad=True)
        differs = {
            "sizes": torch.ones(3, 2),
            "dtype": torch.ones(2, 3, dtype=torch.float64),
            "device": torch.ones(2, 3, device="meta"),
            "requires_grad": grad,
        }

        def run(callee, asserts, inputs=(x, x)):
            args = [("in", k) for k in range(len(inputs))]
            h = _calls_make(
                (
                    len(inputs),
                    0,
                    [],
                    [_cl_entry(callee, args, out=0, flags=["pure"], asserts=asserts)],
                    [],
                )
            )
            return _calls_run(h, inputs, ()), h.stopped

        add = _cl_vector(torch.add)
        fields = tuple(differs)
        self.assertEqual(run(add, [("key", fields, x)]), ((), None))
        for field, other in differs.items():
            with self.subTest(field=field):
                # The one field is compared, and only where it is asked for.
                self.assertEqual(
                    run(add, [("key", (field,), other)]), (None, (0, "assert", True))
                )
                self.assertEqual(
                    run(add, [("key", fields, other)]), (None, (0, "assert", True))
                )
                rest = tuple(f for f in fields if f != field)
                self.assertEqual(run(add, [("key", rest, other)]), ((), None))
        self.assertEqual(
            run(add, [("key", ("requires_grad",), grad)], (grad, x)), ((), None)
        )
        # A key is a tensor's.
        self.assertEqual(
            run(
                _cl_vector(torch.Tensor.item),
                [("key", ("dtype",), x)],
                (torch.ones(()),),
            ),
            (None, (0, "assert", True)),
        )
        # The count of a result tuple.
        max_ = _cl_vector(torch.max)
        self.assertEqual(run(max_, [("count", 2)], (x, 0)), ((), None))
        self.assertEqual(run(max_, [("count", 3)], (x, 0)), (None, (0, "assert", True)))
        self.assertEqual(run(add, [("count", 1)]), (None, (0, "assert", True)))
        # The result is an operand.
        add_ = _cl_vector(torch.Tensor.add_)
        self.assertEqual(run(add_, [("is", 0)], (torch.ones(2), 1.0)), ((), None))
        self.assertEqual(
            run(add_, [("is", 1)], (torch.ones(2), torch.ones(2))),
            (None, (0, "assert", True)),
        )
        self.assertEqual(run(add, [("is", 0)]), (None, (0, "assert", True)))
        # Several at once: each is checked.
        self.assertEqual(
            run(add_, [("is", 0), ("key", fields, x)], (torch.ones(2, 3), 1.0)),
            ((), None),
        )
        self.assertEqual(
            run(add_, [("is", 0), ("key", fields, grad)], (torch.ones(2, 3), 1.0)),
            (None, (0, "assert", True)),
        )

    def test_a_part_that_is_not_there(self):
        x = torch.ones(2, 3)
        for flags, restartable in ((["pure"], True), ([], False)):
            for callee, operands, path in [
                (
                    _cl_vector(list),
                    [("seq", "tuple", [("in", 0)])],
                    (0,),
                ),  # a list is no tuple
                (_cl_vector(torch.max), [("in", 0), ("const", 0)], (2,)),  # too short
                (
                    _cl_vector(torch.max),
                    [("in", 0), ("const", 0)],
                    (0, 0),
                ),  # a tensor is no tuple
            ]:
                with self.subTest(restartable=restartable, path=path):
                    first = _cl_entry(callee, operands, out=0, flags=flags)
                    h = _calls_make(
                        (
                            1,
                            0,
                            [0],
                            [
                                first,
                                _cl_entry(
                                    _cl_vector(torch.neg), [("part", 0, path)], out=1
                                ),
                            ],
                            [("res", 1)],
                        )
                    )
                    if restartable:
                        self.assertIsNone(_calls_run(h, (x,), ()))
                    else:
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "entry 1: a part of a result that the entry is called on is not there",
                        ):
                            _calls_run(h, (x,), ())
                    self.assertEqual(h.stopped, (1, "part", restartable))
                    h = _calls_make((1, 0, [0], [first], [("part", 0, path)]))
                    if restartable:
                        self.assertIsNone(_calls_run(h, (x,), ()))
                    else:
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "entry 1: a part of a result that an output is made of is not there",
                        ):
                            _calls_run(h, (x,), ())
                    self.assertEqual(h.stopped, (1, "part", restartable))

    def test_restartable_is_every_entry_up_to_the_one_that_stopped(self):
        x, y = torch.ones(2, 3), torch.ones(4, 5)
        fresh = _cl_entry(
            _cl_vector(torch.add),
            [("in", 0), ("in", 0)],
            out=0,
            flags=["pure", "fresh"],
        )
        made = _cl_entry(
            _cl_vector(torch.add), [("in", 0), ("in", 0)], out=0, flags=["pure"]
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
            with self.subTest(text=_calls_text(_calls_make((2, 0, [], entries, [])))):
                h = _calls_make((2, 0, [], entries, []))
                with self.assertRaises(RuntimeError):
                    _calls_run(h, (x.clone(), y), ())
                self.assertEqual(h.stopped[1:], ("raised", restartable))

    def test_an_intermediate_dies_where_the_list_kills_it(self):
        """t = a + b; u = t * 2.0; v = u - 1.0: eager frees t when the multiply
        returns (it was a temporary of the call), or when the local is rebound
        later. `alive(k)` asks, between entries, whether t is still there.
        """
        a, b = torch.ones(3), torch.ones(3)

        def program(kill_at):
            entries = [
                _cl_entry(_cl_vector(torch.add), [("in", 0), ("in", 1)], out=0),
                _cl_entry(_cl_vector(weakref.ref), [("res", 0)], out=1),
            ]

            def alive():
                k = len(entries)
                entries.append(
                    _cl_entry(_cl_vector(operator.call), [("res", 1)], out=k)
                )
                entries.append(
                    _cl_entry(
                        _cl_vector(operator.is_not),
                        [("res", k), ("const", 2)],
                        out=k + 1,
                        kills=[k],
                    )
                )
                return ("res", k + 1)

            probes = [alive()]
            entries.append(
                _cl_entry(
                    ("binary", "*"),
                    [("res", 0), ("const", 0)],
                    out=len(entries),
                    kills=[0] if kill_at == "multiply" else [],
                )
            )
            u = len(entries) - 1
            probes.append(alive())
            entries.append(
                _cl_entry(
                    ("binary", "-"),
                    [("res", u), ("const", 1)],
                    out=len(entries),
                    kills=[u] + ([0] if kill_at == "subtract" else []),
                )
            )
            v = len(entries) - 1
            probes.append(alive())
            return (
                2,
                0,
                [2.0, 1.0, None],
                entries,
                [("res", v), ("res", 1), ("seq", "list", probes)],
            )

        for kill_at, want in (
            ("multiply", [True, False, False]),
            ("subtract", [True, True, False]),
            ("never", [True, True, True]),
        ):
            with self.subTest(kill_at=kill_at):
                v, t, alive = _cl_run(program(kill_at), (a, b))
                self.assertEqual(alive, want)
                self.assertTrue(torch.equal(v, (a + b) * 2.0 - 1.0))
                # Behind the run only its outputs are left.
                self.assertIsNone(t())

    def test_slots_are_released_when_an_entry_raises(self):
        a, seen = torch.ones(3), []
        spec = (1, 1, [], [
            _cl_entry(_cl_vector(torch.neg), [("in", 0)], out=0),
            _cl_entry(_cl_vector(weakref.ref), [("res", 0)], out=1),
            _cl_entry(_cl_vector(list.append), [("pin", 0), ("res", 1)], kills=[1]),
            _cl_entry(_cl_vector(torch.matmul), [("res", 0), ("seq", "list", [("res", 0)])], out=3),
        ], [("res", 3)])  # fmt: skip
        with self.assertRaises(TypeError):
            _cl_run(spec, (a,), (seen,))
        self.assertIsNone(seen[0]())


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

            return _calls_make((1, 10, [None, (1, 1), 1, training, 0.1, 1e-05, cudnn], [
                _cl_entry(conv, [("in", 0), ("pin", 0), *tail], out=0, at=0),
                _cl_entry(bn, [("res", 0), *pin(1, 2, 3, 4), *norm], out=1, at=1, kills=[0]),
                _cl_entry(relu_, [("res", 1)], out=2, at=2, asserts=[("is", 0)], kills=[1]),
                _cl_entry(conv, [("res", 2), ("pin", 5), *tail], out=3, at=3, kills=[2]),
                _cl_entry(bn, [("res", 3), *pin(6, 7, 8, 9), *norm], out=4, at=4, kills=[3]),
                _cl_entry(("binary", "+="), [("res", 4), ("in", 0)], out=5, at=5, asserts=[("is", 0)], kills=[4]),
                _cl_entry(relu_, [("res", 5)], out=6, at=6, kills=[5]),
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

        def call(callee, operands, kwnames=None, kills=()):
            entries.append(
                _cl_entry(
                    callee,
                    operands,
                    kwnames,
                    out=len(entries),
                    at=len(entries),
                    flags=["pure"],
                    kills=kills,
                )
            )
            return ("res", len(entries) - 1)

        def head(weight):
            flat = call(lin, [("in", 0), ("pin", weight), ("pin", weight + 1)])
            split_ = call(view, [flat, *split], kills=[flat[1]])
            return call(
                transpose, [split_, ("const", 4), ("const", 9)], kills=[split_[1]]
            )

        q, k, v = head(0), head(2), head(4)
        kt = call(transpose, [k, ("const", 5), ("const", 6)], kills=[k[1]])
        raw = call(matmul, [q, kt], kills=[q[1], kt[1]])
        scores = call(("binary", "/"), [raw, ("const", 7)], kills=[raw[1]])
        probs = call(
            _cl_vector(torch.Tensor.softmax),
            [scores, ("const", 5)],
            ("dim",),
            kills=[scores[1]],
        )
        mixed = call(matmul, [probs, v], kills=[probs[1], v[1]])
        moved = call(transpose, [mixed, ("const", 4), ("const", 9)], kills=[mixed[1]])
        packed = call(_cl_vector(torch.Tensor.contiguous), [moved], kills=[moved[1]])
        context = call(
            view, [packed, ("const", 0), ("const", 1), ("const", 8)], kills=[packed[1]]
        )
        dense = call(lin, [context, ("pin", 6), ("pin", 7)], kills=[context[1]])
        out = call(("binary", "+"), [dense, ("in", 0)], kills=[dense[1]])
        consts = [batch, length, heads, width // heads, 1, -1, -2, scale, width, 2]
        h = _calls_make((1, 8, consts, entries, [out]))
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
        h = _calls_make((n, 2 * n, [0.01, 0.9, -0.1], [
            _cl_entry(_cl_vector(torch._foreach_add), [grads, params, ("const", 0)], ("alpha",), out=0, flags=["pure", "fresh"], asserts=[("count", n)]),
            _cl_entry(_cl_vector(torch._foreach_mul_), [bufs, ("const", 1)], at=1, flags=[("writes", (0,))]),
            _cl_entry(_cl_vector(torch._foreach_add_), [bufs, decayed], at=2, flags=[("writes", (0,))], kills=[0]),
            _cl_entry(_cl_vector(torch._foreach_add_), [params, bufs, ("const", 2)], ("alpha",), at=3, flags=[("writes", (0,))]),
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
    say they write (mapped to the call's operands), whether an operator tagged
    as drawing or as handing out a value ran at any depth, whether Python was
    entered. The rules: it changes nothing a program computes; what it cannot
    classify is opaque, and opaque is everything at once; what no schema
    declares it does not see, and the tests say so.
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
            {"ops": ["aten::add.Tensor"], "below": 0, "wrote": [], "opaque": False, "why": "", "drew": False,
             "read_value": False, "read_value_top": False, "entered_python": False, "pure": True},
        )  # fmt: skip

    def test_a_declared_write_is_mapped_to_the_operand(self, device):
        a, b = self.pair(device)
        _, d = _witness(torch.Tensor.add_, a, b)
        self.assertEqual(
            (d["ops"], d["wrote"]), (["aten::add_.Tensor"], [(0, 0, 0, False)])
        )
        self.assertFalse(d["pure"])
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
        self.assertTrue(d["pure"])

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
        self.assertEqual((d["ops"], d["pure"]), (["aten::batch_norm"], True))

    def test_draws(self, device):
        a, _ = self.pair(device)
        _, d = _witness(torch.dropout, a, 0.5, True)
        self.assertEqual(
            (d["ops"], d["drew"], d["pure"]), (["aten::dropout"], True, False)
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
                (d["ops"], d["read_value_top"], d["pure"]), ([op], True, False)
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
        self.assertEqual((len(r), d["read_value"], d["pure"]), (3, False, True))

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
        self.assertEqual((d["entered_python"], d["pure"]), (True, False))
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
            "entered_python": True,
            "pure": False,
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
RULES_MODULE = """import sys, types
def module(k):
    m = types.ModuleType('vm_rules_mod')
    m.K = k
    sys.modules['vm_rules_mod'] = m
module(2.0)"""
RULES_MODULE_CHANGE = "module(5.0 if second else 2.0)"
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
    "import_name": (
        ("IMPORT_NAME",),
        RULES_MODULE,
        "import vm_rules_mod\nk = vm_rules_mod.K",
        RULES_MODULE_CHANGE,
        "None",
    ),
    "import_from": (
        ("IMPORT_FROM",),
        RULES_MODULE,
        "from vm_rules_mod import K\nk = K",
        RULES_MODULE_CHANGE,
        "None",
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


def rules_source(name):
    _, setup, construct, change, state = RULES_STOPS[name]
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
                self.assertIn(family, ("ops", *RULES_LATER))
                dumped.add(
                    (what, how if how in ("row", "stop") else f"{how}:{rest[0]}")
                )
                if how == "later":
                    self.assertIn(rest[0], RULES_LATER)
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


if __name__ == "__main__":
    run_tests()
