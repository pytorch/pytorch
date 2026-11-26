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
import enum
import functools
import glob
import json
import os
import re
import struct
import subprocess
import sys
import traceback
import unittest
import warnings

import numpy

import torch
from torch.overrides import TorchFunctionMode
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    _restore_fp32_precision,
    _snapshot_fp32_precision,
    DeterministicGuard,
    HardwareClassification,
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


if __name__ == "__main__":
    run_tests()
