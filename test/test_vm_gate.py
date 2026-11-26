# Owner(s): ["module: dynamo"]

"""Tests of the soundness contract of torch.vm's core: what a region checks
before it runs, and what the engine may assume of a traced call.

Every class here states one rule of the contract and tests it over the whole
domain the rule speaks about (every field of what is compared for a tensor,
every instruction the engine knows), not over the cases somebody thought of.
"""

import ast
import collections
import glob
import json
import os
import re
import subprocess
import sys

import torch
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


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


if __name__ == "__main__":
    run_tests()
