# Owner(s): ["module: ci"]

import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import torch
from torch.testing._internal import tci
from torch.testing._internal.common_utils import (
    HardwareClassification,
    IS_SANDCASTLE,
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TEST_CUDA,
    TEST_WITH_CROSSREF,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.testing._internal.tci import machine, plugin
from torch.testing._internal.torchci import environment


L4 = tci.Machine("cuda", 1, "NVIDIA L4", 22.5, (8, 9))
MACHINES = {
    "cpu": tci.Machine(),
    "l4": L4,
    "a10g": tci.Machine("cuda", 1, "NVIDIA A10G", 22.5, (8, 6)),
    "h100x2": tci.Machine("cuda", 2, "NVIDIA H100 80GB HBM3", 79.6, (9, 0)),
    "mi300x": tci.Machine("rocm", 1, "AMD Instinct MI300X", 191.7),
    "mi350x4": tci.Machine("rocm", 4, "AMD Instinct MI350X VF", 287.6),
    "xpu": tci.Machine("xpu", 1, "Intel(R) Data Center GPU Max 1100"),
    "m1": tci.Machine("mps", 1, "Apple M1"),
}
GPUS = "l4 a10g h100x2 mi300x mi350x4 xpu m1"
# A test file, and the conftest.py that makes the machine it is collected on an L4
SAMPLE = """\
import unittest

import pytest

from torch.testing._internal import tci


@tci.only(tci.gpu)
class Base(unittest.TestCase):
    pass


@tci.exclude(tci.rocm, reason="wrong results")
class TestSample(Base):
    def test_kept(self): ...

    @tci.needs(tci.gpus >= 2)
    @tci.only(tci.cuda.h100)
    def test_two_gpus(self): ...


class TestPlain(unittest.TestCase):
    def test_plain(self): ...

    @tci.only(tci.cpu)
    def test_cpu(self): ...


@tci.needs(tci.gpu_memory >= "32GB")
@pytest.mark.parametrize("n", [1, 2])
def test_function(n): ...
"""
CONFTEST = """\
from torch.testing._internal import tci
from torch.testing._internal.tci import machine

machine.capture = lambda: tci.Machine("cuda", 1, "NVIDIA L4", 22.5, (8, 9))
"""


def runs_on(*decorators: Callable[[Any], Any], cls: type | None = None) -> str:
    """The MACHINES that a test under these decorators, in this class, can run on."""

    def test() -> None: ...

    for decorate in reversed(decorators):
        decorate(test)
    found = tci.constraints(cls, test)
    met = [name for name, m in MACHINES.items() if all(c.met_by(m) for c in found)]
    return " ".join(met)


class TestConstraints(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_only(self) -> None:
        self.assertEqual(runs_on(), f"cpu {GPUS}")
        self.assertEqual(runs_on(tci.only(tci.cpu)), "cpu")
        self.assertEqual(runs_on(tci.only(tci.gpu)), GPUS)
        self.assertEqual(runs_on(tci.only(tci.cuda)), "l4 a10g h100x2")
        self.assertEqual(runs_on(tci.only(tci.rocm, tci.xpu)), "mi300x mi350x4 xpu")
        self.assertEqual(runs_on(tci.only(tci.cpu | tci.mps)), "cpu m1")
        # Each only applies: a GPU, and NVIDIA's or none
        both = runs_on(tci.only(tci.gpu), tci.only(tci.cuda, tci.cpu))
        self.assertEqual(both, "l4 a10g h100x2")

    def test_devices(self) -> None:
        self.assertEqual(runs_on(tci.only(tci.cuda.h100)), "h100x2")
        self.assertEqual(runs_on(tci.only(tci.cuda.a10g, tci.cuda.l4)), "l4 a10g")
        self.assertEqual(runs_on(tci.only(tci.cuda.a10)), "")  # an A10G is not one
        self.assertEqual(runs_on(tci.only(tci.rocm.mi350x | tci.mps.m1)), "mi350x4 m1")
        hopper = tci.cuda(min_capability="9.0")
        self.assertEqual(runs_on(tci.only(hopper)), "h100x2")
        self.assertEqual(runs_on(tci.only(tci.FP8_GPU)), "l4 h100x2 mi300x mi350x4")
        with self.assertRaisesRegex(AttributeError, "tci.cuda has no 'h10' among"):
            tci.only(tci.cuda.h10)
        with self.assertRaisesRegex(ValueError, "min_capability is like"):
            tci.cuda(min_capability="9")

    def test_exclude(self) -> None:
        not_rocm = tci.exclude(tci.rocm, reason="wrong results")
        self.assertEqual(runs_on(not_rocm), "cpu l4 a10g h100x2 xpu m1")
        self.assertEqual(runs_on(tci.only(tci.gpu), not_rocm), "l4 a10g h100x2 xpu m1")
        some = tci.exclude(tci.cuda.l4, tci.cpu | tci.rocm, reason="too small")
        self.assertEqual(runs_on(some), "a10g h100x2 xpu m1")
        with self.assertRaisesRegex(TypeError, "reason"):
            tci.exclude(tci.rocm)
        with self.assertRaisesRegex(ValueError, "takes a reason"):
            tci.exclude(tci.rocm, reason=" ")

    def test_needs(self) -> None:
        self.assertEqual(runs_on(tci.needs(tci.gpus >= 2)), "h100x2 mi350x4")
        self.assertEqual(runs_on(tci.needs(2 <= tci.gpus)), "h100x2 mi350x4")
        self.assertEqual(runs_on(tci.needs(tci.gpus == 1)), "l4 a10g mi300x xpu m1")
        self.assertEqual(runs_on(tci.needs(tci.gpus < 1)), "cpu")
        # on= is the if: machines outside it aren't asked
        if_gpu = tci.needs(tci.gpus >= 2, on=tci.gpu)
        self.assertEqual(runs_on(if_gpu), "cpu h100x2 mi350x4")
        if_rocm = tci.needs(tci.gpus <= 2, on=tci.rocm)
        self.assertEqual(runs_on(if_rocm), "cpu l4 a10g h100x2 mi300x xpu m1")
        # All of each GPU's memory. Nothing reports an Intel or an Apple GPU's
        roomy = tci.needs(tci.gpu_memory >= "32GB")
        self.assertEqual(runs_on(roomy), "h100x2 mi300x mi350x4 xpu m1")
        both = tci.needs(tci.gpus >= 2, tci.gpu_memory > "80GB")
        self.assertEqual(runs_on(both), "mi350x4")
        with self.assertRaisesRegex(TypeError, "one comparison at a time"):
            tci.needs(1 <= tci.gpus <= 2)
        with self.assertRaisesRegex(ValueError, "a whole number, not 2.5"):
            tci.needs(tci.gpus >= 2.5)
        with self.assertRaisesRegex(ValueError, 'a size like "32GB": 32'):
            tci.needs(tci.gpu_memory >= 32)

    def test_arguments(self) -> None:
        mistakes: list[tuple[Callable[[], Any], str]] = [
            (lambda: tci.only(), "only() takes groups"),
            (lambda: tci.only("cuda"), "only() takes groups"),
            (lambda: tci.only(tci.gpus >= 2), "only() takes groups"),
            (lambda: tci.needs(), "needs() takes comparisons"),
            (lambda: tci.needs(tci.gpu), "needs() takes comparisons"),
            (lambda: tci.needs(tci.gpus >= 2, on="gpu"), "on= takes a group"),
        ]
        for mistake, message in mistakes:
            with self.subTest(message), self.assertRaises(TypeError) as raised:
                mistake()
            self.assertIn(message, str(raised.exception))

    def test_levels(self) -> None:
        @tci.only(tci.gpu)
        class Base:
            pass

        @tci.needs(tci.gpus >= 2)
        @tci.exclude(tci.rocm, reason="wrong results")
        class Sample(Base):
            @tci.needs(tci.gpus <= 2)
            def test_few(self) -> None: ...

            def test_plain(self) -> None: ...

        written = [c.code for c in tci.constraints(Sample, Sample.test_few)]
        expected = [
            "@tci.only(tci.gpu)",
            "@tci.needs(tci.gpus >= 2)",
            '@tci.exclude(tci.rocm, reason="wrong results")',
            "@tci.needs(tci.gpus <= 2)",
        ]
        self.assertEqual(written, expected)
        self.assertEqual(len(tci.constraints(Sample, Sample.test_plain)), 3)
        self.assertEqual(len(tci.constraints(Base, None)), 1)  # not its subclass's
        self.assertEqual(tci.constraints(None, print), [])
        only_four = tci.only(tci.rocm.mi350x)
        self.assertEqual(runs_on(cls=Sample), "h100x2")
        self.assertEqual(runs_on(only_four, cls=Base), "mi350x4")


class TestMachine(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def capture(self, **facts: Any) -> tci.Machine:
        """The machine tci makes of what torchci captures, with these facts changed
        and, for NVML to report, a compute capability of 9.0."""
        captured = dataclasses.replace(environment.capture(), **facts)
        with (
            mock.patch.object(environment, "capture", return_value=captured),
            mock.patch.object(machine, "_cuda_capability", return_value=(9, 0)),
        ):
            return machine.capture.__wrapped__()

    def test_capture(self) -> None:
        h100 = {"device_name": "NVIDIA H100 80GB HBM3", "device_count": 2}
        memory = {"properties": {"device_memory_mib": "81559"}}
        captured = self.capture(accelerator="cuda", **h100, **memory)
        gib = 81559 / 1024
        expected = tci.Machine("cuda", 2, "NVIDIA H100 80GB HBM3", gib, (9, 0))
        self.assertEqual(captured, expected)
        # Only CUDA has a compute capability
        captured = self.capture(accelerator="rocm", **h100, **memory)
        rocm = dataclasses.replace(expected, accelerator="rocm", capability=(0, 0))
        self.assertEqual(captured, rocm)
        # A CUDA build on a machine without a GPU
        no_gpu = {"device_name": "", "device_count": 0, "properties": {}}
        self.assertEqual(self.capture(accelerator="cuda", **no_gpu), tci.Machine())
        # Nothing reports an Apple GPU's memory
        m1 = {"device_name": "Apple M1", "device_count": 1, "properties": {}}
        self.assertEqual(self.capture(accelerator="mps", **m1), MACHINES["m1"])
        # NVML does report an NVIDIA GPU's, so a blank is a read that failed
        no_memory, no_name = {"properties": {}}, {**memory, "device_name": ""}
        for unread in [no_memory, no_name]:
            with self.assertRaisesRegex(RuntimeError, "can't read the cuda GPU's"):
                self.capture(accelerator="cuda", **{**h100, **unread})


@unittest.skipUnless(TEST_CUDA and not TEST_WITH_ROCM, "needs an NVIDIA GPU")
class TestMachineCUDA(TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_capture(self) -> None:
        captured = machine.capture.__wrapped__()
        self.assertEqual(captured.accelerator, "cuda")
        self.assertEqual(captured.gpus, torch.cuda.device_count())
        self.assertEqual(captured.device, torch.cuda.get_device_name(0))
        self.assertEqual(captured.capability, torch.cuda.get_device_capability(0))
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        # NVML counts what the driver reserves; CUDA doesn't
        self.assertLess(abs(captured.gpu_memory - total), 2)


class TestPlugin(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_reads_the_machine_only_for_a_constraint(self) -> None:
        def test() -> None: ...

        item, config = mock.Mock(obj=test), mock.Mock()
        item.cls = None
        config.getoption.return_value = None
        items = [item]
        with mock.patch.object(machine, "capture", side_effect=AssertionError):
            plugin.pytest_collection_modifyitems(config, items)
        self.assertEqual(items, [item])
        config.hook.pytest_deselected.assert_not_called()
        tci.only(tci.cpu)(test)
        with mock.patch.object(machine, "capture", return_value=L4):
            plugin.pytest_collection_modifyitems(config, items)
        self.assertEqual(items, [])
        config.hook.pytest_deselected.assert_called_once_with(items=[item])


@unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
@unittest.skipIf(IS_SANDCASTLE, "Skipping because doesn't work on sandcastle")
@skipIfTorchDynamo("subprocess test does not need Dynamo coverage")
@unittest.skipIf(TEST_WITH_CROSSREF, "subprocess test does not need crossref coverage")
class TestPluginCollects(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def collect(self, *args: str) -> tuple[list[str], dict[str, str]]:
        """SAMPLE's tests that pytest keeps on an L4 with the plugin and these
        arguments, and what the plugin writes of those it deselects."""
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "test_sample.py").write_text(SAMPLE)
            Path(tmp, "conftest.py").write_text(CONFTEST)
            plugins = ["-p", plugin.__name__, "-p", "no:cacheprovider"]
            wrote = ["--tci-deselected=deselected.json"]
            pytest = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
            command = [*pytest, *plugins, *wrote, *args]
            done = subprocess.run(command, cwd=tmp, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            deselected = json.loads(Path(tmp, "deselected.json").read_text())
        kept = [line.partition("::")[2] for line in done.stdout.splitlines()]
        return list(filter(None, kept)), deselected

    def test_machine_module_prints_the_machine(self) -> None:
        printed = subprocess.check_output([sys.executable, "-m", machine.__name__])
        fields = [field.name for field in dataclasses.fields(tci.Machine)]
        self.assertEqual(list(json.loads(printed.splitlines()[-1])), fields)

    def test_deselects_what_the_machine_cant_run(self) -> None:
        kept, deselected = self.collect()
        self.assertEqual(kept, ["TestSample::test_kept", "TestPlain::test_plain"])
        expected = {
            "test_sample.py::TestSample::test_two_gpus": "@tci.needs(tci.gpus >= 2)",
            "test_sample.py::TestPlain::test_cpu": "@tci.only(tci.cpu)",
            "test_sample.py::test_function[1]": '@tci.needs(tci.gpu_memory >= "32GB")',
            "test_sample.py::test_function[2]": '@tci.needs(tci.gpu_memory >= "32GB")',
        }
        self.assertEqual(deselected, expected)

    def test_judges_only_what_other_filters_keep(self) -> None:
        kept, deselected = self.collect("-k", "TestPlain")
        self.assertEqual(kept, ["TestPlain::test_plain"])
        cpu = {"test_sample.py::TestPlain::test_cpu": "@tci.only(tci.cpu)"}
        self.assertEqual(deselected, cpu)


if __name__ == "__main__":
    run_tests()
