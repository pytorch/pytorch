# Owner(s): ["oncall: jit"]

import io
import threading
import unittest
import warnings

import torch
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
    run_tests,
    TestCase,
)


REMOVED_PUBLIC_APIS = (
    "fork",
    "freeze",
    "interface",
    "load",
    "optimize_for_inference",
    "save",
    "script",
    "script_method",
    "set_fusion_strategy",
    "trace",
    "trace_module",
    "wait",
)


class TestTorchScriptCompatibility(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @unittest.skipIf(IS_FBCODE, "The internal distribution retains these entrypoints")
    @parametrize("name", REMOVED_PUBLIC_APIS)
    def test_retired_public_exports(self, name):
        self.assertNotIn(name, torch.jit.__all__)
        self.assertFalse(hasattr(torch.jit, name))

    def test_retained_type_aliases(self):
        self.assertIs(torch.jit.ScriptModule, torch.jit._script.ScriptModule)
        self.assertIs(torch.jit.CompilationUnit, torch.jit._script.CompilationUnit)

    def test_compilation_unit(self):
        unit = torch.jit.CompilationUnit(
            "def twice(x: Tensor) -> Tensor:\n    return x + x\n"
        )
        value = torch.tensor([1.0, 2.0])
        self.assertEqual(unit.twice(value), value + value)

    def test_annotate_is_passthrough(self):
        value = [1, 2]
        self.assertIs(torch.jit.annotate(list[int], value), value)

    def test_ignore_preserves_eager_function(self):
        def increment(value):
            return value + 1

        ignored = torch.jit.ignore(increment)
        self.assertIs(ignored, increment)
        self.assertEqual(ignored(3), 4)

    def test_shared_future(self):
        future = torch.futures.Future()
        value = torch.tensor([3.0])
        future.set_result(value)
        self.assertEqual(future.wait(), value)

    @staticmethod
    def _script_archive():
        unit = torch.jit.CompilationUnit(
            "def increment(x: Tensor) -> Tensor:\n    return x + 1\n"
        )
        return io.BytesIO(unit.increment.save_to_buffer())

    def test_legacy_archive_compatibility(self):
        archive = self._script_archive()
        if not IS_FBCODE:
            for weights_only in (True, False):
                archive.seek(0)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "TorchScript archives are no longer supported by torch.load",
                ):
                    torch.load(archive, weights_only=weights_only)
            return
        with self.assertRaisesRegex(RuntimeError, "weights_only=True"):
            torch.load(archive, weights_only=True)
        archive.seek(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", UserWarning)
            loaded = torch.load(archive, weights_only=False)
        self.assertEqual(loaded(torch.tensor([2.0])), torch.tensor([3.0]))

    def test_thread_safe_error_stacks(self):
        callstacks = []

        def callstack_creator():
            factory = torch._C._jit_tree_views.SourceRangeFactory(
                "source code", "a.py", 1, 0
            )
            callstack = torch._C.CallStack("a", factory.make_range(1, 0, 1))
            callstacks.append(callstack)

        thread = threading.Thread(target=callstack_creator)
        thread.start()
        thread.join()
        self.assertEqual(len(callstacks), 1)
        del callstacks[0]
        self.assertEqual(callstacks, [])

    def test_ordinary_checkpoint_does_not_warn(self):
        archive = io.BytesIO()
        value = {"weight": torch.tensor([1.0])}
        torch.save(value, archive)
        archive.seek(0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(torch.load(archive, weights_only=True), value)
        self.assertEqual(caught, [])


instantiate_parametrized_tests(TestTorchScriptCompatibility)


if __name__ == "__main__":
    run_tests()
