# Owner(s): ["module: inductor"]

import glob
import os
import tempfile
import types
import unittest
from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.codegen import wrapper
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.virtualized import V

_FRACTION = "aot_inductor.autotune_offload_constants_min_device_fraction"


def _graph(constants, is_const_graph=False):
    return types.SimpleNamespace(constants=constants, is_const_graph=is_const_graph)


def _force_spill():
    return config.patch({_FRACTION: 0.0})


def _force_skip():
    return config.patch({_FRACTION: 1.0})


@unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
class ConstantOffloadTest(TestCase):
    def _constants(self):
        return {
            "w": torch.randn(64, 128, device="cuda"),
            "b": torch.randn(128, device="cuda", dtype=torch.float16),
        }

    def test_constants_are_freed_then_restored_exactly(self):
        constants = self._constants()
        expected = {k: v.cpu().clone() for k, v in constants.items()}
        graph = _graph(constants)

        with _force_spill(), V.set_graph_handler(graph):
            with wrapper._constants_offloaded_to_disk():
                freed = [t.untyped_storage().nbytes() for t in constants.values()]

        self.assertEqual(freed, [0, 0])
        for name, tensor in constants.items():
            self.assertEqual(tensor.cpu(), expected[name])
            self.assertEqual(tensor.device.type, "cuda")
            self.assertEqual(tensor.dtype, expected[name].dtype)
            self.assertEqual(tensor.shape, expected[name].shape)

    def test_device_memory_drops_inside_the_block(self):
        constants = {"w": torch.randn(1024, 1024, device="cuda")}
        graph = _graph(constants)
        before = torch.cuda.memory_allocated()

        with _force_spill(), V.set_graph_handler(graph):
            with wrapper._constants_offloaded_to_disk():
                inside = torch.cuda.memory_allocated()

        self.assertLess(inside, before)
        self.assertEqual(torch.cuda.memory_allocated(), before)

    def test_aliased_constants_are_spilled_once(self):
        shared = torch.randn(32, 32, device="cuda")
        graph = _graph({"a": shared, "b": shared, "c": shared.view(-1)})

        with V.set_graph_handler(graph):
            targets, total = wrapper._constant_offload_targets()

        self.assertEqual(len(targets), 1)
        self.assertEqual(total, shared.untyped_storage().nbytes())

    def test_skipped_below_device_fraction(self):
        constants = self._constants()
        graph = _graph(constants)

        with _force_skip(), V.set_graph_handler(graph):
            with wrapper._constants_offloaded_to_disk():
                inside = [t.untyped_storage().nbytes() for t in constants.values()]

        self.assertTrue(all(n > 0 for n in inside))

    def test_skipped_for_const_graph(self):
        constants = self._constants()
        graph = _graph(constants, is_const_graph=True)

        with _force_spill(), V.set_graph_handler(graph):
            with wrapper._constants_offloaded_to_disk():
                inside = [t.untyped_storage().nbytes() for t in constants.values()]

        self.assertTrue(all(n > 0 for n in inside))

    def test_spill_file_is_unlinked_while_mapped(self):
        graph = _graph(self._constants())
        pattern = os.path.join(tempfile.gettempdir(), "inductor_const_spill_*")

        with _force_spill(), V.set_graph_handler(graph):
            with wrapper._constants_offloaded_to_disk():
                leaked = glob.glob(pattern)

        self.assertEqual(leaked, [])

    def test_failed_spill_leaves_constants_on_device(self):
        constants = self._constants()
        expected = {k: v.cpu().clone() for k, v in constants.items()}
        graph = _graph(constants)

        with (
            _force_spill(),
            V.set_graph_handler(graph),
            mock.patch.object(
                torch.UntypedStorage, "from_file", side_effect=OSError("no space")
            ),
        ):
            with wrapper._constants_offloaded_to_disk():
                inside = [t.untyped_storage().nbytes() for t in constants.values()]

        self.assertTrue(all(n > 0 for n in inside))
        for name, tensor in constants.items():
            self.assertEqual(tensor.cpu(), expected[name])

    def test_restore_failure_does_not_mask_the_block_exception(self):
        graph = _graph(self._constants())

        with (
            _force_spill(),
            V.set_graph_handler(graph),
            mock.patch.object(
                wrapper, "_restore_constants", side_effect=RuntimeError("cuda gone")
            ),
        ):
            # The block's own failure is the cause worth surfacing; a restore
            # that fails alongside it must not replace it.
            with self.assertRaisesRegex(ValueError, "original failure"):
                with wrapper._constants_offloaded_to_disk():
                    raise ValueError("original failure")

    def test_gate_measures_the_device_the_constants_are_on(self):
        constants = self._constants()
        graph = _graph(constants)

        with (
            V.set_graph_handler(graph),
            mock.patch.object(
                torch.cuda, "mem_get_info", wraps=torch.cuda.mem_get_info
            ) as mem_get_info,
        ):
            with wrapper._constants_offloaded_to_disk():
                pass

        self.assertTrue(mem_get_info.called)
        self.assertEqual(
            mem_get_info.call_args.args[0], constants["w"].device
        )

    def test_failed_restore_raises_rather_than_returning_bad_constants(self):
        graph = _graph(self._constants())

        with (
            _force_spill(),
            V.set_graph_handler(graph),
            mock.patch.object(
                wrapper, "_restore_constants", side_effect=RuntimeError("cuda gone")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "no longer usable"):
                with wrapper._constants_offloaded_to_disk():
                    pass


if __name__ == "__main__":
    run_tests()
