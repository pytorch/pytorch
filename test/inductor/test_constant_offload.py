# Owner(s): ["module: inductor"]

import glob
import os
import types
import unittest
from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.codegen import wrapper
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.virtualized import V
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON


_FRACTION = "aot_inductor.autotune_offload_constants_min_device_fraction"

_UNSET = object()


def _graph(constants, is_const_graph=False, aot_mode=True, current_device=_UNSET):
    # current_device mirrors GraphLowering: a real torch.device while a
    # device-specific kernel is being codegen'd, None otherwise.
    if current_device is _UNSET:
        current_device = torch.device(
            GPU_TYPE, torch.accelerator.current_device_index()
        )
    return types.SimpleNamespace(
        constants=constants,
        is_const_graph=is_const_graph,
        aot_mode=aot_mode,
        current_device=current_device,
    )


def _force_spill():
    return config.patch({_FRACTION: 0.0})


def _force_skip():
    return config.patch({_FRACTION: 1.0})


@unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
class ConstantOffloadTest(TestCase):
    def _constants(self):
        return {
            "w": torch.randn(64, 128, device=GPU_TYPE),
            "b": torch.randn(128, device=GPU_TYPE, dtype=torch.float16),
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
            self.assertEqual(tensor.device.type, GPU_TYPE)
            self.assertEqual(tensor.dtype, expected[name].dtype)
            self.assertEqual(tensor.shape, expected[name].shape)

    def test_device_memory_drops_inside_the_block(self):
        constants = {"w": torch.randn(1024, 1024, device=GPU_TYPE)}
        graph = _graph(constants)
        before = torch.accelerator.memory_allocated()

        with _force_spill(), V.set_graph_handler(graph):
            with wrapper._constants_offloaded_to_disk():
                inside = torch.accelerator.memory_allocated()

        self.assertLess(inside, before)
        self.assertEqual(torch.accelerator.memory_allocated(), before)

    def test_aliased_constants_are_spilled_once(self):
        shared = torch.randn(32, 32, device=GPU_TYPE)
        graph = _graph({"a": shared, "b": shared, "c": shared.view(-1)})

        with V.set_graph_handler(graph):
            targets, total = wrapper._constant_offload_targets(shared.device)

        self.assertEqual(len(targets), 1)
        self.assertEqual(total, shared.untyped_storage().nbytes())

    def test_skipped_below_device_fraction(self):
        constants = self._constants()
        graph = _graph(constants)

        with _force_skip(), V.set_graph_handler(graph):
            with wrapper._constants_offloaded_to_disk():
                inside = [t.untyped_storage().nbytes() for t in constants.values()]

        self.assertTrue(all(n > 0 for n in inside))

    def test_default_config_does_not_offload(self):
        # The offload is opt-in: the shipped default must never fire, since
        # constants cannot occupy 100% of the card they already live on.
        self.assertGreaterEqual(
            config.aot_inductor.autotune_offload_constants_min_device_fraction, 1.0
        )
        constants = self._constants()
        graph = _graph(constants)

        with V.set_graph_handler(graph):  # no config patch: shipped default
            with wrapper._constants_offloaded_to_disk():
                inside = [t.untyped_storage().nbytes() for t in constants.values()]

        self.assertTrue(all(n > 0 for n in inside))

    def test_skipped_when_no_current_device(self):
        # current_device is unset outside device-specific kernel codegen -- the
        # cpp_wrapper path reaches here that way. That must skip the offload,
        # not raise.
        constants = self._constants()
        graph = _graph(constants, current_device=None)

        with _force_spill(), V.set_graph_handler(graph):
            with wrapper._constants_offloaded_to_disk():
                inside = [t.untyped_storage().nbytes() for t in constants.values()]

        self.assertTrue(all(n > 0 for n in inside))

    def test_skipped_for_cpu_graph(self):
        constants = self._constants()
        graph = _graph(constants, current_device=torch.device("cpu"))

        with _force_spill(), V.set_graph_handler(graph):
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
        pattern = os.path.join(cache_dir(), "inductor_const_spill_*")

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

    def test_failed_unlink_does_not_leave_the_spill_file_behind(self):
        # from_file creates the file; if the unlink that follows it fails, the
        # spill is abandoned and nothing else would remove a file the size of
        # the whole constant set.
        constants = self._constants()
        graph = _graph(constants)
        pattern = os.path.join(cache_dir(), "inductor_const_spill_*")
        real_unlink = os.unlink
        failures = []

        def failing_unlink(path, *a, **k):
            # Fail only the unlink inside the spill, so the cleanup retry can
            # still succeed -- otherwise the test asserts something the code
            # cannot do.
            if "inductor_const_spill_" in str(path) and not failures:
                failures.append(path)
                raise OSError("unlink refused")
            return real_unlink(path, *a, **k)

        with (
            _force_spill(),
            V.set_graph_handler(graph),
            mock.patch.object(os, "unlink", failing_unlink),
        ):
            with wrapper._constants_offloaded_to_disk():
                pass

        # The spill was abandoned, so the constants stay on device ...
        self.assertTrue(
            all(t.untyped_storage().nbytes() > 0 for t in constants.values())
        )
        # ... and no file is left behind once the real unlink is back.
        self.assertTrue(failures, "the unlink under test never ran")
        self.assertEqual(glob.glob(pattern), [])

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
                torch.accelerator,
                "get_memory_info",
                wraps=torch.accelerator.get_memory_info,
            ) as get_memory_info,
        ):
            with wrapper._constants_offloaded_to_disk():
                pass

        self.assertTrue(get_memory_info.called)
        self.assertEqual(get_memory_info.call_args.args[0], constants["w"].device)

    def test_failure_while_freeing_restores_what_was_already_freed(self):
        # The spill can succeed and the free loop still fail partway, leaving
        # some storages at zero bytes. Those must be put back.
        constants = self._constants()
        expected = {k: v.cpu().clone() for k, v in constants.items()}
        graph = _graph(constants)
        real_resize = torch.UntypedStorage.resize_
        zero_resizes = 0

        def flaky_resize(storage, size):
            nonlocal zero_resizes
            if size == 0:
                zero_resizes += 1
                if zero_resizes == 2:
                    raise RuntimeError("cannot resize")
            return real_resize(storage, size)

        with (
            _force_spill(),
            V.set_graph_handler(graph),
            mock.patch.object(torch.UntypedStorage, "resize_", flaky_resize),
            self.assertRaisesRegex(RuntimeError, "cannot resize"),
        ):
            with wrapper._constants_offloaded_to_disk():
                pass

        self.assertEqual(zero_resizes, 2)
        for name, tensor in constants.items():
            self.assertEqual(tensor.cpu(), expected[name])

    def test_non_resizable_constants_are_not_selected(self):
        # resize_ raises on these even for a zero-size resize, so they have to
        # be filtered up front rather than discovered mid-free.
        graph = _graph(self._constants())
        device = torch.device(GPU_TYPE, torch.accelerator.current_device_index())

        with V.set_graph_handler(graph):
            self.assertTrue(wrapper._constant_offload_targets(device)[0])
            with mock.patch.object(
                torch.UntypedStorage, "resizable", lambda self: False
            ):
                self.assertEqual(wrapper._constant_offload_targets(device)[0], [])

    def test_jit_compile_is_not_offloaded(self):
        # autotune_at_compile_time is reachable outside AOT, where the config
        # this is gated on does not apply.
        constants = self._constants()
        graph = _graph(constants, aot_mode=False)

        with _force_spill(), V.set_graph_handler(graph):
            with wrapper._constants_offloaded_to_disk():
                inside = [t.untyped_storage().nbytes() for t in constants.values()]

        self.assertTrue(all(n > 0 for n in inside))

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
