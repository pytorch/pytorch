# Owner(s): ["module: functorch"]

from __future__ import annotations

from unittest.mock import patch

import torch
from torch._functorch._aot_autograd import utils as aot_utils
from torch._functorch._aot_autograd.utils import _get_autocast_states
from torch.testing._internal.common_utils import run_tests, TestCase


class TestAOTAutocastStates(TestCase):
    def test_get_autocast_states_includes_supported_device(self) -> None:
        with (
            patch.object(
                aot_utils.torch._C,
                "_autocast_supported_devices",
                return_value=["cpu", "privateuseone"],
            ),
            patch.object(aot_utils.torch, "is_autocast_enabled") as enabled,
            patch.object(aot_utils.torch, "get_autocast_dtype") as dtype,
            patch.object(
                aot_utils.torch, "is_autocast_cache_enabled", return_value=False
            ) as cache_enabled,
        ):
            enabled.side_effect = lambda device: device == "privateuseone"
            dtype.side_effect = lambda device: {
                "cpu": torch.bfloat16,
                "privateuseone": torch.float16,
            }[device]

            states = _get_autocast_states()

            self.assertEqual(
                states,
                [
                    False,
                    torch.bfloat16,
                    True,
                    torch.float16,
                    cache_enabled.return_value,
                ],
            )

    def test_get_autocast_states_reads_without_top_level_torch_module(self) -> None:
        device_type = "autocast_only_device"
        if hasattr(torch, device_type):
            self.skipTest(f"torch.{device_type} unexpectedly exists")
        with (
            patch.object(
                aot_utils.torch._C,
                "_autocast_supported_devices",
                return_value=["cpu", device_type],
            ),
            patch.object(aot_utils.torch, "is_autocast_enabled") as enabled,
            patch.object(aot_utils.torch, "get_autocast_dtype") as dtype,
            patch.object(
                aot_utils.torch, "is_autocast_cache_enabled", return_value=False
            ),
        ):
            enabled.side_effect = lambda device: device == device_type
            dtype.side_effect = lambda device: {
                "cpu": torch.float32,
                device_type: torch.float16,
            }[device]

            states = _get_autocast_states()

            enabled.assert_any_call(device_type)
            dtype.assert_any_call(device_type)
            self.assertEqual(
                states,
                [False, torch.float32, True, torch.float16, False],
            )

    def test_get_autocast_states_appends_cache_enabled_once(self) -> None:
        with (
            patch.object(
                aot_utils.torch._C,
                "_autocast_supported_devices",
                return_value=["cpu", "cuda"],
            ),
            patch.object(aot_utils.torch, "is_autocast_enabled", return_value=False),
            patch.object(
                aot_utils.torch, "get_autocast_dtype", return_value=torch.float32
            ),
            patch.object(
                aot_utils.torch, "is_autocast_cache_enabled", return_value=True
            ),
        ):
            states = _get_autocast_states()
            self.assertEqual(states.count(True), 1)
            self.assertEqual(states[-1], True)

    def test_get_autocast_states_preserves_supported_device_order(self) -> None:
        with (
            patch.object(
                aot_utils.torch._C,
                "_autocast_supported_devices",
                return_value=["cpu", "cuda", "xpu"],
            ),
            patch.object(aot_utils.torch, "is_autocast_enabled") as enabled,
            patch.object(aot_utils.torch, "get_autocast_dtype") as dtype,
            patch.object(
                aot_utils.torch, "is_autocast_cache_enabled", return_value=False
            ),
        ):
            enabled.side_effect = lambda device: device != "cpu"
            dtype.side_effect = lambda device: {
                "cpu": torch.float32,
                "cuda": torch.bfloat16,
                "xpu": torch.float16,
            }[device]

            self.assertEqual(
                _get_autocast_states(),
                [
                    False,
                    torch.float32,
                    True,
                    torch.bfloat16,
                    True,
                    torch.float16,
                    False,
                ],
            )

    def test_get_autocast_states_includes_renamed_privateuse1_backend(self) -> None:
        with (
            patch.object(
                aot_utils.torch._C,
                "_autocast_supported_devices",
                return_value=["cpu", "npu"],
            ),
            patch.object(
                aot_utils.torch._C,
                "_get_privateuse1_backend_name",
                return_value="npu",
            ),
            patch.object(aot_utils.torch, "is_autocast_enabled") as enabled,
            patch.object(aot_utils.torch, "get_autocast_dtype") as dtype,
            patch.object(
                aot_utils.torch, "is_autocast_cache_enabled", return_value=True
            ),
        ):
            enabled.side_effect = lambda device: device == "npu"
            dtype.side_effect = lambda device: {
                "cpu": torch.float32,
                "npu": torch.float16,
            }[device]

            self.assertEqual(
                _get_autocast_states(),
                [False, torch.float32, True, torch.float16, True],
            )

    def test_get_autocast_states_reflects_live_supported_devices(self) -> None:
        expected: list[object] = []
        for device_type in torch._C._autocast_supported_devices():
            expected.append(torch.is_autocast_enabled(device_type))
            expected.append(torch.get_autocast_dtype(device_type))
        expected.append(torch.is_autocast_cache_enabled())
        self.assertEqual(_get_autocast_states(), expected)

    def test_legacy_cuda_cpu_snapshot_misses_third_party_autocast(self) -> None:
        with (
            patch.object(
                aot_utils.torch._C,
                "_autocast_supported_devices",
                return_value=["cpu", "privateuseone"],
            ),
            patch.object(aot_utils.torch, "is_autocast_enabled") as enabled,
            patch.object(aot_utils.torch, "get_autocast_dtype") as dtype,
            patch.object(
                aot_utils.torch, "is_autocast_cache_enabled", return_value=False
            ),
        ):
            enabled.side_effect = lambda device: device == "privateuseone"
            dtype.side_effect = lambda device: {
                "cpu": torch.float32,
                "privateuseone": torch.float16,
            }[device]

            new_states = _get_autocast_states()
            legacy_states = [False, False, torch.float16, torch.float32, False]
            self.assertNotEqual(new_states, legacy_states)
            self.assertEqual(
                new_states, [False, torch.float32, True, torch.float16, False]
            )

    def test_collect_metadata_preserves_stable_autocast_state(self) -> None:
        from torch._functorch._aot_autograd.collect_metadata_analysis import (
            run_functionalized_fw_and_collect_metadata,
        )
        from torch._functorch._aot_autograd.descriptors import PlainAOTInput
        from torch._subclasses.fake_tensor import FakeTensorMode

        def f(x: torch.Tensor) -> list[torch.Tensor]:
            return [x + 1]

        collect = run_functionalized_fw_and_collect_metadata(
            f,
            flat_args_descs=[PlainAOTInput(0)],
            keep_input_mutations=False,
        )
        fake_mode = FakeTensorMode()
        x = fake_mode.from_tensor(torch.randn(2))
        before = _get_autocast_states()
        collect(x)
        self.assertEqual(before, _get_autocast_states())

    def test_collect_metadata_detects_autocast_state_mutation(self) -> None:
        from torch._functorch._aot_autograd.collect_metadata_analysis import (
            run_functionalized_fw_and_collect_metadata,
        )
        from torch._functorch._aot_autograd.descriptors import PlainAOTInput
        from torch._subclasses.fake_tensor import FakeTensorMode

        def f(x: torch.Tensor) -> list[torch.Tensor]:
            return [x + 1]

        collect = run_functionalized_fw_and_collect_metadata(
            f,
            flat_args_descs=[PlainAOTInput(0)],
            keep_input_mutations=False,
        )
        fake_mode = FakeTensorMode()
        x = fake_mode.from_tensor(torch.randn(2))
        call_count = 0

        def fake_get_autocast_states() -> list[object]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [False, torch.float32, False]
            return [True, torch.float32, False]

        with patch(
            "torch._functorch._aot_autograd.collect_metadata_analysis._get_autocast_states",
            side_effect=fake_get_autocast_states,
        ):
            with self.assertRaisesRegex(RuntimeError, "mutate the autocast state"):
                collect(x)

    def test_get_autocast_states_changes_when_npu_autocast_toggles(self) -> None:
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            self.skipTest("torch_npu is not available")
        if not torch.npu.is_available():
            self.skipTest("NPU is not available")

        states_off = _get_autocast_states()
        with torch.amp.autocast("npu", enabled=True, dtype=torch.float16):
            states_on = _get_autocast_states()
        self.assertNotEqual(states_off, states_on)


if __name__ == "__main__":
    run_tests()
