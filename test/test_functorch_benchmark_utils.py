# Owner(s): ["module: functorch"]

import os
import tempfile
from unittest.mock import patch

import torch
from torch._functorch import benchmark_utils
from torch.profiler import ProfilerActivity
from torch.testing._internal.common_utils import run_tests, TestCase


class _FakeProf:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def export_chrome_trace(self, filename: str) -> None:
        with open(filename, "w") as handle:
            handle.write('{"traceEvents":[]}')


class TestFunctorchBenchmarkUtils(TestCase):
    def _mock_accelerator(self, acc_type: str):
        return patch.object(
            torch.accelerator,
            "current_accelerator",
            return_value=type("Acc", (), {"type": acc_type})(),
        )

    def _dump(self, devices=None, num_runs=1, acc_type="cuda"):
        calls: list[object] = []

        def fake_sync(device=None, /):
            calls.append(device)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(benchmark_utils, "profile", return_value=_FakeProf()),
            patch.object(torch.accelerator, "is_available", return_value=True),
            self._mock_accelerator(acc_type),
            patch.object(torch.accelerator, "synchronize", side_effect=fake_sync),
        ):
            kwargs = {}
            if devices is not None:
                kwargs["devices"] = devices
            benchmark_utils.dump_chrome_trace(
                lambda x: x,
                (0,),
                os.path.join(tmp, "t.json"),
                torch.enable_grad(),
                [],
                num_runs=num_runs,
                **kwargs,
            )
        return calls

    def test_devices_cpu_does_not_sync(self):
        before = benchmark_utils.synchronize
        calls = self._dump(devices=["cpu"])
        self.assertEqual(calls, [])
        self.assertIs(benchmark_utils.synchronize, before)

    def test_default_devices_syncs_current_accelerator(self):
        calls = self._dump()
        self.assertEqual(len(calls), 8)
        self.assertTrue(all(c is None for c in calls))

    def test_cuda_sentinel_falls_back_when_accelerator_type_differs(self):
        calls = self._dump(acc_type="xpu")
        self.assertEqual(len(calls), 8)
        self.assertTrue(all(c is None for c in calls))

    def test_does_not_rebind_module_synchronize(self):
        before = benchmark_utils.synchronize
        self._dump(devices=["cuda"])
        self.assertIs(benchmark_utils.synchronize, before)
        before()

    def test_explicit_device_index_is_passed_through(self):
        calls = self._dump(devices=["cuda:1"])
        self.assertEqual(len(calls), 8)
        self.assertTrue(all(c == torch.device("cuda:1") for c in calls))

    def test_explicit_wrong_device_type_raises(self):
        with self.assertRaisesRegex(ValueError, "do not match current accelerator"):
            self._dump(devices=["cuda:0"], acc_type="xpu")

    def test_cpu_then_accelerator_does_not_leak_global_sync(self):
        before = benchmark_utils.synchronize
        cpu_calls = self._dump(devices=["cpu"])
        gpu_calls = self._dump(devices=["cuda"])
        self.assertEqual(cpu_calls, [])
        self.assertEqual(len(gpu_calls), 8)
        self.assertIs(benchmark_utils.synchronize, before)

    def test_empty_devices_raises(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self._dump(devices=[])

    def test_multiple_cpu_devices_does_not_sync(self):
        calls = self._dump(devices=["cpu", "cpu"])
        self.assertEqual(calls, [])

    def test_mixed_cpu_and_accelerator_devices(self):
        calls = self._dump(devices=["cpu", "cuda:1"])
        self.assertEqual(len(calls), 8)
        self.assertTrue(all(c == torch.device("cuda:1") for c in calls))

    def test_is_device_process_label_accepts_npu_trace(self):
        with self._mock_accelerator("npu"):
            self.assertTrue(benchmark_utils._is_device_process_label("NPU 0"))

    def test_builtin_accelerator_without_profiler_activity_raises(self):
        with (
            patch.object(torch.accelerator, "is_available", return_value=True),
            self._mock_accelerator("mps"),
        ):
            with self.assertRaisesRegex(RuntimeError, "not supported for accelerator 'mps'"):
                benchmark_utils._device_profiler_activity()

    def test_privateuse1_fallback_only_for_registered_backend(self):
        privateuse1_name = torch._C._get_privateuse1_backend_name()
        with (
            patch.object(torch.accelerator, "is_available", return_value=True),
            self._mock_accelerator(privateuse1_name),
        ):
            self.assertEqual(
                benchmark_utils._device_profiler_activity(),
                ProfilerActivity.PrivateUse1,
            )

    def test_renamed_privateuse1_without_enum_uses_privateuse1(self):
        with (
            patch.object(torch.accelerator, "is_available", return_value=True),
            self._mock_accelerator("npu"),
            patch.object(torch._C, "_get_privateuse1_backend_name", return_value="npu"),
        ):
            if hasattr(ProfilerActivity, "NPU"):
                self.skipTest("ProfilerActivity.NPU is present in this build")
            self.assertEqual(
                benchmark_utils._device_profiler_activity(),
                ProfilerActivity.PrivateUse1,
            )

    def test_unregistered_accelerator_name_raises(self):
        with (
            patch.object(torch.accelerator, "is_available", return_value=True),
            self._mock_accelerator("npu"),
        ):
            if hasattr(ProfilerActivity, "NPU"):
                self.skipTest("ProfilerActivity.NPU aliases npu in this build")
            if torch._C._get_privateuse1_backend_name() == "npu":
                self.skipTest("privateuse1 backend is registered as npu")
            with self.assertRaisesRegex(RuntimeError, "not supported for accelerator 'npu'"):
                benchmark_utils._device_profiler_activity()

    def test_renamed_privateuse1_with_enum_uses_enum(self):
        with (
            patch.object(torch.accelerator, "is_available", return_value=True),
            self._mock_accelerator("npu"),
            patch.object(torch._C, "_get_privateuse1_backend_name", return_value="npu"),
        ):
            if not hasattr(ProfilerActivity, "NPU"):
                self.skipTest("ProfilerActivity.NPU is not present in this build")
            self.assertEqual(
                benchmark_utils._device_profiler_activity(),
                ProfilerActivity.NPU,
            )


if __name__ == "__main__":
    run_tests()
