# Owner(s): ["module: inductor"]

from unittest.mock import call, MagicMock, patch

import torch
from torch._inductor.analysis.device_info import (
    _get_amd_smi,
    _get_pynvml,
    datasheet_tops,
    DeviceInfo,
    DeviceSpec,
    lookup_device_info,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestDeviceInfo(TestCase):
    def _reset_cache(self):
        import torch._inductor.analysis.device_info as device_info_module

        device_info_module._pynvml_cache = None
        device_info_module._pynvml_initialized = False
        device_info_module._amd_smi_cache = None
        device_info_module._amd_smi_name = None

    def setUp(self):
        self._reset_cache()

    def tearDown(self):
        self._reset_cache()

    def test_lookup_device_info(self):
        h100_info = lookup_device_info("NVIDIA H100")
        self.assertIsNotNone(h100_info)
        if h100_info is not None:
            self.assertEqual(h100_info.dram_gb, 80)
            self.assertIn(torch.float32, h100_info.tops)

        unknown_info = lookup_device_info("Unknown Device")
        self.assertIsNone(unknown_info)

    def test_datasheet_tops_function(self):
        with patch("torch.cuda.get_device_name") as mock_get_device_name:
            mock_get_device_name.return_value = "NVIDIA H100"
            tops = datasheet_tops(torch.float32)
            self.assertIsNotNone(tops)
            self.assertEqual(tops, 67.5)

            tops_tf32 = datasheet_tops(torch.float32, is_tf32=True)
            self.assertEqual(tops_tf32, 156.0)

            mock_get_device_name.return_value = "Unknown Device"
            tops_unknown = datasheet_tops(torch.float32)
            self.assertIsNone(tops_unknown)

            mock_get_device_name.return_value = None
            tops_no_device = datasheet_tops(torch.float32)
            self.assertIsNone(tops_no_device)

    def test_lazy_pynvml_import(self):
        import importlib

        import torch._inductor.analysis.device_info as device_info_module

        original_cache = device_info_module._pynvml_cache
        original_initialized = device_info_module._pynvml_initialized

        try:
            device_info_module._pynvml_cache = None
            device_info_module._pynvml_initialized = False

            importlib.reload(device_info_module)

            with patch("builtins.__import__") as mock_import:
                mock_pynvml_module = MagicMock()
                mock_import.return_value = mock_pynvml_module

                pynvml = device_info_module._get_pynvml()
                self.assertEqual(pynvml, mock_pynvml_module)
                self.assertTrue(mock_import.called)

            device_info_module._pynvml_cache = None
            device_info_module._pynvml_initialized = False

            with patch(
                "builtins.__import__", side_effect=ImportError("pynvml not found")
            ):
                pynvml = device_info_module._get_pynvml()
                self.assertIsNone(pynvml)

        finally:
            device_info_module._pynvml_cache = original_cache
            device_info_module._pynvml_initialized = original_initialized

    @patch("torch._inductor.analysis.device_info._get_pynvml")
    def test_hardware_lookup_clock_hz_success(self, mock_get_pynvml):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
        mock_pynvml.nvmlDeviceGetClockInfo.return_value = 1500
        mock_pynvml.NVML_CLOCK_SM = "clock_key"
        mock_pynvml.nvmlShutdown = MagicMock()
        mock_get_pynvml.return_value = mock_pynvml

        result = DeviceInfo._hardware_lookup_clock_hz()
        self.assertEqual(result, 1500 * 1e6)

    def test_lazy_pynvml_import_caching(self):
        with patch("builtins.__import__") as mock_import:
            mock_pynvml_module = MagicMock()
            mock_import.return_value = mock_pynvml_module

            pynvml1 = _get_pynvml()
            self.assertEqual(pynvml1, mock_pynvml_module)
            self.assertEqual(mock_import.call_count, 1)

            pynvml2 = _get_pynvml()
            self.assertEqual(pynvml2, mock_pynvml_module)
            self.assertEqual(mock_import.call_count, 1)

            self.assertEqual(pynvml1, pynvml2)

    def test_hardware_lookup_exception_handling(self):
        with (
            patch("torch.version.hip", None),
            patch(
                "torch.cuda.get_device_properties", side_effect=Exception("CUDA Error")
            ),
            patch(
                "torch._inductor.analysis.device_info._get_pynvml"
            ) as mock_get_pynvml,
        ):
            mock_pynvml = MagicMock()
            mock_pynvml.nvmlInit.side_effect = Exception("NVML Error")
            mock_get_pynvml.return_value = mock_pynvml

            # Test direct hardware lookup methods, not the generic lookup methods
            result = DeviceInfo._hardware_lookup_sm_count()
            self.assertIsNone(result)

            result = DeviceInfo._hardware_lookup_clock_hz()
            self.assertIsNone(result)

    def test_device_mapping_aliases(self):
        mi300x_direct = lookup_device_info("AMD MI300X")
        mi300x_alias = lookup_device_info("AMD INSTINCT MI300X")
        self.assertEqual(mi300x_direct, mi300x_alias)

        mi210x_direct = lookup_device_info("AMD MI210X")
        mi210x_alias = lookup_device_info("AMD INSTINCT MI210X")
        self.assertEqual(mi210x_direct, mi210x_alias)

    def setUp_amd(self):
        import torch._inductor.analysis.device_info as device_info_module

        device_info_module._amd_smi_cache = None
        device_info_module._amd_smi_name = None

    def test_lazy_amd_smi_import_success(self):
        self.setUp_amd()

        with patch("builtins.__import__") as mock_import:
            mock_amd_smi_module = MagicMock()

            def mock_import_func(module_name):
                if module_name == "amdsmi":
                    return mock_amd_smi_module
                raise ImportError(f"No module named '{module_name}'")

            mock_import.side_effect = mock_import_func

            amd_smi = _get_amd_smi()
            self.assertEqual(amd_smi, mock_amd_smi_module)

    def test_lazy_amd_smi_import_failure(self):
        """Test AMD SMI library import failure for all libraries."""
        self.setUp_amd()

        with patch(
            "builtins.__import__", side_effect=ImportError("No AMD library found")
        ):
            amd_smi = _get_amd_smi()
            self.assertIsNone(amd_smi)

    def test_lazy_amd_smi_import_caching(self):
        """Test that AMD SMI import is cached and not repeated."""
        self.setUp_amd()

        with patch("builtins.__import__") as mock_import:
            mock_amd_smi_module = MagicMock()

            def mock_import_func(module_name):
                if module_name == "rocm_smi":
                    return mock_amd_smi_module
                raise ImportError(f"No module named '{module_name}'")

            mock_import.side_effect = mock_import_func

            amd_smi1 = _get_amd_smi()
            self.assertEqual(amd_smi1, mock_amd_smi_module)

            amd_smi2 = _get_amd_smi()
            self.assertEqual(amd_smi2, mock_amd_smi_module)

            self.assertEqual(amd_smi1, amd_smi2)

            expected_calls = [
                call("amdsmi"),
                call("rocm_smi"),
            ]
            mock_import.assert_has_calls(expected_calls, any_order=False)

    @patch("torch.version.hip", "some_hip_version")
    @patch("torch._inductor.analysis.device_info._get_amd_smi")
    def test_amd_hardware_lookup_clock_hz_success(self, mock_get_amd_smi):
        """Test successful AMD clock frequency lookup."""
        mock_amd_smi = MagicMock()
        mock_amd_smi.rsmi_init = MagicMock()
        mock_amd_smi.rsmi_dev_gpu_clk_freq_get.return_value = 2100
        mock_amd_smi.RSMI_CLK_TYPE_SYS = "system_clock"
        mock_amd_smi.rsmi_shut_down = MagicMock()
        mock_get_amd_smi.return_value = mock_amd_smi

        result = DeviceInfo._amd_hardware_lookup_clock_hz()
        self.assertEqual(result, 2100 * 1e6)
        mock_amd_smi.rsmi_dev_gpu_clk_freq_get.assert_called_once_with(
            0, "system_clock"
        )

    @patch("torch.version.hip", "some_hip_version")
    @patch("torch._inductor.analysis.device_info._get_amd_smi")
    def test_amd_hardware_lookup_dram_bw_gbs_not_implemented(self, mock_get_amd_smi):
        """Test AMD memory bandwidth lookup (not implemented)."""
        mock_amd_smi = MagicMock()
        mock_amd_smi.rsmi_init = MagicMock()
        mock_amd_smi.rsmi_shut_down = MagicMock()
        mock_get_amd_smi.return_value = mock_amd_smi

        result = DeviceInfo._amd_hardware_dram_bw_gbs()
        self.assertIsNone(result)

    def test_amd_device_mapping_entries(self):
        """Test that AMD devices are properly represented in device mapping."""
        mi300x = lookup_device_info("AMD MI300X")
        self.assertIsNotNone(mi300x)
        if mi300x is not None:
            self.assertEqual(mi300x.dram_gb, 192.0)
            self.assertEqual(mi300x.dram_bw_gbs, 5300.0)
            self.assertIn(torch.float32, mi300x.tops)

        mi300x_instinct = lookup_device_info("AMD INSTINCT MI300X")
        self.assertEqual(mi300x, mi300x_instinct)

        mi300a = lookup_device_info("AMD MI300A")
        self.assertIsNotNone(mi300a)
        if mi300a is not None:
            self.assertEqual(mi300a.dram_gb, 128.0)
            self.assertEqual(mi300a.dram_bw_gbs, 5300.0)

        mi210x = lookup_device_info("AMD MI210X")
        self.assertIsNotNone(mi210x)
        if mi210x is not None:
            self.assertEqual(mi210x.dram_gb, 64.0)
            self.assertEqual(mi210x.dram_bw_gbs, 1600.0)

        mi210x_instinct = lookup_device_info("AMD INSTINCT MI210X")
        self.assertEqual(mi210x, mi210x_instinct)

    def test_amd_integration_with_datasheet_tops(self):
        """Test datasheet_tops function with AMD devices."""
        with patch("torch.cuda.get_device_name") as mock_get_device_name:
            mock_get_device_name.return_value = "AMD MI300X"

            tops_fp32 = datasheet_tops(torch.float32)
            self.assertEqual(tops_fp32, 163.4)

            tops_fp16 = datasheet_tops(torch.float16)
            self.assertEqual(tops_fp16, 1307.4)

            tops_bf16 = datasheet_tops(torch.bfloat16)
            self.assertEqual(tops_bf16, 1307.4)

            tops_tf32 = datasheet_tops(torch.float32, is_tf32=True)
            self.assertEqual(tops_tf32, 653.7)

    def test_flops_hardware_calculation(self):
        """Test FLOPS calculation using hardware lookup methods."""
        with (
            patch.object(DeviceInfo, "lookup_sm_count", return_value=108),
            patch.object(DeviceInfo, "lookup_cores_per_sm", return_value=64),
            patch.object(DeviceInfo, "lookup_clock_hz", return_value=1.5e9),
            patch.object(DeviceInfo, "lookup_ops_per_core_per_cycle", return_value=2),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="AMD MI300X"),
        ):
            flops = DeviceInfo.lookup_tops(
                device_name="AMD MI300X", dtype=torch.float32, force_datasheet=False
            )
            expected_flops = 108 * 64 * 1.5e9 * 2
            self.assertEqual(flops, expected_flops)

    def test_flops_datasheet_calculation(self):
        """Test FLOPS calculation using datasheet TOPS."""
        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
        ):
            mock_get_device_name.return_value = "NVIDIA H100"

            flops = DeviceInfo.lookup_tops(
                device_name="NVIDIA H100", dtype=torch.float32, force_datasheet=True
            )
            expected_flops = 67.5 * 1e12
            self.assertEqual(flops, expected_flops)

    def test_flops_fallback_to_datasheet(self):
        """Test FLOPS fallback to datasheet when hardware lookup fails."""
        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
        ):
            mock_get_device_name.return_value = "NVIDIA H100"

            flops = DeviceInfo.lookup_tops(
                device_name="NVIDIA H100", dtype=torch.float32, force_datasheet=False
            )
            expected_flops = 67.5 * 1e12
            self.assertEqual(flops, expected_flops)

    def test_flops_clock_adjustment_in_fallback(self):
        """Test clock adjustment when falling back to datasheet."""
        custom_device_info = DeviceSpec(
            memory_clock_hz=100,
            tops={torch.float32: 100.0},
            dram_bw_gbs=1000.0,
            dram_gb=16.0,
            sm_count=None,
            cores_per_sm=None,
            clock_hz=1.5e9,
            ops_per_core_per_cycle=None,
        )

        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch(
                "torch._inductor.analysis.device_info.lookup_device_info"
            ) as mock_lookup,
        ):
            mock_get_device_name.return_value = "Custom Device"
            mock_lookup.return_value = custom_device_info

            with patch.object(
                DeviceInfo, "_hardware_lookup_clock_hz", return_value=3.0e9
            ):
                flops = DeviceInfo.lookup_tops(
                    "Custom Device", dtype=torch.float32, force_datasheet=False
                )

                datasheet_flops = 100.0 * 1e12
                clock_ratio = 3.0e9 / 1.5e9
                expected_flops = datasheet_flops * clock_ratio
                self.assertEqual(flops, expected_flops)

    @patch("torch._inductor.analysis.device_info.lookup_device_info")
    def test_flops_clock_adjustment_no_expected_clock(self, mock_lookup):
        """Test fallback behavior when device mapping has None for clock_hz."""
        device_info = DeviceSpec(
            memory_clock_hz=100,
            tops={torch.float32: 100.0},
            dram_bw_gbs=1000.0,
            dram_gb=16.0,
            sm_count=None,
            cores_per_sm=None,
            clock_hz=None,
            ops_per_core_per_cycle=None,
        )
        mock_lookup.return_value = device_info

        with patch("torch.cuda.get_device_name") as mock_get_device_name:
            mock_get_device_name.return_value = "NVIDIA H100"

            with patch.object(
                DeviceInfo, "_hardware_lookup_clock_hz", return_value=3.0e9
            ):
                flops = DeviceInfo.lookup_tops(
                    "NVIDIA H100", dtype=torch.float32, force_datasheet=False
                )

                expected_flops = 100.0 * 1e12
                self.assertEqual(flops, expected_flops)

    def test_flops_clock_adjustment_none_clock(self):
        """Test fallback behavior when clock lookup returns None."""
        with patch("torch.cuda.get_device_name") as mock_get_device_name:
            mock_get_device_name.return_value = "NVIDIA H100"

            with patch.object(
                DeviceInfo, "_hardware_lookup_clock_hz", return_value=None
            ):
                flops = DeviceInfo.lookup_tops(
                    "NVIDIA H100", dtype=torch.float32, force_datasheet=False
                )

                expected_flops = 67.5 * 1e12
                self.assertEqual(flops, expected_flops)

    def test_flops_no_device_name(self):
        """Test FLOPS calculation when device name is unavailable."""
        with (
            patch("torch.cuda.get_device_name", return_value=None),
            patch("torch.cuda.is_available", return_value=False),
        ):
            # When there's no device name and we force datasheet, it should return None
            with patch(
                "torch._inductor.analysis.device_info.datasheet_tops", return_value=None
            ):
                flops = DeviceInfo.lookup_tops(
                    "NVIDIA H100", dtype=torch.float32, force_datasheet=True
                )
                self.assertIsNone(flops)

            # When cuda is not available, hardware lookup is skipped and datasheet is used
            flops = DeviceInfo.lookup_tops(
                "NVIDIA H100", dtype=torch.float32, force_datasheet=False
            )
            self.assertIsNone(
                flops
            )  # Should be None since cuda.is_available() is False

    def test_flops_unknown_device(self):
        """Test FLOPS calculation with unknown device."""
        with patch("torch.cuda.get_device_name") as mock_get_device_name:
            mock_get_device_name.return_value = "Unknown Device"

            flops = DeviceInfo.lookup_tops(
                "Unknown Device", dtype=torch.float32, force_datasheet=False
            )
            self.assertIsNone(flops)

    def test_flops_partial_hardware_values(self):
        """Test FLOPS calculation with some hardware values missing."""
        with patch("torch.cuda.get_device_name") as mock_get_device_name:
            mock_get_device_name.return_value = "NVIDIA H100"

            flops = DeviceInfo.lookup_tops(
                device_name="NVIDIA H100", dtype=torch.float32, force_datasheet=False
            )
            expected_flops = 67.5 * 1e12
            self.assertEqual(flops, expected_flops)

    def test_flops_exception_handling(self):
        """Test FLOPS calculation handles exceptions gracefully."""
        with (
            patch.object(
                DeviceInfo,
                "_hardware_lookup_sm_count",
                side_effect=Exception("Hardware error"),
            ),
            patch("torch.cuda.get_device_name") as mock_get_device_name,
        ):
            mock_get_device_name.return_value = "NVIDIA H100"

            flops = DeviceInfo.lookup_tops(
                "NVIDIA H100", dtype=torch.float32, force_datasheet=False
            )
            expected_flops = 67.5 * 1e12
            self.assertEqual(flops, expected_flops)

    def test_flops_integration_with_hardware_lookup(self):
        """Test FLOPS integration with actual hardware lookup methods."""
        dn = "NVIDIA H100"

        with (
            patch.object(DeviceInfo, "lookup_sm_count", return_value=108),
            patch.object(DeviceInfo, "lookup_cores_per_sm", return_value=64),
            patch.object(DeviceInfo, "lookup_clock_hz", return_value=1500 * 1e6),
            patch.object(DeviceInfo, "lookup_ops_per_core_per_cycle", return_value=2),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value=dn),
        ):
            flops = DeviceInfo.lookup_tops(
                device_name=dn, dtype=torch.float32, force_datasheet=False
            )
            expected_flops = 108 * 64 * (1500 * 1e6) * 2
            self.assertEqual(flops, expected_flops)


if __name__ == "__main__":
    run_tests()
