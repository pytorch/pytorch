# Owner(s): ["module: inductor"]

import unittest
from unittest.mock import MagicMock, patch

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
    def test_lookup_device_info(self):
        h100_info = lookup_device_info("NVIDIA H100")
        self.assertIsNotNone(h100_info)
        if h100_info is not None:
            self.assertEqual(h100_info.dram_gb, 80)
            self.assertIn(torch.float32, h100_info.tops)

        unknown_info = lookup_device_info("Unknown Device")
        self.assertIsNone(unknown_info)

    def test_datasheet_tops_function(self):
        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
        ):
            mock_get_device_name.return_value = "NVIDIA H100"
            tops = datasheet_tops(torch.float32)
            self.assertIsNotNone(tops)
            self.assertEqual(tops, 67.0)

            tops_tf32 = datasheet_tops(torch.float32, is_tf32=True)
            self.assertEqual(tops_tf32, 989.0)

            mock_get_device_name.return_value = "Unknown Device"
            tops_unknown = datasheet_tops(torch.float32)
            self.assertIsNone(tops_unknown)

            mock_get_device_name.return_value = None
            tops_no_device = datasheet_tops(torch.float32)
            self.assertIsNone(tops_no_device)

    @unittest.skipIf(torch.version.hip, "only nvidia")
    def test_lazy_pynvml_import(self):
        """Test pynvml import through torch.cuda."""
        with (
            patch("torch.cuda._HAS_PYNVML", True),
            patch.object(torch.cuda, "pynvml", MagicMock(), create=True) as mock_pynvml,
        ):
            pynvml = _get_pynvml()
            self.assertEqual(pynvml, mock_pynvml)

        with patch("torch.cuda._HAS_PYNVML", False):
            pynvml = _get_pynvml()
            self.assertIsNone(pynvml)

    @patch("torch.version.hip", None)
    @patch("torch._inductor.analysis.device_info._get_pynvml")
    def test_hardware_lookup_clock_hz_success(self, mock_get_pynvml):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
        mock_pynvml.nvmlDeviceGetMaxClockInfo.return_value = 1500
        mock_pynvml.NVML_CLOCK_SM = "clock_key"
        mock_pynvml.nvmlShutdown = MagicMock()
        mock_get_pynvml.return_value = mock_pynvml

        result = DeviceInfo._hardware_lookup_clock_hz()
        self.assertEqual(result, 1500 * 1e6)

    @unittest.skipIf(torch.version.hip, "only nvidia")
    def test_lazy_pynvml_import_caching(self):
        """Test pynvml caching through torch.cuda (now handled by torch.cuda module)."""
        with (
            patch("torch.cuda._HAS_PYNVML", True),
            patch.object(torch.cuda, "pynvml", MagicMock(), create=True) as mock_pynvml,
        ):
            pynvml1 = _get_pynvml()
            self.assertEqual(pynvml1, mock_pynvml)

            pynvml2 = _get_pynvml()
            self.assertEqual(pynvml2, mock_pynvml)

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

    def test_lazy_amd_smi_import_success(self):
        """Test AMD SMI import through torch.cuda."""
        with patch("torch.cuda._HAS_PYNVML", False):
            amd_smi = _get_amd_smi()
            self.assertIsNone(amd_smi)

    def test_lazy_amd_smi_import_caching(self):
        """Test AMD SMI caching through torch.cuda."""
        # Test consistent behavior across multiple calls
        with patch("torch.cuda._HAS_PYNVML", True):
            amd_smi1 = _get_amd_smi()
            amd_smi2 = _get_amd_smi()
            # Both should return the same result (None in this environment)
            self.assertEqual(amd_smi1, amd_smi2)

        with patch("torch.cuda._HAS_PYNVML", False):
            amd_smi1 = _get_amd_smi()
            amd_smi2 = _get_amd_smi()
            self.assertEqual(amd_smi1, amd_smi2)
            self.assertIsNone(amd_smi1)

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
        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
        ):
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
        """Test FLOPS calculation now uses datasheet values with clock adjustment."""
        with (
            patch.object(DeviceInfo, "lookup_clock_hz", return_value=1.5e9),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="AMD MI300X"),
        ):
            flops = DeviceInfo.lookup_tops(
                device_name="AMD MI300X", dtype=torch.float32
            )
            # Now uses datasheet value (163.4 TOPS) with clock adjustment
            # Device mapping has clock_hz=2100*1e6, so ratio = 1.5e9 / (2100*1e6) = ~0.714
            datasheet_flops = 163.4 * 1e12
            device_info = lookup_device_info("AMD MI300X")
            if device_info and device_info.clock_hz:
                clock_ratio = 1.5e9 / device_info.clock_hz
                expected_flops = datasheet_flops * clock_ratio
            else:
                expected_flops = datasheet_flops
            self.assertEqual(flops, expected_flops)

    def test_flops_datasheet_calculation(self):
        """Test FLOPS calculation using datasheet TOPS."""
        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
            patch.object(
                DeviceInfo, "lookup_clock_hz", return_value=1.98e9 / 2
            ),  # Use datasheet clock
        ):
            mock_get_device_name.return_value = "NVIDIA H100"

            flops = DeviceInfo.lookup_tops(
                device_name="NVIDIA H100", dtype=torch.float32
            )
            expected_flops = 67.0 * 1e12 / 2
            self.assertEqual(flops, expected_flops)

    def test_flops_fallback_to_datasheet(self):
        """Test FLOPS fallback to datasheet when hardware lookup fails."""
        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
            patch.object(
                DeviceInfo, "lookup_clock_hz", return_value=1.98e9 / 2
            ),  # Use datasheet clock
        ):
            mock_get_device_name.return_value = "NVIDIA H100"

            flops = DeviceInfo.lookup_tops(
                device_name="NVIDIA H100", dtype=torch.float32
            )
            expected_flops = 67.0 * 1e12 / 2
            self.assertEqual(flops, expected_flops)

    def test_flops_clock_adjustment_in_fallback(self):
        """Test clock adjustment when falling back to datasheet."""
        custom_device_info = DeviceSpec(
            memory_clock_hz=100,
            tops={torch.float32: 100.0},
            dram_bw_gbs=1000.0,
            dram_gb=16.0,
            sm_count=None,
            clock_hz=1.5e9,
        )

        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "torch._inductor.analysis.device_info.lookup_device_info"
            ) as mock_lookup,
        ):
            mock_get_device_name.return_value = "Custom Device"
            mock_lookup.return_value = custom_device_info

            with patch.object(
                DeviceInfo, "_hardware_lookup_clock_hz", return_value=3.0e9
            ):
                flops = DeviceInfo.lookup_tops("Custom Device", dtype=torch.float32)

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
            clock_hz=None,
        )
        mock_lookup.return_value = device_info

        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
        ):
            mock_get_device_name.return_value = "NVIDIA H100"

            with patch.object(
                DeviceInfo, "_hardware_lookup_clock_hz", return_value=3.0e9
            ):
                flops = DeviceInfo.lookup_tops("NVIDIA H100", dtype=torch.float32)

                expected_flops = 100.0 * 1e12
                self.assertEqual(flops, expected_flops)

    def test_flops_clock_adjustment_none_clock(self):
        """Test fallback behavior when clock lookup returns None."""
        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
        ):
            mock_get_device_name.return_value = "NVIDIA H100"

            with patch.object(
                DeviceInfo, "_hardware_lookup_clock_hz", return_value=None
            ):
                flops = DeviceInfo.lookup_tops("NVIDIA H100", dtype=torch.float32)

                expected_flops = 67.0 * 1e12
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
                flops = DeviceInfo.lookup_tops("NVIDIA H100", dtype=torch.float32)
                self.assertIsNone(flops)

            # When cuda is not available, hardware lookup is skipped and datasheet is used
            flops = DeviceInfo.lookup_tops("NVIDIA H100", dtype=torch.float32)
            self.assertIsNone(
                flops
            )  # Should be None since cuda.is_available() is False

    def test_flops_unknown_device(self):
        """Test FLOPS calculation with unknown device."""
        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
        ):
            mock_get_device_name.return_value = "Unknown Device"

            flops = DeviceInfo.lookup_tops("Unknown Device", dtype=torch.float32)
            # Should be None for unknown device
            self.assertIsNone(flops)

    def test_flops_partial_hardware_values(self):
        """Test FLOPS calculation with some hardware values missing."""
        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
            patch.object(
                DeviceInfo, "lookup_clock_hz", return_value=1.98e9 / 2
            ),  # Use datasheet clock
        ):
            mock_get_device_name.return_value = "NVIDIA H100"

            flops = DeviceInfo.lookup_tops(
                device_name="NVIDIA H100", dtype=torch.float32
            )
            expected_flops = 67.0 * 1e12 / 2
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
            patch("torch.cuda.is_available", return_value=True),
            patch.object(
                DeviceInfo, "lookup_clock_hz", return_value=1.98e9 / 2
            ),  # Use datasheet clock
        ):
            mock_get_device_name.return_value = "NVIDIA H100"

            flops = DeviceInfo.lookup_tops("NVIDIA H100", dtype=torch.float32)
            expected_flops = 67.0 * 1e12 / 2
            self.assertEqual(flops, expected_flops)

    def test_flops_integration_with_hardware_lookup(self):
        """Test FLOPS integration with datasheet values and clock adjustment."""
        dn = "NVIDIA H100"

        with (
            patch.object(DeviceInfo, "lookup_clock_hz", return_value=1500 * 1e6),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value=dn),
        ):
            flops = DeviceInfo.lookup_tops(device_name=dn, dtype=torch.float32)
            # Now uses datasheet value (67.0 TOPS) with clock adjustment
            # Device mapping has clock_hz=1.98e9, so ratio = 1500*1e6 / 1.98e9 = ~0.7576
            datasheet_flops = 67.0 * 1e12
            device_info = lookup_device_info(dn)
            if device_info and device_info.clock_hz:
                clock_ratio = (1500 * 1e6) / device_info.clock_hz
                expected_flops = datasheet_flops * clock_ratio
            else:
                expected_flops = datasheet_flops
            self.assertEqual(flops, expected_flops)

    @unittest.skipIf(
        True, "pynvml and amdsmi are not available in CI, run these tests locally"
    )
    @unittest.skipIf(torch.version.hip, "only nvidia")
    def test_pynvml_integration(self):
        """Test direct pynvml library integration."""
        try:
            import pynvml

            # Test basic NVML initialization and device access
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Test clock frequency retrieval
            sm_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(
                handle, pynvml.NVML_CLOCK_SM
            )
            self.assertIsInstance(sm_clock_mhz, int)
            self.assertGreater(sm_clock_mhz, 0)

            # Test memory clock frequency retrieval
            mem_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(
                handle, pynvml.NVML_CLOCK_MEM
            )
            self.assertIsInstance(mem_clock_mhz, int)
            self.assertGreater(mem_clock_mhz, 0)

            # Test memory bus width retrieval
            bus_width_bits = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
            self.assertIsInstance(bus_width_bits, int)
            self.assertGreater(bus_width_bits, 0)

            # Test bandwidth calculation (same as device_info.py implementation)
            mem_clock_hz = mem_clock_mhz * 1e6
            effective_rate = mem_clock_hz * 2  # GDDR uses DDR so *2
            peak_bw = (effective_rate * bus_width_bits) / 8
            peak_bw_gbs = peak_bw / (1024**3)

            self.assertIsInstance(peak_bw_gbs, float)
            self.assertGreater(peak_bw_gbs, 0)

            pynvml.nvmlShutdown()

        except ImportError:
            self.fail(
                "pynvml library not available - install with 'pip install nvidia-ml-py'"
            )
        except Exception as e:
            self.fail(f"pynvml integration failed: {e}")

    @unittest.skipIf(
        True, "pynvml and amdsmi are not available in CI, run these tests locally"
    )
    @unittest.skipIf(not torch.version.hip, "only amd")
    def test_amdsmi_integration(self):
        """Test direct amdsmi library integration."""
        try:
            import amdsmi

            # Test basic AMD SMI initialization
            amdsmi.amdsmi_init()

            # Test device handle retrieval (matches current implementation)
            device_handle = amdsmi.amdsmi_get_processor_handles()[0]
            self.assertIsNotNone(device_handle)

            # Test GPU clock info retrieval (matches current implementation)
            clock_info = amdsmi.amdsmi_get_clock_info(
                device_handle, amdsmi.AmdSmiClkType.SYS
            )
            self.assertTrue("max_clk" in clock_info)
            self.assertIsInstance(clock_info["max_clk"], int)
            self.assertGreater(clock_info["max_clk"], 0)

            # Test GPU memory clock info retrieval (matches current implementation)
            mem_clock_info = amdsmi.amdsmi_get_clock_info(
                device_handle, amdsmi.AmdSmiClkType.MEM
            )
            self.assertTrue("max_clk" in mem_clock_info)
            self.assertIsInstance(mem_clock_info["max_clk"], int)
            self.assertGreater(mem_clock_info["max_clk"], 0)

            amdsmi.amdsmi_shut_down()

        except ImportError:
            self.fail("amdsmi library not available - install AMD SMI")
        except Exception as e:
            self.fail(f"amdsmi integration failed: {e}")

    @unittest.skipIf(
        True, "pynvml and amdsmi are not available in CI, run these tests locally"
    )
    @unittest.skipIf(torch.version.hip, "only amd")
    def test_pynvml_error_handling(self):
        """Test pynvml error handling for invalid operations."""
        try:
            import pynvml

            pynvml.nvmlInit()

            # Test invalid device index - should raise exception
            with self.assertRaises(Exception):
                pynvml.nvmlDeviceGetHandleByIndex(999)  # Invalid index

            pynvml.nvmlShutdown()

        except ImportError:
            self.skipTest("pynvml library not available")

    @unittest.skipIf(
        True, "pynvml and amdsmi are not available in CI, run these tests locally"
    )
    @unittest.skipIf(not torch.version.hip, "only nvidia")
    def test_amd_smi_error_handling(self):
        """Test AMD SMI error handling for invalid operations."""
        # Try amdsmi only
        try:
            import amdsmi

            amdsmi.amdsmi_init()

            # Test invalid device index - should raise exception
            with self.assertRaises(Exception):
                amdsmi.amdsmi_get_processor_handle(999)  # Invalid index

            amdsmi.amdsmi_shut_down()

        except ImportError:
            self.skipTest("amdsmi library not available")

    @unittest.skipIf(True, "amdsmi is not available in CI, run this test locally")
    @unittest.skipIf(not torch.version.hip, "only amd")
    def test_amd_hardware_lookup_clock_hz(self):
        """Test the _amd_hardware_lookup_clock_hz function with real AMD hardware."""
        # Test the actual function directly
        clock_hz = DeviceInfo._amd_hardware_lookup_clock_hz()

        self.assertIsInstance(clock_hz, float)
        self.assertGreater(clock_hz, 0)
        # Clock frequency should be reasonable (between 500MHz and 5GHz)
        self.assertGreater(clock_hz, 50e6)
        self.assertLess(clock_hz, 5e9)
        # Should return frequency in Hz, not MHz
        # Most AMD clocks are in GHz range, so check it's properly converted
        self.assertGreater(clock_hz, 1e8)  # At least 100MHz in Hz

    @unittest.skipIf(True, "amdsmi is not available in CI, run this test locally")
    @unittest.skipIf(not torch.version.hip, "only amd")
    def test_amd_hardware_lookup_memory_clock_hz(self):
        """Test the _amd_hardware_lookup_memory_clock_hz function with real AMD hardware."""
        try:
            memory_clock_hz = DeviceInfo._amd_hardware_lookup_memory_clock_hz()

            self.assertIsInstance(memory_clock_hz, float)
            self.assertGreater(memory_clock_hz, 0)
            # Memory clock frequency should be reasonable (between 500MHz and 10GHz)
            self.assertGreater(memory_clock_hz, 500e6)
            self.assertLess(memory_clock_hz, 10e9)
            # Should return frequency in Hz, not MHz
            # Most AMD memory clocks are in GHz range, so check it's properly converted
            self.assertGreater(memory_clock_hz, 1e8)  # At least 100MHz in Hz

        except ImportError:
            self.fail("amdsmi library not available - install AMD SMI")
        except Exception:
            # If there's a hardware error or no AMD device, the function should
            # handle it gracefully and return None rather than crash
            self.assertIsNone(DeviceInfo._amd_hardware_lookup_memory_clock_hz())

    def test_dram_bw_hardware_calculation(self):
        """Test DRAM bandwidth calculation with memory clock adjustment."""
        with (
            patch.object(DeviceInfo, "lookup_memory_clock_hz", return_value=7e9),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="AMD MI300X"),
        ):
            dram_bw = DeviceInfo.lookup_dram_bw_gbs(device_name="AMD MI300X")
            # Uses datasheet value (5300.0 GB/s) with memory clock adjustment
            # Device mapping has memory_clock_hz=5200*1e6, so ratio = 7e9 / (5200*1e6) = ~1.346
            datasheet_bw = 5300.0
            device_info = lookup_device_info("AMD MI300X")
            if device_info and device_info.memory_clock_hz:
                memory_clock_ratio = 7e9 / device_info.memory_clock_hz
                expected_bw = datasheet_bw * memory_clock_ratio
            else:
                expected_bw = datasheet_bw
            self.assertEqual(dram_bw, expected_bw)

    def test_dram_bw_datasheet_calculation(self):
        """Test DRAM bandwidth calculation using datasheet values."""
        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
            patch.object(
                DeviceInfo, "lookup_memory_clock_hz", return_value=1.4e10 / 2
            ),  # Use half datasheet memory clock
        ):
            mock_get_device_name.return_value = "NVIDIA H100"

            dram_bw = DeviceInfo.lookup_dram_bw_gbs(device_name="NVIDIA H100")
            expected_bw = 3350 / 2  # Datasheet bandwidth scaled by memory clock ratio
            self.assertEqual(dram_bw, expected_bw)

    def test_dram_bw_fallback_to_datasheet(self):
        """Test DRAM bandwidth fallback to datasheet when hardware lookup fails."""
        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
            patch.object(
                DeviceInfo, "lookup_memory_clock_hz", return_value=1.4e10 / 2
            ),  # Use half datasheet memory clock
        ):
            mock_get_device_name.return_value = "NVIDIA H100"

            dram_bw = DeviceInfo.lookup_dram_bw_gbs(device_name="NVIDIA H100")
            expected_bw = 3350 / 2  # Datasheet bandwidth scaled by memory clock ratio
            self.assertEqual(dram_bw, expected_bw)

    def test_dram_bw_memory_clock_adjustment_in_fallback(self):
        """Test memory clock adjustment when falling back to datasheet."""
        custom_device_info = DeviceSpec(
            memory_clock_hz=2e9,
            tops={torch.float32: 100.0},
            dram_bw_gbs=1000.0,
            dram_gb=16.0,
            sm_count=None,
            clock_hz=1.5e9,
        )

        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "torch._inductor.analysis.device_info.lookup_device_info"
            ) as mock_lookup,
        ):
            mock_get_device_name.return_value = "Custom Device"
            mock_lookup.return_value = custom_device_info

            with patch.object(DeviceInfo, "lookup_memory_clock_hz", return_value=4e9):
                dram_bw = DeviceInfo.lookup_dram_bw_gbs("Custom Device")

                datasheet_bw = 1000.0
                memory_clock_ratio = 4e9 / 2e9
                expected_bw = datasheet_bw * memory_clock_ratio
                self.assertEqual(dram_bw, expected_bw)

    @patch("torch._inductor.analysis.device_info.lookup_device_info")
    def test_dram_bw_memory_clock_adjustment_no_expected_clock(self, mock_lookup):
        """Test fallback behavior when device mapping has None for memory_clock_hz."""
        device_info = DeviceSpec(
            memory_clock_hz=None,
            tops={torch.float32: 100.0},
            dram_bw_gbs=1000.0,
            dram_gb=16.0,
            sm_count=None,
            clock_hz=1.5e9,
        )
        mock_lookup.return_value = device_info

        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
        ):
            mock_get_device_name.return_value = "NVIDIA H100"

            with patch.object(DeviceInfo, "lookup_memory_clock_hz", return_value=4e9):
                dram_bw = DeviceInfo.lookup_dram_bw_gbs("NVIDIA H100")

                expected_bw = 1000.0  # No memory clock adjustment
                self.assertEqual(dram_bw, expected_bw)

    def test_dram_bw_memory_clock_adjustment_none_clock(self):
        """Test fallback behavior when memory clock lookup returns None."""
        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
        ):
            mock_get_device_name.return_value = "NVIDIA H100"

            with patch.object(DeviceInfo, "lookup_memory_clock_hz", return_value=None):
                dram_bw = DeviceInfo.lookup_dram_bw_gbs("NVIDIA H100")

                expected_bw = 3350  # Datasheet value without adjustment
                self.assertEqual(dram_bw, expected_bw)


if __name__ == "__main__":
    run_tests()
