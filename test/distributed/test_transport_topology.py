# Owner(s): ["oncall: distributed"]

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch.distributed._transport import _topology
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


GPU = ("0x10de", "0x030200")
NIC = ("0x15b3", "0x020700")
_accelerator = _topology._accelerator


def root(i):
    """Device directly under its own PCI root ``i``."""
    return f"pci{i:04x}:00/{i:04x}:00:00.0/{i:04x}:01:00.0"


class TestSelectNics(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.devices = tmp / "devices"
        self.pci = tmp / "bus"
        self.ib = tmp / "ib"
        for path in (self.pci, self.ib):
            path.mkdir()
        self.gpus = []
        for target, value in (
            ("_PCI_ROOT", self.pci),
            ("_IB_ROOT", self.ib),
            ("_accelerator", lambda device: device),
            ("_pci_address", lambda device: self.gpus[device.index]),
        ):
            self.enterContext(patch.object(_topology, target, value))

    def add_pci(self, path, kind, numa, vf=False):
        device = self.devices / path
        device.mkdir(parents=True)
        for name, value in zip(("vendor", "class"), kind):
            (device / name).write_text(value)
        (device / "numa_node").write_text(str(numa))
        if vf:
            (device / "physfn").symlink_to(device.parent)
        (self.pci / device.name).symlink_to(device)
        return device.name

    def add_gpu(self, path, numa=0, vf=False):
        address = self.add_pci(path, GPU, numa, vf)
        if not vf:
            self.gpus.append(address)

    def add_nic(self, name, path, numa=0, rate=400, state="4: ACTIVE", vf=False):
        self.add_pci(path, NIC, numa, vf)
        port = self.ib / name / "ports" / "1"
        port.mkdir(parents=True)
        (port / "state").write_text(state)
        (port / "rate").write_text(f"{rate} Gb/sec (4X NDR)")
        (self.ib / name / "device").symlink_to(self.devices / path)

    def select(self, index):
        # Patched to fail if selection hits an unexpected exception.
        with patch.object(_topology.logger, "debug") as debug:
            selected = _topology.select_nics(torch.device("cuda", index))
        debug.assert_not_called()
        return selected

    def test_pcie_switch_beats_numa_and_order(self):
        for i in range(2):
            switch = f"pci0000:00/0000:00:0{i}.0/0000:0{i + 1}:00.0"
            self.add_gpu(f"{switch}/0000:1{i}:00.0", numa=0)
            # NIC names are in the reverse order of the switches.
            self.add_nic(f"mlx5_{1 - i}", f"{switch}/0000:2{i}:00.0", numa=1)
        # Same NUMA node as both GPUs but on another PCI root.
        self.add_nic("mlx5_2", root(5))
        self.assertEqual([self.select(i) for i in range(2)], [["mlx5_1"], ["mlx5_0"]])

    def test_numa_breaks_ties(self):
        # GB300-like: each GPU and NIC has its own PCI root.
        for i in range(4):
            self.add_gpu(root(i), numa=i // 2)
            self.add_nic(f"mlx5_{i}", root(0x10 + i), numa=1 - i // 2)
        self.assertEqual(
            [self.select(i) for i in range(4)],
            [["mlx5_2"], ["mlx5_3"], ["mlx5_0"], ["mlx5_1"]],
        )

    def test_slow_and_down_nics_ignored(self):
        switch = "pci0000:00/0000:00:00.0/0000:01:00.0"
        self.add_gpu(f"{switch}/0000:10:00.0")
        self.add_nic("mlx5_0", f"{switch}/0000:20:00.0", rate=200)
        self.add_nic("mlx5_1", f"{switch}/0000:21:00.0", state="1: DOWN")
        self.add_nic("mlx5_2", root(5))
        self.assertEqual(self.select(0), ["mlx5_2"])

    def test_more_nics_than_gpus(self):
        self.add_gpu(root(9))
        for i in range(2):
            self.add_nic(f"mlx5_{i}", root(i))
        self.assertEqual(self.select(0), ["mlx5_0", "mlx5_1"])

    def test_tied_gpus_share_nics(self):
        for i in range(2):
            self.add_gpu(root(i))
        for i in range(5):
            self.add_nic(f"mlx5_{i}", root(0x10 + i))
        self.assertEqual(
            [self.select(i) for i in range(2)],
            [["mlx5_0", "mlx5_1"], ["mlx5_2", "mlx5_3"]],
        )

    def test_overlapping_nearest_nics_stay_disjoint(self):
        # GPU0 shares a PCIe switch with NIC0; GPU1 is equally far from both.
        switch = "pci0000:00/0000:00:01.0/0000:01:00.0"
        self.add_gpu(f"{switch}/0000:02:00.0")
        self.add_nic("mlx5_0", f"{switch}/0000:03:00.0")
        self.add_gpu("pci0000:00/0000:00:02.0/0000:04:00.0")
        self.add_nic("mlx5_1", "pci0000:00/0000:00:03.0/0000:05:00.0")
        self.assertEqual([self.select(i) for i in range(2)], [["mlx5_0"], ["mlx5_1"]])

    def test_unassigned_nics_stay_free(self):
        # GPU0/1 split mlx5_0..2 under their switch; GPU2/3 get the leftover.
        switch = "pci0000:00/0000:00:01.0/0000:01:00.0"
        for i in range(2):
            self.add_gpu(f"{switch}/0000:0{2 + i}:00.0")
        for i in range(3):
            self.add_nic(f"mlx5_{i}", f"{switch}/0000:0{4 + i}:00.0")
        for i in range(2):
            self.add_gpu(f"pci0000:00/0000:00:0{2 + i}.0/0000:1{i}:00.0")
        self.add_nic("mlx5_3", "pci0000:00/0000:00:04.0/0000:20:00.0")
        self.assertEqual(
            [self.select(i) for i in range(4)],
            [["mlx5_0"], ["mlx5_1"], ["mlx5_2"], ["mlx5_3"]],
        )

    def test_claimed_nics_shared_when_none_free(self):
        # GPU0/1 each share a switch with one NIC; GPU2 ties on both.
        for i in range(2):
            switch = f"pci0000:00/0000:00:0{i + 1}.0/0000:0{i + 1}:00.0"
            self.add_gpu(f"{switch}/0000:1{i}:00.0")
            self.add_nic(f"mlx5_{i}", f"{switch}/0000:2{i}:00.0")
        self.add_gpu("pci0000:00/0000:00:03.0/0000:30:00.0")
        self.assertEqual(
            [self.select(i) for i in range(3)],
            [["mlx5_0"], ["mlx5_1"], ["mlx5_0", "mlx5_1"]],
        )

    def test_group_ranked_by_closest_member(self):
        # GPU1 shares a switch with mlx5_0, so its group (with GPU0, which is
        # nearest mlx5_0 only by NUMA) claims mlx5_0 before GPU2's tie.
        switch = "pci0001:00/0001:00:01.0/0001:01:00.0"
        self.add_gpu(root(0), numa=1)
        self.add_gpu(f"{switch}/0001:02:00.0", numa=1)
        self.add_nic("mlx5_0", f"{switch}/0001:03:00.0", numa=1)
        self.add_nic("mlx5_1", "pci0001:00/0001:00:05.0/0001:06:00.0")
        self.add_gpu("pci0001:00/0001:00:09.0/0001:0a:00.0", numa=2)
        self.assertEqual(
            [self.select(i) for i in range(3)], [["mlx5_0"], ["mlx5_0"], ["mlx5_1"]]
        )

    def test_nic_virtual_functions_ignored(self):
        for i in range(2):
            self.add_gpu(root(i))
        self.add_nic("mlx5_0", root(0x10))
        self.add_nic("mlx5_1", f"{root(0x10)}/0010:02:00.0", vf=True)
        self.add_nic("mlx5_2", root(0x11))
        self.assertEqual([self.select(i) for i in range(2)], [["mlx5_0"], ["mlx5_2"]])

    def test_more_gpus_than_nics(self):
        for i in range(3):
            self.add_gpu(root(i))
        for i in range(2):
            self.add_nic(f"mlx5_{i}", root(0x10 + i))
        self.assertEqual(
            [self.select(i) for i in range(3)], [["mlx5_0"], ["mlx5_1"], ["mlx5_0"]]
        )

    def test_hidden_gpus_and_virtual_functions(self):
        for i in range(2):
            self.add_gpu(root(i))
            self.add_gpu(f"{root(i)}/{i:04x}:02:00.0", vf=True)
            self.add_nic(f"mlx5_{i}", root(0x10 + i))
        # CUDA_VISIBLE_DEVICES=1: the only visible device is the second GPU.
        self.gpus = self.gpus[1:]
        self.assertEqual(self.select(0), ["mlx5_1"])

    def test_fallbacks(self):
        self.add_gpu(root(0))
        self.assertIsNone(self.select(0))
        self.add_nic("mlx5_0", root(1))
        self.assertEqual(self.select(0), ["mlx5_0"])
        with patch.object(_topology, "_pci_address", return_value=None):
            self.assertIsNone(self.select(0))
        with patch.object(_topology, "_accelerator", return_value=None):
            self.assertIsNone(self.select(0))


class TestAccelerator(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def accelerator(self, device, initialized, current=3, local_rank=None, count=8):
        module = SimpleNamespace(is_initialized=lambda: initialized)
        index = MagicMock(return_value=current)
        env = {} if local_rank is None else {"LOCAL_RANK": local_rank}
        with (
            patch.dict(os.environ, env),
            patch.object(torch.accelerator, "device_count", return_value=count),
            patch.object(
                torch.accelerator,
                "current_accelerator",
                return_value=torch.device("cuda"),
            ),
            patch.object(torch, "get_device_module", return_value=module),
            patch.object(torch.accelerator, "current_device_index", index),
        ):
            return _accelerator(device), index.called

    def test_explicit_device(self):
        device = torch.device("cuda", 1)
        self.assertEqual(self.accelerator(device, False), (device, False))

    def test_current_device_once_initialized(self):
        for device in (None, torch.device("cpu")):
            self.assertEqual(
                self.accelerator(device, True), (torch.device("cuda", 3), True)
            )
            # Never initialize the runtime for a CPU-only process.
            self.assertEqual(self.accelerator(device, False), (None, False))

    def test_unset_device_with_nonzero_local_rank(self):
        # CUDA initialized without set_device leaves the current device at 0.
        with self.assertLogs(_topology.logger, "WARNING"):
            self.assertEqual(
                self.accelerator(None, True, current=0, local_rank="3"),
                (None, True),
            )
        for local_rank, count in (("0", 8), ("3", 1)):
            self.assertEqual(
                self.accelerator(None, True, 0, local_rank, count),
                (torch.device("cuda", 0), True),
            )

    def test_device_without_index(self):
        device = torch.device("cuda")
        self.assertEqual(
            self.accelerator(device, False), (torch.device("cuda", 3), True)
        )
        with self.assertLogs(_topology.logger, "WARNING"):
            self.assertEqual(
                self.accelerator(device, False, current=0, local_rank="3"),
                (None, True),
            )

    def test_no_accelerator(self):
        with patch.object(torch.accelerator, "current_accelerator", return_value=None):
            self.assertIsNone(_accelerator(None))


if __name__ == "__main__":
    run_tests()
