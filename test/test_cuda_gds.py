# Owner(s): ["module: cuda"]

import os
import sys
import tempfile
import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    NoTest,
    run_tests,
    TemporaryFileName,
    TestCase,
)


if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811


def _find_nvme_mount() -> str:
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if (
                    len(parts) >= 2
                    and "nvme" in parts[0]
                    and os.path.isdir(parts[1])
                    and os.access(parts[1], os.W_OK)
                ):
                    return parts[1]
    except OSError:
        pass
    return tempfile.gettempdir()


_GDS_TEST_DIR = os.environ.get("GDS_TEST_DIR", _find_nvme_mount())


def _gds_works_on_test_dir() -> bool:
    if not torch.cuda.gds.is_available():
        return False
    try:
        with TemporaryFileName(prefix="gds_probe_", dir=_GDS_TEST_DIR) as path:
            fd = os.open(path, os.O_RDWR | os.O_DIRECT)
            try:
                handle = torch._C._gds_register_handle(fd)
                torch._C._gds_deregister_handle(handle)
                return True
            finally:
                os.close(fd)
    except (OSError, RuntimeError):
        return False


@unittest.skipIf(not _gds_works_on_test_dir(), "GDS not functional on test filesystem")
class TestCudaGds(TestCase):
    def _tmp_path(self, name):
        d = tempfile.TemporaryDirectory(prefix="test_gds_", dir=_GDS_TEST_DIR)
        self.addCleanup(d.cleanup)
        return os.path.join(d.name, name)

    def _roundtrip(self, sd, name):
        path = self._tmp_path(name)
        torch.cuda.gds.save(sd, path)
        loaded = torch.cuda.gds.load(path)
        loaded_cpu = torch.load(path, map_location="cpu", weights_only=True)
        return loaded, loaded_cpu

    def test_save_load_roundtrip(self):
        sd = {f"layer{i}": torch.randn(512, 512, device="cuda") for i in range(4)}
        loaded, loaded_cpu = self._roundtrip(sd, "roundtrip.pt")
        for k in sd:
            self.assertEqual(sd[k], loaded[k])
            self.assertEqual(sd[k].cpu(), loaded_cpu[k])

        # map_location variants
        path = self._tmp_path("map_loc.pt")
        torch.cuda.gds.save(sd, path)
        for loc in ["cuda:0", torch.device("cuda:0"), "cpu"]:
            loaded = torch.cuda.gds.load(path, map_location=loc)
            for k in sd:
                self.assertEqual(sd[k].cpu(), loaded[k].cpu())

        # mixed CPU/GPU
        sd_mixed = {"gpu": torch.randn(64, device="cuda"), "cpu": torch.randn(64)}
        loaded, _ = self._roundtrip(sd_mixed, "mixed.pt")
        self.assertEqual(sd_mixed["gpu"], loaded["gpu"])
        self.assertEqual(sd_mixed["cpu"], loaded["cpu"].cpu())

        # shared storage
        weight = torch.randn(32, 16, device="cuda")
        sd_shared = {"enc": weight, "dec": weight}
        loaded, _ = self._roundtrip(sd_shared, "shared.pt")
        self.assertEqual(sd_shared["enc"], loaded["enc"])
        self.assertEqual(
            loaded["enc"].storage().data_ptr(), loaded["dec"].storage().data_ptr()
        )

    def test_torch_save_load_use_gds(self):
        sd = {f"layer{i}": torch.randn(256, 256, device="cuda") for i in range(4)}
        # save with use_gds, load normally
        path = self._tmp_path("save_gds.pt")
        torch.save(sd, path, use_gds=True)
        loaded = torch.load(path, map_location="cuda:0", weights_only=True)
        for k in sd:
            self.assertEqual(sd[k], loaded[k])
        # save normally, load with use_gds
        path2 = self._tmp_path("load_gds.pt")
        torch.cuda.gds.save(sd, path2)
        loaded = torch.load(path2, map_location="cuda:0", use_gds=True)
        for k in sd:
            self.assertEqual(sd[k], loaded[k])

    def test_use_gds_validation(self):
        path = self._tmp_path("gds_validate.pt")
        torch.save({"w": torch.randn(8)}, path)
        # load: mmap + use_gds
        with self.assertRaises(ValueError):
            torch.load(path, use_gds=True, mmap=True)
        # load: file-like object
        with open(path, "rb") as f:
            with self.assertRaises(ValueError):
                torch.load(f, use_gds=True)
        # save: file-like object
        import io

        with self.assertRaises(ValueError):
            torch.save({"w": torch.randn(8)}, io.BytesIO(), use_gds=True)


if __name__ == "__main__":
    run_tests()
