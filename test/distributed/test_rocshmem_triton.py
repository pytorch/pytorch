# Owner(s): ["oncall: distributed"]

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch.distributed._symmetric_memory._rocshmem_triton import RocshmemLibFinder
from torch.testing._internal.common_utils import run_tests, TestCase


class RocshmemLibFinderTest(TestCase):
    def setUp(self) -> None:
        RocshmemLibFinder.found_device_lib_path = None

    def tearDown(self) -> None:
        RocshmemLibFinder.found_device_lib_path = None

    def test_finds_device_library_in_rocm_sdk(self) -> None:
        for rocm_root_var in ("ROCM_HOME", "ROCM_PATH"):
            with self.subTest(rocm_root_var=rocm_root_var):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    device_lib = Path(tmp_dir) / "lib" / "librocshmem_device_gfx950.bc"
                    device_lib.parent.mkdir()
                    device_lib.touch()

                    with (
                        mock.patch.object(
                            torch.cuda, "is_available", return_value=True
                        ),
                        mock.patch.object(
                            torch.cuda,
                            "get_device_properties",
                            return_value=SimpleNamespace(gcnArchName="gfx950:sramecc+"),
                        ),
                        mock.patch.dict(
                            os.environ, {rocm_root_var: tmp_dir}, clear=False
                        ),
                    ):
                        os.environ.pop("ROCSHMEM_LIB_DIR", None)
                        other_root_var = (
                            "ROCM_PATH" if rocm_root_var == "ROCM_HOME" else "ROCM_HOME"
                        )
                        os.environ.pop(other_root_var, None)
                        self.assertEqual(
                            RocshmemLibFinder.find_device_library(),
                            str(device_lib),
                        )

                RocshmemLibFinder.found_device_lib_path = None


if __name__ == "__main__":
    run_tests()
