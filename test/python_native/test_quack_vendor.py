# Owner(s): ["module: dsl-native-ops"]

import importlib.util
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from torch.testing._internal.common_utils import run_tests, TestCase


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_SCRIPT = REPO_ROOT / "tools" / "vendoring" / "quack" / "vendor.sh"


@unittest.skipIf(
    importlib.util.find_spec("cutlass") is None,
    "vendored QuACK imports require CuTeDSL/CUTLASS",
)
class TestQuackVendor(TestCase):
    def test_vendored_quack_imports_from_torch_vendor(self):
        import torch._vendor.quack as quack
        from torch._vendor.quack.epilogue.frontend import EpiMod
        from torch._vendor.quack.gemm_interface import gemm_symmetric_out
        from torch._vendor.quack.grouped_reduce import feed_main_capable
        from torch._vendor.quack.rmsnorm import rmsnorm

        vendor_root = Path(quack.__file__).resolve().parent
        self.assertIn("torch/_vendor/quack", vendor_root.as_posix())
        for obj in (EpiMod, gemm_symmetric_out, feed_main_capable, rmsnorm):
            self.assertTrue(callable(obj))
        self.assertNotIn("quack", sys.modules)

    def test_vendored_ops_use_torch_vendor_quack_namespace(self):
        import torch
        import torch._vendor.quack.gemm_runtime.torch_op
        import torch._vendor.quack.rmsnorm

        self.assertTrue(hasattr(torch.ops.torch_vendor_quack, "gemm_epi"))
        self.assertTrue(hasattr(torch.ops.torch_vendor_quack, "_rmsnorm_fwd"))


@unittest.skipIf(
    shutil.which("git") is None or shutil.which("patch") is None,
    "re-running the vendoring script requires git and patch",
)
class TestQuackVendorScript(TestCase):
    def test_vendor_script_reproduces_committed_tree(self):
        src = os.environ.get("QUACK_VENDOR_SRC")
        allow_clone = os.environ.get("QUACK_VENDOR_ALLOW_CLONE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        if not src and not allow_clone:
            self.skipTest(
                "set QUACK_VENDOR_SRC to a local quack checkout at the pinned SHA, "
                "or QUACK_VENDOR_ALLOW_CLONE=1 to fetch upstream main"
            )

        cmd = ["bash", str(VENDOR_SCRIPT), "--check"]
        if src:
            cmd += ["--src", str(Path(src).expanduser())]
        self.assertEqual(
            subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode,
            0,
            "vendor.sh --check reported drift; edit the FlexGEMM patchset, "
            "not the vendored files",
        )


if __name__ == "__main__":
    run_tests()
