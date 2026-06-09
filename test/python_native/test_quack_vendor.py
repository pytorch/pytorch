# Owner(s): ["module: dsl-native-ops"]

import importlib.util
import os
import shutil
import subprocess
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
    def test_vendored_quack_gemm_imports_from_torch_vendor(self):
        import torch._vendor.quack as quack
        from torch._vendor.quack.gemm_act import gemm_act
        from torch._vendor.quack.gemm_blockscaled_interface import (
            mxfp8_scaled_mm_epilogue,
            mxfp8_varlen_k_scaled_mm_epilogue,
            mxfp8_varlen_m_scaled_mm_epilogue,
        )
        from torch._vendor.quack.gemm_config import GemmConfig
        from torch._vendor.quack.trace import TraceContext

        vendor_root = Path(quack.__file__).resolve().parent
        self.assertIn("torch/_vendor/quack", vendor_root.as_posix())
        self.assertTrue(
            all(
                callable(obj)
                for obj in (
                    gemm_act,
                    GemmConfig,
                    mxfp8_scaled_mm_epilogue,
                    mxfp8_varlen_m_scaled_mm_epilogue,
                    mxfp8_varlen_k_scaled_mm_epilogue,
                )
            )
        )
        trace_context = TraceContext.create(None)
        trace_context.b("noop")
        trace_context.e("noop")
        trace_context.flush()


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
                "set QUACK_VENDOR_SRC to a local quack checkout, or "
                "QUACK_VENDOR_ALLOW_CLONE=1 to clone the pinned SHA"
            )

        cmd = ["bash", str(VENDOR_SCRIPT), "--check"]
        if src:
            cmd += ["--src", str(Path(src).expanduser())]
        self.assertEqual(
            subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode,
            0,
            "vendor.sh --check reported drift; edit tools/vendoring/quack/patches, "
            "not the vendored files",
        )


if __name__ == "__main__":
    run_tests()
