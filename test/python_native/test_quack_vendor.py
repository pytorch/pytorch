# Owner(s): ["module: dsl-native-ops"]

import importlib.util
import os
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest import mock

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
                )
            )
        )
        trace_context = TraceContext.create(None)
        trace_context.b("noop")
        trace_context.e("noop")
        trace_context.flush()

    def test_precompile_serializes_rmsnorm_tuned_tensor_kwargs(self):
        import torch
        from torch._vendor.quack.autotuner import AutotuneConfig
        from torch._vendor.quack.rmsnorm import rmsnorm_fwd_tuned
        from torch._vendor.quack.rmsnorm_config import RmsNormFwdConfig

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        x = torch.randn(2, 128, dtype=torch.float16, device="cuda")
        weight = torch.randn(128, dtype=torch.float16, device="cuda")
        out = torch.empty_like(x)
        bias = torch.randn(128, dtype=torch.float16, device="cuda")
        residual = torch.randn_like(x)
        residual_out = torch.empty_like(x)
        rstd = torch.empty(2, dtype=torch.float32, device="cuda")
        config = RmsNormFwdConfig(
            num_threads=128,
            threads_per_row=16,
            cluster_n=1,
            reload_from=None,
            delay_w_load=False,
        )
        configs = [AutotuneConfig(config=config), AutotuneConfig(config=config)]

        with (
            mock.patch.dict(os.environ, {"QUACK_COMPILE_WORKERS": "2"}),
            mock.patch(
                "torch._vendor.quack.autotuner.time.time",
                side_effect=[0.0, 1.0],
            ),
        ):
            handle = rmsnorm_fwd_tuned._precompile(
                x,
                weight,
                out,
                configs=configs,
                bias=bias,
                residual=residual,
                residual_out=residual_out,
                rstd=rstd,
            )
        try:
            for i in range(len(configs)):
                handle.wait_for(i)
            self.assertEqual(handle.failures, {})
        finally:
            handle.shutdown()


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
            "vendor.sh --check reported drift; edit the FlexGEMM patchset or "
            "PyTorch vendoring patches, not the vendored files",
        )


if __name__ == "__main__":
    run_tests()
