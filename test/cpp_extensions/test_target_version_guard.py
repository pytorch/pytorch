# Owner(s): ["module: cpp"]

"""Compile-time TORCH_TARGET_VERSION vs TORCH_ABI_VERSION guard."""

import subprocess
import tempfile
from pathlib import Path

from torch.testing._internal.common_utils import run_tests, skipIfWindows, TestCase
from torch.utils.cpp_extension import get_cxx_compiler, include_paths


class TestTargetVersionGuard(TestCase):
    def _compile_target_version_snippet(self, target_expr: str) -> tuple[bool, str]:
        pytorch_includes = [f"-I{path}" for path in include_paths(device_type="cpu")]
        with tempfile.TemporaryDirectory(prefix="target_version_guard_") as tmp:
            src_file = Path(tmp) / "target_version_guard.cpp"
            obj_file = Path(tmp) / "target_version_guard.o"
            src_file.write_text(
                f"""\
#include <torch/headeronly/version.h>
#define TORCH_TARGET_VERSION {target_expr}
#include <torch/csrc/stable/version.h>
int main() {{ return 0; }}
"""
            )
            result = subprocess.run(
                [
                    get_cxx_compiler(),
                    "-c",
                    "-std=c++17",
                    *pytorch_includes,
                    str(src_file),
                    "-o",
                    str(obj_file),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0, result.stderr

    @skipIfWindows(msg="uses gcc/clang compile flags")
    def test_target_version_must_not_exceed_abi_version(self):
        """TORCH_TARGET_VERSION must be <= TORCH_ABI_VERSION of these headers."""
        success, error_msg = self._compile_target_version_snippet("TORCH_ABI_VERSION")
        self.assertTrue(
            success,
            f"Expected TORCH_TARGET_VERSION=TORCH_ABI_VERSION to compile. Error: {error_msg}",
        )

        # test failure of next minor version.
        success, error_msg = self._compile_target_version_snippet(
            "(TORCH_ABI_VERSION + (1ULL << 48))"
        )
        self.assertFalse(
            success,
            "Expected TORCH_TARGET_VERSION newer than TORCH_ABI_VERSION to fail to compile, but it compiled cleanly.",
        )
        self.assertIn(
            "TORCH_TARGET_VERSION is newer than the libtorch headers",
            error_msg,
        )


if __name__ == "__main__":
    run_tests()
