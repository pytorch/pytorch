import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".ci" / "pytorch" / "binary_populate_env.sh"


class TestBinaryPopulateEnv(unittest.TestCase):
    def test_missing_pytorch_root_has_actionable_error(self) -> None:
        env = {
            "BINARY_ENV_FILE": os.devnull,
            "DESIRED_CUDA": "cpu",
            "GPU_ARCH_TYPE": "cuda",
            "PACKAGE_TYPE": "wheel",
            "PATH": os.environ["PATH"],
        }

        result = subprocess.run(
            ["bash", str(SCRIPT)],
            check=False,
            capture_output=True,
            env=env,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "PYTORCH_ROOT must point to the PyTorch checkout root",
            result.stderr,
        )
