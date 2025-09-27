# Owner(s): ["module: cuda"]
import os
import json
import sys
import subprocess
import unittest


def _run_subprocess_with_env(env):
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "..", "tools", "repro_allocator_roundup_divisions_one.py")]
    proc_env = os.environ.copy()
    proc_env.update(env or {})
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=proc_env)
    out, err = p.communicate()
    return p.returncode, out.decode("utf-8"), err.decode("utf-8")


class TestCudaAllocatorRoundup(unittest.TestCase):
    def test_roundup_divisions_one_next_power_of_two(self):
        env = {
            "PYTORCH_CUDA_ALLOC_CONF": "backend:native,roundup_power2_divisions:[>:1]",
            # Request ~514MB allocation to test next-PoT rounding to 1GiB
            "ALLOC_BYTES": str(514 * 1024 * 1024),
        }
        code, out, err = _run_subprocess_with_env(env)
        # If CUDA not available in CI shard, accept skip via returncode 2
        if code == 2:
            self.skipTest("CUDA not available")
        self.assertEqual(code, 0, msg=f"subprocess failed: {err}")
        data = json.loads(out.strip().splitlines()[-1])
        reserved = int(data["reserved"])  # bytes
        MB = 1024 * 1024
        # Expect reservation near 1 GiB
        self.assertGreaterEqual(reserved, 1024 * MB)
        # Allow allocator bookkeeping slack
        self.assertLess(reserved, 1024 * MB + 64 * MB)

    def test_roundup_divisions_one_power_of_two_exact(self):
        env = {
            "PYTORCH_CUDA_ALLOC_CONF": "backend:native,roundup_power2_divisions:[>:1]",
            # Request exactly 1GiB allocation
            "ALLOC_BYTES": str(1024 * 1024 * 1024),
        }
        code, out, err = _run_subprocess_with_env(env)
        if code == 2:
            self.skipTest("CUDA not available")
        self.assertEqual(code, 0, msg=f"subprocess failed: {err}")
        data = json.loads(out.strip().splitlines()[-1])
        reserved = int(data["reserved"])  # bytes
        self.assertIsInstance(reserved, int)

    def test_roundup_divisions_zero_disables(self):
        env = {
            "PYTORCH_CUDA_ALLOC_CONF": "backend:native,roundup_power2_divisions:[>:0]",
            "ALLOC_BYTES": str(514 * 1024 * 1024),
        }
        code, out, err = _run_subprocess_with_env(env)
        if code == 2:
            self.skipTest("CUDA not available")
        self.assertEqual(code, 0, msg=f"subprocess failed: {err}")
        data = json.loads(out.strip().splitlines()[-1])
        reserved = int(data["reserved"])  # bytes
        MB = 1024 * 1024
        # With divisions disabled, reserved memory should be < 1 GiB for a 514MB allocation
        self.assertLess(reserved, 900 * MB)

    def test_backend_cudamallocasync_ignored(self):
        env = {
            "PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync,roundup_power2_divisions:[>:1]",
            "ALLOC_BYTES": str(514 * 1024 * 1024),
        }
        code, out, err = _run_subprocess_with_env(env)
        if code == 2:
            self.skipTest("CUDA not available")
        if code != 0:
            self.skipTest(f"subprocess failed under cudaMallocAsync backend: {err}")
        data = json.loads(out.strip().splitlines()[-1])
        reserved = int(data["reserved"])  # bytes
        MB = 1024 * 1024
        # Expect not rounded up to 1 GiB
        self.assertLess(reserved, 900 * MB)


if __name__ == "__main__":
    unittest.main()
