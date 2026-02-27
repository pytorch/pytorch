"""
Test that FxGraphCache correctly invalidates when TRITON_LIBDEVICE_PATH changes.

Uses bitwise pow differences as a fingerprint for which libdevice was used:
- Triton's bundled libdevice produces ~392/1024 bitwise diffs vs eager CUDA pow
- CUDA toolkit's libdevice produces ~128/1024 diffs (closer match)

Test 1 (env var at process start):
  Sets TRITON_LIBDEVICE_PATH before Python starts.

Test 2 (env var mid-process, before torch.compile):
  Sets TRITON_LIBDEVICE_PATH after triton_hash_with_backend() is cached but
  before torch.compile runs.
"""

import os
import subprocess
import sys
import tempfile

import torch

WORKER_ENV = r"""
import torch

torch.manual_seed(42)
base = torch.rand(1024, device="cuda", dtype=torch.float32) * 10 + 0.1
exp = torch.rand(1024, device="cuda", dtype=torch.float32) * 5
ref = torch.pow(base, exp)

torch._dynamo.reset()

@torch.compile
def compiled_pow(x, y):
    return torch.pow(x, y)

result = compiled_pow(base, exp)
diffs = (ref != result).sum().item()
print(diffs)
"""

WORKER_MIDPROCESS = r"""
import os, sys, torch

libdevice_path = sys.argv[1] if len(sys.argv) > 1 else ""

torch.manual_seed(42)
base = torch.rand(1024, device="cuda", dtype=torch.float32) * 10 + 0.1
exp = torch.rand(1024, device="cuda", dtype=torch.float32) * 5
ref = torch.pow(base, exp)

torch._dynamo.reset()

# Cache triton_hash_with_backend() first (simulating codegen happening early)
_ = torch.utils._triton.triton_hash_with_backend()

# Set env var mid-process (like _set_triton_libdevice_path would)
if libdevice_path:
    os.environ["TRITON_LIBDEVICE_PATH"] = libdevice_path

@torch.compile
def compiled_pow(x, y):
    return torch.pow(x, y)

result = compiled_pow(base, exp)
diffs = (ref != result).sum().item()
print(diffs)
"""


CUDA_LIBDEVICE = f"/usr/local/cuda-{torch.version.cuda}/nvvm/libdevice/libdevice.10.bc"


def run_worker(cache_dir, script, *, libdevice_env=None, libdevice_arg=None):
    env = os.environ.copy()
    env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    env.pop("TRITON_LIBDEVICE_PATH", None)
    if libdevice_env:
        env["TRITON_LIBDEVICE_PATH"] = libdevice_env

    cmd = [sys.executable, "-c", script]
    if libdevice_arg:
        cmd.append(libdevice_arg)

    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(f"STDERR:\n{r.stderr}", file=sys.stderr)
        raise RuntimeError(f"Worker failed (rc={r.returncode})")
    return int(r.stdout.strip())


def test_env_at_startup():
    """TRITON_LIBDEVICE_PATH set before process starts."""
    print("=" * 60)
    print("Test 1: TRITON_LIBDEVICE_PATH set at process startup")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as cache_dir:
        d1 = run_worker(cache_dir, WORKER_ENV)
        print(f"  Run 1 (bundled):      {d1} diffs")

        d2 = run_worker(cache_dir, WORKER_ENV, libdevice_env=CUDA_LIBDEVICE)
        print(f"  Run 2 (cuda toolkit): {d2} diffs")

        d3 = run_worker(cache_dir, WORKER_ENV)
        print(f"  Run 3 (bundled):      {d3} diffs")

        ok = d1 != d2 and d1 == d3
        print(f"  {'PASS' if ok else 'FAIL'}")
        return ok


def test_env_midprocess():
    """TRITON_LIBDEVICE_PATH set mid-process before torch.compile."""
    print()
    print("=" * 60)
    print("Test 2: TRITON_LIBDEVICE_PATH set mid-process")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as cache_dir:
        d1 = run_worker(cache_dir, WORKER_MIDPROCESS)
        print(f"  Run 1 (bundled):      {d1} diffs")

        d2 = run_worker(cache_dir, WORKER_MIDPROCESS, libdevice_arg=CUDA_LIBDEVICE)
        print(f"  Run 2 (cuda mid-proc): {d2} diffs")

        d3 = run_worker(cache_dir, WORKER_MIDPROCESS)
        print(f"  Run 3 (bundled):      {d3} diffs")

        ok = d1 != d2 and d1 == d3
        print(f"  {'PASS' if ok else 'FAIL'}")
        return ok


def main():
    if not os.path.isfile(CUDA_LIBDEVICE):
        print(f"SKIP: CUDA toolkit libdevice not found at {CUDA_LIBDEVICE}")
        return

    ok1 = test_env_at_startup()
    ok2 = test_env_midprocess()

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Test 1 (env at startup):    {'PASS' if ok1 else 'FAIL'}")
    print(f"  Test 2 (env mid-process):   {'PASS' if ok2 else 'FAIL'}")

    if not (ok1 and ok2):
        sys.exit(1)


if __name__ == "__main__":
    main()
