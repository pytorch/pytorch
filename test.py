"""
Test bitwise equivalence of torch.pow between eager CUDA and inductor (Triton).

Problem:
  Triton bundles its own libdevice.10.bc which is a different version than the
  system CUDA libdevice that nvcc links into eager's aten kernels. Both implement
  __nv_powf / __nv_pow, but the different versions produce different results
  (typically ~1 ULP for float32).

Fix:
  Set TRITON_LIBDEVICE_PATH to the system CUDA libdevice that matches the CUDA
  version PyTorch was built with. This makes Triton use the same __nv_powf
  implementation as eager.

Observed results (CUDA 12.9, A100):
  With system libdevice: bitwise equal for all normal-range inputs across
  float32 (370K+), float16 (70K), bfloat16 (60K), float64 (70K).

  Remaining divergences: float32 and bfloat16 subnormal inputs still differ
  due to Triton's FTZ behavior that config.eager_numerics.disable_ftz doesn't
  fully resolve. float64 subnormals are unaffected (FTZ only applies to fp32).

  The inductor config knobs (emulate_precision_casts, disable_ftz,
  division_rounding) have no effect on pow — the divergence is entirely
  from the libdevice version mismatch.

Usage:
  # Default (Triton's bundled libdevice, expect 354/1024 diffs for float32):
  python test_pow_bitwise_equivalence.py

  # With system libdevice (expect bitwise equal for normal-range inputs):
  TRITON_LIBDEVICE_PATH=$(python -c "import torch; print(f'/usr/local/cuda-{torch.version.cuda}/nvvm/libdevice/libdevice.10.bc')") \\
      python test_pow_bitwise_equivalence.py

  # Verify TRITON_LIBDEVICE_PATH is respected (runs subprocesses with each):
  python test_pow_bitwise_equivalence.py --verify-env
"""

import argparse
import hashlib
import os
import shutil

import torch
import torch._inductor.config as inductor_config
from torch._inductor.runtime.runtime_utils import cache_dir


# ── Expected results (CUDA 12.9, A100, torch.manual_seed(42)) ──────────────
#
# With TRITON_LIBDEVICE_PATH=<system cuda libdevice>:
#
#   float32:
#     random 100K                         BITWISE EQUAL (0/100000)
#     subnormal bases (1e-38..1e-45)      DIFFER 3728/10000   (FTZ issue)
#     large bases (10..1e38)              BITWISE EQUAL (0/10000)
#     negative exponents (-10..0)         BITWISE EQUAL (0/50000)
#     large positive exponents (0..50)    BITWISE EQUAL (0/50000)
#     integer exponents (-5..5)           BITWISE EQUAL (0/50000)
#     fractional exponents (roots)        BITWISE EQUAL (0/50000)
#     base near 1.0, large exp            BITWISE EQUAL (0/50000)
#     tiny base & tiny exp                BITWISE EQUAL (0/10000)
#     special values (0, 1, inf, min,max) DIFFER 439/7000     (contains 1e-45 subnormal)
#   float16:
#     random 50K                          BITWISE EQUAL (0/50000)
#     wide range bases                    BITWISE EQUAL (0/10000)
#     subnormal bases                     BITWISE EQUAL (0/10000)
#   bfloat16:
#     random 50K                          BITWISE EQUAL (0/50000)
#     wide range bases                    BITWISE EQUAL (0/10000)
#     subnormal bases                     DIFFER 2512/10000   (FTZ issue)
#   float64:
#     random 50K                          BITWISE EQUAL (0/50000)
#     wide range bases (1e-300..1e300)    BITWISE EQUAL (0/10000)
#     subnormal bases (1e-308..1e-323)    BITWISE EQUAL (0/10000)
#
# Without TRITON_LIBDEVICE_PATH (Triton's bundled libdevice):
#   float32 random 100K                  DIFFER 354/1024 (on a 1024-element test)
#   (libdevice version mismatch causes ~1 ULP diffs on normal-range float32)


def clear_all_caches():
    shutil.rmtree(cache_dir(), ignore_errors=True)
    shutil.rmtree(os.path.expanduser("~/.triton/cache"), ignore_errors=True)


def check_libdevice_paths():
    import triton

    triton_dir = os.path.dirname(triton.__file__)

    triton_path = None
    for root, dirs, files in os.walk(triton_dir):
        for f in files:
            if f == "libdevice.10.bc":
                triton_path = os.path.join(root, f)
                break
        if triton_path:
            break

    system_path = None
    cuda_version = torch.version.cuda
    for candidate in [
        f"/usr/local/cuda-{cuda_version}/nvvm/libdevice/libdevice.10.bc",
        "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
    ]:
        if os.path.exists(candidate):
            system_path = candidate
            break

    def file_info(path):
        if path and os.path.exists(path):
            size = os.path.getsize(path)
            md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
            return f"{path} (size={size}, md5={md5})"
        return path or "not found"

    override = os.environ.get("TRITON_LIBDEVICE_PATH")
    from triton import knobs
    knobs_val = knobs.nvidia.libdevice_path

    print("=== Libdevice configuration ===")
    print(f"  PyTorch CUDA version:        {torch.version.cuda}")
    print(f"  Triton bundled:              {file_info(triton_path)}")
    print(f"  System CUDA:                 {file_info(system_path)}")
    print(f"  TRITON_LIBDEVICE_PATH env:   {override or 'not set'}")
    print(f"  knobs.nvidia.libdevice_path: {knobs_val}")

    if triton_path and system_path:
        triton_md5 = hashlib.md5(open(triton_path, "rb").read()).hexdigest()
        system_md5 = hashlib.md5(open(system_path, "rb").read()).hexdigest()
        match = "MATCH" if triton_md5 == system_md5 else "MISMATCH"
        print(f"  Triton vs system:            {match}")

    print()
    return system_path


def run_test(name, base, exp, compiled_pow):
    eager_result = torch.pow(base, exp)
    compiled_result = compiled_pow(base, exp)

    diff_mask = (eager_result != compiled_result) & ~(
        torch.isnan(eager_result) & torch.isnan(compiled_result)
    )
    n_diff = diff_mask.sum().item()
    total = base.numel()

    if n_diff == 0:
        print(f"  {name:45s}: BITWISE EQUAL (0/{total})")
    else:
        max_diff = (
            (eager_result[diff_mask] - compiled_result[diff_mask])
            .abs()
            .max()
            .item()
        )
        print(f"  {name:45s}: DIFFER {n_diff}/{total}, max_abs_diff={max_diff:.6e}")

    return n_diff


def run_comprehensive_tests():
    torch.manual_seed(42)
    device = "cuda"
    total_diffs = 0

    clear_all_caches()
    torch._dynamo.reset()

    with inductor_config.patch(
        emulate_precision_casts=True,
        **{"eager_numerics.disable_ftz": True},
    ):
        compiled_pow = torch.compile(torch.pow)

        # Warm up compilation
        _ = compiled_pow(
            torch.ones(4, device=device, dtype=torch.float32),
            torch.ones(4, device=device, dtype=torch.float32),
        )

        # --- float32 ---
        print("--- float32 ---")

        b = torch.randn(100000, device=device, dtype=torch.float32).abs() + 1e-6
        e = torch.randn(100000, device=device, dtype=torch.float32)
        total_diffs += run_test("random 100K", b, e, compiled_pow)

        b = torch.logspace(-38, -45, 10000, device=device, dtype=torch.float32)
        e = torch.linspace(0.1, 3.0, 10000, device=device, dtype=torch.float32)
        total_diffs += run_test("subnormal bases (1e-38..1e-45)", b, e, compiled_pow)

        b = torch.logspace(1, 38, 10000, device=device, dtype=torch.float32)
        e = torch.linspace(-1.0, 1.0, 10000, device=device, dtype=torch.float32)
        total_diffs += run_test("large bases (10..1e38)", b, e, compiled_pow)

        b = torch.rand(50000, device=device, dtype=torch.float32) + 0.01
        e = -torch.rand(50000, device=device, dtype=torch.float32) * 10
        total_diffs += run_test("negative exponents (-10..0)", b, e, compiled_pow)

        b = torch.rand(50000, device=device, dtype=torch.float32) * 0.5 + 0.5
        e = torch.rand(50000, device=device, dtype=torch.float32) * 50
        total_diffs += run_test("large positive exponents (0..50)", b, e, compiled_pow)

        b = torch.randn(50000, device=device, dtype=torch.float32).abs() + 0.01
        e = torch.randint(-5, 6, (50000,), device=device, dtype=torch.float32)
        total_diffs += run_test("integer exponents (-5..5)", b, e, compiled_pow)

        b = torch.rand(50000, device=device, dtype=torch.float32) * 100 + 0.001
        e = torch.tensor(
            [0.5, 1.0 / 3, 0.25, 0.1, 0.01], device=device, dtype=torch.float32
        ).repeat(10000)
        total_diffs += run_test("fractional exponents (roots)", b, e, compiled_pow)

        b = 1.0 + (torch.rand(50000, device=device, dtype=torch.float32) - 0.5) * 1e-3
        e = torch.randn(50000, device=device, dtype=torch.float32) * 100
        total_diffs += run_test("base near 1.0, large exp", b, e, compiled_pow)

        b = torch.rand(10000, device=device, dtype=torch.float32) * 1e-6
        e = torch.rand(10000, device=device, dtype=torch.float32) * 1e-3
        total_diffs += run_test("tiny base & tiny exp", b, e, compiled_pow)

        special = torch.tensor(
            [0.0, 1.0, float("inf"), 1e-45, 1e-38, 1e38, 3.4028235e38],
            device=device, dtype=torch.float32,
        )
        b = special.repeat(1000)
        e = torch.linspace(-5, 5, b.shape[0], device=device, dtype=torch.float32)
        total_diffs += run_test("special values (0, 1, inf, min, max)", b, e, compiled_pow)

    # fp16, bf16, fp64 need separate compilation (different kernels per dtype)
    for dtype, dtype_name, test_cases in [
        (
            torch.float16,
            "float16",
            [
                ("random 50K", lambda: (
                    torch.randn(50000, device=device, dtype=torch.float16).abs() + 1e-3,
                    torch.randn(50000, device=device, dtype=torch.float16),
                )),
                ("wide range bases", lambda: (
                    torch.logspace(-4, 4, 10000, device=device, dtype=torch.float32).half(),
                    torch.linspace(-2, 2, 10000, device=device, dtype=torch.float16),
                )),
                ("subnormal bases", lambda: (
                    torch.tensor([6e-8, 1e-7, 5e-8, 2e-8] * 2500, device=device, dtype=torch.float16),
                    torch.linspace(0.1, 2.0, 10000, device=device, dtype=torch.float16),
                )),
            ],
        ),
        (
            torch.bfloat16,
            "bfloat16",
            [
                ("random 50K", lambda: (
                    torch.randn(50000, device=device, dtype=torch.bfloat16).abs() + 1e-3,
                    torch.randn(50000, device=device, dtype=torch.bfloat16),
                )),
                ("wide range bases", lambda: (
                    torch.logspace(-4, 4, 10000, device=device, dtype=torch.float32).bfloat16(),
                    torch.linspace(-2, 2, 10000, device=device, dtype=torch.bfloat16),
                )),
                ("subnormal bases", lambda: (
                    torch.tensor([1e-40, 5e-41, 1e-41, 5e-42] * 2500, device=device, dtype=torch.bfloat16),
                    torch.linspace(0.1, 2.0, 10000, device=device, dtype=torch.bfloat16),
                )),
            ],
        ),
        (
            torch.float64,
            "float64",
            [
                ("random 50K", lambda: (
                    torch.randn(50000, device=device, dtype=torch.float64).abs() + 1e-15,
                    torch.randn(50000, device=device, dtype=torch.float64),
                )),
                ("wide range bases (1e-300..1e300)", lambda: (
                    torch.logspace(-300, 300, 10000, device=device, dtype=torch.float64),
                    torch.linspace(-1, 1, 10000, device=device, dtype=torch.float64),
                )),
                ("subnormal bases (1e-308..1e-323)", lambda: (
                    torch.logspace(-308, -323, 10000, device=device, dtype=torch.float64),
                    torch.linspace(0.1, 1.0, 10000, device=device, dtype=torch.float64),
                )),
            ],
        ),
    ]:
        print(f"\n--- {dtype_name} ---")
        torch._dynamo.reset()
        with inductor_config.patch(
            emulate_precision_casts=True,
            **{"eager_numerics.disable_ftz": True},
        ):
            compiled_pow = torch.compile(torch.pow)
            for name, make_inputs in test_cases:
                b, e = make_inputs()
                total_diffs += run_test(name, b, e, compiled_pow)

    return total_diffs


def verify_env():
    """Verify TRITON_LIBDEVICE_PATH is respected by running each config in a subprocess."""
    import subprocess
    import sys

    script = """\
import os, shutil, torch, torch._inductor.config as cfg
from torch._inductor.runtime.runtime_utils import cache_dir
from triton import knobs

shutil.rmtree(cache_dir(), ignore_errors=True)
shutil.rmtree(os.path.expanduser("~/.triton/cache"), ignore_errors=True)
torch._dynamo.reset()
torch.manual_seed(42)

base = torch.randn(1024, device="cuda", dtype=torch.float32).abs() + 1e-6
exp = torch.randn(1024, device="cuda", dtype=torch.float32)
eager = torch.pow(base, exp)

with cfg.patch(emulate_precision_casts=True, **{"eager_numerics.disable_ftz": True}):
    compiled = torch.compile(torch.pow)(base, exp)

diff = ((eager != compiled) & ~(torch.isnan(eager) & torch.isnan(compiled))).sum().item()
env = os.environ.get("TRITON_LIBDEVICE_PATH", "not set")
knob = knobs.nvidia.libdevice_path
print(f"TRITON_LIBDEVICE_PATH={env}")
print(f"knobs.nvidia.libdevice_path={knob}")
print(f"diffs={diff}/1024")
"""

    print("=== Verification: TRITON_LIBDEVICE_PATH is respected ===")
    print("  (each test runs in a fresh subprocess)\n")

    system_path = None
    cuda_version = torch.version.cuda
    for candidate in [
        f"/usr/local/cuda-{cuda_version}/nvvm/libdevice/libdevice.10.bc",
        "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
    ]:
        if os.path.exists(candidate):
            system_path = candidate
            break

    import triton
    triton_dir = os.path.dirname(triton.__file__)
    triton_path = None
    for root, dirs, files in os.walk(triton_dir):
        for f in files:
            if f == "libdevice.10.bc":
                triton_path = os.path.join(root, f)
                break
        if triton_path:
            break

    tests = [
        ("no env var (Triton default)", {}, "expect ~354 diffs"),
    ]
    if triton_path:
        tests.append(("Triton bundled (explicit)", {"TRITON_LIBDEVICE_PATH": triton_path}, "expect ~354 diffs"))
    if system_path:
        tests.append(("system CUDA libdevice", {"TRITON_LIBDEVICE_PATH": system_path}, "expect 0 diffs"))
    tests.append(("bogus path", {"TRITON_LIBDEVICE_PATH": "/nonexistent/libdevice.10.bc"}, "expect error"))

    for label, extra_env, expectation in tests:
        print(f"--- {label} ({expectation}) ---")
        env = os.environ.copy()
        # Remove TRITON_LIBDEVICE_PATH from parent env for clean test
        env.pop("TRITON_LIBDEVICE_PATH", None)
        env.update(extra_env)
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env, capture_output=True, text=True, timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        for line in output.split("\n"):
            if any(k in line for k in ["TRITON_LIBDEVICE", "knobs.", "diffs=", "Error", "error"]):
                print(f"  {line}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verify-env", action="store_true",
        help="Verify TRITON_LIBDEVICE_PATH is respected (runs subprocesses)",
    )
    args = parser.parse_args()

    system_path = check_libdevice_paths()

    if args.verify_env:
        verify_env()
        return

    print("=== Comprehensive pow bitwise equivalence tests ===\n")
    total_diffs = run_comprehensive_tests()

    print(f"\n{'=' * 75}")
    if total_diffs == 0:
        print("ALL TESTS BITWISE EQUAL")
    else:
        print(f"TOTAL DIFFS: {total_diffs}")

    override = os.environ.get("TRITON_LIBDEVICE_PATH")
    if not override and system_path:
        print(f"\nTip: try running with:")
        print(f"  TRITON_LIBDEVICE_PATH={system_path} python {os.path.basename(__file__)}")


if __name__ == "__main__":
    main()

