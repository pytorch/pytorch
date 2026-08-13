#!/usr/bin/env python3
"""Smoke-test ROCm runtime paths that exercise kernel load, hipsolver, and MIOpen.

Useful for validating ROCm wheel installs (e.g. TheRock nightlies) and for
spot-checking CLR code-object loading issues such as ROCM-29159.

Usage:
    python .ci/rocm/smoke_test_rocm_runtime.py

Requires a working torch build with matching amd-torch-device-gfx* wheel when
using TheRock multi-arch layouts.
"""

from __future__ import annotations

import sys
import traceback

import torch


def _run(name: str, fn) -> bool:
    try:
        fn()
        print(f"PASS: {name}")
        return True
    except Exception as exc:
        print(f"FAIL: {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=2)
        return False


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA/ROCm device available")
        sys.exit(0)

    device = "cuda"
    print(f"torch {torch.__version__} hip {torch.version.hip}")
    print(f"device: {torch.cuda.get_device_name(0)}")

    results = [
        _run(
            "matmul",
            lambda: (
                torch.randn(512, 512, device=device)
                @ torch.randn(512, 512, device=device)
            ).sum().item(),
        ),
        _run(
            "svd/hipsolver",
            lambda: torch.linalg.svd(
                torch.randn(64, 64, device=device, dtype=torch.float32)
            ),
        ),
        _run(
            "conv/MIOpen",
            lambda: torch.nn.functional.conv2d(
                torch.randn(2, 3, 32, 32, device=device),
                torch.randn(8, 3, 3, 3, device=device),
            ).sum().item(),
        ),
    ]

    def inductor_smoke() -> None:
        import torch._inductor.config as inductor_config

        inductor_config.triton.cudagraphs = False

        @torch.compile(backend="inductor")
        def f(x):
            return x * 2 + 1

        x = torch.randn(1024, device=device)
        y = f(x)
        torch.cuda.synchronize()
        assert y.shape == x.shape

    results.append(_run("inductor/compile (hipModuleLoadData path)", inductor_smoke))

    passed = sum(results)
    print(f"\n=== {passed}/{len(results)} tests passed ===")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
