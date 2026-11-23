# Copyright (c) 2026, Tri Dao.
"""Forkserver preload module for the async compile pool.

Imported once inside the multiprocessing *forkserver* process (see
``multiprocessing.set_forkserver_preload``). Every pool worker is then
``fork()``-ed from that warm process and inherits the imported interpreter
state via copy-on-write: worker startup drops from ~13 s (torch 4 s +
cutlass/cute/tvm_ffi 9 s per spawn) to ~0.1 s per fork.

This is the same architecture as PyTorch Inductor's compile-worker
``SubprocPool``: one sidecar pays the import, workers fork from it.

Fork-safety: nothing here may initialize CUDA (a forked child of a
CUDA-initialized process is undefined behavior). Importing torch and
cutlass does not create a CUDA context; workers additionally run with
``CUDA_VISIBLE_DEVICES=""`` + ``QUACK_ARCH``/``CUTE_DSL_ARCH`` overrides so
the compile path never touches the driver (the same mechanism the CPU-only
compile workflow uses).
"""

import os
import subprocess

# Pin both arch overrides BEFORE importing quack: import-time code paths
# (e.g. rmsnorm_config._detect_arch_major) consult QUACK_ARCH via
# get_device_capacity and would otherwise initialize CUDA — which both makes
# the forkserver's context leak into children and trips torch's forked-child
# guard. nvidia-smi queries the capability without creating a CUDA context.
#
# NOTE: by the time this module body runs, its parent packages (``quack``,
# ``quack.cache``) have already been imported — and ``quack/__init__`` pulls
# in cutlass, so the DSL's env manager is constructed before these env vars
# exist. cutlass-dsl >= 4.6.2 snapshots CUTE_DSL_ARCH at that construction,
# so the pinning here is NOT enough for the ptxas target: the explicit
# ``_pin_dsl_arch`` re-latch at the bottom of this module is what actually
# lands it. The env vars still matter for everything that reads them lazily
# (QUACK_ARCH dispatch, the gpu-blind smem-capacity shim).
#
# The two overrides are pinned independently: QUACK_ARCH (dispatch) is
# respected if the caller set it (the CI proxy legs), while CUTE_DSL_ARCH
# (ptxas target) always defaults to the PHYSICAL arch — never derived from
# QUACK_ARCH — because the main process loads the .o on the physical GPU
# (see async_compile._detect_arch_env).
try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    cap = out.stdout.strip().splitlines()[0].strip()  # e.g. "9.0"
    major, minor = cap.split(".")
    os.environ.setdefault("QUACK_ARCH", f"{major}{minor}")
    os.environ.setdefault(
        "CUTE_DSL_ARCH", f"sm_{major}{minor}a" if int(major) >= 9 else f"sm_{major}{minor}"
    )
except Exception:
    pass  # CPU-only box: rely on user-provided env, as before

# Belt and suspenders: even if some import still tries to touch CUDA, make
# it see no devices rather than creating a context in the forkserver.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch._vendor.quack.cache  # noqa: F401, E402  (pulls torch, cutlass.cute, tvm_ffi)

# Re-latch the ptxas target on the already-constructed DSL singleton (see the
# NOTE above): forked workers inherit the latched value. Without this, a
# GPU-blind worker under cutlass-dsl >= 4.6.2 defaults to sm_100a and every
# pool .o fails to load with cudaErrorNoKernelImageForDevice — which the
# 4.6.2 runtime then escalates into a spinlock hang on the next launch.
from torch._vendor.quack.cache.async_compile import _pin_dsl_arch  # noqa: E402

_pin_dsl_arch(os.environ.get("CUTE_DSL_ARCH"))
