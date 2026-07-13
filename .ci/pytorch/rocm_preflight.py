#!/usr/bin/env python3
"""Fast ROCm/RCCL pre-flight health check for distributed CI shards.

Some gfx950 runner pods come up with a container that cannot read part of the
KFD/HSA topology -- RCCL fails every collective init with
"ncclUnhandledCudaError: Call to CUDA function failed / Could not read node #N"
(N is a fixed topology-node index for that pod). When a distributed shard lands
on such a pod, the first collective crashes and a later test hangs the whole
shard until the 270-minute job timeout. Host-side ``rocminfo`` still enumerates
the GPUs on these pods, so the existing GPU-count gate does not catch it.

This runs a tiny multi-rank all_reduce inside the test container before the
suite. If it fails (or is killed by the outer ``timeout``), the job fails in
seconds with a clear message instead of hanging, and the bad pod can be drained.
"""

import datetime
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


_PG_TIMEOUT = datetime.timedelta(seconds=60)


def _worker(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, timeout=_PG_TIMEOUT
    )
    try:
        t = torch.ones(16, device=f"cuda:{rank}")
        dist.all_reduce(t)
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


def _diag() -> None:
    """Print container-side GPU visibility to disambiguate a device_count()==0
    failure: a container KFD/topology-passthrough gap (no /dev/dri render nodes,
    /dev/kfd unreadable, rocminfo sees 0 agents) vs. a torch/HIP-layer problem
    (devices present to the OS/rocminfo but torch still enumerates 0)."""
    import ctypes
    import glob
    import pathlib
    import subprocess

    try:
        print(f"[preflight-diag] /dev/dri: {sorted(glob.glob('/dev/dri/*'))}")
        print(f"[preflight-diag] /dev/kfd readable: {os.access('/dev/kfd', os.R_OK)}")
        for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            print(f"[preflight-diag] {var}={os.environ.get(var, '<unset>')}")
        rocminfo = subprocess.run(
            "rocminfo | grep -c -E 'Name:.*\\sgfx'",
            shell=True,
            capture_output=True,
            text=True,
        )
        print(f"[preflight-diag] rocminfo gfx-agent count: {rocminfo.stdout.strip()!r}")
        print(f"[preflight-diag] torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"[preflight-diag] torch.version.hip: {torch.version.hip}")

        # Driver-truth: the LOADED kernel driver, not image-baked dpkg metadata
        # (which is identical on every pod). This is the load-bearing fact for a
        # KMD/UMD-skew diagnosis.
        uname = subprocess.run(["uname", "-r"], capture_output=True, text=True)
        print(f"[preflight-diag] uname -r: {uname.stdout.strip()}")
        for label, path in (
            ("amdgpu KMD version", "/sys/module/amdgpu/version"),
            ("kfd interface version", "/sys/devices/virtual/kfd/kfd/version"),
            ("kfd topology generation", "/sys/class/kfd/kfd/topology/generation_id"),
        ):
            p = pathlib.Path(path)
            val = p.read_text().strip() if p.exists() else "<absent>"
            print(f"[preflight-diag] {label} ({path}): {val}")

        # Resolve the libamdhip64 torch actually loaded (bare soname is not on
        # the loader path in the wheel container -- torch finds it via its own
        # RPATH). Read the real path from /proc/self/maps.
        import re

        hip_path = None
        try:
            with open("/proc/self/maps") as maps:
                for line in maps:
                    m = re.search(r"(/\S*libamdhip64\.so[.\d]*)", line)
                    if m:
                        hip_path = m.group(1)
                        break
        except Exception as e:  # noqa: BLE001
            print(f"[preflight-diag] could not read /proc/self/maps: {e}")
        print(f"[preflight-diag] resolved libamdhip64 path: {hip_path}")

        # Remove torch from the causal chain: query libamdhip64 directly.
        if hip_path:
            try:
                hip = ctypes.CDLL(hip_path)
                raw = ctypes.c_int(-1)
                rc = hip.hipGetDeviceCount(ctypes.byref(raw))
                print(f"[preflight-diag] raw hipGetDeviceCount rc={rc} count={raw.value}")
            except Exception as e:  # noqa: BLE001
                print(f"[preflight-diag] raw hipGetDeviceCount failed: {e}")

        # CLR verbose init log: on failure HIP names why here (KFD version below
        # minimum, ioctl EINVAL, queue-create failure, no usable device). The
        # child imports torch first so libamdhip64 resolves via torch's RPATH,
        # then re-reads the resolved path and calls hipGetDeviceCount raw.
        try:
            clr = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import torch, ctypes, re;"
                    "p=[re.search(r'(/\\S*libamdhip64\\.so[.\\d]*)',l).group(1)"
                    " for l in open('/proc/self/maps') if 'libamdhip64' in l][0];"
                    "h=ctypes.CDLL(p); n=ctypes.c_int(-1);"
                    "rc=h.hipGetDeviceCount(ctypes.byref(n));"
                    "print('rc-count', rc, n.value)",
                ],
                env={**os.environ, "AMD_LOG_LEVEL": "4"},
                capture_output=True,
                text=True,
                timeout=120,
            )
            lines = (clr.stderr or "").strip().splitlines()
            # On failure the reason can appear anywhere, drowned by per-device
            # spam. Surface flagged lines first, then a bounded tail fallback.
            flagged = [
                ln
                for ln in lines
                if any(
                    k in ln.lower()
                    for k in (
                        "error",
                        "fail",
                        "einval",
                        "no device",
                        "no gpu",
                        "not supported",
                        "unsupported",
                        "version",
                        "minimum",
                        "unable",
                        "abort",
                        "kfd",
                        "ioctl",
                        "queue",
                        "hsakmt",
                        "topology",
                    )
                )
            ]
            print(f"[preflight-diag] raw child stdout: {clr.stdout.strip()!r}")
            print("[preflight-diag] --- CLR AMD_LOG_LEVEL=4 flagged lines ---")
            for line in flagged[-60:]:
                print(f"[preflight-diag-clr] {line}")
            print("[preflight-diag] --- CLR AMD_LOG_LEVEL=4 tail (25 lines) ---")
            for line in lines[-25:]:
                print(f"[preflight-diag-clr] {line}")
        except Exception as e:  # noqa: BLE001
            print(f"[preflight-diag] CLR log capture failed: {e}")
    except Exception as e:  # noqa: BLE001 - diagnostics must never mask the real error
        print(f"[preflight-diag] error while collecting diagnostics: {e}")


def main() -> int:
    n = torch.cuda.device_count()
    if n < 1:
        _diag()
        print("::error::ROCm pre-flight: no GPUs visible to the container")
        return 1
    world_size = min(2, n)
    try:
        mp.spawn(_worker, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        print(
            "::error::ROCm/RCCL pre-flight failed: the container cannot "
            f"initialize a {world_size}-rank collective on this runner -- "
            "likely a broken KFD/HSA topology on this pod (look for "
            '"Could not read node #N"). Failing fast instead of hanging; '
            f"re-run to land on a healthy runner. Underlying error: {e}"
        )
        return 1
    print(f"ROCm/RCCL pre-flight passed ({world_size}-rank all_reduce)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
