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

        # Provenance + arch-mismatch check. If the pod's device ISA is not in
        # the wheel's compiled arch list, fat-binary registration fails and
        # torch filters the device out -> device_count()==0 while raw HIP still
        # counts it. Compare agent ISA names against the wheel's arch flags.
        import socket

        print(f"[preflight-diag] hostname: {socket.gethostname()}")
        print(f"[preflight-diag] torch.__file__: {torch.__file__}")
        try:
            import importlib.metadata as _md

            print(
                f"[preflight-diag] _rocm_sdk_core version: "
                f"{_md.version('rocm-sdk-core')}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[preflight-diag] _rocm_sdk_core version raised: {e!r}")
        # get_arch_list() is gated on is_available(), which is False in the very
        # failure we are diagnosing -> it returns []. Use the ungated binding.
        try:
            print(
                f"[preflight-diag] arch flags (ungated): "
                f"{torch._C._cuda_getArchFlags()}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[preflight-diag] _cuda_getArchFlags raised: {e!r}")
        try:
            print(f"[preflight-diag] get_arch_list (gated): {torch.cuda.get_arch_list()}")
        except Exception as e:  # noqa: BLE001
            print(f"[preflight-diag] get_arch_list raised: {e!r}")
        agents = subprocess.run(["rocminfo"], capture_output=True, text=True)
        for ln in agents.stdout.splitlines():
            s = ln.strip()
            if (s.startswith("Name:") and "gfx" in s) or "Marketing Name:" in s:
                print(f"[preflight-diag] agent: {s}")

        # Split torch's device-counting paths, cheapest-and-most-isolated first
        # (before any amdsmi import, whose side effects could color results).
        # Whichever returns 0 while raw HIP counts >0 is the culprit layer. The
        # C++ path swallows init errors into a 0 return; _lazy_init re-raises the
        # real message.
        try:
            print(
                f"[preflight-diag] _C._cuda_getDeviceCount(): "
                f"{torch._C._cuda_getDeviceCount()}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[preflight-diag] _C._cuda_getDeviceCount raised: {e!r}")
        try:
            torch.cuda._lazy_init()
            print("[preflight-diag] _lazy_init OK")
        except Exception as e:  # noqa: BLE001
            print(f"[preflight-diag] _lazy_init raised: {e!r}")
        for name in ("_device_count_amdsmi", "_raw_device_count_amdsmi"):
            try:
                fn = getattr(torch.cuda, name, None)
                print(f"[preflight-diag] torch.cuda.{name}(): {fn() if fn else '<absent>'}")
            except Exception as e:  # noqa: BLE001
                print(f"[preflight-diag] torch.cuda.{name} raised: {e!r}")

        # Probe amdsmi directly (not through torch's wrapper): if torch's
        # amdsmi-based count zeroes out and short-circuits before HIP, the
        # direct probe names why. Kept last -- its init can mutate state.
        try:
            import amdsmi

            amdsmi.amdsmi_init()
            handles = amdsmi.amdsmi_get_processor_handles()
            print(f"[preflight-diag] amdsmi direct handle count: {len(handles)}")
        except Exception as e:  # noqa: BLE001
            print(f"[preflight-diag] amdsmi direct probe raised: {e!r}")

        # WHY does amdsmi/rsmi enumerate 0 while HIP counts N? amdsmi's GPU
        # count comes from rsmi_num_monitor_devices -> DiscoverAmdgpuDevices(),
        # which walks /sys/class/kfd/kfd/topology/nodes/. Per node (bare metal),
        # rsmi counts it iff ALL hold (rocm_smi_main.cc DiscoverAmdgpuDevices,
        # release/therock-7.10):
        #   - properties file exists & non-empty
        #   - gpu_id file readable & numeric, value != 0
        #   - properties has PARSEABLE (key present, not nonzero!) domain,
        #     location_id, AND drm_render_minor
        # On a VM guest (is_vm_guest) only gpu_id != 0 is required. HIP walks the
        # same topology but is more tolerant. Dump every key rsmi keys on so a
        # node HIP-accepts / rsmi-rejects divergence is legible in one run.
        try:
            import glob as _glob

            # VM-guest detection: rsmi relaxes to gpu_id-only on VMs.
            vm_guest = False
            try:
                cpuinfo = pathlib.Path("/proc/cpuinfo").read_text()
                vm_guest = "hypervisor" in cpuinfo
            except Exception:  # noqa: BLE001
                pass
            print(f"[preflight-diag] rsmi VM-guest path (relaxed): {vm_guest}")

            kfd_root = "/sys/class/kfd/kfd/topology/nodes"
            nodes = sorted(
                _glob.glob(f"{kfd_root}/*"),
                key=lambda p: int(os.path.basename(p)) if os.path.basename(p).isdigit() else -1,
            )
            print(f"[preflight-diag] KFD topology: {len(nodes)} node dir(s) under {kfd_root}")
            print(
                f"[preflight-diag] KFD nodes/ readable: "
                f"{os.access(kfd_root, os.R_OK | os.X_OK)}"
            )
            # Divergence probe (case c): Python getdents vs librsmi getdents.
            try:
                print(f"[preflight-diag] KFD nodes/ os.listdir count: {len(os.listdir(kfd_root))}")
            except Exception as e:  # noqa: BLE001
                print(f"[preflight-diag] KFD nodes/ os.listdir raised: {e!r}")

            def _key(props, k):
                # Return the value if the key is PRESENT & has a token (rsmi's
                # gate is parseable, i.e. present, NOT nonzero), else "<absent>".
                for ln in props.splitlines():
                    if ln.startswith(k + " "):
                        parts = ln.split()
                        return parts[1] if len(parts) > 1 else "<empty>"
                return "<absent>"

            gpu_nodes = 0
            rsmi_countable = 0
            for nd in nodes:
                name = os.path.basename(nd)
                try:
                    gpu_id = pathlib.Path(f"{nd}/gpu_id").read_text().strip()
                except Exception as e:  # noqa: BLE001
                    print(f"[preflight-diag] KFD node {name}: gpu_id UNREADABLE ({e!r})")
                    continue
                props_path = pathlib.Path(f"{nd}/properties")
                props = props_path.read_text() if props_path.exists() else ""
                props_ok = bool(props.strip())
                rmin = _key(props, "drm_render_minor")
                loc = _key(props, "location_id")
                dom = _key(props, "domain")
                simd = _key(props, "simd_count")
                gfxv = _key(props, "gfx_target_version")
                is_gpu = gpu_id != "0"
                if is_gpu:
                    gpu_nodes += 1
                # Faithful rsmi predicate: keys must be PRESENT (parseable), not
                # nonzero. On VM guest only gpu_id!=0 is required.
                if vm_guest:
                    rsmi_ok = props_ok and is_gpu
                else:
                    rsmi_ok = (
                        props_ok
                        and is_gpu
                        and rmin != "<absent>"
                        and loc != "<absent>"
                        and dom != "<absent>"
                    )
                if rsmi_ok:
                    rsmi_countable += 1
                print(
                    f"[preflight-diag] KFD node {name}: gpu_id={gpu_id} "
                    f"props_nonempty={props_ok} drm_render_minor={rmin} "
                    f"location_id={loc} domain={dom} simd_count={simd} "
                    f"gfx_target_version={gfxv} is_gpu={is_gpu} "
                    f"rsmi_would_count={rsmi_ok}"
                )
            print(
                f"[preflight-diag] KFD GPU-node count (gpu_id!=0): {gpu_nodes}; "
                f"rsmi-countable: {rsmi_countable}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[preflight-diag] KFD topology probe raised: {e!r}")

        # The DRM sysfs rsmi reads for card metadata (secondary; does not cause
        # 0, but records container sysfs completeness).
        try:
            import glob as _glob

            drm = sorted(_glob.glob("/sys/class/drm/*"))
            print(f"[preflight-diag] /sys/class/drm entries: {[os.path.basename(d) for d in drm]}")
        except Exception as e:  # noqa: BLE001
            print(f"[preflight-diag] /sys/class/drm probe raised: {e!r}")

        # rocm-smi / amd-smi CLI cross-check + rsmi env knobs.
        for var in ("ROCM_PATH", "RSMI_LOGGING", "ROCP_METRICS", "HSA_ENABLE_SDMA"):
            print(f"[preflight-diag] env {var}={os.environ.get(var, '<unset>')}")
        for cli in (["amd-smi", "list"], ["rocm-smi", "--showid"]):
            try:
                # Merge stderr into stdout: SMI failures often go only to stderr.
                r = subprocess.run(
                    cli,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=60,
                )
                out = (r.stdout or "").strip().splitlines()
                print(f"[preflight-diag] `{' '.join(cli)}` rc={r.returncode}, first 12 lines:")
                for ln in out[:12]:
                    print(f"[preflight-diag-cli] {ln}")
            except Exception as e:  # noqa: BLE001
                print(f"[preflight-diag] `{' '.join(cli)}` raised: {e!r}")

        # Case (c) discriminator: if Python reads the KFD nodes fine but rsmi
        # still counts 0, the divergence is a syscall the library makes that
        # Python's read path does not (getdents/openat blocked by seccomp for
        # librsmi, an ioctl on /dev/kfd, etc). strace amd-smi to name it.
        try:
            r = subprocess.run(
                ["strace", "-f", "-e", "trace=openat,getdents64,ioctl",
                 "amd-smi", "list"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=90,
            )
            lines = (r.stdout or "").strip().splitlines()
            hits = [
                ln for ln in lines
                if ("kfd" in ln or "dri" in ln or "= -1" in ln or "EPERM" in ln
                    or "EACCES" in ln or "ENOENT" in ln)
            ]
            print(f"[preflight-diag] strace amd-smi: {len(lines)} lines, {len(hits)} relevant:")
            for ln in hits[:40]:
                print(f"[preflight-diag-strace] {ln}")
        except FileNotFoundError:
            print("[preflight-diag] strace not installed (skip case-c syscall probe)")
        except Exception as e:  # noqa: BLE001
            print(f"[preflight-diag] strace probe raised: {e!r}")

        # NOTE on reading the wheel's bundled arches off disk: PyTorch's device
        # code lives in the .hip_fatbin section, which is NOBITS (allocated at
        # runtime, zero bytes on disk), so clang-offload-bundler --list /
        # llvm-objdump --offloading / objcopy all return empty and a bare
        # `strings | grep gfx` only matches metadata tokens (false positives).
        # The authoritative bundled-vs-device ISA comparison is only visible at
        # runtime in the comgr narration -- captured by the CLR child below,
        # which now runs the full failing path (import torch; device_count()).
        # Anchored full-triple grep kept as a best-effort signal only.
        try:
            import os as _os

            _lib = _os.path.join(_os.path.dirname(torch.__file__), "lib", "libtorch_hip.so")
            _r = subprocess.run(
                f"grep -aoE 'amdgcn-amd-amdhsa--gfx[0-9a-z:+_-]+' {_lib} "
                f"| sort | uniq -c | sort -rn | head -30",
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            print(
                f"[preflight-diag] offload triples in libtorch_hip.so "
                f"(best-effort, may be empty):\n{_r.stdout.strip() or '<none found on disk>'}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[preflight-diag] offload-triple grep raised: {e!r}")

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

        # CLR/comgr verbose log of the ACTUAL FAILING PATH. A bare
        # hipGetDeviceCount succeeds (last run: rc=0 count=4) and emits none of
        # the flood -- the 682 fat-binary failures and the comgr ISA-match
        # narration happen during torch's full device init. So the child must
        # run `import torch; device_count()`, the path that actually fails, for
        # the first-failure window below to have anything to bite on.
        try:
            clr = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import torch; print('child device_count', torch.cuda.device_count())",
                ],
                env={
                    **os.environ,
                    "AMD_LOG_LEVEL": "4",
                    # comgr unbundles fat binaries and matches ISAs -- the
                    # component most likely failing 682x. It names ISA
                    # mismatches explicitly when verbose.
                    "AMD_COMGR_EMIT_VERBOSE_LOGS": "1",
                    "AMD_COMGR_REDIRECT_LOGS": "stderr",
                },
                capture_output=True,
                text=True,
                timeout=120,
                # cwd=/tmp so the child's `import torch` does not pick up the
                # torch SOURCE tree in the CI workspace (which shadows the
                # installed wheel -> ModuleNotFoundError: torch.version).
                cwd="/tmp",
            )
            lines = (clr.stderr or "").strip().splitlines()
            print(f"[preflight-diag] raw child stdout: {clr.stdout.strip()!r}")

            # The runtime explains itself in the narration BEFORE the first
            # "register fat binary failed"; the 682-line flood after it is
            # noise. Print the window around the first failure.
            spam = "register fat binary failed"
            idx = next((i for i, ln in enumerate(lines) if spam in ln), None)
            if idx is not None:
                print("[preflight-diag] --- context before FIRST fat-binary failure ---")
                for line in lines[max(0, idx - 50) : idx + 3]:
                    print(f"[preflight-diag-clr] {line}")

            # Deduped flagged pass (excluding the flood string itself) catches
            # anything outside that window.
            flagged = [
                ln
                for ln in lines
                if spam not in ln
                and any(
                    k in ln.lower()
                    for k in (
                        "error",
                        "einval",
                        "no device",
                        "no gpu",
                        "not supported",
                        "unsupported",
                        "mismatch",
                        "minimum",
                        "unable",
                        "abort",
                        "kfd",
                        "ioctl",
                        "queue",
                        "hsakmt",
                        "topology",
                        "isa",
                        "comgr",
                        "unbundle",
                    )
                )
            ]
            print("[preflight-diag] --- CLR flagged lines (flood excluded) ---")
            for line in flagged[-60:]:
                print(f"[preflight-diag-clr] {line}")
            print("[preflight-diag] --- CLR tail (25 lines) ---")
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
