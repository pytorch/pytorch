"""Print one machine-parseable line of raw runtime environment facts.

Output: a single line `RUNTIME_PROBE_JSON: {...}`. Values are raw; a field
that fails to collect becomes {"error": "..."}.
Never raises, never blocks for long (subprocess timeouts), stdlib + torch only.
"""

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig


ENV_VARS = [
    "BUILD_ENVIRONMENT",
    "TEST_CONFIG",
    "SHARD_NUMBER",
    "NUM_TEST_SHARDS",
    "JOB_ID",
    "JOB_NAME",
    "GITHUB_REPOSITORY",
    "GITHUB_WORKFLOW",
    "GITHUB_JOB",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
    "RUNNER_NAME",
    "RUNNER_OS",
    "RUNNER_ARCH",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "ZE_AFFINITY_MASK",
    "ATEN_CPU_CAPABILITY",
    "PYTHON_GIL",
    "LD_PRELOAD",
    "TPU_ACCELERATOR_TYPE",
]
PACKAGES = re.compile(
    r"^(numpy|scipy|numba|pandas|einops|dill|opt[-_]einsum|optree|sympy|networkx"
    r"|onnx|onnxruntime|triton|pytorch[-_]triton.*|expecttest|hypothesis"
    r"|torch|torch(vision|audio|ao|comms)|torch[-_](tpu|xla))$"
)
DMI = ["sys_vendor", "product_name", "board_name"]


def run(cmd, timeout=30):
    if not shutil.which(cmd[0]):
        return None
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.stdout.strip() if p.returncode == 0 else None


def read(path):
    with open(path) as f:
        return f.read().strip()


def safe(fn):
    try:
        return fn()
    except BaseException as e:
        return {"error": f"{type(e).__name__}: {e}"}


def text(fn):
    r = safe(fn)
    return r if isinstance(r, str) else ""


def os_info():
    d = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }
    if sys.platform.startswith("linux"):
        rel = dict(
            re.findall(
                r'^(\w+)="?(.*?)"?$',
                text(lambda: read("/etc/os-release")),
                re.MULTILINE,
            )
        )
        d["os_release"] = {
            k: rel.get(k) for k in ("ID", "VERSION_ID", "VERSION_CODENAME")
        }
        d["libc"] = "-".join(platform.libc_ver())
    elif sys.platform == "darwin":
        d["mac_ver"] = platform.mac_ver()[0]
        d["sw_vers"] = safe(lambda: run(["sw_vers", "-productVersion"]))
    elif sys.platform == "win32":
        d["win32_ver"] = list(platform.win32_ver())
        d["win32_edition"] = safe(platform.win32_edition)
    return d


def python_info():
    return {
        "version": list(sys.version_info[:3]),
        "implementation": sys.implementation.name,
        "abiflags": getattr(sys, "abiflags", None),
        "py_gil_disabled": sysconfig.get_config_var("Py_GIL_DISABLED"),
        "gil_enabled": safe(lambda: sys._is_gil_enabled())
        if hasattr(sys, "_is_gil_enabled")
        else None,
        "executable": sys.executable,
    }


def host_cpu():
    d = {}
    if sys.platform.startswith("linux"):
        out = text(lambda: run(["lscpu"]))
        d["lscpu"] = {
            k: v.strip()
            for k, v in re.findall(
                r"^\s*(Model name|Vendor ID|Machine type|CPU\(s\)):(.*)$",
                out,
                re.MULTILINE,
            )
        }
        info = text(lambda: read("/proc/cpuinfo"))
        d["cpuinfo"] = dict(
            re.findall(
                r"^(model name|CPU implementer|CPU part|machine)\s*:\s*(.*)$",
                info,
                re.MULTILINE,
            )
        )
        d["dmi"] = {
            k: safe(lambda k=k: read("/sys/devices/virtual/dmi/id/" + k)) for k in DMI
        }
        d["mem_total_kb"] = safe(
            lambda: int(re.search(r"MemTotal:\s+(\d+)", read("/proc/meminfo")).group(1))
        )
        d["cgroup_mem_max"] = safe(lambda: read("/sys/fs/cgroup/memory.max"))
        d["cgroup_cpu_max"] = safe(lambda: read("/sys/fs/cgroup/cpu.max"))
        d["affinity_cpus"] = safe(lambda: len(os.sched_getaffinity(0)))
    elif sys.platform == "darwin":
        keys = ["machdep.cpu.brand_string", "hw.model", "hw.memsize", "hw.physicalcpu"]
        d["sysctl"] = safe(
            lambda: dict(zip(keys, (run(["sysctl", "-n"] + keys) or "").splitlines()))
        )
    elif sys.platform == "win32":
        import winreg

        key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        d["cpu_name"] = safe(
            lambda: winreg.QueryValueEx(
                winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key), "ProcessorNameString"
            )[0].strip()
        )
    d["os_cpu_count"] = os.cpu_count()
    return d


def torch_info():
    import torch

    v = torch.version
    d = {
        "version": torch.__version__,
        "git": getattr(v, "git_version", None),
        "cuda": v.cuda,
        "hip": v.hip,
        "rocm": getattr(v, "rocm", None),
        "xpu": getattr(v, "xpu", None),
        "debug": v.debug,
        "mps_built": safe(torch.backends.mps.is_built),
    }
    cfg = torch.__config__.show().splitlines()
    d["config"] = [
        l.strip("- ").strip()
        for l in cfg[1:]
        if l.strip() and "Build settings" not in l
    ]
    bs = next((l.split(":", 1)[1] for l in cfg if "Build settings:" in l), "")
    d["build_settings"] = {
        k: val
        for k, val in re.findall(r"(\w+)=(.*?), (?=\w+=|$)", bs.strip() + " ")
        if "FLAGS" not in k
    }
    d["cpu_capability"] = safe(torch.backends.cpu.get_cpu_capability)
    d["cpu_caps"] = safe(
        lambda: {k: val for k, val in torch.cpu.get_capabilities().items() if val}
    )
    d["accelerator"] = safe(lambda: accel_info(torch))
    if sys.platform.startswith("linux"):
        maps = text(lambda: read("/proc/self/maps"))
        d["sanitizer_libs"] = sorted(
            set(re.findall(r"lib\S*(?:asan|tsan|ubsan)\S*\.so", maps))
        )
    return d


def accel_info(torch):
    if not torch.accelerator.is_available():
        return {"type": None, "count": 0}
    acc = torch.accelerator.current_accelerator().type
    d = {"type": acc, "count": torch.accelerator.device_count(), "devices": []}
    mod = getattr(torch, acc, None)
    for i in range(d["count"]):
        p = (
            safe(lambda i=i: mod.get_device_properties(i))
            if hasattr(mod, "get_device_properties")
            else None
        )
        dev = dict(p) if isinstance(p, dict) else {}
        for a in (
            "name",
            "total_memory",
            "gcnArchName",
            "major",
            "minor",
            "multi_processor_count",
            "driver_version",
        ):
            if hasattr(p, a):
                dev[a] = getattr(p, a)
        if acc == "mps":
            dev["recommended_max_memory"] = safe(torch.mps.recommended_max_memory)
        d["devices"].append(dev)
    return d


def driver_info():
    d = {}
    if os.path.exists("/proc/driver/nvidia/version"):
        d["nvidia"] = safe(lambda: read("/proc/driver/nvidia/version").splitlines()[0])
    elif shutil.which("nvidia-smi"):
        d["nvidia"] = safe(
            lambda: run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            )
        )
    if os.path.exists("/sys/module/amdgpu/version"):
        d["amdgpu"] = safe(lambda: read("/sys/module/amdgpu/version"))
    return d


def packages():
    from importlib import metadata

    return {
        n: dist.version
        for dist in metadata.distributions()
        if (n := (dist.metadata["Name"] or "").lower()) and PACKAGES.match(n)
    }


def main():
    probe = {
        "schema": 1,
        "os": safe(os_info),
        "python": safe(python_info),
        "host_cpu": safe(host_cpu),
        "torch": safe(torch_info),
        "driver": safe(driver_info),
        "packages": safe(packages),
        "env": {k: os.environ.get(k) for k in ENV_VARS if k in os.environ},
    }
    print(
        "RUNTIME_PROBE_JSON: "
        + json.dumps(probe, sort_keys=True, default=str, separators=(",", ":")),
        flush=True,
    )


if __name__ == "__main__":
    err = safe(main)
    if err:
        print(
            "RUNTIME_PROBE_JSON: " + json.dumps({"schema": 1, "fatal": err}), flush=True
        )
