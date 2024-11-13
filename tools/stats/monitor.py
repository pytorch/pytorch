#!/usr/bin/env python3

from __future__ import annotations

import datetime
import json
import signal
import time
import subprocess
from datetime import timezone
from typing import Any

import psutil  # type: ignore[import]


def get_processes_running_python_tests() -> list[Any]:
    python_processes = []
    for process in psutil.process_iter():
        try:
            if "python" in process.name() and process.cmdline():
                python_processes.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # access denied or the process died
            pass
    return python_processes

def get_per_process_cpu_info() -> list[dict[str, Any]]:
    processes = get_processes_running_python_tests()
    per_process_info = []
    for p in processes:
        info = {
            "pid": p.pid,
            "cmd": " ".join(p.cmdline()),
            "cpu_percent": p.cpu_percent(),
            "rss_memory": p.memory_info().rss,
            "ppid":p.ppid.pid,
        }

        # https://psutil.readthedocs.io/en/latest/index.html?highlight=memory_full_info
        # requires higher user privileges and could throw AccessDenied error, i.e. mac
        try:
            memory_full_info = p.memory_full_info()

            info["uss_memory"] = memory_full_info.uss
            if "pss" in memory_full_info:
                # only availiable in linux
                info["pss_memory"] = memory_full_info.pss

        except psutil.AccessDenied as e:
            # It's ok to skip this
            pass
        per_process_info.append(info)
    return per_process_info


def get_per_process_gpu_info(handle: Any) -> list[dict[str, Any]]:
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    per_process_info = []
    for p in processes:
        info = {"pid": p.pid, "gpu_memory": p.usedGpuMemory}
        per_process_info.append(info)
    return per_process_info


def rocm_get_per_process_gpu_info(handle: Any) -> list[dict[str, Any]]:
    processes = amdsmi.amdsmi_get_gpu_process_list(handle)
    per_process_info = []
    for p in processes:
        try:
            proc_info = amdsmi.amdsmi_get_gpu_process_info(handle, p)
        except AttributeError:
            # https://github.com/ROCm/amdsmi/commit/c551c3caedbd903ba828e7fdffa5b56d475a15e7
            # BC-breaking change that removes amdsmi_get_gpu_process_info API from amdsmi
            proc_info = p
        info = {
            "pid": proc_info["pid"],
            "gpu_memory": proc_info["memory_usage"]["vram_mem"],
        }
        per_process_info.append(info)
    return per_process_info


if __name__ == "__main__":
    nvml_handle = None
    amdsmi_handle = None
    try:
        import pynvml  # type: ignore[import]
        try:
            pynvml.nvmlInit()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except pynvml.NVMLError:
            pass
    except ModuleNotFoundError:
        # no pynvml avaliable, probably because not cuda
        pass
    try:
        import amdsmi  # type: ignore[import]
        try:
            amdsmi.amdsmi_init()
            amdsmi_handle = amdsmi.amdsmi_get_processor_handles()[0]
        except amdsmi.AmdSmiException:
            pass
    except ModuleNotFoundError:
        # no amdsmi is available
        pass

    kill_now = False

    def exit_gracefully(*args: Any) -> None:
        global kill_now
        kill_now = True

    signal.signal(signal.SIGTERM, exit_gracefully)

    while not kill_now:
        try:
            stats = {
                "time": datetime.datetime.now(timezone.utc).isoformat("T") + "Z",
                "total_cpu_percent": psutil.cpu_percent(),
                "per_process_cpu_info": get_per_process_cpu_info(),
            }
            if nvml_handle is not None:
                stats["per_process_gpu_info"] = get_per_process_gpu_info(nvml_handle)
                # https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html
                gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
                gpu_count = pynvml.nvmlDeviceGetCount()
                # Iterate over the available GPUs
                for i in range(gpu_count):
                    # Get the handle to the current GPU
                    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    # Get the message for the current GPU
                    # Print the message
                    stats[f"total_gpu_utilization-{i}"] = gpu_utilization.gpu
                    stats[f"total_gpu_mem_utilization-{i}"] = gpu_utilization.memory
                # Run the nvidia-smi command and capture its output
                output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])# Decode the output from bytes to string
                output_str = output.decode('utf-8')
                # Print the output
                stats["nvidia-smi-test"] = output_str
            if amdsmi_handle is not None:
                stats["per_process_gpu_info"] = rocm_get_per_process_gpu_info(
                    amdsmi_handle
                )
                stats["total_gpu_utilization"] = amdsmi.amdsmi_get_gpu_activity(
                    amdsmi_handle
                )["gfx_activity"]
                stats["total_gpu_mem_utilization"] = amdsmi.amdsmi_get_gpu_activity(
                    amdsmi_handle
                )["umc_activity"]
        except Exception as e:
            stats = {
                "time": datetime.datetime.now(timezone.utc).isoformat("T") + "Z",
                "error": str(e),
            }
        finally:
            print(json.dumps(stats))
            time.sleep(1)
