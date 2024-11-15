#!/usr/bin/env python3

from __future__ import annotations
from collections import defaultdict

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
                # cmd = " ".join(process.cmdline())
                # if not cmd.startswith("/opt/conda/envs"):
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
            "ppid":p.ppid(),
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

if __name__ == "__main__":
    print("use monitor_linux.py")
    nvml_exists = False
    try:
        import pynvml  # type: ignore[import]
        pynvml.nvmlInit()
        nvml_exists = True
    except ModuleNotFoundError:
        # no pynvml avaliable, probably because not cuda
        pass

    kill_now = False
    def exit_gracefully(*args: Any) -> None:
        global kill_now
        kill_now = True

    signal.signal(signal.SIGTERM, exit_gracefully)

    while not kill_now:
        # each job does not share gpus, they will run in one ec2 instance, but each job will be assigned to one gpu,
        try:
            stats = {
                "time": datetime.datetime.now(timezone.utc).isoformat("T") + "Z",
                "total_cpu_percent": psutil.cpu_percent(),
                "per_process_cpu_info": get_per_process_cpu_info(),
            }
            if nvml_exists:
                # https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html
                gpu_count = pynvml.nvmlDeviceGetCount()
                # Iterate over the available GPUs
                for i in range(gpu_count):
                    # Get the handle to the current GPU
                    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                    # Get the message for the current GPU
                    stats[f"total_gpu_utilization_{i}"] = gpu_utilization.gpu
                    stats[f"total_gpu_mem_utilization_{i}"] = gpu_utilization.memory
                # Run the nvidia-smi command and capture its output
                output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])# Decode the output from bytes to string
                output_str = output.decode('utf-8')
                # Print the output
                stats["nvidia-smi-test"] = output_str
        except Exception as e:
            stats = {
                "time": datetime.datetime.now(timezone.utc).isoformat("T") + "Z",
                "error": str(e),
            }
        finally:
            print(json.dumps(stats))
            time.sleep(5)
