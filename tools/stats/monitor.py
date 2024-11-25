from __future__ import annotations

import argparse
import datetime
import json
import signal
import time
from datetime import timezone
from typing import Any

import psutil  # type: ignore[import]

def get_processes_running_python_tests() -> list[Any]:
    python_processes = []
    for process in psutil.process_iter():
        try:
            if "python" in process.name() and process.cmdline():
                python_processes.append(process)
        except (psutil.ZombieProcess, psutil.NoSuchProcess, psutil.AccessDenied):
            # access denied or the process died
            pass
    return python_processes


def get_process_info() -> list[dict[str, Any]]:
    processes = get_processes_running_python_tests()
    per_process_info = []
    for p in processes:
        try:
            cmdline = p.cmdline()
            info = {
            "pid": p.pid,
            "cmd": " ".join(cmdline),
        }
        except (psutil.ZombieProcess, psutil.NoSuchProcess,psutil.AccessDenied):
            continue
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=" Test Utilization Monitoring")
    args = parser.parse_args()

    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="set time interval for logging utilization data, default is 5 seconds",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    interval = args.log_interval
    has_pynvml = False
    has_amdsmi = False

    try:
        import pynvml  # type: ignore[import]
        try:
            pynvml.nvmlInit()
            has_pynvml = True
        except pynvml.NVMLError:
            pass
    except ModuleNotFoundError:
        pass

    try:
        import amdsmi  # type: ignore[import]
        try:
            amdsmi.amdsmi_init()
            has_amdsmi = True
        except amdsmi.AmdSmiException:
            pass
    except ModuleNotFoundError:
        # no amdsmi is available
        pass

    gpu_handles = []
    kill_now = False
    def exit_gracefully(*args: Any) -> None:
        global kill_now
        kill_now = True
    signal.signal(signal.SIGTERM, exit_gracefully)

    monitor_start_time = datetime.datetime.now(timezone.utc).isoformat("T") + "Z"
    try:
        gpu_libs_detected = []
        if has_pynvml:
            gpu_libs_detected.append("pynvml")
            num_of_gpu = pynvml.nvmlDeviceGetCount()
            gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        if has_amdsmi:
            gpu_libs_detected.append("amdsmi")
            gpu_handles  = amdsmi.amdsmi_get_processor_handles()
        num_cpus = psutil.cpu_count()

        # log info
        info = {
            "type": "metadata",
            "log_interval": f"{interval} seconds",
            "gpu":  gpu_libs_detected,
            "num_of_gpus":len(gpu_handles),
            "num_of_cpus": psutil.cpu_count(logical=False),
        }

        print(json.dumps(info))
    except Exception as e:
        info = {
            "error": str(e)
        }
        print(json.dumps(info))

    while not kill_now:
        try:
            memory_info = psutil.virtual_memory()
            stats = {
                "type": "log",
                "time": datetime.datetime.now(timezone.utc).isoformat("T") + "Z",
                "total_cpu_percent": psutil.cpu_percent(),
                "total_memory_percent":memory_info.percent,
                "processes": get_process_info(),
            }
            if has_pynvml:
                # Iterate over the available GPUs
                for idx,gpu_handle in enumerate(gpu_handles):
                    gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                    stats[f"total_gpu_utilization_{idx}"] = gpu_utilization.gpu
                    stats[f"total_gpu_mem_utilization_{idx}"] = gpu_utilization.memory
            if has_amdsmi:
                for idx,handle in enumerate(gpu_handles):
                    stats[f"total_gpu_utilization_{idx}"] = amdsmi.amdsmi_get_gpu_activity(handle)["gfx_activity"]
                    stats[f"total_gpu_mem_utilization_{idx}"] = amdsmi.amdsmi_get_gpu_activity(handle)["umc_activity"]

        except Exception as e:
            stats = {
                "time": datetime.datetime.now(timezone.utc).isoformat("T") + "Z",
                "error": str(e),
            }
        finally:
            print(json.dumps(stats))
            time.sleep(interval)
    info = {
        "type": "metadata",
        "monitor_start_time": monitor_start_time,
        "monitor_end_time": datetime.datetime.now(timezone.utc).isoformat("T") + "Z",
    }
    print(json.dumps(info))
    # close the connection
    if has_amdsmi:
        try:
            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiException as e:
            pass
    if has_pynvml:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as e:
            pass
