#!/usr/bin/env python3
"""
A Python script that logging the system-level utilization usage in json format.
Data collected: CPU, memory, GPU memeory utilzation, and GPU utilization if available.

Usage:
    python3 monitor.py --log-interval 10

- To log the data to a file, use the following command:
    python3 monitor.py > usage_log.txt 2>&1
- To gracefully exit the script in the local machine, press ctrl+c, or kill the process using:
    kill <pid>
"""

from __future__ import annotations

import argparse
import datetime
import json
import signal
import time
from typing import Any, Dict

import psutil  # type: ignore[import]


_HAS_PYNVML = False
_HAS_AMDSMI = False

try:
    import pynvml  # type: ignore[import]

    _HAS_PYNVML = True
except ModuleNotFoundError:
    pass

try:
    import amdsmi  # type: ignore[import]

    _HAS_AMDSMI = True
except ModuleNotFoundError:
    pass


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=" System-level Usage Logger ")

    # debug mode used in local to gracefully exit the script when ctrl+c is
    # pressed,and print out the json output in a pretty format.
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="set time interval for logging utilization data, default is 5 seconds",
    )
    args = parser.parse_args()
    return args


class UsageLogger:
    """
    Collect and display usage data, including:
    CPU, memory, GPU memory utilization, and GPU utilization.
    """

    def __init__(
        self,
        log_interval: int = 5,
        is_debug_mode: bool = False,
        pynvml_enabled: bool = False,
        amdsmi_enabled: bool = False,
    ) -> None:
        """
        log_interval: Time interval in seconds for collecting usage data; default is 5 seconds.
        is_debug_mode:
            Useful if you're testing on a local machine and want to see the output
            in a pretty format with more information.
        """
        self._log_interval = log_interval
        self._summary_info = {
            "level": "metadata",
            "interval": self._log_interval,
        }
        self._has_pynvml = pynvml_enabled
        self._has_amdsmi = amdsmi_enabled
        self._kill_now = False
        self._gpu_handles: list[Any] = []
        self._gpu_libs_detected: list[str] = []
        self._num_of_cpus = 0
        self._debug_mode = is_debug_mode
        self._initial_gpu_handler()

    def start(self) -> None:
        """
        runs the main loop of the program.
        the first json record is the metadata of the run,
        including the start time, end time, and the interval of the log.
        """

        self._summary_info["start_time"] = datetime.datetime.now().timestamp()
        self.log_json(self._summary_info)

        # start data collection
        while not self._kill_now:
            collecting_start_time = time.time()
            stats = {}
            try:
                stats.update(
                    {
                        "level": "record",
                        "time": datetime.datetime.now().timestamp(),
                    }
                )
                # collect cpu and memory metrics
                memory = psutil.virtual_memory()
                used_cpu_percent = psutil.cpu_percent()

                stats.update(
                    {
                        "total_cpu_percent": used_cpu_percent,
                        "total_memory_percent": memory.percent,
                        "processes": self._get_process_info(),  # type: ignore[dict-item]
                        "gpu_usage": self._collect_gpu_data(),  # type: ignore[dict-item]
                    }
                )

            except Exception as e:
                stats = {
                    "level": "record",
                    "time": datetime.datetime.now().timestamp(),
                    "error": str(e),
                }
            finally:
                collecting_end_time = time.time()
                time_diff = collecting_end_time - collecting_start_time
                stats["log_duration"] = f"{time_diff * 1000:.2f} ms"

                # output the data to stdout
                self.log_json(stats)
                time.sleep(self._log_interval)

        # shut down gpu connections when exiting
        self._shutdown_gpu_connections()

    def stop(self, *args: Any) -> None:
        """
        Exits the program gracefully. this shuts down the logging loop.
        """
        # TODO: add interruptable timer, that if the script is killed, it will stop the sleep immediatly.
        self._kill_now = True

    def log_json(self, stats: Any) -> None:
        """
        Logs the stats in json format to stdout.
        """
        if self._debug_mode:
            print(json.dumps(stats, indent=4))
            return
        print(json.dumps(stats))

    def _collect_gpu_data(self) -> list[Dict[str, Any]]:
        gpu_data_list = []
        if self._has_pynvml:
            # Iterate over the available GPUs
            for gpu_handle in self._gpu_handles:
                # see https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html
                gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                gpu_uuid = pynvml.nvmlDeviceGetUUID(gpu_handle)
                gpu_processes = self._get_per_process_gpu_info(gpu_handle)
                gpu_data_list.append(
                    {
                        "gpu_uuid": gpu_uuid,
                        "total_gpu_utilization": gpu_utilization.gpu,
                        "total_gpu_mem_utilization": gpu_utilization.memory,
                        "gpu_processes": gpu_processes,
                    }
                )
        elif self._has_amdsmi:
            # Iterate over the available GPUs
            for handle in self._gpu_handles:
                # see https://rocm.docs.amd.com/projects/amdsmi/en/docs-5.7.0/py-interface_readme_link.html
                engine_usage = amdsmi.amdsmi_get_gpu_activity(handle)
                gpu_processes = self._rocm_get_per_process_gpu_info(handle)
                gpu_uuid = amdsmi.amdsmi_get_gpu_device_uuid(handle)
                gpu_utilization = engine_usage["gfx_activity"]
                gpu_mem_utilization = gpu_utilization["umc_activity"]
                gpu_data_list.append(
                    {
                        "gpu_uuid": gpu_uuid,
                        "total_gpu_utilization": gpu_utilization,
                        "total_gpu_mem_utilization": gpu_mem_utilization,
                        "gpu_processes": gpu_processes,
                    }
                )
        return gpu_data_list

    def _initial_gpu_handler(self) -> None:
        """
        Initializes the GPU handlers if gpus are available, and updates the log summary info.
        """
        try:
            if self._has_pynvml:
                self._gpu_libs_detected.append("pynvml")
                # Todo: investigate if we can use device uuid instead of index.
                # there is chance that the gpu index can change when the gpu is rebooted.
                self._gpu_handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                    for i in range(pynvml.nvmlDeviceGetCount())
                ]
            if self._has_amdsmi:
                self._gpu_libs_detected.append("amdsmi")
                self._gpu_handles = amdsmi.amdsmi_get_processor_handles()

            self._num_of_cpus = psutil.cpu_count(logical=False)
            # update summary info
            self._summary_info.update(
                {
                    "gpu_libs_detected": self._gpu_libs_detected,
                    "num_of_gpus": len(self._gpu_handles),
                    "num_of_cpus": self._num_of_cpus,
                }
            )
        except Exception as e:
            self._summary_info["error"] = str(e)

    def _shutdown_gpu_connections(self) -> None:
        if self._has_amdsmi:
            try:
                amdsmi.amdsmi_shut_down()
            except amdsmi.AmdSmiException as e:
                pass
        if self._has_pynvml:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as e:
                pass

    def _get_per_process_gpu_info(self, handle: Any) -> list[dict[str, Any]]:
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        per_process_info = []
        for p in processes:
            mem = p.usedGpuMemory / (1024 * 1024)
            pid = p.pid
            info = {"pid": pid, "gpu_memory": mem}
            per_process_info.append(info)
        return per_process_info

    def _rocm_get_per_process_gpu_info(self, handle: Any) -> list[dict[str, Any]]:
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
                "gpu_memory": proc_info["memory_usage"]["vram_mem"] / (1024 * 1024),
            }
            per_process_info.append(info)
        return per_process_info

    def _get_process_info(self) -> list[dict[str, Any]]:
        def get_processes_running_python_tests() -> list[Any]:
            python_processes = []
            for process in psutil.process_iter():
                try:
                    if "python" in process.name() and process.cmdline():
                        python_processes.append(process)
                except (
                    psutil.ZombieProcess,
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                ):
                    # access denied or the process died
                    pass
            return python_processes

        processes = get_processes_running_python_tests()
        per_process_info = []
        for p in processes:
            try:
                cmdline = p.cmdline()
                info = {
                    "pid": p.pid,
                    "cmd": " ".join(cmdline),
                }
            except (psutil.ZombieProcess, psutil.NoSuchProcess, psutil.AccessDenied):
                continue

            # https://psutil.readthedocs.io/en/latest/index.html?highlight=memory_full_info
            # requires higher user privileges and could throw AccessDenied error, i.e. mac
            try:
                memory_full_info = p.memory_full_info()
                info["uss_memory"] = f"{memory_full_info.uss / (1024 * 1024):.2f}"
                if "pss" in memory_full_info:
                    # only availiable in linux
                    info["pss_memory"] = f"{memory_full_info.pss / (1024 * 1024):.2f}"
            except psutil.AccessDenied as e:
                # It's ok to skip this
                pass
            per_process_info.append(info)
        return per_process_info


def main() -> None:
    """
    Main function of the program.
    """

    # initialize gpu management libraries
    pynvml_enabled = False
    amdsmi_enabled = False

    if _HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            pynvml_enabled = True
        except pynvml.NVMLError:
            pass
    if _HAS_AMDSMI:
        try:
            amdsmi.amdsmi_init()
            amdsmi_enabled = True
        except amdsmi.AmdSmiException:
            pass
    args = parse_args()
    usagelogger = UsageLogger(
        args.log_interval, args.debug, pynvml_enabled, amdsmi_enabled
    )

    # gracefully exit the script when pid is killed
    signal.signal(signal.SIGTERM, usagelogger.stop)
    # gracefully exit the script when keyboard ctrl+c is pressed.
    signal.signal(signal.SIGINT, usagelogger.stop)

    # start the logging
    usagelogger.start()


if __name__ == "__main__":
    main()
