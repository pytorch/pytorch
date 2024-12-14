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
import dataclasses
import datetime
import json
import signal
import threading
import time
from collections import defaultdict
from typing import Any

import psutil  # type: ignore[import]


_HAS_PYNVML = False
_HAS_AMDSMI = False


@dataclasses.dataclass
class UsageData:
    """
    Dataclass for storing usage data.
    """

    cpu_percent: float
    memory_percent: float
    processes: list[dict[str, Any]]
    gpu_list: list[GpuData]


@dataclasses.dataclass
class GpuData:
    """
    Dataclass for storing gpu data.
    """

    uuid: str
    utilization: float
    mem_utilization: float


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
        type=float,
        default=5,
        help="set time interval for logging utilization data, default is 5 seconds",
    )
    parser.add_argument(
        "--data-collect-interval",
        type=float,
        default=0.5,
        help="set time interval to collect data, default is 0.5 second, this should not longer than log_interval",
    )
    args = parser.parse_args()
    return args


class UsageLogger:
    """
    Collect and display usage data, including:
    CPU, memory, GPU memory utilization, and GPU utilization.
    By default, data is collected every 0.5 seconds, and output
    the aggregated json log every 5 seconds.
    """

    def __init__(
        self,
        log_interval: float = 5,
        data_collect_interval: float = 0.5,
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
        self._data_collect_interval = data_collect_interval
        self._has_pynvml = pynvml_enabled
        self._has_amdsmi = amdsmi_enabled
        self._gpu_handles: list[Any] = []
        self._gpu_libs_detected: list[str] = []
        self._num_of_cpus = 0
        self._debug_mode = is_debug_mode
        self._initial_gpu_handler()

        self.exit_event = threading.Event()

        self.data_list: list[UsageData] = []
        self.data_errors: list[str] = []
        self.lock = threading.Lock()

    def _collect_data(self) -> None:
        """
        Collects the data every data_collect_interval (in seconds).
        """
        while not self.exit_event.is_set():
            try:
                # collect cpu, memory and gpu metrics
                memory = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent()
                processes = self._get_process_info()
                gpuList = self._collect_gpu_data()

                data = UsageData(
                    cpu_percent=cpu_percent,
                    memory_percent=memory,
                    processes=processes,
                    gpu_list=gpuList,
                )
                if self._debug_mode:
                    print(f"collecting data {data}")
                self._add_data(data)
            except Exception as e:
                self._add_error(e)
            finally:
                time.sleep(self._data_collect_interval)

    def _add_data(self, data: UsageData) -> None:
        with self.lock:
            self.data_list.append(data)

    def _add_error(self, error: Exception) -> None:
        with self.lock:
            self.data_errors.append(str(error))

    def _output_data(self) -> None:
        """
        output the data.
        """
        self._summary_info["start_time"] = datetime.datetime.now().timestamp()
        self.log_json(self._summary_info)

        while not self.exit_event.is_set():
            collecting_start_time = time.time()
            stats = {}
            try:
                with self.lock:
                    if self._debug_mode:
                        print("collected:", len(self.data_list))

                    # if no data is collected and has more than one error during collect interval, log the errors
                    if not self.data_list and len(self.data_errors) > 1:
                        errors = ",".join(set(self.data_errors))
                        self.data_errors.clear()
                        raise ValueError(
                            f"no data is collected but detected multiple errors: [{errors}]"
                        )

                    if not self.data_list:
                        continue
                    # record timestamp
                    stats.update(
                        {
                            "level": "record",
                            "time": datetime.datetime.now().timestamp(),
                        }
                    )
                    # collect cpu and memory metrics
                    total_cpu = sum(
                        usageData.cpu_percent for usageData in self.data_list
                    )
                    avg_cpu = total_cpu / len(self.data_list)
                    max_cpu = max(usageData.cpu_percent for usageData in self.data_list)

                    max_memory = max(
                        usageData.memory_percent for usageData in self.data_list
                    )
                    total_memory = sum(
                        usageData.memory_percent for usageData in self.data_list
                    )
                    avg_memory = total_memory / len(self.data_list)

                    # find all cmds during the interval
                    cmds = list(
                        set(
                            process["cmd"]
                            for data in self.data_list
                            for process in data.processes
                        )
                    )

                    stats.update(
                        {
                            "cpu": {
                                "avg": round(avg_cpu, 2),
                                "max": round(max_cpu, 2),
                            },
                            "memory": {
                                "avg": round(avg_memory, 2),
                                "max": round(max_memory, 2),
                            },
                            "cmds": cmds,
                            "count": len(self.data_list),
                        }
                    )

                    # collect gpu metrics
                    if self._has_pynvml or self._has_amdsmi:
                        gpu_list = self._calculate_gpu_utilization(self.data_list)
                        stats.update(
                            {
                                "gpu_list": gpu_list,
                            }
                        )
                    self.data_list.clear()
            except Exception as e:
                stats = {
                    "level": "record",
                    "time": datetime.datetime.now().timestamp(),
                    "error": str(e),
                }
            finally:
                collecting_end_time = time.time()
                time_diff = collecting_end_time - collecting_start_time
                # verify there is data
                if stats:
                    stats["log_duration"] = f"{time_diff * 1000:.2f} ms"
                    self.log_json(stats)
                time.sleep(self._log_interval)
        # shut down gpu connections when exiting
        self._shutdown_gpu_connections()

    def _calculate_gpu_utilization(self, data_list) -> list[dict[str, Any]]:
        """
        Calculates the GPU utilization.
        """
        calculate_gpu = []
        gpu_mem_utilization = defaultdict(list)
        gpu_utilization = defaultdict(list)

        for data in data_list:
            for gpu in data.gpu_list:
                gpu_mem_utilization[gpu.uuid].append(gpu.mem_utilization)
                gpu_utilization[gpu.uuid].append(gpu.utilization)

        for gpu_uuid in gpu_utilization.keys():
            max_gpu_utilization = max(gpu_utilization[gpu_uuid])
            max_gpu_mem_utilization = max(gpu_mem_utilization[gpu_uuid])

            total_gpu = sum(gpu_utilization[gpu_uuid])
            avg_gpu_utilization = (
                total_gpu / len(gpu_utilization[gpu_uuid])
                if len(gpu_utilization[gpu_uuid]) > 0
                else 0
            )

            total_mem = sum(gpu_mem_utilization[gpu_uuid])
            avg_gpu_mem_utilization = (
                total_mem / len(gpu_mem_utilization[gpu_uuid])
                if len(gpu_mem_utilization[gpu_uuid]) > 0
                else 0
            )

            calculate_gpu.append(
                {
                    "uuid": gpu_uuid,
                    "util_percent": {
                        "count": len(gpu_utilization[gpu_uuid]),
                        "avg": round(avg_gpu_utilization, 2),
                        "max": round(max_gpu_utilization, 2),
                    },
                    "mem_util_percent": {
                        "count": len(gpu_mem_utilization[gpu_uuid]),
                        "avg": round(avg_gpu_mem_utilization, 2),
                        "max": round(max_gpu_mem_utilization, 2),
                    },
                }
            )
        return calculate_gpu

    def start(self) -> None:
        collect_thread = threading.Thread(target=self._collect_data)
        collect_thread.start()
        self._output_data()
        collect_thread.join()

    def stop(self, *args: Any) -> None:
        """
        Exits the program gracefully. this shuts down the logging loop.
        """
        self.exit_event.set()

    def log_json(self, stats: Any) -> None:
        """
        Logs the stats in json format to stdout.
        """
        if self._debug_mode:
            print(json.dumps(stats, indent=4))
            return
        print(json.dumps(stats))

    def _collect_gpu_data(self) -> list[GpuData]:
        gpu_data_list = []
        if self._has_pynvml:
            # Iterate over the available GPUs
            for gpu_handle in self._gpu_handles:
                # see https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html
                gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                gpu_uuid = pynvml.nvmlDeviceGetUUID(gpu_handle)
                gpu_data_list.append(
                    GpuData(
                        uuid=gpu_uuid,
                        utilization=gpu_utilization.gpu,
                        mem_utilization=gpu_utilization.memory,
                    )
                )
        elif self._has_amdsmi:
            # Iterate over the available GPUs
            for handle in self._gpu_handles:
                # see https://rocm.docs.amd.com/projects/amdsmi/en/docs-5.7.0/py-interface_readme_link.html
                engine_usage = amdsmi.amdsmi_get_gpu_activity(handle)
                gpu_uuid = amdsmi.amdsmi_get_gpu_device_uuid(handle)
                gpu_utilization = engine_usage["gfx_activity"]
                gpu_mem_utilization = gpu_utilization["umc_activity"]
                gpu_data_list.append(
                    GpuData(
                        uuid=gpu_uuid,
                        utilization=gpu_utilization,
                        mem_utilization=gpu_mem_utilization,
                    )
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
            except amdsmi.AmdSmiException:
                pass
        if self._has_pynvml:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
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
            python_test_processes = []
            for process in psutil.process_iter():
                try:
                    cmd = " ".join(process.cmdline())
                    processName = process.name()
                    pid = process.pid
                    if "python" in processName and cmd.startswith("python"):
                        python_test_processes.append({"pid": pid, "cmd": cmd})
                except Exception as e:
                    pass
            return python_test_processes

        processes = get_processes_running_python_tests()
        return processes


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
        log_interval=args.log_interval,
        data_collect_interval=args.data_collect_interval,
        is_debug_mode=args.debug,
        pynvml_enabled=pynvml_enabled,
        amdsmi_enabled=amdsmi_enabled,
    )

    # gracefully exit the script when pid is killed
    signal.signal(signal.SIGTERM, usagelogger.stop)
    # gracefully exit the script when keyboard ctrl+c is pressed.
    signal.signal(signal.SIGINT, usagelogger.stop)

    # start the logging
    usagelogger.start()


if __name__ == "__main__":
    main()
