"""
A Python script that monitors the usage of a system's resources, including CPU, memory, and GPU utilization, and logs the data to a JSON file.
"""

from __future__ import annotations

import argparse
import datetime
import json
import signal
import time
from datetime import timezone
from typing import Any

import psutil  # type: ignore[import]


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=" Git Job Test Usage Monitoring")

    # debug mode used in local to gracefully exit the script when ctrl+c is pressed.
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="set time interval for logging utilization data, default is 5 seconds",
    )
    args = parser.parse_args()
    return args


class LogRecord:
    """
    A class to manage log data as record.
    """
    def __init__(self) -> None:
        self._summary_info = {}

    def upsert(self, key, value) -> None:
        self._summary_info[key] = value

    def upsert_pairs(self, pairs) -> None:
        for key, value in pairs.items():
            self._summary_info[key] = value

    def get(self) -> dict[str, Any]:
        return self._summary_info


class UsageLog:
    def __init__(self, log_interval, is_debug_mode=False) -> None:
        """
        Initializes a new instance of the UsageLog class.

        Args:
            log_interval (int): The time interval for logging utilization data in seconds.
        """
        self._log_interval = log_interval
        self._summary_info = LogRecord()
        self._summary_info.upsert_pairs(
            {
                "level": "metadata",
                "interval": self._log_interval,
            }
        )
        self._has_pynvml = False
        self._has_amdsmi = False
        self._kill_now = False
        self._gpu_handles = []
        self._gpu_libs_detected = []
        self._num_of_cpus = 0
        self._debug_mode = is_debug_mode

        # initialize gpu connections
        try:
            import pynvml  # type: ignore[import]

            try:
                pynvml.nvmlInit()
                self._has_pynvml = True
            except pynvml.NVMLError:
                pass
        except ModuleNotFoundError:
            pass
        try:
            import amdsmi  # type: ignore[import]

            try:
                amdsmi.amdsmi_init()
                self._has_amdsmi = True
            except amdsmi.AmdSmiException:
                pass
        except ModuleNotFoundError:
            # no amdsmi is available
            pass
        self._initialGpuHanlders()

    def log_json(self, stats) -> None:
        """
        Logs the given statistics as JSON.

        Args:
            stats (dict): A dictionary containing the statistics to be logged.
        """
        if self._debug_mode:
            # pretty print the json for debug mode
            print(json.dumps(stats, indent=4))
        else:
            print(json.dumps(stats))

    def exit_gracefully(self, *args: Any) -> None:
        """
        Exits the program gracefully. this shuts down the logging loops in execute()
        """
        self._kill_now = True

    def execute(self) -> None:
        """
        Executes the main loop of the program.
        """
        # logs start_time for execution
        self._summary_info.upsert(
            "start_time", datetime.datetime.now(timezone.utc).isoformat("T") + "Z"
        )

        # prints log summary info before execution
        self.log_json(self._summary_info.get())

        # execute the main loop
        while not self._kill_now:
            collecting_start_time = time.time()
            stats = {}
            try:
                valid_record = LogRecord()
                valid_record.upsert_pairs(
                    {
                        "level": "record",
                        "time": datetime.datetime.now(timezone.utc).isoformat("T")
                        + "Z",
                        "total_cpu_percent": psutil.cpu_percent(),
                        "total_memory_percent": psutil.virtual_memory(),
                        "processes": self._get_process_info(),
                    }
                )
                if self._has_pynvml:
                    # Iterate over the available GPUs
                    for idx, gpu_handle in enumerate(self._gpu_handles):
                        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(
                            gpu_handle
                        )
                        valid_record.upsert_pairs(
                            {
                                f"total_gpu_utilization_{idx}": gpu_utilization.gpu,
                                f"total_gpu_mem_utilization_{idx}": gpu_utilization.memory,
                            }
                        )

                if self._has_amdsmi:
                    for idx, handle in enumerate(self._gpu_handles):
                        valid_record.upsert_pairs(
                            {
                                f"total_gpu_utilization_{idx}": amdsmi.amdsmi_get_gpu_activity(
                                    handle
                                )[
                                    "gfx_activity"
                                ],
                                f"total_gpu_mem_utilization_{idx}": amdsmi.amdsmi_get_gpu_activity(
                                    handle
                                )[
                                    "umc_activity"
                                ],
                            }
                        )
                stats = valid_record.get()
            except Exception as e:
                error_record = {
                    "level": "record",
                    "time": datetime.datetime.now(timezone.utc).isoformat("T") + "Z",
                    "error": str(e),
                }
                stats = error_record
            finally:
                collecting_end_time = time.time()
                time_diff = collecting_end_time - collecting_start_time
                stats["collecting_time_interval"] = f"{time_diff*1000:.2f}ms"
                self.log_json(stats)
                # sleep for the remaining time to meet the log interval.
                if time_diff < self._log_interval:
                    time.sleep(self._log_interval - time_diff)

        # prints complete log summary info to terminal
        self._summary_info.upsert(
            "end_time", datetime.datetime.now(timezone.utc).isoformat("T") + "Z"
        )
        self.log_json(self._summary_info.get())
        # shut down gpu connections
        self._shutdown_gpu_connections()

    def _initialGpuHanlders(self) -> None:
        """
        Initializes the GPU handlers if available.
        """

        try:
            if self._has_pynvml:
                self._gpu_libs_detected.append("pynvml")
                self._gpu_handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                    for i in range(pynvml.nvmlDeviceGetCount())
                ]
            if self._has_amdsmi:
                self._gpu_libs_detected.append("amdsmi")
                self._gpu_handles = amdsmi.amdsmi_get_processor_handles()
            self._num_of_cpus = psutil.cpu_count(logical=False)

            # log summary info for handlers
            self._summary_info.upsert("gpu_libs_detected", self._gpu_libs_detected)
            self._summary_info.upsert("num_of_gpus", len(self._gpu_handles))
            self._summary_info.upsert("num_of_cpus", self._num_of_cpus)
        except Exception as e:
            self._summary_info.upsert("error", str(e))

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
            info = {"pid": p.pid, "gpu_memory": p.usedGpuMemory}
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
                "gpu_memory": proc_info["memory_usage"]["vram_mem"],
            }
            per_process_info.append(info)
        return per_process_info

    def _get_processes_running_python_tests(self) -> list[Any]:
        python_processes = []
        for process in psutil.process_iter():
            try:
                if "python" in process.name() and process.cmdline():
                    python_processes.append(process)
            except (psutil.ZombieProcess, psutil.NoSuchProcess, psutil.AccessDenied):
                # access denied or the process died
                pass
        return python_processes

    def _get_process_info(self) -> list[dict[str, Any]]:
        processes = self._get_processes_running_python_tests()
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
                info["uss_memory"] = memory_full_info.uss
                if "pss" in memory_full_info:
                    # only availiable in linux
                    info["pss_memory"] = memory_full_info.pss
            except psutil.AccessDenied as e:
                # It's ok to skip this
                pass
            per_process_info.append(info)
        return per_process_info


def main():
    """
    Main function of the program.
    """
    try:
        args = parse_args()
        usagelog = UsageLog(args.log_interval, args.debug)
        # gracefully exit the script when pid is killed
        signal.signal(signal.SIGTERM, usagelog.exit_gracefully)
        # gracefully exit the script when keyboard ctrl+c is pressed.
        signal.signal(signal.SIGINT, usagelog.exit_gracefully)
        usagelog.execute()
    except Exception as e:
        json.dumps(
            {"level": "main", "error": f"Failed to execute the usage log: {str(e)}"}
        )


if __name__ == "__main__":
    main()
