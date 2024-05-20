import functools
import logging
import os
import re
import subprocess
import time
from threading import Lock
from typing import Optional

logger = logging.getLogger("strobelight_cli_function_profiler")

console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(name)s, line %(lineno)d, %(asctime)s, %(levelname)s: %(message)s"
)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


class StrobelightCLIProfilerError(Exception):
    "Raised when an error happens during strobelight profiling"


# A function decorator that wraps profile, if no profiler is provided one with
# default args is created. A function can be annotated as:
# @strobelight()
# @strobelight(profiler = StrobelightFunctionProfiler(stop_at_error=True,..))
# @strobelight(stop_at_error=True,...)
def strobelight(profiler=None, **kwargs):
    if not profiler:
        profiler = StrobelightCLIFunctionProfiler(**kwargs)

    def strobelight_inner(work_function):
        @functools.wraps(work_function)
        def wrapper_function(*args, **kwargs):
            return profiler.profile(work_function, *args, **kwargs)

        return wrapper_function

    return strobelight_inner


def pid_namespace_link(pid: Optional[int] = None) -> str:
    PID_NAMESPACE_PATH = "/proc/{}/ns/pid"
    """Returns the link to the process's namespace, example: pid:[4026531836]"""
    pid = pid or os.getpid()
    return os.readlink(PID_NAMESPACE_PATH.format(pid))


def pid_namespace(pid: Optional[int] = None) -> int:
    """Returns the process's namespace id"""
    pid = pid or os.getpid()
    link = pid_namespace_link(pid)
    return int(link[link.find("[") + 1 : -1])


def command_to_string(command):
    s = ""
    for item in command:
        s += " "
        s += item
    return s


class StrobelightCLIFunctionProfiler:
    """
    Note: this is a meta only tool.

    StrobelightCLIFunctionProfiler can be used to profile a python function and
    generate a strobelight link with the results. It works on meta servers but
    does not requries an fbcode target.
    When stop_at_error is false(default), error during profiling does not prevent
    the work function from running.

    Check function_profiler_example.py for an example.
    """

    # This lock is used to make sure only one thread is running the profiler at any point.
    lock = Lock()

    def __init__(
        self,
        *,
        stop_at_error=False,
        max_profile_duration_sec=60 * 10,
        sample_each=1e7,  # sample each sample_each cycles.
        run_user_name="pytorch-strobelight-ondemand",
        timeout_wait_for_running_sec=60,
        timeout_wait_for_finished_sec=60,
        recorded_env_variables=None,
        sample_tags=None,
        stack_max_len=127,
        async_stack_max_len=127,
    ):
        self.stop_at_error = stop_at_error
        self.max_profile_duration_sec = max_profile_duration_sec
        self.sample_each = sample_each
        self.run_user_name = run_user_name
        self.timeout_wait_for_running_sec = timeout_wait_for_running_sec
        self.timeout_wait_for_finished_sec = timeout_wait_for_finished_sec
        # Results of the most recent run.
        # Tracks the strobelight run id of the most recent run
        self.current_run_id = None
        self.sample_tags = sample_tags

    def _run_async(self):
        processId = os.getpid()
        namespace = pid_namespace(processId)
        command = [
            "strobeclient",
            "run",
            "--profiler",
            "pyperf",
            "--event",
            "cycles",
            "--async",
            "--sample-interval",
            str(int(self.sample_each)),
            "--duration-ms",
            str(int(self.max_profile_duration_sec * 1000)),
            "--pid",
            f"{namespace}:{processId}",
        ]

        if self.sample_tags:
            command.append("--sample-tags")
            add_comma = False
            for item in self.sample_tags:
                if add_comma:
                    command.append(",")
                command.append(item)
                add_comma = True

        logger.debug(f"running command:{command_to_string(command)}")

        result = subprocess.run(command, capture_output=True)

        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                "failed to start strobelight profiling, error in run_async"
            )

        for line in result.stderr.split(b"\n"):
            logger.debug(line)
            match = re.search(rb"INFO Run Id:.*", line)
            if match:
                self.current_run_id = int(
                    re.search(rb"-?\d+$", match.group(0)).group(0)
                )

    def _wait_for_running(self, counter=0):
        if counter > 20:
            raise StrobelightCLIProfilerError(
                "wait_for_running called more than 20 times"
            )

        command = ["strobeclient", "getRunStatus", "--run-id", str(self.current_run_id)]

        logger.debug(f"running command:{command_to_string(command)}")

        result = subprocess.run(command, capture_output=True)

        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                "failed to start strobelight profiling, error in wait_for_running"
            )

        for line in result.stderr.split(b"\n"):
            logger.debug(line)
            match = re.search(rb"Profile run status: (.*)", line)
            if match:
                current_status = match.group(1)
                if current_status == b"RUNNING":
                    return
                elif current_status == b"PREPARING":
                    time.sleep(10)
                    self._wait_for_running(counter + 1)
                    return
                else:
                    raise StrobelightCLIProfilerError(
                        f"unexpected {current_status} phase"
                    )

        raise StrobelightCLIProfilerError("unreachable")

    def _stop_run(self):
        command = ["strobeclient", "stopRun", "--run-id", str(self.current_run_id)]

        logger.debug(f"running command:{command_to_string(command)}")

        result = subprocess.run(command, capture_output=True)

        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                "failed to stop strobelight profiling, return code is not 0"
            )

        for line in result.stderr.split(b"\n"):
            logger.debug(line)
            match = re.search(rb"INFO ::1:(.*)", line)
            if match:
                current_status = match.group(1)
                if current_status.__contains__(b"Success!"):
                    return
                elif current_status.__contains__(b"Failed!"):
                    raise StrobelightCLIProfilerError(
                        "failed to stop strobelight profiling, got Failed!"
                    )
                else:
                    raise StrobelightCLIProfilerError(
                        "failed to stop strobelight profiling, unexpected response"
                    )

        raise StrobelightCLIProfilerError("unreachable")

    def _get_results(self):
        command = ["strobeclient", "getRunStatus", "--run-id", str(self.current_run_id)]

        logger.debug(f"running command:{command_to_string(command)}")

        result = subprocess.run(command, capture_output=True)

        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                "failed to extract profiling results, return code is not 0"
            )

        for line in result.stderr.split(b"\n"):
            match = re.search(
                rb"(Total samples(.*)$|GraphProfiler(.*)$|Icicle view \(python stack\)(.*)$)",
                line,
            )
            if match:
                logger.info(match.group(1).decode("utf-8"))
            else:
                logger.debug(line)
            match = re.search(rb"INFO ::1:(.*)", line)
            if match:
                current_status = match.group(1)
                if current_status.__contains__(b"Profile run finished with SUCCESS"):
                    continue
                else:
                    raise StrobelightCLIProfilerError(
                        "failed to extract profiling results, unexpected response"
                    )

    def _stop_strobelight_no_throw(
        self,
        collect_results,
    ):
        try:
            # call stop run
            result = self._stop_run()
            logger.info("strobelight profiling stopped")

            logger.debug("collection stopped")

            if not collect_results:
                return

            self._get_results()
        except Exception as error:
            logger.warning("error during stop_strobelight %s", error)

    # Return true if strobelight started and is running.
    def _start_strobelight(self):
        strobelight_started = False
        try:
            self._run_async()
            strobelight_started = True
            logger.info("strobelight run id is: %s", self.current_run_id)
            self._wait_for_running()
            logger.info("strobelight profiling running")
            return True

        except Exception as error:
            logger.warning("error during start_strobelight: %s", error)
            if strobelight_started:
                self._stop_strobelight_no_throw(collect_results=False)
            return False

    def profile(self, work_function, *args, **kwargs):
        self.current_run_id = None

        if StrobelightCLIFunctionProfiler.lock.locked():
            if self.stop_at_error:
                raise StrobelightProfileError("simultaneous runs not supported")

            return work_function(*args, **kwargs)

        with StrobelightCLIFunctionProfiler.lock:
            started = self._start_strobelight()
            if not started:
                if self.stop_at_error:
                    raise StrobelightCLIProfilerError(
                        "failed to start strobelight profiling"
                    )
                return work_function(*args, **kwargs)

            try:
                logger.debug("collection started")
                result = work_function(*args, **kwargs)
                self._stop_strobelight_no_throw(collect_results=True)
                return result
            except Exception as error:
                logger.warning("work function throw exception")
                self._stop_strobelight_no_throw(collect_results=False)
                raise error
