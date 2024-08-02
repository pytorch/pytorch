# mypy: disallow-untyped-defs

import logging
import os
from datetime import datetime
from socket import gethostname
from typing import Any, Optional

from torch._strobelight.cli_function_profiler import StrobelightCLIFunctionProfiler


logger = logging.getLogger("strobelight_compile_time_profiler")

console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(name)s, line %(lineno)d, %(asctime)s, %(levelname)s: %(message)s"
)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


class StrobelightCompileTimeProfiler:
    success_profile_count: int = 0
    failed_profile_count: int = 0
    ignored_profile_runs: int = 0
    inside_profile_compile_time: bool = False
    enabled: bool = False
    # A unique identifier that is used as the run_user_name in the strobelight profile to
    # associate all compile time profiles together.
    identifier: Optional[str] = None

    current_phase: Optional[str] = None

    profiler: Optional[Any] = None

    max_stack_length: int = int(
        os.environ.get("COMPILE_STROBELIGHT_MAX_STACK_LENGTH", 127)
    )
    max_profile_time: int = int(
        os.environ.get("COMPILE_STROBELIGHT_MAX_PROFILE_TIME", 60 * 30)
    )
    # Collect sample each x cycles.
    sample_each: int = int(
        float(os.environ.get("COMPILE_STROBELIGHT_SAMPLE_RATE", 1e7))
    )

    @classmethod
    def enable(cls, profiler_class: Any = StrobelightCLIFunctionProfiler) -> None:
        if cls.enabled:
            logger.info("compile time strobelight profiling already enabled")
            return

        logger.info("compile time strobelight profiling enabled")

        if profiler_class is StrobelightCLIFunctionProfiler:
            import shutil

            if not shutil.which("strobeclient"):
                logger.info(
                    "strobeclient not found, cant enable compile time strobelight profiling, seems"
                    "like you are not on a FB machine."
                )
                return

        cls.enabled = True
        cls._cls_init()
        # profiler_class should have public API similar to that of StrobelightCLIFunctionProfiler.
        # we have pass different functionProfilerClass for meta-internal fbcode targets.
        cls.profiler = profiler_class(
            sample_each=cls.sample_each,
            max_profile_duration_sec=cls.max_profile_time,
            stack_max_len=cls.max_stack_length,
            async_stack_max_len=cls.max_stack_length,
            run_user_name="pt2-profiler/"
            + os.environ.get("USER", os.environ.get("USERNAME", "")),
            sample_tags={cls.identifier},
        )

    @classmethod
    def _cls_init(cls) -> None:
        cls.identifier = "{date}{pid}{hostname}".format(
            date=datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
            pid=os.getpid(),
            hostname=gethostname(),
        )

        logger.info("Unique sample tag for this run is: %s", cls.identifier)
        logger.info(
            "You can use the following link to access the strobelight profile at the end of the run: %s",
            (
                "https://www.internalfb.com/intern/scuba/query/?dataset=pyperf_experime"
                "ntal%2Fon_demand&drillstate=%7B%22purposes%22%3A[]%2C%22end%22%3A%22no"
                "w%22%2C%22start%22%3A%22-30%20days%22%2C%22filterMode%22%3A%22DEFAULT%"
                "22%2C%22modifiers%22%3A[]%2C%22sampleCols%22%3A[]%2C%22cols%22%3A[%22n"
                "amespace_id%22%2C%22namespace_process_id%22]%2C%22derivedCols%22%3A[]%"
                "2C%22mappedCols%22%3A[]%2C%22enumCols%22%3A[]%2C%22return_remainder%22"
                "%3Afalse%2C%22should_pivot%22%3Afalse%2C%22is_timeseries%22%3Afalse%2C"
                "%22hideEmptyColumns%22%3Afalse%2C%22timezone%22%3A%22America%2FLos_Ang"
                "eles%22%2C%22compare%22%3A%22none%22%2C%22samplingRatio%22%3A%221%22%2"
                "C%22metric%22%3A%22count%22%2C%22aggregation_field%22%3A%22async_stack"
                "_complete%22%2C%22top%22%3A10000%2C%22aggregateList%22%3A[]%2C%22param"
                "_dimensions%22%3A[%7B%22dim%22%3A%22py_async_stack%22%2C%22op%22%3A%22"
                "edge%22%2C%22param%22%3A%220%22%2C%22anchor%22%3A%220%22%7D]%2C%22orde"
                "r%22%3A%22weight%22%2C%22order_desc%22%3Atrue%2C%22constraints%22%3A[["
                "%7B%22column%22%3A%22sample_tags%22%2C%22op%22%3A%22all%22%2C%22value%"
                f"22%3A[%22[%5C%22{cls.identifier}%5C%22]%22]%7D]]%2C%22c_constraints%22%3A[[]]%2C%22b"
                "_constraints%22%3A[[]]%2C%22ignoreGroupByInComparison%22%3Afalse%7D&vi"
                "ew=GraphProfilerView&&normalized=1712358002&pool=uber"
            ),
        )

    @classmethod
    def _log_stats(cls) -> None:
        logger.info(
            "%s strobelight success runs out of %s non-recursive compilation events.",
            cls.success_profile_count,
            cls.success_profile_count + cls.failed_profile_count,
        )

    # TODO use threadlevel meta data to tags to record phases.
    @classmethod
    def profile_compile_time(
        cls, func: Any, phase_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        if not cls.enabled:
            return func(*args, **kwargs)

        if cls.profiler is None:
            logger.error("profiler is not set")
            return

        if cls.inside_profile_compile_time:
            cls.ignored_profile_runs += 1
            logger.info(
                "profile_compile_time is requested for phase: %s while already in running phase: %s, recursive call ignored",
                phase_name,
                cls.current_phase,
            )
            return func(*args, **kwargs)

        cls.inside_profile_compile_time = True
        cls.current_phase = phase_name

        work_result = cls.profiler.profile(func, *args, **kwargs)

        if cls.profiler.profile_result is not None:
            cls.success_profile_count += 1
        else:
            cls.failed_profile_count += 1

        cls._log_stats()
        cls.inside_profile_compile_time = False
        return work_result
