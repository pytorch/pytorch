import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional, Type, Union
from typing_extensions import TypeAlias


OnExitType: TypeAlias = Callable[
    [Dict[str, Any], Optional[Type[BaseException]], Optional[BaseException]], None
]


class MetricsContext:
    def __init__(self, on_exit: OnExitType):
        """
        Use this class as a contextmanager to create a context under which to accumulate
        a set of metrics, e.g., metrics gathered during a compilation. On exit of the
        contextmanager, call the provided 'on_exit' function and pass a dictionary of
        all metrics set during the lifetime of the contextmanager.
        """
        self._on_exit = on_exit
        self._metrics: Dict[str, Any] = {}
        self._start_time: Dict[str, int] = {}
        self._recording = False

    def __enter__(self) -> "MetricsContext":
        """
        Initialize metrics recording.
        """
        if self._recording:
            raise RuntimeError("Cannot re-enter a MetricsContext")
        self._metrics = {}
        self._start_time = {}
        self._recording = True
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        _traceback: Any,
    ) -> None:
        """
        Post-process metrics where appropriate and call the provided on_exit function.
        """
        self._on_exit(self._metrics, exc_type, exc_value)
        self._recording = False

    def _check(self) -> None:
        """
        Raise if we haven't entered the contextmanager.
        """
        if not self._recording:
            raise RuntimeError("Cannot set metrics outside of a MetricsContext")

    def increment(self, metric: str, value: Union[str, float]) -> None:
        """
        Increment a metric by a given amount.
        """
        self._check()
        if metric not in self._metrics:
            self._metrics[metric] = 0
        self._metrics[metric] += value

    def set(self, metric: str, value: Any) -> None:
        """
        Set a metric to a given value. Use set_once to raise if the metric is set
        more than once.
        """
        self._check()
        self._metrics[metric] = value

    def set_once(self, metric: str, value: Any) -> None:
        """
        Set a metric to a given value. Raise if the metrics has been set previously
        in the current context.
        """
        self._check()
        if metric in self._metrics:
            raise RuntimeError(
                f"Metric '{metric}' has already been set in the current context"
            )
        self._metrics[metric] = value

    def update(self, values: Dict[str, Any]) -> None:
        """
        Set multiple metrics.
        """
        self._check()
        self._metrics.update(values)

    @contextmanager
    def timed(self, metric: str) -> Generator[Any, None, None]:
        """
        Use this context manager to record execution time. Automatically adjusts the
        recorded time according to the suffix of the provided metric name: "_ns", "_us",
        "_ms", or "_s" for nanoseconds, microseconds, etc.
        """

        # To properly record timing when there's recursion, start and stop the timer
        # only for the outermost instance of any metric.
        outermost = False
        if metric not in self._start_time:
            self._start_time[metric] = time.time_ns()
            outermost = True
        try:
            yield
        finally:
            if outermost:
                elapsed = time.time_ns() - self._start_time[metric]

                # Adjust according to suffix. Assume seconds are floats and everything
                # else is an integer.
                if metric.endswith("_ns"):
                    pass
                elif metric.endswith("_us"):
                    elapsed = elapsed // 1000
                elif metric.endswith("_ms"):
                    elapsed = elapsed // 1000**2
                else:
                    # Assume anything else is seconds.
                    elapsed = float(elapsed) / 1000**3

                self.increment(metric, elapsed)
                del self._start_time[metric]
