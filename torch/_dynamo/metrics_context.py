import time
from typing import Any, Callable, Dict, Optional, Type
from typing_extensions import TypeAlias


OnExitType: TypeAlias = Callable[
    [int, int, Dict[str, Any], Optional[Type[BaseException]], Optional[BaseException]],
    None,
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
        self._start_time_ns: int = 0
        self._level: int = 0

    def __enter__(self) -> "MetricsContext":
        """
        Initialize metrics recording.
        """
        if self._level == 0:
            # In case of recursion, track at the outermost context.
            self._metrics = {}
            self._start_time_ns = time.time_ns()

        self._level += 1
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        _traceback: Any,
    ) -> None:
        """
        At exit, call the provided on_exit function.
        """
        self._level -= 1
        assert self._level >= 0
        if self._level == 0:
            end_time_ns = time.time_ns()
            self._on_exit(
                self._start_time_ns, end_time_ns, self._metrics, exc_type, exc_value
            )

    def in_progress(self) -> bool:
        """
        True if we've entered the context.
        """
        return self._level > 0

    def increment(self, metric: str, value: int) -> None:
        """
        Increment a metric by a given amount.
        """
        if self._level == 0:
            raise RuntimeError(f"Cannot increment {metric} outside of a MetricsContext")
        if metric not in self._metrics:
            self._metrics[metric] = 0
        self._metrics[metric] += value

    def set(self, metric: str, value: Any) -> None:
        """
        Set a metric to a given value. Raises if the metric has been assigned previously
        in the current context.
        """
        if self._level == 0:
            raise RuntimeError(f"Cannot set {metric} outside of a MetricsContext")
        if metric in self._metrics:
            raise RuntimeError(
                f"Metric '{metric}' has already been set in the current context"
            )
        self._metrics[metric] = value

    def set_key_value(self, metric: str, key: str, value: Any) -> None:
        """
        Treats a give metric as a dictionary and set the k and value within it.
        Note that the metric must be a dictionary or not present.

        We allow this to be called multiple times (i.e. for features, it's not uncommon
        for them to be used multiple times within a single compilation).
        """
        if self._level == 0:
            raise RuntimeError(f"Cannot set {metric} outside of a MetricsContext")
        if metric not in self._metrics:
            self._metrics[metric] = {}
        self._metrics[metric][key] = value

    def update(self, values: Dict[str, Any]) -> None:
        """
        Set multiple metrics directly. This method does NOT increment. Raises if any
        metric has been assigned previously in the current context.
        """
        if self._level == 0:
            raise RuntimeError("Cannot update metrics outside of a MetricsContext")
        existing = self._metrics.keys() & values.keys()
        if existing:
            raise RuntimeError(
                f"Metric(s) {existing} have already been set in the current context"
            )
        self._metrics.update(values)

    def update_outer(self, values: Dict[str, Any]) -> None:
        """
        Update, but only when at the outermost context.
        """
        if self._level == 0:
            raise RuntimeError("Cannot update metrics outside of a MetricsContext")
        if self._level == 1:
            self.update(values)

    def add_to_set(self, metric: str, value: Any) -> None:
        """
        Records a metric as a set() of values.
        """
        if self._level == 0:
            raise RuntimeError(f"Cannot add {metric} outside of a MetricsContext")
        if metric not in self._metrics:
            self._metrics[metric] = set()
        self._metrics[metric].add(value)
