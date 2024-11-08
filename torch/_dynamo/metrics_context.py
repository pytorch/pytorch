from typing import Any, Callable, Dict, Optional, Type
from typing_extensions import TypeAlias


OnExitType: TypeAlias = Callable[[Dict[str, Any]], None]


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
        self._level = 0

    def __enter__(self) -> "MetricsContext":
        """
        Initialize metrics recording.
        """
        if self._level == 0:
            # In case of recursion, track at the outermost context.
            self._metrics = {}

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
            self._on_exit(self._metrics)

    def increment(self, metric: str, value: int) -> None:
        """
        Increment a metric by a given amount.
        """
        if self._level == 0:
            raise RuntimeError(f"Cannot increment {metric} outside of a MetricsContext")
        if metric not in self._metrics:
            self._metrics[metric] = 0
        self._metrics[metric] += value

    def set(self, metric: str, value: Any, overwrite: bool = False) -> None:
        """
        Set a metric to a given value. If overwrite=False (the default), raise if the
        metric has been assigned previously in the current context.
        """
        if self._level == 0:
            raise RuntimeError(f"Cannot set {metric} outside of a MetricsContext")
        if not overwrite and metric in self._metrics:
            raise RuntimeError(
                f"Metric '{metric}' has already been set in the current context"
            )
        self._metrics[metric] = value

    def update(self, values: Dict[str, Any], overwrite: bool = False) -> None:
        """
        Set multiple metrics directly. This method does NOT increment. If overwrite=False
        (the default), raise if any metric has been assigned previously in the current
        context.
        """
        if self._level == 0:
            raise RuntimeError("Cannot update metrics outside of a MetricsContext")
        if not overwrite:
            existing = self._metrics.keys() & values.keys()
            if existing:
                raise RuntimeError(
                    f"Metric(s) {existing} have already been set in the current context"
                )
        self._metrics.update(values)
