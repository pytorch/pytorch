from typing import Any, Callable, Dict, Optional, Type, Union
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
            self._on_exit(self._metrics, exc_type, exc_value)

    def recording(self) -> bool:
        """
        Return True if we've entered the context manager.
        """
        return self._level > 0

    def _check(self) -> None:
        """
        Raise if we haven't entered the contextmanager.
        """
        if self._level == 0:
            raise RuntimeError("Cannot set metrics outside of a MetricsContext")

    def increment(self, metric: str, value: Union[str, float]) -> None:
        """
        Increment a metric by a given amount.
        """
        # TODO: do we want to be safe and grab a lock whenever modifying an entry? For
        # example, I saw that we may be adding a helper thread for the remote cache and
        # the get/put timing could be bumped on that helper...
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
