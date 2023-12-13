from torch._dynamo.utils import CompilationMetrics
from typing import Sequence
from collections import deque
from torch._utils_internal import log_compilation_event


DEFAULT_SIZE = 64


_compilation_metrics: deque[CompilationMetrics] = deque(maxlen=DEFAULT_SIZE)


def record_compilation_metrics(compilation_metrics: CompilationMetrics):
    global _compilation_metrics
    _compilation_metrics.append(compilation_metrics)
    log_compilation_event(compilation_metrics)


def set_size(new_size: int) -> None:
    global _compilation_metrics
    while len(_compilation_metrics) > new_size:
        _compilation_metrics.popleft()
    new_deque = deque(_compilation_metrics, maxlen=new_size)
    _compilation_metrics = new_deque


def clear_compilation_metrics() -> None:
    global _compilation_metrics
    _compilation_metrics.clear()


def get_compilation_metrics() -> Sequence[CompilationMetrics]:
    return list(_compilation_metrics)
