from typing import List, Optional, Any, Tuple
from enum import Enum
from triton.testing import do_bench
from .common import Device


class MetricResult:
    # The first dimension is the sample index, the second dimension is the metric value
    execution_time: List[List[float]] = []
    mem_throughput: List[List[float]] = []
    cpu_peak_mem: float = None
    gpu_peak_mem: float = None
    input: List[Tuple[Any, Any]] = []  # Correlate metrics with inputs. indexing by sample


# Define an Enum for metrics
class Metrics(Enum):
    EXECUTION_TIME = "execution_time"
    THROUGHPUT = "throughput"
    CPU_PEAK_MEM = "cpu_peak_mem"
    GPU_PEAK_MEM = "gpu_peak_mem"


def get_execution_time(fn, quantiles=None, grad_to_none=None, device=None, **kwargs):
    """
    Get the execution time of a function.
    For CUDA, we use triton's do_bench. Note: it has a default repeat of 100 and warmup of 25.
    """
    if device == Device.CUDA:
        return do_bench(fn, quantiles=quantiles, grad_to_none=grad_to_none, **kwargs)
    else:
        raise ValueError(f"Device {device} is not supported")
