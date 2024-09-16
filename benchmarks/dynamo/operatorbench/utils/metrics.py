from enum import Enum
from typing import Any, List, Tuple

from triton.testing import do_bench

from .common import Device


class MetricResult:
    def __init__(self) -> None:
        self.op_name: str = ""
        self.op_variantant: str = ""
        # The first dimension is the sample index, the second dimension is the metric value for each repeat
        self.execution_time: List[List[float]] = []  # List of lists for execution times
        self.mem_throughput: List[
            List[float]
        ] = []  # List of lists for memory throughput
        self.cpu_peak_mem: float = None  # Peak CPU memory usage
        self.gpu_peak_mem: float = None  # Peak GPU memory usage
        self.input: List[
            Tuple[Any, Any]
        ] = []  # Correlate metrics with inputs, indexed by sample

    def __str__(self) -> str:
        return (
            f"MetricResult(op_name={self.op_name}, "
            f"op_variantant={self.op_variantant}, "
            f"execution_time={self.execution_time}, "
            f"mem_throughput={self.mem_throughput}, "
            f"cpu_peak_mem={self.cpu_peak_mem}, "
            f"gpu_peak_mem={self.gpu_peak_mem})"
        )


# Define an Enum for metrics
class Metrics(Enum):
    EXECUTION_TIME = "execution_time"
    MEM_THROUGHPUT = "mem_throughput"
    CPU_PEAK_MEM = "cpu_peak_mem"
    GPU_PEAK_MEM = "gpu_peak_mem"


def get_execution_time(fn, grad_to_none=None, device=None, **kwargs):
    """
    Get the execution time of a function.
    For CUDA, we use triton's do_bench. Note: it has a default repeat of 100 and warmup of 25.
    """
    if device == Device.CUDA:
        return do_bench(fn, grad_to_none=grad_to_none, **kwargs)
    else:
        raise ValueError(f"Device {device} is not supported")
