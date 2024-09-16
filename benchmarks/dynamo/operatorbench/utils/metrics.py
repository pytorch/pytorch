from enum import Enum
from typing import Any, List, Tuple

from triton.testing import do_bench

from .common import Device


class MetricResult:
    # The first dimension is the sample index, the second dimension is the metric value
    op_name: str = ""
    op_variantant: str = ""
    execution_time: List[List[float]] = []
    mem_throughput: List[List[float]] = []
    cpu_peak_mem: float = None
    gpu_peak_mem: float = None
    input: List[
        Tuple[Any, Any]
    ] = []  # Correlate metrics with inputs. indexing by sample

    def __str__(self):
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
