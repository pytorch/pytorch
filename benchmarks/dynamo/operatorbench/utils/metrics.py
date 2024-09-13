from typing import List, Optional
from enum import Enum
from triton.testing import do_bench
from .common import Device

class MetricResult:
    execution_time: List[float] = []
    throughputs: Optional[List[float]] = None
    cpu_peak_mem: Optional[float] = None
    gpu_peak_mem: Optional[float] = None



# Define an Enum for metrics
class Metrics(Enum):
    EXECUTION_TIME = "execution_time"
    THROUGHPUT = "throughput"
    CPU_PEAK_MEM = "cpu_peak_mem"
    GPU_PEAK_MEM = "gpu_peak_mem"

def get_execution_time(fn, quantiles=None, grad_to_none=None, device=None, **kwargs):
    if device == Device.CUDA:
        return do_bench(fn, quantiles=quantiles, grad_to_none=grad_to_none, **kwargs)
    else:
        raise ValueError(f"Device {device} is not supported")