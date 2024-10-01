from enum import Enum
from typing import Any, List, Tuple

from triton.testing import do_bench
import nvtx
from contextlib import contextmanager
import torch
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


class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"

@contextmanager
def profile_range(range_name):
    with nvtx.annotate(range_name):
        yield

def get_execution_time(fn, grad_to_none=None, device=None, **kwargs):
    """
    Get the execution time of a function.
    For CUDA, we use triton's do_bench. Note: it has a default repeat of 100 and warmup of 25.
    """
    if device == Device.CUDA:
        return do_bench(fn, grad_to_none=grad_to_none, **kwargs)
    else:
        raise ValueError(f"Device {device} is not supported")

    
def do_profile_bench(fn, n_repeat=5, grad_to_none=None):
    """
    :param fn: Function to benchmark
    :type fn: Callable
    :param n_repeat: Repetition number. Because this is for ncu profiling, 
        we don't need to repeat the function many times. So we use number instead of time.
    :type n_repeat: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    """
    torch.cuda.synchronize()
    for _ in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        fn()
    torch.cuda.synchronize()

def do_profile_warmup(fn, warmup=25, fast_flush=True):
    """
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    """
    fn()
    torch.cuda.synchronize()
    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')
    # Estimate the runtime of the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    # Warm-up
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()