from typing import Callable, List, Union

import torch


def benchmark_func(
    func: Callable,
    sizes: Union[int, List[Union[List[int], int]]],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> "torch.utils.benchmark.Measurement":
    """Convenienence  wrapper for benchmarking an individual operators with specific input sizes and dtypes
    Measures the wall clock time of running the function , and synchronizing on device if needed.
    Example:

      >>> torch.testing._benchmark_func(torch.sin, [1024,])
      <torch.utils.benchmark.utils.common.Measurement object at 0x10b2a96a0>
      f(*args);
      setup: args = [torch.testing.make_tensor(s, dtype=torch.float32, device='cpu') for s in [1024]]
      Median: 1.29 us
      IQR:    0.13 us (1.29 to 1.42)
      148525 measurements, 1 runs per measurement, 1 thread
    """
    from timeit import default_timer

    from torch.utils.benchmark import Timer

    if device == "mps":
        sync_cmd = "torch.mps.synchronize()"
    elif device == "cuda":
        sync_cmd = "torch.cuda.synchronize()"
    else:
        sync_cmd = ""

    if isinstance(sizes, int):
        sizes = [sizes]
    t = Timer(
        stmt=f"f(*args);{sync_cmd}",
        setup=f"args = [torch.testing.make_tensor(s, dtype={dtype}, device='{device}') for s in {sizes}]",
        globals={
            "f": func,
        },
        language="python",
        timer=default_timer,
    )
    return t.blocked_autorange()
