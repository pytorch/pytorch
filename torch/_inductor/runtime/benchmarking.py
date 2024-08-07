import time
from functools import cached_property, partial
from statistics import median
from typing import Any, Callable, Dict, List, Tuple
from typing_extensions import Self

from torch._inductor.utils import is_cpu_device


MILLISECONDS_PER_SECOND = 1000


class Benchmarker:
    def __init__(self: Self) -> None:
        pass

    def benchmark(
        self: Self,
        fn: Callable[..., Any],
        fn_args: Tuple[Any],
        fn_kwargs: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> float:
        """Dispatch benchmark request to CPU or GPU depending on device of `fn_args` and `fn_kwargs`"""
        if is_cpu_device(list(fn_args) + list(fn_kwargs.values())):
            return self.benchmark_cpu(
                lambda: fn(*fn_args, **fn_kwargs), *args, **kwargs
            )
        return self.benchmark_gpu(lambda: fn(*fn_args, **fn_kwargs), *args, **kwargs)

    def benchmark_cpu(
        self: Self, _callable: Callable[[], Any], warmup: int = 20, rep: int = 100
    ) -> float:
        """Benchmark a CPU callable, and return the median runtime in milliseconds.

        Parameters:
        - fn: The callable.
        - warmup: Duration, in milliseconds, to run the callable before starting benchmarking.
        - rep: Duration, in milliseconds, to run the benchmarking.

        Returns:
        - The median runtime, in milliseconds, of the callable.
        """

        def run_for(ms: int) -> List[float]:
            timings = []
            run_start_s = time.perf_counter()
            while True:
                start_s = time.perf_counter()
                _callable()
                end_s = time.perf_counter()
                timings.append((end_s - start_s) * MILLISECONDS_PER_SECOND)
                if ((end_s - run_start_s) * MILLISECONDS_PER_SECOND) > ms:
                    break
            return timings

        run_for(warmup)
        return median(run_for(rep))

    @cached_property
    def triton_do_bench(self: Self) -> Callable[..., Any]:
        """Lazily import Triton's do_bench, and set return mode to median."""
        try:
            from triton.testing import do_bench
        except ImportError as e:
            raise NotImplementedError("requires Triton") from e
        return partial(do_bench, return_mode="median")

    def benchmark_gpu(self: Self, *args: Any, **kwargs: Any) -> float:
        """Benchmark a GPU callable using Triton's do_bench, and return the median runtime in milliseconds."""
        return self.triton_do_bench(*args, **kwargs)


benchmarker = Benchmarker()
