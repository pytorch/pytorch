import time

from strobelight_cli_function_profiler import (
    strobelight,
    StrobelightCLIFunctionProfiler,
)

import torch

if __name__ == "__main__":

    def fn(x, y, z):
        return x * y + z

    # use decorator with default profiler.
    @strobelight()
    @torch.compile()
    def work():
        for i in range(100):
            for j in range(5):
                fn(torch.rand(j, j), torch.rand(j, j), torch.rand(j, j))

    work()

    # you can also pass profiler constructor arguments.
    @strobelight(stop_at_error=False)
    def work2():
        sum = 0
        for i in range(100000000):
            sum += 1

    work2()

    # or pass a profiler instance.
    profiler = StrobelightCLIFunctionProfiler(stop_at_error=False)

    @strobelight(profiler)
    def work3():
        for i in range(10):
            time.sleep(1)

    work3()
