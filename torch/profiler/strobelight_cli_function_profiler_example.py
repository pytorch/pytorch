import torch
from torch.profiler.strobelight_cli_function_profiler import (
    strobelight,
    StrobelightCLIFunctionProfiler,
)

if __name__ == "__main__":

    def fn(x, y, z):
        return x * y + z

    # use decorator with default profiler or optional profile arguments.
    @strobelight(sample_each=100, stop_at_error=False)
    @torch.compile()
    def work():
        for i in range(10000):
            torch._dynamo.reset()
            for j in range(5):
                fn(torch.rand(j, j), torch.rand(j, j), torch.rand(j, j))

    work()

    # or pass a profiler instance.
    profiler = StrobelightCLIFunctionProfiler(stop_at_error=False)

    @strobelight(profiler)
    def work2():
        sum = 0
        for i in range(100000000):
            sum += 1

    work2()
