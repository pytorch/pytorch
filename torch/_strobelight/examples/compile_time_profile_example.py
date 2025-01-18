# mypy: allow-untyped-defs
import torch
from torch._strobelight.compile_time_profiler import StrobelightCompileTimeProfiler


if __name__ == "__main__":
    # You can pass TORCH_COMPILE_STROBELIGHT=True instead.
    StrobelightCompileTimeProfiler.enable()

    def fn(x, y, z):
        return x * y + z

    @torch.compile()
    def work(n):
        for _ in range(3):
            for _ in range(5):
                fn(torch.rand(n, n), torch.rand(n, n), torch.rand(n, n))

    # Strobelight will be called only 3 times because dynamo will be disabled after
    # 3rd iteration.
    for i in range(3):
        torch._dynamo.reset()
        work(i)
