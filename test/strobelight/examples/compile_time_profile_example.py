# mypy: allow-untyped-defs
import torch
from torch._strobelight.compile_time_profiler import StrobelightCompileTimeProfiler


if __name__ == "__main__":
    # You can pass TORCH_COMPILE_STROBELIGHT=True instead.
    StrobelightCompileTimeProfiler.enable()

    # You can use the code below to filter what frames to be profiled.
    StrobelightCompileTimeProfiler.frame_id_filter = "1/.*"
    # StrobelightCompileTimeProfiler.frame_id_filter='0/.*'
    # StrobelightCompileTimeProfiler.frame_id_filter='.*'
    # You can set env variable COMPILE_STROBELIGHT_FRAME_FILTER to set the filter also.

    def fn(x, y, z):
        return x * y + z

    @torch.compile()
    def work(n):
        for _ in range(3):
            for _ in range(5):
                fn(torch.rand(n, n), torch.rand(n, n), torch.rand(n, n))

    # Strobelight will be called only 3 times because dynamo will be disabled after
    # 3rd iteration.
    # Frame 0/0
    for i in range(3):
        torch._dynamo.reset()
        work(i)

    @torch.compile(fullgraph=True)
    def func4(x):
        return x * x

    # Frame 1/0
    func4(torch.rand(10))
