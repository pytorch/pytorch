import torch

from tools.strobelight.compile_time_profiler import StrobelightCompileTimeProfiler

if __name__ == "__main__":
    # You can pass TORCH_COMPILE_STROBELIGHT=True instead.
    StrobelightCompileTimeProfiler.enable()

    def fn(x, y, z):
        return x * y + z

    @torch.compile()
    def work():
        for i in range(10):
            for j in range(5):
                fn(torch.rand(j, j), torch.rand(j, j), torch.rand(j, j))

    # compilation will happen 3 times.
    work()
