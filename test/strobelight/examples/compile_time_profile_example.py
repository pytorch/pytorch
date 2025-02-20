# mypy: allow-untyped-defs
import torch
from torch._strobelight.compile_time_profiler import StrobelightCompileTimeProfiler


if __name__ == "__main__":
    # You can pass TORCH_COMPILE_STROBELIGHT=True instead.
    # StrobelightCompileTimeProfiler.enable()

    def fn(x, y, z):
        return x * y + z

    # @torch.compile(fullgraph=True)
    # def e(n):
    #     for _ in range(3):
    #         for _ in range(5):
    #             fn(torch.rand(n, n), torch.rand(n, n), torch.rand(n, n))

    # Strobelight will be called only 3 times because dynamo will be disabled after
    # 3rd iteration.
    # Those will be 0/0
    
    # set COMPILE_STROBELIGHT_FRAME_FILTER="0" to only profile this. 
    # for i in range(3):
    #     # torch._dynamo.reset()
    #     e(i)
    
    # print("finished compiling first frame")

    @torch.compile(fullgraph=True)
    def func2(x):
        for i in range(0, 100):
            x  = x*2
        return x

    # set COMPILE_STROBELIGHT_FRAME_FILTER="1" to only profile this. 
    func2(torch.rand(10))


    @torch.compile(fullgraph=True)
    def func4(x):
        return x*x
    
    func4(torch.rand(10))
