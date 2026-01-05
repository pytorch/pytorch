# Owner(s): ["module: unknown"]

import torch
from torch.testing._internal.common_utils import run_tests, TemporaryFileName, TestCase
from torch.utils import ThroughputBenchmark


class TwoLayerNet(torch.jit.ScriptModule):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(2 * H, D_out)

    @torch.jit.script_method
    def forward(self, x1, x2):
        h1_relu = self.linear1(x1).clamp(min=0)
        h2_relu = self.linear1(x2).clamp(min=0)
        cat = torch.cat((h1_relu, h2_relu), 1)
        y_pred = self.linear2(cat)
        return y_pred


class TwoLayerNetModule(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(2 * H, D_out)

    def forward(self, x1, x2):
        h1_relu = self.linear1(x1).clamp(min=0)
        h2_relu = self.linear1(x2).clamp(min=0)
        cat = torch.cat((h1_relu, h2_relu), 1)
        y_pred = self.linear2(cat)
        return y_pred


class TestThroughputBenchmark(TestCase):
    def linear_test(self, Module, profiler_output_path=""):
        D_in = 10
        H = 5
        D_out = 15
        B = 8
        NUM_INPUTS = 2

        module = Module(D_in, H, D_out)

        inputs = []

        for _ in range(NUM_INPUTS):
            inputs.append([torch.randn(B, D_in), torch.randn(B, D_in)])
        bench = ThroughputBenchmark(module)

        for input in inputs:
            # can do both args and kwargs here
            bench.add_input(input[0], x2=input[1])

        for i in range(NUM_INPUTS):
            # or just unpack the list of inputs
            module_result = module(*inputs[i])
            bench_result = bench.run_once(*inputs[i])
            torch.testing.assert_close(bench_result, module_result)

        stats = bench.benchmark(
            num_calling_threads=4,
            num_warmup_iters=100,
            num_iters=1000,
            profiler_output_path=profiler_output_path,
        )

        print(stats)

    def test_script_module(self):
        self.linear_test(TwoLayerNet)

    def test_module(self):
        self.linear_test(TwoLayerNetModule)

    def test_profiling(self):
        with TemporaryFileName() as fname:
            self.linear_test(TwoLayerNetModule, profiler_output_path=fname)

    def linear_with_compile_test(self, Module, dtype):
        from contextlib import nullcontext

        from torch._dynamo import config
        from torch._inductor import config as inductor_config

        config.error_on_recompile = True
        inductor_config.cpp_wrapper = True
        inductor_config.freezing = True
        D_in = 10
        H = 5
        D_out = 15
        B = 8

        autocast = dtype != torch.float32
        module = Module(D_in, H, D_out)

        input = (torch.randn(B, D_in), torch.randn(B, D_in))

        with torch.no_grad(), torch.amp.autocast("cpu", enabled=autocast, dtype=dtype):
            torch._dynamo.reset()
            module(*input)
            module = torch.compile(module)
            module(*input)
            module(*input)

        ctx = nullcontext()
        if dtype == torch.float16 or dtype == torch.bfloat16:
            ctx = torch.amp.autocast("cpu", enabled=autocast, dtype=dtype)
        with torch.no_grad(), ctx:
            bench = ThroughputBenchmark(module)
            bench.add_input(*input)

            module_result = module(*input)
            bench_result = bench.run_once(*input)
            torch.testing.assert_close(bench_result, module_result)

            stats = bench.benchmark(
                num_calling_threads=4, num_warmup_iters=100, num_iters=1000
            )

            print(stats)

    def test_compile(self):
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        for dtype in dtypes:
            self.linear_with_compile_test(TwoLayerNetModule, dtype)


if __name__ == "__main__":
    run_tests()
