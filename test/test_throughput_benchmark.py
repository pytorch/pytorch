
import torch
from torch.utils import ThroughputBenchmark
from torch.testing import assert_allclose

from torch.testing._internal.common_utils import run_tests, TestCase, TemporaryFileName

class TwoLayerNet(torch.jit.ScriptModule):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
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
        super(TwoLayerNetModule, self).__init__()
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

        for i in range(NUM_INPUTS):
            inputs.append([torch.randn(B, D_in), torch.randn(B, D_in)])
        bench = ThroughputBenchmark(module)

        for input in inputs:
            # can do both args and kwargs here
            bench.add_input(input[0], x2=input[1])

        for i in range(NUM_INPUTS):
            # or just unpack the list of inputs
            module_result = module(*inputs[i])
            bench_result = bench.run_once(*inputs[i])
            assert_allclose(bench_result, module_result)

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


if __name__ == '__main__':
    run_tests()
