
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
operatorbench_dir = os.path.join(current_dir, '..', 'benchmarks', 'dynamo')
sys.path.append(operatorbench_dir)
from operatorbench.run import run_benchmarks
from click.testing import CliRunner


import torch
from torch._inductor.test_case import TestCase, run_tests
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch._dynamo.utils import counters

@instantiate_parametrized_tests
class OperatorBenchTestCase(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(23456)
        counters.clear()

    @parametrize("device", [GPU_TYPE])
    @parametrize("op", ["FusedLinearCrossEntropy"])
    # @parametrize("dtype", ["float32", "float16", "bfloat16"])
    @parametrize("dtype", ["float32"])
    @parametrize("phase", ["forward", "backward", "full"])
    def test_FusedLinearCrossEntropy(self, device, op, dtype, phase):
        args = [
            "--op", op,
            "--dtype", dtype,
            "--max-samples", "1",
            "--device", device,
            "--phase", phase,
            "--repeat", "1",
            "--metrics", "execution_time",
        ]
        runner = CliRunner()
        result = runner.invoke(run_benchmarks, args)
        if result.exit_code != 0:
            print("args:", args)
            print("Error:", result.output)
            print(result)
            raise RuntimeError("Failed to run benchmarks")
if __name__ == "__main__":

    if HAS_GPU:
        run_tests(needs="filelock")
