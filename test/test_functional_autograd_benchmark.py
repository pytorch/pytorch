import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_utils import slowTest

import subprocess
import tempfile
import os

# This is a very simple smoke test for the functional autograd benchmarking script.
class TestFunctionalAutogradBenchmark(TestCase):
    def _test_runner(self, model):
        with tempfile.NamedTemporaryFile() as out_file:
            cmd = ['python', '../benchmarks/functional_autograd_benchmark/functional_autograd_benchmark.py']
            # Only run the warmup
            cmd += ['--num-iters', '0']
            # Only run the vjp task (fastest one)
            cmd += ['--task-filter', 'vjp']
            # Only run the specified model
            cmd += ['--model-filter', model]
            # Output file
            cmd += ['--output', out_file.name]

            res = subprocess.run(cmd)

            self.assertTrue(res.returncode == 0)
            # Check that somethink was writen to the file
            out_file.seek(0, os.SEEK_END)
            self.assertTrue(out_file.tell() > 0)


    def test_fast_tasks(self):
        fast_tasks = ['resnet18', 'ppl_simple_reg', 'ppl_robust_reg', 'wav2letter',
                      'transformer', 'multiheadattn']

        for task in fast_tasks:
            self._test_runner(task)

    @slowTest
    def test_slow_tasks(self):
        slow_tasks = ['fcn_resnet', 'detr']
        # deepspeech is volontarily excluded as it takes too long to run without
        # proper tuning of the number of threads it should use.

        for task in slow_tasks:
            self._test_runner(task)


if __name__ == '__main__':
    run_tests()
