import unittest
from caffe2.python import convnet_benchmarks as cb
from caffe2.python import test_util, workspace


@unittest.skipIf(not workspace.has_gpu_support, "no gpu")
class TestConvnetBenchmarks(test_util.TestCase):
    def testConvnetBenchmarks(self):
        all_args = [
            '--batch_size 16 --order NCHW --iterations 1 '
            '--warmup_iterations 1',
            '--batch_size 16 --order NCHW --iterations 1 '
            '--warmup_iterations 1 --forward_only',
        ]
        for model in [cb.AlexNet, cb.OverFeat, cb.VGGA, cb.Inception]:
            for arg_str in all_args:
                args = cb.GetArgumentParser().parse_args(arg_str.split(' '))
                cb.Benchmark(model, args)


if __name__ == '__main__':
    unittest.main()
