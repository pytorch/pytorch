## @package device_reduce_sum_bench
# Module caffe2.experiments.python.device_reduce_sum_bench
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import itertools
import logging
import os

from six import add_metaclass
import numpy as np

from caffe2.python import workspace, core
from caffe2.python.hypothesis_test_util import runOpBenchmark, gpu_do

logging.basicConfig()
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)

ALL_BENCHMARKS = {}


class BenchmarkMeta(type):
    def __new__(metacls, name, bases, class_dict):
        cls = type.__new__(metacls, name, bases, class_dict)
        if name != 'Benchmark':
            ALL_BENCHMARKS[name] = cls
        return cls


@add_metaclass(BenchmarkMeta)
class Benchmark(object):

    def __init__(self):
        self.results = []

    def display(self):
        print('Results ({}):'.format(type(self).__name__))
        print('input size                      ms/iter')
        print('------------------------------  -----------')
        for size, ms in self.results:
            print('{!s:<30}  {:.4f}'.format(size, ms))


class SumElements(Benchmark):
    def run(self):
        op = core.CreateOperator(
            "SumElements",
            ["X"],
            ["y"]
        )

        for n in itertools.imap(pow, itertools.cycle([10]), range(10)):
            X = np.random.rand(n).astype(np.float32)
            logger.info('Running benchmark for n = {}'.format(n))
            ret = runOpBenchmark(gpu_do, op, inputs=[X])
            self.results.append((n, ret[1]))


class SumSqrElements(Benchmark):
    def run(self):
        op = core.CreateOperator(
            "SumSqrElements",
            ["X"],
            ["y"]
        )

        for n in itertools.imap(pow, itertools.cycle([10]), range(10)):
            X = np.random.rand(n).astype(np.float32)
            logger.info('Running benchmark for n = {}'.format(n))
            ret = runOpBenchmark(gpu_do, op, inputs=[X])
            self.results.append((n, ret[1]))


class SoftMaxWithLoss(Benchmark):
    def run(self):
        op = core.CreateOperator(
            "SoftmaxWithLoss",
            ["X", "label"],
            ["probs", "avgloss"],
        )

        for n in itertools.imap(pow, itertools.cycle([10]), range(8)):
            for D in itertools.imap(pow, itertools.cycle([10]), range(3)):
                X = np.random.rand(n, D).astype(np.float32)
                label = (np.random.rand(n) * D).astype(np.int32)
                logger.info('Running benchmark for n = {}, D= {}'.format(n, D))
                ret = runOpBenchmark(gpu_do, op, inputs=[X, label])
                self.results.append(((n, D), ret[1]))


def parse_args():
    parser = argparse.ArgumentParser(os.path.basename(__file__))
    parser.add_argument('-b', '--benchmarks', nargs='+',
                        default=ALL_BENCHMARKS.keys(),
                        help='benchmarks to run (default: %(default)s))')
    return parser.parse_args()


def main():
    args = parse_args()

    benchmarks = [ALL_BENCHMARKS[name]() for name in args.benchmarks]
    for bench in benchmarks:
        bench.run()
    for bench in benchmarks:
        bench.display()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()
