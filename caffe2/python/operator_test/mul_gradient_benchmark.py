from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np

from caffe2.python import core, workspace


def benchmark_mul_gradient(args):
    workspace.FeedBlob("dC", np.random.rand(args.m, args.n).astype(np.float32))
    workspace.FeedBlob("A", np.random.rand(args.m, args.n).astype(np.float32))
    workspace.FeedBlob("B", np.random.rand(args.m).astype(np.float32))

    net = core.Net("mynet")
    net.MulGradient(["dC", "A", "B"], ["dA", "dB"], broadcast=True, axis=0)
    workspace.CreateNet(net)

    workspace.BenchmarkNet(net.Name(), 1, args.iteration, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="benchmark for MulGradient.")
    parser.add_argument(
        '-m', type=int, default=9508,
        help="The number of rows of A")
    parser.add_argument(
        "-n", type=int, default=80,
        help="The number of columns of A")
    parser.add_argument(
        '-i', "--iteration", type=int, default=100,
        help="The number of iterations.")
    args, extra_args = parser.parse_known_args()
    core.GlobalInit(['python'] + extra_args)
    benchmark_mul_gradient(args)
