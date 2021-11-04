




import argparse
import numpy as np

from caffe2.python import core, workspace


def benchmark_mul_gradient(args):
    workspace.FeedBlob("dC", np.random.rand(args.m, args.n).astype(np.float32))
    workspace.FeedBlob("A", np.random.rand(args.m, args.n).astype(np.float32))
    workspace.FeedBlob("B", np.random.rand(args.n).astype(np.float32))

    net = core.Net("mynet")
    net.MulGradient(
        ["dC", "A", "B"],
        ["dC" if args.inplace else "dA", "dB"],
        broadcast=True,
        axis=1,
        allow_broadcast_fastpath=args.allow_broadcast_fastpath,
    )
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
    parser.add_argument(
        "--inplace",
        action='store_true', help="Whether to perform the op in-place.")
    parser.add_argument(
        "--allow-broadcast-fastpath",
        action='store_true', help="Whether the broadcast fastpath is enabled.")
    args, extra_args = parser.parse_known_args()
    core.GlobalInit(['python'] + extra_args)
    benchmark_mul_gradient(args)
