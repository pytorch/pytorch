import argparse

import numpy as np
from caffe2.python import core, workspace


def benchmark_concat(num_inputs, input_dim, axis, add_axis, iterations):
    input_names = [f"input{i}" for i in range(num_inputs)]
    for n in input_names:
        workspace.FeedBlob(n, np.random.randn(*input_dim).astype(np.float32))

    net = core.Net("benchmark_net")
    net.Concat(input_names, ["output", "split_info"], axis=axis, add_axis=add_axis)
    workspace.CreateNet(net)

    runtimes = workspace.BenchmarkNet(net.Name(), 1, iterations, True)
    print(f"{num_inputs * np.prod(input_dim) * 4 / runtimes[1] / 1e6} GB/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="minimal benchmark for concat.")
    parser.add_argument("--num_inputs", type=int, default=2)
    parser.add_argument("--input_dim", nargs="+", type=int, required=True)
    parser.add_argument("--axis", type=int, default=-1)
    parser.add_argument("--add_axis", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=64)
    args, extra_args = parser.parse_known_args()
    core.GlobalInit(["python"] + extra_args)
    benchmark_concat(
        args.num_inputs, args.input_dim, args.axis, args.add_axis, args.iterations
    )
