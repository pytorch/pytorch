

import argparse
import datetime

import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, workspace


def benchmark_sparse_lengths_sum(
    categorical_limit,
    embedding_size,
    average_len,
    batch_size,
    iterations,
    flush_cache,
    bit_rate=st.sampled_from([2, 4]),
):
    print("Preparing lookup table. " + str(datetime.datetime.now()))

    # We will use a constant, but non-trivial value so we save initialization
    # time.
    data = np.ones([categorical_limit, embedding_size], dtype=np.float32)
    data *= 17.01

    init_net = core.Net("init_net")
    op = core.CreateOperator(
        "FloatToFused" + str(bit_rate) + "BitRowwiseQuantized", "X", "X_q"
    )
    init_net.Proto().op.extend([op])
    workspace.FeedBlob("X", data)

    print("Data has shape {} {}".format(data.shape, datetime.datetime.now()))

    # In order to produce truly random lengths and indices, we will embed a
    # Python operator in the net to generate them.
    def f(_, outputs):
        lengths = np.random.randint(
            int(average_len * 0.75), int(average_len * 1.25), batch_size
        ).astype(np.int32)
        indices = np.random.randint(0, categorical_limit, np.sum(lengths)).astype(
            np.int64
        )
        outputs[0].feed(indices)
        outputs[1].feed(lengths)

    init_net.Python(f)([], ["indices", "lengths"])
    workspace.RunNetOnce(init_net)

    net = core.Net("mynet")
    if flush_cache:
        l3_cache_size = 30 * 2 ** 20 // 4
        workspace.FeedBlob(
            "huge_blob", np.random.randn(l3_cache_size).astype(np.float32)
        )
        net.Scale("huge_blob", "huge_blob_2x", value=2.0)
    op = core.CreateOperator(
        "SparseLengthsSumFused" + str(bit_rate) + "BitRowwise",
        ["X_q", "indices", "lengths"],
        "Y",
    )
    net.Proto().op.extend([op])
    workspace.CreateNet(net)

    # Set random seed, so that repeated runs will keep the same sequence of
    # random indices.
    np.random.seed(1701)

    print("Preparation finished. " + str(datetime.datetime.now()))

    runtimes = workspace.BenchmarkNet(net.Name(), 1, iterations, True)
    print(
        "{} billion sums per sec".format(
            embedding_size
            * workspace.FetchBlob("indices").size
            / runtimes[2 if flush_cache else 1]
            / 1e6
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="minimal benchmark for sparse lengths sum."
    )
    parser.add_argument(
        "-e", "--embedding-size", type=int, default=6000000, help="Lookup table size."
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=128, help="Embedding dimension."
    )
    parser.add_argument(
        "--average_len",
        type=int,
        default=27,
        help="Sparse feature average lengths, default is 27",
    )
    parser.add_argument("--batch_size", type=int, default=100, help="The batch size.")
    parser.add_argument(
        "-i", "--iteration", type=int, default=100000, help="The number of iterations."
    )
    parser.add_argument(
        "--flush-cache", action="store_true", help="If true, flush cache"
    )
    parser.add_argument("--bit-rate", type=int, default=4)
    args, extra_args = parser.parse_known_args()
    core.GlobalInit(["python"] + extra_args)
    benchmark_sparse_lengths_sum(
        args.embedding_size,
        args.embedding_dim,
        args.average_len,
        args.batch_size,
        args.iteration,
        args.flush_cache,
        args.bit_rate,
    )
