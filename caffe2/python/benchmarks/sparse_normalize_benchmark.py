import argparse
import datetime

# import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, workspace


def benchmark_sparse_normalize(
    categorical_limit,
    embedding_size,
    average_len,
    batch_size,
    iterations,
    flush_cache,
    fp16,
):
    print("Preparing lookup table. " + str(datetime.datetime.now()))

    # We will use a constant, but non-trivial value so we save initialization
    # time.
    data = np.ones([categorical_limit, embedding_size], dtype=np.float32)
    data *= 17.01

    init_net = core.Net("init_net")
    if fp16:
        op = core.CreateOperator("FloatToHalf", "X", "X_fp16")
        init_net.Proto().op.extend([op])
    l3_cache_size = 30 * 2 ** 20 // 4

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

    workspace.FeedBlob("X", data)
    workspace.FeedBlob("huge_blob", np.random.randn(l3_cache_size).astype(np.float32))

    print("Data has shape {} {}".format(data.shape, datetime.datetime.now()))

    init_net.Python(f)([], ["indices"])
    workspace.RunNetOnce(init_net)

    net = core.Net("mynet")
    op = core.CreateOperator(
        "Float16SparseNormalize" if fp16 else "SparseNormalize",
        ["X_fp16", "indices"] if fp16 else ["X", "indices"],
        "X_fp16" if fp16 else "X",
    )
    net.Proto().external_input.append("X")
    net.Proto().external_input.append("X_fp16")
    net.Proto().external_input.append("indices")
    net.Proto().op.extend([op])
    if flush_cache:
        net.Scale("huge_blob", "huge_blob_2x", value=2.0)

    workspace.CreateNet(net)

    # Set random seed, so that repeated runs will keep the same sequence of
    # random indices.
    np.random.seed(1701)

    print("Preparation finished. " + str(datetime.datetime.now()))

    runtimes = workspace.BenchmarkNet(net.Name(), 1, iterations, True)

    print("{} ms".format(runtimes[2 if flush_cache else 1]))
    print("indice_size: " + str(workspace.FetchBlob("indices").size))
    print(
        "{} GB/sec".format(
            (2 if fp16 else 4)
            * embedding_size
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
        "-e", "--embedding-size", type=int, default=600000, help="Lookup table size."
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=128, help="Embedding dimension."
    )
    parser.add_argument(
        "--average-len",
        type=int,
        default=27,
        help="Sparse feature average lengths, default is 27",
    )
    parser.add_argument("--batch_size", type=int, default=100, help="The batch size.")
    parser.add_argument(
        "-i", "--iteration", type=int, default=100, help="The number of iterations."
    )
    parser.add_argument(
        "--flush-cache", action="store_true", help="If true, flush cache"
    )
    parser.add_argument("--fp16", action="store_true", help="If true, use fp16")
    args, extra_args = parser.parse_known_args()
    core.GlobalInit(["python"] + extra_args)

    benchmark_sparse_normalize(
        args.embedding_size,
        args.embedding_dim,
        args.average_len,
        args.batch_size,
        args.iteration,
        args.flush_cache,
        args.fp16,
    )
