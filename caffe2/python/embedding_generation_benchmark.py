## @package embedding_generation_benchmark
# Module caffe2.python.embedding_generation_benchmark





from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, core, utils, model_helper

import argparse
import numpy as np
import time

import logging

logging.basicConfig()
log = logging.getLogger("embedding_generation_benchmark")
log.setLevel(logging.DEBUG)


def generate_data(T, batch_size, max_seq_length):
    '''
    Fill a queue with input data
    '''
    log.info("Generating T={} batches".format(T))

    generate_input_init_net = core.Net('generate_input_init')
    queue = generate_input_init_net.CreateBlobsQueue(
        [], "inputqueue", num_blobs=1, capacity=T,
    )
    workspace.RunNetOnce(generate_input_init_net)

    generate_input_net = core.Net('generate_input')
    generate_input_net.EnqueueBlobs([queue, "scratch"], ["scratch"])
    np.random.seed(2603)

    for t in range(T):
        if (t % (max(10, T // 10)) == 0):
            log.info("Generating data {}/{}".format(t, T))
        X = np.tile(np.arange(max_seq_length), [batch_size, 1]).transpose()
        workspace.FeedBlob("scratch", X)
        workspace.RunNetOnce(generate_input_net.Proto())

    log.info("Finished data generation")
    return queue


def generate_embedding_table(vocab_size, embedding_size):
    log.info("Generating embedding table with dimensions {}"
             .format([vocab_size, embedding_size]))

    generate_table_net = core.Net('generate_table')
    table = generate_table_net.GaussianFill(
        [],
        ['embedding_table'],
        shape=[vocab_size, embedding_size],
    )

    workspace.RunNetOnce(generate_table_net)
    return table


def create_model(args, queue, embedding_table, embedding_size):
    model = model_helper.ModelHelper(name='embedding_generation_bench')
    input_blob = model.net.DequeueBlobs(queue, 'input_data')

    if args.implementation == 'sinusoid':
        model.net.SinusoidPositionEncoding(
            [input_blob],
            ['output'],
            embedding_size=embedding_size
        )
    else:
        model.net.Gather(
            [embedding_table, input_blob],
            ['output'],
        )

    return model


def Caffe2EmbeddingGeneration(args):
    T = args.data_size // args.batch_size

    queue = generate_data(T, args.batch_size, args.seq_length)

    embedding_table = None
    if args.implementation == 'table':
        embedding_table = generate_embedding_table(
            args.seq_length,
            args.embedding_size,
        )

    model = create_model(args, queue, embedding_table, args.embedding_size)

    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)

    start_time = time.time()
    num_iters = T
    total_iters = 0

    # Run the Benchmark
    log.info("------ Warming up ------")
    workspace.RunNet(model.net.Proto().name)

    log.info("------ Starting benchmark ------")
    start_time = time.time()
    last_time = time.time()
    for iteration in range(1, num_iters, args.iters_to_report):
        iters_once = min(args.iters_to_report, num_iters - iteration)
        total_iters += iters_once
        workspace.RunNet(model.net.Proto().name, iters_once)

        new_time = time.time()
        log.info(
            "Iter: {} / {}. Embeddings Generated Per Second: {}k.".format(
                iteration,
                num_iters,
                (iters_once * args.batch_size * args.seq_length) /
                (new_time - last_time) // 100 / 10,
            )
        )
        last_time = new_time

    total_per_sec = (num_iters - 1) * args.batch_size * args.seq_length
    total_per_sec = total_per_sec / (time.time() - start_time) // 100 / 10

    log.info("Done. Total embeddings generated per second " +
             "excluding 1st iteration: {}k".format(total_per_sec))

    return time.time() - start_time


@utils.debug
def Benchmark(args):
    return Caffe2EmbeddingGeneration(args)


def GetArgumentParser():
    parser = argparse.ArgumentParser(
        description="Embedding generation benchmark."
    )

    parser.add_argument(
        "--embedding_size",
        type=int,
        default=512,
        help="Embedding size",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="The batch size."
    )
    parser.add_argument(
        "--data_size",
        type=int,
        default=10000,
        help="Number of sequences to generate"
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=128,
        help="Max sequence length"
    )
    parser.add_argument(
        "--iters_to_report",
        type=int,
        default=20,
        help="Number of iterations to report progress"
    )
    parser.add_argument(
        "--implementation",
        type=str,
        default="sinusoid",
        help="'table' or 'sinusoid'",
    )
    return parser


if __name__ == '__main__':
    args, extra_args = GetArgumentParser().parse_known_args()

    workspace.GlobalInit([
        'caffe2',
        '--caffe2_log_level=0',
        '--caffe2_print_blob_sizes_at_exit=0'] + extra_args)

    device = core.DeviceOption(caffe2_pb2.CPU)

    with core.DeviceScope(device):
        Benchmark(args)
