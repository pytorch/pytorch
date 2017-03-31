## @package lstm_benchmark
# Module caffe2.python.lstm_benchmark
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
from caffe2.python import cnn, workspace, core, utils, recurrent

import argparse
import numpy as np
import time

import logging

logging.basicConfig()
log = logging.getLogger("lstm_bench")
log.setLevel(logging.DEBUG)


def generate_data(T, shape, num_labels):
    '''
    Fill a queue with input data
    '''
    log.info("Generating T={} sequence batches".format(T))

    generate_input_init_net = core.Net('generate_input_init')
    queue = generate_input_init_net.CreateBlobsQueue(
        [], "inputqueue", num_blobs=1, capacity=T,
    )
    label_queue = generate_input_init_net.CreateBlobsQueue(
        [], "labelqueue", num_blobs=1, capacity=T,
    )

    workspace.RunNetOnce(generate_input_init_net)
    generate_input_net = core.Net('generate_input')
    generate_input_net.EnqueueBlobs([queue, "scratch"], ["scratch"])
    generate_input_net.EnqueueBlobs([label_queue, "label_scr"], ["label_scr"])
    np.random.seed(2603)

    for t in range(T):
        if (t % 50 == 0):
            print("Generating data {}/{}".format(t, T))
        # Randomize the seqlength
        random_shape = [np.random.randint(1, shape[0])] + shape[1:]
        X = np.random.rand(*random_shape).astype(np.float32)
        batch_size = random_shape[1]
        L = num_labels * batch_size
        labels = (np.random.rand(random_shape[0]) * L).astype(np.int32)
        workspace.FeedBlob("scratch", X)
        workspace.FeedBlob("label_scr", labels)
        workspace.RunNetOnce(generate_input_net.Proto())
    log.info("Finished data generation")
    return queue, label_queue


def create_model(args, queue, label_queue):
    model = cnn.CNNModelHelper(name="LSTM_bench")
    seq_lengths, hidden_init, cell_init, target = \
        model.net.AddExternalInputs(
            'seq_lengths',
            'hidden_init',
            'cell_init',
            'target',
        )
    input_blob = model.DequeueBlobs(queue, "input_data")
    labels = model.DequeueBlobs(label_queue, "label")
    all_hidden, last_hidden, _, last_state = recurrent.LSTM(
        model=model,
        input_blob=input_blob,
        seq_lengths=seq_lengths,
        initial_states=(hidden_init, cell_init),
        dim_in=args.input_dim,
        dim_out=args.hidden_dim,
        scope="lstm1",
    )
    weights = model.UniformFill(labels, "weights")

    softmax, loss = model.SoftmaxWithLoss(
        [model.Flatten(all_hidden), labels, weights],
        ['softmax', 'loss'],
    )

    model.AddGradientOperators([loss])

    # carry states over
    model.net.Copy(last_hidden, hidden_init)
    model.net.Copy(last_hidden, cell_init)

    workspace.FeedBlob(hidden_init, np.zeros(
        [1, args.batch_size, args.hidden_dim], dtype=np.float32
    ))
    workspace.FeedBlob(cell_init, np.zeros(
        [1, args.batch_size, args.hidden_dim], dtype=np.float32
    ))
    return model


def Caffe2LSTM(args):
    T = args.data_size // args.batch_size
    input_blob_shape = [args.seq_length, args.batch_size, args.input_dim]
    queue, label_queue = generate_data(T // args.seq_length,
                                       input_blob_shape,
                                       args.hidden_dim)

    workspace.FeedBlob(
        "seq_lengths",
        np.array([args.seq_length] * args.batch_size, dtype=np.int32)
    )

    model = create_model(args, queue, label_queue)

    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)

    last_time = time.time()
    start_time = last_time
    num_iters = T // args.seq_length
    entries_per_iter = args.seq_length * args.batch_size

    # Run the Benchmark
    log.info("------ Starting benchmark ------")
    for iteration in range(0, num_iters, args.iters_to_report):
        iters_once = min(args.iters_to_report, num_iters - iteration)
        workspace.RunNet(model.net.Proto().name, iters_once)
        new_time = time.time()
        log.info("Iter: {} / {}. Entries Per Second: {}k". format(
            iteration,
            num_iters,
            entries_per_iter * iters_once / (new_time - last_time) // 1000
        ))
        last_time = new_time

    log.info("Done. Total EPS: {}k".format(
        entries_per_iter * num_iters / (time.time() - start_time) // 1000
    ))


@utils.debug
def Benchmark(args):
    Caffe2LSTM(args)


def GetArgumentParser():
    parser = argparse.ArgumentParser(description="LSTM benchmark.")

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=40,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=40,
        help="Input dimension",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="The batch size."
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=20,
        help="Max sequence length"
    )
    parser.add_argument(
        "--data_size",
        type=int,
        default=10000000,
        help="Number of data points to generate"
    )
    parser.add_argument(
        "--iters_to_report",
        type=int,
        default=100,
        help="Number of iteration to report progress"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run all on GPU",
    )

    return parser


if __name__ == '__main__':
    args = GetArgumentParser().parse_args()

    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])

    device = core.DeviceOption(
        caffe2_pb2.CUDA if args.gpu else caffe2_pb2.CPU, 0)

    with core.DeviceScope(device):
        Benchmark(args)
