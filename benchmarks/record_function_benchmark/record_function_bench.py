import argparse
from functools import partial
import statistics
import sys
import timeit
import torch

from benchmarks.fastrnns.factory import lstm_creator
from torchvision.models import resnet50

def prepare_lstm_jit(bench_args):
    model_def = lstm_creator(
        script=True,
        seqLength=bench_args.lstmSeqLength,
        numLayers=bench_args.lstmNumLayers,
        inputSize=bench_args.lstmInputSize,
        hiddenSize=bench_args.lstmHiddenSize,
        miniBatch=bench_args.lstmMiniBatch,
        device='cpu')
    return model_def.inputs, model_def.forward

def prepare_resnet50_jit(bench_args):
    model = resnet50()
    inputs = (torch.randn(32, 3, 224, 224),)
    model = torch.jit.trace(model, inputs)
    return inputs, model

MODELS = {
    'lstm_jit' : prepare_lstm_jit,
    'resnet50_jit' : prepare_resnet50_jit
}

def run_bench(model_name, bench_args):
    model_creator = MODELS[model_name]
    inputs, model = model_creator(bench_args)

    print("Benchmarking RecordFunction overhead for", model_name)
    print("Running warmup...", end=" ")
    for _ in range(bench_args.warmup):
        model(*inputs)
    print("finished")

    for with_rec_fn in [True, False]:
        torch.autograd._enable_record_function(with_rec_fn)
        torch.autograd._clear_callbacks()
        if with_rec_fn:
            torch.autograd._set_empty_test_observer(True, 0.0001)
        print("Running {} iterations {} RecordFunction...".format(
            bench_args.nloops, "with" if with_rec_fn else "without"), end=" ")
        sys.stdout.flush()
        runtimes = timeit.repeat(
            partial(model, *inputs),
            repeat=bench_args.nloops,
            number=1)
        print("finished")
        avg_time = statistics.mean(runtimes) * 1000.0
        stddev_time = statistics.stdev(runtimes) * 1000.0
        print("N = {}, avg. time: {:.3f} ms, stddev: {:.3f} ms".format(
            bench_args.nloops, avg_time, stddev_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark RecordFunction overhead for ResNet and LSTM models')

    parser.add_argument('--models', nargs='*',
                        help='What model to run: ' + str(MODELS))

    parser.add_argument('--lstmSeqLength', default='100', type=int)
    parser.add_argument('--lstmNumLayers', default='1', type=int)
    parser.add_argument('--lstmInputSize', default='512', type=int)
    parser.add_argument('--lstmHiddenSize', default='512', type=int)
    parser.add_argument('--lstmMiniBatch', default='64', type=int)
    parser.add_argument('--warmup', default='10', type=int)
    parser.add_argument('--nloops', default='100', type=int)

    args = parser.parse_args()

    models = args.models or MODELS.keys()

    for model in models:
        assert model in MODELS
        run_bench(model, args)
