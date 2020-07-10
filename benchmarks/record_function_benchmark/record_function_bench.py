import argparse
from functools import partial
import statistics
import sys
import timeit
import torch

try:
    from benchmarks.fastrnns.factory import lstm_creator
except ImportError:
    from caffe2.benchmarks.fastrnns.factory import lstm_creator

try:
    from benchmarks.experimental_components.utils import Timer, Compare
    HAS_TIMER = True
except ImportError:
    HAS_TIMER = False

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
    'resnet50_jit' : prepare_resnet50_jit,
    'lstm_jit' : prepare_lstm_jit,
}

NUM_THREADS = [1, 2, 4, 8, 16, 32]

def run_bench(model_names, bench_args):
    results = []
    for model_name in model_names:
        model_creator = MODELS[model_name]
        inputs, model = model_creator(bench_args)

        print("Benchmarking RecordFunction overhead for", model_name)
        print("Running warmup...", end=" ")
        sys.stdout.flush()
        for _ in range(bench_args.warmup):
            model(*inputs)
        print("finished")

        for num_threads in NUM_THREADS:
            for with_rec_fn in [True, False]:
                torch.autograd._enable_record_function(with_rec_fn)
                torch.autograd._clear_callbacks()
                if with_rec_fn:
                    torch.autograd._set_empty_test_observer(True, 0.0001)

                if bench_args.use_timer and HAS_TIMER:
                    print("Running {} RecordFunction, num threads {} ...".format(
                        "with" if with_rec_fn else "without", num_threads), end=" ")
                    sys.stdout.flush()
                    timer = Timer(
                        stmt="model(*inputs)",
                        globals={"model": model, "inputs": inputs},
                        description=model_name,
                        label="Record function overhead",
                        sub_label=f"with{'' if with_rec_fn else 'out'}_rec_fn, num_threads {num_threads}",
                        num_threads=num_threads)
                    result = timer.blocked_autorange(min_run_time=bench_args.timer_min_run_time)
                    print("finished")
                    print(result)
                    sys.stdout.flush()
                    results.append(result)
                else:
                    print("Running {} iterations {} RecordFunction, num threads {} ...".format(
                        bench_args.nloops, "with" if with_rec_fn else "without", num_threads), end=" ")
                    sys.stdout.flush()
                    torch.set_num_threads(num_threads)
                    runtimes = timeit.repeat(
                        partial(model, *inputs),
                        repeat=bench_args.nloops,
                        number=1)
                    print("finished")
                    avg_time = statistics.mean(runtimes) * 1000.0
                    stddev_time = statistics.stdev(runtimes) * 1000.0
                    print("N = {}, avg. time: {:.3f} ms, stddev: {:.3f} ms".format(
                        bench_args.nloops, avg_time, stddev_time))
                    sys.stdout.flush()

    if bench_args.use_timer and HAS_TIMER:
        comparison = Compare(results)
        comparison.trim_significant_figures()
        comparison.highlight_warnings()
        comparison.print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark RecordFunction overhead for ResNet and LSTM models')

    parser.add_argument('--models', nargs='*', default=['lstm_jit'],
                        help='What model to run: ' + str(MODELS.keys()))

    parser.add_argument('--lstmSeqLength', default='100', type=int)
    parser.add_argument('--lstmNumLayers', default='1', type=int)
    parser.add_argument('--lstmInputSize', default='512', type=int)
    parser.add_argument('--lstmHiddenSize', default='512', type=int)
    parser.add_argument('--lstmMiniBatch', default='64', type=int)
    parser.add_argument('--warmup', default='2', type=int)
    parser.add_argument('--nloops', default='50', type=int)
    parser.add_argument('--use_timer', default=True, type=bool)
    parser.add_argument('--timer_min_run_time', default=120, type=int)

    args = parser.parse_args()

    models = args.models or MODELS.keys()

    if args.use_timer and not HAS_TIMER:
        print("Warning: benchmark Timer not available")

    for model in models:
        assert model in MODELS
    run_bench(models, args)
