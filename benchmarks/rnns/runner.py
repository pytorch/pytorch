from __future__ import print_function
import inspect
from itertools import product

from benchmarks.memnn import run_memnn
from benchmarks.mlstm import run_mlstm
from benchmarks.lstm import run_lstm
from benchmarks.cudnn_lstm import run_cudnn_lstm
from benchmarks.tensor import run_tensor
from benchmarks.lstm_variants_test import run_lstm_variant
from benchmarks.bnlstm import run_bnlstm
from benchmarks.sru_test import run_sru
from benchmarks.qrnn import run_qrnn

from benchmarks.sequence_labeler import test_wsj
from benchmarks.sequence_labeler import Example

from benchmarks.common import AttrDict


class over(object):
    def __init__(self, *args):
        self.values = args


def make_params(**kwargs):
    keys = list(kwargs.keys())
    iterables = [kwargs[k].values if isinstance(kwargs[k], over) else (kwargs[k],) for k in keys]
    all_values = list(product(*iterables))
    param_dicts = [AttrDict({k: v for k, v in zip(keys, values)}) for values in all_values]
    return [param_dicts]


def skip(fn):
    def wrapper(*args, **kwargs):
        return [None]


class Benchmarks(object):

    # @skip
    def sequence_labeler(self):
        params = make_params(cuda=over(False, True), jit=over(False, True))[0]
        return [test_wsj(**p) for p in params]

    # @skip
    def lstm_one_layer_train_cuda(self):
        return [
            run_lstm(seq_len=256, batch_size=32, backward=True, fused=True),
            run_lstm(seq_len=256, batch_size=32, backward=True, jit=True),
            run_cudnn_lstm(seq_len=256, batch_size=32, backward=True),
        ]

    # @skip
    def mlstm_forward_cuda(self):
        return [run_mlstm(autograd=True), run_mlstm(jit=True)]

    # @skip
    def tensor_mul_broadcast(self):
        # Basic example
        return [
            run_tensor(broadcast=True),
            run_tensor(broadcast=False),
        ]

    # @skip
    def memnn(self):
        # This doesn't actually use a RNN. (in its current form, at least)
        # CUDA is much slower than CPU for memnn (CPU is still slow).
        # My guess is that the code wasn't optmized for CUDA
        return [
            run_memnn(warmup=5, benchmark=15, cuda=False),
            run_memnn(warmup=5, benchmark=15, cuda=True),
        ]

    # @skip
    def bnlstm(self):
        # Slow on CPU
        return [
            run_bnlstm(cuda=True, jit=True, num_batches=9),
            run_bnlstm(cuda=True, num_batches=9),
        ]

    # @skip
    def lstm_variant(self):
        # Slow on CPU
        variants = over(
            'SlowLSTM',
            'LSTM',
            'GalLSTM',
            'MoonLSTM',
            'SemeniutaLSTM',
            'LayerNormLSTM',
            'LayerNormGalLSTM',
            'LayerNormMoonLSTM',
            'LayerNormSemeniutaLSTM',
        )

        params = make_params(variant=variants, cuda=True)[0]
        params.extend(make_params(variant='SlowLSTM', cuda=True, jit=True)[0])
        return [run_lstm_variant(**p) for p in params]

    # @skip
    def sru(self):
        params = make_params(backward=over(False, True),
                             use_kernel=over(False, True))[0]
        params_jit = make_params(backward=over(False, True), jit=True)[0]
        params.extend(params_jit)
        return [run_sru(**p) for p in params]

    # @skip
    def qrnn(self):
        params = make_params(cuda=True,
                             use_kernel=over(False, True))[0]
        params_jit = make_params(cuda=True, jit=True)[0]
        params.extend(params_jit)
        return [run_qrnn(**p) for p in params]


def discover_benchmarks():
    benchmarks = Benchmarks()
    return inspect.getmembers(benchmarks, predicate=inspect.ismethod)


def title(text='title', width=80):
    reserve = len(text) + 2
    num_lines = int((width - reserve) / 2)
    lines = '-' * num_lines
    return '{} {} {}'.format(lines, text, lines)


def summarize(result):
    if result is None:
        return 'ERR [{0}] '.format(result.name)
    gpu_summary, cpu_summary = result.summary()
    samples = result.results.__len__() - result.warmup_iters
    use_summary = gpu_summary
    if gpu_summary.max == 0 and gpu_summary.min == 0:
        use_summary = cpu_summary

    range_middle = (use_summary.max + use_summary.min) / 2
    deviation = use_summary.max - range_middle

    return '{2:10.4f} ± {3:8.4f} msec (average {1:10.4f} msec, {4} samples) [{0}]'.format(
        result.name, use_summary.mean, range_middle, deviation, samples)


def main():
    timing_fns = discover_benchmarks()
    results = []
    for name, fn in timing_fns:
        try:
            print(title(name))
            result = fn()
            results.append((name, result))
        except RuntimeError as err:
            print(err)
            results.append((name, None))

    print(title('summary'))
    for name, benches in results:
        print('')
        print(name)
        for result in benches:
            print(summarize(result))


if __name__ == '__main__':
    main()
