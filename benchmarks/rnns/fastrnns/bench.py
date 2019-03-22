import argparse
from collections import namedtuple
import torch
import gc
import sys
import json
import copy

from .runner import get_nn_runners


BenchResult = namedtuple('BenchResult', [
    'name', 'avg_fwd', 'std_fwd', 'avg_bwd', 'std_bwd',
])


def fit_str(string, colwidth=16):
    if len(string) < colwidth:
        return (colwidth - len(string)) * ' ' + string
    else:
        return string[:colwidth]


def to_str(item):
    if isinstance(item, float):
        return '%.4g' % item
    return str(item)


def print_header(colwidth=16, sep=' '):
    items = []
    for item in BenchResult._fields:
        items.append(fit_str(item))
    return sep.join(items)


def pretty_print(benchresult, colwidth=16, sep=' '):
    items = []
    for thing in benchresult:
        items.append(fit_str(to_str(thing)))
    return sep.join(items)


def trainbench(name, rnn_creator, nloops=100, warmup=10,
               seqLength=100, numLayers=1, inputSize=512, hiddenSize=512,
               miniBatch=64, device='cuda', seed=None):
    def train_batch(modeldef):
        # CUDA events for timing
        fwd_start_event = torch.cuda.Event(enable_timing=True)
        fwd_end_event = torch.cuda.Event(enable_timing=True)
        bwd_start_event = torch.cuda.Event(enable_timing=True)
        bwd_end_event = torch.cuda.Event(enable_timing=True)

        gc.collect()

        fwd_start_event.record()
        forward_output = modeldef.forward(*modeldef.inputs)
        fwd_end_event.record()

        # XXX: Use if need to print something
        # print(modeldef.forward.graph_for(*modeldef.inputs))

        if modeldef.backward_setup is not None:
            backward_input = modeldef.backward_setup(forward_output)
        else:
            backward_input = forward_output

        gc.collect()

        bwd_start_event.record()
        if modeldef.backward is not None:
            modeldef.backward(*backward_input)
        bwd_end_event.record()

        if modeldef.backward is not None:
            for param in modeldef.params:
                assert param.grad is not None
                param.grad.data.zero_()

        torch.cuda.synchronize()

        fwd_time = fwd_start_event.elapsed_time(fwd_end_event)
        bwd_time = bwd_start_event.elapsed_time(bwd_end_event)
        return fwd_time, bwd_time

    assert device == 'cuda'
    creator_args = dict(seqLength=seqLength, numLayers=numLayers,
                        inputSize=inputSize, hiddenSize=hiddenSize,
                        miniBatch=miniBatch, device=device, seed=seed)
    modeldef = rnn_creator(**creator_args)

    [train_batch(modeldef) for _ in range(warmup)]

    results = [train_batch(modeldef) for _ in range(nloops)]
    fwd_times, bwd_times = zip(*results)

    fwd_times = torch.tensor(fwd_times)
    bwd_times = torch.tensor(bwd_times)

    return BenchResult(name=name,
                       avg_fwd=fwd_times.mean().item(),
                       std_fwd=fwd_times.std().item(),
                       avg_bwd=bwd_times.mean().item(),
                       std_bwd=bwd_times.std().item())


def print_stderr(*args, **kwargs):
    return print(*args, **kwargs, file=sys.stderr)


def bench(rnn_runners, group_name, print_json=False, sep=' ', **params):
    print_stderr(print_header(sep=sep))
    results = {}
    for name, creator, context in rnn_runners:
        with context():
            try:
                result = trainbench(name, creator, **params)
                print_stderr(pretty_print(result, sep=sep))
                results[name] = result
            except Exception as e:
                if not print_json:
                    raise

    return {
        group_name: {k: v.avg_fwd for k, v in results.items()},
        group_name + '-backward': {k: v.avg_bwd for k, v in results.items()},
    }


def bench_group(model_list, bench_name, bench_group, bench_args):
    print_stderr('Benchmarking {}s...'.format(bench_name))
    nn_results = bench(get_nn_runners(*model_list), bench_group, **bench_args)
    print_stderr('')
    return nn_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Profile RNNs')

    # groups help control which test group you want to run
    # if you only want to run one/two benchmark, run it with
    # e.g: python -m fastrnns.bench --rnns jit and --group rnns
    default_groups = ['cnns', 'rnns']

    parser.add_argument('--seqLength', default='100', type=int)
    parser.add_argument('--numLayers', default='1', type=int)
    parser.add_argument('--inputSize', default='512', type=int)
    parser.add_argument('--hiddenSize', default='512', type=int)
    parser.add_argument('--miniBatch', default='64', type=int)
    parser.add_argument('--warmup', default='10', type=int)
    parser.add_argument('--nloops', default='100', type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--variable_lstms', action='store_true',
                        help='Also benchmark variable sequence length lstms '
                        'Note that some of these run really slowly '
                        'and that the `seqLength` flag will be ignored.')
    parser.add_argument('--sep', default=' ', type=str)
    parser.add_argument('--print-json', action='store_true')
    parser.add_argument('--rnns', nargs='*',
                        help='What to run. cudnn, aten, jit, etc')
    parser.add_argument('--cnns', nargs='*',
                        help='What to run. resnet18, resnet18_jit, resnet50, etc')
    parser.add_argument('--group', nargs='*', default=default_groups, help='Which group to run. cnns, rnns, etc.')

    args = parser.parse_args()
    rnns = args.rnns or ['cudnn', 'aten', 'jit', 'jit_premul', 'jit_simple',
                         'jit_multilayer', 'py']
    cnns = args.cnns or ['resnet18', 'resnet18_jit', 'resnet50', 'resnet50_jit']
    # TODO: Maybe add a separate section for the layernorm/dropout lstms
    # 'cudnn_layernorm', jit_layernorm', 'jit_layernom_decom',
    # 'jit', 'jit_dropout', 'cudnn_dropout'
    vlrnns = ['vl_cudnn', 'vl_jit', 'vl_py']

    if args.print_json:
        print_stderr = lambda *args, **kwargs: None    # noqa
    print_stderr(args)

    bench_args = copy.deepcopy(vars(args))
    should_bench_varlen_lstms = args.variable_lstms
    del bench_args['group']
    del bench_args['rnns']
    del bench_args['cnns']
    del bench_args['variable_lstms']

    results = dict()
    if should_bench_varlen_lstms:
        if args.nloops + args.warmup > 30:
            print_stderr(
                'WARNING: some of the variable sequence length lstms are '
                'very unoptimized and therefore take forever to run.')
        results.update(bench_group(vlrnns, 'variable-length sequence LSTM', 'vl_lstm', bench_args))

    if 'rnns' in args.group:
        results.update(bench_group(rnns, 'LSTM', 'lstm', bench_args))
    if 'cnns' in args.group:
        results.update(bench_group(cnns, 'ResNet', 'resnet', bench_args))

    if args.print_json:
        print(json.dumps(results))
