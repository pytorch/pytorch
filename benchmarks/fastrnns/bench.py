import argparse
import copy
import gc
import json
import sys
import time
from collections import namedtuple

import torch
from torch.autograd.profiler import record_function

from .fuser import set_fuser
from .runner import get_nn_runners


BenchResult = namedtuple(
    "BenchResult",
    [
        "name",
        "avg_fwd",
        "std_fwd",
        "info_fwd",
        "avg_bwd",
        "std_bwd",
        "info_bwd",
    ],
)


def fit_str(string, colwidth=16):
    if len(string) < colwidth:
        return (colwidth - len(string)) * " " + string
    else:
        return string[:colwidth]


def to_str(item):
    if isinstance(item, float):
        return f"{item:.4g}"
    return str(item)


def print_header(colwidth=16, sep=" "):
    items = []
    for item in BenchResult._fields:
        items.append(fit_str(item))
    return sep.join(items)


def pretty_print(benchresult, colwidth=16, sep=" "):
    items = []
    for thing in benchresult:
        items.append(fit_str(to_str(thing)))
    return sep.join(items)


# shim for torch.cuda.Event when running on cpu
class Event:
    def __init__(self, enable_timing):
        pass

    def record(self):
        self.time = time.perf_counter()

    def elapsed_time(self, end_event):
        assert isinstance(end_event, Event)
        return end_event.time - self.time


def trainbench(
    name,
    rnn_creator,
    nloops=100,
    warmup=10,
    seqLength=100,
    numLayers=1,
    inputSize=512,
    hiddenSize=512,
    miniBatch=64,
    device="cuda",
    seed=None,
):
    def train_batch(modeldef):
        # CUDA events for timing
        if device == "cuda":
            timer_class = torch.cuda.Event
        else:
            timer_class = Event

        fwd_start_event = timer_class(enable_timing=True)
        fwd_end_event = timer_class(enable_timing=True)
        bwd_start_event = timer_class(enable_timing=True)
        bwd_end_event = timer_class(enable_timing=True)

        gc.collect()

        fwd_start_event.record()
        with record_function("## forward ##"):
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
            with torch.no_grad():
                for param in modeldef.params:
                    assert param.grad is not None
                    param.grad.zero_()

        if device == "cuda":
            torch.cuda.synchronize()

        fwd_time = fwd_start_event.elapsed_time(fwd_end_event)
        bwd_time = bwd_start_event.elapsed_time(bwd_end_event)
        return fwd_time, bwd_time

    creator_args = creator_args = {
        "seqLength": seqLength,
        "numLayers": numLayers,
        "inputSize": inputSize,
        "hiddenSize": hiddenSize,
        "miniBatch": miniBatch,
        "device": device,
        "seed": seed,
    }

    modeldef = rnn_creator(**creator_args)

    [train_batch(modeldef) for _ in range(warmup)]

    results = [train_batch(modeldef) for _ in range(nloops)]
    fwd_times, bwd_times = zip(*results)

    fwd_times = torch.tensor(fwd_times)
    bwd_times = torch.tensor(bwd_times)
    return BenchResult(
        name=name,
        avg_fwd=fwd_times.mean().item(),
        std_fwd=fwd_times.std().item(),
        info_fwd=fwd_times,
        avg_bwd=bwd_times.mean().item(),
        std_bwd=bwd_times.std().item(),
        info_bwd=bwd_times,
    )


def print_stderr(*args, **kwargs):
    kwargs["file"] = sys.stderr
    return print(*args, **kwargs)


def print_json_oss_format(results):
    oss_results = {}
    for group_name, group_val in results.items():
        oss_results[group_name] = {}
        for model_name, run_time in group_val.items():
            # Output for OSS
            oss_results[group_name][model_name] = run_time["avg"]

    print(json.dumps(oss_results))


def print_json_pep_format(results):
    # print the AI-PEP format json string for each model
    for group_name, group_val in results.items():
        for model_name, run_time in group_val.items():
            # Output for AI-PEP
            num_iters = len(run_time["info"])
            info = run_time["info"].tolist()
            for i in range(num_iters):
                print(
                    "Caffe2Observer "
                    + json.dumps(
                        {
                            "type": "NET",
                            "metric": group_name + "-" + model_name,
                            "unit": "ms",
                            "value": str(info[i]),
                        }
                    )
                )


def bench(rnn_runners, group_name, print_json=False, sep=" ", **params):
    print_stderr(print_header(sep=sep))
    results = {}
    for name, creator, context in rnn_runners:
        with context():
            try:
                result = trainbench(name, creator, **params)
                # Replace the value of info_fwd and info_bwd to None
                result_with_no_info = result._replace(info_fwd="None", info_bwd="None")
                print_stderr(pretty_print(result_with_no_info, sep=sep))
                results[name] = result
            except Exception as e:
                if not print_json:
                    raise

    return {
        group_name: {
            k: {"avg": v.avg_fwd, "std": v.std_fwd, "info": v.info_fwd}
            for k, v in results.items()
        },
        group_name
        + "-backward": {
            k: {"avg": v.avg_bwd, "std": v.std_bwd, "info": v.info_bwd}
            for k, v in results.items()
        },
    }


def bench_group(model_list, bench_name, bench_group, bench_args):
    print_stderr(f"Benchmarking {bench_name}s...")
    nn_results = bench(get_nn_runners(*model_list), bench_group, **bench_args)
    print_stderr("")
    return nn_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile RNNs")

    # groups help control which test group you want to run
    # if you only want to run one/two benchmark, run it with
    # e.g: python -m fastrnns.bench --rnns jit and --group rnns
    default_groups = ["cnns", "rnns"]

    parser.add_argument("--seqLength", default="100", type=int)
    parser.add_argument("--numLayers", default="1", type=int)
    parser.add_argument("--inputSize", default="512", type=int)
    parser.add_argument("--hiddenSize", default="512", type=int)
    parser.add_argument("--miniBatch", default="64", type=int)
    parser.add_argument("--warmup", default="10", type=int)
    parser.add_argument("--nloops", default="100", type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--variable-lstms",
        "--variable_lstms",
        action="store_true",
        help="Also benchmark variable sequence length lstms "
        "Note that some of these run really slowly "
        "and that the `seqLength` flag will be ignored.",
    )
    parser.add_argument("--sep", default=" ", type=str)
    parser.add_argument("--print-json", nargs="?", default=None, const="oss")
    parser.add_argument("--rnns", nargs="*", help="What to run. cudnn, aten, jit, etc")
    parser.add_argument(
        "--cnns", nargs="*", help="What to run. resnet18, resnet18_jit, resnet50, etc"
    )
    parser.add_argument(
        "--group",
        nargs="*",
        default=default_groups,
        help="Which group to run. cnns, rnns, etc.",
    )
    parser.add_argument(
        "--fuser",
        default="te",
        type=str,
        help="The fuser backend to use. One of: te, old, or none",
    )
    parser.add_argument(
        "--executor",
        default=None,
        type=str,
        help="The executor to use. One of: legacy, simple, profiling",
    )
    parser.add_argument(
        "--cuda-pointwise-loop-level",
        "--cuda_pointwise_loop_level",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--cuda-pointwise-block-count",
        "--cuda_pointwise_block_count",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--cuda-pointwise-block-size",
        "--cuda_pointwise_block_size",
        default=None,
        type=int,
    )

    args = parser.parse_args()
    set_fuser(args.fuser, args.executor)

    if args.cuda_pointwise_loop_level:
        torch._C._jit_set_te_cuda_pointwise_loop_levels(args.cuda_pointwise_loop_level)
    if args.cuda_pointwise_block_count:
        torch._C._jit_set_te_cuda_pointwise_block_count(args.cuda_pointwise_block_count)
    if args.cuda_pointwise_block_size:
        torch._C._jit_set_te_cuda_pointwise_block_size(args.cuda_pointwise_block_size)

    rnns = args.rnns or [
        "cudnn",
        "aten",
        "jit",
        "jit_premul",
        "jit_premul_bias",
        "jit_simple",
        "jit_multilayer",
        "py",
    ]
    cnns = args.cnns or ["resnet18", "resnet18_jit", "resnet50", "resnet50_jit"]
    # TODO: Maybe add a separate section for the layernorm/dropout lstms
    # 'cudnn_layernorm', jit_layernorm', 'jit_layernom_decom',
    # 'jit', 'jit_dropout', 'cudnn_dropout'
    vlrnns = ["vl_cudnn", "vl_jit", "vl_py"]

    if args.print_json:
        print_stderr = lambda *args, **kwargs: None  # noqa: E731,F811
    print_stderr(args)

    bench_args = copy.deepcopy(vars(args))
    should_bench_varlen_lstms = args.variable_lstms
    del bench_args["group"]
    del bench_args["rnns"]
    del bench_args["cnns"]
    del bench_args["variable_lstms"]
    del bench_args["fuser"]
    del bench_args["executor"]
    del bench_args["cuda_pointwise_loop_level"]
    del bench_args["cuda_pointwise_block_count"]
    del bench_args["cuda_pointwise_block_size"]

    results = {}
    if should_bench_varlen_lstms:
        if args.nloops + args.warmup > 30:
            print_stderr(
                "WARNING: some of the variable sequence length lstms are "
                "very unoptimized and therefore take forever to run."
            )
        results.update(
            bench_group(vlrnns, "variable-length sequence LSTM", "vl_lstm", bench_args)
        )

    if "rnns" in args.group:
        results.update(bench_group(rnns, "LSTM", "lstm", bench_args))
    if "cnns" in args.group:
        results.update(bench_group(cnns, "ResNet", "resnet", bench_args))

    if args.print_json == "oss":
        print_json_oss_format(results)
    elif args.print_json == "pep":
        print_json_pep_format(results)
