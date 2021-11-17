
import argparse
import collections
import copy
import csv
import functools
import gc
import io
import itertools
import logging
import math
import numpy as np
import os
import re
import sys
import textwrap
import time
import torch
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import warnings
from torch import nn
from torch.nn import Module
from torch.jit import fuser
from os.path import abspath
from os.path import exists
from scipy.stats import gmean
from scipy.stats import ttest_ind

# from caffe2.python import workspace
# workspace.GlobalInit(['caffe2', '--caffe2_log_level=-5'])

import lazy_tensor_core
import lazy_tensor_core.core.lazy_model as ltm
import lazy_tensor_core.debug.metrics as met
lazy_tensor_core._LAZYC._ltc_init_ts_backend()


os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam
torchbench_dir = abspath("/home/whc/benchmark")
assert os.path.exists(torchbench_dir)
# os.chdir(torchbench_dir)
sys.path.append(torchbench_dir)
log = logging.getLogger(__name__)
SKIP = {}
current_name = ""
current_device = ""

def synchronize():
    pass

@functools.lru_cache(1)
def output_csv(name, headers):
    output = csv.writer(
        io.TextIOWrapper(
            open(name, "wb", buffering=0),
            "utf-8",
            write_through=True,
        )
    )
    output.writerow(headers)
    return output


class Fusion(nn.Module):

    def __init__(self, dims=[128, 16, 128, 128], device='cuda', jit=False):
        super(Fusion, self).__init__()
        self.attention_head_size = dims[1]
        self.example_inputs = (
            torch.randn(*dims, device=device, dtype=torch.float32),
            torch.randn(*dims, device=device, dtype=torch.float32),
        )
        
    name = "DivAddMul"

    def get_module(self):
        return self, self.example_inputs

    def forward(self, inputs, mask):
        out1 = inputs / math.sqrt(self.attention_head_size)
        out2 = out1 + mask
        out3 = out2 * 5.0
        return out3

def list_toy_models():
    yield Fusion

def pick_grad(name):
    if name in ("maml",):
        return torch.enable_grad()
    else:
        return torch.no_grad()

def short_name(name, limit=20):
    """Truncate a model name to limit chars"""
    return name if len(name) <= limit else f"{name[:limit - 3].rstrip('_')}..."


# Iter torchbench models and toy models
def iter_models(args):
    from fastNLP.core import logger

    logger.setLevel(logging.WARNING)
    from torchbenchmark import list_models  # noqa
    for benchmark_cls in itertools.chain(list_toy_models(), list_models()):
        if (
            (len(args.filter) and (not re.search("|".join(args.filter), benchmark_cls.name, re.I)))
            or (len(args.exclude) and re.search("|".join(args.exclude), benchmark_cls.name, re.I))
            or benchmark_cls.name in SKIP
        ):
            continue
        for device in args.devices:
            try:
                torch.manual_seed(1337)
                benchmark = benchmark_cls(device=device, jit=False)
                torch.manual_seed(1337)
                lazy_benchmark = benchmark_cls(device='lazy', jit=False)
                model, example_inputs = benchmark.get_module()
                lazy_model, lazy_example_inputs = lazy_benchmark.get_module()
                model.eval()
                lazy_model.eval()
                gc.collect()
                global current_name, current_device
                current_device = device
                current_name = short_name(benchmark.name)
                yield device, current_name, model, example_inputs, lazy_model, lazy_example_inputs
            except NotImplementedError:
                print("NotImplementedError")
                pass
            except Exception as e:
                print(f"Exception in iter_models for {benchmark_cls.name}", e)
                log.exception(f"misconfigured model {benchmark_cls.name}")

def call_model_with(model, inputs):
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        return model(*inputs)
    elif isinstance(inputs, dict):
        return model(**inputs)
    elif isistance(inputs, torch.Tensor):
        return model(inputs)
    raise RuntimeError("invalid example inputs ", inputs)

def timed(model, example_inputs, times=1, iter_sync_fn=lambda:(), final_sync_fn=lambda:()):
    
    final_sync_fn()
    gc.collect()
    torch.manual_seed(1337)
    # keep the lazy tensor results alive until the final sync
    results = []
    t0 = time.perf_counter()
    for _ in range(times):
        results.append(call_model_with(model, example_inputs))
        # may be just an async 'mark_step' for lazy, or no-op for cuda
        # iter_sync_fn()
        to_device(results[-1], 'cpu')
    
    # should be a hard sync for lazy and cuda
    # unless strictly measuring lazy trace overhead, then no-op
    to_device(results[-1], 'cpu')
    # final_sync_fn()
    t1 = time.perf_counter()
    return results[-1], t1 - t0

def to_device(inputs, device):
    try:
        import transformer
        if isinstance(inputs, transformers.modeling_outputs.MaskedLMOutput) \
        or isinstance(inputs, transformers.modeling_outputs.Seq2SeqLMOutput):
            correct_result = correct_result.to_tuple()[0]
    except ImportError:
        pass

    if isinstance(inputs, tuple):
        return tuple(to_device(i, device) for i in inputs)
    elif isinstance(inputs, dict):
        return {k: to_device(inputs[k], device) for k in inputs}
    elif isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    raise RuntimeError("invalid example inputs ", inputs)

def lazy_overhead_experiment(results, args, model, example_inputs, lazy_model, lazy_inputs):
    timings = np.zeros((args.repeat, 2), np.float64)
    for rep in range(args.warmup):
        # interleave the runs to handle frequency scaling and load changes
        timed(model, example_inputs, iter_sync_fn=torch.cuda.synchronize, final_sync_fn=torch.cuda.synchronize)
        timed(lazy_model, lazy_inputs)
# lazy_inputs = to_device(example_inputs, 'lazy')
    # lazy_model = model.to('lazy')
    for rep in range(args.repeat):
        # interleave the runs to handle frequency scaling and load changes
        _, timings[rep, 0] = timed(model, example_inputs, iter_sync_fn=torch.cuda.synchronize, final_sync_fn=torch.cuda.synchronize)
        _, timings[rep, 1] = timed(lazy_model, lazy_inputs)
    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    median = np.median(timings, axis=0)
    overhead = median[1] / median[0]
    results.append(overhead)
    output_csv(
        "lazy_overheads.csv",
        ("dev", "name", "overhead"),
    ).writerow([current_device, current_name, f"{overhead:.4f}"])
    return (overhead, pvalue)

def lazy_compute_experiment(results, args, model, example_inputs, lazy_model, lazy_inputs, times=10):
    timings = np.zeros((args.repeat, 2), np.float64)
    for rep in range(args.warmup):
        # interleave the runs to handle frequency scaling and load changes
        timed(model, example_inputs, final_sync_fn=torch.cuda.synchronize)
        with fuser('fuser2'):
            timed(lazy_model, lazy_inputs, iter_sync_fn=ltm.mark_step, final_sync_fn=ltm.wait_device_ops)
    
    for rep in range(args.repeat):
        # interleave the runs to handle frequency scaling and load changes
        _, timings[rep, 0] = timed(model, example_inputs, times=times, final_sync_fn=torch.cuda.synchronize)
        _, timings[rep, 1] = timed(lazy_model, lazy_inputs, times=times, iter_sync_fn=ltm.mark_step, final_sync_fn=ltm.wait_device_ops)
    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    median = np.median(timings, axis=0)
    speedup = median[0] / median[1]
    results.append(speedup)
    output_csv(
        "lazy_compute.csv",
        ("dev", "name", "speedup"),
    ).writerow([current_device, current_name, f"{speedup:.4f}"])
    return (speedup, pvalue)

def check_results(name, correct_result, lazy_result, device):
    import transformers #noqa
    if isinstance(correct_result, transformers.modeling_outputs.MaskedLMOutput) \
      or isinstance(correct_result, transformers.modeling_outputs.Seq2SeqLMOutput):
        correct_result = correct_result.to_tuple()[0]
        lazy_result = lazy_result.to_tuple()[0]
    lazy_result = lazy_result.to(device)
    return torch.allclose(correct_result, lazy_result)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", "-k", action="append", default=["DivAddMul", "hf_Bert", "hf_Bart"], help="filter benchmarks")
    parser.add_argument("--exclude", "-x", action="append", default=[], help="filter benchmarks")
    parser.add_argument("--devices", "-d", action="append", default=['cuda'], help="cpu or cuda")
    parser.add_argument("--warmup", type=int, default=4, help="number of warmup runs")
    parser.add_argument("--repeat", "-n", type=int, default=16, help="number of timing runs")
    parser.add_argument("--fuser", type=str, default='fuser2', choices=['fuser0', 'fuser1', 'fuser2'], help="0=legacy, 1=nnc, 2=nvfuser")
    args = parser.parse_args()
    results = []

    for device, name, model, example_inputs, lazy_model, lazy_inputs in iter_models(args):
        if device == 'cuda':
            assert 'LTC_TS_CUDA' in os.environ and bool(os.environ['LTC_TS_CUDA'])

        with pick_grad(name):
            try:
                torch.manual_seed(1337)
                correct_result = call_model_with(copy.deepcopy(model), example_inputs)
                torch.manual_seed(1337)
                lazy_result = call_model_with(lazy_model, lazy_inputs)
            except Exception:
                logging.exception("unhandled error")
                print("ERROR")
                continue
            if not check_results(name, correct_result, lazy_result, device):
                print("INCORRECT")
                import ipdb; ipdb.set_trace()
                continue
            overhead, pvalue = lazy_overhead_experiment(results, args, model, example_inputs, lazy_model, lazy_inputs)
            print(f"name: {name},  trace overhead: {overhead}, pvalue: {pvalue}")

            with fuser(args.fuser): 
                speedup, pvalue = lazy_compute_experiment(results, args, model, example_inputs, lazy_model, lazy_inputs)
                print(f"name: {name},  amortized compute speedup: {speedup}, pvalue: {pvalue} (using {args.fuser})")
            
            with fuser(args.fuser): 
                speedup, pvalue = lazy_compute_experiment(results, args, model, example_inputs, lazy_model, lazy_inputs, times=1)
                print(f"name: {name},  unamortized compute speedup: {speedup}, pvalue: {pvalue} (using {args.fuser})")
        