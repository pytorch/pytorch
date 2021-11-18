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
sys.path.append(torchbench_dir)
log = logging.getLogger(__name__)
SKIP = {}
current_name = ""
current_device = ""

@functools.cache
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

class DivAddMulBenchmark:
    """This wrapper helps interface with the same iterator as torchbench models
    """
    def __init__(self, dims):
        self.name = "DivAddMul[" + ','.join([str(d) for d in dims]) + ']'
        self.dims = dims

    def __call__(self, device, jit):
        return DivAddMul(self.dims, device, jit)

class DivAddMul(nn.Module):
    def __init__(self, dims=[128, 16, 128, 128], device='cuda', jit=False):
        super(DivAddMul, self).__init__()
        self.attention_head_size = dims[1]
        self.name = "DivAddMul[" + ','.join([str(d) for d in dims]) + ']'
        self.example_inputs = (
            torch.randn(*dims, device=device, dtype=torch.float32),
            torch.randn(*dims, device=device, dtype=torch.float32),
        )

    def get_module(self):
        return self, self.example_inputs

    def name(self):
        return self.name

    def forward(self, inputs, mask):
        out1 = inputs / math.sqrt(self.attention_head_size)
        out2 = out1 + mask
        out3 = out2 * 5.0
        return out3

def list_toy_models():
    yield DivAddMulBenchmark(dims=[1,1,1,1])
    yield DivAddMulBenchmark(dims=[1,16,128,128])
    yield DivAddMulBenchmark(dims=[128,16,128,128])
    yield DivAddMulBenchmark(dims=[256,16,128,128])

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
        name = benchmark_cls.name if hasattr(benchmark_cls, 'name') else benchmark_cls.name()
        if (
            (len(args.filter) and (not re.search("|".join(args.filter), name, re.I)))
            or (len(args.exclude) and re.search("|".join(args.exclude), name, re.I))
            or name in SKIP
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
                print(f"Exception in iter_models for {name}", e)
                log.exception(f"misconfigured model {name}")

def call_model_with(model, inputs):
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        return model(*inputs)
    elif isinstance(inputs, dict):
        return model(**inputs)
    elif isistance(inputs, torch.Tensor):
        return model(inputs)
    raise RuntimeError("invalid example inputs ", inputs)

# TODO(whc)
#2 understand delta between wait_device_ops, to_cuda
#  see why to_cpu is _so much slower_, given cuda is also copying to cpu
# - bert to_cpu amortized v unamortized huge delta... why?
# - hf-bart, shows wait_device_ops method is way off from reality.  why?
#3 see if bert vs bart data makes sense
#4 see if divaddmul is getting fused into a single kernel or not
#5 how much do my 'overhead fixes' from last night help?
class CudaSync:
    def __init__(self, sync_every_iter=False):
        self.sync_every_iter = sync_every_iter

    def iter_sync(self, results):
        if self.sync_every_iter:
           torch.cuda.synchronize()

    def final_sync(self, results):
       torch.cuda.synchronize()

class NoOpSync:
    def iter_sync(self, results):
        pass

    def final_sync(self, results):
        pass

class LazySync:
    def __init__(self, sync_every_iter=False):
        self.sync_every_iter = sync_every_iter

    def iter_sync(self, results):
        ltm.mark_step()
        if self.sync_every_iter:
            ltm.wait_device_ops()
            if current_device == 'cuda':
                torch.cuda.synchronize()
            else:
                raise RuntimeError("unexpected to not have current device=cuda")

    def final_sync(self, results):
        ltm.mark_step()
        ltm.wait_device_ops()
        if current_device == 'cuda':
            torch.cuda.synchronize()
        else:
            raise RuntimeError("unexpected to not have current device=cuda")

class ToDeviceSync:
    def __init__(self, device, sync_every_iter=False):
        self.sync_every_iter = sync_every_iter
        self.device = device

    def iter_sync(self, results):
        if self.sync_every_iter:
            to_device(results[-1], self.device)
            if current_device == 'cuda':
                torch.cuda.synchronize()

    def final_sync(self, results):
        if len(results):
            if self.sync_every_iter:
                to_device(results[-1], self.device)
            else:
                to_device(results, self.device)

        if current_device == 'cuda':
            torch.cuda.synchronize()

def timed(model, example_inputs, sync, times=1):
    results = []
    sync.final_sync(results)
    gc.collect()
    torch.manual_seed(1337)
    # keep the lazy tensor results alive until the final sync
    t0 = time.perf_counter()
    for _ in range(times):
        results.append(call_model_with(model, example_inputs))
        # may be just an async 'mark_step' for lazy, or no-op for cuda
        sync.iter_sync(results)
    
    # should be a hard sync for lazy and cuda
    # unless strictly measuring lazy trace overhead, then no-op
    sync.final_sync(results)
    t1 = time.perf_counter()
    return results[-1], t1 - t0

def to_device(inputs, device):
    if inputs is None:
        return None

    try:
        import transformer
        if isinstance(inputs, transformers.modeling_outputs.MaskedLMOutput) \
        or isinstance(inputs, transformers.modeling_outputs.Seq2SeqLMOutput):
            correct_result = correct_result.to_tuple()[0]
    except ImportError:
        pass

    if isinstance(inputs, tuple) or isinstance(inputs, list):
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
        timed(model, example_inputs, sync=CudaSync(sync_every_iter=True))
        timed(lazy_model, lazy_inputs, sync=LazySync(sync_every_iter=True))
    for rep in range(args.repeat):
        # interleave the runs to handle frequency scaling and load changes
        _, timings[rep, 0] = timed(model, example_inputs, sync=CudaSync(sync_every_iter=True))
        _, timings[rep, 1] = timed(lazy_model, lazy_inputs, sync=NoOpSync())
    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    median = np.median(timings, axis=0)
    overhead = median[1] / median[0]
    results.append(overhead)
    output_csv(
        "lazy_overheads.csv",
        ("dev", "name", "overhead", "pvalue"),
    ).writerow([current_device, current_name, f"{overhead:.4f}", f"{pvalue:.4e}"])
    print(f"{short_name(name, 30):<30} {current_device:<4}  {'trace overheads':<20} overhead: {overhead:.4f} pvalue: {pvalue:.4e}")
    return (overhead, pvalue)

def lazy_compute_experiment(experiment, results, args, model, example_inputs, lazy_model, lazy_inputs, sync_every_iter=False, to_dev_sync=None):
    timings = np.zeros((args.repeat, 2), np.float64)
    if to_dev_sync is not None:
        cuda_sync = ToDeviceSync(to_dev_sync, sync_every_iter=sync_every_iter) 
        lazy_sync = ToDeviceSync(to_dev_sync, sync_every_iter=sync_every_iter)
    else:
        cuda_sync = CudaSync(sync_every_iter=sync_every_iter) 
        lazy_sync = LazySync(sync_every_iter=sync_every_iter)
    
    # interleave the runs to handle frequency scaling and load changes
    for rep in range(args.warmup):
        # warmup
        timed(model, example_inputs, sync=cuda_sync)
        timed(lazy_model, lazy_inputs, sync=lazy_sync)
    for rep in range(args.repeat):
        # measure
        _, timings[rep, 0] = timed(model, example_inputs, times=args.inner_loop_repeat, sync=cuda_sync)
        _, timings[rep, 1] = timed(lazy_model, lazy_inputs, times=args.inner_loop_repeat, sync=lazy_sync)
    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    median = np.median(timings, axis=0)
    speedup = median[0] / median[1]
    results.append(speedup)
    output_csv(
        "lazy_compute.csv",
        ("name", "dev", "experiment", "speedup", "pvalue"),
    ).writerow([current_name, current_device, experiment, f"{speedup:.4f}", f"{pvalue:.4e}"])
    print(f"{short_name(current_name, 30):<30} {current_device:<4}  {experiment:<20} speedup: {speedup:.4f} pvalue: {pvalue:.4e}")
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
    parser.add_argument("--repeat", "-n", type=int, default=6, help="number of timing runs (samples)")
    parser.add_argument("--inner_loop_repeat", type=int, default=6, help="repeat the computation this many times per sample")
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
                print(f"ERROR ({name})")
                continue
            if not check_results(name, correct_result, lazy_result, device):
                print(f"INCORRECT ({name})")
                import ipdb; ipdb.set_trace()
                continue
            lazy_overhead_experiment(results, args, model, example_inputs, lazy_model, lazy_inputs)

            with fuser(args.fuser): 
                lazy_compute_experiment("amortized", results, args, model, example_inputs, lazy_model, lazy_inputs)
                lazy_compute_experiment("unamortized", results, args, model, example_inputs, lazy_model, lazy_inputs, sync_every_iter=True)

                # using to_cpu sync
                lazy_compute_experiment("to_cpu amortized", results, args, model, example_inputs, lazy_model, lazy_inputs, to_dev_sync='cpu')
                lazy_compute_experiment("to_cpu unamortized", results, args, model, example_inputs, lazy_model, lazy_inputs, sync_every_iter=True, to_dev_sync='cpu')

                # using to_cuda sync
                lazy_compute_experiment("to_cuda amortized", results, args, model, example_inputs, lazy_model, lazy_inputs, to_dev_sync='cuda')
                lazy_compute_experiment("to_cuda unamortized", results, args, model, example_inputs, lazy_model, lazy_inputs, sync_every_iter=True, to_dev_sync='cuda')
