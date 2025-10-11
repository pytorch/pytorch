#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import csv
import dataclasses
import functools
import gc
import importlib
import itertools
import json
import logging
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import weakref
from contextlib import contextmanager
from typing import Any, NamedTuple, Optional, overload, TYPE_CHECKING, TypeVar
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import psutil
import yaml
from scipy.stats import gmean, ttest_ind
from tqdm.auto import tqdm, trange

import torch
import torch._dynamo
import torch._dynamo.utils
import torch._export
import torch.distributed
import torch.multiprocessing as mp
from torch._C import _has_cuda as HAS_CUDA, _has_xpu as HAS_XPU
from torch._C._nativert import PyModelRunner
from torch._dynamo.profiler import fx_insert_profiling, Profiler
from torch._dynamo.testing import (
    dummy_fx_compile,
    format_speedup,
    reset_rng_state,
    same,
)
from torch._logging.scribe import open_source_signpost


try:
    from torch._dynamo.utils import clone_inputs, graph_break_reasons
    from torch._inductor.utils import fresh_cache
except ImportError:
    from _dynamo.utils import clone_inputs, graph_break_reasons
    from _inductor.utils import fresh_cache

import torch._functorch.config
from torch._functorch.aot_autograd import set_model_name
from torch._inductor import config as inductor_config, metrics
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map, tree_map_only


try:
    import torch_xla
    import torch_xla.core.xla_model as xm

    # This is to workaround the backward issue https://github.com/pytorch/xla/issues/4174
    torch_xla._XLAC._init_computation_client()
except ImportError:
    # ignore the error if torch_xla is not installed
    pass


if TYPE_CHECKING:
    from collections.abc import Sequence

_D = TypeVar("_D", bound=dict[str, Any])
_T = TypeVar("_T")


log = logging.getLogger(__name__)

# We are primarily interested in TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)

# Suppress torch.profiler spam
os.environ["KINETO_LOG_LEVEL"] = "5"

current_name = ""
current_device = ""
current_backend = ""
current_mode = ""
current_dtype = ""
current_quantization = ""
current_settings = None
current_batch_size = None
output_filename = None
disable_output = False

MAX_DOWNLOAD_ATTEMPTS = 5


class CI(NamedTuple):
    backend: str  # aot_eager or inductor
    training: bool
    dynamic: bool = False
    device: str = "cuda"


CI_SKIP_OPTIMIZER = {
    # HF
    "MobileBertForMaskedLM",  # Stack issue in fx
}

try:
    from .fb.common import INTERNAL_CI_SKIP_DYNAMIC_BATCH_ONLY
except ImportError:
    INTERNAL_CI_SKIP_DYNAMIC_BATCH_ONLY = set()

try:
    from pytorch.benchmark.fb.run_utils import trace_handler
except ImportError:
    trace_handler = None


CI_SKIP_DYNAMIC_BATCH_ONLY = {
    "sam",
    # See https://github.com/mindee/doctr/blob/f2114758d529ed8d3d0030581638f0520b6b98d8/doctr/models/detection/core.py#L89
    # It iterates over the batch, which is dynamic, and dynamo chokes
    # We should be able to graphbreak there.
    "doctr_det_predictor",
    "dlrm",
    "pyhpc_isoneutral_mixing",
    "pyhpc_equation_of_state",
    "pyhpc_turbulent_kinetic_energy",
    "detectron2_fcos_r_50_fpn",
    "detectron2_fasterrcnn_r_101_c4",
    "detectron2_fasterrcnn_r_101_dc5",
    "detectron2_fasterrcnn_r_101_fpn",
    "detectron2_fasterrcnn_r_50_c4",
    "detectron2_fasterrcnn_r_50_dc5",
    "detectron2_fasterrcnn_r_50_fpn",
    "Reformer",
    "llama",
}.union(INTERNAL_CI_SKIP_DYNAMIC_BATCH_ONLY)

# These models currently fail accuracy with eager Adam optimizer
# so we use SGD when running the full benchmarks
# https://github.com/pytorch/pytorch/issues/115966
BENCHMARK_USE_SGD = {
    # TorchBench
    "BERT_pytorch",
    "LearningToPaint",
    "alexnet",
    "dcgan",
    "demucs",
    "densenet121",
    "dlrm",
    "fastNLP_Bert",
    "mobilenet_v2",
    "phlippe_densenet",
    "phlippe_resnet",
    "pytorch_stargan",
    "resnet18",
    "shufflenet_v2_x1_0",
    "speech_transformer",
    "squeezenet1_1",
    "stable_diffusion_text_encoder",
    "vgg16",
    # HF
    "AlbertForMaskedLM",
    "BartForCausalLM",
    "ElectraForCausalLM",
    "M2M100ForConditionalGeneration",
    "MBartForCausalLM",
    "OPTForCausalLM",
    "PLBartForCausalLM",
    "PegasusForCausalLM",
    "TrOCRForCausalLM",
    "XGLMForCausalLM",
    # TIMM
    "adv_inception_v3",
    "tf_efficientnet_b0",
    "ghostnet_100",
}

# These models OOM in CI
# due to the extra memory of Adam optimizer states,
# so we fall back to SGD in CI
CI_USE_SGD = {
    "torchrec_dlrm",
    "demucs",
    "detectron2_fasterrcnn_r_101_c4",
    "detectron2_fasterrcnn_r_101_dc5",
    "detectron2_fasterrcnn_r_101_fpn",
    "detectron2_fasterrcnn_r_50_c4",
    "detectron2_fasterrcnn_r_50_dc5",
    "detectron2_fasterrcnn_r_50_fpn",
    "detectron2_maskrcnn_r_101_c4",
    "detectron2_maskrcnn_r_101_fpn",
    "detectron2_maskrcnn_r_50_c4",
    "detectron2_maskrcnn_r_50_fpn",
    "llama_v2_7b_16h",
    "mobilenet_v2_quantized_qat",
    "phi_1_5 resnet50_quantized_qat",
    "BlenderbotForCausalLM",
    "DALLE2_pytorch",
    "moco",
    "timm_efficientdet",
    "ghostnet_100",
    "inception_v3",
    "mobilevit_s",
    "pytorch_CycleGAN_and_pix2pix",
    "vision_maskrcnn",
    "dlrm",
    "resnet50",
    "dm_nfnet_f0",
}


DO_NOT_CAST_INPUTS = {"stable_diffusion"}


# Maps a benchmark model name to a list of status codes. For any listed entry, we'll
# capture TORCH_COMPILE_DEBUG logs in CI runs and preserve them (i.e., for upload) if
# the result status matches one listed.
CI_PRESERVE_COMPILE_DEBUG = {
    # For example:
    # "mnasnet1_0": ["fail_accuracy"],
}


@functools.lru_cache(maxsize=1)
def load_yaml_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)

    with open(filepath) as f:
        data = yaml.safe_load(f)

    internal_file_path = os.path.join(os.path.dirname(__file__), "fb", filename)
    if os.path.exists(internal_file_path):
        with open(internal_file_path) as f:
            internal_data = yaml.safe_load(f)
            data.update(internal_data)

    def flatten(lst):
        for item in lst:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item

    def maybe_list_to_set(obj):
        if isinstance(obj, dict):
            return {k: maybe_list_to_set(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return set(flatten(obj))
        return obj

    return maybe_list_to_set(data)


def model_specified_by_path(path_and_class_str):
    return ":" in path_and_class_str


def load_model_from_path(path_and_class_str):
    configs = {}
    for kvstr in path_and_class_str.split(","):
        k, v = kvstr.split(":")
        configs[k] = v

    for name in ["path", "class"]:
        if name not in configs:
            raise RuntimeError(
                "Invalid --only arguments. Check help message for the correct format"
            )

    path = configs["path"]
    class_name = configs["class"]

    if path[:1] != "/":
        raise RuntimeError(
            "Use absolute path since dynamo may change the current working directory which makes using relative path tricky"
        )

    spec = importlib.util.spec_from_file_location("module_name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_class = getattr(module, class_name)
    assert issubclass(model_class, torch.nn.Module)
    model = model_class()
    assert hasattr(model, "get_example_inputs")
    inputs = model.get_example_inputs()
    return model, inputs


def write_outputs(filename, headers, row, upload_to_benchmark_db: bool = True):
    """
    Write both CSV and JSON outputs using the original CSV output interface
    """
    global disable_output
    if disable_output:
        return

    output_csv(filename, headers, row)
    if upload_to_benchmark_db:
        output_json(filename, headers, row)


def output_csv(filename, headers, row):
    if os.path.exists(filename):
        with open(filename) as fd:
            lines = list(csv.reader(fd)) or [[]]
            if headers and len(headers) > len(lines[0]):
                # if prior results failed the header might not be filled in yet
                lines[0] = headers
            else:
                headers = lines[0]
    else:
        lines = [headers]
    lines.append([(f"{x:.6f}" if isinstance(x, float) else x) for x in row])
    with open(filename, "w") as fd:
        writer = csv.writer(fd, lineterminator="\n")
        for line in lines:
            writer.writerow(list(line) + ["0"] * (len(headers) - len(line)))


def output_json(filename, headers, row):
    """
    Write the result into JSON format, so that it can be uploaded to the benchmark database
    to be displayed on OSS dashboard. The JSON format is defined at
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    origin = ""
    if "torchbench" in filename:
        origin = "torchbench"
    elif "huggingface" in filename:
        origin = "huggingface"
    elif "timm_models" in filename:
        origin = "timm_models"

    extra_info = {
        "device": current_device,
        "quantization": current_quantization,
        "batch_size": current_batch_size,
    }
    if current_settings:
        extra_info.update(current_settings)

    mapping_headers = {headers[i]: v for i, v in enumerate(row)}
    with open(f"{os.path.splitext(filename)[0]}.json", "a") as f:
        for header, value in mapping_headers.items():
            # These headers are not metric names
            if header in ("dev", "name", "batch_size"):
                continue

            # Make sure that the record is valid
            if not current_name:
                continue

            record = {
                "benchmark": {
                    "name": "TorchInductor",
                    "mode": current_mode,
                    "dtype": current_dtype,
                    "extra_info": extra_info,
                },
                "model": {
                    "name": current_name,
                    "type": "OSS model",
                    "backend": current_backend,
                    "origins": [origin],
                },
            }

            # NB: When the metric is accuracy, its value is actually a string, i.e. pass, and
            # not a number. ClickHouse doesn't support mix types atm. It has a Variant type
            # https://clickhouse.com/docs/en/sql-reference/data-types/variant, but this isn't
            # recommended by CH team themselves. The workaround here is to store that value
            # in the extra_info field instead.
            if isinstance(value, str):
                record["metric"] = {
                    "name": header,
                    "extra_info": {"benchmark_values": [value]},
                }
            else:
                record["metric"] = {
                    "name": header,
                    "benchmark_values": [value],
                }

            print(json.dumps(record), file=f)


def get_suite_from_model_iter_fn(model_iter_fn):
    # TODO: This is a bit of a hack
    suite = None
    if (runner := getattr(model_iter_fn, "__self__", None)) and hasattr(
        runner, "suite_name"
    ):
        suite = runner.suite_name
    return suite


def output_signpost(data, args, suite, error=None):
    from torch.utils._stats import simple_call_counter

    data = data.copy()

    if "name" not in data:
        data["name"] = current_name

    if "dev" not in data:
        data["dev"] = current_device

    filtered_args = vars(args).copy()
    # I generated this list by reading through all the configs and dropping
    # ones that looked irrelevant or redundant
    for k in [
        "filter",
        "exclude",
        "exclude_exact",
        "dump_raw_metrics",
        "log_operator_inputs",
        "distributed_master_port",
        "skip_accuracy_check",
        "generate_aot_autograd_stats",
        "output",
        "output_directory",
        "disable_output",
        "export_profiler_trace",
        "profiler_trace_name",
        "explain",
        "stats",
        "print_memory",
        "print_compilation_time",
        "print_dataframe_summary",
        "print_graph_breaks",
        "log_graph_breaks",
        "timing",
        "progress",
        "timeout",
        "per_process_memory_fraction",
        "minify",
        "verbose",
        "quiet",
        "print_fx",
        "print_aten_ops",
        "log_conv_args",
        "recompile_profiler",
        "find_batch_sizes",
        # Redundant
        "batch_size",
        "batch_size_file",
        "only",
        "diff_branch",
        "tag",
        "coverage",
        "overhead",
        "speedup_dynamo_ts",
        "speedup_fx2trt",
        "speedup_fx2trt_fp16",
        "accuracy",
        "performance",
        "tolerance",
    ]:
        del filtered_args[k]

    event_name = "unknown"
    if args.accuracy:
        event_name = "accuracy"
    elif args.quantization:
        event_name = "quantization"
    elif args.performance:
        event_name = "performance"

    from torch._dynamo.utils import calculate_time_spent, compilation_time_metrics

    wall_time_by_phase = calculate_time_spent()

    open_source_signpost(
        subsystem="dynamo_benchmark",
        name=event_name,
        parameters=json.dumps(
            {
                **data,
                # TODO: Arguably the rest of these should be in the CSV too
                "suite": suite,
                # Better than using compile_times utils directly
                # NB: Externally, compilation_metrics colloquially refers to
                # the coarse-grained phase timings, even though internally
                # they are called something else
                "compilation_metrics": wall_time_by_phase,
                "agg_compilation_metrics": {
                    k: sum(v) for k, v in compilation_time_metrics.items()
                },
                "detailed_compilation_metrics": compilation_time_metrics,
                "simple_call_counter": simple_call_counter,
                # NB: args has training vs inference
                "args": filtered_args,
                "error": error,
            }
        ),
    )

    return wall_time_by_phase["total_wall_time"]


def nothing(f):
    return f


@functools.cache
def patch_torch_manual_seed():
    """Make torch manual seed deterministic. Helps with accuracy testing."""

    def deterministic_torch_manual_seed(*args, **kwargs):
        from torch._C import default_generator

        seed = 1337
        if HAS_CUDA:
            import torch.cuda

            if not torch.cuda._is_in_bad_fork():
                torch.cuda.manual_seed_all(seed)
        if HAS_XPU:
            import torch.xpu

            if not torch.xpu._is_in_bad_fork():
                torch.xpu.manual_seed_all(seed)
        return default_generator.manual_seed(seed)

    torch.manual_seed = deterministic_torch_manual_seed


def empty_gpu_cache(device):
    """
    Explicitly empty gpu cache to avoid OOM in subsequent run.
    """

    if device not in ["cuda", "xpu", "mps"]:
        log.warning(
            "Trying to call the empty_gpu_cache for device: %s, which is not in list [cuda, xpu]",
            device,
        )
        return

    getattr(torch, device).empty_cache()


def synchronize():
    pass


def summarize_graph_break(filename):
    """
    Sorts and de-dupes the graphs breaks on the reason string. Note that this
    function is just a best effort to reduce the logging information. We could
    miss some graph breaks because of de-duping. We can further refine this
    function as need arises.
    """
    log_file = f"{filename.rstrip('.csv')}_graph_breaks.csv"
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = df.sort_values("reason").drop_duplicates(subset="reason")

        # Specialize for multi tensor sgd as reason is not identical
        multi_tensor_sgd_row = df.loc[df["reason"].str.contains("_multi_tensor_sgd")]
        if len(multi_tensor_sgd_row):
            df = df[
                ~df["reason"].str.contains("_multi_tensor_sgd")
            ]  # Drop all sgd rows
            df = pd.concat(
                [df, pd.DataFrame([multi_tensor_sgd_row.iloc[0]])], axis=0
            )  # Add back a single row
        df.to_csv(f"{log_file.rstrip('.csv')}_deduped.csv", index=False)


def print_summary(filename, print_dataframe=False):
    if not (filename and os.path.exists(filename)):
        return
    data = pd.read_csv(filename)
    if "tag" in data.columns:
        for tag in data.tag.unique():
            if tag == "0.0000":
                continue  # This happens for failed runs
            print(f"\nSummary for tag={tag}:")
            print_summary_table(data[data.tag == tag], print_dataframe=print_dataframe)
    else:
        print_summary_table(data, print_dataframe=print_dataframe)
    summarize_graph_break(filename)


def print_summary_table(data, print_dataframe=False):
    if print_dataframe:
        pd.options.display.max_rows = 1000
        pd.options.display.max_columns = 1000
        pd.options.display.width = 2000
        print(data)
    width = max(map(len, data.columns))
    for col in data.columns:
        try:
            if col in ("dev", "name", "batch_size", "tag"):
                continue
            elif col in ("pct_ops", "pct_time"):
                print(col.ljust(width), f"{data[col].mean():.3%}")
            elif col in ("graphs", "graph_calls", "captured_ops", "total_ops"):
                print(col.ljust(width), f"{data[col].mean():.3f}")
            elif col in ("compilation_latency"):
                print(col.ljust(width), f"mean={data[col].mean():.3f} seconds")
            elif col in ("compression_ratio"):
                print(col.ljust(width), f"mean={data[col].mean():.3f}x")
            elif col in ("accuracy"):
                pass_rate = (data[col] == "pass").mean()
                print(col.ljust(width), f"pass_rate={100 * pass_rate:.2f}%")
            else:
                cdata = data[col]
                print(
                    col.ljust(width),
                    f"gmean={gmean(cdata):.2f}x mean={cdata.mean():.3f}x",
                )
        except Exception:
            pass


def tensor_is_on_xla(tensors):
    def visit(x: torch.Tensor):
        nonlocal result
        if x.device.type == "xla":
            result = True

    result = False
    tree_map_only(torch.Tensor, visit, tensors)
    return result


def timed(
    model,
    model_iter_fn,
    example_inputs,
    times=1,
    return_result=False,
    collect_outputs=False,
    batch_size=None,
):
    use_xla = tensor_is_on_xla(example_inputs)
    synchronize()

    if batch_size:
        patch_torch_manual_seed()

    if use_xla:
        xm.mark_step()
        xm.wait_device_ops()

    def vary_batch(t: torch.Tensor, new_batch_size) -> torch.Tensor:
        for i, s in enumerate(t.size()):
            if s == batch_size:
                # If new batch is smaller, we truncate
                if new_batch_size < batch_size:
                    indexer = [slice(None)] * t.ndim
                    indexer[i] = slice(0, new_batch_size)
                    t = t[tuple(indexer)]
                # If new batch is greater, we just duplicate the last row
                # over and over until we hit the desired batch size
                elif new_batch_size > batch_size:
                    indexer = [slice(None)] * t.ndim
                    indexer[i] = -1
                    last_slice = t[tuple(indexer)].unsqueeze(i)
                    repeat_shape = list(t.shape)
                    repeat_shape[i] = new_batch_size - batch_size
                    padding = last_slice.expand(*repeat_shape)
                    t = torch.cat([t, padding], dim=i)
                break
        return t

    time_total = 0
    # Dont collect outputs to correctly measure timing
    for i in range(times):
        # If batch_size is 1, it too often collides with other non batch size
        # dimensions resulting in errors.
        if batch_size and batch_size > 1:
            # Calculate new batch size by varying the original batch size by up to 20%
            # Ensure it's at least greater than 1
            variation = random.uniform(0.8, 1.2)
            new_batch_size = max(2, int(batch_size * variation))
            example_inputs = tree_map_only(
                torch.Tensor, lambda x: vary_batch(x, new_batch_size), example_inputs
            )
        # Put this call inside the loop to reset the seed for each iteration.
        # Don't include reset_rng_state() to correctly measure timing
        reset_rng_state(use_xla)
        t_iter_begin = time.perf_counter()
        result = model_iter_fn(model, example_inputs, collect_outputs=collect_outputs)

        # instead of calling sync on result_list, we should call mark_step.
        # In training case, result_list may be empty, but we want to
        # send all the pending graphs for compilation.
        if use_xla:
            # For the model running on regular torchxla (baseline), we need the
            # mark step to send the accumulated graph for compilation.
            #
            # For the model running with dynamo/torchxla bridge, in training case,
            # we need the mark step to send the optimizer graph out for
            # compilation.
            xm.mark_step()
        t_iter_end = time.perf_counter()
        time_total += t_iter_end - t_iter_begin

    t_0 = time.perf_counter()
    if use_xla:
        xm.wait_device_ops()
    synchronize()
    t_1 = time.perf_counter()
    time_total += t_1 - t_0
    return (time_total, result) if return_result else time_total


@overload
def _normalize_bench_inputs(example_inputs: _D) -> tuple[tuple[()], _D]: ...


@overload
def _normalize_bench_inputs(
    example_inputs: Sequence[_T],
) -> tuple[tuple[_T, ...], dict[str, Any]]: ...


def _normalize_bench_inputs(example_inputs):
    # NOTE(bowbao): For huggingface benchmark, example_inputs are formatted as dictionary,
    # and consumed like `model(**example_inputs)`.
    # For other benchmarks, example_inputs are formatted as tuple and consumed
    # like `model(*example_inputs)`.
    if isinstance(example_inputs, dict):
        return (), example_inputs
    else:
        return tuple(example_inputs), {}


def _register_dataclass_output_as_pytree(example_outputs) -> None:
    # NOTE(angelayi): For huggingface benchmark, some example outputs are
    # formatted as a dataclass which pytree cannot consume. So we want
    # to register the pytree implementation here
    example_outputs_flat = pytree.tree_leaves(example_outputs)
    output_dataclass_types = [
        type(out) for out in example_outputs_flat if dataclasses.is_dataclass(type(out))
    ]
    for output_type in output_dataclass_types:
        from torch._export.utils import register_dataclass_as_pytree_node

        register_dataclass_as_pytree_node(
            output_type,
            serialized_type_name=f"{output_type.__module__}.{output_type.__name__}",
        )


class Stats:
    totals = collections.defaultdict(collections.Counter)

    @classmethod
    def reset_counters(cls):
        for k, v in torch._dynamo.utils.counters.items():
            cls.totals[k].update(v)
        ok = torch._dynamo.utils.counters["frames"]["ok"]
        total = torch._dynamo.utils.counters["frames"]["total"]
        torch._dynamo.utils.counters.clear()
        return ok, total

    @classmethod
    def print_summary(cls):
        for k, v in sorted(cls.totals.items()):
            lines = "\n  ".join(map(str, v.most_common(50)))
            print(f"STATS {k}\n  {lines}")

    @classmethod
    def aot_summary(cls):
        return [cls.totals["aot_autograd"]["total"], cls.totals["aot_autograd"]["ok"]]


def coverage_experiment(args, model_iter_fn, model, example_inputs):
    """
    Test operator/model coverage of TorchDynamo and record statistics
    taken from a profiler.  This target is mainly intended to check
    correctness.

    Writes to ./coverage.csv
    """
    profiler = Profiler()
    frozen_model_iter_fn = torch._dynamo.run(model_iter_fn)
    with profiler.prof:
        frozen_model_iter_fn(model, example_inputs)
    coverage_result = profiler.results()
    write_outputs(
        output_filename,
        (
            "dev",
            "name",
            "batch_size",
            "graphs",
            "graph_calls",
            "captured_ops",
            "total_ops",
            "pct_ops",
            "pct_time",
        ),
        [
            current_device,
            current_name,
            current_batch_size,
        ]
        + coverage_result.tocsv(),
    )
    return coverage_result


def speedup_experiment_fx2trt(args, model_iter_fn, model, example_inputs):
    """
    Measure speedups over eager using the trt inference backend. TRT backend is based fx graph
    generated by torch._dynamo.
    Writes to ./speedups_fx2trt.csv
    """
    return speedup_experiment(args, model_iter_fn, model, example_inputs)


# TODO: CompilerProfiler is deprecated, remove this
def recompile_profiler_experiment(args, model_iter_fn, model, example_inputs):
    prof = torch._dynamo.utils.CompilerProfiler()
    opt_model_iter_fn = torch._dynamo.optimize(prof, nopython=args.nopython)(
        model_iter_fn
    )
    opt_model_iter_fn(model, example_inputs)
    write_outputs(
        output_filename, ["model", "profiler report"], [current_name, prof.report()]
    )
    met = prof.get_metrics()
    guard_failures = len(met["guard_failures"])
    return [guard_failures]


def randomize_input(inputs):
    if isinstance(inputs, (list, tuple)):
        return type(inputs)([randomize_input(x) for x in inputs])
    elif isinstance(inputs, torch.Tensor):
        if inputs.dtype in (torch.float32, torch.float64):
            torch._dynamo.utils.counters["randomize_input"]["times"] += 1
            return torch.randn_like(inputs)
        elif inputs.dtype == torch.int64:
            # Note: we can not simply tune integer tensors as follows
            #   `return torch.randint_like(inputs, high=inputs.max().item())`
            # This may break some invariants between tensors.
            # E.g. in embedding lookup case, one tensor is the length
            # and another is an indices tensor.
            return inputs
        else:
            raise RuntimeError(
                f"randomize_input need support tensor of type {inputs.dtype}"
            )
    else:
        raise RuntimeError(
            f"randomize_input can not handle input of type {type(inputs)}"
        )


def maybe_mark_step(args):
    if args.trace_on_xla:
        xm.mark_step()


def latency_experiment(args, model_iter_fn, model, example_inputs, mark, **kwargs):
    """
    Measure latency on a specific backend.
    """

    timings = np.zeros((args.repeat,), np.float64)
    # if we randomize the input, we should also check the result is correct
    should_randomize_input = args.randomize_input

    import contextlib

    from torch._inductor.utils import maybe_profile

    @contextlib.contextmanager
    def maybe_mark_profile(*args, **kwargs):
        prof: torch.profiler.profile = kwargs.pop("p", None)
        mark = kwargs.pop("mark", None)
        if prof:
            with torch.profiler.record_function(mark):
                yield
        else:
            yield

    times = args.iterations_per_run

    with maybe_profile(args.export_profiler_trace, **args.profile_details) as p:
        for rep in trange(args.repeat, desc="running benchmark"):
            inputs = (
                randomize_input(copy.deepcopy(example_inputs))
                if should_randomize_input
                else example_inputs
            )
            # need call mark_step to perform the computation
            # on randomize_input. Otherwise the first call using the
            # inputs will incur high penalty then the next one.
            maybe_mark_step(args)

            with maybe_mark_profile(p=p, mark=mark):
                timings[rep], actual_output = timed(
                    model,
                    model_iter_fn,
                    inputs,
                    return_result=True,
                    times=times,
                    collect_outputs=args.collect_outputs,
                )

    if args.export_profiler_trace:
        name = args.profiler_trace_name + "_" + model.name
        if hasattr(args, "rank"):
            name += f"_rank_{args.rank}"
        name += ".json"
        name = os.path.join(torch._dynamo.config.base_dir, name)
        p.export_chrome_trace(name)
    return timings


# TODO: This seems to be specifically triggered by torchao testing
def latency_experiment_summary(suite_name, args, model, timings, **kwargs):
    median = np.median(timings, axis=0)
    speedup = median[0] / median[1]
    if args.dump_raw_metrics:
        np.save(
            f"{output_filename[:-4]}-raw_timings-{current_name}-{current_device}.npy",
            timings,
        )

    first_headers = ["dev", "name", "batch_size"]
    first_fields = [current_device, current_name, current_batch_size]
    if "tag" in kwargs:
        first_headers.append("tag")
        first_fields.append(kwargs["tag"])
    headers = first_headers + ["speedup", "abs_latency"]
    row = first_fields + [float(speedup), median[1] * 1000]
    msg = f"{speedup:.3f}x"
    if args.baseline:
        headers.extend(
            [
                "baseline",
                "speedup_vs_baseline",
            ]
        )
        df = pd.read_csv(args.baseline)
        try:
            baseline_speedup = df[df["name"] == current_name]["speedup"].item()
            row.extend([baseline_speedup, speedup / baseline_speedup])
            msg = f"{baseline_speedup:.3f}x -> {speedup:.3f}x [{speedup / baseline_speedup:.3f}x]"
        except (KeyError, ZeroDivisionError):
            row.extend(
                [
                    0.0,
                    0.0,
                ]
            )
    if "compilation_latency" in kwargs:
        headers += [
            "compilation_latency",
            "compression_ratio",
            "eager_peak_mem",
            "dynamo_peak_mem",
        ]
        row.append(kwargs["compilation_latency"])
        row.append(kwargs["compression_ratio"])
        row.append(kwargs["eager_peak_mem"])
        row.append(kwargs["dynamo_peak_mem"])

    if "cache_lookup_latency" in kwargs:
        headers.append("cache_lookup_latency")
        row.append(kwargs["cache_lookup_latency"])

    if "dynamo_stats" in kwargs:
        for k, v in kwargs["dynamo_stats"].items():
            headers.append(k)
            row.append(v)
    write_outputs(
        output_filename,
        headers,
        row,
    )
    c_headers, c_data = torch._dynamo.utils.compile_times(repr="csv", aggregate=True)
    assert output_filename.find(".csv") > 0, (
        f"expected output_filename to be a .csv, but got {output_filename}"
    )
    write_outputs(
        output_filename[:-4] + "_compilation_metrics.csv",
        first_headers + c_headers,
        first_fields + c_data,
    )

    # Hypothetically you can use this from other places, but it's currently
    # inaccessible, and when this assert fails you need to update the
    # event_name here to account for the other cases you are using this
    assert args.quantization is not None
    output_signpost(
        dict(zip(headers, row)),
        args,
        suite_name,
    )

    return msg


def speedup_experiment(args, model_iter_fn, model, example_inputs, **kwargs):
    """
    Measure speedups over eager.

    Writes to ./speedups.csv
    """
    timings = np.zeros((args.repeat, 2), np.float64)
    # if we randomize the input, we should also check the result is correct
    should_randomize_input = args.randomize_input

    import contextlib

    from torch._inductor.utils import maybe_profile

    @contextlib.contextmanager
    def maybe_mark_profile(*args, **kwargs):
        prof: torch.profiler.profile = kwargs.pop("p", None)
        mark = kwargs.pop("mark", None)
        if prof:
            with torch.profiler.record_function(mark):
                yield
        else:
            yield

    times = args.iterations_per_run

    # Use higher tolerance for XLA since XLA cause numerical instability when
    # graph size changes
    tolerance = args.xla_tolerance if args.trace_on_xla else 1e-4
    torch._dynamo.config.repro_tolerance = tolerance

    with maybe_profile(args.export_profiler_trace, **args.profile_details) as p:
        if args.export_aot_inductor:
            frozen_model_iter_fn = export_aot_inductor(
                model, example_inputs, args.inductor_compile_mode
            )
        elif args.export_nativert:
            frozen_model_iter_fn = export_nativert(model, example_inputs)
        elif args.torchscript_jit_trace:
            frozen_model_iter_fn = torchscript_jit_trace(model, example_inputs)
        else:
            if kwargs["hf_llm"]:
                # If it's an llm, we want to optimize model.forward, and use
                # the generate function
                model.forward = torch._dynamo.run(model)
                frozen_model_iter_fn = model_iter_fn
            else:
                frozen_model_iter_fn = torch._dynamo.run(model_iter_fn)

        for rep in trange(args.repeat, desc="running benchmark"):
            inputs = (
                randomize_input(copy.deepcopy(example_inputs))
                if should_randomize_input
                else example_inputs
            )
            # need call mark_step to perform the computation
            # on randomize_input. Otherwise the first call using the
            # inputs will incur high penalty then the next one.
            maybe_mark_step(args)

            # interleave the runs to handle frequency scaling and load changes
            with (
                maybe_mark_profile(p=p, mark="expected"),
                torch.compiler.set_stance("force_eager"),
            ):
                timings[rep, 0], expected_output = timed(
                    model,
                    model_iter_fn,
                    inputs,
                    return_result=True,
                    times=times,
                    collect_outputs=args.collect_outputs,
                    batch_size=kwargs.get("batch_size"),
                )

            # call mark_step between the 2 calls to make the comparison fair.
            maybe_mark_step(args)

            with maybe_mark_profile(p=p, mark="actual"):
                timings[rep, 1], actual_output = timed(
                    model,
                    frozen_model_iter_fn,
                    inputs,
                    return_result=True,
                    times=times,
                    collect_outputs=args.collect_outputs,
                )

    if args.export_profiler_trace:
        name = args.profiler_trace_name + "_" + model.name
        if hasattr(args, "rank"):
            name += f"_rank_{args.rank}"
        if args.export_perfdoctor and trace_handler:
            trace_handler(name, p)
        else:
            name += ".json"
            name = os.path.join(torch._dynamo.config.base_dir, name)
            p.export_chrome_trace(name)

    median = np.median(timings, axis=0)
    speedup = median[0] / median[1]
    if args.dump_raw_metrics:
        np.save(
            f"{output_filename[:-4]}-raw_timings-{current_name}-{current_device}.npy",
            timings,
        )

    first_headers = ["dev", "name", "batch_size"]
    first_fields = [current_device, current_name, current_batch_size]
    if "tag" in kwargs:
        first_headers.append("tag")
        first_fields.append(kwargs["tag"])
    headers = first_headers + ["speedup", "abs_latency"]
    row = first_fields + [float(speedup), median[1] * 1000]
    msg = f"{speedup:.3f}x"
    if args.baseline:
        headers.extend(
            [
                "baseline",
                "speedup_vs_baseline",
            ]
        )
        df = pd.read_csv(args.baseline)
        try:
            baseline_speedup = df[df["name"] == current_name]["speedup"].item()
            row.extend([baseline_speedup, speedup / baseline_speedup])
            msg = f"{baseline_speedup:.3f}x -> {speedup:.3f}x [{speedup / baseline_speedup:.3f}x]"
        except (KeyError, ZeroDivisionError):
            row.extend(
                [
                    0.0,
                    0.0,
                ]
            )
    if "compilation_latency" in kwargs:
        headers += [
            "compilation_latency",
            "compression_ratio",
            "eager_peak_mem",
            "dynamo_peak_mem",
        ]
        row.append(kwargs["compilation_latency"])
        row.append(kwargs["compression_ratio"])
        row.append(kwargs["eager_peak_mem"])
        row.append(kwargs["dynamo_peak_mem"])

    if "cache_lookup_latency" in kwargs:
        headers.append("cache_lookup_latency")
        row.append(kwargs["cache_lookup_latency"])

    if "dynamo_stats" in kwargs:
        for k, v in kwargs["dynamo_stats"].items():
            headers.append(k)
            row.append(v)
    write_outputs(
        output_filename,
        headers,
        row,
    )
    c_headers, c_data = torch._dynamo.utils.compile_times(repr="csv", aggregate=True)
    assert output_filename.find(".csv") > 0, (
        f"expected output_filename to be a .csv, but got {output_filename}"
    )
    write_outputs(
        output_filename[:-4] + "_compilation_metrics.csv",
        first_headers + c_headers,
        first_fields + c_data,
    )

    output_signpost(
        dict(zip(headers, row)),
        args,
        get_suite_from_model_iter_fn(model_iter_fn),
    )

    return msg


def overhead_experiment(*args, model_iter_fn):
    """
    Measure overheads of TorchDynamo by running with no backend (only
    eager+FX), and reporting speedup/slowdown over eager.

    Writes to ./overheads.csv
    """
    return speedup_experiment(*args, model_iter_fn)


def print_fx(gm, example_inputs):
    print(gm.graph)
    return gm


def print_aten_ops(gm, example_inputs):
    from functorch.compile import aot_module

    def trace_printer(gm, _):
        print(gm.graph)
        return gm

    return aot_module(gm, fw_compiler=trace_printer, bw_compiler=trace_printer)


def baselines(models, model_iter_fn, example_inputs, args):
    """
    Common measurement code across all baseline experiments.
    """
    models = list(models)
    for idx, (name, model) in enumerate(models):
        if idx == 0:
            result0 = model_iter_fn(model, example_inputs)
        elif model is not None:
            try:
                result = model_iter_fn(model, example_inputs)
                if same(result0, result):
                    continue
                print(name, "is INCORRECT")
            except Exception:
                log.exception("error checking %s", name)
            models[idx] = (name, None)
    timings = np.zeros((args.repeat, len(models)), np.float64)
    timings.fill(1.0e10)
    for rep in range(args.repeat):
        for idx, (name, model) in enumerate(models):
            if model is not None:
                try:
                    timings[rep, idx] = timed(model, model_iter_fn, example_inputs)
                except Exception:
                    pass
    pvalue = [
        ttest_ind(timings[:, 0], timings[:, i]).pvalue
        for i in range(1, timings.shape[1])
    ]
    median = np.median(timings, axis=0)
    speedup = median[0] / median[1:]
    for idx, (name, model) in enumerate(models[1:]):
        if model is None:
            speedup[idx] = 0.0
    result = " ".join(
        [
            format_speedup(s, p, m is not None)
            for s, p, m in zip(speedup, pvalue, [m for n, m in models[1:]])
        ]
    )
    write_outputs(
        output_filename,
        ("dev", "name", "batch_size") + tuple(n for n, m in models[1:]),
        [current_device, current_name, current_batch_size]
        + [f"{x:.4f}" for x in speedup],
    )
    return result


def xla(args, model_iter_fn, model, example_inputs):
    xla_dev = xm.xla_device(devkind=current_device)
    model_xla = copy.deepcopy(model).to("cpu").to(device=xla_dev)
    example_inputs_xla = tree_map_only(
        torch.Tensor, lambda x: x.to("cpu").to(device=xla_dev), example_inputs
    )
    for _ in range(3):  # warmup
        timed(model, model_iter_fn, example_inputs)
        timed(model_xla, model_iter_fn, example_inputs_xla)
    timings = np.zeros((args.repeat, 2), np.float64)
    timings.fill(1.0e10)
    for rep in range(args.repeat):
        timings[rep, 0] = timed(model, model_iter_fn, example_inputs)
        timings[rep, 1] = timed(model_xla, model_iter_fn, example_inputs_xla)

    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    time_baseline, time_xla = np.median(timings, axis=0)
    speedup = time_baseline / time_xla
    write_outputs(
        output_filename,
        ("dev", "name", "batch_size", "speedup", "time_baseline", "time_xla"),
        [
            current_device,
            current_name,
            current_batch_size,
            speedup,
            time_baseline,
            time_xla,
        ],
    )
    return format_speedup(speedup, pvalue)


def try_script(model, example_inputs):
    try:
        return torch.jit.script(model)
    except Exception:
        return None


def _produce_dynamic_shapes_for_export(path, x):
    # mark_dynamic() is ignored for export.
    # use this to produce dynamic_shapes spec instead.
    from torch.export.dynamic_shapes import Dim

    if not isinstance(x, torch.Tensor):
        return None
    return dict.fromkeys(getattr(x, "_dynamo_dynamic_indices", {}), Dim.AUTO)


class AOTInductorModelCache:
    cache: dict[weakref.ref, tuple[Any, float]] = {}

    @classmethod
    def load(cls, model, example_inputs, mode):
        import torch._inductor
        from torch.export.dynamic_shapes import _combine_args, _tree_map_with_path

        key = weakref.ref(model)
        if key not in cls.cache:
            # Register the output dataclass to pytree
            example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
            with torch.no_grad():
                # copy.deepcopy is required to prevent any surprising side-effect,
                # see https://github.com/pytorch/pytorch/issues/113029
                # This will cause memory stats to be overshadowed by this eager run.
                # To fix that, memory stats will be reset later.
                example_outputs = copy.deepcopy(model)(*example_args, **example_kwargs)

            if pytree.is_namedtuple_instance(example_outputs):
                typ = type(example_outputs)
                pytree._register_namedtuple(
                    typ,
                    serialized_type_name=f"{typ.__module__}.{typ.__name__}",
                )
            else:
                _register_dataclass_output_as_pytree(example_outputs)

            combined_args = _combine_args(model, example_args, example_kwargs)
            dynamic_shapes = _tree_map_with_path(
                _produce_dynamic_shapes_for_export, combined_args
            )

            # delete example_outputs and reset memory stats here
            del example_outputs
            if current_device == "cuda":
                empty_gpu_cache(current_device)
                torch.cuda.reset_peak_memory_stats()
                pre_clone_memory_used = torch.cuda.max_memory_allocated()
            elif current_device == "hpu":
                torch.hpu.reset_peak_memory_stats()
                pre_clone_memory_used = torch.hpu.max_memory_allocated()

            # Clone the model pre-exporting.  This prevents scenarios observed in a few
            # models, where the forward pass modifies model state while exporting, and
            # FakeTensors are thus saved as model data members.  This invalidates model
            # reuse in eager mode, so it's safest to export a model clone.
            model_clone = copy.deepcopy(model)

            # Since CPU doesn't monitor max memory allocation, anything measuring peak
            # memory will miss our transient model clone on CPU anyway.
            #
            # The justification for tracking this value (in order to remove it from the
            # AOTInductor memory measurements) is that normal usage of AOTInductor would
            # not clone the model, since the eager model would be unused post-export.
            clone_memory_used = 0.0
            if current_device == "cuda":
                clone_memory_used = (
                    torch.cuda.max_memory_allocated() - pre_clone_memory_used
                ) / 1e9
            elif current_device == "hpu":
                clone_memory_used = (
                    torch.hpu.max_memory_allocated() - pre_clone_memory_used
                ) / 1e9

            inductor_configs = {}
            if mode == "max-autotune":
                inductor_configs["max_autotune"] = True
            ep = torch.export.export(
                model_clone,
                example_args,
                example_kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
            )
            with torch.no_grad():
                package_path = torch._inductor.aoti_compile_and_package(
                    ep, inductor_configs=inductor_configs
                )  # type: ignore[arg-type]

            cls.cache[key] = (
                torch._inductor.aoti_load_package(package_path),
                clone_memory_used,
            )

        return cls.cache[key][0]

    @classmethod
    def get_excess_memory(cls, model) -> float:
        return cls.cache.get(weakref.ref(model), (None, 0.0))[1]


class NativeRTCache:
    cache: dict[weakref.ref, Any] = {}

    @classmethod
    def load(cls, model, example_inputs):
        from torch.export.dynamic_shapes import _combine_args, _tree_map_with_path

        key = weakref.ref(model)
        if key not in cls.cache:
            example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
            example_outputs = model(*example_args, **example_kwargs)
            _register_dataclass_output_as_pytree(example_outputs)

            combined_args = _combine_args(model, example_args, example_kwargs)
            dynamic_shapes = _tree_map_with_path(
                _produce_dynamic_shapes_for_export, combined_args
            )

            ep = torch.export.export(
                model, example_args, example_kwargs, dynamic_shapes=dynamic_shapes
            )
            ep = ep.run_decompositions({})
            with tempfile.NamedTemporaryFile(delete=False) as f:
                torch.export.pt2_archive._package.package_pt2(
                    f, exported_programs={"forward": ep}
                )
                filename = f.name
            cls.cache[key] = PyModelRunner(filename, "forward")

        return cls.cache[key]


class JitTracedCache:
    cache: dict[weakref.ref, Any] = {}

    @classmethod
    def load(cls, model, example_inputs):
        key = weakref.ref(model)
        if key not in cls.cache:
            example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
            if example_args:
                jit_traced_module = torch.jit.trace(
                    model, example_inputs=example_args, strict=False
                )
            else:
                jit_traced_module = torch.jit.trace(
                    model, example_kwarg_inputs=example_kwargs, strict=False
                )

            cls.cache[key] = jit_traced_module

        return cls.cache[key]


def export(model, example_inputs):
    from torch.export.dynamic_shapes import _combine_args, _tree_map_with_path

    example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
    example_outputs = model(*example_args, **example_kwargs)
    _register_dataclass_output_as_pytree(example_outputs)

    combined_args = _combine_args(model, example_args, example_kwargs)
    dynamic_shapes = _tree_map_with_path(
        _produce_dynamic_shapes_for_export, combined_args
    )

    # NOTE: if args.export is ever enabled for --performance mode (rather than solely
    # --accuracy), we'll need to clone the model and subtract out extra memory usage, as
    # done in AOTInductorModelCache.
    ep = torch.export.export(
        model, example_args, example_kwargs, dynamic_shapes=dynamic_shapes, strict=True
    )

    def opt_export(_, example_inputs):
        example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
        return ep.module()(*example_args, **example_kwargs)

    return opt_export


def export_nativert(model, example_inputs):
    optimized = NativeRTCache.load(model, example_inputs)

    def opt_nativert(_, example_inputs, collect_outputs=False):
        example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
        return optimized.run(*example_args, **example_kwargs)

    return opt_nativert


def export_aot_inductor(model, example_inputs, mode):
    optimized = AOTInductorModelCache.load(model, example_inputs, mode)

    def opt_aot_inductor(_, example_inputs, collect_outputs=False):
        example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
        return optimized(*example_args, **example_kwargs)

    return opt_aot_inductor


def torchscript_jit_trace(model, example_inputs):
    optimized = JitTracedCache.load(model, example_inputs)

    def opt_jit_trace(_, example_inputs, collect_outputs=False):
        example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
        return optimized(*example_args, **example_kwargs)

    return opt_jit_trace


def download_retry_decorator(download_fn):
    """
    Decorator function for applying retry logic to a download function.

    The wrapped function will be called up to 5 times and raises an exception if the function fails each time.
    After each unsuccessful attempt, there is a delay before the next attempt, which is increased linearly with the number of tries.

    Usage:
    @download_retry_decorator
    def download_function(model_name: str):
        # download logic goes here
    """

    @functools.wraps(download_fn)
    def wrapper(self, *args, **kwargs) -> Any:
        tries = 0
        total_allowed_tries = MAX_DOWNLOAD_ATTEMPTS
        while tries <= total_allowed_tries:
            try:
                model = download_fn(self, *args, **kwargs)
                return model
            except Exception as e:
                tries += 1
                if tries <= total_allowed_tries:
                    wait = tries * 30
                    print(
                        f"Failed to load model: {e}. Trying again ({tries}/{total_allowed_tries}) after {wait}s"
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(  # noqa: B904
                        f"Failed to load model '{args}' with following error(s): {str(e)}."
                    )

    return wrapper


def read_batch_size_from_file(args, filename, model_name):
    batch_size = None
    if os.path.exists("benchmarks"):
        filename = os.path.join("benchmarks", filename)
    assert os.path.exists(filename), filename
    with open(filename) as f:
        lines = f.readlines()
        lines = [i.split(",") for i in lines if len(i.strip()) > 0]
        for val in lines:
            cur_name, b = val
            if model_name == cur_name:
                batch_size = int(b)
    if batch_size is None:
        log.warning("Could not find batch size for %s", model_name)
    elif batch_size == -1:
        raise RuntimeError(
            f"Batch size is unset for {model_name} in {args.batch_size_file}"
        )
    print(f"batch size: {batch_size}")
    return batch_size


class TimeOutException(Exception):
    pass


def alarm_handler(signum, frame):
    raise TimeOutException


def exit_after(s):
    """
    Decorator to raise TimeoutException if the fn is taking more than s seconds
    to run.
    """

    def outer(fn):
        def inner(*args, **kwargs):
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(s)
            try:
                result = fn(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return inner

    return outer


def get_peak_memory():
    return torch.cuda.max_memory_allocated() / 10**9


def null_experiment(args, model_iter_fn, model, example_inputs):
    """
    A no-op experiment useful for making sure TorchBenchark alone works properly.
    """

    return []


def cast_to(dtype, model, inputs):
    # cast model and inputs to fp16
    if dtype == torch.float16:
        model = model.half()
    else:
        model = model.to(dtype)

    inputs = tree_map(
        lambda x: x.to(dtype)
        if isinstance(x, torch.Tensor) and x.is_floating_point()
        else x,
        inputs,
    )
    return model, inputs


def cast_to_bf16(model, inputs):
    return cast_to(torch.bfloat16, model, inputs)


def cast_to_fp16(model, inputs):
    return cast_to(torch.float16, model, inputs)


def cast_to_fp64(model, inputs):
    return cast_to(torch.float64, model, inputs)


def cast_to_fp32(model, inputs):
    return cast_to(torch.float32, model, inputs)


class DummyGradScaler:
    def scale(self, loss):
        return loss


def get_dynamo_stats():
    # TODO: consider deepcopy'ing the entire counters struct and
    # adding a helper to do subtraction on it
    return collections.Counter(
        {
            "calls_captured": torch._dynamo.utils.counters["stats"]["calls_captured"],
            "unique_graphs": torch._dynamo.utils.counters["stats"]["unique_graphs"],
            "graph_breaks": sum(torch._dynamo.utils.counters["graph_break"].values()),
            # NB: The plus removes zero counts
            "unique_graph_breaks": len(+torch._dynamo.utils.counters["graph_break"]),
            "autograd_captures": torch._dynamo.utils.counters["compiled_autograd"][
                "captures"
            ],
            "autograd_compiles": torch._dynamo.utils.counters["compiled_autograd"][
                "compiles"
            ],
            "cudagraph_skips": torch._dynamo.utils.counters["inductor"][
                "cudagraph_skips"
            ],
        }
    )


@contextmanager
def maybe_init_distributed(should_init_distributed, rank, world_size, port="6789"):
    try:
        if should_init_distributed:
            torch.cuda.set_device(rank)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = port
            torch.distributed.init_process_group(
                "nccl", rank=rank, world_size=world_size
            )
        yield
    finally:
        if should_init_distributed:
            torch.distributed.destroy_process_group()


@contextmanager
def maybe_snapshot_memory(should_snapshot_memory, suffix):
    # Enables Memory Snapshot tool for memory deep dives:
    # https://pytorch.org/blog/understanding-gpu-memory-1/
    try:
        if should_snapshot_memory:
            torch.cuda.memory._record_memory_history(max_entries=100000)
        yield
    finally:
        if should_snapshot_memory:
            try:
                torch.cuda.memory._dump_snapshot(
                    os.path.join(
                        torch._dynamo.config.base_dir,
                        f"{output_filename.rstrip('.csv')}_{suffix}.pickle",
                    )
                )
            except Exception as e:
                log.error("Failed to save memory snapshot, %s", e)

            torch.cuda.memory._record_memory_history(enabled=None)


class BenchmarkRunner:
    def __init__(self):
        self.model_iter_fn = None
        self.grad_scaler = DummyGradScaler()
        self.autocast = contextlib.nullcontext
        self.autocast_arg = {}
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._args = None

    def setup_amp(self, current_device=None):
        if self.args.only in self.fp32_only_models:
            return

        devices = [current_device] if current_device else self.args.devices
        if self.args.amp:
            # AMP training can lead to small loss values which can underflow
            # gradient values returning in zero gradients. To solve this
            # problem, PyTorch introduces GradScaler. GradScaler is a stateful
            # structure, that scales the loss values to prevent underflow. Loss
            # values are big at the beginning of training (therefore not
            # requiring scaling), while loss value tends to be small as network
            # starts getting better (requiring scaling). GradScaler manages all
            # of this fine tuning, checking the gradients are turning to inf,
            # discarding such batches.

            # Since we are not running a long iteration, default value of
            # init_scale 65536 is going to turn all gradients to inf. Therefore,
            # we just use a init_scale of 2.0 for benchmarking purpose.

            # Disabling Gradscaler because
            #  1) Benchmark setup runs 2 iterations of fwd-bwd. So, not useful.
            #  2) Current setup shares grad_scaler for eager and dynamo model,
            #  which is bad as Gradscaler has state and can adjust the scaling
            #  factor between eager and dynamo run, making accuracy check
            #  harder.
            # self.grad_scaler = torch.amp.GradScaler(device="cuda", init_scale=2.0)
            self.autocast = functools.partial(
                torch.amp.autocast, device_type=devices[0]
            )
            if self.args.amp_dtype:
                amp_dtype = (
                    torch.float16
                    if self.args.amp_dtype == "float16"
                    else torch.bfloat16
                )
                self.autocast_arg["dtype"] = amp_dtype

    def init_optimizer(self, name, device, params):
        if device == "cuda" and self.args.training and name not in CI_SKIP_OPTIMIZER:
            if (name in CI_USE_SGD and self.args.ci) or name in BENCHMARK_USE_SGD:
                self.optimizer = torch.optim.SGD(params, lr=0.01, foreach=True)
                # Disable multi_tensor_sgd for benchmarking, there isn't a large performance benefit (~1%) to compiling
                # this optimizer because it is a single foreach add, and increases compile time.
                # After autotuning and fake tensor caching lands, we can enable, because the compile time impact will be lower.
                # Fake Tensor caching: https://github.com/pytorch/pytorch/pull/113873
                # Autotuning: https://github.com/pytorch/pytorch/issues/117447
                self.optimizer.step = torch._dynamo.disable(self.optimizer.step)
            else:
                self.optimizer = torch.optim.Adam(
                    params, lr=0.01, capturable=True, foreach=True
                )
        else:
            self.optimizer = None

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, args):
        self._args = args

    @property
    def skip_models(self):
        return set()

    @property
    def skip_models_for_cuda(self):
        return set()

    @property
    def skip_models_for_cpu(self):
        return set()

    @property
    def skip_models_for_cpu_aarch64(self):
        return set()

    @property
    def skip_models_for_freezing_cpu(self):
        return set()

    @property
    def skip_models_for_freezing_cuda(self):
        return set()

    @property
    def slow_models(self):
        return set()

    @property
    def very_slow_models(self):
        return set()

    @property
    def non_deterministic_models(self):
        return set()

    @property
    def fp32_only_models(self):
        return set()

    @property
    def force_amp_for_fp16_bf16_models(self):
        return set()

    @property
    def force_fp16_for_bf16_models(self):
        return set()

    @property
    def skip_not_suitable_for_training_models(self):
        return set()

    @property
    def failing_torchinductor_models(self):
        return set()

    @property
    def failing_fx2trt_models(self):
        return set()

    @property
    def skip_accuracy_checks_large_models_dashboard(self):
        return set()

    @property
    def skip_accuracy_check_as_eager_non_deterministic(self):
        return set()

    @property
    def skip_multiprocess_models(self):
        return set()

    @property
    def skip_models_due_to_control_flow(self):
        return set()

    @property
    def skip_models_due_to_export_not_supported(self):
        return set()

    @property
    def disable_cudagraph_models(self):
        return set()

    @property
    def guard_on_nn_module_models(self):
        return set()

    @property
    def inline_inbuilt_nn_modules_models(self):
        return set()

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        raise NotImplementedError

    @property
    def equal_nan(self):
        equal_nan = True
        if self.args.float32:
            equal_nan = False
        return equal_nan

    def use_larger_multiplier_for_smaller_tensor(self, name):
        return False

    def iter_models(self, args):
        for model_name in self.iter_model_names(args):
            for device in args.devices:
                try:
                    yield self.load_model(
                        device,
                        model_name,
                        batch_size=args.batch_size,
                    )
                except NotImplementedError:
                    continue  # bad benchmark implementation

    def deepcopy_model(self, model):
        return copy.deepcopy(model)

    def cast_based_on_args(self, model, example_inputs):
        if self.args.float32 or self.args.only in self.fp32_only_models:
            if not self.args.float32:
                log.warning("Model %s supports float32 only", self.args.only)
            model, example_inputs = cast_to_fp32(model, example_inputs)
        elif self.args.float16:
            if self.args.only in self.force_amp_for_fp16_bf16_models:
                log.warning(
                    "Model %s does not support float16, running with amp instead",
                    self.args.only,
                )
                self.args.amp = True
                self.setup_amp()
            else:
                model, example_inputs = cast_to_fp16(model, example_inputs)
        elif self.args.bfloat16:
            if self.args.only in self.force_amp_for_fp16_bf16_models:
                log.warning(
                    "Model %s does not support bfloat16, running with amp instead",
                    self.args.only,
                )
                self.args.amp = True
                self.setup_amp()
            elif self.args.only in self.force_fp16_for_bf16_models:
                log.warning(
                    "Model %s does not support bfloat16, running with float16 instead",
                    self.args.only,
                )
                model, example_inputs = cast_to_fp16(model, example_inputs)
            else:
                model, example_inputs = cast_to_bf16(model, example_inputs)

        return model, example_inputs

    def validate_model(self, model, example_inputs):
        """
        Runs the eager model with example inputs to ensure that eager passes.
        """
        model = self.deepcopy_model(model)
        example_inputs = clone_inputs(example_inputs)
        model, example_inputs = self.cast_based_on_args(model, example_inputs)
        try:
            self.model_iter_fn(model, example_inputs)
        except Exception as e:
            raise RuntimeError("Eager run failed") from e

    def maybe_cast(self, model, example_inputs):
        model, example_inputs = self.cast_based_on_args(model, example_inputs)
        return model, example_inputs

    def decay_batch_exp(self, batch_size, factor=0.5, divisor=2):
        out_batch_size = batch_size * factor
        if out_batch_size > divisor:
            out_batch_size = (out_batch_size + 1) // divisor * divisor
        else:
            out_batch_size = batch_size - 1
        return max(0, int(out_batch_size))

    def batch_size_finder(self, device, model_name, initial_batch_size=1024):
        batch_size = initial_batch_size
        while batch_size >= 1:
            empty_gpu_cache(current_device)
            try:
                device, name, model, example_inputs, _ = self.load_model(
                    device,
                    model_name,
                    batch_size,
                )
                self.model_iter_fn(model, example_inputs)
                return batch_size
            except RuntimeError as e:
                error_str = str(e)
                if "channels_last" in error_str:
                    break
            batch_size = self.decay_batch_exp(batch_size)
        return 1

    def run_n_iterations(self, mod, inputs, model_iter_fn):
        n = self.args.iterations
        for _ in range(n - 1):
            model_iter_fn(mod, inputs, collect_outputs=False)
        return model_iter_fn(mod, inputs, collect_outputs=True)

    @torch._disable_dynamo(recursive=True)
    def optimizer_zero_grad(self, mod):
        if self.optimizer is not None:
            self.optimizer.zero_grad(True)
        else:
            mod.zero_grad(True)

    def optimizer_step(self):
        if self.optimizer is not None:
            self.optimizer.step()

    def get_benchmark_indices(self, length):
        start = self._args.partition_id * (length // self._args.total_partitions)
        end = (
            (self._args.partition_id + 1) * (length // self._args.total_partitions)
            if self._args.partition_id < self._args.total_partitions - 1
            else length
        )
        return start, end

    def get_fsdp_auto_wrap_policy(self, model_name: str):
        from diffusers.models.transformer_2d import Transformer2DModel
        from torchbenchmark.models.nanogpt.model import Block
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        from torch.distributed.fsdp.wrap import (
            ModuleWrapPolicy,
            size_based_auto_wrap_policy,
        )

        # handcrafted wrap policy
        MODEL_FSDP_WRAP = {
            "stable_diffusion_unet": (Transformer2DModel,),
            "llama_v2_7b_16h": (LlamaDecoderLayer,),
            "nanogpt": (Block,),
        }

        if model_name not in MODEL_FSDP_WRAP:
            # default to using wrap policy based on module size
            return functools.partial(
                size_based_auto_wrap_policy, recurse=True, min_num_params=int(1e5)
            )

        return ModuleWrapPolicy(MODEL_FSDP_WRAP[model_name])

    def deepcopy_and_maybe_parallelize(self, model):
        model = self.deepcopy_model(model)
        if self.args.ddp:
            assert torch.distributed.is_available(), (
                "Can't use DDP without a distributed enabled build"
            )
            from torch.nn.parallel import DistributedDataParallel as DDP

            model = DDP(model, find_unused_parameters=True)
        elif self.args.fsdp:
            assert torch.distributed.is_available(), (
                "Can't use FSDP without a distributed enabled build"
            )
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                MixedPrecision,
            )

            if self.args.float16:
                dtype = torch.float16
            elif self.args.bfloat16:
                dtype = torch.bfloat16
            else:
                dtype = torch.float32

            mp_policy = MixedPrecision(
                param_dtype=dtype,
                # Gradient communication precision.
                reduce_dtype=dtype,
                # Buffer precision.
                buffer_dtype=dtype,
            )

            model = FSDP(
                model,
                use_orig_params=True,
                device_id=torch.cuda.current_device()
                if self.args.devices[-1] == "cuda"
                else None,
                mixed_precision=mp_policy,
                limit_all_gathers=True,
                auto_wrap_policy=self.get_fsdp_auto_wrap_policy(self.args.only),
            )
        return model

    def check_accuracy(
        self, name, model, example_inputs, optimize_ctx, experiment, tag
    ):
        """
        Checks accuracy.
        1) Collect the outputs with fp64 datatype. This is useful for error checking.
        2) Checks if eager itself has variations.
        """
        start_stats = get_dynamo_stats()

        def record_status(accuracy_status, dynamo_start_stats):
            """
            Records the status in the csv file
            """
            if current_name in self.non_deterministic_models:
                if accuracy_status in (
                    "pass",
                    "eager_two_runs_differ",
                    "fail_accuracy",
                ):
                    accuracy_status = "pass"

            headers = ["dev", "name", "batch_size", "accuracy"]
            fields = [current_device, current_name, current_batch_size, accuracy_status]

            if tag is not None:
                headers.insert(3, "tag")
                fields.insert(3, tag)

            o_headers = list(headers)
            o_fields = list(fields)

            dynamo_stats = get_dynamo_stats()
            dynamo_stats.subtract(dynamo_start_stats)
            for k, v in dynamo_stats.items():
                headers.append(k)
                fields.append(v)

            total_wall_time = output_signpost(
                dict(zip(o_headers, o_fields)),
                self.args,
                self.suite_name,
            )
            headers.append("compilation_latency")
            fields.append(total_wall_time)
            write_outputs(output_filename, headers, fields)

            if self.args.print_compilation_time:
                print(f"Compilation time (from dynamo_timed): {total_wall_time}")

            return accuracy_status

        if name in self.skip_accuracy_checks_large_models_dashboard:
            return record_status("pass_due_to_skip", dynamo_start_stats=start_stats)

        # Skip all accuracy check for the torchao backend
        if self.args.backend == "torchao":
            return record_status("pass_due_to_skip", dynamo_start_stats=start_stats)

        with self.pick_grad(name, self.args.training):
            # Collect the fp64 reference outputs to be used later for accuracy checking.
            fp64_outputs = None
            model_fp64 = None
            inputs_fp64 = None
            try:
                model_fp64, inputs_fp64 = cast_to_fp64(
                    self.deepcopy_and_maybe_parallelize(model),
                    clone_inputs(example_inputs),
                )
                self.init_optimizer(name, current_device, model_fp64.parameters())
                fp64_outputs = self.run_n_iterations(
                    model_fp64, inputs_fp64, self.model_iter_fn
                )
                fp64_outputs = tree_map(
                    lambda x: x.to(torch.float64)
                    if isinstance(x, torch.Tensor) and x.is_floating_point()
                    else x,
                    fp64_outputs,
                )
            except Exception:
                log.warning(
                    "fp64 golden ref were not generated for %s. Setting accuracy check to cosine",
                    name,
                )
                self.args.cosine = True
                fp64_outputs = None
            finally:
                del model_fp64, inputs_fp64
                empty_gpu_cache(current_device)

            tolerance, cos_similarity = self.get_tolerance_and_cosine_flag(
                self.args.training, current_device, name
            )

            # Cast the model to float16/float32 as necessary
            model, example_inputs = self.maybe_cast(model, example_inputs)
            accuracy_status = "pass"

            # Get results of native pytorch
            reset_rng_state()
            model_copy = None
            try:
                with torch.compiler.set_stance("force_eager"):
                    model_copy = self.deepcopy_and_maybe_parallelize(model)
                    self.init_optimizer(name, current_device, model_copy.parameters())
                    correct_result = self.run_n_iterations(
                        model_copy, clone_inputs(example_inputs), self.model_iter_fn
                    )
            except Exception as e:
                accuracy_status = (
                    "eager_1st_run_OOM"
                    if isinstance(e, torch.cuda.OutOfMemoryError)
                    else "eager_1st_run_fail"
                )
                log.exception("")
                return record_status(accuracy_status, dynamo_start_stats=start_stats)
            finally:
                del model_copy
                empty_gpu_cache(current_device)

            # Rerun native pytorch
            reset_rng_state()
            model_copy = None
            try:
                with torch.compiler.set_stance("force_eager"):
                    model_copy = self.deepcopy_and_maybe_parallelize(model)
                    self.init_optimizer(name, current_device, model_copy.parameters())
                    correct_rerun_result = self.run_n_iterations(
                        model_copy, clone_inputs(example_inputs), self.model_iter_fn
                    )
            except Exception as e:
                accuracy_status = (
                    "eager_2nd_run_OOM"
                    if isinstance(e, torch.cuda.OutOfMemoryError)
                    else "eager_2nd_run_fail"
                )
                log.exception("")
                return record_status(accuracy_status, dynamo_start_stats=start_stats)
            finally:
                del model_copy
                empty_gpu_cache(current_device)

            # Two eager runs should have exactly same result, within tolerance.
            # TODO If we want the above to be true, then deterministic should be set.
            # For example, MIOpen convolutions could be implemented with non-deterministic algos.
            is_same = True
            try:
                if (
                    name not in self.skip_accuracy_check_as_eager_non_deterministic
                    and not same(
                        correct_result,
                        correct_rerun_result,
                        fp64_ref=None,
                        cos_similarity=False,
                        tol=tolerance if torch.version.hip else 0,
                        equal_nan=self.equal_nan,
                        use_larger_multiplier_for_smaller_tensor=self.use_larger_multiplier_for_smaller_tensor(
                            name
                        ),
                    )
                ):
                    is_same = False
            except Exception:
                # Sometimes torch.allclose may throw RuntimeError
                is_same = False

            if not is_same:
                accuracy_status = "eager_two_runs_differ"
                return record_status(accuracy_status, dynamo_start_stats=start_stats)

            correct_rerun_result = None

            # Run with Dynamo
            reset_rng_state()
            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()
            model_copy = None
            try:
                model_copy = self.deepcopy_and_maybe_parallelize(model)
                self.init_optimizer(name, current_device, model_copy.parameters())
                if (
                    self.args.export
                    or self.args.export_aot_inductor
                    or self.args.export_nativert
                    or self.args.torchscript_jit_trace
                ):
                    # apply export on module directly
                    # no need for n iterations
                    # the logic should be the same to self.model_iter_fn (forward_pass)
                    with self.autocast(**self.autocast_arg):
                        optimized_model_iter_fn = optimize_ctx(
                            model_copy, example_inputs
                        )
                        new_result = optimized_model_iter_fn(model_copy, example_inputs)
                else:
                    optimized_model_iter_fn = optimize_ctx(self.model_iter_fn)
                    new_result = self.run_n_iterations(
                        model_copy, example_inputs, optimized_model_iter_fn
                    )
            except Exception as e:
                log.exception("")
                print(
                    "TorchDynamo optimized model failed to run because of following error"
                )
                accuracy_status = (
                    "OOM"
                    if isinstance(e, torch.cuda.OutOfMemoryError)
                    else "fail_to_run"
                )
                return record_status(accuracy_status, dynamo_start_stats=start_stats)
            finally:
                del model_copy

            if name in self.skip_accuracy_check_as_eager_non_deterministic:
                return record_status("pass_due_to_skip", dynamo_start_stats=start_stats)

            force_max_multiplier = False
            if (
                self.args.freezing
                and self.args.bfloat16
                and torch._dynamo.utils.counters["inductor"]["binary_folding_conv"] > 0
            ):
                force_max_multiplier = True

            try:
                if self.args.training and self.args.amp:
                    if process_fn := self.get_output_amp_train_process_func.get(
                        name, None
                    ):
                        correct_result = process_fn(correct_result)
                        new_result = process_fn(new_result)
                        fp64_outputs = process_fn(fp64_outputs)

                if not same(
                    correct_result,
                    new_result,
                    fp64_outputs,
                    equal_nan=self.equal_nan,
                    use_larger_multiplier_for_smaller_tensor=self.use_larger_multiplier_for_smaller_tensor(
                        name
                    ),
                    cos_similarity=cos_similarity,
                    tol=tolerance,
                    force_max_multiplier=force_max_multiplier,
                ):
                    is_same = False
            except Exception:
                # Sometimes torch.allclose may throw RuntimeError
                is_same = False

            if not is_same:
                if self.args.skip_accuracy_check:
                    accuracy_status = "pass_due_to_skip"
                else:
                    accuracy_status = "fail_accuracy"
                return record_status(accuracy_status, dynamo_start_stats=start_stats)

        return record_status(accuracy_status, dynamo_start_stats=start_stats)

    def check_tolerance(
        self, name, model, example_inputs, optimize_ctx, base_device="cpu"
    ):
        """
        Checks tolerance based on https://pytorch.org/docs/stable/generated/torch.allclose.html.
        """
        tolerance_status = "pass"
        if name in self.skip_accuracy_checks_large_models_dashboard:
            tolerance_status = "pass_due_to_skip"
            return tolerance_status
        # Cast the model to float16/float32 as necessary
        model, example_inputs = self.maybe_cast(model, example_inputs)

        with self.pick_grad(name, self.args.training):
            # Get results of native pytorch
            reset_rng_state()
            model_copy = copy.deepcopy(model)
            model_copy = model_copy.to(base_device)
            example_inputs_copy = copy.deepcopy(example_inputs)
            example_inputs_copy = tree_map(
                lambda x: x.to(base_device), example_inputs_copy
            )
            self.init_optimizer(name, base_device, model_copy.parameters())
            correct_result = self.run_n_iterations(
                model_copy, example_inputs_copy, self.model_iter_fn
            )

            # Run with Dynamo
            # Sometime CI fails with random triton compilation failure which will be skipped for now
            # TODO: revisit this after switching to new Triton runtime
            reset_rng_state()
            torch._dynamo.reset()
            try:
                self.init_optimizer(name, current_device, model.parameters())
                optimized_model_iter_fn = optimize_ctx(self.model_iter_fn)
                new_result = self.run_n_iterations(
                    model_copy, example_inputs, optimized_model_iter_fn
                )
            except Exception:
                log.exception("")
                print(
                    "TorchDynamo optimized model failed to run because of following error"
                )
                return "fail_to_run"

            def dump_max_mean_values(tol, ref, res):
                if isinstance(ref, (list, tuple, torch.nn.ParameterList, torch.Size)):
                    for refi, resi in zip(ref, res):
                        dump_max_mean_values(tol, refi, resi)
                elif isinstance(ref, dict):
                    for k in ref.keys():
                        dump_max_mean_values(tol, ref[k], res[k])
                elif isinstance(ref, torch.Tensor):
                    res = res.to(base_device)
                    t = torch.abs(ref - res) / (1 + torch.abs(ref))
                    tol.append(t.flatten().to(torch.float32))
                return tol

            tol = []
            dump_max_mean_values(tol, correct_result, new_result)
            tol = torch.cat(tol)
            tol = torch.tensor(tol)
            max = torch.max(tol)
            mean = torch.mean(tol)
            div = torch.std(tol)
            headers = ["dev", "name", "batch_size", "max", "mean", "std"]
            fields = [
                current_device,
                current_name,
                current_batch_size,
                max.item(),
                mean.item(),
                div.item(),
            ]
            write_outputs(output_filename, headers, fields)
        return tolerance_status

    def run_performance_test_non_alternate(
        self, name, model, example_inputs, optimize_ctx, experiment, tag=None
    ):
        "Run performance test in non-alternately."
        assert experiment.func is latency_experiment, (
            "Must run with latency_experiment."
        )

        def warmup(fn, model, example_inputs, mode, niters=10):
            gc.collect()
            peak_mem = 0
            start_stats = get_dynamo_stats()
            try:
                if current_device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    empty_gpu_cache(current_device)
                elif current_device == "hpu":
                    torch.hpu.reset_peak_memory_stats()
                t0 = time.perf_counter()
                for _ in range(niters):
                    fn(model, example_inputs)
                t1 = time.perf_counter()
                latency = t1 - t0
                if current_device == "cuda":
                    peak_mem = get_peak_memory()
                elif current_device == "hpu":
                    peak_mem = torch.hpu.max_memory_allocated() / 10**9
                elif current_device == "cpu":
                    total = psutil.virtual_memory().total
                    percentage = psutil.Process(os.getpid()).memory_percent()
                    peak_mem = percentage * total / 10**9
            except Exception:
                log.exception("Backend %s failed in warmup()", mode)
                write_csv_when_exception(
                    self.args, current_name, "warmup_failed", current_device
                )
                output_signpost({}, self.args, self.suite_name, error="warmup_failed")
                return sys.exit(-1)
            dynamo_stats = get_dynamo_stats()
            dynamo_stats.subtract(start_stats)
            return latency, peak_mem, dynamo_stats

        # Cast the model to float16/float32 as necessary
        model, example_inputs = self.maybe_cast(model, example_inputs)

        # Use distributed wrapping as necessary
        model = self.deepcopy_and_maybe_parallelize(model)

        if not hasattr(model, name):
            model.name = name
        self.init_optimizer(name, current_device, model.parameters())

        # The self.autocast context is needed for the model we export with aot_compile,
        # similar to what we do in the check_accuracy function
        ctx = (
            self.autocast(**self.autocast_arg)
            if self.args.export_aot_inductor
            else contextlib.nullcontext()
        )

        with self.pick_grad(name, self.args.training), ctx:
            ok, total = Stats.reset_counters()
            experiment_kwargs = {}
            if tag is not None:
                experiment_kwargs["tag"] = tag
            results = []

            with maybe_snapshot_memory(
                self.args.snapshot_memory, f"eager_{self.args.only}"
            ):
                eager_latency, eager_peak_mem, _ = warmup(
                    self.model_iter_fn, model, example_inputs, "eager"
                )
                if self.args.use_warm_peak_memory:
                    _, eager_peak_mem, _ = warmup(
                        self.model_iter_fn, model, example_inputs, "eager", niters=1
                    )

            baseline_timings = experiment(
                self.model_iter_fn,
                model,
                example_inputs,
                mark="expected",
                **experiment_kwargs,
            )

            if self.args.export_aot_inductor:
                optimized_model_iter_fn = optimize_ctx
            else:
                optimized_model_iter_fn = optimize_ctx(self.model_iter_fn)

            with maybe_snapshot_memory(
                self.args.snapshot_memory, f"compiled_{self.args.only}"
            ):
                dynamo_latency, dynamo_peak_mem, dynamo_stats = warmup(
                    optimized_model_iter_fn, model, example_inputs, "dynamo"
                )
                if self.args.use_warm_peak_memory:
                    _, dynamo_peak_mem, _ = warmup(
                        optimized_model_iter_fn,
                        model,
                        example_inputs,
                        "dynamo",
                        niters=1,
                    )
                # If we use warm peak memory, the AOT model loading transient memory
                # won't be present on the warm measurement.  We only have to account for
                # it when using cold memory.
                elif self.args.export_aot_inductor:
                    dynamo_peak_mem -= AOTInductorModelCache.get_excess_memory(model)

            if self.args.profile_dynamo_cache_lookup:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as prof:
                    warmup(optimized_model_iter_fn, model, example_inputs, "dynamo")

                events = list(
                    filter(
                        lambda event: "TorchDynamo Cache Lookup" in event.key,
                        prof.key_averages(),
                    )
                )
                dynamo_cache_lookup_latency = events[0].self_cpu_time_total

            compilation_time = dynamo_latency - eager_latency
            compression_ratio = (
                eager_peak_mem / dynamo_peak_mem if dynamo_peak_mem else 0.0
            )
            if self.args.print_memory:
                print(
                    f"memory: eager: {eager_peak_mem:.2f} GB, "
                    f"dynamo: {dynamo_peak_mem:.2f} GB, "
                    f"ratio: {compression_ratio:.2f}"
                )

            if self.args.print_compilation_time:
                print(f"Compilation time: {compilation_time:.2f}")

            if experiment.func is speedup_experiment:
                experiment_kwargs["compilation_latency"] = compilation_time
                experiment_kwargs["compression_ratio"] = compression_ratio
                experiment_kwargs["eager_peak_mem"] = eager_peak_mem
                experiment_kwargs["dynamo_peak_mem"] = dynamo_peak_mem
                experiment_kwargs["dynamo_stats"] = dynamo_stats
                if self.args.profile_dynamo_cache_lookup:
                    experiment_kwargs["cache_lookup_latency"] = (
                        dynamo_cache_lookup_latency
                    )

            backend_timings = experiment(
                self.model_iter_fn,
                model,
                example_inputs,
                mark="expected",
                **experiment_kwargs,
            )
            timings = np.stack((baseline_timings, backend_timings), axis=1)
            result_summary = latency_experiment_summary(
                self.suite_name, self.args, model, timings, **experiment_kwargs
            )
            results.append(result_summary)
            return " ".join(map(str, results))

    def run_performance_test(
        self,
        name,
        model,
        example_inputs,
        optimize_ctx,
        experiment,
        tag=None,
        batch_size=None,
    ):
        niters = 5
        if getattr(self, "hf_llm", False):
            # If we're benchmarking an llm, we want to use the generate function
            self.model_iter_fn = self.generate
            niters = 1

        if self.args.xla:
            with self.pick_grad(name, self.args.training):
                return experiment(
                    self.model_iter_fn, *self.maybe_cast(model, example_inputs)
                )

        def warmup(fn, model, example_inputs, mode, niters=5):
            gc.collect()
            peak_mem = 0
            start_stats = get_dynamo_stats()
            try:
                if current_device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    empty_gpu_cache(current_device)
                elif current_device == "hpu":
                    torch.hpu.reset_peak_memory_stats()
                t0 = time.perf_counter()
                for _ in range(niters):
                    fn(model, example_inputs)
                t1 = time.perf_counter()
                latency = t1 - t0
                if current_device == "cuda":
                    peak_mem = get_peak_memory()
                elif current_device == "hpu":
                    peak_mem = torch.hpu.max_memory_allocated() / 10**9
                elif current_device == "cpu":
                    total = psutil.virtual_memory().total
                    percentage = psutil.Process(os.getpid()).memory_percent()
                    peak_mem = percentage * total / 10**9
            except Exception:
                log.exception("Backend %s failed in warmup()", mode)
                write_csv_when_exception(
                    self.args, current_name, "warmup_failed", current_device
                )
                output_signpost({}, self.args, self.suite_name, error="warmup_failed")
                return sys.exit(-1)
            dynamo_stats = get_dynamo_stats()
            dynamo_stats.subtract(start_stats)
            return latency, peak_mem, dynamo_stats

        # Cast the model to float16/float32 as necessary
        model, example_inputs = self.maybe_cast(model, example_inputs)

        # Use distributed wrapping as necessary
        model = self.deepcopy_and_maybe_parallelize(model)

        if not hasattr(model, name):
            model.name = name

        self.init_optimizer(name, current_device, model.parameters())

        # The self.autocast context is needed for the model we export with aot_compile,
        # similar to what we do in the check_accuracy function
        ctx = (
            self.autocast(**self.autocast_arg)
            if self.args.export_aot_inductor
            else contextlib.nullcontext()
        )

        with self.pick_grad(name, self.args.training), ctx:
            ok, total = Stats.reset_counters()
            experiment_kwargs = {}
            experiment_kwargs["batch_size"] = batch_size
            if tag is not None:
                experiment_kwargs["tag"] = tag
            results = []
            with maybe_snapshot_memory(
                self.args.snapshot_memory, f"eager_{self.args.only}"
            ):
                with torch.compiler.set_stance("force_eager"):
                    eager_latency, eager_peak_mem, _ = warmup(
                        self.model_iter_fn,
                        copy.deepcopy(model),
                        example_inputs,
                        "eager",
                        niters=niters,
                    )
                    if self.args.use_warm_peak_memory:
                        _, eager_peak_mem, _ = warmup(
                            self.model_iter_fn,
                            copy.deepcopy(model),
                            example_inputs,
                            "eager",
                            niters=1,
                        )

            if (
                self.args.export_aot_inductor
                or self.args.export_nativert
                or self.args.torchscript_jit_trace
            ):
                optimized_model_iter_fn = optimize_ctx
            else:
                if getattr(self, "hf_llm", False):
                    # If it's an llm, we want to optimize model.forward, and use
                    # the generate function
                    model = optimize_ctx(model)
                    optimized_model_iter_fn = self.model_iter_fn
                else:
                    optimized_model_iter_fn = optimize_ctx(self.model_iter_fn)

            with maybe_snapshot_memory(
                self.args.snapshot_memory, f"compiled_{self.args.only}"
            ):
                dynamo_latency, dynamo_peak_mem, dynamo_stats = warmup(
                    optimized_model_iter_fn, model, example_inputs, "dynamo"
                )
                if self.args.use_warm_peak_memory:
                    _, dynamo_peak_mem, _ = warmup(
                        optimized_model_iter_fn,
                        model,
                        example_inputs,
                        "dynamo",
                        niters=1,
                    )
                # If we use warm peak memory, the AOT model loading transient memory
                # won't be present on the warm measurement.  We only have to account for
                # it when using cold memory.
                elif self.args.export_aot_inductor:
                    dynamo_peak_mem -= AOTInductorModelCache.get_excess_memory(model)

            if self.args.profile_dynamo_cache_lookup:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as prof:
                    warmup(optimized_model_iter_fn, model, example_inputs, "dynamo")

                events = list(
                    filter(
                        lambda event: "TorchDynamo Cache Lookup" in event.key,
                        prof.key_averages(),
                    )
                )
                dynamo_cache_lookup_latency = events[0].self_cpu_time_total

            compilation_time = dynamo_latency - eager_latency
            compression_ratio = (
                eager_peak_mem / dynamo_peak_mem if dynamo_peak_mem else 0.0
            )
            if self.args.print_memory:
                print(
                    f"memory: eager: {eager_peak_mem:.2f} GB, "
                    f"dynamo: {dynamo_peak_mem:.2f} GB, "
                    f"ratio: {compression_ratio:.2f}"
                )

            if self.args.print_compilation_time:
                print(f"Compilation time: {compilation_time:.2f}")

            if experiment.func is speedup_experiment:
                experiment_kwargs["compilation_latency"] = compilation_time
                experiment_kwargs["compression_ratio"] = compression_ratio
                experiment_kwargs["eager_peak_mem"] = eager_peak_mem
                experiment_kwargs["dynamo_peak_mem"] = dynamo_peak_mem
                experiment_kwargs["dynamo_stats"] = dynamo_stats
                if self.args.profile_dynamo_cache_lookup:
                    experiment_kwargs["cache_lookup_latency"] = (
                        dynamo_cache_lookup_latency
                    )

            if experiment.func is coverage_experiment:
                ok, total = Stats.reset_counters()
                results = []
                # run with torch._dynamo few times to populate the cache
                for _ in range(3):
                    optimized_model_iter_fn(model, example_inputs)
                _, frames_second_pass = Stats.reset_counters()  # should be 0
                if frames_second_pass > 0:
                    optimized_model_iter_fn(model, example_inputs)
                    _, frames_third_pass = Stats.reset_counters()  # should be 0
                else:
                    frames_third_pass = 0

                results.append(
                    f"{ok:3}/{total:3} +{frames_third_pass} frames {compilation_time:3.0f}s"
                )

            experiment_kwargs["hf_llm"] = getattr(self, "hf_llm", False)

            results.append(
                experiment(
                    self.model_iter_fn, model, example_inputs, **experiment_kwargs
                )
            )
            return " ".join(map(str, results))

    def minify_model(
        self,
        name,
        model,
        example_inputs,
        optimize_ctx,
        experiment,
        tag,
    ):
        log.info("Minifying %s...", name)
        os.environ["TORCH_COMPILE_DEBUG"] = "1"
        os.environ["TORCHDYNAMO_REPRO_AFTER"] = "dynamo"
        os.environ["TORCHDYNAMO_REPRO_LEVEL"] = "4"

        self.check_accuracy(name, model, example_inputs, optimize_ctx, experiment, tag)

        if self.args.output_directory:
            repro_dir = self.args.output_directory
        else:
            repro_dir = torch._dynamo.config.base_dir

        try:
            shutil.move("repro.py", f"{repro_dir}/{name}_repro.py")
        except OSError:
            log.error("Could not find repro script for model %s", name)
        else:
            log.info(
                "Repro script for model %s with minified graph saved to %s",
                name,
                repro_dir,
            )

    def maybe_preserve_compile_debug(self, name, status):
        if (
            name in CI_PRESERVE_COMPILE_DEBUG
            and status in CI_PRESERVE_COMPILE_DEBUG[name]
        ):
            src_dir = torch._dynamo.utils.get_debug_dir()
            if os.path.isdir(src_dir):
                dbg_dir = os.path.join(
                    os.getcwd(), "test", "debug", "torch_compile_debug"
                )
                dst_dir = os.path.join(dbg_dir, os.path.basename(src_dir))
                try:
                    os.makedirs(dbg_dir, exist_ok=True)
                    os.rename(src_dir, dst_dir)
                    log.warning("Moved %s to %s", src_dir, dst_dir)
                except OSError:
                    log.exception("Failed to preserve %s", src_dir)

    def run_one_model(
        self,
        name,
        model,
        example_inputs,
        optimize_ctx,
        experiment,
        explain=False,
        tag=None,
        batch_size=None,
    ):
        mode = "train" if self.args.training else "eval"
        msg = f"{current_device:4} {mode:5} {current_name:34} "
        if tag:
            msg += f" {tag:26}"
        print(msg, flush=True)

        start_stats = get_dynamo_stats()

        if self.args.accuracy:
            status = self.check_accuracy(
                name, model, example_inputs, optimize_ctx, experiment, tag
            )
            print(status)
            if status == "fail_accuracy" and self.args.minify:
                self.minify_model(
                    name, model, example_inputs, optimize_ctx, experiment, tag
                )
        elif self.args.tolerance:
            status = self.check_tolerance(name, model, example_inputs, optimize_ctx)
            print(status)
        elif self.args.performance:
            if self.args.backend == "torchao":
                status = self.run_performance_test_non_alternate(
                    name, model, example_inputs, optimize_ctx, experiment, tag
                )
            else:
                status = self.run_performance_test(
                    name,
                    model,
                    example_inputs,
                    optimize_ctx,
                    experiment,
                    tag,
                    batch_size=batch_size,
                )
            print(status)
        empty_gpu_cache(current_device)

        self.maybe_preserve_compile_debug(name, status)

        if self.args.timing:
            from torch._dynamo.utils import op_count, print_time_report
            from torch.utils._stats import simple_call_counter

            print_time_report()
            stats = "STATS: "
            stats = stats + " | ".join(
                itertools.chain(
                    [f"call_* op count: {op_count}"],
                    (f"{key}:{value}" for key, value in simple_call_counter.items()),
                )
            )
            print(stats)
        stats = get_dynamo_stats()
        stats.subtract(start_stats)

        if explain:
            print(
                f"Dynamo produced {stats['unique_graphs']} graphs "
                f"covering {stats['calls_captured']} ops with "
                f"{stats['graph_breaks']} graph breaks ({stats['unique_graph_breaks']} unique)"
            )

        if explain or self.args.log_graph_breaks or self.args.print_graph_breaks:
            filename = f"{output_filename.rstrip('.csv')}_graph_breaks.csv"

            def add_double_quotes(x):
                # Delimiter because reason could have comma
                return f'"{x}"'

            for graph_break in graph_break_reasons:
                reason = add_double_quotes(graph_break.reason)
                user_stack = add_double_quotes(
                    ", ".join([str(x) for x in graph_break.user_stack])
                )

                # NB: Don't upload them to the benchmark database as they are debugging
                # information. There are also around a million records a day which is
                # wasteful to store
                write_outputs(
                    filename,
                    ["model", "reason", "user_stack"],
                    [current_name, reason, user_stack],
                    False,
                )

        if self.args.stats:
            Stats.print_summary()


def help(fn):
    return fn.__doc__


diff_branch_default = "DIFF-BRANCH-DEFAULT"


def should_diff_branch(args):
    return args.diff_branch != diff_branch_default


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter", "-k", action="append", help="filter benchmarks with regexp"
    )
    parser.add_argument(
        "--exclude", "-x", action="append", help="filter benchmarks with regexp"
    )
    parser.add_argument(
        "--exclude-exact", action="append", help="filter benchmarks with exact match"
    )
    parser.add_argument(
        "--total-partitions",
        type=int,
        default=1,
        choices=range(1, 16),
        help="Total number of partitions we want to divide the benchmark suite into",
    )
    parser.add_argument(
        "--partition-id",
        type=int,
        default=0,
        help="ID of the benchmark suite partition to be run. Used to divide CI tasks",
    )
    parser.add_argument(
        "--devices", "--device", "-d", action="append", help="cpu, cuda or hpu"
    )
    parser.add_argument("--device-index", help="CUDA device index")
    parser.add_argument(
        "--repeat", "-n", type=int, default=30, help="number of timing runs"
    )
    iterations_per_run_help = """
        Run this may iterations for each time measurement. This is mainly used for
        XLA training. We want to run multiple iterations per measurement so the
        tracing and computation for different iterations can overlap with each
        other. This makes sure we have an accurate xla baseline.
    """
    parser.add_argument(
        "--iterations-per-run", type=int, default=1, help=iterations_per_run_help
    )
    parser.add_argument(
        "--randomize-input",
        action="store_true",
        help="Whether to randomize the input values. Dimensions will be kept the same.",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        help="number of threads to use for eager and inductor",
    )
    parser.add_argument(
        "--nopython", action="store_true", help="Turn graph breaks into errors"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="run models that are in the global SKIP list",
    )
    parser.add_argument(
        "--prims-nvfuser", action="store_true", help="user prims + nvfuser backend"
    )
    parser.add_argument(
        "--dump-raw-metrics",
        action="store_true",
        help="dump raw timing metrics from speedup experiment",
    )
    parser.add_argument(
        "--log-operator-inputs",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        default=False,
        help="use channels last format",
    )
    parser.add_argument(
        "--batch-size", "--batch_size", type=int, help="batch size for benchmarking"
    )
    parser.add_argument(
        "--iterations", type=int, default=2, help="how many iterations to run"
    )
    parser.add_argument(
        "--batch-size-file", type=str, help="String to load batch size from"
    )
    parser.add_argument("--cosine", action="store_true", help="use cosine similarity")
    parser.add_argument(
        "--freezing", action="store_true", help="turn on freezing", default=False
    )
    parser.add_argument(
        "--inductor-config",
        "-c",
        action="append",
        help="key=value in torch._inductor.config",
    )
    parser.add_argument(
        "--ci", action="store_true", help="Flag to tell that its a CI run"
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="Flag to tell that its a Dashboard run"
    )
    parser.add_argument(
        "--skip-fp64-check", action="store_true", help="skip accuracy check using fp64"
    )
    parser.add_argument(
        "--fast", "-f", action="store_true", help="skip slow benchmarks"
    )
    parser.add_argument(
        "--only",
        help="""Run just one model from torchbench. Or
        specify the path and class name of the model in format like:
        --only=path:<MODEL_FILE_PATH>,class:<CLASS_NAME>

        Due to the fact that dynamo changes current working directory,
        the path should be an absolute path.

        The class should have a method get_example_inputs to return the inputs
        for the model. An example looks like
        ```
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

            def get_example_inputs(self):
                return (torch.randn(2, 10),)
        ```
    """,
    )
    parser.add_argument(
        "--multiprocess",
        action="store_true",
        help="Create n processes based on the number of devices (distributed use case).",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Wraps model in DDP before running it, and uses dynamo DDPOptmizer (graph breaks) by default.",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="""Wraps model in FSDP before running it.
        Doesn't recursively wrap, mainly useful for checking dynamo UnspecNNModule compatibility
    """,
    )
    parser.add_argument(
        "--optimize-ddp-mode",
        type=str,
        default="ddp_optimizer",
        help="Specify the DDP optimization mode -- the value of torch._dynamo.config.optimize_ddp.",
    )
    parser.add_argument(
        "--distributed-master-port",
        default="6789",
        help="Port to bind for for torch.distributed.  Use the default unless it's conflicting with another user",
    )
    parser.add_argument(
        "--dynamic-shapes",
        action="store_true",
        help="Runs a dynamic shapes version of the benchmark, if available.",
    )
    parser.add_argument(
        "--propagate-real-tensors",
        action="store_true",
        help="Capture as much data dependent as you can by unsoundly propagating real tensors",
    )
    parser.add_argument(
        "--dynamic-batch-only",
        action="store_true",
        help="Only assume batch dimension is dynamic.  Implies --dynamic-shapes",
    )
    parser.add_argument(
        "--specialize-int", action="store_true", help="Run with specialize_int=True."
    )
    parser.add_argument(
        "--use-eval-mode",
        action="store_true",
        help="sets model.eval() to reduce randomness",
    )
    parser.add_argument(
        "--skip-accuracy-check",
        action="store_true",
        help="keeps running even when accuracy fails",
    )
    parser.add_argument(
        "--generate-aot-autograd-stats",
        action="store_true",
        help="Generates AOT Autograd stats like how many graphs are sent to AOT",
    )
    parser.add_argument(
        "--inductor-settings",
        action="store_true",
        help="Use same settings as --inductor for baseline comparisons",
    )
    parser.add_argument(
        "--suppress-errors",
        action="store_true",
        help="Suppress errors instead of raising them",
    )
    parser.add_argument(
        "--output",
        help="Overrides the output filename",
    )
    parser.add_argument(
        "--output-directory",
        help="Overrides the directory to place output files.",
    )
    parser.add_argument(
        "--disable-output",
        action="store_true",
        help="Disable writing of output files, e.g., for warm-up runs",
    )
    parser.add_argument(
        "--baseline",
        help="Compare with a prior --output",
    )
    parser.add_argument(
        "--part",
        default=None,
        help="Specify the part of the model to run.",
    )
    parser.add_argument(
        "--export-profiler-trace",
        action="store_true",
        help="exports trace of kineto profiler",
    )
    parser.add_argument(
        "--profiler-trace-name",
        "--profiler_trace_name",
        help="Overwrites exported trace name",
    )
    parser.add_argument(
        "--profile-details", action="store_true", help="More detailed profiler trace."
    )
    parser.add_argument(
        "--export-perfdoctor",
        action="store_true",
        help="Export Chrome trace to perf doctor. (internal only)",
    )
    parser.add_argument(
        "--diff-branch",
        default=diff_branch_default,
        help="delta current branch against given branch.",
    )
    parser.add_argument(
        "--tag", default=None, help="Specify a tag to be included in csv files."
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="print some graph/op statistics during the run, similar to .explain()",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="print graph counter stats",
    )
    parser.add_argument(
        "--use-warm-peak-memory",
        "--use_warm_peak_memory",
        action="store_true",
        help="Measure peak memory using a warm run to reduce autotuning noise",
    )
    parser.add_argument(
        "--print-memory",
        action="store_true",
        help="print extra memory statistics",
    )
    parser.add_argument(
        "--print-compilation-time",
        action="store_true",
        help="print compilation latency",
    )
    parser.add_argument(
        "--print-dataframe-summary",
        action="store_true",
        help="print dataframe result used for calculating accuracy",
    )
    parser.add_argument(
        "--disable-cudagraphs",
        action="store_true",
        help="Disables cudagraphs for Inductor",
    )
    parser.add_argument(
        "--disable-split-reductions",
        action="store_true",
        help="Disables split reductions for Inductor",
    )
    parser.add_argument(
        "--disable-persistent-reductions",
        action="store_true",
        help="Disables split reductions for Inductor",
    )
    parser.add_argument(
        "--disable-divisible-by-16",
        action="store_true",
        help="Disables divisible by 16 hint to Triton for Inductor",
    )
    parser.add_argument(
        "--inductor-compile-mode",
        default=None,
        help="torch.compile mode argument for inductor runs.",
    )
    parser.add_argument(
        "--print-graph-breaks",
        action="store_true",
        help="Show a warning whenever graph break",
    )
    parser.add_argument(
        "--log-graph-breaks",
        action="store_true",
        help="log graph breaks in a file",
    )
    parser.add_argument(
        "--trace-on-xla",
        action="store_true",
        help="Whether to trace the model on XLA or on eager device",
    )
    parser.add_argument(
        "--xla-tolerance",
        type=float,
        default=1e-2,
        help="XLA needs a loose tolerance to pass the correctness check",
    )
    parser.add_argument(
        "--collect-outputs",
        action="store_true",
        help="""Whether to collect outputs for training. Set this to true if we
        want to verify the numerical correctness of graidents. But that may
        cause time measurement not accurate""",
    )
    parser.add_argument(
        "--enable-activation-checkpointing",
        action="store_true",
        help="Enables activation checkpointing for HF models",
    )
    parser.add_argument("--timing", action="store_true", help="Emits phase timing")

    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print n/k models message between each model run.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=2000,
        help="timeout (second) for benchmarking.",
    )

    parser.add_argument(
        "--per_process_memory_fraction",
        type=float,
        default=1,
        help="Set per-process GPU memory fraction (limit) for reducing usable size and reproducing OOMs",
    )

    parser.add_argument(
        "--no-translation-validation",
        action="store_true",
        help="Disable translation validation for accuracy builds.",
    )

    parser.add_argument(
        "--minify",
        action="store_true",
        help="Enable minification when failure is below tolerance. Save repro script for each model.",
    )

    parser.add_argument(
        "--compiled-autograd",
        action="store_true",
        help="Enables compiled autograd on compiled benchmark",
    )

    parser.add_argument(
        "--profile_dynamo_cache_lookup",
        "--profile-dynamo-cache-lookup",
        action="store_true",
        help="profiles TorchDynamo cache lookup",
    )

    parser.add_argument(
        "--snapshot-memory",
        "--snapshot_memory",
        action="store_true",
        help="Enables Memory Snapshot tool for memory deep dives: https://pytorch.org/blog/understanding-gpu-memory-1/",
    )

    parser.add_argument(
        "--retain-output",
        action="store_true",
        help="Enables appending to the already existing output file if it exists \
            instead of deleting it and creating a new one.",
    )

    parser.add_argument(
        "--caching-precompile",
        action="store_true",
        help="Enables caching precompile, serializing artifacts to DynamoCache between runs",
    )

    group_latency = parser.add_mutually_exclusive_group()
    group_latency.add_argument(
        "--cold-start-latency",
        "--cold_start_latency",
        action="store_true",
        help="Use a fresh triton cachedir when running each model, to force cold-start compile.",
    )
    group_latency.add_argument(
        "--warm-start-latency",
        "--warm_start_latency",
        action="store_true",
        help="Run model(s) twice and preserve caches in between to enable a 'warm start' on the 2nd run",
    )

    group_fuser = parser.add_mutually_exclusive_group()
    # --nvfuser is now the default, keep the option to not break scripts
    group_fuser.add_argument("--nvfuser", action="store_true", help=argparse.SUPPRESS)
    group_fuser.add_argument("--nnc", action="store_true", help="enable NNC for GPUs")

    group_prec = parser.add_mutually_exclusive_group()
    group_prec.add_argument("--float16", action="store_true", help="cast model to fp16")
    group_prec.add_argument(
        "--bfloat16", action="store_true", help="cast model to bf16"
    )
    group_prec.add_argument("--float32", action="store_true", help="cast model to fp32")
    group_prec.add_argument(
        "--amp", action="store_true", help="use automatic mixed precision"
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16"),
        help="the data type used with automatic mixed precision",
    )
    group_printout = parser.add_mutually_exclusive_group()
    group_printout.add_argument(
        "--verbose", "-v", action="store_true", help="enable verbose debug printouts"
    )
    group_printout.add_argument(
        "--quiet", "-q", action="store_true", help="suppress debug printouts"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--coverage", action="store_true", help="(default) " + help(coverage_experiment)
    )
    group.add_argument(
        "--overhead", action="store_true", help=help(overhead_experiment)
    )
    group.add_argument(
        "--speedup-dynamo-ts",
        action="store_true",
        help="TorchDynamo frontend with torchscript backend",
    )
    group.add_argument(
        "--speedup-fx2trt", action="store_true", help=help(speedup_experiment_fx2trt)
    )
    group.add_argument(
        "--speedup-fx2trt-fp16",
        action="store_true",
        help=help(speedup_experiment_fx2trt),
    )
    group.add_argument(
        "--print-fx",
        action="store_true",
        help="Print fx traces captured from model",
    )
    group.add_argument(
        "--print-aten-ops",
        action="store_true",
        help="Print traces of aten ops captured by AOT autograd",
    )
    group.add_argument(
        "--inductor",
        action="store_true",
        help="Measure speedup with TorchInductor",
    )
    group.add_argument(
        "--quantization",
        choices=[
            "int8dynamic",
            "int8weightonly",
            "int4weightonly",
            "autoquant",
            "noquant",
        ],
        default=None,
        help="Measure speedup of torchao quantization with TorchInductor baseline",
    )
    group.add_argument(
        "--export",
        action="store_true",
        help="Measure pass rate with export",
    )
    group.add_argument(
        "--export-aot-inductor",
        action="store_true",
        help="Measure pass rate with Export+AOTInductor",
    )
    group.add_argument(
        "--export-nativert",
        action="store_true",
        help="Measure pass rate with Export+NativeRT",
    )
    group.add_argument(
        "--torchscript-jit-trace",
        action="store_true",
        help="Measure pass rate with TorchScript jit.trace",
    )
    group.add_argument(
        "--xla", action="store_true", help="Compare TorchXLA to eager PyTorch"
    )
    group.add_argument(
        "--backend",
        choices=torch._dynamo.list_backends(exclude_tags=None),
        help="measure speedup with a given backend",
    )
    group.add_argument("--nothing", action="store_true", help=help(null_experiment))
    group.add_argument(
        "--log-conv-args",
        action="store_true",
        help="Dump convolution input/weight/bias's shape/stride/dtype and other options to json",
    )
    group.add_argument(
        "--recompile-profiler",
        "--recompile_profiler",
        action="store_true",
        help="Run the dynamo recompilation profiler on each model.",
    )
    group.add_argument(
        "--find-batch-sizes",
        action="store_true",
        help="finds the largest batch size that could fit on GPUs",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--accuracy",
        action="store_true",
        help="Checks accuracy with small batch size and eval mode",
    )
    mode_group.add_argument(
        "--performance", action="store_true", help="Measures performance speedup"
    )
    mode_group.add_argument(
        "--tolerance",
        action="store_true",
        help="extracts the tolerance for each model with small batch size and eval mode",
    )
    run_mode_group = parser.add_mutually_exclusive_group(required=True)
    run_mode_group.add_argument(
        "--training",
        action="store_true",
        help="Performs training",
    )
    run_mode_group.add_argument(
        "--inference", action="store_true", help="Performs inference"
    )
    return parser.parse_args(args)


def process_caching_precompile():
    """
    After every process_entry, save precompile artifacts to DynamoCache
    """
    assert torch._dynamo.config.caching_precompile, (
        "Caching precompile should be enabled with --caching-precompile"
    )
    from torch._dynamo.precompile_context import PrecompileContext

    debug_info = PrecompileContext.save_to_dynamo_cache()
    print(
        f"Saved {len(debug_info['dynamo'])} precompile artifacts with {len(debug_info['backends'])} backends"
    )


def process_entry(rank, runner, original_dir, args):
    args.rank = rank
    with maybe_init_distributed(
        args.init_distributed,
        rank=rank,
        world_size=args.world_size,
        port=args.distributed_master_port,
    ):
        result = run(runner, args, original_dir)
        if args.caching_precompile:
            process_caching_precompile()
        return result


def maybe_fresh_cache(args):
    cache_dir_assigned = "TORCHINDUCTOR_CACHE_DIR" in os.environ
    if not cache_dir_assigned and (
        args.cold_start_latency or args.warm_start_latency or args.ci
    ):
        return fresh_cache()
    else:
        return contextlib.nullcontext()


def main(runner, original_dir=None, args=None):
    if original_dir:
        os.chdir(original_dir)
    args = parse_args() if not args else parse_args(args)
    if args.baseline:
        args.baseline = os.path.abspath(args.baseline)

    if should_diff_branch(args):
        import git

        # We do this here so we error out earlier if there's an issue
        repo = git.Repo()
        if repo.is_dirty():
            raise RuntimeError(
                "--diff-branch called on dirty branch. Commit, stash, or reset."
            )
        main_branch = repo.active_branch.name
        if main_branch == args.diff_branch:
            raise RuntimeError(
                f"--diff-branch: current branch is same as {args.diff_branch} branch, what are you diffing?"
            )

    with maybe_fresh_cache(args):
        if args.caching_precompile:
            os.environ["TORCH_CACHING_PRECOMPILE"] = "1"
            torch._dynamo.config.caching_precompile = True

        args.init_distributed = args.only and args.multiprocess
        if args.init_distributed:
            # NB: Do NOT query device count before CUDA initialization; we're
            # going to overwrite CUDA_VISIBLE_DEVICES and this will result in
            # https://github.com/pytorch/pytorch/issues/107300
            device_count = torch.cuda.device_count()
            if device_count <= 1:
                log.warning(
                    "The use multiprocess flag is set but there are <= 1 devices available."
                )
            # multiprocess path
            args.world_size = device_count
            mp.spawn(
                process_entry, args=(runner, original_dir, args), nprocs=device_count
            )
        elif args.only and args.warm_start_latency:
            # Warm start mode. Enable FX graph caching and perform back-to-back runs in
            # separate processes (but ensure the inductor cache is preserved across runs).
            env = os.environ.copy()
            env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
            cmd = [sys.executable] + sys.argv
            cmd.remove("--warm-start-latency")

            print(f"Performing cold-start run for {args.only}")
            warmup_cmd = cmd + ["--repeat=1", "--disable-output"]
            subprocess.check_call(warmup_cmd, timeout=args.timeout, env=env)

            print(f"Performing warm-start run for {args.only}")
            subprocess.check_call(cmd, timeout=args.timeout, env=env)
        else:
            # single process path just uses the main process
            args.world_size = 1
            process_entry(0, runner, original_dir, args)


def write_csv_when_exception(args, name: str, status: str, device=None):
    print(status)
    placeholder_batch_size = 0
    devices = [device] if device is not None else args.devices
    if args.accuracy:
        headers = ["dev", "name", "batch_size", "accuracy"]
        rows = [[device, name, placeholder_batch_size, status] for device in devices]
    elif args.performance:
        headers = ["dev", "name", "batch_size", "speedup", "abs_latency"]
        rows = [[device, name, placeholder_batch_size, 0.0, 0.0] for device in devices]
    else:
        headers = []
        rows = [[device, name, placeholder_batch_size, 0.0] for device in devices]

    for row in rows:
        write_outputs(output_filename, headers, row)


def run(runner, args, original_dir=None):
    # Pass the parsed args object to benchmark runner object
    torch._dynamo.reset()
    runner.args = args

    args.filter = args.filter or [r"."]
    args.exclude = args.exclude or [r"^$"]
    args.exclude_exact = args.exclude_exact or []

    if args.inductor:
        assert args.backend is None
        args.backend = "inductor"
    if args.quantization:
        assert args.backend is None
        args.backend = "torchao"
    if args.dynamic_batch_only:
        args.dynamic_shapes = True
        torch._dynamo.config.assume_static_by_default = True
    if args.dynamic_shapes:
        if not args.dynamic_batch_only:
            torch._dynamo.config.assume_static_by_default = False
    if args.compiled_autograd:
        torch._dynamo.config.compiled_autograd = True
    if args.propagate_real_tensors:
        # TODO: Separate flag for data dependent
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._functorch.config.fake_tensor_propagate_real_tensors = True
    if args.specialize_int:
        torch._dynamo.config.specialize_int = True
    if args.ci:
        if args.accuracy:
            # Run fewer iterations when checking accuracy
            args.repeat = min(args.repeat, 2)

            # Set translation validation on by default on CI accuracy runs.
            torch.fx.experimental._config.translation_validation = True

    if args.ddp:
        assert args.training, "DDP benchmark requires --training mode"
        torch._dynamo.config.optimize_ddp = args.optimize_ddp_mode
        if args.only == "dlrm":
            log.error(
                "DLRM+DDP is unsupported as it requires sharding the embedding layer separately from DDP"
            )
            return sys.exit(-1)
    if args.accuracy:
        # Use small batch size. We use >1 batch size to ensure we test
        # batch_norm type of operators that work on batch dims.
        # TODO - Go through the failures for batch size = 2
        if args.batch_size is None:
            if runner.suite_name == "huggingface":
                args.batch_size = 1
            elif runner.suite_name == "torchbench":
                args.batch_size = 4
            else:
                # Larger batch size of TIMM models to have stable batch_norm
                assert runner.suite_name == "timm_models"
                args.batch_size = 8

        # Remove sources of randomness
        if runner.suite_name not in ("timm_models", "huggingface"):
            # TODO - Using train mode for timm_models and HF models. Move to train mode for Torchbench as well.
            args.use_eval_mode = True
        inductor_config.fallback_random = True
        if args.only is not None and args.only not in {
            "alexnet",
            "Background_Matting",
            "pytorch_CycleGAN_and_pix2pix",
            "pytorch_unet",
            "Super_SloMo",
            "vgg16",
            # https://github.com/pytorch/pytorch/issues/96724
            "Wav2Vec2ForCTC",
            "Wav2Vec2ForPreTraining",
            "sam",
            "sam_fast",
            "resnet50_quantized_qat",
            "mobilenet_v2_quantized_qat",
            "detectron2_maskrcnn",
            "detectron2_maskrcnn_r_101_c4",
            "detectron2_maskrcnn_r_101_fpn",
            "detectron2_maskrcnn_r_50_c4",
            "detectron2_maskrcnn_r_50_fpn",
            "detectron2_fasterrcnn_r_101_c4",
            "detectron2_fasterrcnn_r_101_dc5",
            "detectron2_fasterrcnn_r_101_fpn",
            "detectron2_fasterrcnn_r_50_c4",
            "detectron2_fasterrcnn_r_50_dc5",
            "detectron2_fasterrcnn_r_50_fpn",
        }:
            # some of the models do not support use_deterministic_algorithms
            torch.use_deterministic_algorithms(True)
        if args.devices == ["xpu"]:
            torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if args.only is not None and args.only in {
            "nvidia_deeprecommender",
        }:
            # These seem unhappy with numerics of larger cuBLASLt workspace
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)

        torch.backends.mkldnn.deterministic = True

        # Remove randomness when torch manual seed is called
        patch_torch_manual_seed()

        # Some models e.g. yolov3 assert batch size on n_gpus
        if "CUDA_VISIBLE_DEVICES" not in os.environ and not args.multiprocess:
            args.device_index = "0"

        # Stricter check to disable fallbacks
        args.suppress_errors = False

        if not args.disable_cudagraphs:
            runner.skip_models.update(
                {
                    # xfail: https://github.com/pytorch/pytorch/issues/145773
                    "llama",
                    "cm3leon_generate",
                }
            )

    if args.device_index is not None:
        if args.multiprocess:
            print("Cannot specify both --device_index and --multiprocess")
            return sys.exit(-1)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_index

    elif args.performance:
        # Ensure that we test on real scenarios
        args.use_eval_mode = False

    if args.partition_id > args.total_partitions or args.partition_id < 0:
        print("Invalid partition id")
        return sys.exit(-1)

    if not args.devices:
        if torch.cuda.is_available():
            args.devices = ["cuda"]
        else:
            log.warning("torch.cuda.is_available() == False, using CPU")
            args.devices = ["cpu"]

    if args.devices != ["cpu"] and (HAS_CUDA or HAS_XPU):
        global synchronize
        synchronize = torch.cuda.synchronize if HAS_CUDA else torch.xpu.synchronize

    if args.nnc:
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(True)
        torch._C._jit_set_nvfuser_enabled(False)

    if args.threads:
        torch.set_num_threads(args.threads)

    if args.verbose:
        torch._logging.set_logs(dynamo=logging.DEBUG)

    if args.print_graph_breaks:
        torch._logging.set_logs(graph_breaks=True)

    if args.quiet:
        torch._logging.set_logs(dynamo=logging.ERROR)

    torch._dynamo.config.suppress_errors = args.suppress_errors

    if args.training:
        runner.model_iter_fn = runner.forward_and_backward_pass
        runner.skip_models.update(runner.skip_not_suitable_for_training_models)
    else:
        runner.model_iter_fn = runner.forward_pass

    if args.fast:
        runner.skip_models.update(runner.slow_models)

    if args.devices == ["cpu"]:
        arch = platform.machine()
        runner.skip_models.update(runner.skip_models_for_cpu)
        if arch == "aarch64":
            runner.skip_models.update(runner.skip_models_for_cpu_aarch64)
    elif args.devices == ["cuda"]:
        runner.skip_models.update(runner.skip_models_for_cuda)

    if not args.multiprocess:
        runner.skip_models.update(runner.skip_multiprocess_models)

    if args.freezing:
        if args.devices == ["cpu"]:
            runner.skip_models.update(runner.skip_models_for_freezing_cpu)
        elif args.devices == ["cuda"]:
            runner.skip_models.update(runner.skip_models_for_freezing_cuda)

    if args.no_skip:
        runner.skip_models.clear()

    experiment = null_experiment
    global \
        current_name, \
        current_device, \
        current_batch_size, \
        current_backend, \
        current_mode, \
        current_dtype, \
        current_quantization, \
        current_settings, \
        output_filename, \
        disable_output, \
        optimize_ctx
    optimize_ctx = contextlib.nullcontext()

    if args.disable_output:
        disable_output = True

    if args.overhead:
        optimize_ctx = torch._dynamo.optimize(dummy_fx_compile, nopython=args.nopython)
        experiment = speedup_experiment
        output_filename = "overheads.csv"
    elif args.inductor:
        inductor_config.debug = args.verbose
        if args.threads:
            inductor_config.cpp.threads = args.threads

        optimize_ctx = functools.partial(
            torch.compile,
            backend="inductor",
            fullgraph=args.nopython,
            mode=args.inductor_compile_mode,
        )
        experiment = speedup_experiment
        output_filename = "inductor.csv"
    elif args.export:
        optimize_ctx = export
        experiment = speedup_experiment
        output_filename = "export.csv"
    elif args.export_nativert:
        optimize_ctx = export_nativert
        experiment = speedup_experiment
        output_filename = "export_nativert.csv"
    elif args.torchscript_jit_trace:
        optimize_ctx = torchscript_jit_trace
        experiment = speedup_experiment
        output_filename = "torchscript_jit_trace.csv"
    elif args.xla:
        (dev,) = args.devices
        os.environ["PJRT_DEVICE"] = {"cuda": "GPU", "cpu": "CPU"}[dev]
        torch._dynamo.mark_dynamic = MagicMock()
        experiment = xla
        output_filename = "xla.csv"
    elif args.speedup_dynamo_ts:
        optimize_ctx = torch._dynamo.optimize("ts", nopython=args.nopython)
        experiment = speedup_experiment
        output_filename = "speedup_dynamo_ts.csv"
    elif args.prims_nvfuser:
        optimize_ctx = torch._dynamo.optimize("prims_nvfuser", nopython=args.nopython)
        experiment = speedup_experiment
        backend_str = "prims_nvfuser"
        output_filename = f"accuracy_aot_{backend_str}.csv"
    elif args.print_fx:
        optimize_ctx = torch._dynamo.optimize(
            print_fx,
            nopython=args.nopython,
        )
    elif args.print_aten_ops:
        optimize_ctx = torch._dynamo.optimize(
            print_aten_ops,
            nopython=args.nopython,
        )
    elif args.nothing:
        optimize_ctx = nothing
        experiment = speedup_experiment
        output_filename = "nothing.csv"
    elif args.backend or args.export_aot_inductor:
        if args.export_aot_inductor:
            assert not args.training, "AOTInductor only supports inference"
            optimize_ctx = functools.partial(
                export_aot_inductor, mode=args.inductor_compile_mode
            )

            # AOTInductor doesn't support control flow yet
            runner.skip_models.update(runner.skip_models_due_to_control_flow)
            runner.skip_models.update(runner.skip_models_due_to_export_not_supported)
        elif args.backend == "torchao":
            assert "cuda" in args.devices, "Quantization requires CUDA device."
            assert args.bfloat16, "Quantization requires dtype bfloat16."
            try:
                from torchao_backend import setup_baseline, torchao_optimize_ctx
            except ImportError:
                try:
                    from .torchao_backend import setup_baseline, torchao_optimize_ctx
                except ImportError:
                    from userbenchmark.dynamo.dynamobench.torchao_backend import (
                        setup_baseline,
                        torchao_optimize_ctx,
                    )

            setup_baseline()
            baseline_ctx = functools.partial(
                torch.compile,
                backend="inductor",
                fullgraph=args.nopython,
                mode=args.inductor_compile_mode,
            )
            model_iter_fn = baseline_ctx(runner.model_iter_fn)

            # needed to avoid error that causes inconsistent timing due to:
            # Unable to hit fast path of CUDAGraphs because of pending, uninvoked backwards
            def model_iter_fn_and_mark_step(*args, **kwargs):
                torch.compiler.cudagraph_mark_step_begin()
                model_iter_fn(*args, **kwargs)

            runner.model_iter_fn = model_iter_fn_and_mark_step
            optimize_ctx = torchao_optimize_ctx(args.quantization)
        else:
            optimize_ctx = torch._dynamo.optimize(args.backend, nopython=args.nopython)
        experiment = (
            speedup_experiment if not args.backend == "torchao" else latency_experiment
        )
        if args.accuracy:
            output_filename = f"accuracy_{args.backend}.csv"
        elif args.tolerance:
            output_filename = f"tolerance_{args.backend}.csv"
        else:
            output_filename = f"speedup_{args.backend}.csv"
    elif args.recompile_profiler:
        output_filename = "recompile_profiler_log.csv"
        experiment = recompile_profiler_experiment
    else:
        optimize_ctx = torch._dynamo.optimize(
            fx_insert_profiling, nopython=args.nopython
        )
        experiment = coverage_experiment
        output_filename = "coverage.csv"

    if args.only in runner.disable_cudagraph_models:
        args.disable_cudagraphs = True

    if args.inductor or args.backend == "inductor" or args.export_aot_inductor:
        inductor_config.triton.cudagraphs = not args.disable_cudagraphs
        inductor_config.triton.persistent_reductions = (
            not args.disable_persistent_reductions
        )
        inductor_config.split_reductions = not args.disable_split_reductions
        inductor_config.triton.divisible_by_16 = not args.disable_divisible_by_16
        if args.inference:
            inductor_config.freezing = args.freezing
        if args.inductor_config:
            for config in args.inductor_config:
                key, value = config.split("=")
                typ = type(inductor_config.__getattr__(key))
                if issubclass(typ, bool):
                    assert value in ("0", "1", "True", "False")
                    value = value in ("1", "True")
                elif issubclass(typ, (str, int, float)):
                    value = typ(value)
                else:
                    raise NotImplementedError(typ)
                inductor_config.__setattr__(key, value)

    runner.setup_amp()

    if args.output:
        output_filename = args.output

    if output_filename:
        if args.output_directory:
            output_filename = os.path.join(args.output_directory, output_filename)
        else:
            output_filename = os.path.join(
                torch._dynamo.config.base_dir, output_filename
            )

    if args.find_batch_sizes and args.only:
        for device in args.devices:
            batch_size = runner.batch_size_finder(device, args.only)
            print(args.only, batch_size)
            write_outputs(output_filename, [], [args.only, batch_size])
        return

    args.profile_details = {}
    if args.export_profiler_trace:
        if args.profile_details:
            args.profile_details = {
                "record_shapes": True,
                "profile_memory": True,
                "with_stack": True,
                "with_modules": True,
            }

        if args.profiler_trace_name is None:
            if args.backend:
                args.profiler_trace_name = args.backend
            elif args.inductor:
                args.profiler_trace_name = "inductor"
            else:
                args.profiler_trace_name = "profile"
        else:
            args.profiler_trace_name = args.profiler_trace_name

    if args.no_translation_validation:
        # Overwrite 'translation_validation' config, if specified.
        torch.fx.experimental._config.translation_validation = False

    experiment = functools.partial(experiment, args)

    if args.only and should_diff_branch(args):
        import git

        repo = git.Repo()
        main_branch = repo.active_branch.name
        try:
            # Adding diff-branch again to the args will override previous value
            call_args = (
                [sys.executable] + sys.argv + [f"--diff-branch={diff_branch_default}"]
            )
            # Run for main branch
            subprocess.check_call(call_args + [f"--tag={main_branch}"])
            # Run for comparison branch
            repo.git.checkout(args.diff_branch)
            subprocess.check_call(call_args + [f"--tag={args.diff_branch}"])
        finally:
            # Go back to main branch
            repo.git.checkout(main_branch)
    elif args.only:
        model_name = args.only
        for device in args.devices:
            batch_size = args.batch_size
            if args.batch_size_file:
                batch_size = read_batch_size_from_file(
                    args, args.batch_size_file, model_name
                )
            if model_specified_by_path(args.only):
                model, example_inputs = load_model_from_path(args.only)
                name = model.__class__.__name__
                model = model.to(device=device)
                example_inputs = tree_map_only(
                    torch.Tensor, lambda x: x.to(device=device), example_inputs
                )
            else:
                name = model_name
                try:
                    with tqdm(desc="loading model"):
                        extra_args = []
                        if hasattr(args, "rank") and hasattr(args, "world_size"):
                            extra_args += [
                                "--rank",
                                str(args.rank),
                                "--world_size",
                                str(args.world_size),
                            ]

                        if args.part:
                            (
                                device,
                                name,
                                model,
                                example_inputs,
                                batch_size,
                            ) = runner.load_model(
                                device,
                                model_name,
                                batch_size=batch_size,
                                part=args.part,
                                extra_args=extra_args,
                            )
                        else:
                            if args.fsdp:
                                # Always load model on cpu for fsdp
                                # When initializing FSDP, we will use the cuda device if args.cuda is set
                                (
                                    _,
                                    name,
                                    model,
                                    example_inputs,
                                    batch_size,
                                ) = runner.load_model(
                                    "cpu",
                                    model_name,
                                    batch_size=batch_size,
                                    extra_args=extra_args,
                                )
                            else:
                                (
                                    device,
                                    name,
                                    model,
                                    example_inputs,
                                    batch_size,
                                ) = runner.load_model(
                                    device,
                                    model_name,
                                    batch_size=batch_size,
                                    extra_args=extra_args,
                                )
                except Exception as e:
                    import traceback

                    mode = "train" if args.training else "eval"
                    print(f"{device:4} {mode:5} {name:34} ")
                    print(traceback.format_exc())
                    status = (
                        "model_fail_to_load"
                        if isinstance(e, NotImplementedError)
                        else "eager_fail_to_run"
                    )
                    write_csv_when_exception(args, name, status, device)
                    # NB: current_name/current_device not set, so pass
                    # explicitly
                    output_signpost(
                        {"name": name, "dev": device},
                        args,
                        runner.suite_name,
                        error=status,
                    )
                    continue  # bad benchmark implementation

            if args.trace_on_xla:
                xla_dev = xm.xla_device()
                model = model.to(device=xla_dev)
                example_inputs = tree_map_only(
                    torch.Tensor, lambda x: x.to(device=xla_dev), example_inputs
                )

            current_name = name
            current_device = device
            current_batch_size = batch_size
            current_backend = args.backend
            current_mode = (
                "training" if args.training else "inference" if args.inference else ""
            )
            if args.float16:
                current_dtype = "float16"
            elif args.bfloat16:
                current_dtype = "bfloat16"
            elif args.float32:
                current_dtype = "float32"
            elif args.amp:
                current_dtype = "amp"
            else:
                current_dtype = ""
            current_quantization = args.quantization
            # Keep the remaining of the settings
            current_settings = vars(args)
            set_model_name(name)

            # Look for stuff that looks like batch size, and mark it dynamic.
            # Better integration would integrate directly with benchmark suite
            # but cannot conveniently do this
            # NB: This must be done late enough so that we don't do more
            # conversions on the inputs
            # NB: Assumes only the first batch-y like dimension is the batch
            marked = False

            def detect_and_mark_batch(t):
                nonlocal marked
                for i, s in enumerate(t.size()):
                    if s == batch_size:
                        torch._dynamo.maybe_mark_dynamic(t, i)
                        marked = True
                        break

            if (
                args.dynamic_batch_only
                and batch_size > 1
                and model_name not in CI_SKIP_DYNAMIC_BATCH_ONLY
            ):
                tree_map_only(torch.Tensor, detect_and_mark_batch, example_inputs)
                assert marked, f"nothing in example_inputs had a dim with {batch_size}"

            if args.log_operator_inputs:
                log_operator_inputs(
                    model, example_inputs, runner.model_iter_fn, name, args
                )
                continue

            if args.per_process_memory_fraction != 1:
                torch.cuda.set_per_process_memory_fraction(
                    args.per_process_memory_fraction
                )
            if model_name in DO_NOT_CAST_INPUTS:
                model, _ = runner.cast_based_on_args(model, example_inputs)

            else:
                model, example_inputs = runner.cast_based_on_args(model, example_inputs)
            runner.setup_amp(current_device)
            guard_ctx = contextlib.nullcontext()
            if name in runner.guard_on_nn_module_models:
                guard_ctx = torch._dynamo.config.patch(guard_nn_modules=True)

            inline_ctx = contextlib.nullcontext()
            if name in runner.inline_inbuilt_nn_modules_models:
                inline_ctx = torch._dynamo.config.patch(inline_inbuilt_nn_modules=True)

            with guard_ctx:
                with inline_ctx:
                    runner.run_one_model(
                        name,
                        model,
                        example_inputs,
                        optimize_ctx,
                        experiment,
                        explain=args.explain,
                        tag=args.tag,
                        batch_size=batch_size if args.dynamic_batch_only else None,
                    )
        if args.generate_aot_autograd_stats:
            stats_file = output_filename.split(".csv")[0] + "_stats.csv"
            write_outputs(
                stats_file,
                ("dev", "name", "batch_size", "total_aot_graphs", "ok_aot_graphs"),
                [
                    current_device,
                    current_name,
                    current_batch_size,
                    *Stats.aot_summary(),
                ],
            )
    else:
        metrics.purge_old_log_files()
        if (
            output_filename
            and os.path.exists(output_filename)
            and not args.retain_output
        ):
            os.unlink(output_filename)
        if original_dir:
            os.chdir(original_dir)
        model_names = list(runner.iter_model_names(args))
        nmodels = len(model_names)
        for i, name in enumerate(model_names):
            current_name = name
            if args.progress:
                print(f"Running model {i + 1}/{nmodels}", flush=True)

            try:
                timeout = args.timeout
                if should_diff_branch(args):
                    timeout *= 2
                env = os.environ.copy()
                if args.ci and name in CI_PRESERVE_COMPILE_DEBUG:
                    env["TORCH_COMPILE_DEBUG"] = "1"
                subprocess.check_call(
                    [sys.executable] + sys.argv + [f"--only={name}"],
                    timeout=timeout,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                write_csv_when_exception(args, name, "timeout")
                # NB: device is potentially multiple here, though we should
                # try our best to report in anyway TODO
                output_signpost(
                    {"name": name}, args, runner.suite_name, error="timeout"
                )
            except subprocess.CalledProcessError as e:
                print("Run failed with return code: ", e.returncode, file=sys.stderr)
                print("Output: ", e.output, file=sys.stderr)
                print("Error: ", e.stderr, file=sys.stderr)
        print_summary(output_filename, print_dataframe=args.print_dataframe_summary)


def log_operator_inputs(model, example_inputs, model_iter_fn, name, args):
    mode = "training" if args.training else "eval"
    output = os.path.join(os.path.dirname(args.output), f"{name}_{mode}.txt")

    # TODO - add option for coalescing inputs over multiple runs
    if os.path.exists(output):
        print(f"Skipping {name}, {output} already exists")
        return

    print(f"Running {name}")
    try:
        from .microbenchmarks.operator_inp_utils import OperatorInputsMode
    except ImportError:
        from microbenchmarks.operator_inp_utils import OperatorInputsMode

    operator_mode = OperatorInputsMode()
    fake_tensor_mode = FakeTensorMode()

    with torch._subclasses.fake_tensor.FakeCopyMode(fake_tensor_mode):
        model_fake = copy.deepcopy(model)
        example_inputs_fake = copy.deepcopy(example_inputs)
    try:
        with fake_tensor_mode, operator_mode:
            model_iter_fn(model_fake, example_inputs_fake, collect_outputs=False)
    except Exception as e:
        print(f"{name} failed to run with fake tensors, trying real. Exception: {e}")
        operator_mode = OperatorInputsMode()
        try:
            with operator_mode:
                model_iter_fn(model, example_inputs, collect_outputs=False)
        except Exception as e2:
            print(f"{name} failed to run with real. Exception: {e2}")
            raise

    print(f"Writing output to {output}")
    operator_mode.log_to_file(output)


if __name__ == "__main__":
    raise RuntimeError(
        f"You shouldn't run {sys.argv[0]} directly, instead try timm_model.py, torchbench.py or huggingface.py"
    )
