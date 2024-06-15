#!/usr/bin/env python3
from __future__ import annotations

import abc

import argparse
import collections
import contextlib
import copy
import csv
import dataclasses
import functools
import importlib
import itertools
import logging
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time
import weakref
from contextlib import contextmanager

from typing import (
    Any,
    Callable,
    Generator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
)
from typing_extensions import Self
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import psutil
from scipy.stats import gmean, ttest_ind
from tqdm.auto import tqdm, trange

import torch
import torch._dynamo
import torch._dynamo.utils
import torch._export
import torch.distributed
import torch.multiprocessing as mp
from torch._C import _has_cuda as HAS_CUDA, _has_xpu as HAS_XPU
from torch._dynamo.profiler import fx_insert_profiling, Profiler
from torch._dynamo.testing import (
    dummy_fx_compile,
    format_speedup,
    reset_rng_state,
    same,
)

try:
    from torch._dynamo.utils import (
        clone_inputs,
        graph_break_reasons,
        maybe_enable_compiled_autograd,
    )
    from torch._inductor.utils import fresh_inductor_cache
except ImportError:
    from _dynamo.utils import (
        clone_inputs,
        graph_break_reasons,
        maybe_enable_compiled_autograd,
    )

import torch._functorch.config
from torch._functorch.aot_autograd import set_model_name
from torch._inductor import config as inductor_config, metrics
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map, tree_map_only

try:
    import torch_xla
    import torch_xla.core.xla_model as xm

    # This is to woraround the backward issue https://github.com/pytorch/xla/issues/4174
    torch_xla._XLAC._init_computation_client()
except ImportError:
    # ignore the error if torch_xla is not installed
    pass


if TYPE_CHECKING:
    from torch.onnx._internal.fx import diagnostics


log = logging.getLogger(__name__)

# We are primarily interested in TF32
torch.backends.cuda.matmul.allow_tf32 = True

# Suppress torch.profiler spam
os.environ["KINETO_LOG_LEVEL"] = "5"

current_name = ""
current_device = ""
current_onnx_compiler = ""
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
    # TIMM
    "convmixer_768_32",  # accuracy
    "hrnet_w18",  # Stack issue in fx
    # HF
    "pnasnet5large",  # Stack issue in fx
    "MobileBertForMaskedLM",  # Stack issue in fx
    "MobileBertForQuestionAnswering",  # Stack issue in fx
    "PegasusForConditionalGeneration",  # OOM
}

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
    "hf_T5_generate",
}

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
    "timm_efficientdet",
    "timm_nfnet",
    "timm_regnet",
    "timm_vision_transformer",
    "timm_vovnet",
    "vgg16",
    "hf_T5",  # Fails dynamic https://github.com/pytorch/pytorch/issues/115968
    # HF
    "AlbertForMaskedLM",
    "BartForCausalLM",
    "BartForConditionalGeneration",
    "BlenderbotSmallForCausalLM",
    "BlenderbotSmallForConditionalGeneration",
    "DebertaV2ForQuestionAnswering",  # eager OOM
    "ElectraForCausalLM",
    "M2M100ForConditionalGeneration",
    "MBartForCausalLM",
    "MBartForConditionalGeneration",
    "OPTForCausalLM",
    "PLBartForCausalLM",
    "PLBartForConditionalGeneration",
    "PegasusForCausalLM",
    "Speech2Text2ForCausalLM",
    "TrOCRForCausalLM",
    "XGLMForCausalLM",
    # TIMM
    "adv_inception_v3",
    "botnet26t_256",
    "cait_m36_384",  # OOM
    "coat_lite_mini",
    "convit_base",
    "dpn107",
    "fbnetv3_b",
    "gernet_l",
    "lcnet_050",
    "mixnet_l",
    "res2net101_26w_4s",
    "res2net50_14w_8s",
    "res2next50",
    "resnest101e",
    "sebotnet33ts_256",
    "swsl_resnext101_32x16d",
    "tf_efficientnet_b0",
    "ghostnet_100",
    "gmixer_24_224",
    "tinynet_a",
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
    "hf_T5_base",
    "hf_clip",
    "llama_v2_7b_16h",
    "mobilenet_v2_quantized_qat",
    "phi_1_5 resnet50_quantized_qat",
    "BlenderbotForCausalLM",
    "cait_m36_384",
    "DALLE2_pytorch",
    "moco",
    "timm_efficientdet",
    "ghostnet_100",
    "regnety_002",
    "poolformer_m36",
    "inception_v3",
    "tinynet_a",
    "selecsls42b",
    "mobilevit_s",
    "pytorch_CycleGAN_and_pix2pix",
    "vision_maskrcnn",
    "resmlp_12_224",
    "dlrm",
    "resnet50",
    "dm_nfnet_f0",
    "pit_b_224",
    "tf_mixnet_l",
}


DO_NOT_CAST_INPUTS = {"stable_diffusion"}


# Maps a benchmark model name to a list of status codes. For any listed entry, we'll
# capture TORCH_COMPILE_DEBUG logs in CI runs and preseve them (i.e., for upload) if
# the result status matches one listed.
CI_PRESERVE_COMPILE_DEBUG = {
    # For example:
    # "mnasnet1_0": ["fail_accuracy"],
}


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


def output_csv(filename, headers, row):
    global disable_output
    if disable_output:
        return
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


def nothing(f):
    return f


@functools.lru_cache(None)
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

    if device not in ["cuda", "xpu"]:
        log.warning(
            "Trying to call the empty_gpu_cache for device: %s, which is not in list [cuda, xpu]",
            device,
        )
        return

    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "xpu":
        torch.xpu.empty_cache()


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
                print(col.ljust(width), f"pass_rate={100*pass_rate:.2f}%")
            else:
                cdata = data[col]
                print(
                    col.ljust(width),
                    f"gmean={gmean(cdata):.2f}x mean={cdata.mean():.3f}x",
                )
        except Exception as e:
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
):
    use_xla = tensor_is_on_xla(example_inputs)
    synchronize()

    if use_xla:
        xm.mark_step()
        xm.wait_device_ops()

    time_total = 0
    # Dont collect outputs to correctly measure timing
    for _ in range(times):
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


def _normalize_bench_inputs(example_inputs) -> Tuple[Tuple[Any], Mapping[str, Any]]:
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
    output_csv(
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


def recompile_profiler_experiment(args, model_iter_fn, model, example_inputs):
    prof = torch._dynamo.utils.CompilerProfiler()
    opt_model_iter_fn = torch._dynamo.optimize(prof, nopython=args.nopython)(
        model_iter_fn
    )
    opt_model_iter_fn(model, example_inputs)
    output_csv(
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


def speedup_experiment(args, model_iter_fn, model, example_inputs, **kwargs):
    """
    Measure speedups over eager.

    Writes to ./speedups.csv
    """
    # if args.dynamic_shapes:
    #     return speedup_experiment_ds(args, model_iter_fn, model, example_inputs)

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

    # Use higher tolerance for XLA since XLA cause numerical unstability when
    # graph size changes
    tolerance = args.xla_tolerance if args.trace_on_xla else 1e-4
    torch._dynamo.config.repro_tolerance = tolerance

    with maybe_profile(args.export_profiler_trace) as p:
        if args.export_aot_inductor:
            frozen_model_iter_fn = export_aot_inductor(
                model, example_inputs, args.devices[0]
            )
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
            with maybe_mark_profile(p=p, mark="expected"):
                timings[rep, 0], expected_output = timed(
                    model,
                    model_iter_fn,
                    inputs,
                    return_result=True,
                    times=times,
                    collect_outputs=args.collect_outputs,
                )

            # call mark_step between the 2 calls to make the comparison fair.
            maybe_mark_step(args)

            with maybe_mark_profile(p=p, mark="actual"), maybe_enable_compiled_autograd(
                args.compiled_autograd
            ):
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
    output_csv(
        output_filename,
        headers,
        row,
    )
    headers, data = torch._dynamo.utils.compile_times(repr="csv", aggregate=True)
    assert (
        output_filename.find(".csv") > 0
    ), f"expected output_filename to be a .csv, but got {output_filename}"
    output_csv(
        output_filename[:-4] + "_compilation_metrics.csv",
        first_headers + headers,
        first_fields + data,
    )
    return msg


def speedup_experiment_ds(args, model_iter_fn, model, example_inputs):
    """
    Run dynamic shapes benchmarks.

    Requires dynamic shape compatible models, which provide a list of example inputs.

    Warms up using the first input example and then iterates the inputs,
    measuring (and expecting minimal) variance between the runtime for different examples.

    """
    timings = np.zeros((args.repeat, len(example_inputs), 2), np.float64)

    if args.repeat > 5:
        print(
            f"\ndynamic shapes experiments are slow, consider setting --repeat less than {args.repeat}\n"
        )

    nwarmup = 4
    for rep in range(args.repeat):
        # Start each rep fresh, e.g. only warmup on example 0
        torch._dynamo.reset()
        optimized_model_iter_fn = optimize_ctx(model_iter_fn)
        for _ in range(nwarmup):
            optimized_model_iter_fn(model, example_inputs[0])

        for input_idx, inputs in enumerate(example_inputs):
            # interleave the runs to handle frequency scaling and load changes
            timings[rep, input_idx, 0] = timed(
                model, model_iter_fn, inputs, return_result=False
            )
            # different from regular speedup_experiment, we _DO_ want to allow recompilation
            timings[rep, input_idx, 1] = timed(
                model, optimized_model_iter_fn, inputs, return_result=False
            )
    medians = np.median(timings, axis=0)
    speedups = list(medians[:, 0] / medians[:, 1])
    speedups_mean = np.mean(speedups)
    speedups_median = np.median(speedups)
    speedups_var = np.var(speedups)

    # TODO this x[0] is not going to work in general but bert only has 1 input
    shapes = [x[0].shape for x in example_inputs]
    shape_keys = sorted(set(shapes))
    shape_speedups = {
        shape: [
            it[1] for it in filter(lambda it: it[0] == shape, zip(shapes, speedups))
        ]
        for shape in shape_keys
    }
    output_str = (
        f"mean: {speedups_mean:.3f}, median: {speedups_median:.3f}, var: {speedups_var:.3f}"
        + "\nSpeedups by shape: "
        + "\n".join(
            [
                f"{shape}: "
                + ", ".join([f"{speedup: .3g}" for speedup in shape_speedups[shape]])
                for shape in shape_keys
            ]
        )
    )
    output_csv(
        output_filename,
        ("dev", "name", "batch_size", "speedup mean", "speedup median", "speedup var"),
        [
            current_device,
            current_name,
            current_batch_size,
            speedups_mean,
            speedups_median,
            speedups_var,
        ],
    )
    return output_str


@contextlib.contextmanager
def override_synchronize_with_onnx_iobinding(iobinding):
    global synchronize
    prev_synchrnoize = synchronize
    try:
        if iobinding is not None:

            def new_synchronize():
                iobinding.synchronize_inputs()
                iobinding.synchronize_outputs()

            synchronize = new_synchronize
        yield
    finally:
        synchronize = prev_synchrnoize


def speedup_experiment_onnx(
    args,
    model_iter_fn,
    onnx_model: OnnxModel,
    model,
    example_inputs,
    **kwargs,
):
    """
    Measure speedups over eager.

    This function is responsible for the following:
        1. Creating iobinding with OnnxModel if device is CUDA, which is essential for perf measurement.
        2. Running ORT with OnnxModel.

    Writes to ./{output_filename}, which should be
        `pathlib.Path(self.output_dir) / f"{self.compiler}_{suite}_{self.dtype}_{self.mode}_{self.device}_{self.testing}.csv".

    TODO(bowbao): Record export time and export peak memory usage.
    """
    timings = np.zeros((args.repeat, 2), np.float64)
    is_correct = True
    should_randomize_input = args.randomize_input
    times = args.iterations_per_run

    def create_onnx_input_binded_fn(onnx_model: OnnxModel, pt_inputs, example_outputs):
        # Goal is to move the iobinding creation outside of the timer function.
        iobinding, outputs = onnx_model.create_iobinding(pt_inputs, example_outputs)

        def onnxrt_model_iter_fn(model, inputs, collect_outputs=True):
            onnx_model.run_with_iobinding(iobinding, outputs)
            if collect_outputs:
                return outputs

        return onnxrt_model_iter_fn, iobinding

    def create_onnx_fn(onnx_model: OnnxModel, pt_inputs):
        # NOTE: Making perf comparison fair by moving out the i/o adapting part.
        # 1. Pre-adapt `pt_inputs` to `onnx_inputs` here.
        # 2. Drop `onnx_outputs` to `pt_outputs` adapting. Output comparison is not part of perf measurement.
        onnx_inputs = onnx_model.adapt_pt_inputs_to_onnx(pt_inputs)

        def onnxrt_model_iter_fn(model, inputs, collect_outputs=True):
            return onnx_model.run_with_onnx_inputs(onnx_inputs)

        return onnxrt_model_iter_fn

    def timed_onnx(model, onnx_model: OnnxModel, inputs):
        if current_device == "cpu" or onnx_model.is_cpu():
            onnxrt_model_iter_fn = create_onnx_fn(onnx_model, inputs)
            iobinding = None
        else:
            onnxrt_model_iter_fn, iobinding = create_onnx_input_binded_fn(
                onnx_model, inputs, expected_output
            )
        with override_synchronize_with_onnx_iobinding(iobinding):
            return timed(
                model,
                onnxrt_model_iter_fn,
                inputs,
                return_result=True,
                times=times,
                collect_outputs=args.collect_outputs,
            )

    # Insert ONNX warm-up
    inputs = (
        randomize_input(copy.deepcopy(example_inputs))
        if should_randomize_input
        else example_inputs
    )
    _, expected_output = timed(
        model,
        model_iter_fn,
        inputs,
        return_result=True,
        times=times,
        collect_outputs=args.collect_outputs,
    )
    for _ in range(2):
        timed_onnx(model, onnx_model, inputs)

    for rep in range(args.repeat):
        inputs = (
            randomize_input(copy.deepcopy(example_inputs))
            if should_randomize_input
            else example_inputs
        )
        if torch.cuda.device_count() > 1:
            # Manually set correct torch.cuda.current_device to ensure torch.cuda.synchronize() works as intended.
            # When there are more than 1 cuda devices, the first one is used for pytorch eager.
            # The second one is used for onnx ort.
            torch.cuda.set_device(0)
        timings[rep, 0], expected_output = timed(
            model,
            model_iter_fn,
            inputs,
            return_result=True,
            times=times,
            collect_outputs=args.collect_outputs,
        )
        if torch.cuda.device_count() > 1:
            # Manually set correct torch.cuda.current_device to ensure torch.cuda.synchronize() works as intended.
            # When there are more than 1 cuda devices, the first one is used for pytorch eager.
            # The second one is used for onnx ort.
            torch.cuda.set_device(1)
        timings[rep, 1], actual_output = timed_onnx(model, onnx_model, inputs)

    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    median = np.median(timings, axis=0)
    speedup = median[0] / median[1]
    if args.dump_raw_metrics:
        np.save(
            f"{output_filename[:-4]}-raw_timings-{current_name}-{current_device}.npy",
            timings,
        )

    headers = ["dev", "name", "batch_size", "speedup", "abs_latency"]
    row = [
        current_device,
        current_name,
        current_batch_size,
        float(speedup),
        median[1] * 1000,
    ]
    if "compilation_latency" in kwargs:
        headers = headers + ["compilation_latency", "compression_ratio"]
        row.append(kwargs["compilation_latency"])
        row.append(kwargs["compression_ratio"])

    output_csv(
        output_filename,
        headers,
        row,
    )
    headers, data = torch._dynamo.utils.compile_times(repr="csv", aggregate=True)
    assert (
        output_filename.find(".csv") > 0
    ), f"expected output_filename to be a .csv, but got {output_filename}"
    output_csv(
        output_filename[:-4] + "_compilation_metrics.csv",
        ["dev", "name", "batch_size"] + headers,
        [current_device, current_name, current_batch_size] + data,
    )
    return format_speedup(speedup, pvalue, is_correct=is_correct)


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
    output_csv(
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
    output_csv(
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


class AOTInductorModelCache:
    cache = dict()

    @classmethod
    def load(cls, model, example_inputs, device):
        import torch._inductor
        import torch.export._trace

        key = weakref.ref(model)
        if key not in cls.cache:
            # Register the output dataclass to pytree
            example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
            with torch.no_grad():
                # copy.deepcopy is required to prevent any surprising side-effect,
                # see https://github.com/pytorch/pytorch/issues/113029
                example_outputs = copy.deepcopy(model)(*example_args, **example_kwargs)

            if pytree._is_namedtuple_instance(example_outputs):
                typ = type(example_outputs)
                pytree._register_namedtuple(
                    typ,
                    serialized_type_name=f"{typ.__module__}.{typ.__name__}",
                )
            else:
                _register_dataclass_output_as_pytree(example_outputs)

            # TODO(angelayi): change this to predispatch
            # https://github.com/pytorch/pytorch/issues/127513 needs to be fixed before changing
            # to predispatch to avoid performance regressions
            gm = torch.export._trace._export_to_torch_ir(
                model,
                example_args,
                example_kwargs,
            )
            with torch.no_grad():
                so_path = torch._inductor.aot_compile(
                    gm, example_args, example_kwargs
                )  # type: ignore[arg-type]

            cls.cache[key] = torch._export.aot_load(so_path, device)

        return cls.cache[key]


def export(model, example_inputs):
    example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
    example_outputs = model(*example_args, **example_kwargs)
    _register_dataclass_output_as_pytree(example_outputs)

    ep = torch.export.export(model, example_args, example_kwargs)

    def opt_export(_, example_inputs):
        example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
        return ep(*example_args, **example_kwargs)

    return opt_export


def export_aot_inductor(model, example_inputs, device):
    optimized = AOTInductorModelCache.load(model, example_inputs, device)

    def opt_aot_inductor(_, example_inputs, collect_outputs=False):
        example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
        return optimized(*example_args, **example_kwargs)

    return opt_aot_inductor


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


class OnnxModel(abc.ABC):
    TORCH_TO_NUMPY_DTYPE = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.uint8: np.uint8,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.longlong,
        torch.bool: np.bool_,
    }

    _COMPILER_NAME: str

    def __init__(
        self,
        output_directory,
        model,
        example_inputs,
        dynamic_shapes: bool,
        copy_before_export: bool = False,
    ):
        model_name = current_name
        self.copy_before_export = copy_before_export
        self.model_dir = self._generate_onnx_model_directory(
            output_directory, self._COMPILER_NAME, model_name
        )
        self.model_path = str(
            self.model_dir / f"{model_name}_{self._COMPILER_NAME}.onnx"
        )

    def _determine_deepcopy_target_device(self):
        if current_device == "cpu":
            target_device = "cpu"
        else:
            if torch.cuda.device_count() > 1:
                # Copy to another cuda device to avoid OOM.
                target_device = "cuda:1"
            else:
                target_device = "cuda"
        return target_device

    def deepcopy_model_and_inputs_to_device(self, model, example_inputs, target_device):
        # Deepcopy model before export to avoid modification to baseline model.
        # To avoid OOM, the model is first moved to CPU. Both models are then moved to device.
        model_device = next(model.parameters()).device
        model.to("cpu")
        model_copy = copy.deepcopy(model).to(target_device)
        model.to(model_device)

        target_device_example_inputs = tree_map_only(
            torch.Tensor, lambda x: x.to(device=target_device), example_inputs
        )

        return model_copy, target_device_example_inputs

    @classmethod
    def _generate_onnx_model_directory(
        cls, output_directory: str, compiler_name: str, model_name: str
    ) -> pathlib.Path:
        model_path = pathlib.Path(
            output_directory,
            ".onnx_models",
            model_name,
            compiler_name,
        )
        if model_path.exists() and model_path.is_dir():
            shutil.rmtree(model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        return model_path

    @abc.abstractmethod
    def format_pt_inputs(self, pt_inputs: Any) -> Sequence[torch.Tensor]:
        ...

    @abc.abstractmethod
    def format_pt_outputs(self, pt_outputs: Any) -> Sequence[torch.Tensor]:
        ...

    def adapt_pt_inputs_to_onnx(self, pt_inputs) -> Mapping[str, np.ndarray]:
        pt_inputs = self.format_pt_inputs(pt_inputs)
        return {
            ort_input.name: pt_input.cpu().numpy()
            for ort_input, pt_input in zip(self.onnx_session.get_inputs(), pt_inputs)
        }

    def adapt_onnx_outputs_to_pt(self, onnx_outputs: List[np.ndarray]) -> Any:
        pt_outputs = [
            torch.from_numpy(onnx_output).to(current_device)
            for onnx_output in onnx_outputs
        ]
        if len(pt_outputs) == 1:
            return pt_outputs[0]
        return pt_outputs

    def _init_ort_session(self, model_path: str):
        import onnxruntime

        if current_device == "cpu":
            ort_providers = ["CPUExecutionProvider"]
        else:
            # NOTE(bowbao): Reduce OOM by running ORT on another gpu.
            # TODO(bowbao): This works to avoid OOM, but performance is surprisingly very bad.
            cuda_provider_options = {
                "device_id": 1 if torch.cuda.device_count() > 1 else 0,
            }
            ort_providers = [("CUDAExecutionProvider", cuda_provider_options)]
        session_options = onnxruntime.SessionOptions()
        session_options.log_severity_level = 3  # Error

        ort_session = onnxruntime.InferenceSession(
            self.model_path,
            providers=ort_providers,
            sess_options=session_options,
        )
        return ort_session

    def is_cpu(self) -> bool:
        return self.onnx_session.get_providers()[0] == "CPUExecutionProvider"

    def cpu(self) -> Self:
        self.onnx_session.set_providers(["CPUExecutionProvider"])
        return self

    def create_outputs(self, *example_outputs):
        return tuple(torch.empty_like(x) for x in example_outputs)

    def create_iobinding(self, pt_inputs, example_outputs):
        pt_inputs = self.format_pt_inputs(pt_inputs)
        example_outputs = self.format_pt_outputs(example_outputs)

        iobinding = self.onnx_session.io_binding()
        args = [arg.contiguous() for arg in pt_inputs]
        for ort_input, arg in zip(self.onnx_session.get_inputs(), args):
            # NOTE: Run ORT on another cuda device to reduce OOM.
            if torch.cuda.device_count() > 1:
                arg = arg.detach().to("cuda:1")
            device = arg.device
            iobinding.bind_input(
                ort_input.name,
                device.type,
                device.index or 0,
                self.TORCH_TO_NUMPY_DTYPE[arg.dtype],
                arg.size(),
                arg.data_ptr(),
            )

        outputs = self.create_outputs(*example_outputs)
        for ort_output, output in zip(self.onnx_session.get_outputs(), outputs):
            if torch.cuda.device_count() > 1:
                output = output.detach().to("cuda:1")
            device = output.device
            iobinding.bind_output(
                ort_output.name,
                device.type,
                device.index or 0,
                self.TORCH_TO_NUMPY_DTYPE[output.dtype],
                output.size(),
                output.data_ptr(),
            )
        return iobinding, outputs

    def run_with_iobinding(self, iobinding, outputs):
        # 'outputs' are torch empty tensors binded to 'iobinding'.
        self.onnx_session.run_with_iobinding(iobinding)
        return outputs

    def run_with_onnx_inputs(self, onnx_inputs):
        return self.onnx_session.run(None, onnx_inputs)

    @classmethod
    def save_tensor_data(cls, numpy_tensor, output_path):
        from onnx import numpy_helper

        proto_tensor = numpy_helper.from_array(numpy_tensor)
        with open(output_path, "wb") as f:
            f.write(proto_tensor.SerializeToString())

    def run_and_serialize_inputs_outputs(self, pt_inputs):
        test_data_dir = self.model_dir / "test_data_set_0"
        test_data_dir.mkdir(parents=True, exist_ok=True)

        onnx_inputs = self.adapt_pt_inputs_to_onnx(pt_inputs)
        for i, onnx_input in enumerate(onnx_inputs.values()):
            self.save_tensor_data(onnx_input, str(test_data_dir / f"input_{i}.pb"))

        onnx_outputs = self.run_with_onnx_inputs(onnx_inputs)

        for i, onnx_output in enumerate(onnx_outputs):
            self.save_tensor_data(onnx_output, str(test_data_dir / f"output_{i}.pb"))

        return self.adapt_onnx_outputs_to_pt(onnx_outputs)

    def run(self, pt_inputs):
        # NOTE: For CUDA performance testing, use `run_with_iobinding` to exclude memory
        # copying overhead for inputs/outputs between cpu and gpu.
        # Otherwise perf number is inaccurate.
        onnx_inputs = self.adapt_pt_inputs_to_onnx(pt_inputs)
        onnx_outputs = self.run_with_onnx_inputs(onnx_inputs)
        return self.adapt_onnx_outputs_to_pt(onnx_outputs)


class OnnxModelFromTorchScript(OnnxModel):
    """TorchScript based onnx export. `torch.onnx.export`

    TODO(bowbao):
    * large model export failed.
          Onnx Model is larger than 2GB, but exporter makes decision based pt model size, which is
          smaller than 2GB.
    * OOM on slightly larger model.
          Both pt model and ort inference session are on gpu. Attempt has been made to move ORT to
          cuda:1, however ORT perf drop significantly.
          For now running everything with batch_size 1 set in launch script.
    """

    _COMPILER_NAME = "torchscript"

    def __init__(
        self, output_directory, model, example_inputs, dynamic_shapes: bool, **kwargs
    ):
        if dynamic_shapes:
            raise NotImplementedError("NYI dynamic shapes for OnnxModelFromTorchScript")
        super().__init__(
            output_directory, model, example_inputs, dynamic_shapes, **kwargs
        )
        self._export(
            model,
            example_inputs,
            self.model_path,
            opset_version=17,
            do_constant_folding=False,
            verbose=False,
        )
        self.onnx_session = self._init_ort_session(self.model_path)

    def _export(self, model, example_inputs, output_path: str, /, **kwargs) -> None:
        if self.copy_before_export:
            # Deepcopy model before export to avoid modification to baseline model.
            model, example_inputs = self.deepcopy_model_and_inputs_to_device(
                model, example_inputs, self._determine_deepcopy_target_device()
            )

        # Hack for huggingface models (kwargs only).
        if isinstance(example_inputs, dict):

            class WrapperModel(torch.nn.Module):
                def __init__(self, model, keys):
                    super().__init__()
                    self.model = model
                    self.keys = keys

                def forward(self, *args):
                    return self.model(**dict(zip(self.keys, args)))

            model = WrapperModel(model, list(example_inputs.keys()))

        torch.onnx.export(
            model,
            self.format_pt_inputs(example_inputs),
            output_path,
            **kwargs,
        )

    def format_pt_inputs(self, pt_inputs):
        # NOTE(bowbao): For huggingface benchmark, pt_inputs are formatted as dictionary,
        # and consumed like `model(**pt_inputs)`.
        # For other benchmarks, pt_inputs are formatted as tuple and consumed
        # like `model(*pt_inputs)`.
        if isinstance(pt_inputs, dict):
            pt_inputs = list(pt_inputs.values())
        if isinstance(pt_inputs, torch.Tensor):
            pt_inputs = (pt_inputs,)
        return tuple(arg.contiguous() for arg in pt_inputs)

    def format_pt_outputs(self, pt_outputs):
        if isinstance(pt_outputs, torch.Tensor):
            pt_outputs = (pt_outputs,)

        pt_outputs = pytree.tree_leaves(pt_outputs)

        # Hack for huggingface model outputs
        try:
            from transformers import modeling_outputs
        except ImportError:
            pass
        else:

            def _to_tuple(x):
                if isinstance(x, modeling_outputs.ModelOutput):
                    return x.to_tuple()
                return x

            pt_outputs = pytree.tree_map(_to_tuple, pt_outputs)
            pt_outputs = pytree.tree_leaves(pt_outputs)

        return pt_outputs


class OnnxModelFromDynamo(OnnxModel):
    """Dynamo and Fx based export. `torch.onnx.dynamo_export`."""

    _COMPILER_NAME = "dynamo"

    def __init__(
        self, output_directory, model, example_inputs, dynamic_shapes: bool, **kwargs
    ):
        super().__init__(
            output_directory, model, example_inputs, dynamic_shapes, **kwargs
        )
        self._dynamic_shapes = dynamic_shapes
        self._onnx_program = self._export(model, example_inputs, self.model_path)
        # Clear the model proto to save memory.
        # The model proto is saved to disk and no longer needed from `onnx_program`.
        # `onnx_program` is kept for i/o adapter usage.
        self._onnx_program.model_proto.Clear()
        self.onnx_session = self._init_ort_session(self.model_path)

    def _export(
        self, model, example_inputs, output_path: str
    ) -> torch.onnx.ONNXProgram:
        if self.copy_before_export:
            # Deepcopy model before export to avoid modification to baseline model.
            model, example_inputs = self.deepcopy_model_and_inputs_to_device(
                model, example_inputs, self._determine_deepcopy_target_device()
            )

        example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
        options = torch.onnx.ExportOptions(dynamic_shapes=self._dynamic_shapes)
        onnx_program = torch.onnx.dynamo_export(
            model, *example_args, **example_kwargs, export_options=options
        )

        onnx_program.save(output_path)
        return onnx_program

    def format_pt_inputs(self, pt_inputs):
        pt_args, pt_kwargs = _normalize_bench_inputs(pt_inputs)
        return self._onnx_program.adapt_torch_inputs_to_onnx(*pt_args, **pt_kwargs)

    def format_pt_outputs(self, pt_outputs):
        return self._onnx_program.adapt_torch_outputs_to_onnx(pt_outputs)


class OnnxModelFromDynamoAotInline(OnnxModelFromDynamo):
    """Dynamo and Fx based export, with AOT inline post export. `torch.onnx.dynamo_export`."""

    _COMPILER_NAME = "dynamo_aot_inline"

    def _export(
        self, model, example_inputs, output_path: str
    ) -> torch.onnx.ONNXProgram:
        if self.copy_before_export:
            # Deepcopy model before export to avoid modification to baseline model.
            model, example_inputs = self.deepcopy_model_and_inputs_to_device(
                model, example_inputs, self._determine_deepcopy_target_device()
            )

        example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
        options = torch.onnx.ExportOptions(dynamic_shapes=self._dynamic_shapes)
        onnx_program = torch.onnx.dynamo_export(
            model, *example_args, **example_kwargs, export_options=options
        )
        # Apply AOT inline post export.
        # Requires onnx >= 1.15
        import onnx
        import onnx.inliner

        # Workaround for inliner not supporting with models larger than 2GB.
        # Save model to disk first separating out external data,
        # and load back without external data for inliner to work on.
        model_proto = onnx_program.model_proto
        onnx.save_model(model_proto, output_path, save_as_external_data=True)
        model_proto = onnx.load(output_path, load_external_data=False)
        model_proto = onnx.inliner.inline_local_functions(model_proto)
        onnx.save_model(model_proto, output_path)
        return onnx_program


class OnnxModelFromDynamoAotOptimize(OnnxModelFromDynamo):
    """Dynamo and Fx based export, with AOT optimize post export. `torch.onnx.dynamo_export`."""

    _COMPILER_NAME = "dynamo_aot_optimize"

    def _export(
        self, model, example_inputs, output_path: str
    ) -> torch.onnx.ONNXProgram:
        if self.copy_before_export:
            # Deepcopy model before export to avoid modification to baseline model.
            model, example_inputs = self.deepcopy_model_and_inputs_to_device(
                model, example_inputs, self._determine_deepcopy_target_device()
            )

        example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
        options = torch.onnx.ExportOptions(dynamic_shapes=self._dynamic_shapes)
        export_output = torch.onnx.dynamo_export(
            model, *example_args, **example_kwargs, export_options=options
        )

        import onnx
        from onnxscript.rewriter.onnxruntime import rewrite

        model_proto = rewrite(export_output.model_proto)
        onnx.save_model(
            model_proto,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
        )

        return export_output


class _OnnxPatch:
    @classmethod
    def patch_non_tensor_outputs(cls, correct_result, new_result, fp64_outputs):
        """Patch non-tensor outputs to make them comparable with the correct result.

        ONNX model always returns a flat tuple of tensors, but the PyTorch model outputs
        `correct_result` and `fp64_outputs` can be arbitrary types. This function normalizes
        the outputs to make them comparable with the ONNX model output.
        """
        try:
            from transformers import modeling_outputs
        except ImportError:
            has_transformers = False
        else:
            has_transformers = True

        if has_transformers and isinstance(
            correct_result, modeling_outputs.ModelOutput
        ):
            correct_result = correct_result.to_tuple()
            fp64_outputs = fp64_outputs.to_tuple() if fp64_outputs is not None else None
        elif type(correct_result).__name__ in (
            "MaskedLMOutput",
            "Seq2SeqLMOutput",
            "CausalLMOutputWithCrossAttentions",
            "LongformerMaskedLMOutput",
            "Instances",
            "SquashedNormal",
            "Boxes",
            "Normal",
            "TanhTransform",
            "Foo",
            "Variable",
        ):
            # Copied from `same` function in `torch._dynamo.utils`
            correct_result = [
                value
                for key in correct_result.__dict__.keys()
                if (value := getattr(correct_result, key)) is not None
            ]
            fp64_outputs = (
                [
                    value
                    for key in fp64_outputs.__dict__.keys()
                    if (value := getattr(fp64_outputs, key)) is not None
                ]
                if fp64_outputs is not None
                else None
            )

        # Flatten nested tuple of tensors, i.e. past_key_values
        correct_result = pytree.tree_leaves(correct_result)
        # Hack to put results from different runs on same device.
        # This is needed for ONNX CPU fallback benchmark, where PyTorch eager is run on GPU.
        # Assuming outputs from a single run are always on same device!
        devices = [x.device for x in correct_result if isinstance(x, torch.Tensor)]
        assert devices and all(
            x == devices[0] for x in devices
        ), "All tensors must be on same device!"
        device = devices[0]
        new_result = pytree.tree_leaves(new_result)
        new_result = pytree.tree_map(
            lambda x: x.to(device=device) if isinstance(x, torch.Tensor) else x,
            new_result,
        )
        fp64_outputs = pytree.tree_leaves(fp64_outputs)

        return correct_result, new_result, fp64_outputs


@dataclasses.dataclass
class OnnxExportErrorRow:
    device: str
    model_name: str
    batch_size: int
    rule_id: Optional[str] = None
    rule_name: Optional[str] = None
    diagnostic_level: Optional[str] = None
    diagnostic_message: Optional[str] = None
    exception_type_name: Optional[str] = None
    exception_message: Optional[str] = None

    def __post_init__(self):
        assert (
            self.rule_id is not None
            and self.rule_name is not None
            and self.diagnostic_level is not None
            and self.diagnostic_message is not None
        ) or self.exception_type_name, (
            "Either rule_id, rule_name, diagnostic_level and diagnostic_message "
            "must be set or exception_type_name must be set"
        )

    @property
    def headers(self) -> List[str]:
        return [field.name for field in dataclasses.fields(self)]

    @property
    def row(self) -> List[str]:
        return [getattr(self, field.name) for field in dataclasses.fields(self)]


class OnnxExportErrorParser:
    def __init__(self, device: str, model_name: str, batch_size: int):
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size

    def _qualified_exception_class_name(self, exception: Exception) -> str:
        if exception.__class__.__module__ == "builtins":
            return exception.__class__.__name__
        return f"{exception.__class__.__module__}.{exception.__class__.__name__}"

    def parse_diagnostic_context(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
    ) -> Generator[OnnxExportErrorRow, Any, Any]:
        from torch.onnx._internal.fx import diagnostics

        for diagnostic in diagnostic_context.diagnostics:
            if diagnostic.level >= diagnostics.levels.ERROR:
                yield OnnxExportErrorRow(
                    device=self.device,
                    model_name=self.model_name,
                    batch_size=self.batch_size,
                    rule_id=diagnostic.rule.id,
                    rule_name=diagnostic.rule.name,
                    diagnostic_level=diagnostic.level.name,
                    diagnostic_message=diagnostic.message,
                )

    def parse_exception(self, exception: Exception) -> OnnxExportErrorRow:
        return OnnxExportErrorRow(
            device=self.device,
            model_name=self.model_name,
            batch_size=self.batch_size,
            exception_type_name=self._qualified_exception_class_name(exception),
            exception_message=str(exception),
        )


@dataclasses.dataclass
class OnnxContext:
    onnx_model: Optional[OnnxModel] = None


def optimize_onnx_ctx(
    output_directory: str,
    onnx_model_cls: Type[OnnxModel],
    run_n_iterations: Callable,
    dynamic_shapes: bool = False,
    copy_before_export: bool = False,
) -> Callable:
    # NOTE(bowbao): This function creates and returns the onnx version of 'run_n_iterations',
    # which does the following:
    #   1. Export and cache model.
    #   2. Create iobinding for ORT.
    #   3. Run ORT for n iterations.
    # The cached model is stored in 'context' under the returned callable.
    context = OnnxContext()
    test_data_dumped = False

    def run_n_iterations_onnx(model, inputs, n=2):
        from torch.onnx._internal import exporter
        from torch.onnx._internal.fx import diagnostics

        # NOTE(bowbao): Capture all export & ort errors and diagnostics.
        # Serialize to csv, to be parsed and summarized later by '._onnx/reporter.py'.
        # TODO: Accuracy mismatch is not reported here in csv.
        assert (
            output_filename.find(".csv") > 0
        ), f"expected output_filename to be a .csv, but got {output_filename}"
        output_error_filename = output_filename[:-4] + "_export_error.csv"
        parser = OnnxExportErrorParser(current_device, current_name, current_batch_size)
        try:
            nonlocal context
            if context.onnx_model is None:
                context.onnx_model = onnx_model_cls(
                    output_directory,
                    model,
                    copy.deepcopy(inputs),
                    dynamic_shapes=dynamic_shapes,
                    copy_before_export=copy_before_export,
                )
            onnx_model = context.onnx_model

            for _ in range(n):
                nonlocal test_data_dumped
                if not test_data_dumped:
                    # Serializes inputs and outputs to .pb files for further offline analysis.
                    # Due to this, this function is not and should not be used for perf measurement.
                    outputs = onnx_model.run_and_serialize_inputs_outputs(inputs)
                    test_data_dumped = True
                else:
                    outputs = onnx_model.run(inputs)
            return outputs
        except exporter.OnnxExporterError as e:
            # `torch.onnx.dynamo_export` raises error that encloses diagnostics.
            diagnostic_context = e.onnx_program.diagnostic_context
            for parsed_error in parser.parse_diagnostic_context(diagnostic_context):
                output_csv(
                    output_error_filename, parsed_error.headers, parsed_error.row
                )
            if context.onnx_model is not None:
                e.onnx_program.save_diagnostics(
                    f"{context.onnx_model.model_dir}/"
                    f"{current_onnx_compiler}_{current_name}_{current_device}.sarif"
                )

            # Check also the raw exception that caused export failure.
            # Skip if it is already analyzed by diagnostics.
            cause_of_exception = e.__cause__
            if not isinstance(
                cause_of_exception, diagnostics.RuntimeErrorWithDiagnostic
            ):
                parsed_error = parser.parse_exception(cause_of_exception)
                output_csv(
                    output_error_filename, parsed_error.headers, parsed_error.row
                )
            raise
        except Exception as e:
            # `torch.onnx.export` errors.
            # ORT errors.
            parsed_error = parser.parse_exception(e)
            output_csv(output_error_filename, parsed_error.headers, parsed_error.row)
            raise

    run_n_iterations_onnx.context = context

    return run_n_iterations_onnx


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
                logging.error("Failed to save memory snapshot, %s", e)

            torch.cuda.memory._record_memory_history(enabled=None)


class BenchmarkRunner:
    def __init__(self):
        self.model_iter_fn = None
        self.grad_scaler = DummyGradScaler()
        self.autocast = contextlib.nullcontext
        self.autocast_arg = {}
        self.optimizer = None
        self._args = None

    def setup_amp(self, current_device=None):
        if self.args.only in self.fp32_only_models:
            return

        devices = [current_device] if current_device else self.args.devices
        if self.args.amp:
            # AMP training can lead to small loss values which can undeflow
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
                # After autotuning and fake tensor caching lands, we can enable, becuase the compile time impact will be lower.
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
    def skip_models_for_freezing(self):
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
    def guard_on_nn_module_models(self):
        return set()

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        raise NotImplementedError

    @property
    def equal_nan(self):
        equal_nan = True
        if self.args.float32:
            equal_nan = False
        return equal_nan

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

    def run_n_iterations(self, mod, inputs):
        n = self.args.iterations
        for _ in range(n - 1):
            self.model_iter_fn(mod, inputs, collect_outputs=False)
        return self.model_iter_fn(mod, inputs, collect_outputs=True)

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

        from transformers.models.t5.modeling_t5 import T5Block
        from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

        from torch.distributed.fsdp.wrap import (
            ModuleWrapPolicy,
            size_based_auto_wrap_policy,
        )

        # handcrafted wrap policy
        MODEL_FSDP_WRAP = {
            "stable_diffusion_unet": (Transformer2DModel,),
            "hf_T5": (T5Block,),
            "hf_T5_base": (T5Block,),
            "hf_T5_large": (T5Block,),
            "hf_Whisper": (WhisperEncoderLayer,),
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
            assert (
                torch.distributed.is_available()
            ), "Can't use DDP without a distributed enabled build"
            from torch.nn.parallel import DistributedDataParallel as DDP

            model = DDP(model, find_unused_parameters=True)
        elif self.args.fsdp:
            assert (
                torch.distributed.is_available()
            ), "Can't use FSDP without a distributed enabled build"
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

            dynamo_stats = get_dynamo_stats()
            dynamo_stats.subtract(dynamo_start_stats)
            for k, v in dynamo_stats.items():
                headers.append(k)
                fields.append(v)

            output_csv(output_filename, headers, fields)
            return accuracy_status

        if name in self.skip_accuracy_checks_large_models_dashboard:
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
                fp64_outputs = self.run_n_iterations(model_fp64, inputs_fp64)
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
                model_copy = self.deepcopy_and_maybe_parallelize(model)
                self.init_optimizer(name, current_device, model_copy.parameters())
                correct_result = self.run_n_iterations(
                    model_copy, clone_inputs(example_inputs)
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
                model_copy = self.deepcopy_and_maybe_parallelize(model)
                self.init_optimizer(name, current_device, model_copy.parameters())
                correct_rerun_result = self.run_n_iterations(
                    model_copy, clone_inputs(example_inputs)
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

            # Two eager runs should have exactly same result
            is_same = True
            try:
                if (
                    name not in self.skip_accuracy_check_as_eager_non_deterministic
                    and not same(
                        correct_result,
                        correct_rerun_result,
                        fp64_ref=None,
                        cos_similarity=False,
                        tol=0,
                        equal_nan=self.equal_nan,
                    )
                ):
                    is_same = False
            except Exception as e:
                # Sometimes torch.allclose may throw RuntimeError
                is_same = False

            if not is_same:
                accuracy_status = "eager_two_runs_differ"
                return record_status(accuracy_status, dynamo_start_stats=start_stats)

            correct_rerun_result = None

            # Run with Dynamo
            reset_rng_state()
            torch._dynamo.reset()
            model_copy = None
            try:
                model_copy = self.deepcopy_and_maybe_parallelize(model)
                self.init_optimizer(name, current_device, model_copy.parameters())
                if self.args.export or self.args.export_aot_inductor:
                    # apply export on module directly
                    # no need for n iterations
                    # the logic should be the same to self.model_iter_fn (forward_pass)
                    with self.autocast(**self.autocast_arg):
                        optimized_model_iter_fn = optimize_ctx(
                            model_copy, example_inputs
                        )
                        new_result = optimized_model_iter_fn(model_copy, example_inputs)
                else:
                    optimized_model_iter_fn = optimize_ctx(self.run_n_iterations)
                    with maybe_enable_compiled_autograd(self.args.compiled_autograd):
                        new_result = optimized_model_iter_fn(model_copy, example_inputs)
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

            if (
                current_onnx_compiler == "torchscript"
                or current_onnx_compiler == "dynamo"
            ):
                # Workaround for ONNX for non-tensor outputs
                (
                    correct_result,
                    new_result,
                    fp64_outputs,
                ) = _OnnxPatch.patch_non_tensor_outputs(
                    correct_result, new_result, fp64_outputs
                )
                # Relax tolerance for ONNX cuda
                if current_device == "cuda":
                    tolerance = 1e-2

                # TODO: store correct_result into the dumped file for offline onnx model validation.
                # The downside and potential problem, is that the output formats may be different.
                # E.g., the output order might not match, None might be part of output, etc.

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
                    cos_similarity=cos_similarity,
                    tol=tolerance,
                ):
                    is_same = False
            except Exception as e:
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
            correct_result = self.run_n_iterations(model_copy, example_inputs_copy)

            # Run with Dynamo
            # Sometime CI fails with random triton compilation failure which will be skipped for now
            # TODO: revisit this after switching to new Triton runtime
            reset_rng_state()
            torch._dynamo.reset()
            try:
                self.init_optimizer(name, current_device, model.parameters())
                optimized_model_iter_fn = optimize_ctx(self.run_n_iterations)
                new_result = optimized_model_iter_fn(model, example_inputs)
            except Exception as e:
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
            output_csv(output_filename, headers, fields)
        return tolerance_status

    def run_performance_test(
        self, name, model, example_inputs, optimize_ctx, experiment, tag=None
    ):
        if self.args.xla:
            with self.pick_grad(name, self.args.training):
                return experiment(*self.maybe_cast(model, example_inputs))

        def warmup(fn, model, example_inputs, mode, niters=5):
            peak_mem = 0
            start_stats = get_dynamo_stats()
            try:
                if current_device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    empty_gpu_cache(current_device)
                t0 = time.perf_counter()
                for _ in range(niters):
                    fn(model, example_inputs)
                t1 = time.perf_counter()
                latency = t1 - t0
                if current_device == "cuda":
                    peak_mem = get_peak_memory()
                elif current_device == "cpu":
                    total = psutil.virtual_memory().total
                    percentage = psutil.Process(os.getpid()).memory_percent()
                    peak_mem = percentage * total / 10**9
            except Exception:
                log.exception("Backend %s failed in warmup()", mode)
                return sys.exit(-1)
            dynamo_stats = get_dynamo_stats()
            dynamo_stats.subtract(start_stats)
            return latency, peak_mem, dynamo_stats

        # Cast the model to float16/float32 as necessary
        model, example_inputs = self.maybe_cast(model, example_inputs)

        # Use distributed wrapping as necessary
        model = self.deepcopy_and_maybe_parallelize(model)

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

            if self.args.export_aot_inductor:
                t_0 = time.perf_counter()
                optimized_model_iter_fn = optimize_ctx
                t_1 = time.perf_counter()
                aot_compilation_time = t_1 - t_0
            else:
                optimized_model_iter_fn = optimize_ctx(self.model_iter_fn)
                aot_compilation_time = 0

            with maybe_enable_compiled_autograd(
                self.args.compiled_autograd
            ), maybe_snapshot_memory(
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

            if self.args.profile_dynamo_cache_lookup:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as prof:
                    with maybe_enable_compiled_autograd(self.args.compiled_autograd):
                        warmup(optimized_model_iter_fn, model, example_inputs, "dynamo")

                events = list(
                    filter(
                        lambda event: "TorchDynamo Cache Lookup" in event.key,
                        prof.key_averages(),
                    )
                )
                dynamo_cache_lookup_latency = events[0].self_cpu_time_total

            compilation_time = dynamo_latency - eager_latency + aot_compilation_time
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
                    experiment_kwargs[
                        "cache_lookup_latency"
                    ] = dynamo_cache_lookup_latency

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

            if experiment.func is speedup_experiment_onnx:
                experiment = functools.partial(
                    experiment, optimized_model_iter_fn.context.onnx_model
                )

            if not hasattr(model, name):
                model.name = name
            results.append(experiment(model, example_inputs, **experiment_kwargs))
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
        logging.info("Minifying %s...", name)
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
        except OSError as e:
            logging.error("Could not find repro script for model %s", name)
        else:
            logging.info(
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
            status = self.run_performance_test(
                name, model, example_inputs, optimize_ctx, experiment, tag
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
                output_csv(
                    filename,
                    ["model", "reason", "user_stack"],
                    [current_name, reason, user_stack],
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
        choices=range(1, 10),
        help="Total number of partitions we want to divide the benchmark suite into",
    )
    parser.add_argument(
        "--partition-id",
        type=int,
        default=0,
        help="ID of the benchmark suite partition to be run. Used to divide CI tasks",
    )
    parser.add_argument(
        "--devices", "--device", "-d", action="append", help="cpu or cuda"
    )
    parser.add_argument("--device-index", help="CUDA device index")
    parser.add_argument(
        "--repeat", "-n", type=int, default=30, help="number of timing runs"
    )
    iterations_per_run_help = """
        Run this may iterations for each time measurement. This is mainly used for
        XLA training. We want to run multiple iterations per measurement so the
        tracing and computation for different iteartions can overlap with each
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
        help="Generates AOT Autograd stats like how mnay graphs are sent to AOT",
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
        help="Run model(s) twice and preseve caches in between to enable a 'warm start' on the 2nd run",
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
        "--xla", action="store_true", help="Compare TorchXLA to eager PyTorch"
    )
    group.add_argument(
        "--torchscript-onnx",
        "--torchscript_onnx",
        action="store_true",
        help="Measure speedup with TorchScript ONNX, i.e. `torch.onnx.export`",
    )
    group.add_argument(
        "--dynamo-onnx",
        "--dynamo_onnx",
        action="store_true",
        help="Measure speedup with Dynamo ONNX, i.e. `torch.onnx.dynamo_export`",
    )
    group.add_argument(
        "--dynamo-onnx-aot-inline",
        "--dynamo_onnx_aot_inline",
        action="store_true",
        help="Measure speedup with Dynamo ONNX AOT Inline, i.e. `torch.onnx.dynamo_export`",
    )
    group.add_argument(
        "--dynamo-onnx-aot-optimize",
        "--dynamo_onnx_aot_optimize",
        action="store_true",
        help="Measure speedup with Dynamo ONNX w/ ort fusions, i.e. `torch.onnx.dynamo_export`",
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


def process_entry(rank, runner, original_dir, args):
    args.rank = rank
    with maybe_init_distributed(
        args.init_distributed,
        rank=rank,
        world_size=args.world_size,
        port=args.distributed_master_port,
    ):
        return run(runner, args, original_dir)


def maybe_fresh_cache(args):
    cache_dir_assigned = "TORCHINDUCTOR_CACHE_DIR" in os.environ
    if not cache_dir_assigned and (
        args.cold_start_latency or args.warm_start_latency or args.ci
    ):
        return fresh_inductor_cache()
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
        output_csv(output_filename, headers, row)


def run(runner, args, original_dir=None):
    # Pass the parsed args object to benchmark runner object
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

        ci = functools.partial(
            CI, args.backend, training=args.training, dynamic=args.dynamic_shapes
        )
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
        }:
            # some of the models do not support use_deterministic_algorithms
            torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False

        # Remove randomeness when torch manual seed is called
        patch_torch_manual_seed()

        # Some models e.g. yolov3 assert batch size on n_gpus
        if "CUDA_VISIBLE_DEVICES" not in os.environ and not args.multiprocess:
            args.device_index = "0"

        # Stricter check to disable fallbacks
        args.suppress_errors = False

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

    if (
        args.devices == ["cuda"]
        and torch.cuda.get_device_properties(0).total_memory < 25 * 2**30
    ):
        # OOM errors on an RTX 3090 with 24gb RAM
        runner.skip_models.update(
            {
                # torchbench
                "hf_Longformer",
                "timm_nfnet",
                "timm_efficientdet",
            }
        )
        if args.training:
            runner.skip_models.add("hf_T5")

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
        runner.skip_models.update(runner.very_slow_models)
        runner.skip_models.update(runner.skip_models_for_cpu)
    elif args.devices == ["cuda"]:
        runner.skip_models.update(runner.skip_models_for_cuda)

    if not args.multiprocess:
        runner.skip_models.update(runner.skip_multiprocess_models)

    if args.freezing:
        runner.skip_models.update(runner.skip_models_for_freezing)

    if args.no_skip:
        runner.skip_models.clear()

    experiment = null_experiment
    global current_name, current_device, current_batch_size, output_filename, disable_output, optimize_ctx, current_onnx_compiler
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
    elif args.xla:
        (dev,) = args.devices
        os.environ["PJRT_DEVICE"] = {"cuda": "GPU", "cpu": "CPU"}[dev]
        torch._dynamo.mark_dynamic = MagicMock()
        experiment = xla
        output_filename = "xla.csv"
    elif args.torchscript_onnx:
        optimize_ctx = functools.partial(
            optimize_onnx_ctx,
            args.output_directory or ".",
            OnnxModelFromTorchScript,
            copy_before_export=args.performance,  # Accuarcy bench already did deepcopy
        )
        experiment = speedup_experiment_onnx
        output_filename = "torchscript_onnx.csv"
        current_onnx_compiler = "torchscript"
    elif args.dynamo_onnx:
        optimize_ctx = functools.partial(
            optimize_onnx_ctx,
            args.output_directory or ".",
            OnnxModelFromDynamo,
            dynamic_shapes=args.dynamic_shapes,
            copy_before_export=args.performance,
        )
        experiment = speedup_experiment_onnx
        output_filename = "dynamo_onnx.csv"
        current_onnx_compiler = "dynamo"
    elif args.dynamo_onnx_aot_inline:
        optimize_ctx = functools.partial(
            optimize_onnx_ctx,
            args.output_directory or ".",
            OnnxModelFromDynamoAotInline,
            dynamic_shapes=args.dynamic_shapes,
            copy_before_export=args.performance,
        )
        experiment = speedup_experiment_onnx
        output_filename = "dynamo_onnx_aot_inline.csv"
        current_onnx_compiler = "dynamo"
    elif args.dynamo_onnx_aot_optimize:
        optimize_ctx = functools.partial(
            optimize_onnx_ctx,
            args.output_directory or ".",
            OnnxModelFromDynamoAotOptimize,
            dynamic_shapes=args.dynamic_shapes,
            copy_before_export=args.performance,
        )
        experiment = speedup_experiment_onnx
        output_filename = "dynamo_onnx_aot_optimize.csv"
        current_onnx_compiler = "dynamo"
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
                export_aot_inductor, device=args.devices[0]
            )

            # AOTInductor doesn't support control flow yet
            runner.skip_models.update(runner.skip_models_due_to_control_flow)
        elif args.backend == "torchao":
            assert "cuda" in args.devices, "Quantization requires CUDA device."
            assert args.bfloat16, "Quantization requires dtype bfloat16."
            try:
                from torchao_backend import setup_baseline, torchao_optimize_ctx
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
            runner.model_iter_fn = baseline_ctx(runner.model_iter_fn)
            optimize_ctx = torchao_optimize_ctx(args.quantization)
        else:
            optimize_ctx = torch._dynamo.optimize(args.backend, nopython=args.nopython)
        experiment = speedup_experiment
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

    if args.inductor or args.backend == "inductor" or args.export_aot_inductor:
        inductor_config.triton.cudagraphs = not args.disable_cudagraphs
        inductor_config.triton.persistent_reductions = (
            not args.disable_persistent_reductions
        )
        inductor_config.split_reductions = not args.disable_split_reductions
        inductor_config.triton.divisible_by_16 = not args.disable_divisible_by_16
        if args.inference:
            inductor_config.freezing = args.freezing

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
            output_csv(output_filename, [], [args.only, batch_size])
        return

    if args.export_profiler_trace:
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

    experiment = functools.partial(experiment, args, runner.model_iter_fn)

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
                        torch._dynamo.mark_dynamic(t, i)
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

            with guard_ctx:
                runner.run_one_model(
                    name,
                    model,
                    example_inputs,
                    optimize_ctx,
                    experiment,
                    explain=args.explain,
                    tag=args.tag,
                )
        if args.generate_aot_autograd_stats:
            stats_file = output_filename.split(".csv")[0] + "_stats.csv"
            output_csv(
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
        if output_filename and os.path.exists(output_filename):
            os.unlink(output_filename)
        if original_dir:
            os.chdir(original_dir)
        model_names = list(runner.iter_model_names(args))
        nmodels = len(model_names)
        for i, name in enumerate(model_names):
            current_name = name
            if args.progress:
                print(f"Running model {i+1}/{nmodels}", flush=True)

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
