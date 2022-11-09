#!/usr/bin/env python3
import argparse
import collections
import copy
import csv
import functools
import io
import logging
import os
import random
import signal
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch

import torch._dynamo
import torch._dynamo.utils
import torch.distributed
from microbenchmarks.operator_inp_utils import OperatorInputsMode
from scipy.stats import gmean, ttest_ind
from torch._dynamo.optimizations import backends
from torch._dynamo.optimizations.log_args import conv_args_analysis
from torch._dynamo.profiler import fx_insert_profiling, Profiler
from torch._dynamo.testing import dummy_fx_compile, format_speedup, same
from torch._dynamo.utils import clone_inputs
from torch._inductor import config as inductor_config
from torch._inductor.utils import fresh_inductor_cache
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils._pytree import tree_map

try:
    from functorch._src.aot_autograd import set_model_name
except ImportError:

    def set_model_name(name):
        pass


log = logging.getLogger(__name__)

# We are primarily interested in TF32
torch.backends.cuda.matmul.allow_tf32 = True

current_name = ""
current_device = ""
current_batch_size = None
output_filename = None

CI_SKIP_AOT_EAGER_INFERENCE = [
    # TorchBench
    "demucs",  # OOM
    # Huggingface
    "AllenaiLongformerBase",
    "BartForConditionalGeneration",  # OOM
]

CI_SKIP_AOT_EAGER_TRAINING = [
    *CI_SKIP_AOT_EAGER_INFERENCE,
    # TorchBench
    "Background_Matting",  # fp64_OOM
    "moco",
    "pytorch_struct",
    "vision_maskrcnn",
    # Huggingface
    "AlbertForMaskedLM",  # OOM
    "AlbertForQuestionAnswering",  # OOM
    "BigBird",
    "M2M100ForConditionalGeneration",  # OOM
    "PegasusForConditionalGeneration",  # OOM
    "XGLMForCausalLM",  # OOM
    "XLNetLMHeadModel",  # OOM
    "YituTechConvBert",
    # TIMM
    "cait_m36_384",  # fp64_OOM
    "convit_base",  # fp64_OOM
    "mobilevit_s",  # Accuracy
    "xcit_large_24_p8_224",  # fp64_OOM
]

CI_SKIP_INDCUTOR_INFERENCE = [
    *CI_SKIP_AOT_EAGER_INFERENCE,
    # TorchBench
    "detectron2",
    "hf_Reformer",
    "moco",  # accuracy
    "pyhpc_equation_of_state",  # Accuracy
    "pyhpc_turbulent_kinetic_energy",  # Accuracy
    "tacotron2",
    "vision_maskrcnn",  # accuracy
    "yolov3",  # Accuracy
    # Huggingface
    "BigBird",
    "YituTechConvBert",
    # TIMM
    "cait_m36_384",  # Accuracy
    "ghostnet_100",  # Accuracy
    "swin_base_patch4_window7_224",  # Accuracy
    # Trying to get CI working - https://github.com/pytorch/pytorch/pull/87588
    "visformer_small",  # fails accuracy on CI but passes locally
]

CI_SKIP_INDUCTOR_TRAINING = [
    # CI does not check accuracy for inductor training yet
    # *CI_SKIP_AOT_EAGER_TRAINING,
    # *CI_SKIP_INDCUTOR_INFERENCE,
    # TorchBench
    "attention_is_all_you_need_pytorch",
    "drq",
    "hf_Albert",
    "hf_Bart",
    "hf_GPT2",
    "hf_Reformer",
    "mobilenet_v3_large",
    "moco",
    "pytorch_struct",
    "vgg16",
    "speech_transformer",  # from functionalization
    "vision_maskrcnn",  # from functionalization
    "timm_efficientnet",  # from functionalization (only fails for inductor)
    "hf_Bert",
    "soft_actor_critic",
    "tacotron2",
    "yolov3",
    # OOM
    "Background_Matting",
    "fastNLP_Bert",
    "hf_BigBird",
    "mobilenet_v2",
    "mobilenet_v2_quantized_qat",
    "resnet50_quantized_qat",
    "timm_regnet",
    # Huggingface
    "AllenaiLongformerBase",
    "AlbertForMaskedLM",  # OOM
    "BartForConditionalGeneration",  # OOM
    "M2M100ForConditionalGeneration",  # OOM
    "MBartForConditionalGeneration",  # OOM
    "MT5ForConditionalGeneration",  # OOM
    "PegasusForConditionalGeneration",  # OOM
    "XGLMForCausalLM",  # fp64_OOM
    # OOM
    "BigBird",
    "TrOCRForCausalLM",
    "AlbertForQuestionAnswering",
    # TIMM
    "cait_m36_384",  # fp64_OOM
    "coat_lite_mini",  # time out
    "convit_base",  # fp64_OOM
    "gernet_l",  # accuracy
    "gluon_xception65",
    "hrnet_w18",  # accuracy
    "lcnet_0500",  # accuracy
    "levit_128",  # levit_128
    "rexnet_100",  # accuracy
    "swin_base_patch4_window7_224",
    "twins_pcpvt_base",  # time out
    "xcit_large_24_p8_224",  # fp64_OOM
]


def output_csv(filename, headers, row):
    assert filename
    existed = os.path.exists(filename)
    output = csv.writer(
        io.TextIOWrapper(
            open(filename, "ab", buffering=0),
            "utf-8",
            write_through=True,
        ),
        lineterminator="\n",
    )
    if not existed:
        output.writerow(headers)
    output.writerow([(f"{x:.4f}" if isinstance(x, float) else x) for x in row])


class NullContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@functools.lru_cache(None)
def patch_torch_manual_seed():
    """Make torch manual seed deterministic. Helps with accuracy testing."""

    def deterministic_torch_manual_seed(*args, **kwargs):
        from torch._C import default_generator

        seed = 1337
        import torch.cuda

        if not torch.cuda._is_in_bad_fork():
            torch.cuda.manual_seed_all(seed)
        return default_generator.manual_seed(seed)

    torch.manual_seed = deterministic_torch_manual_seed


def synchronize():
    pass


def print_summary(filename):
    if not (filename and os.path.exists(filename)):
        return
    data = pd.read_csv(filename)
    width = max(map(len, data.columns))
    for col in data.columns:
        try:
            if col in ("dev", "name", "batch_size"):
                continue
            elif col in ("pct_ops", "pct_time"):
                print(col.ljust(width), f"{data[col].mean():.1%}")
            elif col in ("graphs", "graph_calls", "captured_ops", "total_ops"):
                print(col.ljust(width), f"{data[col].mean():.1f}")
            elif col in ("compilation_latency"):
                print(col.ljust(width), f"mean={data[col].mean():.1f} seconds")
            elif col in ("compression_ratio"):
                print(col.ljust(width), f"mean={data[col].mean():.1f}x")
            else:
                cdata = data[col].clip(1)
                print(
                    col.ljust(width),
                    f"gmean={gmean(cdata):.2f}x mean={cdata.mean():.2f}x",
                )
        except Exception:
            pass


def timed(model, model_iter_fn, example_inputs, times=1, return_result=False):
    synchronize()
    reset_rng_state()
    t0 = time.perf_counter()
    # Dont collect outputs to correctly measure timing
    for _ in range(times):
        result = model_iter_fn(model, example_inputs, collect_outputs=False)
        synchronize()
    t1 = time.perf_counter()
    return (t1 - t0, result) if return_result else t1 - t0


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
    prof = torch._dynamo.utils.CompileProfiler()
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


def speedup_experiment(args, model_iter_fn, model, example_inputs, **kwargs):
    """
    Measure speedups over eager.

    Writes to ./speedups.csv
    """
    if args.dynamic_shapes:
        return speedup_experiment_ds(args, model_iter_fn, model, example_inputs)

    timings = np.zeros((args.repeat, 2), np.float64)
    # if we randomize the input, we should also check the result is correct
    should_check_result = should_randomize_input = args.randomize_input
    is_correct = True

    baseline_model_iter_fn = get_baseline_model_iter_fn(args, model_iter_fn)
    baseline_model = get_baseline_model(args, model)

    import contextlib

    @contextlib.contextmanager
    def maybe_profile(*args, **kwargs):
        if kwargs.pop("enabled", True):
            with torch.profiler.profile(*args, **kwargs) as p:
                yield p
        else:
            yield

    with maybe_profile(enabled=args.export_profiler_trace) as p:
        frozen_model_iter_fn = torch._dynamo.run(model_iter_fn)
        for rep in range(args.repeat):
            inputs = (
                randomize_input(copy.deepcopy(example_inputs))
                if should_randomize_input
                else example_inputs
            )

            # interleave the runs to handle frequency scaling and load changes
            timings[rep, 0], expected_output = timed(
                baseline_model, baseline_model_iter_fn, inputs, return_result=True
            )
            timings[rep, 1], actual_output = timed(
                model, frozen_model_iter_fn, inputs, return_result=True
            )
            if should_check_result:
                is_correct = is_correct and same(expected_output, actual_output)
    if args.export_profiler_trace:
        name = args.profiler_trace_name + "_" + model.name + ".json"
        name = os.path.join(torch._dynamo.config.base_dir, name)
        p.export_chrome_trace(name)
    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    median = np.median(timings, axis=0)
    speedup = median[0] / median[1]
    if args.dump_raw_metrics:
        np.save(
            f"{output_filename[:-4]}-raw_timings-{current_name}-{current_device}.npy",
            timings,
        )

    headers = ("dev", "name", "batch_size", "speedup", "abs_latency")
    row = [current_device, current_name, current_batch_size, float(speedup), median[1]]
    if "compilation_latency" in kwargs:
        headers = headers + ("compilation_latency", "compression_ratio")
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
        shape: list(
            map(
                lambda it: it[1],
                filter(lambda it: it[0] == shape, zip(shapes, speedups)),
            )
        )
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


def try_script(model, example_inputs):
    try:
        return torch.jit.script(model)
    except Exception:
        return None


def speedup_experiment_onnx(args, model_iter_fn, model, example_inputs):
    """
    Measure baseline performance (without using TorchDynamo) of ONNXRT and TensorFlow.

    Writes to ./baseline_onnx.csv
    """
    if current_device == "cpu":
        m_onnxrt = backends.onnxrt_cpu(
            try_script(model, example_inputs), example_inputs
        )
    else:
        m_onnxrt = backends.onnxrt_cuda(
            try_script(model, example_inputs), example_inputs
        )

    if current_name != "timm_resnest":
        m_onnx2tf = backends.onnx2tf(try_script(model, example_inputs), example_inputs)
    else:
        # this one takes 8+ hours to finish
        m_onnx2tf = None

    return baselines(
        [
            ("eager", model),
            ("onnxrt", m_onnxrt),
            ("onnx2tf", m_onnx2tf),
        ],
        model_iter_fn,
        example_inputs,
        args,
    )


def speedup_experiment_trt(args, model_iter_fn, model, example_inputs):
    """
    Measure baseline performance (without using TorchDynamo) of TensorRT.

    Writes to ./baseline_trt.csv
    """
    m_onnx2trt = backends.onnx2tensorrt(
        try_script(model, example_inputs), example_inputs
    )

    m_torch2trt = backends.torch2trt(model, example_inputs)

    if current_name != "opacus_cifar10":
        m_fx2trt = backends.fx2trt(model, example_inputs)
    else:
        # fx2trt infinite loops on one model
        m_fx2trt = None

    return baselines(
        [
            ("eager", model),
            ("onnx2trt", m_onnx2trt),
            ("torch2trt", m_torch2trt),
            ("fx2trt", m_fx2trt),
        ],
        model_iter_fn,
        example_inputs,
        args,
    )


def read_batch_size_from_file(args, filename, model_name):
    batch_size = None
    if os.path.exists("benchmarks"):
        filename = os.path.join("benchmarks", filename)
    assert os.path.exists(filename), filename
    with open(filename, "r") as f:
        lines = f.readlines()
        lines = [i.split(",") for i in lines if len(i.strip()) > 0]
        for val in lines:
            cur_name, b = val
            if model_name == cur_name:
                batch_size = int(b)
    if batch_size is None:
        log.warning("Could not find batch size for {}".format(model_name))
    elif batch_size == -1:
        raise RuntimeError(
            f"Batch size is unset for {model_name} in {args.batch_size_file}"
        )
    print(f"batch size: {batch_size}")
    return batch_size


class TimeOutException(Exception):
    pass


def alarm_handler(signum, frame):
    raise TimeOutException()


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


def cast_to_fp16(model, inputs):
    return cast_to(torch.float16, model, inputs)


def cast_to_fp64(model, inputs):
    return cast_to(torch.float64, model, inputs)


def cast_to_fp32(model, inputs):
    return cast_to(torch.float32, model, inputs)


def reset_rng_state():
    torch.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)


class DummyGradScaler:
    def scale(self, loss):
        return loss


def maybe_fresh_cache(fn, is_cold_start):
    def inner(*args, **kwargs):
        cache_minder = NullContext()
        if is_cold_start:
            cache_entries = {}
            cache_minder = fresh_inductor_cache(cache_entries)

        try:
            with cache_minder:
                return fn(*args, **kwargs)
        finally:
            dump_cache = False
            if dump_cache and is_cold_start:
                output_csv(
                    output_filename[:-4] + "_triton_cache.csv",
                    ["dev", "name", "batch_size", "triton_cache"],
                    [
                        current_device,
                        current_name,
                        current_batch_size,
                        cache_entries,
                    ],
                )

    return inner


@contextmanager
def maybe_init_distributed(should_init_distributed, port="6789", rank=0, world_size=1):
    # To avoid multiple inheritance from _dynamo.test_case.TestCase and MultiProcessTestCase,
    # Just manually implement the most important part of the dynamo behavior to reset/clear.
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


def xla_wrapper(model_iter_fn):
    """
    Wrap the model_iter_fn to run the model on XLA devices.
    """

    def wrapper(xla_mod, inputs, collect_outputs=True):
        import torch_xla.core.xla_model as xm

        # Make sure the model is already moved to the xla device. Moving
        # the model to xla device can be very expensive since model parameters
        # need to be copied. We should not do that inside the wrapper since
        # the wrapper will be calles for each set of inputs.
        assert (
            next(xla_mod.parameters()).device.type == "xla"
        ), "The model should be already on xla device"

        xla_dev = xm.xla_device()
        eager_dev = inputs[0].device
        xla_inputs = tree_map(lambda x: x.to(device=xla_dev), inputs)
        xla_out = model_iter_fn(xla_mod, xla_inputs, collect_outputs)
        if isinstance(xla_out, torch.Tensor):
            return xla_out.to(device=eager_dev)
        elif hasattr(xla_out, "__dict__"):
            for k in xla_out.__dict__.keys():
                if xla_out.__dict__[k] is None:
                    continue
                xla_out.__dict__[k] = tree_map(
                    lambda x: x.to(device=eager_dev), xla_out.__dict__[k]
                )
            return xla_out
        else:
            raise RuntimeError(f"Can not handle type {type(xla_out)}")

    return wrapper


def get_baseline_model_iter_fn(args, model_iter_fn):
    return xla_wrapper(model_iter_fn) if args.use_xla_baseline else model_iter_fn


def get_baseline_model(args, model):
    if args.use_xla_baseline:
        import torch_xla.core.xla_model as xm

        xla_dev = xm.xla_device()
        return copy.deepcopy(model).to(device=xla_dev)
    else:
        return model


class BenchmarkRunner:
    def __init__(self):
        self.model_iter_fn = None
        self.use_amp = False
        self.grad_scaler = DummyGradScaler()
        self.autocast = NullContext
        self._args = None

    def setup_amp(self):
        if self.args.amp and self.args.training:
            assert self.args.devices == ["cuda"], "AMP is supported only for CUDA"
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
            self.grad_scaler = torch.cuda.amp.GradScaler(init_scale=2.0)
            self.autocast = torch.cuda.amp.autocast

    def init_optimizer(self, device, params):
        self.optimizer = None
        # TODO - Currently, optimizers are used incorrectly. Fix optimizers with
        # https://github.com/pytorch/pytorch/pull/87492
        # param_list = list(params)
        # if device == "cuda" and len(param_list) != 0:
        #     # capturable is only supported on cuda at the moment
        #     self.optimizer = torch.optim.Adam(param_list, capturable=True)
        # else:
        #     self.optimizer = None

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
    def slow_models(self):
        return set()

    @property
    def very_slow_models(self):
        return set()

    @property
    def non_deterministic_models(self):
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
    def failing_dynamic_shape_models(self):
        return set()

    @property
    def skip_accuracy_checks_large_models_dashboard(self):
        return set()

    @property
    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        raise NotImplementedError()

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

    def validate_model(self, model, example_inputs):
        """
        Runs the eager model with example inputs to ensure that eager passes.
        """
        model = copy.deepcopy(model)
        example_inputs = clone_inputs(example_inputs)
        if self.args.float32:
            model, example_inputs = cast_to_fp32(model, example_inputs)
        elif self.args.float16:
            model, example_inputs = cast_to_fp16(model, example_inputs)

        try:
            self.model_iter_fn(model, example_inputs)
        except Exception:
            raise NotImplementedError("Eager model failed to run")

    def maybe_cast(self, model, example_inputs):
        model = copy.deepcopy(model)
        example_inputs = clone_inputs(example_inputs)
        if self.args.float32:
            model, example_inputs = cast_to_fp32(model, example_inputs)
        elif self.args.float16:
            model, example_inputs = cast_to_fp16(model, example_inputs)
        return model, example_inputs

    def decay_batch_exp(self, batch_size, factor=0.5, divisor=2):
        out_batch_size = batch_size * factor
        if out_batch_size > divisor:
            out_batch_size = (out_batch_size + 1) // divisor * divisor
        else:
            out_batch_size = batch_size - 1
        return max(0, int(out_batch_size))

    def batch_size_finder(self, device, model_name, initial_batch_size=128):
        batch_size = initial_batch_size
        while batch_size >= 1:
            torch.cuda.empty_cache()
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

    def run_n_iterations(self, mod, inputs, n=2):
        for _ in range(n - 1):
            self.model_iter_fn(mod, inputs, collect_outputs=False)
        return self.model_iter_fn(mod, inputs, collect_outputs=True)

    def optimizer_zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad(True)

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

    def check_accuracy(self, name, model, example_inputs, optimize_ctx, experiment):
        """
        Checks accuracy.
        1) Collect the outputs with fp64 datatype. This is useful for error checking.
        2) Checks if eager itself has variations.
        """

        def record_status(accuracy_status):
            """
            Records the status in the csv file
            """
            if current_name in self.non_deterministic_models:
                if accuracy_status in ("pass", "eager_variation", "fail_accuracy"):
                    accuracy_status = "pass"

            output_csv(
                output_filename,
                ("dev", "name", "batch_size", "accuracy"),
                [current_device, current_name, current_batch_size, accuracy_status],
            )
            return "PASS" if accuracy_status in ("pass", "pass_due_to_skip") else "FAIL"

        tolerance, cos_similarity = self.get_tolerance_and_cosine_flag(
            self.args.training, current_device, name
        )

        if name in self.skip_accuracy_checks_large_models_dashboard:
            return record_status("pass_due_to_skip")

        def deepcopy_and_maybe_ddp(model):
            model = copy.deepcopy(model)
            if self.args.ddp:
                model = DDP(model)
            return model

        # Collect the fp64 reference outputs to be used later for accuracy checking.
        fp64_outputs = None
        try:
            fp64_outputs = self.run_n_iterations(
                *cast_to_fp64(
                    deepcopy_and_maybe_ddp(model),
                    clone_inputs(example_inputs),
                )
            )
        except Exception:
            log.warning(f"fp64 golden ref were not generated for {name}")
            fp64_outputs = None
            if self.args.ci and self.args.training:
                return record_status("fp64_OOM")

        # Cast the model to float16/float32 as necessary
        model, example_inputs = self.maybe_cast(model, example_inputs)
        accuracy_status = "pass"

        with self.pick_grad(name, self.args.training):
            # Get results of native pytorch
            reset_rng_state()
            correct_result = self.run_n_iterations(
                deepcopy_and_maybe_ddp(model), clone_inputs(example_inputs)
            )

            # Rerun native pytorch
            reset_rng_state()
            correct_rerun_result = self.run_n_iterations(
                deepcopy_and_maybe_ddp(model), clone_inputs(example_inputs)
            )
            if not same(
                correct_result,
                correct_rerun_result,
                fp64_outputs,
                equal_nan=self.equal_nan,
            ):
                accuracy_status = "eager_variation"
                return record_status(accuracy_status)
            correct_rerun_result = None

            # Run with Dynamo
            reset_rng_state()
            torch._dynamo.reset()
            try:
                optimized_model_iter_fn = optimize_ctx(self.run_n_iterations)

                new_result = optimized_model_iter_fn(
                    deepcopy_and_maybe_ddp(model), example_inputs
                )
            except Exception as e:
                accuracy_status = "fail_to_run"
                print(
                    "TorchDynamo optimized model failed to run because of following error"
                )
                log.exception(e)
                return record_status(accuracy_status)

            if not same(
                correct_result,
                new_result,
                fp64_outputs,
                equal_nan=self.equal_nan,
                cos_similarity=cos_similarity,
                tol=tolerance,
            ):
                if self.args.skip_accuracy_check:
                    accuracy_status = "pass_due_to_skip"
                else:
                    accuracy_status = "fail_accuracy"
                return record_status(accuracy_status)

        return record_status(accuracy_status)

    def run_performance_test(
        self, name, model, example_inputs, optimize_ctx, experiment
    ):
        def warmup(fn, model, example_inputs, mode, niters=5):
            peak_mem = 0
            try:
                if current_device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                t0 = time.perf_counter()
                for _ in range(niters):
                    fn(model, example_inputs)
                t1 = time.perf_counter()
                latency = t1 - t0
                if current_device == "cuda":
                    peak_mem = get_peak_memory()
            except Exception as e:
                log.exception(f"Failed for {mode} {e}")
                return sys.exit(-1)
            return latency, peak_mem

        # Cast the model to float16/float32 as necessary
        model, example_inputs = self.maybe_cast(model, example_inputs)
        with self.pick_grad(name, self.args.training):
            ok, total = Stats.reset_counters()
            experiment_kwargs = {}
            results = []

            eager_latency, eager_peak_mem = warmup(
                self.model_iter_fn, model, example_inputs, "eager"
            )
            optimized_model_iter_fn = optimize_ctx(self.model_iter_fn)
            dynamo_latency, dynamo_peak_mem = warmup(
                optimized_model_iter_fn, model, example_inputs, "dynamo"
            )

            compilation_time = dynamo_latency - eager_latency
            compression_ratio = (
                eager_peak_mem / dynamo_peak_mem if dynamo_peak_mem else 0.0
            )
            # print(
            #     f"memory: eager: {eager_peak_mem:.2f} GB, "
            #     f"dynamo: {dynamo_peak_mem:.2f} GB, "
            #     f"ratio: {compression_ratio:.2f}"
            # )

            if experiment.func is speedup_experiment:
                experiment_kwargs["compilation_latency"] = compilation_time
                experiment_kwargs["compression_ratio"] = compression_ratio

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

            if not hasattr(model, name):
                model.name = name
            results.append(experiment(model, example_inputs, **experiment_kwargs))
            return " ".join(map(str, results))

    def compare_branches(
        self,
        name,
        model,
        example_inputs,
        optimize_ctx,
        experiment,
        diff=False,
        branch=None,
    ):
        assert branch is None, "Branch set during top level flow."
        import git

        repo = git.Repo(
            "../torch._dynamo"
        )  # Hack assumption of torchbenchmark positioning
        curr_branch = repo.active_branch.name
        if curr_branch != "main":
            if repo.is_dirty():
                raise RuntimeError(
                    "--diff_main called on dirty branch. Commit, stash, or reset."
                )
            # Run current
            try:
                self.run_one_model(
                    name,
                    model,
                    self.model_iter_fn,
                    example_inputs,
                    optimize_ctx,
                    experiment,
                    diff=False,
                    branch=curr_branch,
                )
                # Swap to main
                repo.git.checkout("main")
                # Run main
                self.run_one_model(
                    name,
                    model,
                    self.model_iter_fn,
                    example_inputs,
                    optimize_ctx,
                    experiment,
                    diff=False,
                    branch="main",
                )
            finally:
                # Swap back
                repo.git.checkout(curr_branch)
            return
        else:
            raise RuntimeError(
                "--diff_main called on main branch, what are you diffing?"
            )

    def run_one_model(
        self,
        name,
        model,
        example_inputs,
        optimize_ctx,
        experiment,
        diff=False,
        branch=None,
    ):
        if diff:
            self.compare_branches(
                name, model, example_inputs, optimize_ctx, experiment, diff, branch
            )
        elif branch:
            print("RUNNING ON BRANCH:", branch)
        mode = "train" if self.args.training else "eval"
        print(f"{current_device:4} {mode:5} {current_name:34} ", end="", flush=True)
        if self.args.accuracy:
            status = self.check_accuracy(
                name, model, example_inputs, optimize_ctx, experiment
            )
            print(status)
        elif self.args.performance:
            status = self.run_performance_test(
                name, model, example_inputs, optimize_ctx, experiment
            )
            print(status)


def help(fn):
    return fn.__doc__


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter", "-k", action="append", help="filter benchmarks with regexp"
    )
    parser.add_argument(
        "--exclude", "-x", action="append", help="filter benchmarks with regexp"
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
    parser.add_argument("--devices", "-d", action="append", help="cpu or cuda")
    parser.add_argument(
        "--repeat", "-n", type=int, default=30, help="number of timing runs"
    )
    parser.add_argument(
        "--randomize-input",
        action="store_true",
        help="Whether to randomize the input values. Dimensions will be kept the same.",
    )
    parser.add_argument(
        "--threads", "-t", type=int, help="number of threads to use for eager"
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
    parser.add_argument("--batch_size", type=int, help="batch size for benchmarking")
    parser.add_argument(
        "--batch-size-file", type=str, help="String to load batch size from"
    )
    parser.add_argument("--cosine", action="store_true", help="use cosine similarity")
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
    parser.add_argument("--only", help="Run just one model")
    parser.add_argument(
        "--training",
        action="store_true",
        help="Performs training",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Wraps model in DDP before running it, and uses dynamo DDPOptmizer (graph breaks) by default.",
    )
    parser.add_argument(
        "--no-optimize-ddp",
        action="store_true",
        help="Disables dynamo DDPOptimizer (graph breaks). (Applies only when using --ddp benchmark mode).",
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
        "--export-profiler-trace",
        action="store_true",
        help="exports trace of kineto profiler",
    )
    parser.add_argument("--profiler_trace_name", help="Overwrites exported trace name")

    parser.add_argument(
        "--diff_main",
        action="store_true",
        help="Delta this branch against main. In the future, we may add support for picking the branch.",
    )

    parser.add_argument(
        "--cold_start_latency",
        action="store_true",
        help="Use a fresh triton cachedir when running each model, to force cold-start compile.",
    )
    parser.add_argument(
        "--disable-cudagraphs",
        action="store_true",
        help="Disables cudagraphs for Inductor",
    )
    parser.add_argument(
        "--use-xla-baseline",
        action="store_true",
        help="Whether to run baseline on XLA devices or eager devices",
    )

    group_fuser = parser.add_mutually_exclusive_group()
    # --nvfuser is now the default, keep the option to not break scripts
    group_fuser.add_argument("--nvfuser", action="store_true", help=argparse.SUPPRESS)
    group_fuser.add_argument("--nnc", action="store_true", help="enable NNC for GPUs")

    group_prec = parser.add_mutually_exclusive_group()
    group_prec.add_argument("--float16", action="store_true", help="cast model to fp16")
    group_prec.add_argument("--float32", action="store_true", help="cast model to fp32")
    group_prec.add_argument(
        "--amp", action="store_true", help="use automatic mixed precision"
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
        "--speedup-onnx", action="store_true", help=help(speedup_experiment_onnx)
    )
    group.add_argument(
        "--speedup-trt", action="store_true", help=help(speedup_experiment_trt)
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
        "--inductor-dynamic",
        action="store_true",
        help="Measure speedup with TorchInductor",
    )
    group.add_argument(
        "--backend",
        choices=torch._dynamo.list_backends(),
        help="measure speedup with a given backend",
    )
    group.add_argument("--nothing", action="store_true", help=help(null_experiment))
    group.add_argument(
        "--log-conv-args",
        action="store_true",
        help="Dump convolution input/weight/bias's shape/stride/dtype and other options to json",
    )
    group.add_argument(
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
    args = parser.parse_args()
    return args


def main(runner, original_dir=None):
    args = parse_args()
    with maybe_init_distributed(
        args.ddp and args.only, port=args.distributed_master_port
    ):
        return maybe_fresh_cache(run, args.cold_start_latency and args.only)(
            runner, args, original_dir
        )


def run(runner, args, original_dir=None):
    # Pass the parsed args object to benchmark runner object
    runner.args = args

    args.filter = args.filter or [r"."]
    args.exclude = args.exclude or [r"^$"]

    if args.ci:
        # Only dump error on CI
        args.quiet = True
        args.repeat = 2
        if args.backend == "aot_eager":
            args.exclude = (
                CI_SKIP_AOT_EAGER_TRAINING
                if args.training
                else CI_SKIP_AOT_EAGER_INFERENCE
            )
        elif args.inductor:
            args.exclude = (
                CI_SKIP_INDUCTOR_TRAINING
                if args.training
                else CI_SKIP_INDCUTOR_INFERENCE
            )
    if args.ddp:
        # TODO: we could also hook DDP bench up to --speedup bench, _not_ for mgpu e2e perf,
        # but just to measure impact on singlenode of performing graph-breaks.
        # Left it as a follow up to keep this PR isolated.
        assert (
            args.accuracy
        ), "DDP benchmark is currently only hooked up to --accuracy bench"
        assert args.training, "DDP benchmark requires --training mode"
        if args.no_optimize_ddp:
            torch._dynamo.config.optimize_ddp = False
        else:
            # TODO(whc) after enabling DDPOptimizer by default this could be removed or assert
            torch._dynamo.config.optimize_ddp = True

    if args.accuracy:
        # Use small batch size. We use >1 batch size to ensure we test
        # batch_norm type of operators that work on batch dims.
        # TODO - Go through the failures for batch size = 2
        if args.batch_size is None:
            if runner.suite_name == "huggingface":
                args.batch_size = 1
            else:
                args.batch_size = 2

        # Remove sources of randomness
        args.use_eval_mode = True

        # Remove randomeness when torch manual seed is called
        patch_torch_manual_seed()

        # Some models e.g. yolov3 assert batch size on n_gpus
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # Stricter check to disable fallbacks
        args.suppress_errors = False

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

    if args.devices != ["cpu"] and torch.cuda.is_available():
        global synchronize
        synchronize = torch.cuda.synchronize

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
                # timm
                "beit_base_patch16_224",
                "cait_m36_384",
                "convmixer_768_32",
                "deit_base_distilled_patch16_224",
                "dm_nfnet_f0",
                "dpn107",
                "dm_nfnet_f0",
            }
        )
        if args.training:
            runner.skip_models.add("hf_T5")

    if torch._dynamo.config.dynamic_shapes:
        # TODO(jansel): fix bugs in these
        runner.skip_models.update(runner.failing_dynamic_shape_models)

    if args.nnc:
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(True)
        torch._C._jit_set_nvfuser_enabled(False)

    if args.threads:
        torch.set_num_threads(args.threads)

    if args.verbose:
        torch._dynamo.config.log_level = logging.DEBUG

    if args.quiet:
        torch._dynamo.config.log_level = logging.ERROR

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

    if args.inductor or args.inductor_dynamic or args.inductor_settings:
        runner.skip_models.update(runner.failing_torchinductor_models)
        if args.float16:
            # TODO(jansel): check if correctness issue is real
            runner.skip_models.add("yolov3")

    if args.float16:
        # these give `INCORRECT - Variation in Eager runs itself` sometimes
        runner.non_deterministic_models.update(
            {
                "demucs",
                "pyhpc_equation_of_state",
                "timm_efficientdet",
                "pyhpc_isoneutral_mixing",
                "pyhpc_turbulent_kinetic_energy",
                "shufflenet_v2_x1_0",
            }
        )

    if args.no_skip:
        runner.skip_models.clear()

    experiment = null_experiment
    global current_name, current_device, current_batch_size, output_filename, optimize_ctx
    optimize_ctx = NullContext()

    if args.overhead:
        optimize_ctx = torch._dynamo.optimize(dummy_fx_compile, nopython=args.nopython)
        experiment = speedup_experiment
        output_filename = "overheads.csv"
    elif args.inductor or args.inductor_dynamic:
        inductor_config.debug = args.verbose
        if args.threads:
            inductor_config.cpp.threads = args.threads

        if args.inductor_dynamic:
            inductor_config.triton.cudagraphs = False
            inductor_config.dynamic_shapes = True
        else:
            inductor_config.dynamic_shapes = False
            if args.export_profiler_trace:
                print("Profiling requested, setting cudagraphs to False")
                inductor_config.triton.cudagraphs = False

        optimize_ctx = torch._dynamo.optimize("inductor", nopython=args.nopython)
        experiment = speedup_experiment
        output_filename = "inductor.csv"
    elif args.speedup_onnx:
        experiment = speedup_experiment_onnx
        output_filename = "baseline_onnx.csv"
    elif args.speedup_trt:
        experiment = speedup_experiment_trt
        output_filename = "baseline_trt.csv"
    elif args.speedup_dynamo_ts:
        optimize_ctx = torch._dynamo.optimize(backends.ts, nopython=args.nopython)
        experiment = speedup_experiment
        output_filename = "speedup_dynamo_ts.csv"
    elif args.speedup_fx2trt:
        optimize_ctx = torch._dynamo.optimize(
            backends.fx2trt_compiler, nopython=args.nopython
        )
        experiment = speedup_experiment_fx2trt
        output_filename = "speedups_fx2trt.csv"
        runner.skip_models.update(runner.failing_fx2trt_models)
        args.float32 = True
        args.float16 = False
        args.cosine = True
    elif args.speedup_fx2trt_fp16:
        optimize_ctx = torch._dynamo.optimize(
            backends.fx2trt_compiler_fp16, nopython=args.nopython
        )
        experiment = speedup_experiment_fx2trt
        output_filename = "speedups_fx2trt_fp16.csv"
        args.float32 = False
        args.float16 = True
        args.cosine = True
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
        pass
    elif args.backend:
        optimize_ctx = torch._dynamo.optimize(args.backend, nopython=args.nopython)
        experiment = speedup_experiment
        if args.accuracy:
            output_filename = f"accuracy_{args.backend}.csv"
        else:
            output_filename = f"speedup_{args.backend}.csv"
    elif args.log_conv_args:
        optimize_ctx = torch._dynamo.optimize(
            conv_args_analysis, nopython=args.nopython
        )
        output_filename = "log_conv_args.csv"
    elif args.recompile_profiler:
        output_filename = "recompile_profiler_log.csv"
        experiment = recompile_profiler_experiment
    else:
        optimize_ctx = torch._dynamo.optimize(
            fx_insert_profiling, nopython=args.nopython
        )
        experiment = coverage_experiment
        output_filename = "coverage.csv"

    if args.inductor or args.backend == "inductor":
        if args.disable_cudagraphs:
            inductor_config.triton.cudagraphs = False

    runner.setup_amp()

    if args.output:
        output_filename = args.output

    if output_filename:
        output_filename = os.path.join(torch._dynamo.config.base_dir, output_filename)

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
            elif args.inductor or args.inductor_dynamic:
                args.profiler_trace_name = "inductor"
            else:
                args.profiler_trace_name = "profile"
        else:
            args.profiler_trace_name = args.profiler_trace_name

    experiment = functools.partial(experiment, args, runner.model_iter_fn)

    if args.only:
        model_name = args.only
        for device in args.devices:
            batch_size = args.batch_size
            if args.batch_size_file:
                batch_size = read_batch_size_from_file(
                    args, args.batch_size_file, model_name
                )
            try:
                device, name, model, example_inputs, batch_size = runner.load_model(
                    device,
                    model_name,
                    batch_size=batch_size,
                )
            except NotImplementedError as e:
                print(e)
                import traceback

                print(traceback.format_exc())
                logging.warn(f"{args.only} failed to load")
                continue  # bad benchmark implementation

            current_name = name
            current_device = device
            current_batch_size = batch_size
            set_model_name(name)

            if args.float32:
                model, example_inputs = cast_to_fp32(model, example_inputs)
            elif args.float16:
                model, example_inputs = cast_to_fp16(model, example_inputs)

            if args.log_operator_inputs:
                log_operator_inputs(
                    model, example_inputs, runner.model_iter_fn, name, args
                )
                continue

            runner.run_one_model(
                name,
                model,
                example_inputs,
                optimize_ctx,
                experiment,
                diff=args.diff_main,
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
        if output_filename and os.path.exists(output_filename):
            os.unlink(output_filename)
        if original_dir:
            os.chdir(original_dir)
        for name in runner.iter_model_names(args):
            current_name = name
            placeholder_batch_size = 0
            try:
                subprocess.check_call([sys.executable] + sys.argv + [f"--only={name}"])
            except subprocess.SubprocessError:
                print("ERROR")
                for device in args.devices:
                    output_csv(
                        output_filename, [], [device, name, placeholder_batch_size, 0.0]
                    )
        print_summary(output_filename)


def log_operator_inputs(model, example_inputs, model_iter_fn, name, args):
    mode = "training" if args.training else "eval"
    output = os.path.join(os.path.dirname(args.output), f"{name}_{mode}.txt")

    # TODO - add option for coalescing inputs over multiple runs
    if os.path.exists(output):
        print(f"Skipping {name}, {output} already exists")
        return

    print(f"Running {name}")

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
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main()
