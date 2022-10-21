import argparse
import copy
import functools
import importlib
import itertools
import os
import sys
import time
from contextlib import nullcontext
from functools import partial
from os.path import abspath, exists
from typing import List

import functorch.compile
import numpy as np
import tabulate
import torch

import torch._dynamo as dynamo
import torch.distributed as dist
import torch.fx as fx
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils._pytree as pytree
import transformers
from torch._dynamo.optimizations import BACKENDS
from torch._dynamo.optimizations.distributed import DDPOptimizer
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import (
    always_wrap_policy,
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity, record_function
from traitlets.config.loader import ArgumentError


def setup_torchbench_env():
    original_dir = abspath(os.getcwd())

    os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam
    for torchbench_dir in (
        "./torchbenchmark",
        "../torchbenchmark",
        "../torchbench",
        "../benchmark",
        "../../torchbenchmark",
        "../../torchbench",
        "../../benchmark",
    ):
        if exists(torchbench_dir):
            break

    if exists(torchbench_dir):
        torchbench_dir = abspath(torchbench_dir)
        os.chdir(torchbench_dir)
        sys.path.append(torchbench_dir)
    return original_dir


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class CustomLinear(torch.nn.Module):
    def __init__(self, a, b):
        super(CustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(a, b))

    def forward(self, x):
        return torch.mm(x, self.weight)


class MyModule(torch.nn.Module):
    def __init__(self, a, b):
        super(MyModule, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(a, b),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net = nn.Sequential(
            *[nn.Linear(10, 10000), nn.ReLU()]
            + [nn.Linear(10000, 10000), nn.ReLU()]
            + [MyModule(10000, 10000)]
            + [MyModule(10000, 1000)]
            + [MyModule(1000, 1000)]
            + [MyModule(1000, 1000)]
            + [MyModule(1000, 1000)]
            + [MyModule(1000, 1000)]
            + [MyModule(1000, 1000)]
            + [MyModule(1000, 1000)]
            + [MyModule(1000, 1000)]
            + [nn.Linear(1000, 5)]
        )

    def forward(self, x):
        return self.net(x)


def fsdp_checkpointing_base(model, blocks):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, blocks)

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )


from transformers.models.t5.modeling_t5 import T5Block

MODEL_FSDP_WRAP = {
    ToyModel: (MyModule,)
    # TODO T5: (T5Block,)
}


def apply_fsdp(model, use_checkpointing=False, use_wrap_policy=True):
    blocks = MODEL_FSDP_WRAP[model.__class__]

    wrap_policy = None
    if use_wrap_policy:
        # transformer policy is really a generic policy that wraps modules of specified classes
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=blocks
        )

    model = FSDP(model, auto_wrap_policy=wrap_policy)
    if use_checkpointing:
        fsdp_checkpointing_base(model, blocks)

    return model


unpack_logits_types = (
    transformers.modeling_outputs.MaskedLMOutput,
    transformers.modeling_outputs.Seq2SeqLMOutput,
)


def unpack_outputs(outputs):
    if isinstance(outputs, unpack_logits_types):
        return outputs.logits
    return outputs


def run_model(args, model, inputs, rank, world_size, key, result_q):
    setup(rank, world_size)
    if args.device == "cuda":
        # needed for FSDP
        torch.cuda.set_device(rank)

    dev_rank = f"{args.device}:{rank}"
    model = model.to(dev_rank)

    def move_tensor(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(dev_rank)
        return maybe_tensor

    inputs = pytree.tree_map(move_tensor, inputs)

    if args.fsdp:
        model = apply_fsdp(
            model,
            use_checkpointing=args.fsdp_checkpoint,
            use_wrap_policy=args.fsdp_wrap,
        )
        print(model)
    elif args.ddp:
        model = DDP(model)

    if args.dynamo:
        if args.disable_fake_tensor:
            dynamo.config.aot_use_fake_tensor = False
            functorch.compile.config.use_fake_tensor = False
        if args.verbose:
            dynamo.config.verbose = True
        def print_compile(gm, ex):
            print(f"print_compile:\n{str(gm.graph)}\n-----------------------------------------")
            return gm
        dynamo_ctx = dynamo.optimize(print_compile if args.dynamo == "print" else args.dynamo)
        model = dynamo_ctx(model)

    # warmup
    for i in range(3):
        outputs = model(*inputs)
        outputs = unpack_outputs(outputs)
        outputs.sum().backward()

    # timing
    times = []
    for i in range(args.repeat):
        t0 = time.time()
        outputs = model(*inputs)
        outputs = unpack_outputs(outputs)
        outputs.sum().backward()
        t1 = time.time()
        times.append(t1 - t0)

    if rank == 0:
        result_q.put(times)

    if args.profile:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i in range(3):
                with record_function("Forward"):
                    outputs = model(*inputs)
                    outputs = unpack_outputs(outputs)
                with record_function("Backward"):
                    outputs.sum().backward()
        if rank == 0:
            prof.export_chrome_trace(args.trace_file)

    cleanup()


def experiment(fn, key, world_size, results):
    # tag = "opt" if optimize_ddp else "noopt"
    key = f"{key}_{world_size}"
    dynamo.reset()
    ctx = mp.get_context("spawn")
    # just get a time from rank0
    result_q = ctx.SimpleQueue()
    f_args = (world_size, key, result_q)
    if world_size > 1:
        mp.spawn(
            fn,
            args=f_args,
            nprocs=world_size,
            join=True,
        )
    else:
        # rank 0
        fn(0, *f_args)
    times = result_q.get()

    results.append((key, np.median(times)))
    # print(key, times, np.median(times))


def print_ddp_buckets(args, model, inputs):
    setup(0, 1)

    def move_tensor(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        return maybe_tensor

    inputs = pytree.tree_map(move_tensor, inputs)
    model = model.cuda()
    ddp_model = DDP(copy.deepcopy(model))
    for _ in range(3):
        # warmup
        outputs = ddp_model(*inputs)
        outputs = unpack_outputs(outputs)
        outputs.sum().backward()
    buckets = ddp_model.reducer._get_zeros_like_grad_buckets()
    assert all([b.buffer().dim() == 1 for b in buckets])
    ddp_buckets = [int(b.buffer().storage().nbytes()) for b in buckets]
    # print(f"DDP Buckets {ddp_buckets}")

    # build our own ddp-optimizer so we can get its internal state- so don't double-optimize
    dynamo.config.optimize_ddp = False
    ddp_opt = DDPOptimizer(
        ddp_model.bucket_bytes_cap,
        parameters_to_ignore=[],
        backend_compile_fn=BACKENDS["aot_eager"],
        debug=True,
    )
    dynamo_ctx = dynamo.optimize(ddp_opt.compile_fn)
    # don't reuse ddp_model since we want to ensure we're not changing the behavior of dynamo+ddp
    dynamo_model = dynamo_ctx(DDP(copy.deepcopy(model)))
    for _ in range(1):
        # warmup
        outputs = ddp_model(*inputs)
        outputs = unpack_outputs(outputs)
        outputs.sum().backward()
    opt_buckets = list(reversed(ddp_opt.bucket_actual_sizes))
    # opt_names = "\n".join(map(str, ddp_opt.bucket_param_names))
    opt_names = ""  # todo
    headers = ("index", "DDP sz", "DDP-Opt sz", "Status", "DDP-Opt params")
    rows = []
    n_buckets = len(ddp_buckets)
    for i in range(n_buckets):
        opt = opt_buckets[i] if i < len(opt_buckets) else None
        mismatch = "error" if opt != ddp_buckets[i] else ""
        rows.append([i, ddp_buckets[i], opt, mismatch, opt_names])
    for i, opt in enumerate(opt_buckets[n_buckets:]):
        rows.append([i, "", opt, "!!!", ""])

    rows.append([])
    s_d = sum(ddp_buckets)
    s_o = sum(opt_buckets)
    rows.append(["SUM", s_d, s_o, "error" if s_d != s_o else None, None])

    print(tabulate.tabulate(rows, headers=headers, tablefmt="rounded_grid"))
    print(
        "Buckets printed in order of execution (0 first, corresponding to last output layers of fwd)"
    )
    cleanup()


def get_model(args):
    if args.torchbench_model:
        setup_torchbench_env()
        module = importlib.import_module(
            f"torchbenchmark.models.{args.torchbench_model}"
        )
        benchmark_cls = getattr(module, "Model", None)
        bm = benchmark_cls(
            test="train", device=args.device, jit=False, batch_size=args.batch_size
        )
        model, inputs = bm.get_module()
    elif args.toy_model:
        model = ToyModel()
        inputs = (torch.randn(20, 10),)
    else:
        raise argparse.ArgumentError(
            args.torchbench_model, message="Must specify a model"
        )

    return model, inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dynamo",
        default=None,
        help="if set to a str, uses dynamo[str] backend. else, eager",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--disable_fake_tensor", action="store_true")
    parser.add_argument("--batch_size", default=None)
    parser.add_argument("--print_ddp_buckets", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Run the profiler")
    parser.add_argument("--trace_file", default="profile.json", help="Run the profiler")
    parser.add_argument("--repeat", default=10, help="Repeats for timing run")
    parser.add_argument(
        "--world_size", type=int, default=2, help="Number of ranks/gpus for experiments"
    )
    parser.add_argument(
        "--fsdp_checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing via model-specific policy",
    )
    parser.add_argument(
        "--fsdp_wrap",
        action="store_true",
        help="whether to apply fsdp to submodules via model-specific policy",
    )

    dist_arg = parser.add_mutually_exclusive_group()
    dist_arg.add_argument("--ddp", action="store_true")
    dist_arg.add_argument("--fsdp", action="store_true")

    model_arg = parser.add_mutually_exclusive_group(required=True)
    model_arg.add_argument(
        "--torchbench_model", help="name of torchbench model, e.g. hf_Bert"
    )
    model_arg.add_argument(
        "--toy_model", action="store_true", help="use toy model instead"
    )
    args = parser.parse_args()

    if args.disable_fake_tensor and (args.ddp or args.print_ddp_buckets):
        raise ArgumentError(
            args.disable_fake_tensor, "can't disable fake tensor with DDP, it crashes"
        )

    model_name = "ToyModel" if args.toy_model else args.torchbench_model
    model, inputs = get_model(args)

    fn = partial(run_model, args, model, inputs)

    if args.print_ddp_buckets:
        print_ddp_buckets(args, model, inputs)
        exit(0)

    times = []
    # if args.world_size > 1:
    experiment(fn, model_name, args.world_size, times)
    # else:
    # fn(0, 1, model_name, times)
    print("\nExperiment Results:")
    print(tabulate.tabulate(times, headers=("key", "time")))
