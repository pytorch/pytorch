import argparse
from functools import partial

import numpy as np
import tabulate
import torch

import torch._dynamo as dynamo
import torch.multiprocessing as mp
import torch.utils._pytree as pytree
from torch._dynamo.testing import reduce_to_scalar_loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity, record_function

try:
    from .common import timed
    from .dist_util import apply_fsdp, cleanup, get_model, model_iter_fn, setup
except ImportError:
    from common import timed
    from dist_util import apply_fsdp, cleanup, get_model, model_iter_fn, setup


def profile_model(args, model, inputs, rank):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(args.repeat):
            with record_function("Forward"):
                outputs = model(*inputs)
                loss = reduce_to_scalar_loss(outputs)
            with record_function("Backward"):
                loss.backward()
    if rank == 0:
        prof.export_chrome_trace(args.trace_file)


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
    elif args.ddp:
        model = DDP(model)

    if args.verbose:
        print(model)

    if args.dynamo:
        if args.verbose:
            dynamo.config.verbose = True
        if args.dynamo_optimize_ddp:
            dynamo.config.optimize_ddp = True

        def print_compile(gm, ex):
            print(
                f"print_compile:\n{str(gm.graph)}\n-----------------------------------------"
            )
            return gm

        dynamo_ctx = dynamo.optimize(
            print_compile if args.dynamo == "print" else args.dynamo
        )
        model = dynamo_ctx(model)

    # warmup
    _ = timed(model, model_iter_fn, inputs, times=3, return_result=False)
    times = []
    t_total = timed(
        model, model_iter_fn, inputs, times=args.repeat, return_result=False
    )
    times.append(t_total / args.repeat)

    if rank == 0:
        result_q.put(times)

    if args.profile:
        profile_model(args, model, inputs, rank)

    cleanup()


def experiment(fn, key, world_size, results):
    key = f"{key}_{world_size}"
    dynamo.reset()
    ctx = mp.get_context("spawn")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dynamo",
        default=None,
        help="if set to a str, uses dynamo[str] backend. else, eager",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--batch_size", default=None)
    parser.add_argument("--profile", action="store_true", help="Run the profiler")
    parser.add_argument("--trace_file", default="profile.json", help="Run the profiler")
    parser.add_argument("--repeat", default=10, help="Repeats for timing run")
    parser.add_argument(
        "--world_size", type=int, default=2, help="Number of ranks/gpus for experiments"
    )
    parser.add_argument(
        "--dynamo_optimize_ddp",
        action="store_true",
        help="Enable dynamo's ddp optimizer",
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
    model_arg.add_argument(
        "--toy_bert", action="store_true", help="use toy model instead"
    )
    args = parser.parse_args()

    model_name = args.torchbench_model
    if args.toy_model:
        model_name =  "ToyModel"
    elif args.toy_bert:
        model_name = "BertLMPredictionHead"
    model, inputs = get_model(args)

    fn = partial(run_model, args, model, inputs)

    times = []
    experiment(fn, model_name, args.world_size, times)
    print("\nExperiment Results:")
    print(tabulate.tabulate(times, headers=("key", "time")))
