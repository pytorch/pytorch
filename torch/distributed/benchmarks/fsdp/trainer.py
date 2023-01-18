import argparse
import contextlib
import datetime
import gc
import os
import time
from pathlib import Path
from statistics import stdev
import numpy as np

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist

from config import pytorch_fsdp_config, train_default_config

from pkg_resources import packaging

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def setup():
    """we use torchrun for init so no params needed here"""
    dist.init_process_group("nccl")
    torch.cuda.empty_cache()


def cleanup():
    dist.destroy_process_group()


def sync_all_device():
    # setup() has already configured CUDA_VISIBLE_DEVICES such that each
    # process exclusively works on its own set of devices. So it's safe to
    # do device sync here
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def print_memory_summary(prefix, device):
    rank = int(os.getenv("RANK"))
    if rank == 0:
        peak_memory_active = torch.cuda.memory_stats().get("active_bytes.all.peak", 0)
        print(
            f"{prefix}, GPU peak memory allocation: {torch.cuda.max_memory_allocated(device) // 1e9}GB, "
            f"GPU peak memory reserved: {torch.cuda.max_memory_reserved(device) // 1e9}GB, "
            f"GPU peak memory active: {peak_memory_active // 1e9}GB"
        )
        torch.cuda.reset_peak_memory_stats(device)


def print_latency(prefix, start_event, end_event):
    rank = int(os.getenv("RANK"))
    if rank == 0:
        print(f"{prefix}: {start_event.elapsed_time(end_event) / 1000}sec")


def model_init(args, cfg):
    rank = int(os.getenv("RANK"))

    if cfg.use_meta_device_init and "GPT" in args.model_name:
        model = model_builder.build(args.model_name, device="meta")
    else:
        model = model_builder.build(args.model_name)

    if rank == 0:
        print(f"--> {args.model_name} built.")
        num_params = (sum(p.numel() for p in model.parameters())) / 1e6
        print(f"built model with {num_params}M params")
        print(f"parameters: {list(model.parameters())[0]}")

    if args.mode == "pytorch_fsdp":
        fsdp_config = pytorch_fsdp_config()
        dist_model = FSDP(
            model,
            auto_wrap_policy=model_builder.get_fsdp_wrapping_policy(),
            mixed_precision=fsdp_config.mixed_precision,
            backward_prefetch=fsdp_config.backward_prefetch,
            sharding_strategy=fsdp_config.sharding_strategy,
            limit_all_gathers=fsdp_config.limit_all_gathers,
            use_orig_params=fsdp_config.use_orig_params,
            device_id=torch.cuda.current_device(),
        )

    # apply checkpointing after FSDP, so that we could have wrapping like FSDP(Checkpoint(Model))
    if cfg.activation_checkpointing:
        model_builder.apply_checkpointing(model)

    if rank == 0:
        print(dist_model)

    return dist_model


# ------ main code loop -----------------
def distributed_train(args):
    """main process,  within each rank process"""
    cfg = train_default_config()

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    before_forward_event = torch.cuda.Event(enable_timing=True)
    after_forward_event = torch.cuda.Event(enable_timing=True)
    after_backward_event = torch.cuda.Event(enable_timing=True)
    after_step_event = torch.cuda.Event(enable_timing=True)
    after_zero_grad_event = torch.cuda.Event(enable_timing=True)

    init_start_event.record()
    dist_model = model_init(args, cfg)
    init_end_event.record()
    sync_all_device()
    dist.barrier()
    print_latency("Building model time", init_start_event, init_end_event)
    print_memory_summary("After model init", torch.cuda.current_device())

    opt = torch.optim.AdamW(
        dist_model.parameters(), lr=1e-3, weight_decay=0, amsgrad=True
    )

    print_memory_summary("After optimizer", torch.cuda.current_device())

    inputs = model_builder.get_inputs(args.batch_size, torch.cuda.current_device())

    if rank == 0: 
        print(f"inputs: {inputs}")

    # warmup
    for i in range(cfg.total_steps_to_warm_up):
        loss = model_builder.get_loss(dist_model, inputs)
        if rank == 0:
            print(f"warm up, Step {i}, loss: {loss}")
        print_memory_summary(f"Step {i} After forward", torch.cuda.current_device())
        loss.backward()
        print_memory_summary(f"Step {i} After backward", torch.cuda.current_device())
        del loss
        opt.step()
        print_memory_summary(f"Step {i} After optimizer", torch.cuda.current_device())
        opt.zero_grad()
        print_memory_summary(f"Step {i} After zero grad", torch.cuda.current_device())

    # training loop
    start_time = time.time()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
        with_flops=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.profile_folder),
    ) if cfg.run_profiler else contextlib.nullcontext() as prof:
        for i in range(cfg.total_steps_to_run):
            before_forward_event.record()
            loss = model_builder.get_loss(dist_model, inputs)
            if rank == 0:
                print(f"training, Step {i}, loss: {loss}")
            after_forward_event.record()
            loss.backward()
            after_backward_event.record()
            del loss
            gc.collect()
            opt.step()
            after_step_event.record()
            opt.zero_grad()
            after_zero_grad_event.record()
            torch.cuda.synchronize()
            fwd_list = []
            bwd_list = []
            opt_list = []
            total_latency = (
                before_forward_event.elapsed_time(after_zero_grad_event) / 1000
            )
            forward_time = before_forward_event.elapsed_time(after_forward_event) / 1000
            backward_time = (
                after_forward_event.elapsed_time(after_backward_event) / 1000
            )
            step_time = after_backward_event.elapsed_time(after_step_event) / 1000
            zero_grad_time = after_step_event.elapsed_time(after_zero_grad_event) / 1000
            fwd_list.append(forward_time * 100.0 / total_latency)
            bwd_list.append(backward_time * 100.0 / total_latency)
            opt_list.append((step_time + zero_grad_time) * 100.0 / total_latency)
            if rank == 0:
                print(
                    f"train {i}th step, total_latency: {total_latency}sec, "
                    f"forward_time: {forward_time}sec, "
                    f"backward_time: {backward_time}sec, "
                    f"step_time: {step_time}sec, "
                    f"zero_grad_time: {zero_grad_time}sec, "
                )
    sync_all_device()
    end_time = time.time()
    delays = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(
        delays, (end_time - start_time) / cfg.total_steps_to_run
    )
    tflops_per_gpu = model_builder.get_flops(args.model_name, args.batch_size) / 10**12 * np.reciprocal(np.array(delays))

    if rank == 0:
        name = f"{args.mode}_{args.model_name}_" f"ws{world_size}_bs{args.batch_size}"
        print(f"===========================+++++++++++final results++++++++++++++==============================")
        Path("delay").mkdir(parents=True, exist_ok=True)
        fout = open(f"delay/{name}.txt", "w")
        fout.write(f"{train_default_config()} \n")
        print(f"{train_default_config()}")
        dist_config = pytorch_fsdp_config() if args.mode=="pytorch_fsdp" else ""
        fout.write(f"{dist_config} \n")
        print(f"{dist_config}")
        avg_delay = sum(delays) / len(delays)
        fout.write(f"delays = {avg_delay:.2f}sec ({stdev(delays):.2f}sec)\n")
        print(f"delays = {avg_delay:.2f}sec ({stdev(delays):.2f}sec)")
        fout.write(f"tflops/gpu = {sum(tflops_per_gpu) / len(tflops_per_gpu):.2f} ({stdev(tflops_per_gpu):.2f})\n")
        print(f"tflops/gpu = {sum(tflops_per_gpu) / len(tflops_per_gpu):.2f} ({stdev(tflops_per_gpu):.2f})")
        fout.write(f"QPS = {args.batch_size / avg_delay} samples per sec\n")
        print(f"QPS = {args.batch_size / avg_delay} samples per sec")
        peak_memory_allocated = max(
            [
                torch.cuda.max_memory_allocated(i)
                for i in range(torch.cuda.device_count())
            ]
        )
        peak_memory_reserved = max(
            [
                torch.cuda.max_memory_reserved(i)
                for i in range(torch.cuda.device_count())
            ]
        )
        peak_memory_active = torch.cuda.memory_stats().get("active_bytes.all.peak", 0)
        fout.write(f"peak allocated mem = {peak_memory_allocated // 1e9}GB\n")
        print(f"peak allocated mem = {peak_memory_allocated // 1e9}GB\n")
        fout.write(f"peak active mem = {peak_memory_active // 1e9}GB\n")
        print(f"peak active mem = {peak_memory_active // 1e9}GB\n")
        fout.write(f"peak reserved mem = {peak_memory_reserved // 1e9}GB\n")
        print(f"peak reserved mem = {peak_memory_reserved // 1e9}GB\n")
        fout.write(
            f"forward percent = {sum(fwd_list) / len(fwd_list):.2f}, backward percent = {sum(bwd_list) / len(bwd_list):.2f}, optimizer percent = {sum(opt_list) / len(opt_list):.2f}"
        )
        print(
            f"forward percent = {sum(fwd_list) / len(fwd_list):.2f}, backward percent = {sum(bwd_list) / len(bwd_list):.2f}, optimizer percent = {sum(opt_list) / len(opt_list):.2f}"
        )
        fout.close()

        if cfg.run_profiler:
            Path("chrome").mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(f"chrome/{name}.json.gz")

    dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch FSDP benchmarks")
    parser.add_argument(
        "--model-name",
        type=str,
        default="t5-small",
        choices=["t5-small", "t5-base", "t5-large", "t5-xl", "t5-xxl", "t5-11b", "t5-11b", "GPTSmall", "GPTMedium", "GPTLarge", "GPTXL", "GPTXXL", " GPTXXXL", "GPT13B", "GPT175B", "GPT1T"],
        help="choose model to run, available: `t5-small`, `t5-base`, `t5-large`, `t5-xl`, `t5-xxl`, `t5-11b`, `t5-11b`, `GPTSmall`, `GPTMedium`, `GPTLarge`, `GPTXL`, `GPTXXL`, `GPTXXXL`, `GPT13B`, `GPT175B`, `GPT1T` (default: t5-small)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="pytorch_fsdp",
        choices=["pytorch_fsdp"],
        help="choose training algorithm to run, available: `pytorch_fsdp` (default: pytorch_fsdp)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    cfg = train_default_config()

    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"--> World Size = {world_size}\n")
        print(f"--> Device_count = {torch.cuda.device_count()}")
        print(f"--> running with these defaults {cfg}")

    setup()

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)

    if "t5" in args.model_name:
        import t5_builder as model_builder
    elif "GPT" in args.model_name:
        import min_gpt_builder as model_builder

    torch.backends.cuda.matmul.allow_tf32 = True

    is_bfloat_supported = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )
    assert is_bfloat_supported, "bfloat16 support is not availabel"

    distributed_train(args)

    cleanup()
