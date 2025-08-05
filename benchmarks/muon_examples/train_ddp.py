"""
Training script adopted from https://github.com/MoonshotAI/Moonlight.
"""

import logging
import os

from train_common_utils import get_model_and_dataset, get_optimizer
from transformers import get_cosine_schedule_with_warmup

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="openwebtext-100k")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--log_dir", type=str, default="/home/czhuge/muon_exp_logs")
    parser.add_argument("--experiment", type=str, default="pytorch_muon")
    parser.add_argument(
        "--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0))
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_profiler", type=bool, default=False)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    is_main = args.local_rank == 0

    fqn_file = f"{args.model}_muon_param_fqns.txt"
    fqn_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), fqn_file)
    if not os.path.isfile(fqn_file):
        raise FileNotFoundError(f"Muon FQNs file: {fqn_file} not found")

    with open(fqn_file) as f:
        muon_param_fqns = [line.strip() for line in f if line.strip()]

    model, train_dataset = get_model_and_dataset(
        args.model, args.dataset, args.hidden_size
    )

    sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True
    )

    optimizer = get_optimizer(
        args.optimizer, model, lr=args.lr, muon_param_fqns=muon_param_fqns
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    model.train()
    epoch = 1
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * epoch,
        num_cycles=0.5,
    )
    if is_main:
        logger.info("num_train_steps: %s", len(train_loader))
        for fqn in muon_param_fqns:
            logger.info("muon param: %s", fqn)
        log_dir = os.path.join(args.log_dir, args.experiment)
        os.makedirs(log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=log_dir)
        if args.use_profiler:
            profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=0, warmup=0, active=10, repeat=0),
                on_trace_ready=tensorboard_trace_handler(log_dir),
                record_shapes=True,
                profile_memory=True,
            )
            profiler.start()

    for e in range(epoch):
        sampler.set_epoch(e)
        for step, batch in enumerate(train_loader):
            # total number of steps: 831

            batch = batch.to(device, non_blocking=True)
            loss = model.module(input_ids=batch, labels=batch).loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if is_main:
                tb_writer.add_scalar("train/loss", loss.item(), global_step=step)
                tb_writer.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"], global_step=step
                )
                logger.info(
                    "Epoch: %s Step: %s LR: %.6f Training loss: %.6f",
                    e,
                    step,
                    optimizer.param_groups[0]["lr"],
                    loss.item(),
                )
                if args.use_profiler:
                    profiler.step()
                    if step == 10:
                        profiler.stop()

    if is_main:
        tb_writer.flush()
        tb_writer.close()

    dist.destroy_process_group()
