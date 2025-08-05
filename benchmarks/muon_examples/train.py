"""
Training script adopted from https://github.com/MoonshotAI/Moonlight.
"""

import logging
import os

from train_common_utils import get_model_and_dataset, get_optimizer
from transformers import get_cosine_schedule_with_warmup

import torch
from torch.utils.data import DataLoader
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
    args = parser.parse_args()

    fqn_file = f"{args.model}_muon_param_fqns.txt"
    fqn_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), fqn_file)
    if not os.path.isfile(fqn_file):
        raise FileNotFoundError(f"Muon FQNs file: {fqn_file} not found")

    with open(fqn_file) as f:
        muon_param_fqns = [line.strip() for line in f if line.strip()]

    model, train_dataset = get_model_and_dataset(
        args.model, args.dataset, args.hidden_size
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    logger.info("num_train_steps: %s", len(train_loader))
    for fqn in muon_param_fqns:
        logger.info("muon param: %s", fqn)

    optimizer = get_optimizer(
        args.optimizer, model, lr=args.lr, muon_param_fqns=muon_param_fqns
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    epoch = 1
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * epoch,
        num_cycles=0.5,
    )
    tb_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.experiment))
    for e in range(epoch):
        for step, batch in enumerate(train_loader):
            # total number of steps: 13299
            batch = batch.to(device)
            input_ids = batch
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
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

    tb_writer.flush()
    tb_writer.close()
