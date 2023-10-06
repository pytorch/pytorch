import argparse
import math
import os
import time

import torch
import torch.nn as nn

from benchmark_dataset import BenchmarkLMDataset, collate_sentences_lm
from torch.distributed import rpc

from torch.distributed.pipeline.sync import Pipe
from torch.distributed.pipeline.sync.utils import partition_model
from torch.optim import Adam
from torch.utils.data import DataLoader


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti"]:
        if abs(num) < 1024.0:
            return f"{num:3.2f}{unit}B"
        num /= 1024.0


def init_random_seed(seed: int):
    import numpy

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)


iteration_count = 0


class EmbeddingLayer(nn.Embedding):
    def __init__(self, ntoken, ninp, initrange):
        super().__init__(ntoken, ninp)
        self.ninp = ninp
        nn.init.uniform_(self.weight, -initrange, initrange)

    def forward(self, src):
        return super().forward(src) * math.sqrt(self.ninp)


class PositionalEncodingLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerDecoderLayer(nn.TransformerEncoderLayer):
    """Though this class inherits from torch.nn.TransformerEncoderLayer,
    it functions as a decoder in this model"""

    def __init__(self, ninp, nhead, nhid, droupout):
        super().__init__(ninp, nhead, nhid, droupout)
        self.src_mask = None

    def forward(self, src):
        global iteration_count
        iteration_count += 1

        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        return super().forward(src, self.src_mask)


class LinearLayer(nn.Linear):
    def __init__(self, ninp, ntoken, initrange):
        super().__init__(ninp, ntoken)
        nn.init.zeros_(self.bias)
        nn.init.uniform_(self.weight, -initrange, initrange)


class TransformerLMSequential(nn.Sequential):
    """A small language model based on the design of GPT-2 using nn.Sequential
    for compatibility with Pipe"""

    def __init__(self, ntokens, ninp, nhead, nhid, dropout, initrange, ndecoder):
        layers = [
            EmbeddingLayer(ntokens, ninp, initrange),
            PositionalEncodingLayer(ninp, dropout),
        ]
        for _ in range(ndecoder):
            layers.append(TransformerDecoderLayer(ninp, nhead, nhid, dropout))

        layers.append(LinearLayer(ninp, ntokens, initrange))
        super().__init__(*layers)


def make_model(args, device, ntokens):
    ninp = 2048  # embedding dimension
    nhid = (
        2048  # the dimension of the feedforward network model in nn.TransformerEncoder
    )
    nhead = 32  # the number of heads in the multiheadattention models
    dropout = 0
    initrange = 0.1
    ndecoder = args.num_decoder_layers

    model = TransformerLMSequential(
        ntokens, ninp, nhead, nhid, dropout, initrange, ndecoder
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 0.01  # learning rate

    def make_adam(model):
        return Adam(model.parameters(), lr=lr)

    optimizer = make_adam

    return model, criterion, optimizer


def train(lm_dataloader, model, criterion, optimizer, vocab_size, args):
    model.train()

    vocab_size = 10000
    total_loss = 0.0
    start_time = time.time()
    word_counter = 0

    optimizer = optimizer(model)

    def get_first_device(model):
        if model.devices:
            return model.devices[0]
        else:
            return torch.cuda.current_device()

    def get_last_device(model):
        if model.devices:
            return model.devices[-1]
        else:
            return torch.cuda.current_device()

    print(
        f"Number of parameters for model: {sum(p.numel() for p in model.parameters())}"
    )
    for i, batch in enumerate(lm_dataloader):
        bi = batch["input"]
        if args.max_batch and i > args.max_batch:
            break
        optimizer.zero_grad()
        try:
            tmp = batch["input"].to(get_first_device(model))
            output = model(tmp).local_value()
        except Exception as e:
            raise RuntimeError(
                f"training failed on {torch.distributed.get_rank()}"
            ) from e

        target = batch["target"].to(get_last_device(model))
        output = output.to(target.device)

        loss = criterion(output.view(-1, vocab_size), target.view(-1))
        loss.backward()
        del target
        del output

        torch.nn.utils.clip_grad_value_(model.parameters(), 0.05)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 1
        word_counter += batch["ntokens"]
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(
                "| batch {:5d} | wps {:5.2f} | loss {:5.2f} | ppl {:8.2f}".format(
                    i, word_counter / elapsed, cur_loss, math.exp(cur_loss)
                )
            )
            word_counter = 0
            total_loss = 0
            start_time = time.time()

    print("Peak memory usage for GPUs: ", end="")
    for i in range(len(model.devices)):
        print(
            f"cuda:{i}: {sizeof_fmt(torch.cuda.memory_stats(i)['allocated_bytes.all.peak'])}, ",
            end="",
        )
    print()


def generate_balance(num_devices, num_layers):
    balance = []
    layers_assigned = 0
    for i in range(num_devices):
        x = (num_layers - layers_assigned) / (num_devices - i)
        if x.is_integer():
            balance.append(int(x))
            layers_assigned += x
        else:
            balance.append(math.ceil(x))
            layers_assigned += math.ceil(x)
    return balance


def make_model_and_data(args, device):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    vocab_size = 10000
    model, criterion, optimizer = make_model(args, device, vocab_size)
    lm_dataset = BenchmarkLMDataset()
    lm_dataloader = DataLoader(
        lm_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_sentences_lm,
    )
    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "data": lm_dataloader,
        "vocab_size": vocab_size,
    }


def bench_single_process(args):
    os.environ.update({"MASTER_ADDR": args.host})
    os.environ.update({"MASTER_PORT": "10638"})

    rpc.init_rpc(
        "worker",
        rank=0,
        world_size=1,
    )

    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_devices = min(args.num_devices, num_devices)
    assert num_devices > 0
    init_random_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    blob = make_model_and_data(args, None)
    model = blob["model"]

    balance = generate_balance(num_devices, len(model))
    model = partition_model(model, balance)
    p = Pipe(model, chunks=args.chunks, checkpoint=args.checkpoint)
    del model
    del blob["model"]

    train(
        blob["data"], p, blob["criterion"], blob["optimizer"], blob["vocab_size"], args
    )


parser = argparse.ArgumentParser(description="benchmark")
parser.add_argument("--host", "-o", type=str, default="localhost", help="hostname")
parser.add_argument(
    "--chunks", type=int, default=4, help="number of microbatches per batch"
)
parser.add_argument("--batch-size", type=int, default=8, help="size of a batch")
parser.add_argument("--max-batch", type=int, default=10, help="Max number of batches")
parser.add_argument(
    "--num-decoder-layers",
    type=int,
    default=10,
    help="Number of decoder layers in the model",
)
parser.add_argument(
    "--checkpoint",
    default="except_last",
    choices=["always", "except_last", "never"],
    help="Checkpointing strategy for pipe",
)
parser.add_argument(
    "--num-devices", type=int, default=4, help="Number of GPU devices to use"
)

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running benchmark with args: {args}")
    bench_single_process(args)
