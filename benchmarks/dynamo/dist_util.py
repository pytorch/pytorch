import argparse
import functools
import importlib
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._dynamo.testing import reduce_to_scalar_loss
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy


try:
    from .torchbench import setup_torchbench_cwd
except ImportError:
    from torchbench import setup_torchbench_cwd

from transformers.models.bert.modeling_bert import BertLayer, BertLMPredictionHead
from transformers.models.t5.modeling_t5 import T5Block


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    os.environ["RANK"] = os.getenv("RANK", "0")
    os.environ["WORLD_SIZE"] = os.getenv("WORLD_SIZE", "1")
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


class CustomLinear(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(a, b))

    def forward(self, x):
        return torch.mm(x, self.weight)


class MyModule(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(a, b),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
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


def model_iter_fn(model, example_inputs, collect_outputs=False):
    outputs = model(*example_inputs)
    loss = reduce_to_scalar_loss(outputs)
    loss.backward()
    if collect_outputs:
        return outputs


def get_model(args):
    if args.torchbench_model:
        old_cwd = setup_torchbench_cwd()
        module = importlib.import_module(
            f"torchbenchmark.models.{args.torchbench_model}"
        )
        benchmark_cls = getattr(module, "Model", None)
        bm = benchmark_cls(test="train", device=args.device, batch_size=args.batch_size)
        model, inputs = bm.get_module()
    elif args.toy_model:
        model = ToyModel()
        inputs = (torch.randn(20, 10),)
    else:
        raise argparse.ArgumentError(
            args.torchbench_model, message="Must specify a model"
        )

    return model, inputs


def fsdp_checkpointing_base(model, blocks):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    def check_fn(submodule):
        return isinstance(submodule, blocks)

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )


MODEL_FSDP_WRAP = {
    "toy_model": (MyModule,),
    "hf_Bert": (BertLayer, BertLMPredictionHead),
    "hf_T5": (T5Block,),
}


def apply_fsdp(args, model, use_checkpointing=False, use_wrap_policy=True):
    wrap_policy = None
    blocks = MODEL_FSDP_WRAP[
        "toy_model" if model.__class__ is ToyModel else args.torchbench_model
    ]
    if use_wrap_policy:
        wrap_policy = ModuleWrapPolicy(blocks)

    model = FSDP(model, auto_wrap_policy=wrap_policy, use_orig_params=True)
    if use_checkpointing:
        fsdp_checkpointing_base(model, blocks)
    return model
