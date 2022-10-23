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
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

try:
    from .torchbench import setup_torchbench_cwd
except ImportError:
    from torchbench import setup_torchbench_cwd


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


# from transformers.models.t5.modeling_t5 import T5Block

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
