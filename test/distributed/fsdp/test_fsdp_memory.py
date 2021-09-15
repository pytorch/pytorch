# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with GPU memory usage. """

import contextlib

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim

from torch.utils.checkpoint import checkpoint
from torch.distributed._fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._fsdp.utils import get_process_group_cached
from torch.testing._internal.common_fsdp import (
    dist_init,
    dump_all_tensors,
    teardown,
    temp_files_ctx,
)


def to_fsdp(module, fsdp_config):
    return FSDP(module, process_group=get_process_group_cached(), **fsdp_config)


def get_cur_mem(rank, result, prefix):
    """Collect memory allocated values in a result dict in MB"""
    result[prefix] = round(torch.cuda.memory_allocated() / 1024 / 1024)


class Model(nn.Module):
    def __init__(self, hidden_dim, with_checkpoint=False, with_fsdp=False):
        super().__init__()
        # TODO (Min): for both fast and memory efficient conv kernels, we should be using
        #     AMP/fp16 + channel_last input format. Otherwise, cudnn internally does conversion
        #     to channel_last when it is fp16 weights. Leave this knowledge here and perhaps
        #     future test can cover it.
        batch_norm_fsdp_config = {
            "mixed_precision": False,  # Keep the weights in FP32.
            "flatten_parameters": False,  # Do not flatten.
            # Reshard==False is good for performance. When FSDP(checkpoint(FSDP(bn))) is used, this
            # **must** be False because BN's FSDP wrapper's pre-backward callback isn't called
            # within the checkpoint's outer backward when multiple forward passes are used.
            "reshard_after_forward": False,
            # No bucketing or small bucketing should be enough for BNs.
            "bucket_cap_mb": 0,
            # Setting this for SyncBatchNorm. This may have a performance impact. If
            # SyncBatchNorm is used, this can be enabled by passing in the `fsdp_config` argument.
            "force_input_to_fp32": False,
        }
        if with_fsdp:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3),
                to_fsdp(nn.BatchNorm2d(64), batch_norm_fsdp_config),
                nn.ReLU(inplace=True)
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        if with_fsdp:
            self.blocks = nn.Sequential(
                nn.Conv2d(64, hidden_dim, kernel_size=5, padding=2),
                to_fsdp(nn.BatchNorm2d(hidden_dim), batch_norm_fsdp_config),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                to_fsdp(nn.BatchNorm2d(hidden_dim), batch_norm_fsdp_config),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                to_fsdp(nn.BatchNorm2d(hidden_dim), batch_norm_fsdp_config),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),
            )
        else:
            self.blocks = nn.Sequential(
                nn.Conv2d(64, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),
            )

        self.head = nn.Linear(hidden_dim, 10)
        self.with_checkpoint = with_checkpoint

    def forward(self, x):
        if self.with_checkpoint:
            return self.head(checkpoint(self.blocks, self.stem(x)))
        else:
            return self.head(self.blocks(self.stem(x)))


def create_model(with_fsdp, with_checkpoint, model_hidden_dim, fsdp_config):
    model = Model(model_hidden_dim, with_checkpoint, with_fsdp)
    if with_fsdp:
        model.stem = to_fsdp(model.stem, fsdp_config)
        model.blocks = to_fsdp(model.blocks, fsdp_config)
        model.head = to_fsdp(model.head, fsdp_config)
    return model


def _distributed_worker(
    gpu_id, world_size, with_fsdp, with_checkpoint, filename, filename_rpc, expected, model_hidden_dim, fsdp_config
):
    torch.cuda.set_device(gpu_id)

    rank = gpu_id
    result = dist_init(rank, world_size, filename, filename_rpc)
    assert result, "Dist init failed"

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    # Note that FSDP auto-cast the input in AMP mode. So we don't need to call half() here.
    batch = torch.randn(size=(2, 3, 224, 224)).cuda()

    model = create_model(with_fsdp, with_checkpoint, model_hidden_dim, fsdp_config)
    model = model.cuda()
    if with_fsdp:
        model = to_fsdp(model, fsdp_config)
    else:
        model = DistributedDataParallel(model, device_ids=[gpu_id], bucket_cap_mb=500)

    # We enable momentum so that after the first iteration, the optimizer state is added
    # to the total memory used.
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    # Set AMP context if needed.
    context = contextlib.suppress()
    if "mixed_precision" in fsdp_config and fsdp_config["mixed_precision"]:
        context = torch.cuda.amp.autocast(enabled=True)

    # We have observed that sometimes after 3rd iteration, 4th one can fail (not on this
    # test but on much bigger scale tests). We run 4 iterations here just in case it happens.
    iterations = 4

    results = {}  # results of memory stats
    for iteration in range(iterations):
        get_cur_mem(gpu_id, results, f"iter {iteration}: start")

        with context:
            out = model(batch)
            get_cur_mem(gpu_id, results, f"iter {iteration}: after fwd")

            out = sum(o.sum() for o in out[0])
            fake_loss = criterion(out, torch.tensor(0.0).cuda())
            get_cur_mem(gpu_id, results, f"iter {iteration}: after loss")

        fake_loss.backward()
        get_cur_mem(gpu_id, results, f"iter {iteration}: after bwd")

        optimizer.step()
        get_cur_mem(gpu_id, results, f"iter {iteration}: after step")

        # It is important to use `set_to_none` below, not optimizer.zero_grad() to reclaim memory.
        model.zero_grad(set_to_none=True)
        get_cur_mem(gpu_id, results, f"iter {iteration}: done")

    dump_all_tensors(gpu_id)
    print(results)

    def cmp(results, expected):
        ret = ""
        assert results.keys() == expected.keys(), f"{list(results.keys())} vs. {list(expected.keys())}"
        for k, v in results.items():
            exp = expected[k]
            if abs(exp - v) > 1:  # allow 1MB rounding differences
                ret += f"{k}: got {v}, expected {exp}\n"
        return ret

    output = cmp(results, expected)
    assert not output, output

    teardown()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="multiple GPUs required"
)
@pytest.mark.parametrize("ckpt", ["no_ckpt", "ckpt"])
@pytest.mark.parametrize("fsdp", ["ddp", "fsdp", "fsdp_amp_default", "fsdp_amp_compute_dtype32"])
def test_fsdp_memory(fsdp, ckpt):
    expected = {
        ("ddp", "no_ckpt"): {
            "iter 0: start": 9,
            "iter 0: after fwd": 346,
            "iter 0: after loss": 346,
            "iter 0: after bwd": 14,
            "iter 0: after step": 17,
            "iter 0: done": 13,
            "iter 1: start": 13,
            "iter 1: after fwd": 350,
            "iter 1: after loss": 350,
            "iter 1: after bwd": 17,
            "iter 1: after step": 17,
            "iter 1: done": 13,
            "iter 2: start": 13,
            "iter 2: after fwd": 350,
            "iter 2: after loss": 350,
            "iter 2: after bwd": 17,
            "iter 2: after step": 17,
            "iter 2: done": 13,
            "iter 3: start": 13,
            "iter 3: after fwd": 350,
            "iter 3: after loss": 350,
            "iter 3: after bwd": 17,
            "iter 3: after step": 17,
            "iter 3: done": 13,
        },
        ("fsdp", "no_ckpt"): {
            "iter 0: start": 3,
            "iter 0: after fwd": 340,
            "iter 0: after loss": 340,
            "iter 0: after bwd": 16,
            "iter 0: after step": 18,
            "iter 0: done": 5,
            "iter 1: start": 5,
            "iter 1: after fwd": 342,
            "iter 1: after loss": 342,
            "iter 1: after bwd": 18,
            "iter 1: after step": 18,
            "iter 1: done": 5,
            "iter 2: start": 5,
            "iter 2: after fwd": 342,
            "iter 2: after loss": 342,
            "iter 2: after bwd": 18,
            "iter 2: after step": 18,
            "iter 2: done": 5,
            "iter 3: start": 5,
            "iter 3: after fwd": 342,
            "iter 3: after loss": 342,
            "iter 3: after bwd": 18,
            "iter 3: after step": 18,
            "iter 3: done": 5,
        },
        ("fsdp_amp_default", "no_ckpt"): {
            "iter 0: start": 28,
            "iter 0: after fwd": 630,
            "iter 0: after loss": 630,
            "iter 0: after bwd": 67,
            "iter 0: after step": 93,
            "iter 0: done": 54,
            "iter 1: start": 54,
            "iter 1: after fwd": 657,
            "iter 1: after loss": 657,
            "iter 1: after bwd": 93,
            "iter 1: after step": 93,
            "iter 1: done": 54,
            "iter 2: start": 54,
            "iter 2: after fwd": 657,
            "iter 2: after loss": 657,
            "iter 2: after bwd": 93,
            "iter 2: after step": 93,
            "iter 2: done": 54,
            "iter 3: start": 54,
            "iter 3: after fwd": 657,
            "iter 3: after loss": 657,
            "iter 3: after bwd": 93,
            "iter 3: after step": 93,
            "iter 3: done": 54,
        },
        ("fsdp_amp_compute_dtype32", "no_ckpt"): {
            "iter 0: start": 28,
            "iter 0: after fwd": 657,
            "iter 0: after loss": 657,
            "iter 0: after bwd": 67,
            "iter 0: after step": 93,
            "iter 0: done": 54,
            "iter 1: start": 54,
            "iter 1: after fwd": 684,
            "iter 1: after loss": 684,
            "iter 1: after bwd": 93,
            "iter 1: after step": 93,
            "iter 1: done": 54,
            "iter 2: start": 54,
            "iter 2: after fwd": 684,
            "iter 2: after loss": 684,
            "iter 2: after bwd": 93,
            "iter 2: after step": 93,
            "iter 2: done": 54,
            "iter 3: start": 54,
            "iter 3: after fwd": 684,
            "iter 3: after loss": 684,
            "iter 3: after bwd": 93,
            "iter 3: after step": 93,
            "iter 3: done": 54,
        },
        ("ddp", "ckpt"): {
            "iter 0: start": 9,
            "iter 0: after fwd": 57,
            "iter 0: after loss": 57,
            "iter 0: after bwd": 14,
            "iter 0: after step": 17,
            "iter 0: done": 13,
            "iter 1: start": 13,
            "iter 1: after fwd": 61,
            "iter 1: after loss": 61,
            "iter 1: after bwd": 17,
            "iter 1: after step": 17,
            "iter 1: done": 13,
            "iter 2: start": 13,
            "iter 2: after fwd": 61,
            "iter 2: after loss": 61,
            "iter 2: after bwd": 17,
            "iter 2: after step": 17,
            "iter 2: done": 13,
            "iter 3: start": 13,
            "iter 3: after fwd": 61,
            "iter 3: after loss": 61,
            "iter 3: after bwd": 17,
            "iter 3: after step": 17,
            "iter 3: done": 13,
        },
        ("fsdp", "ckpt"): {
            "iter 0: start": 3,
            "iter 0: after fwd": 51,
            "iter 0: after loss": 51,
            "iter 0: after bwd": 16,
            "iter 0: after step": 18,
            "iter 0: done": 5,
            "iter 1: start": 5,
            "iter 1: after fwd": 53,
            "iter 1: after loss": 53,
            "iter 1: after bwd": 18,
            "iter 1: after step": 18,
            "iter 1: done": 5,
            "iter 2: start": 5,
            "iter 2: after fwd": 53,
            "iter 2: after loss": 53,
            "iter 2: after bwd": 18,
            "iter 2: after step": 18,
            "iter 2: done": 5,
            "iter 3: start": 5,
            "iter 3: after fwd": 53,
            "iter 3: after loss": 53,
            "iter 3: after bwd": 18,
            "iter 3: after step": 18,
            "iter 3: done": 5,
        },
        ("fsdp_amp_default", "ckpt"): {
            "iter 0: start": 28,
            "iter 0: after fwd": 52,
            "iter 0: after loss": 52,
            "iter 0: after bwd": 67,
            "iter 0: after step": 93,
            "iter 0: done": 54,
            "iter 1: start": 54,
            "iter 1: after fwd": 79,
            "iter 1: after loss": 79,
            "iter 1: after bwd": 93,
            "iter 1: after step": 93,
            "iter 1: done": 54,
            "iter 2: start": 54,
            "iter 2: after fwd": 79,
            "iter 2: after loss": 79,
            "iter 2: after bwd": 93,
            "iter 2: after step": 93,
            "iter 2: done": 54,
            "iter 3: start": 54,
            "iter 3: after fwd": 79,
            "iter 3: after loss": 79,
            "iter 3: after bwd": 93,
            "iter 3: after step": 93,
            "iter 3: done": 54,
        },
        ("fsdp_amp_compute_dtype32", "ckpt"): {
            "iter 0: start": 28,
            "iter 0: after fwd": 52,
            "iter 0: after loss": 52,
            "iter 0: after bwd": 67,
            "iter 0: after step": 93,
            "iter 0: done": 54,
            "iter 1: start": 54,
            "iter 1: after fwd": 79,
            "iter 1: after loss": 79,
            "iter 1: after bwd": 93,
            "iter 1: after step": 93,
            "iter 1: done": 54,
            "iter 2: start": 54,
            "iter 2: after fwd": 79,
            "iter 2: after loss": 79,
            "iter 2: after bwd": 93,
            "iter 2: after step": 93,
            "iter 2: done": 54,
            "iter 3: start": 54,
            "iter 3: after fwd": 79,
            "iter 3: after loss": 79,
            "iter 3: after bwd": 93,
            "iter 3: after step": 93,
            "iter 3: done": 54,
        },
    }[(fsdp, ckpt)]

    # Compute the FSDP config.
    fsdp_config = {}

    # Set mixed precision.
    if "amp" in fsdp:
        fsdp_config["mixed_precision"] = True

    # When compute_dtype is FP32, make sure we use clear_autocast_cache.
    # Setting fp32_reduce_scatter and verbose for more code coverage.
    if "compute_dtype32" in fsdp:
        fsdp_config["compute_dtype"] = torch.float32
        fsdp_config["fp32_reduce_scatter"] = True
        fsdp_config["clear_autocast_cache"] = True
        fsdp_config["verbose"] = True

    # Using bigger hidden dimension for AMP to increase the model size
    # so that bug in handling params will show up but we don't do that
    # in the base case to keep the test fast.
    #   - hidden_dim 128: model size ~4MB
    #   - hidden_dim 512: model size ~55MB
    #   - hidden_dim 1024: model size ~200MB (seems to be too big for CI tests though)
    model_hidden_dim = 128
    if "amp" in fsdp:
        model_hidden_dim = 512

    # Get the fsdp and checkpoint flags.
    with_fsdp = "fsdp" in fsdp
    with_ckpt = ckpt == "ckpt"

    world_size = 2
    with temp_files_ctx(num=2) as temp_files:
        mp.spawn(
            _distributed_worker,
            (world_size, with_fsdp, with_ckpt, temp_files[0], temp_files[1], expected, model_hidden_dim, fsdp_config),
            nprocs=world_size,
        )
