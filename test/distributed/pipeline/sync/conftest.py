# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import tempfile
import pytest

import torch
import torch.distributed as dist

@pytest.fixture(autouse=True)
def manual_seed_zero():
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def cuda_sleep():
    # Warm-up CUDA.
    torch.empty(1, device="cuda")

    # From test/test_cuda.py in PyTorch.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(1000000)
    end.record()
    end.synchronize()
    cycles_per_ms = 1000000 / start.elapsed_time(end)

    def cuda_sleep(seconds):
        torch.cuda._sleep(int(seconds * cycles_per_ms * 1000))

    return cuda_sleep


def pytest_report_header():
    return f"torch: {torch.__version__}"

@pytest.fixture
def setup_rpc(scope="session"):
    file = tempfile.NamedTemporaryFile()
    dist.rpc.init_rpc(
        name="worker0",
        rank=0,
        world_size=1,
        rpc_backend_options=dist.rpc.TensorPipeRpcBackendOptions(
            init_method="file://{}".format(file.name),
        )
    )
    yield
    dist.rpc.shutdown()

def pytest_ignore_collect(path, config):
    "Skip this directory if distributed modules are not enabled."
    return not dist.is_available()
