# Owner(s): ["oncall: distributed"]

import pytest

import torch.distributed as dist
from torch.testing._internal.common_device_type import target_devices
from torch.testing._internal.common_utils import run_tests


"""
common backend API tests
"""


@pytest.mark.parametrize(
    ("device", "backend"), [("cuda", "nccl"), ("cpu", "gloo"), ("hpu", "hccl")]
)
def test_device_to_backend_mapping(device, backend):
    if device in target_devices:
        assert dist.get_default_backend_for_device(device) == backend


def test_invalid_device():
    with pytest.raises(ValueError):
        dist.get_default_backend_for_device("invalid_device")


@pytest.mark.parametrize("device", target_devices)
def test_create_process_group(monkeypatch, device):
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29500")

    backend = dist.get_default_backend_for_device(device)
    dist.init_process_group(backend=backend, rank=0, world_size=1, init_method="env://")
    pg = dist.distributed_c10d._get_default_group()
    backend_pg = pg._get_backend_name()
    assert backend_pg == backend
    dist.destroy_process_group()


if __name__ == "__main__":
    run_tests(use_pytest=True)
