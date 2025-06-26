import itertools
import platform
from dataclasses import dataclass
from typing import Optional

import torch


torch.manual_seed(1234)


def get_arch_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name()
    else:
        # This returns x86_64 or arm64 (for aarch64)
        return platform.machine()


@dataclass
class Experiment:
    name: str
    dtype: str
    device: str
    arch: str
    test_config: str
    compilation_time: int
    tokens_per_second: Optional[int] = None
    memory_bandwidth: Optional[int] = None
    real_time_factor: Optional[float] = None


N_ITER = 10
batch_size_combinations = [1, 4]
max_new_token_combinations = [16, 256]
cache_implementation_combinations = ["hybrid", "static"]


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif "cpu" in device:
        pass
    else:
        print(f"device={device} is not yet suppported")


# Only count activated parameters and buffers.
def _get_model_size(model):
    model_size = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(child.parameters(), child.buffers())
            )

    return model_size
