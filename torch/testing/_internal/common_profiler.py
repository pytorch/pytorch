import torch
from torch.profiler import (
    kineto_available,
    profile,
    ProfilerActivity,
    supported_activities,
)


def initialize_kineto_with_cuda():
    if (
        kineto_available()
        and torch.cuda.is_available()
        and ProfilerActivity.CUDA in supported_activities()
    ):
        # Kineto's process-global profiler cannot currently upgrade from a
        # CPU-only first initialization to CUDA-capable profiling. Prime it with
        # CUDA so CPU-only tests do not poison later CUDA profiler tests.
        x = torch.ones(1, device="cuda")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
            x + x
            torch.cuda.synchronize()
        return True
    return False
