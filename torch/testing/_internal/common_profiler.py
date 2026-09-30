import torch
from torch.profiler import (
    kineto_available,
    profile,
    ProfilerActivity,
    supported_activities,
)


def get_profiler_activities(device_type):
    activities = [ProfilerActivity.CPU]
    if device_type not in ("cpu", "meta"):
        device_activity = getattr(ProfilerActivity, device_type.upper(), None)
        if device_activity and device_activity in supported_activities():
            activities.append(device_activity)
    return activities


def initialize_kineto_with_accelerator():
    if not kineto_available():
        return False
    device = torch.accelerator.current_accelerator(check_available=True)
    # Only CUDA and XPU priming is validated; other backends stay no-op.
    if device is None or device.type not in ("cuda", "xpu"):
        return False
    activities = get_profiler_activities(device.type)
    if len(activities) < 2:
        return False
    # Kineto's process-global profiler cannot currently upgrade from a CPU-only
    # first initialization to accelerator-capable profiling. Prime it with the
    # accelerator so CPU-only tests do not poison later device profiler tests.
    x = torch.ones(1, device=device)
    with profile(activities=activities):
        x + x
        torch.accelerator.synchronize()
    return True
