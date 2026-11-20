import argparse
import ctypes
import resource
import threading

import torch
from torch.profiler import profile, ProfilerActivity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    args = parser.parse_args()
    # A crash would otherwise spend minutes writing a core dump.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    tensors = list(torch.zeros(100_000).unbind())
    config = torch._C._profiler._ExperimentalConfig(profile_all_threads=True)
    p = profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=args.profile_memory,
        experimental_config=config,
    )
    p.start()
    if args.stop:
        p.stop()
    # Dispatches one op per tensor with the GIL released. Each op allocates a
    # temporary for the scalar, so the worker also reports memory.
    worker = threading.Thread(
        target=torch._foreach_add_, args=(tensors, 1.0), daemon=True
    )
    worker.start()
    while tensors[0].item() == 0:
        pass
    print(tensors[-1].item() == 0, flush=True)
    # Skip Python finalization so the worker is still dispatching ops when the
    # C++ atexit handlers and static destructors run.
    ctypes.PyDLL(None).exit(0)
