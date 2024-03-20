"""
Adapted from fsdp.py in https://github.com/pytorch/pytorch/pull/110609.
"""

"""
CUDA_VISIBLE_DEVICES=6,7 TORCH_LOGS_RANKS=0 TORCH_COMPILE_DEBUG=1 torchrun --standalone --nproc_per_node=2 test_dynamo_fsdp.py >output.txt 2>&1

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=6,7 TORCH_LOGS_RANKS=0 TORCH_COMPILE_DEBUG=1 torchrun --standalone --nproc_per_node=2 test_dynamo_fsdp.py >output.txt 2>&1

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=6 TORCH_LOGS_RANKS=0 TORCH_COMPILE_DEBUG=1 torchrun --standalone --nproc_per_node=1 test_dynamo_fsdp.py >output.txt 2>&1

CUDA_VISIBLE_DEVICES=6,7 TORCH_LOGS_RANKS=0 torchrun --standalone --nproc_per_node=2 test_dynamo_fsdp.py >output.txt 2>&1

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=6 TORCH_LOGS_RANKS=0 torchrun --standalone --nproc_per_node=1 test_dynamo_fsdp.py >output.txt 2>&1

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=6 TORCH_LOGS_RANKS=0 TORCH_COMPILE_DEBUG=1 torchrun --standalone --nproc_per_node=1 test_dynamo_fsdp.py >output.txt 2>&1

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=6 TORCH_LOGS_RANKS=0 gdb --args python3 /data/users/willfeng/miniconda3/bin/torchrun --standalone --nproc_per_node=1 test_dynamo_fsdp.py
"""
import contextlib
import logging
import os
import sys
import traceback

import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
from torch._dynamo import compiled_autograd
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed._tensor.placement_types import Shard
from torch.distributed.device_mesh import DeviceMesh
# from torchviz import make_dot

torch_log = logging.getLogger("torch")

hidden_dim = 12340

device_type = "cuda"

# ======== REMOVE WHEN READY TO MERGE ========
import argparse
import os
import subprocess
import sys
import urllib
import urllib.parse
import uuid

from typing import Optional

PERFETTO_UI_ROOT_URL = (
    "https://interncache-all.fbcdn.net/manifold/perfetto-artifacts/tree/ui/index.html"
)
MANIFOLD_FOLDER = "perfetto_internal_traces/tree/shared_trace"
DEFAULT_TTL_SEC = 28 * 24 * 60 * 60


def upload_trace_file(local_path: str, overwrite: bool = False) -> Optional[str]:
    file_name = os.path.basename(local_path)
    manifold_path = os.path.join(
        MANIFOLD_FOLDER, f"{os.getlogin()}_{str(uuid.uuid4())}_{file_name}"
    )
    cmd = [
        "manifold",
        "put",
        local_path,
        manifold_path,
        "--ttl",
        str(DEFAULT_TTL_SEC),
        "--userData",
        "false",
    ]
    ret = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    if ret.returncode == 0:
        print("Uploaded trace successfully.")
        return manifold_path
    else:
        print("[ERROR] Upload failed, maybe the trace file exists.")
        return None


def print_perfetto_ui_url(manifold_path: str) -> None:
    url = (
        PERFETTO_UI_ROOT_URL
        + "#!/?url=https://interncache-all.fbcdn.net/manifold/"
        + urllib.parse.quote_plus(manifold_path)
    )
    print(f"The trace is accessible at:\n{url}")


import socket
from datetime import datetime, timedelta

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

def start_record_memory_history() -> None:
   if not torch.cuda.is_available():
       print("CUDA unavailable. Not recording memory history")
       return

   print("Starting snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(
       max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
   )

def stop_record_memory_history() -> None:
   if not torch.cuda.is_available():
       print("CUDA unavailable. Not recording memory history")
       return

   print("Stopping snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(enabled=None)

def export_memory_snapshot(file_prefix) -> None:
   if not torch.cuda.is_available():
       print("CUDA unavailable. Not exporting memory snapshot")
       return

   try:
       print(f"Saving snapshot to local file: {file_prefix}.pickle")
       torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
   except Exception as e:
       print(f"Failed to capture memory snapshot {e}")
       return
# ======== REMOVE WHEN READY TO MERGE ========



def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    torch_log.error(
        "Uncaught exception\n%s",
        "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
    )


sys.excepthook = handle_exception


def init():
    per_param_fsdp = True

    torch.manual_seed(0)
    # Expectation:
    # - FWD: 2 all-gathers
    # - BWD: 2 all-gathers + 2 reduce-scatters
    model = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim, device=device_type),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim, device=device_type),  # FC->RELU->FC is a good test
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim, device=device_type),
    )
    if per_param_fsdp:
        torch.distributed._composable.fsdp.fully_shard(model, reshard_after_forward=True, _reshard_after_forward_root=True)
    else:
        fsdp_kwargs = {
            "use_orig_params": True,
            "auto_wrap_policy": ModuleWrapPolicy({nn.Linear}),
            # "limit_all_gathers": False,
        }
        model = FSDP(
            model,
            **fsdp_kwargs,
        )
    optim = torch.optim.SGD(model.parameters(), lr=1e-6)
    return model, optim


def printing_eager(gm, inputs):
    gm.graph.print_tabular()
    return gm.forward


local_rank = int(os.environ["LOCAL_RANK"])

def create_input():
    torch.manual_seed(0)
    inp = torch.randn((2, hidden_dim), device=device_type, requires_grad=False)
    return inp


def run(model, optim, n_iter):
    torch.manual_seed(42)
    losses = []
    for _ in range(n_iter):
        optim.zero_grad(set_to_none=True)
        inp = create_input()
        torch.storage.resize_count_and_loc = {}
        torch_log.warning("FORWARD")
        out = model(inp)
        torch_log.warning("END FORWARD")
        # torch.storage.resize_count_and_loc = {}
        loss = out.sum()
        losses.append(loss.item())
        torch.storage.resize_count_and_loc = {}
        torch_log.warning("BACKWARD")
        # torch_log.warning("OUT GRAPH\n%s", make_dot(loss))
        loss.backward()
        torch_log.warning("END BACKWARD")
        optim.step()
        torch.cuda.synchronize()
    print(f"losses: {losses}")
    return losses


def main_compiled(n_iter):
    model, optim = init()
    # per-param FSDP does lazy init using 1st run, so run it once to init using eager mode
    run(model, optim, 1)
    print("done eager 1st run!")

    dynamic = False

    def compiler_fn(gm):
        torch_log.warning("Compiling autograd?")
        return torch.compile(gm, backend="inductor", fullgraph=True, dynamic=dynamic)

    torch._dynamo.config.trace_distributed = True
    torch._inductor.config.triton.unique_kernel_names = True

    # if dist.get_rank() == 0:
    #     # HACK: delay rank 0 by X seconds, so that rank 1 will always fail first.
    #     import time
    #     time.sleep(600)
    model = torch.compile(model, backend="inductor", fullgraph=True, dynamic=dynamic)
    with compiled_autograd.enable(compiler_fn):
        res = run(model, optim, n_iter)
    print(f"res: {res}")
    return res


def main_eager(n_iter):
    model, optim = init()
    # per-param FSDP does lazy init using 1st run, so run it once to init using eager mode
    run(model, optim, 1)

    res = run(model, optim, n_iter)
    return res


def execute_and_profile(callable, profiler_trace_path, memory_snapshot_file_prefix):
    from torch.profiler import profile, ProfilerActivity
    if dist.get_rank() == 0:
        start_record_memory_history()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        ret = callable()
    if dist.get_rank() == 0:
        prof.export_chrome_trace(profiler_trace_path)
        if not os.path.exists(profiler_trace_path):
            raise Exception(f"[ERROR] The trace file doesn't exist: {profiler_trace_path}")
        manifold_path = upload_trace_file(profiler_trace_path)
        if manifold_path:
            print_perfetto_ui_url(manifold_path)
        export_memory_snapshot(memory_snapshot_file_prefix)
        stop_record_memory_history()
    return ret


if __name__ == "__main__":
    assert device_type == "cuda"
    device = f"{device_type}:{local_rank}"
    if device_type == "cuda":
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(device)
    else:
        dist.init_process_group(backend="gloo")
        # torch.set_device(device)

    n_iter = 5
    losses_compiled = execute_and_profile(
        lambda: main_compiled(n_iter=n_iter),
        "compiled_trace.json",
        "compiled_memory_snapshot",
    )
    print(f"losses_compiled: {losses_compiled}")
    losses_eager = execute_and_profile(
        lambda: main_eager(n_iter=n_iter),
        "eager_trace.json",
        "eager_memory_snapshot",
    )
    print(f"losses_eager: {losses_eager}")
    for loss_compiled, loss_eager in zip(losses_compiled, losses_eager):
        assert torch.allclose(torch.tensor(loss_compiled), torch.tensor(loss_eager), rtol=1e-3), f"{loss_compiled} vs {loss_eager}"
