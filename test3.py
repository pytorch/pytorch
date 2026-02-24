"""Example: passing DTensor directly to debug_log_rank (no .to_local() needed)."""

import logging

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils.debug_log import debug_log_rank

log = logging.getLogger("torch.utils.debug_log")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())
log.propagate = False


def main():
    store = FakeStore()
    dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
    mesh = DeviceMesh("cpu", range(2))

    torch.manual_seed(42)
    x = torch.randn(4, 4, requires_grad=True)
    dt = DTensor.from_local(x, mesh, [Replicate()])

    # Pass DTensor directly to debug_log_rank — no .to_local() needed
    print("=== DTensor directly ===")
    y = dt * 2
    debug_log_rank(y, "after_mul", ranks=0)
    y.sum().backward()
    print()

    # Also works with plain tensors
    print("=== Plain tensor ===")
    x2 = torch.randn(4, 4, requires_grad=True)
    y2 = x2 * 2
    debug_log_rank(y2, "after_mul", ranks=0)
    y2.sum().backward()
    print()

    # Compiled with aot_eager backend (uses .to_local() since compile traces through DTensor)
    print("=== Compiled (aot_eager) ===")
    torch._dynamo.reset()

    def f(x):
        y = x * 2
        debug_log_rank(y, "compiled", ranks=0)
        return y

    local = dt.to_local().clone().detach().requires_grad_(True)
    compiled_f = torch.compile(f, backend="aot_eager", fullgraph=True)
    out = compiled_f(local)
    out.sum().backward()
    print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
