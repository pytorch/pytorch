import random

import torch
from torch._C._distributed_c10d import FakeWork


used_ids: set[int] = set()


def generate_unique_id() -> int:
    while True:
        new_id = random.randint(1, 10**9)
        if new_id not in used_ids:
            used_ids.add(new_id)
            return new_id


# Function to create and return FakeWork object
def create_fakework(args, return_first_arg=True):  # type: ignore[no-untyped-def]
    work = FakeWork()
    work.seq_id = generate_unique_id()
    fakework_script_obj = work.boxed()
    return (args[0], fakework_script_obj) if return_first_arg else fakework_script_obj


# Dictionary mapping collective operations to their meta functions
# All 20 ops from torch.csrc.distributed.c10d.Ops.cpp are included
# _DEPRECATED_META_FUNCTIONS = {
#     "allreduce_coalesced_": lambda *args: create_fakework(args, return_first_arg=False),
#     "allgather_coalesced_": lambda *args: create_fakework(args, return_first_arg=False),
#     "allgather_into_tensor_coalesced_": lambda *args: create_fakework(args, return_first_arg=False),
#     "reduce_scatter_tensor_coalesced_": lambda *args: create_fakework(args, return_first_arg=False),
# }
_META_FUNCTIONS = {
    "broadcast_": lambda *args: create_fakework(args),
    "allreduce_": lambda *args: create_fakework(args),
    "allgather_": lambda *args: create_fakework(args),
    "_allgather_base_": lambda *args: create_fakework(args),
    "reduce_scatter_": lambda *args: create_fakework(args),
    "_reduce_scatter_base_": lambda *args: create_fakework(args),
    "reduce_": lambda *args: create_fakework(args, return_first_arg=False),
    "gather_": lambda *args: create_fakework(args, return_first_arg=False),
    "scatter_": lambda *args: create_fakework(args),
    "alltoall_": lambda *args: create_fakework(args),
    "alltoall_base_": lambda *args: create_fakework(args, return_first_arg=False),
    "barrier": lambda *args: create_fakework(args, return_first_arg=False),
    "monitored_barrier_": lambda *args: None,
    "send": lambda *args: create_fakework(args, return_first_arg=False),
    "recv_": lambda *args: create_fakework(args, return_first_arg=False),
    "recv_any_source_": lambda *args: create_fakework(args, return_first_arg=False),
}

lib_impl = torch.library.Library("c10d", "IMPL")  # noqa: TOR901
for op, meta_func in _META_FUNCTIONS.items():
    lib_impl.impl(op, meta_func, "Meta")
