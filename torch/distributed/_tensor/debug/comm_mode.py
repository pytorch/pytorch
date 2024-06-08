# mypy: allow-untyped-defs
from collections import defaultdict
from typing import Any, Dict

import torch
from torch.distributed._tensor.api import DTensor
from torch.utils._python_dispatch import TorchDispatchMode


funcol_native = torch.ops._c10d_functional
funcol_py = torch.ops.c10d_functional
funcol_autograd = torch.ops._c10d_functional_autograd
c10d_ops = torch.ops.c10d

NATIVE_TO_PY_MAPPING = {
    funcol_native.all_gather_into_tensor: funcol_py.all_gather_into_tensor,
    funcol_native.all_gather_into_tensor_coalesced: funcol_py.all_gather_into_tensor_coalesced,
    funcol_native.all_reduce: funcol_py.all_reduce,
    funcol_native.all_reduce_coalesced: funcol_py.all_reduce_coalesced,
    funcol_native.all_to_all_single: funcol_py.all_to_all_single,
    funcol_native.broadcast: funcol_py.broadcast,
    funcol_native.reduce_scatter_tensor: funcol_py.reduce_scatter_tensor,
    funcol_native.reduce_scatter_tensor_coalesced: funcol_py.reduce_scatter_tensor_coalesced,
    # functional ops
    funcol_autograd.all_to_all_single: funcol_py.all_to_all_single,
}

c10d_collective_ops = {
    c10d_ops._allgather_base_,
    c10d_ops._reduce_scatter_base_,
    c10d_ops.allgather_,
    c10d_ops.allgather_coalesced_,
    c10d_ops.allgather_into_tensor_coalesced_,
    c10d_ops.allreduce_,
    c10d_ops.allreduce_coalesced_,
    c10d_ops.alltoall_,
    c10d_ops.alltoall_base_,
    c10d_ops.broadcast_,
    c10d_ops.gather_,
    c10d_ops.scatter_,
    c10d_ops.reduce_,
    c10d_ops.reduce_scatter_,
    c10d_ops.reduce_scatter_tensor_coalesced_,
}


class CommDebugMode(TorchDispatchMode):
    """
    ``CommDebugMode`` is a context manager that counts the number of
    functional collectives within its context. It does this using a
    ``TorchDispatchMode``.

    NOTE: this mode only works for functional collective atm and the
    distributed_c10d collectives are not supported yet.

    Example usage

    .. code-block:: python

        mod = ...
        comm_mode = CommDebugMode()
        with comm_mode:
            mod.sum().backward()

    """

    def __init__(self):
        self.comm_counts: Dict[Any, int] = defaultdict(int)
        self.comm_registry = set()
        for native_op, py_op in NATIVE_TO_PY_MAPPING.items():
            self.comm_registry.add(native_op)
            self.comm_registry.add(py_op)

        self.comm_registry.add(torch.ops._dtensor.shard_dim_alltoall)

    def get_total_counts(self) -> int:
        return sum(self.comm_counts.values())

    def get_comm_counts(self) -> Dict[Any, int]:
        """Returns the communication counts as a dictionary.

        Returns:
            Dict[Any, int]: The communication counts as a dictionary.
        """
        return self.comm_counts

    def __enter__(self):
        self.comm_counts.clear()
        super().__enter__()
        return self

    def __exit__(self, *args):
        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # When running this mode with DTensor, ordinarily all modes will
        # run **before** subclasses get a chance to run.
        # Returning NotImplemented here gives us a chance to let DTensor
        # run and desugar into comms ops, before CommDebugMode sees them.
        if any(t == DTensor for t in types):
            return NotImplemented
        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        func_packet = func._overloadpacket
        # We have many tests that use CommDebugMode to verify the occurrence of
        # collectives. These tests do so by querying comm_counts with legacy
        # funcol ops as key. For the purpose of native funcol migration, we
        # need these tests to work for both legacy and native funcol. To avoid
        # the need to modify all tests to accommodate the two implementations,
        # we make CommDebugMode translate native funcol ops into legacy funcol
        # ops until the migration finishes.

        if func_packet in self.comm_registry or func_packet in c10d_collective_ops:
            if func_packet in NATIVE_TO_PY_MAPPING:
                func_packet = NATIVE_TO_PY_MAPPING[func_packet]
            self.comm_counts[func_packet] += 1

        return out
