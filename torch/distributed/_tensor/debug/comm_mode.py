from collections import defaultdict
from typing import Any, Dict

import torch
from torch.utils._python_dispatch import TorchDispatchMode


native = torch.ops._c10d_functional
legacy = torch.ops.c10d_functional

NATIVE_TO_LEGACY_MAPPING = {
    native.all_gather_into_tensor: legacy.all_gather_into_tensor,
    native.all_gather_into_tensor_coalesced: legacy.all_gather_into_tensor_coalesced,
    native.all_reduce: legacy.all_reduce,
    native.all_to_all_single: legacy.all_to_all_single,
    native.broadcast: legacy.broadcast,
    native.reduce_scatter_tensor: legacy.reduce_scatter_tensor,
    native.reduce_scatter_tensor_coalesced: legacy.reduce_scatter_tensor_coalesced,
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
        for native_op, legacy_op in NATIVE_TO_LEGACY_MAPPING.items():
            self.comm_registry.add(native_op)
            self.comm_registry.add(legacy_op)

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
        if func_packet in self.comm_registry:
            if func_packet in NATIVE_TO_LEGACY_MAPPING:
                func_packet = NATIVE_TO_LEGACY_MAPPING[func_packet]
            self.comm_counts[func_packet] += 1

        return out
