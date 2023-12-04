from collections import defaultdict
from typing import Any, Dict

import torch
from torch.utils._python_dispatch import TorchDispatchMode


funcol = torch.ops.c10d_functional


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
        self.comm_registry = {
            funcol.all_gather_into_tensor,
            funcol.all_gather_into_tensor_coalesced,
            funcol.all_reduce,
            funcol.all_to_all_single,
            funcol.broadcast,
            funcol.reduce_scatter_tensor,
            funcol.reduce_scatter_tensor_coalesced,
        }

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
        if func_packet in self.comm_registry:
            self.comm_counts[func_packet] += 1

        return out
