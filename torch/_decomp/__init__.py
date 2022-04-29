import torch
import torch._ops
from typing import Callable, Union, Dict, Sequence
from torch.utils._pytree import tree_map
from collections import defaultdict

__all__ = ["decomposition_table", "register_decomposition", "get_decompositions"]

# TODO: relax key type here; torch registrations should be possible to; but
# right now this type is accurate
decomposition_table: Dict[torch._ops.OpOverload, Callable] = {}


def register_decomposition(aten_op, registry=None):
    """
    A decorator to register a function as a decomposition to the Python
    decomposition table.  Use it like this::

        @register_decomposition(torch.ops.aten.clamp_min)
        def clamp_min(x):
            return torch.clamp(self, min=min)

    If you are writing a new decomposition, consider contributing it
    directly to PyTorch in torch._decomp.decompositions.

    This API is experimental; we are almost certainly going to extend
    the API when we make decompositions eligible for use in transforms (e.g.,
    autograd) and not just backend tracing, where we then need to know if a
    decomposition can be used to simulate a transform.
    """
    def decomposition_decorator(f):
        nonlocal registry
        if registry is None:
            registry = decomposition_table

        def add_op_to_table(aten_op):
            # Converts aten.foo to aten.foo.default
            # Done so I can be lazy and not write default on all of these ops
            if not isinstance(aten_op, torch._ops.OpOverload):
                op_overload = aten_op.default
            else:
                op_overload = aten_op
            registry[op_overload] = f

        # To handle allowing multiple aten_ops at once
        tree_map(add_op_to_table, aten_op)
        return f

    return decomposition_decorator


def get_decompositions(
    aten_ops: Sequence[Union[torch._ops.OpOverload, torch._ops.OpOverloadPacket]]
) -> Dict[torch._ops.OpOverload, Callable]:
    """
    Retrieve a dictionary of decompositions corresponding to the list of
    operator overloads and overload packets passed as input.  Overload
    packets will include all decomposed overloads in the packet.  If there is
    no decomposition for a requested operator, it is silently ignored.

    This API is experimental; we are almost certainly going to give an alternate,
    more recommended formulation, where a user provides the set of operators
    they know how to implement, and we provide decompositions for everything
    not in this set.
    """
    packets_to_overloads = defaultdict(list)
    for opo in decomposition_table:
        packets_to_overloads[opo.overloadpacket].append(opo)
    decompositions = {}
    for op in aten_ops:
        if isinstance(op, torch._ops.OpOverloadPacket) and op in packets_to_overloads:
            for op_overload in packets_to_overloads[op]:
                decompositions[op_overload] = decomposition_table[op_overload]
        elif isinstance(op, torch._ops.OpOverload) and op in decomposition_table:
            decompositions[op] = decomposition_table[op]
    return decompositions

# populate the table
import torch._decomp.decompositions
