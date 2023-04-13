import torch
from torch._ops import OpOverload, OpOverloadPacket


def _add_op_to_registry(registry, op, fn):
    """
    This is an internal API for adding an op to the decomposition table.

    If op is OpOverload, it will be added to the registry directly.
    If op is OpOverloadPacket, all the valid op_overloads in the packet will be added to the registry.
    """
    overloads = []
    if isinstance(op, OpOverload):
        overloads.append(op)
    else:
        assert isinstance(op, OpOverloadPacket)
        for ol in op.overloads():
            overloads.append(getattr(op, ol))

    for op_overload in overloads:
        if op_overload in registry:
            raise RuntimeError(f"duplicate registrations for {op_overload}")

        # TorchScript dumps a bunch of extra nonsense overloads
        # which don't have corresponding dispatcher entries, we need
        # to filter those out, e.g aten.add.float_int
        if torch._C._dispatch_has_kernel(op_overload.name()):
            registry[op_overload] = fn
