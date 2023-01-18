import inspect
from collections import defaultdict
from functools import wraps
from itertools import chain
from typing import Callable, Dict, Sequence, Union

import torch
import torch.library
from torch._ops import OpOverload, OpOverloadPacket
from torch.utils._pytree import tree_map

__all__ = [
    "decomposition_table",
    "pre_autograd_decomposition_table",
    "meta_table",
    "register_decomposition",
    "get_decompositions",
]


# TODO: relax key type here; torch registrations should be possible to; but
# right now this type is accurate
global_decomposition_table: Dict[str, Dict[OpOverload, Callable]] = defaultdict(dict)

decomposition_table = global_decomposition_table["post_autograd"]
pre_autograd_decomposition_table = global_decomposition_table["pre_autograd"]
meta_table = global_decomposition_table["meta"]


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


def register_decomposition(aten_op, registry=None, *, type="post_autograd"):
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

    By default, we also will register it to the Meta key of dispatcher,
    and replace the c++ Meta implementation if there is already one.
    """

    assert type in {"post_autograd", "pre_autograd", "meta"}

    def decomposition_decorator(f: Callable) -> Callable:
        sig = inspect.signature(f)
        out_annotation = f.__annotations__.get("out")
        # Hack to detect when out is a Tuple. There seems to be no pretty way of doing this
        fn = f
        if out_annotation and getattr(out_annotation, "__origin__", None) is tuple:
            out_names = sig.return_annotation._fields
            # If out is a tuple, we need to register a function that unpacks all the out
            # elements as this is what native_functions.yaml expects

            @wraps(f)
            def _fn(*args, **kwargs):
                out_kwargs = tuple(kwargs.pop(o, None) for o in out_names)
                # Either all of the out kwargs are set or none of them
                is_none = out_kwargs[0] is None
                assert all((o is None) == is_none for o in out_kwargs)
                return f(*args, **kwargs, out=None if is_none else out_kwargs)

            out_params = [
                inspect.Parameter(
                    o,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=t,
                )
                for o, t in zip(out_names, out_annotation.__args__)
            ]
            # Drop the out parameter and concatenate the new kwargs in the signature
            params = chain(
                (v for k, v in sig.parameters.items() if k != "out"), out_params
            )
            _fn.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
                parameters=params, return_annotation=sig.return_annotation  # type: ignore[arg-type]
            )
            # Drop the out parameter and concatenate the new kwargs in the annotations
            _fn.__annotations__ = {
                k: v for k, v in f.__annotations__.items() if k != "out"
            }
            for o in out_params:
                _fn.__annotations__[o.name] = o.annotation

            fn = _fn

        nonlocal registry
        if registry is None:
            registry = global_decomposition_table[type]

        def register(op):
            _add_op_to_registry(registry, op, fn)

        # To handle allowing multiple aten_ops at once
        tree_map(register, aten_op)
        return fn

    return decomposition_decorator


def get_decompositions(
    aten_ops: Sequence[Union[OpOverload, OpOverloadPacket]],
    type: str = "post_autograd",
) -> Dict[OpOverload, Callable]:
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
    assert type in {"post_autograd", "pre_autograd", "meta"}

    registry = global_decomposition_table[type]
    packets_to_overloads = defaultdict(list)
    for opo in registry:
        packets_to_overloads[opo.overloadpacket].append(opo)
    decompositions = {}
    for op in aten_ops:
        if isinstance(op, OpOverloadPacket) and op in packets_to_overloads:
            for op_overload in packets_to_overloads[op]:
                decompositions[op_overload] = registry[op_overload]
        elif isinstance(op, OpOverload) and op in registry:
            decompositions[op] = registry[op]
    return decompositions


# populate the table
import torch._decomp.decompositions
import torch._refs
