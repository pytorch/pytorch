from dataclasses import dataclass
from typing import Annotated, Optional

from . import V8
# better way of setting the attributes of V9
from .V8 import *

from ..schema_upgrader import upgraded_from

SCHEMA_VERSION = (9, 0)
TREESPEC_VERSION = 1


@upgraded_from(V8.ModuleCallSignature)
@dataclass
class ModuleCallSignature:
    # unchanged fields
    inputs: Annotated[list[V8.Argument], 10]
    outputs: Annotated[list[V8.Argument], 20]
    in_spec: Annotated[str, 30]
    out_spec: Annotated[str, 40]

    # change the type from list[str]] to tuple[str]
    forward_arg_names: Annotated[Optional[tuple[str]], 50] = None

    @classmethod
    def upgrade(cls, v8_signature: V8.ModuleCallSignature):
        new_forward_arg_names = None
        if v8_signature.forward_arg_names is not None:
            assert isinstance(v8_signature.forward_arg_names, list)
            new_forward_arg_names = tuple(v8_signature.forward_arg_names)
            assert isinstance(new_forward_arg_names, tuple)

        return cls(
            inputs = v8_signature.inputs,
            outputs = v8_signature.outputs,
            in_spec = v8_signature.in_spec,
            out_spec = v8_signature.out_spec,
            forward_arg_names=new_forward_arg_names,
        )

__all__ = V8.__all__
