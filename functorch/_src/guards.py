import dataclasses
from typing import Optional

@dataclasses.dataclass
class TensorReference(object):
    """
    TensorReference objects are entirely optional. They are created to give us hints
    into where the symbolic shape came from.

    ref_id: The id of the tensor
    kind: A string tracking where in the tensor this value came from ("size","stride", etc)
    idx: An index in the structure

    NOTE - A symbolic shape coming from tensor at id 12345's shape dim 2, would be
    TensorReference(ref_id=12345, kind="size", idx=2)
    """

    ref_id: Optional[int] = None
    kind: Optional[str] = None
    idx: Optional[int] = None
    # Note - this is untyped because of TypeError: '_SpecialForm' object does not support item assignment
    # But it is a Optional[Union["sympy.Expr", int]]
    sym_expr: Optional[object] = None  # Populated after association
    tensor_idx: Optional[int] = None

    def __hash__(self):
        return hash((self.ref_id, self.kind, self.idx))