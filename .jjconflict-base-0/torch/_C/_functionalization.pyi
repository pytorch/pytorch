from torch import Tensor
from torch.types import _bool

# Defined in torch/csrc/functionalization/Module.cpp

class ViewMeta:
    has_symbolic_inputs: _bool

# Returns the list of ViewMeta instances of the given functional tensor.
#
# Although we do have python bindings for their types, we won't
# expose them here, since they should not be used by users.
def get_view_meta_sequence(tensor: Tensor) -> list[ViewMeta]: ...

# Applies the ViewMeta sequence on top of the given base.
def apply_view_meta_sequence(base: Tensor, sequence: list[ViewMeta]) -> Tensor: ...
