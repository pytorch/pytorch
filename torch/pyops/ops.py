from typing import Sequence

# Note [Composite ops written in Python must be valid Python]
# When the torch module is available the operations defined in this file must
#   be valid Python that computes the expected operation and supports type annotations.
#
# Additionally, when the torch module is unavailable the module must be importable
#   so its functions can be programmatically read. In the future this property
#   may be relaxed, but it is very convenient to have (an alternative would be
#   to not import this file but to read it as a string).
#
# To accomplish this the file attempts to import the torch module and, if
#   successful, defines all its types using actual torch types. If the import
#   is unsuccessful then the types are trivially defined as object subclasses.
#   Note that when importing this file the function signatures must be valid
#   Python but the function bodies do not have to be. That's why calls to
#   operations like torch.foo in a function body don't require special handling.
torch_imported = False
try:
    import torch
    torch_imported = True
except ImportError:
    pass

if torch_imported:
    TensorList = Sequence[torch.Tensor]
    Tensor = torch.Tensor

    # Represents a mutable tensor and its annotation
    # e.g. ..., Tensor(a!) out) -> Tensor(a!) in native_functions.yaml
    class MutatedTensorA(torch.Tensor):
        pass
else:
    # Defines above types trivially
    # TODO: this can probably be automated
    class TensorList(object):
        pass


    class Tensor(object):
        pass


    class MutatedTensorA(object):
        pass


from .pyops_builtins import TORCH_CHECK

# TODO: in the future we may be able to more aggressively replace the current
#   native_functions.yaml signatures. If that happens we should also consider
#   autogenerating signature variants from a canonical python implementation
#   of an operator.

# dstack
def dstack(tensors: TensorList) -> Tensor:
    TORCH_CHECK(len(tensors) > 0, "dstack expects a non-empty list of tensors")
    rep = torch.atleast_3d(tensors)
    return torch.cat(rep, 2)

# dstack.out
def dstack_DOT_out(tensors: TensorList, *, out: MutatedTensorA) -> MutatedTensorA:
    TORCH_CHECK(len(tensors) > 0, "dstack expects a non-empty list of tensors")
    rep = torch.atleast_3d(tensors)
    return torch.cat(rep, 2, out=out)