# This file is never automatically imported within PyTorch so it is ok to
# always warn here
import warnings

warnings.warn("The `torch.nn.utils._stateless` code is deprecated now that "
              "it is publicly available. Please use `torch.nn.utils.stateless "
              "instead.", DeprecationWarning)

# Import * wouldn't work as most things are private and thus wouldn't be imported
# here.
from torch.nn.utils.stateless import functional_call  # noqa: F401
from torch.nn.utils.stateless import _apply_func_submodules, _change_class  # noqa: F401
# This one used to look public but should actually be private. This was fixed when making the module
# public and is kept here for BC
from torch.nn.utils.stateless import _reparametrize_module as reparametrize_module  # noqa: F401
