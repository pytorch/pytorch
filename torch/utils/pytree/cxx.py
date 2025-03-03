import sys

import torch.utils._cxx_pytree


# This allows the following statements to work properly:
#
# ```python
# from torch.utils.pytree.cxx import tree_map
# ```
#
# ```python
# from torch.utils.pytree import cxx
# ```
#
sys.modules[__name__] = torch.utils._cxx_pytree

del sys, torch
