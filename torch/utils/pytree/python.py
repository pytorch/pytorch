import sys

import torch.utils._pytree


# This allows the following statements to work properly:
#
# ```python
# from torch.utils.pytree.python import tree_map
# ```
#
# ```python
# from torch.utils.pytree import python
# ```
#
sys.modules[__name__] = torch.utils._pytree

del sys, torch
