import sys
import torch
from torch.backends import ContextProp, PropModule

# Write:
#
#   torch.experimental.deterministic = True
#
# to globally enforce deterministic algorithms

class ExperimentalModule(PropModule):
    def __init__(self, m, name):
        super(ExperimentalModule, self).__init__(m, name)

    deterministic = ContextProp(torch._C._get_experimental_deterministic, torch._C._set_experimental_deterministic)

# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = ExperimentalModule(sys.modules[__name__], __name__)
