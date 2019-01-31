from torch._np_compat import *

import torch._np_compat

class ndarray(torch._np_compat._ndarray_base):
    def __repr__(self):
        return self._torch().__repr__()

torch._np_compat._np_compat_init()
