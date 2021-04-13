import scipy.special
import numpy as np

import torch

def composite_i0e(t):
    return t.abs().neg().exp() * t.i0()

a = np.array([-501.,], dtype=np.float32)
t = torch.from_numpy(a)

print(scipy.special.i0e(a))

# Works as computations happen in `double`
print(torch.special.i0e(t))

print(composite_i0e(t))
print(torch.i0(t), t.abs().neg().exp())
