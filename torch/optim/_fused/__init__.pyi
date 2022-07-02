from functools import partial
from torch import optim

Adam = partial(optim.Adam, fused=True)
