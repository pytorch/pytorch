import numpy as np
from numpy._typing import _96Bit, _128Bit

from typing import assert_type

assert_type(np.float96(), np.floating[_96Bit])
assert_type(np.float128(), np.floating[_128Bit])
assert_type(np.complex192(), np.complexfloating[_96Bit, _96Bit])
assert_type(np.complex256(), np.complexfloating[_128Bit, _128Bit])
