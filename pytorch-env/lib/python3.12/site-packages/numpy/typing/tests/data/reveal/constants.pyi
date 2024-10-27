import sys

import numpy as np

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

assert_type(np.e, float)
assert_type(np.euler_gamma, float)
assert_type(np.inf, float)
assert_type(np.nan, float)
assert_type(np.pi, float)

assert_type(np.little_endian, bool)
assert_type(np.True_, np.bool)
assert_type(np.False_, np.bool)

