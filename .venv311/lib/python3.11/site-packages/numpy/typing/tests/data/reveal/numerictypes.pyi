from typing import Literal, assert_type

import numpy as np

assert_type(np.ScalarType[0], type[int])
assert_type(np.ScalarType[3], type[bool])
assert_type(np.ScalarType[8], type[np.complex64])
assert_type(np.ScalarType[9], type[np.complex128])
assert_type(np.ScalarType[-1], type[np.void])
assert_type(np.bool_(object()), np.bool)

assert_type(np.typecodes["Character"], Literal["c"])
assert_type(np.typecodes["Complex"], Literal["FDG"])
assert_type(np.typecodes["All"], Literal["?bhilqnpBHILQNPefdgFDGSUVOMm"])

assert_type(np.sctypeDict["uint8"], type[np.generic])
