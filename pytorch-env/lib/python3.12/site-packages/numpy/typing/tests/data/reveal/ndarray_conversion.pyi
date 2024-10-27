import sys
from typing import Any

import numpy as np
import numpy.typing as npt

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

nd: npt.NDArray[np.int_]

# item
assert_type(nd.item(), int)
assert_type(nd.item(1), int)
assert_type(nd.item(0, 1), int)
assert_type(nd.item((0, 1)), int)

# tolist
assert_type(nd.tolist(), Any)

# itemset does not return a value
# tostring is pretty simple
# tobytes is pretty simple
# tofile does not return a value
# dump does not return a value
# dumps is pretty simple

# astype
assert_type(nd.astype("float"), npt.NDArray[Any])
assert_type(nd.astype(float), npt.NDArray[Any])
assert_type(nd.astype(np.float64), npt.NDArray[np.float64])
assert_type(nd.astype(np.float64, "K"), npt.NDArray[np.float64])
assert_type(nd.astype(np.float64, "K", "unsafe"), npt.NDArray[np.float64])
assert_type(nd.astype(np.float64, "K", "unsafe", True), npt.NDArray[np.float64])
assert_type(nd.astype(np.float64, "K", "unsafe", True, True), npt.NDArray[np.float64])

assert_type(np.astype(nd, np.float64), npt.NDArray[np.float64])

# byteswap
assert_type(nd.byteswap(), npt.NDArray[np.int_])
assert_type(nd.byteswap(True), npt.NDArray[np.int_])

# copy
assert_type(nd.copy(), npt.NDArray[np.int_])
assert_type(nd.copy("C"), npt.NDArray[np.int_])

assert_type(nd.view(), npt.NDArray[np.int_])
assert_type(nd.view(np.float64), npt.NDArray[np.float64])
assert_type(nd.view(float), npt.NDArray[Any])
assert_type(nd.view(np.float64, np.matrix), np.matrix[Any, Any])

# getfield
assert_type(nd.getfield("float"), npt.NDArray[Any])
assert_type(nd.getfield(float), npt.NDArray[Any])
assert_type(nd.getfield(np.float64), npt.NDArray[np.float64])
assert_type(nd.getfield(np.float64, 8), npt.NDArray[np.float64])

# setflags does not return a value
# fill does not return a value
