import sys
from typing import Any

import numpy as np
import numpy.typing as npt

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

nd: npt.NDArray[np.int64]

# reshape
assert_type(nd.reshape(), npt.NDArray[np.int64])
assert_type(nd.reshape(4), npt.NDArray[np.int64])
assert_type(nd.reshape(2, 2), npt.NDArray[np.int64])
assert_type(nd.reshape((2, 2)), npt.NDArray[np.int64])

assert_type(nd.reshape((2, 2), order="C"), npt.NDArray[np.int64])
assert_type(nd.reshape(4, order="C"), npt.NDArray[np.int64])

# resize does not return a value

# transpose
assert_type(nd.transpose(), npt.NDArray[np.int64])
assert_type(nd.transpose(1, 0), npt.NDArray[np.int64])
assert_type(nd.transpose((1, 0)), npt.NDArray[np.int64])

# swapaxes
assert_type(nd.swapaxes(0, 1), npt.NDArray[np.int64])

# flatten
assert_type(nd.flatten(), npt.NDArray[np.int64])
assert_type(nd.flatten("C"), npt.NDArray[np.int64])

# ravel
assert_type(nd.ravel(), npt.NDArray[np.int64])
assert_type(nd.ravel("C"), npt.NDArray[np.int64])

# squeeze
assert_type(nd.squeeze(), npt.NDArray[np.int64])
assert_type(nd.squeeze(0), npt.NDArray[np.int64])
assert_type(nd.squeeze((0, 2)), npt.NDArray[np.int64])
