import numpy as np

i8 = np.int64()
i4 = np.int32()
u8 = np.uint64()
b_ = np.bool()
i = int()

f8 = np.float64()

b_ >> f8  # type: ignore[operator]
i8 << f8  # type: ignore[operator]
i | f8  # type: ignore[operator]
i8 ^ f8  # type: ignore[operator]
u8 & f8  # type: ignore[operator]
~f8  # type: ignore[operator]
# TODO: Certain mixes like i4 << u8 go to float and thus should fail
