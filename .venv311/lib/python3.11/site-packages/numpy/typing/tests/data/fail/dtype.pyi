import numpy as np

class Test1:
    not_dtype = np.dtype(float)

class Test2:
    dtype = float

np.dtype(Test1())  # type: ignore[call-overload]
np.dtype(Test2())  # type: ignore[arg-type]

np.dtype(  # type: ignore[call-overload]
    {
        "field1": (float, 1),
        "field2": (int, 3),
    }
)
