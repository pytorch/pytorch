import numpy as np
from numpy._typing import ArrayLike

class A: ...

x1: ArrayLike = (i for i in range(10))  # type: ignore[assignment]
x2: ArrayLike = A()  # type: ignore[assignment]
x3: ArrayLike = {1: "foo", 2: "bar"}  # type: ignore[assignment]

scalar = np.int64(1)
scalar.__array__(dtype=np.float64)  # type: ignore[call-overload]
array = np.array([1])
array.__array__(dtype=np.float64)  # type: ignore[call-overload]

array.setfield(np.eye(1), np.int32, (0, 1))  # type: ignore[arg-type]
