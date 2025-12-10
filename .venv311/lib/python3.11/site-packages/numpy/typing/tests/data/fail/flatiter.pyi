import numpy as np
import numpy._typing as npt

class Index:
    def __index__(self) -> int: ...

a: np.flatiter[npt.NDArray[np.float64]]
supports_array: npt._SupportsArray[np.dtype[np.float64]]

a.base = object()  # type: ignore[assignment, misc]
a.coords = object()  # type: ignore[assignment, misc]
a.index = object()  # type: ignore[assignment, misc]
a.copy(order='C')  # type: ignore[call-arg]

# NOTE: Contrary to `ndarray.__getitem__` its counterpart in `flatiter`
# does not accept objects with the `__array__` or `__index__` protocols;
# boolean indexing is just plain broken (gh-17175)
a[np.bool()]  # type: ignore[index]
a[Index()]  # type: ignore[call-overload]
a[supports_array]  # type: ignore[index]
