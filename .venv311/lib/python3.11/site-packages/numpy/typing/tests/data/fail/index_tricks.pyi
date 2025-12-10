import numpy as np

AR_LIKE_i: list[int]
AR_LIKE_f: list[float]

np.ndindex([1, 2, 3])  # type: ignore[call-overload]
np.unravel_index(AR_LIKE_f, (1, 2, 3))  # type: ignore[arg-type]
np.ravel_multi_index(AR_LIKE_i, (1, 2, 3), mode="bob")  # type: ignore[call-overload]
np.mgrid[1]  # type: ignore[index]
np.mgrid[...]  # type: ignore[index]
np.ogrid[1]  # type: ignore[index]
np.ogrid[...]  # type: ignore[index]
np.fill_diagonal(AR_LIKE_f, 2)  # type: ignore[arg-type]
np.diag_indices(1.0)  # type: ignore[arg-type]
