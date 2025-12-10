from typing import Any

import numpy as np
import numpy.typing as npt

AR_f8: npt.NDArray[np.float64]
AR_c16: npt.NDArray[np.complex128]
AR_m: npt.NDArray[np.timedelta64]
AR_M: npt.NDArray[np.datetime64]
AR_O: npt.NDArray[np.object_]
AR_b_list: list[npt.NDArray[np.bool]]

def fn_none_i(a: None, /) -> npt.NDArray[Any]: ...
def fn_ar_i(a: npt.NDArray[np.float64], posarg: int, /) -> npt.NDArray[Any]: ...

np.average(AR_m)  # type: ignore[arg-type]
np.select(1, [AR_f8])  # type: ignore[arg-type]
np.angle(AR_m)  # type: ignore[arg-type]
np.unwrap(AR_m)  # type: ignore[arg-type]
np.unwrap(AR_c16)  # type: ignore[arg-type]
np.trim_zeros(1)  # type: ignore[arg-type]
np.place(1, [True], 1.5)  # type: ignore[arg-type]
np.vectorize(1)  # type: ignore[arg-type]
np.place(AR_f8, slice(None), 5)  # type: ignore[arg-type]

np.piecewise(AR_f8, True, [fn_ar_i], 42)  # type: ignore[call-overload]
# TODO: enable these once mypy actually supports ParamSpec (released in 2021)
# NOTE: pyright correctly reports errors for these (`reportCallIssue`)
# np.piecewise(AR_f8, AR_b_list, [fn_none_i])  # type: ignore[call-overload]s
# np.piecewise(AR_f8, AR_b_list, [fn_ar_i])  # type: ignore[call-overload]
# np.piecewise(AR_f8, AR_b_list, [fn_ar_i], 3.14)  # type: ignore[call-overload]
# np.piecewise(AR_f8, AR_b_list, [fn_ar_i], 42, None)  # type: ignore[call-overload]
# np.piecewise(AR_f8, AR_b_list, [fn_ar_i], 42, _=None)  # type: ignore[call-overload]

np.interp(AR_f8, AR_c16, AR_f8)  # type: ignore[arg-type]
np.interp(AR_c16, AR_f8, AR_f8)  # type: ignore[arg-type]
np.interp(AR_f8, AR_f8, AR_f8, period=AR_c16)  # type: ignore[call-overload]
np.interp(AR_f8, AR_f8, AR_O)  # type: ignore[arg-type]

np.cov(AR_m)  # type: ignore[arg-type]
np.cov(AR_O)  # type: ignore[arg-type]
np.corrcoef(AR_m)  # type: ignore[arg-type]
np.corrcoef(AR_O)  # type: ignore[arg-type]
np.corrcoef(AR_f8, bias=True)  # type: ignore[call-overload]
np.corrcoef(AR_f8, ddof=2)  # type: ignore[call-overload]
np.blackman(1j)  # type: ignore[arg-type]
np.bartlett(1j)  # type: ignore[arg-type]
np.hanning(1j)  # type: ignore[arg-type]
np.hamming(1j)  # type: ignore[arg-type]
np.hamming(AR_c16)  # type: ignore[arg-type]
np.kaiser(1j, 1)  # type: ignore[arg-type]
np.sinc(AR_O)  # type: ignore[arg-type]
np.median(AR_M)  # type: ignore[arg-type]

np.percentile(AR_f8, 50j)  # type: ignore[call-overload]
np.percentile(AR_f8, 50, interpolation="bob")  # type: ignore[call-overload]
np.quantile(AR_f8, 0.5j)  # type: ignore[call-overload]
np.quantile(AR_f8, 0.5, interpolation="bob")  # type: ignore[call-overload]
np.meshgrid(AR_f8, AR_f8, indexing="bob")  # type: ignore[call-overload]
np.delete(AR_f8, AR_f8)  # type: ignore[arg-type]
np.insert(AR_f8, AR_f8, 1.5)  # type: ignore[arg-type]
np.digitize(AR_f8, 1j)  # type: ignore[call-overload]
