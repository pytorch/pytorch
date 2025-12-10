from typing import TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
from numpy._typing import _Shape

_ScalarT = TypeVar("_ScalarT", bound=np.generic)
MaskedArray: TypeAlias = np.ma.MaskedArray[_Shape, np.dtype[_ScalarT]]

MAR_1d_f8: np.ma.MaskedArray[tuple[int], np.dtype[np.float64]]
MAR_b: MaskedArray[np.bool]
MAR_c: MaskedArray[np.complex128]
MAR_td64: MaskedArray[np.timedelta64]

AR_b: npt.NDArray[np.bool]

MAR_1d_f8.shape = (3, 1)  # type: ignore[assignment]
MAR_1d_f8.dtype = np.bool  # type: ignore[assignment]

np.ma.min(MAR_1d_f8, axis=1.0)  # type: ignore[call-overload]
np.ma.min(MAR_1d_f8, keepdims=1.0)  # type: ignore[call-overload]
np.ma.min(MAR_1d_f8, out=1.0)  # type: ignore[call-overload]
np.ma.min(MAR_1d_f8, fill_value=lambda x: 27)  # type: ignore[call-overload]

MAR_1d_f8.min(axis=1.0)  # type: ignore[call-overload]
MAR_1d_f8.min(keepdims=1.0)  # type: ignore[call-overload]
MAR_1d_f8.min(out=1.0)  # type: ignore[call-overload]
MAR_1d_f8.min(fill_value=lambda x: 27)  # type: ignore[call-overload]

np.ma.max(MAR_1d_f8, axis=1.0)  # type: ignore[call-overload]
np.ma.max(MAR_1d_f8, keepdims=1.0)  # type: ignore[call-overload]
np.ma.max(MAR_1d_f8, out=1.0)  # type: ignore[call-overload]
np.ma.max(MAR_1d_f8, fill_value=lambda x: 27)  # type: ignore[call-overload]

MAR_1d_f8.max(axis=1.0)  # type: ignore[call-overload]
MAR_1d_f8.max(keepdims=1.0)  # type: ignore[call-overload]
MAR_1d_f8.max(out=1.0)  # type: ignore[call-overload]
MAR_1d_f8.max(fill_value=lambda x: 27)  # type: ignore[call-overload]

np.ma.ptp(MAR_1d_f8, axis=1.0)  # type: ignore[call-overload]
np.ma.ptp(MAR_1d_f8, keepdims=1.0)  # type: ignore[call-overload]
np.ma.ptp(MAR_1d_f8, out=1.0)  # type: ignore[call-overload]
np.ma.ptp(MAR_1d_f8, fill_value=lambda x: 27)  # type: ignore[call-overload]

MAR_1d_f8.ptp(axis=1.0)  # type: ignore[call-overload]
MAR_1d_f8.ptp(keepdims=1.0)  # type: ignore[call-overload]
MAR_1d_f8.ptp(out=1.0)  # type: ignore[call-overload]
MAR_1d_f8.ptp(fill_value=lambda x: 27)  # type: ignore[call-overload]

MAR_1d_f8.argmin(axis=1.0)  # type: ignore[call-overload]
MAR_1d_f8.argmin(keepdims=1.0)  # type: ignore[call-overload]
MAR_1d_f8.argmin(out=1.0)  # type: ignore[call-overload]
MAR_1d_f8.argmin(fill_value=lambda x: 27)  # type: ignore[call-overload]

np.ma.argmin(MAR_1d_f8, axis=1.0)  # type: ignore[call-overload]
np.ma.argmin(MAR_1d_f8, axis=(1,))  # type: ignore[call-overload]
np.ma.argmin(MAR_1d_f8, keepdims=1.0)  # type: ignore[call-overload]
np.ma.argmin(MAR_1d_f8, out=1.0)  # type: ignore[call-overload]
np.ma.argmin(MAR_1d_f8, fill_value=lambda x: 27)  # type: ignore[call-overload]

MAR_1d_f8.argmax(axis=1.0)  # type: ignore[call-overload]
MAR_1d_f8.argmax(keepdims=1.0)  # type: ignore[call-overload]
MAR_1d_f8.argmax(out=1.0)  # type: ignore[call-overload]
MAR_1d_f8.argmax(fill_value=lambda x: 27)  # type: ignore[call-overload]

np.ma.argmax(MAR_1d_f8, axis=1.0)  # type: ignore[call-overload]
np.ma.argmax(MAR_1d_f8, axis=(0,))  # type: ignore[call-overload]
np.ma.argmax(MAR_1d_f8, keepdims=1.0)  # type: ignore[call-overload]
np.ma.argmax(MAR_1d_f8, out=1.0)  # type: ignore[call-overload]
np.ma.argmax(MAR_1d_f8, fill_value=lambda x: 27)  # type: ignore[call-overload]

MAR_1d_f8.all(axis=1.0)  # type: ignore[call-overload]
MAR_1d_f8.all(keepdims=1.0)  # type: ignore[call-overload]
MAR_1d_f8.all(out=1.0)  # type: ignore[call-overload]

MAR_1d_f8.any(axis=1.0)  # type: ignore[call-overload]
MAR_1d_f8.any(keepdims=1.0)  # type: ignore[call-overload]
MAR_1d_f8.any(out=1.0)  # type: ignore[call-overload]

MAR_1d_f8.sort(axis=(0,1))  # type: ignore[arg-type]
MAR_1d_f8.sort(axis=None)  # type: ignore[arg-type]
MAR_1d_f8.sort(kind='cabbage')  # type: ignore[arg-type]
MAR_1d_f8.sort(order=lambda: 'cabbage')  # type: ignore[arg-type]
MAR_1d_f8.sort(endwith='cabbage')  # type: ignore[arg-type]
MAR_1d_f8.sort(fill_value=lambda: 'cabbage')  # type: ignore[arg-type]
MAR_1d_f8.sort(stable='cabbage')  # type: ignore[arg-type]
MAR_1d_f8.sort(stable=True)  # type: ignore[arg-type]

MAR_1d_f8.take(axis=1.0)  # type: ignore[call-overload]
MAR_1d_f8.take(out=1)  # type: ignore[call-overload]
MAR_1d_f8.take(mode="bob")  # type: ignore[call-overload]

np.ma.take(None)  # type: ignore[call-overload]
np.ma.take(axis=1.0)  # type: ignore[call-overload]
np.ma.take(out=1)  # type: ignore[call-overload]
np.ma.take(mode="bob")  # type: ignore[call-overload]

MAR_1d_f8.partition(['cabbage'])  # type: ignore[arg-type]
MAR_1d_f8.partition(axis=(0,1))  # type: ignore[arg-type, call-arg]
MAR_1d_f8.partition(kind='cabbage')  # type: ignore[arg-type, call-arg]
MAR_1d_f8.partition(order=lambda: 'cabbage')  # type: ignore[arg-type, call-arg]
MAR_1d_f8.partition(AR_b)  # type: ignore[arg-type]

MAR_1d_f8.argpartition(['cabbage'])  # type: ignore[arg-type]
MAR_1d_f8.argpartition(axis=(0,1))  # type: ignore[arg-type, call-arg]
MAR_1d_f8.argpartition(kind='cabbage')  # type: ignore[arg-type, call-arg]
MAR_1d_f8.argpartition(order=lambda: 'cabbage')  # type: ignore[arg-type, call-arg]
MAR_1d_f8.argpartition(AR_b)  # type: ignore[arg-type]

np.ma.ndim(lambda: 'lambda')  # type: ignore[arg-type]

np.ma.size(AR_b, axis='0')  # type: ignore[arg-type]

MAR_1d_f8 >= (lambda x: 'mango') # type: ignore[operator]
MAR_1d_f8 > (lambda x: 'mango') # type: ignore[operator]
MAR_1d_f8 <= (lambda x: 'mango') # type: ignore[operator]
MAR_1d_f8 < (lambda x: 'mango') # type: ignore[operator]

MAR_1d_f8.count(axis=0.)  # type: ignore[call-overload]

np.ma.count(MAR_1d_f8, axis=0.)  # type: ignore[call-overload]

MAR_1d_f8.put(4, 999, mode='flip')  # type: ignore[arg-type]

np.ma.put(MAR_1d_f8, 4, 999, mode='flip')  # type: ignore[arg-type]

np.ma.put([1,1,3], 0, 999)  # type: ignore[arg-type]

np.ma.compressed(lambda: 'compress me')  # type: ignore[call-overload]

np.ma.allequal(MAR_1d_f8, [1,2,3], fill_value=1.5)  # type: ignore[arg-type]

np.ma.allclose(MAR_1d_f8, [1,2,3], masked_equal=4.5)  # type: ignore[arg-type]
np.ma.allclose(MAR_1d_f8, [1,2,3], rtol='.4')  # type: ignore[arg-type]
np.ma.allclose(MAR_1d_f8, [1,2,3], atol='.5')  # type: ignore[arg-type]

MAR_1d_f8.__setmask__('mask')  # type: ignore[arg-type]

MAR_b *= 2  # type: ignore[arg-type]
MAR_c //= 2  # type: ignore[misc]
MAR_td64 **= 2  # type: ignore[misc]

MAR_1d_f8.swapaxes(axis1=1, axis2=0)  # type: ignore[call-arg]
