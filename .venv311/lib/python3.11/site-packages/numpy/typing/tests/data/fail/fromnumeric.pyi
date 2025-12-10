"""Tests for :mod:`numpy._core.fromnumeric`."""

import numpy as np
import numpy.typing as npt

A = np.array(True, ndmin=2, dtype=bool)
A.setflags(write=False)
AR_U: npt.NDArray[np.str_]
AR_M: npt.NDArray[np.datetime64]
AR_f4: npt.NDArray[np.float32]

a = np.bool(True)

np.take(a, None)  # type: ignore[call-overload]
np.take(a, axis=1.0)  # type: ignore[call-overload]
np.take(A, out=1)  # type: ignore[call-overload]
np.take(A, mode="bob")  # type: ignore[call-overload]

np.reshape(a, None)  # type: ignore[call-overload]
np.reshape(A, 1, order="bob")  # type: ignore[call-overload]

np.choose(a, None)  # type: ignore[call-overload]
np.choose(a, out=1.0)  # type: ignore[call-overload]
np.choose(A, mode="bob")  # type: ignore[call-overload]

np.repeat(a, None)  # type: ignore[call-overload]
np.repeat(A, 1, axis=1.0)  # type: ignore[call-overload]

np.swapaxes(A, None, 1)  # type: ignore[call-overload]
np.swapaxes(A, 1, [0])  # type: ignore[call-overload]

np.transpose(A, axes=1.0)  # type: ignore[call-overload]

np.partition(a, None)  # type: ignore[call-overload]
np.partition(a, 0, axis="bob") # type: ignore[call-overload]
np.partition(A, 0, kind="bob")  # type: ignore[call-overload]
np.partition(A, 0, order=range(5))  # type: ignore[arg-type]

np.argpartition(a, None)  # type: ignore[arg-type]
np.argpartition(a, 0, axis="bob")  # type: ignore[arg-type]
np.argpartition(A, 0, kind="bob") # type: ignore[arg-type]
np.argpartition(A, 0, order=range(5))  # type: ignore[arg-type]

np.sort(A, axis="bob")  # type: ignore[call-overload]
np.sort(A, kind="bob")  # type: ignore[call-overload]
np.sort(A, order=range(5)) # type: ignore[arg-type]

np.argsort(A, axis="bob")  # type: ignore[arg-type]
np.argsort(A, kind="bob")  # type: ignore[arg-type]
np.argsort(A, order=range(5))  # type: ignore[arg-type]

np.argmax(A, axis="bob")  # type: ignore[call-overload]
np.argmax(A, kind="bob")  # type: ignore[call-overload]
np.argmax(A, out=AR_f4)  # type: ignore[type-var]

np.argmin(A, axis="bob")  # type: ignore[call-overload]
np.argmin(A, kind="bob")  # type: ignore[call-overload]
np.argmin(A, out=AR_f4)  # type: ignore[type-var]

np.searchsorted(A[0], 0, side="bob")  # type: ignore[call-overload]
np.searchsorted(A[0], 0, sorter=1.0)  # type: ignore[call-overload]

np.resize(A, 1.0)  # type: ignore[call-overload]

np.squeeze(A, 1.0)  # type: ignore[call-overload]

np.diagonal(A, offset=None)  # type: ignore[call-overload]
np.diagonal(A, axis1="bob")  # type: ignore[call-overload]
np.diagonal(A, axis2=[])  # type: ignore[call-overload]

np.trace(A, offset=None)  # type: ignore[call-overload]
np.trace(A, axis1="bob")  # type: ignore[call-overload]
np.trace(A, axis2=[])  # type: ignore[call-overload]

np.ravel(a, order="bob")  # type: ignore[call-overload]

np.nonzero(0)  # type: ignore[arg-type]

np.compress([True], A, axis=1.0)  # type: ignore[call-overload]

np.clip(a, 1, 2, out=1)  # type: ignore[call-overload]

np.sum(a, axis=1.0)  # type: ignore[call-overload]
np.sum(a, keepdims=1.0)  # type: ignore[call-overload]
np.sum(a, initial=[1])  # type: ignore[call-overload]

np.all(a, axis=1.0)  # type: ignore[call-overload]
np.all(a, keepdims=1.0)  # type: ignore[call-overload]
np.all(a, out=1.0)  # type: ignore[call-overload]

np.any(a, axis=1.0)  # type: ignore[call-overload]
np.any(a, keepdims=1.0)  # type: ignore[call-overload]
np.any(a, out=1.0)  # type: ignore[call-overload]

np.cumsum(a, axis=1.0)  # type: ignore[call-overload]
np.cumsum(a, dtype=1.0)  # type: ignore[call-overload]
np.cumsum(a, out=1.0)  # type: ignore[call-overload]

np.ptp(a, axis=1.0)  # type: ignore[call-overload]
np.ptp(a, keepdims=1.0)  # type: ignore[call-overload]
np.ptp(a, out=1.0)  # type: ignore[call-overload]

np.amax(a, axis=1.0)  # type: ignore[call-overload]
np.amax(a, keepdims=1.0)  # type: ignore[call-overload]
np.amax(a, out=1.0)  # type: ignore[call-overload]
np.amax(a, initial=[1.0])  # type: ignore[call-overload]
np.amax(a, where=[1.0])  # type: ignore[arg-type]

np.amin(a, axis=1.0)  # type: ignore[call-overload]
np.amin(a, keepdims=1.0)  # type: ignore[call-overload]
np.amin(a, out=1.0)  # type: ignore[call-overload]
np.amin(a, initial=[1.0])  # type: ignore[call-overload]
np.amin(a, where=[1.0])  # type: ignore[arg-type]

np.prod(a, axis=1.0)  # type: ignore[call-overload]
np.prod(a, out=False)  # type: ignore[call-overload]
np.prod(a, keepdims=1.0)  # type: ignore[call-overload]
np.prod(a, initial=int)  # type: ignore[call-overload]
np.prod(a, where=1.0)  # type: ignore[call-overload]
np.prod(AR_U)  # type: ignore[arg-type]

np.cumprod(a, axis=1.0)  # type: ignore[call-overload]
np.cumprod(a, out=False)  # type: ignore[call-overload]
np.cumprod(AR_U)  # type: ignore[arg-type]

np.size(a, axis=1.0)  # type: ignore[arg-type]

np.around(a, decimals=1.0)  # type: ignore[call-overload]
np.around(a, out=type)  # type: ignore[call-overload]
np.around(AR_U)  # type: ignore[arg-type]

np.mean(a, axis=1.0)  # type: ignore[call-overload]
np.mean(a, out=False)  # type: ignore[call-overload]
np.mean(a, keepdims=1.0)  # type: ignore[call-overload]
np.mean(AR_U)  # type: ignore[arg-type]
np.mean(AR_M)  # type: ignore[arg-type]

np.std(a, axis=1.0)  # type: ignore[call-overload]
np.std(a, out=False)  # type: ignore[call-overload]
np.std(a, ddof='test')  # type: ignore[call-overload]
np.std(a, keepdims=1.0)  # type: ignore[call-overload]
np.std(AR_U)  # type: ignore[arg-type]

np.var(a, axis=1.0)  # type: ignore[call-overload]
np.var(a, out=False)  # type: ignore[call-overload]
np.var(a, ddof='test')  # type: ignore[call-overload]
np.var(a, keepdims=1.0)  # type: ignore[call-overload]
np.var(AR_U)  # type: ignore[arg-type]
