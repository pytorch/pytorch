import sys
from typing import Any, TypeVar
from pathlib import Path
from collections import deque

import numpy as np
import numpy.typing as npt

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

_SCT = TypeVar("_SCT", bound=np.generic, covariant=True)

class SubClass(npt.NDArray[_SCT]): ...

i8: np.int64

A: npt.NDArray[np.float64]
B: SubClass[np.float64]
C: list[int]
D: SubClass[np.float64 | np.int64]

def func(i: int, j: int, **kwargs: Any) -> SubClass[np.float64]: ...

assert_type(np.empty_like(A), npt.NDArray[np.float64])
assert_type(np.empty_like(B), SubClass[np.float64])
assert_type(np.empty_like([1, 1.0]), npt.NDArray[Any])
assert_type(np.empty_like(A, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.empty_like(A, dtype='c16'), npt.NDArray[Any])

assert_type(np.array(A), npt.NDArray[np.float64])
assert_type(np.array(B), npt.NDArray[np.float64])
assert_type(np.array([1, 1.0]), npt.NDArray[Any])
assert_type(np.array(deque([1, 2, 3])), npt.NDArray[Any])
assert_type(np.array(A, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.array(A, dtype='c16'), npt.NDArray[Any])
assert_type(np.array(A, like=A), npt.NDArray[np.float64])
assert_type(np.array(A, subok=True), npt.NDArray[np.float64])
assert_type(np.array(B, subok=True), SubClass[np.float64])
assert_type(np.array(B, subok=True, ndmin=0), SubClass[np.float64])
assert_type(np.array(B, subok=True, ndmin=1), SubClass[np.float64])
assert_type(np.array(D), npt.NDArray[np.float64 | np.int64])

assert_type(np.zeros([1, 5, 6]), npt.NDArray[np.float64])
assert_type(np.zeros([1, 5, 6], dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.zeros([1, 5, 6], dtype='c16'), npt.NDArray[Any])

assert_type(np.empty([1, 5, 6]), npt.NDArray[np.float64])
assert_type(np.empty([1, 5, 6], dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.empty([1, 5, 6], dtype='c16'), npt.NDArray[Any])

assert_type(np.concatenate(A), npt.NDArray[np.float64])
assert_type(np.concatenate([A, A]), Any)
assert_type(np.concatenate([[1], A]), npt.NDArray[Any])
assert_type(np.concatenate([[1], [1]]), npt.NDArray[Any])
assert_type(np.concatenate((A, A)), npt.NDArray[np.float64])
assert_type(np.concatenate(([1], [1])), npt.NDArray[Any])
assert_type(np.concatenate([1, 1.0]), npt.NDArray[Any])
assert_type(np.concatenate(A, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.concatenate(A, dtype='c16'), npt.NDArray[Any])
assert_type(np.concatenate([1, 1.0], out=A), npt.NDArray[np.float64])

assert_type(np.asarray(A), npt.NDArray[np.float64])
assert_type(np.asarray(B), npt.NDArray[np.float64])
assert_type(np.asarray([1, 1.0]), npt.NDArray[Any])
assert_type(np.asarray(A, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.asarray(A, dtype='c16'), npt.NDArray[Any])

assert_type(np.asanyarray(A), npt.NDArray[np.float64])
assert_type(np.asanyarray(B), SubClass[np.float64])
assert_type(np.asanyarray([1, 1.0]), npt.NDArray[Any])
assert_type(np.asanyarray(A, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.asanyarray(A, dtype='c16'), npt.NDArray[Any])

assert_type(np.ascontiguousarray(A), npt.NDArray[np.float64])
assert_type(np.ascontiguousarray(B), npt.NDArray[np.float64])
assert_type(np.ascontiguousarray([1, 1.0]), npt.NDArray[Any])
assert_type(np.ascontiguousarray(A, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.ascontiguousarray(A, dtype='c16'), npt.NDArray[Any])

assert_type(np.asfortranarray(A), npt.NDArray[np.float64])
assert_type(np.asfortranarray(B), npt.NDArray[np.float64])
assert_type(np.asfortranarray([1, 1.0]), npt.NDArray[Any])
assert_type(np.asfortranarray(A, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.asfortranarray(A, dtype='c16'), npt.NDArray[Any])

assert_type(np.fromstring("1 1 1", sep=" "), npt.NDArray[np.float64])
assert_type(np.fromstring(b"1 1 1", sep=" "), npt.NDArray[np.float64])
assert_type(np.fromstring("1 1 1", dtype=np.int64, sep=" "), npt.NDArray[np.int64])
assert_type(np.fromstring(b"1 1 1", dtype=np.int64, sep=" "), npt.NDArray[np.int64])
assert_type(np.fromstring("1 1 1", dtype="c16", sep=" "), npt.NDArray[Any])
assert_type(np.fromstring(b"1 1 1", dtype="c16", sep=" "), npt.NDArray[Any])

assert_type(np.fromfile("test.txt", sep=" "), npt.NDArray[np.float64])
assert_type(np.fromfile("test.txt", dtype=np.int64, sep=" "), npt.NDArray[np.int64])
assert_type(np.fromfile("test.txt", dtype="c16", sep=" "), npt.NDArray[Any])
with open("test.txt") as f:
    assert_type(np.fromfile(f, sep=" "), npt.NDArray[np.float64])
    assert_type(np.fromfile(b"test.txt", sep=" "), npt.NDArray[np.float64])
    assert_type(np.fromfile(Path("test.txt"), sep=" "), npt.NDArray[np.float64])

assert_type(np.fromiter("12345", np.float64), npt.NDArray[np.float64])
assert_type(np.fromiter("12345", float), npt.NDArray[Any])

assert_type(np.frombuffer(A), npt.NDArray[np.float64])
assert_type(np.frombuffer(A, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.frombuffer(A, dtype="c16"), npt.NDArray[Any])

assert_type(np.arange(False, True), npt.NDArray[np.signedinteger[Any]])
assert_type(np.arange(10), npt.NDArray[np.signedinteger[Any]])
assert_type(np.arange(0, 10, step=2), npt.NDArray[np.signedinteger[Any]])
assert_type(np.arange(10.0), npt.NDArray[np.floating[Any]])
assert_type(np.arange(start=0, stop=10.0), npt.NDArray[np.floating[Any]])
assert_type(np.arange(np.timedelta64(0)), npt.NDArray[np.timedelta64])
assert_type(np.arange(0, np.timedelta64(10)), npt.NDArray[np.timedelta64])
assert_type(np.arange(np.datetime64("0"), np.datetime64("10")), npt.NDArray[np.datetime64])
assert_type(np.arange(10, dtype=np.float64), npt.NDArray[np.float64])
assert_type(np.arange(0, 10, step=2, dtype=np.int16), npt.NDArray[np.int16])
assert_type(np.arange(10, dtype=int), npt.NDArray[Any])
assert_type(np.arange(0, 10, dtype="f8"), npt.NDArray[Any])

assert_type(np.require(A), npt.NDArray[np.float64])
assert_type(np.require(B), SubClass[np.float64])
assert_type(np.require(B, requirements=None), SubClass[np.float64])
assert_type(np.require(B, dtype=int), npt.NDArray[Any])
assert_type(np.require(B, requirements="E"), npt.NDArray[Any])
assert_type(np.require(B, requirements=["ENSUREARRAY"]), npt.NDArray[Any])
assert_type(np.require(B, requirements={"F", "E"}), npt.NDArray[Any])
assert_type(np.require(B, requirements=["C", "OWNDATA"]), SubClass[np.float64])
assert_type(np.require(B, requirements="W"), SubClass[np.float64])
assert_type(np.require(B, requirements="A"), SubClass[np.float64])
assert_type(np.require(C), npt.NDArray[Any])

assert_type(np.linspace(0, 10), npt.NDArray[np.floating[Any]])
assert_type(np.linspace(0, 10j), npt.NDArray[np.complexfloating[Any, Any]])
assert_type(np.linspace(0, 10, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.linspace(0, 10, dtype=int), npt.NDArray[Any])
assert_type(np.linspace(0, 10, retstep=True), tuple[npt.NDArray[np.floating[Any]], np.floating[Any]])
assert_type(np.linspace(0j, 10, retstep=True), tuple[npt.NDArray[np.complexfloating[Any, Any]], np.complexfloating[Any, Any]])
assert_type(np.linspace(0, 10, retstep=True, dtype=np.int64), tuple[npt.NDArray[np.int64], np.int64])
assert_type(np.linspace(0j, 10, retstep=True, dtype=int), tuple[npt.NDArray[Any], Any])

assert_type(np.logspace(0, 10), npt.NDArray[np.floating[Any]])
assert_type(np.logspace(0, 10j), npt.NDArray[np.complexfloating[Any, Any]])
assert_type(np.logspace(0, 10, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.logspace(0, 10, dtype=int), npt.NDArray[Any])

assert_type(np.geomspace(0, 10), npt.NDArray[np.floating[Any]])
assert_type(np.geomspace(0, 10j), npt.NDArray[np.complexfloating[Any, Any]])
assert_type(np.geomspace(0, 10, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.geomspace(0, 10, dtype=int), npt.NDArray[Any])

assert_type(np.zeros_like(A), npt.NDArray[np.float64])
assert_type(np.zeros_like(C), npt.NDArray[Any])
assert_type(np.zeros_like(A, dtype=float), npt.NDArray[Any])
assert_type(np.zeros_like(B), SubClass[np.float64])
assert_type(np.zeros_like(B, dtype=np.int64), npt.NDArray[np.int64])

assert_type(np.ones_like(A), npt.NDArray[np.float64])
assert_type(np.ones_like(C), npt.NDArray[Any])
assert_type(np.ones_like(A, dtype=float), npt.NDArray[Any])
assert_type(np.ones_like(B), SubClass[np.float64])
assert_type(np.ones_like(B, dtype=np.int64), npt.NDArray[np.int64])

assert_type(np.full_like(A, i8), npt.NDArray[np.float64])
assert_type(np.full_like(C, i8), npt.NDArray[Any])
assert_type(np.full_like(A, i8, dtype=int), npt.NDArray[Any])
assert_type(np.full_like(B, i8), SubClass[np.float64])
assert_type(np.full_like(B, i8, dtype=np.int64), npt.NDArray[np.int64])

assert_type(np.ones(1), npt.NDArray[np.float64])
assert_type(np.ones([1, 1, 1]), npt.NDArray[np.float64])
assert_type(np.ones(5, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.ones(5, dtype=int), npt.NDArray[Any])

assert_type(np.full(1, i8), npt.NDArray[Any])
assert_type(np.full([1, 1, 1], i8), npt.NDArray[Any])
assert_type(np.full(1, i8, dtype=np.float64), npt.NDArray[np.float64])
assert_type(np.full(1, i8, dtype=float), npt.NDArray[Any])

assert_type(np.indices([1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.indices([1, 2, 3], sparse=True), tuple[npt.NDArray[np.int_], ...])

assert_type(np.fromfunction(func, (3, 5)), SubClass[np.float64])

assert_type(np.identity(10), npt.NDArray[np.float64])
assert_type(np.identity(10, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.identity(10, dtype=int), npt.NDArray[Any])

assert_type(np.atleast_1d(A), npt.NDArray[np.float64])
assert_type(np.atleast_1d(C), npt.NDArray[Any])
assert_type(np.atleast_1d(A, A), tuple[npt.NDArray[Any], ...])
assert_type(np.atleast_1d(A, C), tuple[npt.NDArray[Any], ...])
assert_type(np.atleast_1d(C, C), tuple[npt.NDArray[Any], ...])

assert_type(np.atleast_2d(A), npt.NDArray[np.float64])
assert_type(np.atleast_2d(A, A), tuple[npt.NDArray[Any], ...])

assert_type(np.atleast_3d(A), npt.NDArray[np.float64])
assert_type(np.atleast_3d(A, A), tuple[npt.NDArray[Any], ...])

assert_type(np.vstack([A, A]), np.ndarray[Any, Any])
assert_type(np.vstack([A, A], dtype=np.float64), npt.NDArray[np.float64])
assert_type(np.vstack([A, C]), npt.NDArray[Any])
assert_type(np.vstack([C, C]), npt.NDArray[Any])

assert_type(np.hstack([A, A]), np.ndarray[Any, Any])
assert_type(np.hstack([A, A], dtype=np.float64), npt.NDArray[np.float64])

assert_type(np.stack([A, A]), Any)
assert_type(np.stack([A, A], dtype=np.float64), npt.NDArray[np.float64])
assert_type(np.stack([A, C]), npt.NDArray[Any])
assert_type(np.stack([C, C]), npt.NDArray[Any])
assert_type(np.stack([A, A], axis=0), Any)
assert_type(np.stack([A, A], out=B), SubClass[np.float64])

assert_type(np.block([[A, A], [A, A]]), npt.NDArray[Any])
assert_type(np.block(C), npt.NDArray[Any])

if sys.version_info >= (3, 12):
    from collections.abc import Buffer

    def create_array(obj: npt.ArrayLike) -> npt.NDArray[Any]: ...

    buffer: Buffer
    assert_type(create_array(buffer), npt.NDArray[Any])
