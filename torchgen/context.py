import contextlib

import functools
from typing import Callable, Dict, Iterator, Optional, TypeVar, Union

import torchgen.local as local
from torchgen.model import (
    BackendIndex,
    DispatchKey,
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
)
from torchgen.utils import context, S, T

# Helper functions for defining generators on things in the model

F = TypeVar(
    "F",
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
    Union[NativeFunction, NativeFunctionsGroup],
    Union[NativeFunction, NativeFunctionsViewGroup],
)

F2 = TypeVar(
    "F2",
    NativeFunction,
    NativeFunctionsGroup,
    Optional[NativeFunction],
    bool,
    str,
)


@contextlib.contextmanager
def native_function_manager(
    g: Union[NativeFunctionsGroup, NativeFunctionsViewGroup, NativeFunction]
) -> Iterator[None]:
    if isinstance(g, NativeFunctionsGroup):
        # By default, we associate all errors with structured native functions
        # with the out variant.  In some cases, it might be better to have
        # a more specific place to hang things; if so, use
        # native_function_manager again on the inside
        f = g.out
    elif isinstance(g, NativeFunctionsViewGroup):
        # We associate errors with the view operator
        f = g.view
    else:
        f = g
    with context(lambda: f"in native_functions.yaml line {f.loc}:\n  {f.func}"):
        with local.parametrize(
            use_const_ref_for_mutable_tensors=f.use_const_ref_for_mutable_tensors,
            use_ilistref_for_tensor_lists=f.part_of_structured_group,
        ):
            yield


# Given a function that operates on NativeFunction, wrap it into a new function
# that sets some appropriate context managers for that native function.
# YOU MUST WRAP FUNCTIONS IN THIS for calls to api modules to be sound
# (you will get an error if we try to access the local variables without having
# set them).
def with_native_function(func: Callable[[F], T]) -> Callable[[F], T]:
    @functools.wraps(func)
    def wrapper(f: F) -> T:
        with native_function_manager(f):
            return func(f)

    return wrapper


def with_native_function_and(func: Callable[[F, F2], T]) -> Callable[[F, F2], T]:
    @functools.wraps(func)
    def wrapper(f: F, f2: F2) -> T:
        # The first native_function is assumed to be the one with the appropriate context.
        with native_function_manager(f):
            return func(f, f2)

    return wrapper


def method_with_native_function(func: Callable[[S, F], T]) -> Callable[[S, F], T]:
    @functools.wraps(func)
    def wrapper(slf: S, f: F) -> T:
        with native_function_manager(f):
            return func(slf, f)

    return wrapper


# Convenience decorator for functions that explicitly take in a BackendIndex,
# instead of indirectly taking one in as a closure
def with_native_function_and_index(
    func: Callable[[F, BackendIndex], T]
) -> Callable[[F, BackendIndex], T]:
    @functools.wraps(func)
    def wrapper(f: F, backend_index: BackendIndex) -> T:
        with native_function_manager(f):
            return func(f, backend_index)

    return wrapper


# Convenience decorator for functions that explicitly take in a Dict of BackendIndices
def with_native_function_and_indices(
    func: Callable[[F, Dict[DispatchKey, BackendIndex]], T]
) -> Callable[[F, Dict[DispatchKey, BackendIndex]], T]:
    @functools.wraps(func)
    def wrapper(f: F, backend_indices: Dict[DispatchKey, BackendIndex]) -> T:
        with native_function_manager(f):
            return func(f, backend_indices)

    return wrapper
