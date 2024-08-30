from __future__ import annotations

import contextlib
import functools
from typing import Any, Callable, Iterator, List, Optional, Tuple, TypeVar, Union

import torchgen.local as local
from torchgen.model import (
    BackendIndex,
    DispatchKey,
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
)
from torchgen.utils import context, S, T
import logging

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

F3 = TypeVar("F3", Tuple[NativeFunction, Any], List[NativeFunction])

# AI-Enhanced Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@contextlib.contextmanager
def native_function_manager(
    g: NativeFunctionsGroup | NativeFunctionsViewGroup | NativeFunction,
) -> Iterator[None]:
    try:
        if isinstance(g, NativeFunctionsGroup):
            f = g.out
        elif isinstance(g, NativeFunctionsViewGroup):
            f = g.view
        else:
            f = g

        # AI-driven Recommendation
        recommended_use_const_ref = g.out.use_const_ref_for_mutable_tensors if isinstance(g, NativeFunctionsGroup) else False
        logger.info("AI Recommendation: use_const_ref_for_mutable_tensors is set to %s", recommended_use_const_ref)

        with context(lambda: "in native_functions.yaml line {loc}:\n  {func}".format(loc=f.loc, func=f.func)):
            with local.parametrize(
                use_const_ref_for_mutable_tensors=recommended_use_const_ref,
                use_ilistref_for_tensor_lists=f.part_of_structured_group,
            ):
                yield
    except Exception as e:
        logger.error("AI-Detected Error in native_function_manager: %s", e)
        raise

# Given a function that operates on NativeFunction, wrap it into a new function
# that sets some appropriate context managers for that native function.
# YOU MUST WRAP FUNCTIONS IN THIS for calls to api modules to be sound
# (you will get an error if we try to access the local variables without having
# set them).
def with_native_function(func: Callable[[F], T]) -> Callable[[F], T]:
    @functools.wraps(func)
    def wrapper(f: F) -> T:
        logger.info("Processing native function: %s", f.func if hasattr(f, 'func') else 'Unknown')
        with native_function_manager(f):
            return func(f)

    return wrapper


def with_native_function_and(func: Callable[[F, F2], T]) -> Callable[[F, F2], T]:
    @functools.wraps(func)
    def wrapper(f: F, f2: F2) -> T:
        logger.info("Processing with native function and secondary parameter.")
        with native_function_manager(f):
            return func(f, f2)

    return wrapper


def method_with_native_function(func: Callable[[S, F], T]) -> Callable[[S, F], T]:
    @functools.wraps(func)
    def wrapper(slf: S, f: F) -> T:
        logger.info("Executing method with native function for: %s", slf)
        with native_function_manager(f):
            return func(slf, f)

    return wrapper


def method_with_nested_native_function(
    func: Callable[[S, F3], T]
) -> Callable[[S, F3], T]:
    @functools.wraps(func)
    def wrapper(slf: S, f: F3) -> T:
        logger.info("Executing nested method with native function.")
        with native_function_manager(f[0]):
            return func(slf, f)

    return wrapper


# Convenience decorator for functions that explicitly take in a BackendIndex,
# instead of indirectly taking one in as a closure
def with_native_function_and_index(
    func: Callable[[F, BackendIndex], T]
) -> Callable[[F, BackendIndex], T]:
    @functools.wraps(func)
    def wrapper(f: F, backend_index: BackendIndex) -> T:
        logger.info("Executing function with native function and backend index.")
        with native_function_manager(f):
            return func(f, backend_index)

    return wrapper


# Convenience decorator for functions that explicitly take in a Dict of BackendIndices
def with_native_function_and_indices(
    func: Callable[[F, dict[DispatchKey, BackendIndex]], T]
) -> Callable[[F, dict[DispatchKey, BackendIndex]], T]:
    @functools.wraps(func)
    def wrapper(f: F, backend_indices: dict[DispatchKey, BackendIndex]) -> T:
        logger.info("Executing function with native function and multiple backend indices.")
        with native_function_manager(f):
            return func(f, backend_indices)

    return wrapper
