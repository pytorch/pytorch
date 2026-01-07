# pyre-strict

"""
Custom decoder functions for use with PersistentMemoizer.

This module provides reusable decoder functions that reconstruct function results
from cached data for deterministic cache replay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import ParamSpec, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable


# Type alias for the encoded result
TunedKernelEncodedResultDict = dict[str, object]

# Type variable for function parameters
_P = ParamSpec("_P")
# Type variable for return type
_R = TypeVar("_R")


def tuned_kernel_result_decoder(
    fn: Callable[_P, _R],
) -> Callable[_P, Callable[[TunedKernelEncodedResultDict], _R]]:
    """Factory factory that returns a params-to-decoder factory for tuned kernel results.

    This is a placeholder implementation that doesn't actually decode cached results.
    It just re-executes the underlying function.

    Args:
        fn: The underlying unwrapped function (passed by the memoizer)

    Returns:
        A factory that takes (*args, **kwargs) and returns a decoder function
    """

    def params_to_decoder(
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Callable[[TunedKernelEncodedResultDict], _R]:
        """Factory that returns a decoder function for the given params."""

        def decode_result(encoded: TunedKernelEncodedResultDict) -> _R:
            return fn(*args, **kwargs)

        return decode_result

    return params_to_decoder


# Operation-specific decoders (all use placeholder for now)
tuned_mm_result_decoder = tuned_kernel_result_decoder
tuned_addmm_result_decoder = tuned_kernel_result_decoder
tuned_bmm_result_decoder = tuned_kernel_result_decoder
tuned_baddbmm_result_decoder = tuned_kernel_result_decoder
