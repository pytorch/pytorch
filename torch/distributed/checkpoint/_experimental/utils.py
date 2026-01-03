"""
Utility functions for the experimental checkpoint module.

This module contains helper functions and utilities used across the experimental
checkpoint functionality.
"""

from concurrent.futures import Future
from typing import Any


def wrap_future(original_result: Any) -> Future[None]:
    """
    Wraps a result (Future or not) to return a Future with None result.

    If the input is a Future, returns a new Future that completes with None when
    the original Future completes successfully, or propagates any exception.
    If the input is not a Future, returns a completed Future with None result.

    Args:
        original_result: The result to wrap (Future or any other value).

    Returns:
        A Future that completes with None on success or propagates exceptions.
    """
    masked_future: Future[None] = Future()

    if isinstance(original_result, Future):

        def on_complete(_: Future[Any]) -> None:
            try:
                original_result.result()
                masked_future.set_result(None)
            except Exception as e:
                masked_future.set_exception(e)

        original_result.add_done_callback(on_complete)
    else:
        # Return a completed future with None result
        masked_future.set_result(None)

    return masked_future
