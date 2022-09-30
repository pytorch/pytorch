"""Utility for deprecating functions."""

import functools
import textwrap
import warnings


def deprecated(since: str, removed_in: str, instructions: str):
    """Marks functions as deprecated.

    It will result in a warning when the function is called.

    Args:
        since: The version when the function was first deprecated.
        removed_in: The version when the function will be removed.
        instructions: The action users should take.
    """

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"`{function.__module__}.{function.__name__}` is deprecated in version {since} and will be "
                f"removed in version {removed_in}. Please {instructions}.",
                category=FutureWarning,
                stacklevel=2,
            )
            return function(*args, **kwargs)


        wrapper.__doc__ = textwrap.dedent(
            f"""\
            .. deprecated:: {since}
                Will be removed in version {removed_in}. Please {instructions}.

            """
        ) + (function.__doc__ or "")

        return wrapper

    return decorator
