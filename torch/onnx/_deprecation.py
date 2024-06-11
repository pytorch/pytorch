# mypy: allow-untyped-defs
"""Utility for deprecating functions."""

import functools
import textwrap
import warnings


def deprecated(since: str, removed_in: str, instructions: str):
    """Marks functions as deprecated.

    It will result in a warning when the function is called and a note in the
    docstring.

    Args:
        since: The version when the function was first deprecated.
        removed_in: The version when the function will be removed.
        instructions: The action users should take.
    """

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"'{function.__module__}.{function.__name__}' "
                f"is deprecated in version {since} and will be "
                f"removed in {removed_in}. Please {instructions}.",
                category=FutureWarning,
                stacklevel=2,
            )
            return function(*args, **kwargs)

        # Add a deprecation note to the docstring.
        docstring = function.__doc__ or ""

        # Add a note to the docstring.
        deprecation_note = textwrap.dedent(
            f"""\
            .. deprecated:: {since}
                Deprecated and will be removed in version {removed_in}.
                Please {instructions}.
            """
        )

        # Split docstring at first occurrence of newline
        summary_and_body = docstring.split("\n\n", 1)

        if len(summary_and_body) > 1:
            summary, body = summary_and_body

            # Dedent the body. We cannot do this with the presence of the summary because
            # the body contains leading whitespaces when the summary does not.
            body = textwrap.dedent(body)

            new_docstring_parts = [deprecation_note, "\n\n", summary, body]
        else:
            summary = summary_and_body[0]

            new_docstring_parts = [deprecation_note, "\n\n", summary]

        wrapper.__doc__ = "".join(new_docstring_parts)

        return wrapper

    return decorator
