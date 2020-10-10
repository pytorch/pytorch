"""For neatly implementing static typing in pip.

`mypy` - the static type analysis tool we use - uses the `typing` module, which
provides core functionality fundamental to mypy's functioning.

Generally, `typing` would be imported at runtime and used in that fashion -
it acts as a no-op at runtime and does not have any run-time overhead by
design.

As it turns out, `typing` is not vendorable - it uses separate sources for
Python 2/Python 3. Thus, this codebase can not expect it to be present.
To work around this, mypy allows the typing import to be behind a False-y
optional to prevent it from running at runtime and type-comments can be used
to remove the need for the types to be accessible directly during runtime.

This module provides the False-y guard in a nicely named fashion so that a
curious maintainer can reach here to read this.

In pip, all static-typing related imports should be guarded as follows:

    from pip._internal.utils.typing import MYPY_CHECK_RUNNING

    if MYPY_CHECK_RUNNING:
        from typing import ...

Ref: https://github.com/python/mypy/issues/3216
"""

MYPY_CHECK_RUNNING = False


if MYPY_CHECK_RUNNING:
    from typing import cast
else:
    # typing's cast() is needed at runtime, but we don't want to import typing.
    # Thus, we use a dummy no-op version, which we tell mypy to ignore.
    def cast(type_, value):  # type: ignore
        return value
