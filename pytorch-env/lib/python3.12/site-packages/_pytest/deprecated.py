"""Deprecation messages and bits of code used elsewhere in the codebase that
is planned to be removed in the next pytest release.

Keeping it in a central location makes it easy to track what is deprecated and should
be removed when the time comes.

All constants defined in this module should be either instances of
:class:`PytestWarning`, or :class:`UnformattedWarning`
in case of warnings which need to format their messages.
"""

from __future__ import annotations

from warnings import warn

from _pytest.warning_types import PytestDeprecationWarning
from _pytest.warning_types import PytestRemovedIn9Warning
from _pytest.warning_types import UnformattedWarning


# set of plugins which have been integrated into the core; we use this list to ignore
# them during registration to avoid conflicts
DEPRECATED_EXTERNAL_PLUGINS = {
    "pytest_catchlog",
    "pytest_capturelog",
    "pytest_faulthandler",
}


# This can be* removed pytest 8, but it's harmless and common, so no rush to remove.
# * If you're in the future: "could have been".
YIELD_FIXTURE = PytestDeprecationWarning(
    "@pytest.yield_fixture is deprecated.\n"
    "Use @pytest.fixture instead; they are the same."
)

# This deprecation is never really meant to be removed.
PRIVATE = PytestDeprecationWarning("A private pytest class or function was used.")


HOOK_LEGACY_PATH_ARG = UnformattedWarning(
    PytestRemovedIn9Warning,
    "The ({pylib_path_arg}: py.path.local) argument is deprecated, please use ({pathlib_path_arg}: pathlib.Path)\n"
    "see https://docs.pytest.org/en/latest/deprecations.html"
    "#py-path-local-arguments-for-hooks-replaced-with-pathlib-path",
)

NODE_CTOR_FSPATH_ARG = UnformattedWarning(
    PytestRemovedIn9Warning,
    "The (fspath: py.path.local) argument to {node_type_name} is deprecated. "
    "Please use the (path: pathlib.Path) argument instead.\n"
    "See https://docs.pytest.org/en/latest/deprecations.html"
    "#fspath-argument-for-node-constructors-replaced-with-pathlib-path",
)

HOOK_LEGACY_MARKING = UnformattedWarning(
    PytestDeprecationWarning,
    "The hook{type} {fullname} uses old-style configuration options (marks or attributes).\n"
    "Please use the pytest.hook{type}({hook_opts}) decorator instead\n"
    " to configure the hooks.\n"
    " See https://docs.pytest.org/en/latest/deprecations.html"
    "#configuring-hook-specs-impls-using-markers",
)

MARKED_FIXTURE = PytestRemovedIn9Warning(
    "Marks applied to fixtures have no effect\n"
    "See docs: https://docs.pytest.org/en/stable/deprecations.html#applying-a-mark-to-a-fixture-function"
)

# You want to make some `__init__` or function "private".
#
#   def my_private_function(some, args):
#       ...
#
# Do this:
#
#   def my_private_function(some, args, *, _ispytest: bool = False):
#       check_ispytest(_ispytest)
#       ...
#
# Change all internal/allowed calls to
#
#   my_private_function(some, args, _ispytest=True)
#
# All other calls will get the default _ispytest=False and trigger
# the warning (possibly error in the future).


def check_ispytest(ispytest: bool) -> None:
    if not ispytest:
        warn(PRIVATE, stacklevel=3)
