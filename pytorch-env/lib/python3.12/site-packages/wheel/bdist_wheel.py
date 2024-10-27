from warnings import warn

from ._bdist_wheel import bdist_wheel as bdist_wheel

warn(
    "The 'wheel' package is no longer the canonical location of the 'bdist_wheel' "
    "command, and will be removed in a future release. Please update to setuptools "
    "v70.1 or later which contains an integrated version of this command.",
    DeprecationWarning,
    stacklevel=1,
)
