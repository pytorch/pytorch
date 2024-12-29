import sys


if sys.version_info < (3, 10):
    import importlib_metadata as metadata  # pragma: no cover
else:
    import importlib.metadata as metadata  # noqa: F401


if sys.version_info < (3, 9):
    import importlib_resources as resources  # pragma: no cover
else:
    import importlib.resources as resources  # noqa: F401
