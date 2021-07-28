import _warnings
import os.path

# note: implementations
# copied from cpython's import code


# _zip_searchorder defines how we search for a module in the Zip
# archive: we first search for a package __init__, then for
# non-package .pyc, and .py entries. The .pyc entries
# are swapped by initzipimport() if we run in optimized mode. Also,
# '/' is replaced by path_sep there.

_zip_searchorder = (
    ("/__init__.py", True),
    (".py", False),
)

# Replace any occurrences of '\r\n?' in the input string with '\n'.
# This converts DOS and Mac line endings to Unix line endings.
def _normalize_line_endings(source):
    source = source.replace(b"\r\n", b"\n")
    source = source.replace(b"\r", b"\n")
    return source


def _resolve_name(name, package, level):
    """Resolve a relative module name to an absolute one."""
    bits = package.rsplit(".", level - 1)
    if len(bits) < level:
        raise ValueError("attempted relative import beyond top-level package")
    base = bits[0]
    return "{}.{}".format(base, name) if name else base


def _sanity_check(name, package, level):
    """Verify arguments are "sane"."""
    if not isinstance(name, str):
        raise TypeError("module name must be str, not {}".format(type(name)))
    if level < 0:
        raise ValueError("level must be >= 0")
    if level > 0:
        if not isinstance(package, str):
            raise TypeError("__package__ not set to a string")
        elif not package:
            raise ImportError(
                "attempted relative import with no known parent " "package"
            )
    if not name and level == 0:
        raise ValueError("Empty module name")


def _calc___package__(globals):
    """Calculate what __package__ should be.

    __package__ is not guaranteed to be defined or could be set to None
    to represent that its proper value is unknown.

    """
    package = globals.get("__package__")
    spec = globals.get("__spec__")
    if package is not None:
        if spec is not None and package != spec.parent:
            _warnings.warn(
                "__package__ != __spec__.parent " f"({package!r} != {spec.parent!r})",
                ImportWarning,
                stacklevel=3,
            )
        return package
    elif spec is not None:
        return spec.parent
    else:
        _warnings.warn(
            "can't resolve package from __spec__ or __package__, "
            "falling back on __name__ and __path__",
            ImportWarning,
            stacklevel=3,
        )
        package = globals["__name__"]
        if "__path__" not in globals:
            package = package.rpartition(".")[0]
    return package


def _normalize_path(path):
    """Normalize a path by ensuring it is a string.

    If the resulting string contains path separators, an exception is raised.
    """
    parent, file_name = os.path.split(path)
    if parent:
        raise ValueError("{!r} must be only a file name".format(path))
    else:
        return file_name
