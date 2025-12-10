"""A mypy_ plugin for managing a number of platform-specific annotations.
Its functionality can be split into three distinct parts:

* Assigning the (platform-dependent) precisions of certain `~numpy.number`
  subclasses, including the likes of `~numpy.int_`, `~numpy.intp` and
  `~numpy.longlong`. See the documentation on
  :ref:`scalar types <arrays.scalars.built-in>` for a comprehensive overview
  of the affected classes. Without the plugin the precision of all relevant
  classes will be inferred as `~typing.Any`.
* Removing all extended-precision `~numpy.number` subclasses that are
  unavailable for the platform in question. Most notably this includes the
  likes of `~numpy.float128` and `~numpy.complex256`. Without the plugin *all*
  extended-precision types will, as far as mypy is concerned, be available
  to all platforms.
* Assigning the (platform-dependent) precision of `~numpy.ctypeslib.c_intp`.
  Without the plugin the type will default to `ctypes.c_int64`.

  .. versionadded:: 1.22

.. deprecated:: 2.3

Examples
--------
To enable the plugin, one must add it to their mypy `configuration file`_:

.. code-block:: ini

    [mypy]
    plugins = numpy.typing.mypy_plugin

.. _mypy: https://mypy-lang.org/
.. _configuration file: https://mypy.readthedocs.io/en/stable/config_file.html

"""

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Final, TypeAlias, cast

import numpy as np

__all__: list[str] = []


def _get_precision_dict() -> dict[str, str]:
    names = [
        ("_NBitByte", np.byte),
        ("_NBitShort", np.short),
        ("_NBitIntC", np.intc),
        ("_NBitIntP", np.intp),
        ("_NBitInt", np.int_),
        ("_NBitLong", np.long),
        ("_NBitLongLong", np.longlong),

        ("_NBitHalf", np.half),
        ("_NBitSingle", np.single),
        ("_NBitDouble", np.double),
        ("_NBitLongDouble", np.longdouble),
    ]
    ret: dict[str, str] = {}
    for name, typ in names:
        n = 8 * np.dtype(typ).itemsize
        ret[f"{_MODULE}._nbit.{name}"] = f"{_MODULE}._nbit_base._{n}Bit"
    return ret


def _get_extended_precision_list() -> list[str]:
    extended_names = [
        "float96",
        "float128",
        "complex192",
        "complex256",
    ]
    return [i for i in extended_names if hasattr(np, i)]

def _get_c_intp_name() -> str:
    # Adapted from `np.core._internal._getintp_ctype`
    return {
        "i": "c_int",
        "l": "c_long",
        "q": "c_longlong",
    }.get(np.dtype("n").char, "c_long")


_MODULE: Final = "numpy._typing"

#: A dictionary mapping type-aliases in `numpy._typing._nbit` to
#: concrete `numpy.typing.NBitBase` subclasses.
_PRECISION_DICT: Final = _get_precision_dict()

#: A list with the names of all extended precision `np.number` subclasses.
_EXTENDED_PRECISION_LIST: Final = _get_extended_precision_list()

#: The name of the ctypes equivalent of `np.intp`
_C_INTP: Final = _get_c_intp_name()


try:
    if TYPE_CHECKING:
        from mypy.typeanal import TypeAnalyser

    import mypy.types
    from mypy.build import PRI_MED
    from mypy.nodes import ImportFrom, MypyFile, Statement
    from mypy.plugin import AnalyzeTypeContext, Plugin

except ModuleNotFoundError as e:

    def plugin(version: str) -> type:
        raise e

else:

    _HookFunc: TypeAlias = Callable[[AnalyzeTypeContext], mypy.types.Type]

    def _hook(ctx: AnalyzeTypeContext) -> mypy.types.Type:
        """Replace a type-alias with a concrete ``NBitBase`` subclass."""
        typ, _, api = ctx
        name = typ.name.split(".")[-1]
        name_new = _PRECISION_DICT[f"{_MODULE}._nbit.{name}"]
        return cast("TypeAnalyser", api).named_type(name_new)

    def _index(iterable: Iterable[Statement], id: str) -> int:
        """Identify the first ``ImportFrom`` instance the specified `id`."""
        for i, value in enumerate(iterable):
            if getattr(value, "id", None) == id:
                return i
        raise ValueError("Failed to identify a `ImportFrom` instance "
                         f"with the following id: {id!r}")

    def _override_imports(
        file: MypyFile,
        module: str,
        imports: list[tuple[str, str | None]],
    ) -> None:
        """Override the first `module`-based import with new `imports`."""
        # Construct a new `from module import y` statement
        import_obj = ImportFrom(module, 0, names=imports)
        import_obj.is_top_level = True

        # Replace the first `module`-based import statement with `import_obj`
        for lst in [file.defs, cast("list[Statement]", file.imports)]:
            i = _index(lst, module)
            lst[i] = import_obj

    class _NumpyPlugin(Plugin):
        """A mypy plugin for handling versus numpy-specific typing tasks."""

        def get_type_analyze_hook(self, fullname: str) -> _HookFunc | None:
            """Set the precision of platform-specific `numpy.number`
            subclasses.

            For example: `numpy.int_`, `numpy.longlong` and `numpy.longdouble`.
            """
            if fullname in _PRECISION_DICT:
                return _hook
            return None

        def get_additional_deps(
            self, file: MypyFile
        ) -> list[tuple[int, str, int]]:
            """Handle all import-based overrides.

            * Import platform-specific extended-precision `numpy.number`
              subclasses (*e.g.* `numpy.float96` and `numpy.float128`).
            * Import the appropriate `ctypes` equivalent to `numpy.intp`.

            """
            fullname = file.fullname
            if fullname == "numpy":
                _override_imports(
                    file,
                    f"{_MODULE}._extended_precision",
                    imports=[(v, v) for v in _EXTENDED_PRECISION_LIST],
                )
            elif fullname == "numpy.ctypeslib":
                _override_imports(
                    file,
                    "ctypes",
                    imports=[(_C_INTP, "_c_intp")],
                )
            return [(PRI_MED, fullname, -1)]

    def plugin(version: str) -> type:
        import warnings

        plugin = "numpy.typing.mypy_plugin"
        # Deprecated 2025-01-10, NumPy 2.3
        warn_msg = (
            f"`{plugin}` is deprecated, and will be removed in a future "
            f"release. Please remove `plugins = {plugin}` in your mypy config."
            f"(deprecated in NumPy 2.3)"
        )
        warnings.warn(warn_msg, DeprecationWarning, stacklevel=3)

        return _NumpyPlugin
