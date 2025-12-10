WINDOWS_EXPORT_ALL_SYMBOLS
--------------------------

.. versionadded:: 3.4

This property is implemented only for MS-compatible tools on Windows.

Enable this boolean property to automatically create a module definition
(``.def``) file with all global symbols found in the input ``.obj`` files
for a ``SHARED`` library (or executable with :prop_tgt:`ENABLE_EXPORTS`)
on Windows.  The module definition file will be passed to the linker
causing all symbols to be exported from the ``.dll``.

This simplifies porting projects to Windows by reducing the need for
explicit ``dllexport`` markup, even in ``C++`` classes.  Function
symbols will be automatically exported and may be linked by callers.
However, there are some cases when compiling code in a consumer may
require explicit ``dllimport`` markup:

* Global *data* symbols must be explicitly marked with
  ``__declspec(dllimport)`` in order to link to data in the ``.dll``.

* In cases that the compiler generates references to the virtual function
  table, such as in a delegating constructor of a class with virtual
  functions, the whole class must be marked with ``__declspec(dllimport)``
  in order to link to the vftable in the ``.dll``.

* See the `MSVC Linker /EXPORT Option`_ for more information on data symbols.

.. _`MSVC Linker /EXPORT Option`: https://learn.microsoft.com/en-us/cpp/build/reference/export-exports-a-function

When this property is enabled, zero or more ``.def`` files may also be
specified as source files of the target.  The exports named by these files
will be merged with those detected from the object files to generate a
single module definition file to be passed to the linker.  This can be
used to export symbols from a ``.dll`` that are not in any of its object
files but are added by the linker from dependencies (e.g. ``msvcrt.lib``).

This property is initialized by the value of
the :variable:`CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS` variable if it is set
when a target is created.
