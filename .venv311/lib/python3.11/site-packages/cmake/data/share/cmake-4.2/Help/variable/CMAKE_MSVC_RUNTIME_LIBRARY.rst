CMAKE_MSVC_RUNTIME_LIBRARY
--------------------------

.. versionadded:: 3.15

Select the MSVC runtime library for use by compilers targeting the MSVC ABI.
This variable is used to initialize the :prop_tgt:`MSVC_RUNTIME_LIBRARY`
property on all targets as they are created.  It is also propagated by
calls to the :command:`try_compile` command into the test project.

The allowed values are:

.. include:: ../prop_tgt/include/MSVC_RUNTIME_LIBRARY-VALUES.rst

Use :manual:`generator expressions <cmake-generator-expressions(7)>` to
support per-configuration specification.  For example, the code:

.. code-block:: cmake

  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

selects for all following targets a multi-threaded statically-linked runtime
library with or without debug information depending on the configuration.

If this variable is not set then the :prop_tgt:`MSVC_RUNTIME_LIBRARY` target
property will not be set automatically.  If that property is not set then
CMake uses the default value ``MultiThreaded$<$<CONFIG:Debug>:Debug>DLL``
to select a MSVC runtime library.

.. note::

  This variable has effect only when policy :policy:`CMP0091` is set to ``NEW``
  prior to the first :command:`project` or :command:`enable_language` command
  that enables a language using a compiler targeting the MSVC ABI.
