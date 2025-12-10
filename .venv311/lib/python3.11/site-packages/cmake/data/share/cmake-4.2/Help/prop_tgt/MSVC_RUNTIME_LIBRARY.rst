MSVC_RUNTIME_LIBRARY
--------------------

.. versionadded:: 3.15

Select the MSVC runtime library for use by compilers targeting the MSVC ABI.

The allowed values are:

.. include:: include/MSVC_RUNTIME_LIBRARY-VALUES.rst

Use :manual:`generator expressions <cmake-generator-expressions(7)>` to
support per-configuration specification.  For example, the code:

.. code-block:: cmake

  add_executable(foo foo.c)
  set_property(TARGET foo PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

selects for the target ``foo`` a multi-threaded statically-linked runtime
library with or without debug information depending on the configuration.

The property is initialized from the value of the
:variable:`CMAKE_MSVC_RUNTIME_LIBRARY` variable, if it is set.
If the property is not set, then CMake uses the default value
``MultiThreaded$<$<CONFIG:Debug>:Debug>DLL`` to select a MSVC runtime library.

.. note::

  This property has effect only when policy :policy:`CMP0091` is set to ``NEW``
  prior to the first :command:`project` or :command:`enable_language` command
  that enables a language using a compiler targeting the MSVC ABI.
