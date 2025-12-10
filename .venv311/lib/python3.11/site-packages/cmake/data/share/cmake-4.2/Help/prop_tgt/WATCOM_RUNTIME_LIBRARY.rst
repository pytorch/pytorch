WATCOM_RUNTIME_LIBRARY
----------------------

.. versionadded:: 3.24

Select the Watcom runtime library for use by compilers targeting the Watcom ABI.

The allowed values are:

.. include:: include/WATCOM_RUNTIME_LIBRARY-VALUES.rst

Use :manual:`generator expressions <cmake-generator-expressions(7)>` to
support per-configuration specification.

For example, the code:

.. code-block:: cmake

  add_executable(foo foo.c)
  set_property(TARGET foo PROPERTY
    WATCOM_RUNTIME_LIBRARY "MultiThreaded")

selects for the target ``foo`` a multi-threaded statically-linked runtime
library.

If this property is not set then CMake uses the default value
``MultiThreadedDLL`` on Windows and ``SingleThreaded`` on other
platforms to select a Watcom runtime library.

.. note::

  This property has effect only when policy :policy:`CMP0136` is set to ``NEW``
  prior to the first :command:`project` or :command:`enable_language` command
  that enables a language using a compiler targeting the Watcom ABI.
