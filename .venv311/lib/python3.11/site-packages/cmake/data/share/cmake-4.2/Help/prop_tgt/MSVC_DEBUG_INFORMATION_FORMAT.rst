MSVC_DEBUG_INFORMATION_FORMAT
-----------------------------

.. versionadded:: 3.25

Select debug information format when targeting the MSVC ABI.

The allowed values are:

.. include:: include/MSVC_DEBUG_INFORMATION_FORMAT-VALUES.rst

Use :manual:`generator expressions <cmake-generator-expressions(7)>` to
support per-configuration specification.  For example, the code:

.. code-block:: cmake

  add_executable(foo foo.c)
  set_property(TARGET foo PROPERTY
    MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>")

selects for the target ``foo`` the program database debug information format
for the ``Debug`` and ``RelWithDebInfo`` configurations.

This property is initialized from the value of the
:variable:`CMAKE_MSVC_DEBUG_INFORMATION_FORMAT` variable, if it is set.
If this property is not set, CMake selects a debug information format using
the default value ``$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>``, if
supported by the compiler, and otherwise
``$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>``.

.. note::

  This property has effect only when policy :policy:`CMP0141` is set to ``NEW``
  prior to the first :command:`project` or :command:`enable_language` command
  that enables a language using a compiler targeting the MSVC ABI.
