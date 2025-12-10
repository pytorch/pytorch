CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
-----------------------------------

.. versionadded:: 3.25

Select the MSVC debug information format targeting the MSVC ABI.
This variable is used to initialize the
:prop_tgt:`MSVC_DEBUG_INFORMATION_FORMAT` property on all targets as they are
created.  It is also propagated by calls to the :command:`try_compile` command
into the test project.

The allowed values are:

.. include:: ../prop_tgt/include/MSVC_DEBUG_INFORMATION_FORMAT-VALUES.rst

Use :manual:`generator expressions <cmake-generator-expressions(7)>` to
support per-configuration specification.  For example, the code:

.. code-block:: cmake

  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>")

selects for all following targets the program database debug information format
for the ``Debug`` and ``RelWithDebInfo`` configurations.

If this variable is not set, the :prop_tgt:`MSVC_DEBUG_INFORMATION_FORMAT`
target property will not be set automatically.  If that property is not set,
CMake selects a debug information format using the default value
``$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>``, if supported by the
compiler, and otherwise ``$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>``.

.. note::

  This variable has effect only when policy :policy:`CMP0141` is set to ``NEW``
  prior to the first :command:`project` or :command:`enable_language` command
  that enables a language using a compiler targeting the MSVC ABI.
