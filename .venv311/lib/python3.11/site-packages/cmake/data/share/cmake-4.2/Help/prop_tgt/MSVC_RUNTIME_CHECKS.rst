MSVC_RUNTIME_CHECKS
-------------------

.. versionadded:: 4.0

Select the list of enabled runtime checks when targeting the MSVC ABI.

The allowed values are:

.. include:: include/MSVC_RUNTIME_CHECKS-VALUES.rst

Use :manual:`generator expressions <cmake-generator-expressions(7)>` to
support per-configuration specification.  For example, the code:

.. code-block:: cmake

  add_executable(foo foo.c)
  set_property(TARGET foo PROPERTY
    MSVC_RUNTIME_CHECKS "$<$<CONFIG:Debug,RelWithDebInfo>:PossibleDataLoss;UninitializedVariable>")

enables for the target ``foo`` the possible data loss and uninitialized
variables checks for the ``Debug`` and ``RelWithDebInfo`` configurations.

This property is initialized from the value of the
:variable:`CMAKE_MSVC_RUNTIME_CHECKS` variable, if it is set.
If this property is not set, CMake selects runtime checks using the default
value ``$<$<CONFIG:Debug>:StackFrameErrorCheck;UninitializedVariable>``,
if supported by the compiler, or an empty value otherwise.

.. note::

  This property has effect only when policy :policy:`CMP0184` is set to ``NEW``
  prior to the first :command:`project` or :command:`enable_language` command
  that enables a language using a compiler targeting the MSVC ABI.
