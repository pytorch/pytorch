CMAKE_MSVC_RUNTIME_CHECKS
-------------------------

.. versionadded:: 4.0

Select the list of enabled runtime checks when targeting the MSVC ABI.
This variable is used to initialize the
:prop_tgt:`MSVC_RUNTIME_CHECKS` property on all targets as they are
created. It is also propagated by calls to the :command:`try_compile` command
into the test project.

The allowed values are:

.. include:: ../prop_tgt/include/MSVC_RUNTIME_CHECKS-VALUES.rst

Use :manual:`generator expressions <cmake-generator-expressions(7)>` to
support per-configuration specification. For example, the code:

.. code-block:: cmake

  set(CMAKE_MSVC_RUNTIME_CHECKS "$<$<CONFIG:Debug,RelWithDebInfo>:PossibleDataLoss;UninitializedVariable>")

enables for the target ``foo`` the possible data loss and uninitialized variables checks
for the ``Debug`` and ``RelWithDebInfo`` configurations.

If this variable is not set, the :prop_tgt:`MSVC_RUNTIME_CHECKS`
target property will not be set automatically.  If that property is not set,
CMake selects runtime checks using the default value
``$<$<CONFIG:Debug>:StackFrameErrorCheck;UninitializedVariable>``,
if supported by the compiler, or empty value otherwise.

.. note::

  This variable has effect only when policy :policy:`CMP0184` is set to ``NEW``
  prior to the first :command:`project` or :command:`enable_language` command
  that enables a language using a compiler targeting the MSVC ABI.
