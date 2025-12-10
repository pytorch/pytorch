CMAKE_C_KNOWN_FEATURES
----------------------

.. versionadded:: 3.1

List of C features known to this version of CMake.

The features listed in this global property may be known to be available to the
C compiler.  If the feature is available with the C compiler, it will
be listed in the :variable:`CMAKE_C_COMPILE_FEATURES` variable.

The features listed here may be used with the :command:`target_compile_features`
command.  See the :manual:`cmake-compile-features(7)` manual for information on
compile features and a list of supported compilers.

The features known to this version of CMake are listed below.

High level meta features indicating C standard support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.8

``c_std_90``
  Compiler mode is at least C 90.

``c_std_99``
  Compiler mode is at least C 99.

``c_std_11``
  Compiler mode is at least C 11.

``c_std_17``
  .. versionadded:: 3.21

  Compiler mode is at least C 17.

``c_std_23``
  .. versionadded:: 3.21

  Compiler mode is at least C 23.

.. include:: include/CMAKE_LANG_STD_FLAGS.rst

Low level individual compile features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``c_function_prototypes``
  Function prototypes, as defined in ``ISO/IEC 9899:1990``.

``c_restrict``
  ``restrict`` keyword, as defined in ``ISO/IEC 9899:1999``.

``c_static_assert``
  Static assert, as defined in ``ISO/IEC 9899:2011``.

``c_variadic_macros``
  Variadic macros, as defined in ``ISO/IEC 9899:1999``.
