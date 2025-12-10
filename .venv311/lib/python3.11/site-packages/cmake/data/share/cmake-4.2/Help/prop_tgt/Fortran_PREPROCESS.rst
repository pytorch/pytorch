Fortran_PREPROCESS
------------------

.. versionadded:: 3.18

Control whether the Fortran source file should be unconditionally
preprocessed.

If unset or empty, rely on the compiler to determine whether the file
should be preprocessed. If explicitly set to ``OFF`` then the file does not
need to be preprocessed. If explicitly set to ``ON``, then the file does
need to be preprocessed as part of the compilation step.

When using the :generator:`Ninja` generator, all source files are
first preprocessed in order to generate module dependency
information. Setting this property to ``OFF`` will make ``Ninja``
skip this step.

Use the source-specific :prop_sf:`Fortran_PREPROCESS` property if a single
file needs to be preprocessed. If the variable
:variable:`CMAKE_Fortran_PREPROCESS` is set when a target is created its
value is used to initialize this property.

.. note:: For some compilers, ``NAG``, ``PGI`` and ``Solaris Studio``,
          setting this to ``OFF`` will have no effect.
