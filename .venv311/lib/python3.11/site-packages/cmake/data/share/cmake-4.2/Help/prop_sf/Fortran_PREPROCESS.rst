Fortran_PREPROCESS
------------------

.. versionadded:: 3.18

Control whether the Fortran source file should be unconditionally preprocessed.

If unset or empty, rely on the compiler to determine whether the file
should be preprocessed. If explicitly set to ``OFF`` then the file
does not need to be preprocessed. If explicitly set to ``ON``, then
the file does need to be preprocessed as part of the compilation step.

When using the :generator:`Ninja` generator, all source files are
first preprocessed in order to generate module dependency
information. Setting this property to ``OFF`` will make ``Ninja``
skip this step.

Consider using the target-wide :prop_tgt:`Fortran_PREPROCESS` property
if all source files in a target need to be preprocessed.
