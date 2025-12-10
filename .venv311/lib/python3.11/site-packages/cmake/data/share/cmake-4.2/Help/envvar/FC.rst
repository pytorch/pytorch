FC
--

.. include:: include/ENV_VAR.rst

Preferred executable for compiling ``Fortran`` language files. Will only be used
by CMake on the first configuration to determine ``Fortran`` compiler, after
which the value for ``Fortran`` is stored in the cache as
:variable:`CMAKE_Fortran_COMPILER <CMAKE_<LANG>_COMPILER>`. For any
configuration run (including the first), the environment variable will be
ignored if the :variable:`CMAKE_Fortran_COMPILER <CMAKE_<LANG>_COMPILER>`
variable is defined.

.. note::
  Options that are required to make the compiler work correctly can be included;
  they can not be changed.

.. code-block:: console

  $ export FC="custom-compiler --arg1 --arg2"
