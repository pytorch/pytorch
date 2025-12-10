RC
--

.. include:: include/ENV_VAR.rst

Preferred executable for compiling ``resource`` files. Will only be used by CMake
on the first configuration to determine ``resource`` compiler, after which the
value for ``RC`` is stored in the cache as
:variable:`CMAKE_RC_COMPILER <CMAKE_<LANG>_COMPILER>`. For any configuration run
(including the first), the environment variable will be ignored if the
:variable:`CMAKE_RC_COMPILER <CMAKE_<LANG>_COMPILER>` variable is defined.

.. note::
  Options that are required to make the compiler work correctly can be included;
  they can not be changed.

.. code-block:: console

  $ export RC="custom-compiler --arg1 --arg2"
