ASM<DIALECT>
------------

.. include:: include/ENV_VAR.rst

Preferred executable for compiling a specific dialect of assembly language
files. ``ASM<DIALECT>`` can be one of:

* ``ASM``
* ``ASM_NASM`` (Netwide Assembler)
* ``ASM_MASM`` (Microsoft Assembler)
* ``ASM_MARMASM`` (Microsoft ARM Assembler)
* ``ASM-ATT`` (Assembler AT&T)

Will only be used by CMake on the first configuration to determine
``ASM<DIALECT>`` compiler, after which the value for ``ASM<DIALECT>`` is stored
in the cache as
:variable:`CMAKE_ASM<DIALECT>_COMPILER <CMAKE_<LANG>_COMPILER>`. For subsequent
configuration runs, the environment variable will be ignored in favor of
:variable:`CMAKE_ASM<DIALECT>_COMPILER <CMAKE_<LANG>_COMPILER>`.

.. note::
  Options that are required to make the compiler work correctly can be included;
  they can not be changed.

.. code-block:: console

  $ export ASM="custom-compiler --arg1 --arg2"
