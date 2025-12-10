Handling Compiler Driver Differences
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 4.0

To pass options to the linker tool, each compiler driver has its own syntax.
The ``LINKER:`` prefix and ``,`` separator can be used to specify, in a portable
way, options to pass to the linker tool. ``LINKER:`` is replaced by the
appropriate driver option and ``,`` by the appropriate driver separator.
The driver prefix and driver separator are given by the values of the
:variable:`CMAKE_<LANG>_LINKER_WRAPPER_FLAG` and
:variable:`CMAKE_<LANG>_LINKER_WRAPPER_FLAG_SEP` variables.

For example, ``"LINKER:-z,defs"`` becomes ``-Xlinker -z -Xlinker defs`` for
``Clang`` and ``-Wl,-z,defs`` for ``GNU GCC``.

The ``LINKER:`` prefix supports, as an alternative syntax, specification of
arguments using the ``SHELL:`` prefix and space as separator. The previous
example then becomes ``"LINKER:SHELL:-z defs"``.

.. note::

  Specifying the ``SHELL:`` prefix anywhere other than at the beginning of the
  ``LINKER:`` prefix is not supported.
