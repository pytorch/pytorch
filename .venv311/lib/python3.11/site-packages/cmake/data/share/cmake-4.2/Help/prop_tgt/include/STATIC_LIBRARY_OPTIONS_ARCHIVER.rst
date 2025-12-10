Handling Archiver Driver Differences
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 4.0

To pass options to the archiver tool, each compiler driver has its own syntax.
The ``ARCHIVER:`` prefix and ``,`` separator can be used to specify, in a portable
way, options to pass to the archiver tool. ``ARCHIVER:`` is replaced by the
appropriate driver option and ``,`` by the appropriate driver separator.
The driver prefix and driver separator are given by the values of the
:variable:`CMAKE_<LANG>_ARCHIVER_WRAPPER_FLAG` and
:variable:`CMAKE_<LANG>_ARCHIVER_WRAPPER_FLAG_SEP` variables.

The ``ARCHIVER:`` prefix can be specified as part of a ``SHELL:`` prefix
expression.

The ``ARCHIVER:`` prefix supports, as an alternative syntax, specification of
arguments using the ``SHELL:`` prefix and space as separator.

.. note::

  Specifying the ``SHELL:`` prefix anywhere other than at the beginning of the
  ``ARCHIVER:`` prefix is not supported.
