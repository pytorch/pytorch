CMAKE_LINKER_TYPE
-----------------

.. versionadded:: 3.29

Specify which linker will be used for the link step.

This variable is used to initialize the :prop_tgt:`LINKER_TYPE` property
on each target created by a call to :command:`add_library` or
:command:`add_executable`.  It is meaningful only for targets having a
link step.  If set, its value is also used by the :command:`try_compile`
command.

.. include:: include/LINKER_PREDEFINED_TYPES.rst
