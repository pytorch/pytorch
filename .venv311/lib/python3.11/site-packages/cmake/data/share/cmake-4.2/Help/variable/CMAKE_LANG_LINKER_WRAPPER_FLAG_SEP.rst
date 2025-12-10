CMAKE_<LANG>_LINKER_WRAPPER_FLAG_SEP
------------------------------------

.. versionadded:: 3.13

This variable is used with :variable:`CMAKE_<LANG>_LINKER_WRAPPER_FLAG`
variable to format ``LINKER:`` prefix in the link options
(see :command:`add_link_options` and :command:`target_link_options`).

When specified, arguments of the ``LINKER:`` prefix will be concatenated using
this value as separator.
