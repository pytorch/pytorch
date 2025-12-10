CMAKE_ADD_CUSTOM_COMMAND_DEPENDS_EXPLICIT_ONLY
----------------------------------------------

.. versionadded:: 3.27

Whether to enable the ``DEPENDS_EXPLICIT_ONLY`` option by default in
:command:`add_custom_command`.

This variable affects the default behavior of the :command:`add_custom_command`
command.  Setting this variable to ``ON`` is equivalent to using the
``DEPENDS_EXPLICIT_ONLY`` option in all uses of that command.

See also :variable:`CMAKE_OPTIMIZE_DEPENDENCIES`.
