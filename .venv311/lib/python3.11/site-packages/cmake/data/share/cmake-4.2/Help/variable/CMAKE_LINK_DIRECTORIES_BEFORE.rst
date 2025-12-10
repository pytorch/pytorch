CMAKE_LINK_DIRECTORIES_BEFORE
-----------------------------

.. versionadded:: 3.13

Whether to append or prepend directories by default in
:command:`link_directories`.

This variable affects the default behavior of the :command:`link_directories`
command.  Setting this variable to ``ON`` is equivalent to using the ``BEFORE``
option in all uses of that command.
