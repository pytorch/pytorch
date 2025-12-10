variable_watch
--------------

Watch the CMake variable for change.

.. code-block:: cmake

  variable_watch(<variable> [<command>])

If the specified ``<variable>`` changes and no ``<command>`` is given,
a message will be printed to inform about the change.

If ``<command>`` is given, this command will be executed instead.
The command will receive the following arguments:
``COMMAND(<variable> <access> <value> <current_list_file> <stack>)``

``<variable>``
 Name of the variable being accessed.

``<access>``
 One of ``READ_ACCESS``, ``UNKNOWN_READ_ACCESS``, ``MODIFIED_ACCESS``,
 ``UNKNOWN_MODIFIED_ACCESS``, or ``REMOVED_ACCESS``.  The ``UNKNOWN_``
 values are only used when the variable has never been set.  Once set,
 they are never used again during the same CMake run, even if the
 variable is later unset.

``<value>``
 The value of the variable.  On a modification, this is the new
 (modified) value of the variable.  On removal, the value is empty.

``<current_list_file>``
 Full path to the file doing the access.

``<stack>``
 List of absolute paths of all files currently on the stack of file
 inclusion, with the bottom-most file first and the currently
 processed file (that is, ``current_list_file``) last.

Note that for some accesses such as :command:`list(APPEND)`, the watcher
is executed twice, first with a read access and then with a write one.
Also note that an :command:`if(DEFINED)` query on the variable does not
register as an access and the watcher is not executed.

Only non-cache variables can be watched using this command.  Access to
cache variables is never watched.  However, the existence of a cache
variable ``var`` causes accesses to the non-cache variable ``var`` to
not use the ``UNKNOWN_`` prefix, even if a non-cache variable ``var``
has never existed.
