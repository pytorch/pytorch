- ``reset``: Reset to the unmodified value, ignoring all modifications to
  ``MYVAR`` prior to this entry. Note that this will reset the variable to
  the value set by :prop_test:`ENVIRONMENT`, if it was set, and otherwise
  to its state from the rest of the CTest execution.
- ``set``: Replaces the current value of ``MYVAR`` with ``VALUE``.
- ``unset``: Unsets the current value of ``MYVAR``.
- ``string_append``: Appends singular ``VALUE`` to the current value of
  ``MYVAR``.
- ``string_prepend``: Prepends singular ``VALUE`` to the current value of
  ``MYVAR``.
- ``path_list_append``: Appends singular ``VALUE`` to the current value of
  ``MYVAR`` using the host platform's path list separator (``;`` on Windows
  and ``:`` elsewhere).
- ``path_list_prepend``: Prepends singular ``VALUE`` to the current value of
  ``MYVAR`` using the host platform's path list separator (``;`` on Windows
  and ``:`` elsewhere).
- ``cmake_list_append``: Appends singular ``VALUE`` to the current value of
  ``MYVAR`` using ``;`` as the separator.
- ``cmake_list_prepend``: Prepends singular ``VALUE`` to the current value of
  ``MYVAR`` using ``;`` as the separator.
