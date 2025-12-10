unset
-----

Unset a variable, cache variable, or environment variable.

Unset Normal Variable
^^^^^^^^^^^^^^^^^^^^^

.. signature::
  unset(<variable> [PARENT_SCOPE])
  :target: normal

  Removes a normal variable from the current scope, causing it
  to become undefined.

  If ``PARENT_SCOPE`` is present then the variable is removed from the scope
  above the current scope.  See the same option in the :command:`set` command
  for further details.

.. include:: include/UNSET_NOTE.rst

Unset Cache Entry
^^^^^^^^^^^^^^^^^

.. signature::
  unset(CACHE{<variable>})
  :target: CACHE

  .. versionadded:: 4.2

  Removes ``<variable>`` from the cache, causing it to become undefined.

.. signature::
  unset(<variable> CACHE)
  :target: CACHE_legacy

  This signature is supported for compatibility purpose. Use preferably the
  other one.

Unset Environment Variable
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. signature::
  unset(ENV{<variable>})
  :target: ENV

  Removes ``<variable>`` from the currently available
  :manual:`Environment Variables <cmake-env-variables(7)>`.
  Subsequent calls of ``$ENV{<variable>}`` will return the empty string.

  This command affects only the current CMake process, not the process
  from which CMake was called, nor the system environment at large,
  nor the environment of subsequent build or test processes.

See Also
^^^^^^^^

* :command:`set`
