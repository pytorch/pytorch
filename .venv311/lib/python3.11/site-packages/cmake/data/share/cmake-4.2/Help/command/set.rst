set
---

Set a normal, cache, or environment variable to a given value.
See the :ref:`cmake-language(7) variables <CMake Language Variables>`
documentation for the scopes and interaction of normal variables
and cache entries.

Signatures of this command that specify a ``<value>...`` placeholder
expect zero or more arguments.  Multiple arguments will be joined as
a :ref:`semicolon-separated list <CMake Language Lists>` to form the
actual variable value to be set.

Set Normal Variable
^^^^^^^^^^^^^^^^^^^

.. signature::
  set(<variable> <value>... [PARENT_SCOPE])
  :target: normal

  Set or unset ``<variable>`` in the current function or directory scope:

  * If at least one ``<value>...`` is given, set the variable to that value.
  * If no value is given, unset the variable.  This is equivalent to
    :command:`unset(<variable>) <unset>`.

  If the ``PARENT_SCOPE`` option is given the variable will be set in
  the scope above the current scope.  Each new directory or :command:`function`
  command creates a new scope.  A scope can also be created with the
  :command:`block` command. ``set(PARENT_SCOPE)`` will set the value
  of a variable into the parent directory, calling function, or
  encompassing scope (whichever is applicable to the case at hand).
  The previous state of the variable's value stays the same in the
  current scope (e.g., if it was undefined before, it is still undefined
  and if it had a value, it is still that value).

  The :command:`block(PROPAGATE)` and :command:`return(PROPAGATE)` commands
  can be used as an alternate method to the :command:`set(PARENT_SCOPE)`
  and :command:`unset(PARENT_SCOPE)` commands to update the parent scope.

.. include:: include/UNSET_NOTE.rst

Set Cache Entry
^^^^^^^^^^^^^^^

.. signature::
  set(CACHE{<variable>} [TYPE <type>] [HELP <helpstring>...] [FORCE]
                        VALUE [<value>...])
  :target: CACHE

  .. versionadded:: 4.2

  Sets the given cache ``<variable>`` (cache entry). The options are:

  ``TYPE <type>``
    Specify the type of the cache entry. The ``<type>`` must be one of:

    ``BOOL``
      Boolean ``ON/OFF`` value.
      :manual:`cmake-gui(1)` offers a checkbox.

    ``FILEPATH``
      Path to a file on disk.
      :manual:`cmake-gui(1)` offers a file dialog.

    ``PATH``
      Path to a directory on disk.
      :manual:`cmake-gui(1)` offers a file dialog.

    ``STRING``
      A line of text.
      :manual:`cmake-gui(1)` offers a text field or a drop-down selection
      if the :prop_cache:`STRINGS` cache entry property is set.

    ``INTERNAL``
      A line of text.
      :manual:`cmake-gui(1)` does not show internal entries.
      They may be used to store variables persistently across runs.
      Use of this type implies ``FORCE``.

    If ``TYPE`` is not specified, if the cache variable already exist and its
    type is not ``UNINITIALIZED``, the type previously specified will be kept
    otherwise, ``STRING`` will be used.

  ``HELP <helpstring>...``
    The ``<helpstring>`` must be specified as a line of text providing a quick
    summary of the option for presentation to :manual:`cmake-gui(1)` users. If
    more than one string is given, they are concatenated into a single string
    with no separator between them.

    If ``HELP`` is not specified, an empty string will be used.

  ``FORCE``
    Since cache entries are meant to provide user-settable values this does not
    overwrite existing cache entries by default.  Use the ``FORCE`` option to
    overwrite existing entries.

  ``VALUE <value>...``
    List of values to be set to the cache ``<variable>``. This argument must be
    always the last one.

  If the cache entry does not exist prior to the call or the ``FORCE``
  option is given then the cache entry will be set to the given value.

  .. note::

    The content of the cache variable will not be directly accessible
    if a normal variable of the same name already exists
    (see :ref:`rules of variable evaluation <CMake Language Variables>`).
    If policy :policy:`CMP0126` is set to ``OLD``, any normal variable
    binding in the current scope will be removed.

  It is possible for the cache entry to exist prior to the call but
  have no type set if it was created on the :manual:`cmake(1)` command
  line by a user through the :option:`-D\<var\>=\<value\> <cmake -D>` option
  without specifying a type.  In this case the ``set`` command will add the
  type.  Furthermore, if the ``<type>`` is ``PATH`` or ``FILEPATH``
  and the ``<value>`` provided on the command line is a relative path,
  then the ``set`` command will treat the path as relative to the
  current working directory and convert it to an absolute path.

.. signature::
  set(<variable> <value>... CACHE <type> <docstring> [FORCE])
  :target: CACHE_legacy

  This signature is supported for compatibility purpose. Use preferably the
  other one.

Set Environment Variable
^^^^^^^^^^^^^^^^^^^^^^^^

.. signature::
  set(ENV{<variable>} [<value>])
  :target: ENV

  Sets an :manual:`Environment Variable <cmake-env-variables(7)>`
  to the given value.
  Subsequent calls of ``$ENV{<variable>}`` will return this new value.

  This command affects only the current CMake process, not the process
  from which CMake was called, nor the system environment at large,
  nor the environment of subsequent build or test processes.

  If no argument is given after ``ENV{<variable>}`` or if ``<value>`` is
  an empty string, then this command will clear any existing value of the
  environment variable.

  Arguments after ``<value>`` are ignored. If extra arguments are found,
  then an author warning is issued.

See Also
^^^^^^^^

* :command:`unset`
