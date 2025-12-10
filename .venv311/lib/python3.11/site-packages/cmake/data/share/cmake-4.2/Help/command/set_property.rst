set_property
------------

Set a named property in a given scope.

.. code-block:: cmake

  set_property(<GLOBAL                      |
                DIRECTORY [<dir>]           |
                TARGET    [<target1> ...]   |
                SOURCE    [<src1> ...]
                          [DIRECTORY <dirs> ...]
                          [TARGET_DIRECTORY <targets> ...] |
                INSTALL   [<file1> ...]     |
                TEST      [<test1> ...]
                          [DIRECTORY <dir>] |
                CACHE     [<entry1> ...]    >
               [APPEND] [APPEND_STRING]
               PROPERTY <name> [<value1> ...])

Sets one property on zero or more objects of a scope.

The first argument determines the scope in which the property is set.
It must be one of the following:

``GLOBAL``
  Scope is unique and does not accept a name.

``DIRECTORY``
  Scope defaults to the current directory but other directories
  (already processed by CMake) may be named by full or relative path.
  Relative paths are treated as relative to the current source directory.
  See also the :command:`set_directory_properties` command.

  .. versionadded:: 3.19
    ``<dir>`` may reference a binary directory.

``TARGET``
  Scope may name zero or more existing targets.
  See also the :command:`set_target_properties` command.

  :ref:`Alias Targets` do not support setting target properties.

``SOURCE``
  Scope may name zero or more source files.  By default, source file properties
  are only visible to targets added in the same directory (``CMakeLists.txt``).

  .. versionadded:: 3.18
    Visibility can be set in other directory scopes using one or both of the
    following sub-options:

    ``DIRECTORY <dirs>...``
      The source file property will be set in each of the ``<dirs>``
      directories' scopes.  CMake must already know about
      each of these directories, either by having added them through a call to
      :command:`add_subdirectory` or it being the top level source directory.
      Relative paths are treated as relative to the current source directory.

      .. versionadded:: 3.19
        ``<dirs>`` may reference a binary directory.

    ``TARGET_DIRECTORY <targets>...``
      The source file property will be set in each of the directory scopes
      where any of the specified ``<targets>`` were created (the ``<targets>``
      must therefore already exist).

  See also the :command:`set_source_files_properties` command.

``INSTALL``
  .. versionadded:: 3.1

  Scope may name zero or more installed file paths.
  These are made available to CPack to influence deployment.

  Both the property key and value may use generator expressions.
  Specific properties may apply to installed files and/or directories.

  Path components have to be separated by forward slashes,
  must be normalized and are case sensitive.

  To reference the installation prefix itself with a relative path use ``.``.

  Currently installed file properties are only defined for
  the WIX generator where the given paths are relative
  to the installation prefix.

``TEST``
  Scope is limited to the directory the command is called in. It may name zero
  or more existing tests. See also command :command:`set_tests_properties`.

  Test property values may be specified using
  :manual:`generator expressions <cmake-generator-expressions(7)>`
  for tests created by the :command:`add_test(NAME)` signature.

  .. versionadded:: 3.28

    Visibility can be set in other directory scopes using the following sub-option:

    ``DIRECTORY <dir>``
      The test property will be set in the ``<dir>`` directory's scope. CMake must
      already know about this directory, either by having added it through a call
      to :command:`add_subdirectory` or it being the top level source directory.
      Relative paths are treated as relative to the current source directory.
      ``<dir>`` may reference a binary directory.

``CACHE``
  Scope must name zero or more existing cache entries.

The required ``PROPERTY`` option is immediately followed by the name of
the property to set.  Remaining arguments are used to compose the
property value in the form of a semicolon-separated list.

If the ``APPEND`` option is given the list is appended to any existing
property value (except that empty values are ignored and not appended).
If the ``APPEND_STRING`` option is given the string is
appended to any existing property value as string, i.e. it results in a
longer string and not a list of strings.  When using ``APPEND`` or
``APPEND_STRING`` with a property defined to support ``INHERITED``
behavior (see :command:`define_property`), no inheriting occurs when
finding the initial value to append to.  If the property is not already
directly set in the nominated scope, the command will behave as though
``APPEND`` or ``APPEND_STRING`` had not been given.

.. note::

  The :prop_sf:`GENERATED` source file property may be globally visible.
  See its documentation for details.

See Also
^^^^^^^^

* :command:`define_property`
* :command:`get_property`
* The :manual:`cmake-properties(7)` manual for a list of properties
  in each scope.
