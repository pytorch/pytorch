get_property
------------

Get a property.

.. code-block:: cmake

  get_property(<variable>
               <GLOBAL             |
                DIRECTORY [<dir>]  |
                TARGET    <target> |
                SOURCE    <source>
                          [DIRECTORY <dir> | TARGET_DIRECTORY <target>] |
                INSTALL   <file>   |
                TEST      <test>
                          [DIRECTORY <dir>] |
                CACHE     <entry>  |
                VARIABLE           >
               PROPERTY <name>
               [SET | DEFINED | BRIEF_DOCS | FULL_DOCS])

Gets one property from one object in a scope.

The first argument specifies the variable in which to store the result.
The second argument determines the scope from which to get the property.
It must be one of the following:

``GLOBAL``
  Scope is unique and does not accept a name.

``DIRECTORY``
  Scope defaults to the current directory, but another
  directory (already processed by CMake) may be named by the
  full or relative path ``<dir>``.
  Relative paths are treated as relative to the current source directory.
  See also the :command:`get_directory_property` command.

  .. versionadded:: 3.19
    ``<dir>`` may reference a binary directory.

``TARGET``
  Scope must name one existing target.
  See also the :command:`get_target_property` command.

``SOURCE``
  Scope must name one source file.  By default, the source file's property
  will be read from the current source directory's scope.

  .. versionadded:: 3.18
    Directory scope can be overridden with one of the following sub-options:

    ``DIRECTORY <dir>``
      The source file property will be read from the ``<dir>`` directory's
      scope.  CMake must already know about
      the directory, either by having added it through a call
      to :command:`add_subdirectory` or ``<dir>`` being the top level directory.
      Relative paths are treated as relative to the current source directory.

      .. versionadded:: 3.19
        ``<dir>`` may reference a binary directory.

    ``TARGET_DIRECTORY <target>``
      The source file property will be read from the directory scope in which
      ``<target>`` was created (``<target>`` must therefore already exist).

  See also the :command:`get_source_file_property` command.

``INSTALL``
  .. versionadded:: 3.1

  Scope must name one installed file path.

``TEST``
  Scope must name one existing test.
  See also the :command:`get_test_property` command.

  .. versionadded:: 3.28
    Directory scope can be overridden with the following sub-option:

    ``DIRECTORY <dir>``
      The test property will be read from the ``<dir>`` directory's
      scope.  CMake must already know about the directory, either by having
      added it through a call to :command:`add_subdirectory` or ``<dir>`` being
      the top level directory. Relative paths are treated as relative to the
      current source directory. ``<dir>`` may reference a binary directory.

``CACHE``
  Scope must name one cache entry.

``VARIABLE``
  Scope is unique and does not accept a name.

The required ``PROPERTY`` option is immediately followed by the name of
the property to get.  If the property is not set, the named ``<variable>``
will be unset in the calling scope upon return, although some properties
support inheriting from a parent scope if defined to behave that way
(see :command:`define_property`).

If the ``SET`` option is given, the variable is set to a boolean
value indicating whether the property has been set.  If the ``DEFINED``
option is given, the variable is set to a boolean value indicating
whether the property has been defined, such as with the
:command:`define_property` command.

If ``BRIEF_DOCS`` or ``FULL_DOCS`` is given, then the variable is set to a
string containing documentation for the requested property.  If
documentation is requested for a property that has not been defined,
``NOTFOUND`` is returned.

.. note::

  The :prop_sf:`GENERATED` source file property may be globally visible.
  See its documentation for details.

See Also
^^^^^^^^

* :command:`define_property`
* :command:`set_property`
