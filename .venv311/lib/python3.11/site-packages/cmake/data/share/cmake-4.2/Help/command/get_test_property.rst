get_test_property
-----------------

Get a property of the test.

.. code-block:: cmake

  get_test_property(<test> <property> [DIRECTORY <dir>] <variable>)

Get a property from the test.  The value of the property is stored in
the specified ``<variable>``.  If the ``<test>`` is not defined, or the
test property is not found, ``<variable>`` will be set to ``NOTFOUND``.
If the test property was defined to be an ``INHERITED`` property (see
:command:`define_property`), the search will include the relevant parent
scopes, as described for the :command:`define_property` command.

For a list of standard properties you can type
:option:`cmake --help-property-list`.

.. versionadded:: 3.28
  Directory scope can be overridden with the following sub-option:

  ``DIRECTORY <dir>``
    The test property will be read from the ``<dir>`` directory's
    scope.  CMake must already know about that source directory, either by
    having added it through a call to :command:`add_subdirectory` or ``<dir>``
    being the top level source directory.  Relative paths are treated as
    relative to the current source directory. ``<dir>`` may reference a binary
    directory.

See Also
^^^^^^^^

* :command:`define_property`
* the more general :command:`get_property` command
