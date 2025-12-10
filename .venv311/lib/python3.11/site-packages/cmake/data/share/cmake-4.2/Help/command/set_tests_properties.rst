set_tests_properties
--------------------

Set a property of the tests.

.. code-block:: cmake

  set_tests_properties(<tests>...
                       [DIRECTORY <dir>]
                       PROPERTIES <prop1> <value1>
                       [<prop2> <value2>]...)

Sets a property for the tests.  If the test is not found, CMake
will report an error.

Test property values may be specified using
:manual:`generator expressions <cmake-generator-expressions(7)>`
for tests created by the :command:`add_test(NAME)` signature.

.. versionadded:: 3.28
  Visibility can be set in other directory scopes using the following option:

  ``DIRECTORY <dir>``
    The test properties will be set in the ``<dir>`` directory's scope.
    CMake must already know about this directory, either by having added it
    through a call to :command:`add_subdirectory` or it being the top level
    source directory. Relative paths are treated as relative to the current
    source directory. ``<dir>`` may reference a binary directory.

See Also
^^^^^^^^

* :command:`add_test`
* :command:`define_property`
* the more general :command:`set_property` command
* :ref:`Test Properties` for the list of properties known to CMake
