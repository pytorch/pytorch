set_directory_properties
------------------------

Set properties of the current directory and subdirectories.

.. code-block:: cmake

  set_directory_properties(PROPERTIES <prop1> <value1> [<prop2> <value2>] ...)

Sets properties of the current directory and its subdirectories in key-value
pairs.

See also the :command:`set_property(DIRECTORY)` command.

See :ref:`Directory Properties` for the list of properties known to CMake
and their individual documentation for the behavior of each property.

See Also
^^^^^^^^

* :command:`define_property`
* :command:`get_directory_property`
* the more general :command:`set_property` command
