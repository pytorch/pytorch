get_target_property
-------------------

Get a property from a target.

.. code-block:: cmake

  get_target_property(<variable> <target> <property>)

Get a property from a target.  The value of the property is stored in the
specified ``<variable>``.  If the target property is not found, ``<variable>``
will be set to ``<variable>-NOTFOUND``.  If the target property was defined to
be an ``INHERITED`` property (see :command:`define_property`), the search will
include the relevant parent scopes, as described for the
:command:`define_property` command.

Use :command:`set_target_properties` to set target property values.
Properties are usually used to control how a target is built, but some
query the target instead.  This command can get properties for any
target so far created.  The targets do not need to be in the current
``CMakeLists.txt`` file.

See Also
^^^^^^^^

* :command:`define_property`
* the more general :command:`get_property` command
* :command:`set_target_properties`
* :ref:`Target Properties` for the list of properties known to CMake
