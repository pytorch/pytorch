EXPORT_PROPERTIES
-----------------

.. versionadded:: 3.12

List additional properties to export for a target.

This property contains a list of property names that should be exported by
the :command:`install(EXPORT)` and :command:`export` commands.  By default
only a limited number of properties are exported. This property can be used
to additionally export other properties as well.

Properties starting with ``INTERFACE_`` or ``IMPORTED_`` are not allowed as
they are reserved for internal CMake use.

Properties containing generator expressions are also not allowed.

.. note::

  Since CMake 3.19, :ref:`Interface Libraries` may have arbitrary
  target properties.  If a project exports an interface library
  with custom properties, the resulting package may not work with
  dependents configured by older versions of CMake that reject the
  custom properties.
