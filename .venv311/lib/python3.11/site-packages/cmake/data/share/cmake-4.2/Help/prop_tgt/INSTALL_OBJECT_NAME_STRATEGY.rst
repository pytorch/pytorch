INSTALL_OBJECT_NAME_STRATEGY
----------------------------

.. versionadded:: 4.2

``INSTALL_OBJECT_NAME_STRATEGY`` is a string target property variable
specifying the strategy to use when naming installed object files. The
supported values are:

- ``FULL``: Object files are named after the associated source file or
  its :prop_sf:`OBJECT_NAME` property.
- ``SHORT``: Object files are named based on the hash of the source file name
  to reduce path lengths.

When unset or the named strategy is not supported, the ``FULL`` strategy is
used.

This property is initialized by the value of the variable
:variable:`CMAKE_INSTALL_OBJECT_NAME_STRATEGY` if it is set when a target is
created.

.. note::
  Not all generators support all strategies and paths may differ between
  generators.
