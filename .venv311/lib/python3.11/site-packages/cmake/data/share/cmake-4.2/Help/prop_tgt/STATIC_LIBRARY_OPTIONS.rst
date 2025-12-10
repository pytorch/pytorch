STATIC_LIBRARY_OPTIONS
----------------------

.. versionadded:: 3.13

Archiver (or MSVC librarian) flags for a static library target.
Targets that are shared libraries, modules, or executables need to use
the :prop_tgt:`LINK_OPTIONS` target property.

This property holds a :ref:`semicolon-separated list <CMake Language Lists>` of options
specified so far for its target.  Use :command:`set_target_properties` or
:command:`set_property` commands to set its content.

Contents of ``STATIC_LIBRARY_OPTIONS`` may use "generator expressions" with the
syntax ``$<...>``.  See the :manual:`cmake-generator-expressions(7)` manual
for available expressions.  See the :manual:`cmake-buildsystem(7)` manual
for more on defining buildsystem properties.

.. note::

  This property must be used in preference to :prop_tgt:`STATIC_LIBRARY_FLAGS`
  property.

.. include:: ../command/include/OPTIONS_SHELL.rst

.. include:: ../prop_tgt/include/STATIC_LIBRARY_OPTIONS_ARCHIVER.rst
