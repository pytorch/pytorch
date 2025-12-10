EXPORT_FIND_PACKAGE_NAME
------------------------

.. note::

  Experimental. Gated by ``CMAKE_EXPERIMENTAL_EXPORT_PACKAGE_DEPENDENCIES``.

Control the package name associated with a dependency target when exporting a
:command:`find_dependency` call in :command:`install(PACKAGE_INFO)`,
:command:`export(PACKAGE_INFO)`, :command:`install(EXPORT)` or
:command:`export(EXPORT)`. This can be used to assign a package name to a
package that is built by CMake and exported, or a package that was provided by
:module:`FetchContent`.

This property is initialized by :variable:`CMAKE_EXPORT_FIND_PACKAGE_NAME`.
