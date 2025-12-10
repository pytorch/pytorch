CMP0084
-------

.. versionadded:: 3.14

The :module:`FindQt` module does not exist for :command:`find_package`.

The existence of :module:`FindQt` means that for Qt upstream to provide
package config files that can be found by ``find_package(Qt)``, the consuming
project has to explicitly specify ``find_package(Qt CONFIG)``. Removing this
module gives Qt a path forward for exporting its own config files which can
easily be found by consuming projects.

This policy pretends that CMake's internal :module:`FindQt` module does not
exist for :command:`find_package`. If a project really wants to use Qt 3 or 4,
it can call ``find_package(Qt[34])``, ``include(FindQt)``, or add
:module:`FindQt` to their :variable:`CMAKE_MODULE_PATH`.

The ``OLD`` behavior of this policy is for :module:`FindQt` to exist for
:command:`find_package`. The ``NEW`` behavior is to pretend that it doesn't
exist for :command:`find_package`.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: 3.14
.. |WARNS_OR_DOES_NOT_WARN| replace:: warns
.. include:: include/STANDARD_ADVICE.rst

.. include:: include/DEPRECATED.rst
