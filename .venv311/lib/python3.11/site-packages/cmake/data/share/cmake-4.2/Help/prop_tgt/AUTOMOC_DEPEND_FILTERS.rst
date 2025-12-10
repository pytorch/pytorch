AUTOMOC_DEPEND_FILTERS
----------------------

.. versionadded:: 3.9

Filter definitions used by :prop_tgt:`AUTOMOC` to extract file names from a
source file that are registered as additional dependencies for the
``moc`` file of the source file.

Filters are defined as ``KEYWORD;REGULAR_EXPRESSION`` pairs. First the file
content is searched for ``KEYWORD``. If it is found at least once, then file
names are extracted by successively searching for ``REGULAR_EXPRESSION`` and
taking the first match group.

The file name found in the first match group is searched for

- first in the vicinity of the source file
- and afterwards in the target's :prop_tgt:`INCLUDE_DIRECTORIES`.

If any of the extracted files changes, then the ``moc`` file for the source
file gets rebuilt even when the source file itself doesn't change.

If any of the extracted files is :prop_sf:`GENERATED` or if it is not in the
target's sources, then it might be necessary to add it to the
:ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>` target dependencies.
See :prop_tgt:`AUTOGEN_TARGET_DEPENDS` for reference.

By default ``AUTOMOC_DEPEND_FILTERS`` is initialized from
:variable:`CMAKE_AUTOMOC_DEPEND_FILTERS`, which is empty by default.

From Qt 5.15.0 on this variable is ignored as ``moc`` is able to output the
correct dependencies.

See the :manual:`cmake-qt(7)` manual for more information on using CMake
with Qt.


Example 1
^^^^^^^^^

A header file ``my_class.hpp`` uses a custom macro ``JSON_FILE_MACRO`` which
is defined in an other header ``macros.hpp``.
We want the ``moc`` file of ``my_class.hpp`` to depend on the file name
argument of ``JSON_FILE_MACRO``:

.. code-block:: c++

  // my_class.hpp
  class My_Class : public QObject
  {
    Q_OBJECT
    JSON_FILE_MACRO ( "info.json" )
  ...
  };

In ``CMakeLists.txt`` we add a filter to
:variable:`CMAKE_AUTOMOC_DEPEND_FILTERS` like this:

.. code-block:: c++

  list(APPEND CMAKE_AUTOMOC_DEPEND_FILTERS
    "JSON_FILE_MACRO"
    "[\n][ \t]*JSON_FILE_MACRO[ \t]*\\([ \t]*\"([^\"]+)\""
  )

We assume ``info.json`` is a plain (not :prop_sf:`GENERATED`) file that is
listed in the target's source.  Therefore we do not need to add it to
:prop_tgt:`AUTOGEN_TARGET_DEPENDS`.

Example 2
^^^^^^^^^

In the target ``my_target`` a header file ``complex_class.hpp`` uses a
custom macro ``JSON_BASED_CLASS`` which is defined in an other header
``macros.hpp``:

.. code-block:: c++

  // macros.hpp
  ...
  #define JSON_BASED_CLASS(name, json) \
  class name : public QObject \
  { \
    Q_OBJECT \
    Q_PLUGIN_METADATA(IID "demo" FILE json) \
    name() {} \
  };
  ...

.. code-block:: c++

  // complex_class.hpp
  #pragma once
  JSON_BASED_CLASS(Complex_Class, "meta.json")
  // end of file

Since ``complex_class.hpp`` doesn't contain a ``Q_OBJECT`` macro it would be
ignored by :prop_tgt:`AUTOMOC`.  We change this by adding ``JSON_BASED_CLASS``
to :variable:`CMAKE_AUTOMOC_MACRO_NAMES`:

.. code-block:: cmake

  list(APPEND CMAKE_AUTOMOC_MACRO_NAMES "JSON_BASED_CLASS")

We want the ``moc`` file of ``complex_class.hpp`` to depend on
``meta.json``.  So we add a filter to
:variable:`CMAKE_AUTOMOC_DEPEND_FILTERS`:

.. code-block:: cmake

  list(APPEND CMAKE_AUTOMOC_DEPEND_FILTERS
    "JSON_BASED_CLASS"
    "[\n^][ \t]*JSON_BASED_CLASS[ \t]*\\([^,]*,[ \t]*\"([^\"]+)\""
  )

Additionally we assume ``meta.json`` is :prop_sf:`GENERATED` which is
why we have to add it to :prop_tgt:`AUTOGEN_TARGET_DEPENDS`:

.. code-block:: cmake

  set_property(TARGET my_target APPEND PROPERTY AUTOGEN_TARGET_DEPENDS "meta.json")
