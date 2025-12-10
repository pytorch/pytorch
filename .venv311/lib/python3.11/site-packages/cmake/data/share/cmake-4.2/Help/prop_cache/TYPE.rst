TYPE
----

Widget type for entry in GUIs.

Cache entry values are always strings, but CMake GUIs present widgets
to help users set values.  The GUIs use this property as a hint to
determine the widget type.  Valid ``TYPE`` values are:

.. table::
  :align: left

  =================  ========================================
  ``BOOL``           Boolean ON/OFF value.
  ``PATH``           Path to a directory.
  ``FILEPATH``       Path to a file.
  ``STRING``         Generic string value.
  ``INTERNAL``       Do not present in GUI at all.
  ``STATIC``         Value managed by CMake, do not change.
  ``UNINITIALIZED``  Type not yet specified.
  =================  ========================================

Generally the ``TYPE`` of a cache entry should be set by the command which
creates it (:command:`set`, :command:`option`, :command:`find_library`, etc.).
