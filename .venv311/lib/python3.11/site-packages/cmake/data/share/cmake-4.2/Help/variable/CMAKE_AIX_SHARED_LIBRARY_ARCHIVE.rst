CMAKE_AIX_SHARED_LIBRARY_ARCHIVE
--------------------------------

.. versionadded:: 3.31

On AIX, enable or disable creation of shared library archives.

This variable initializes the :prop_tgt:`AIX_SHARED_LIBRARY_ARCHIVE`
target property on non-imported ``SHARED`` library targets as they are
created by :command:`add_library`.  See that target property for details.
