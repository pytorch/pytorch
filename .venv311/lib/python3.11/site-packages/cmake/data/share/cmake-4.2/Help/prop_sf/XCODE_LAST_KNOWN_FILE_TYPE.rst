XCODE_LAST_KNOWN_FILE_TYPE
--------------------------

.. versionadded:: 3.1

Set the :generator:`Xcode` ``lastKnownFileType`` attribute on its reference to
a source file.  CMake computes a default based on file extension but
can be told explicitly with this property.

See also :prop_sf:`XCODE_EXPLICIT_FILE_TYPE`, which is preferred
over this property if set.
