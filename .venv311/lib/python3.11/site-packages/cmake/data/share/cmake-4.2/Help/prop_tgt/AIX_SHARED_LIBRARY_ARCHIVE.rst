AIX_SHARED_LIBRARY_ARCHIVE
--------------------------

.. versionadded:: 3.31

On AIX, enable or disable creation of a shared library archive
for a ``SHARED`` library target:

* If enabled, the shared object ``.so`` file is placed inside
  an archive ``.a`` file.  This is the preferred convention on AIX.

  The shared object name in the archive encodes version information from
  the :prop_tgt:`SOVERSION` target property, if set, and otherwise from
  the :prop_tgt:`VERSION` target property, if set.

* If disabled, a plain shared object ``.so`` file is produced.
  This is consistent with other UNIX platforms.

This property defaults to :variable:`CMAKE_AIX_SHARED_LIBRARY_ARCHIVE`
if that variable is set when a non-imported ``SHARED`` library target
is created by :command:`add_library`.  Imported targets must explicitly
enable :prop_tgt:`!AIX_SHARED_LIBRARY_ARCHIVE` if they import an AIX
shared library archive.

.. versionchanged:: 4.0

  For a non-imported target, if this property is not set, the
  default is *enabled*.  See policy :policy:`CMP0182`.

  In CMake 3.31, policy :policy:`CMP0182` did not exist,
  so the default was *disabled*.

  In CMake 3.30 and lower, :prop_tgt:`!AIX_SHARED_LIBRARY_ARCHIVE`
  did not exist, so the default was *disabled*.
