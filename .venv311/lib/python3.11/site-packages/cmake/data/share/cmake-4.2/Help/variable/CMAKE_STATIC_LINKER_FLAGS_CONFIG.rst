CMAKE_STATIC_LINKER_FLAGS_<CONFIG>
----------------------------------

Flags to be used to create static libraries.  These flags will be passed
to the archiver when creating a static library in the ``<CONFIG>``
configuration.

See also :variable:`CMAKE_STATIC_LINKER_FLAGS`.

.. note::
  Static libraries do not actually link.  They are essentially archives
  of object files.  The use of the name "linker" in the name of this
  variable is kept for compatibility.
