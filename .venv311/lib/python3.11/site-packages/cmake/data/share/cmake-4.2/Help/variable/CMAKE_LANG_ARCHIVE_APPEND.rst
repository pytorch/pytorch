CMAKE_<LANG>_ARCHIVE_APPEND
---------------------------

Rule variable to append to a static archive.

This is a rule variable that tells CMake how to append to a static
archive.  It is used in place of :variable:`CMAKE_<LANG>_CREATE_STATIC_LIBRARY`
on some platforms in order to support large object counts.  See also
:variable:`CMAKE_<LANG>_ARCHIVE_CREATE` and
:variable:`CMAKE_<LANG>_ARCHIVE_FINISH`.
