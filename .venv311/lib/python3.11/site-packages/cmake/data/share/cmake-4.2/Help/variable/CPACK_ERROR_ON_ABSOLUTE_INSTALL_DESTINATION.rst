CPACK_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION
-------------------------------------------

Ask CPack to error out as soon as a file with absolute ``INSTALL DESTINATION``
is encountered.

The fatal error is emitted before the installation of the offending
file takes place.  Some CPack generators, like ``NSIS``, enforce this
internally.  This variable triggers the definition
of :variable:`CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION` when CPack
runs.
