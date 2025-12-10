CPACK_WARN_ON_ABSOLUTE_INSTALL_DESTINATION
------------------------------------------

Ask CPack to warn each time a file with absolute ``INSTALL DESTINATION`` is
encountered.

This variable triggers the definition of
:variable:`CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION` when CPack runs
``cmake_install.cmake`` scripts.
