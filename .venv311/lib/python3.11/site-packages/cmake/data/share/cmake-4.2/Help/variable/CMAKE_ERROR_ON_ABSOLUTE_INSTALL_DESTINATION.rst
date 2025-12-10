CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION
-------------------------------------------

Ask ``cmake_install.cmake`` script to error out as soon as a file with
absolute ``INSTALL DESTINATION`` is encountered.

The fatal error is emitted before the installation of the offending
file takes place.  This variable is used by CMake-generated
``cmake_install.cmake`` scripts.  If one sets this variable to ``ON`` while
running the script, it may get fatal error messages from the script.
