CMAKE_<LANG>_CREATE_SHARED_LIBRARY_ARCHIVE
------------------------------------------

.. versionadded:: 3.31

Rule variable to create a shared library with archive.

This is a rule variable that tells CMake how to create a shared
library with an archive for the language <LANG>.  This rule variable
is a ; delimited list of commands to run to perform the linking step.
