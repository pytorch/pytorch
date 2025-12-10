CMAKE_HOST_SYSTEM_NAME
----------------------

Name of the OS CMake is running on.

On systems that have the uname command, this variable is set to the
output of ``uname -s``.  ``Linux``, ``Windows``, and ``Darwin`` for macOS
are the values found on the big three operating systems.

For a list of possible values, see :variable:`CMAKE_SYSTEM_NAME`.
